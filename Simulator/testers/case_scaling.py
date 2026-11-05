"""Evaluate and summarize the retained 500-device continuous scaling cases.

Commands: ``evaluate --horizon 24|96`` and ``summarize``. Evaluations preserve
the original directional squared-distance protocol and checkpoints. Summaries
only read saved results. Importing or requesting help does not load solvers.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import zipfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from Simulator.paths import CODE_ROOT, RESULTS_ROOT


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _worker(horizon, directions, region, output, start, end):
    """Each process owns its physical model and single-threaded solver."""
    from Simulator.runners.train_aggregation import build_aggregator
    from Simulator.testers.polytope_evaluation import evaluate_native

    agg = build_aggregator(horizon, seed=0, counts=(100, 200, 200))
    output = Path(output)
    progress_path = output / f"evaluation_progress_part_{start}_{end}.json"

    def progress(record):
        _write_json(progress_path, record)

    return evaluate_native(
        agg, directions, region, progress=progress,
        checkpoint_path=output / f"evaluation_checkpoint_part_{start}_{end}.npz")


def _context_offset(records, offset):
    for record in records:
        context = record.get("context")
        if context is not None and "direction_index" in context:
            context["direction_index"] += offset
    return records


def evaluate_parallel(agg, directions, region, output, progress=None, workers=3):
    """Resume a validated sequential prefix, then evaluate contiguous blocks.

    The existing full checkpoint is validated before reading its prefix. Each
    block has a stable native checkpoint and validates its complete manifest.
    The merged full checkpoint is saved only after all blocks succeed. Every
    direction retains fresh solver state, exactly as in sequential mode.
    """
    from Simulator.testers.polytope_evaluation import (
        SOLVER_OPTIONS, RETRY_POLICY, _VERSION, _array_hash, _build_approximation,
        _save_checkpoint, _translate_pyomo,
    )

    import gurobipy as gp

    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    started = time.perf_counter()
    directions = np.asarray(directions, dtype=float)
    if (directions.ndim != 2 or directions.shape[1] != agg.T or not len(directions)
            or not np.isfinite(directions).all()):
        raise ValueError("Invalid evaluation directions")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Build the same signatures as evaluate_native before accepting any prefix.
    with gp.Env(empty=True) as env:
        for key, value in SOLVER_OPTIONS.items():
            env.setParam(key, value)
        env.start()
        original, physical_hash = _translate_pyomo(agg.case_aggregator(build_only=True), env)
        approximation, region_hash = _build_approximation(agg.T, region, env)
        try:
            sizes = {"original_variables": original.model.NumVars,
                     "original_rows": original.model.NumConstrs,
                     "approximation_variables": approximation.model.NumVars,
                     "approximation_rows": approximation.model.NumConstrs}
        finally:
            original.model.dispose()
            approximation.model.dispose()
    manifest = {"version": _VERSION, "directions_sha256": _array_hash(directions),
                "region_sha256": region_hash, "model_sha256": physical_hash,
                "solver_options": SOLVER_OPTIONS, "gurobi_version": list(gp.gurobi.version())}
    total = len(directions)
    feasibility, optimality = np.empty(total), np.empty(total)
    prefix, old_elapsed, solver_seconds, solves, retry_count = 0, 0., 0., 0, 0
    retry_records = []
    checkpoint = output / "evaluation_checkpoint.npz"
    if checkpoint.is_file():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if json.loads(str(saved["manifest"].item())) != manifest:
                raise ValueError("Sequential checkpoint manifest differs")
            prefix = int(saved["completed"].item())
            if not 0 <= prefix <= total:
                raise ValueError("Invalid sequential prefix count")
            for key, target in (("error_feas", feasibility), ("error_opt", optimality)):
                values = saved[key]
                if values.shape != (prefix,) or not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError(f"Invalid sequential prefix {key}")
                target[:prefix] = values
            old_elapsed = float(saved["elapsed_seconds"].item())
            solver_seconds = float(saved["solver_seconds"].item())
            solves = int(saved["solves"].item()) if "solves" in saved else 4 * prefix
            retry_count = int(saved["retry_count"].item()) if "retry_count" in saved else 0
            retry_records = json.loads(str(saved["retry_records"].item())) if "retry_records" in saved else []
            if (not np.isfinite(old_elapsed) or old_elapsed < 0
                    or not np.isfinite(solver_seconds) or solver_seconds < 0
                    or solves < 4 * prefix or retry_count < 0 or not isinstance(retry_records, list)):
                raise ValueError("Invalid sequential checkpoint timing or retry data")
    setup_seconds = time.perf_counter() - started
    blocks = []
    for indices in np.array_split(np.arange(prefix, total), min(workers, total - prefix) or 1):
        if len(indices):
            blocks.append((int(indices[0]), int(indices[-1]) + 1))
    done_blocks = {}
    block_build_seconds = []

    def report(event):
        if progress is None:
            return
        completed = prefix
        active_solver = 0.
        active_retries = 0
        for start, end in blocks:
            if (start, end) in done_blocks:
                completed += end - start
                continue
            path = output / f"evaluation_progress_part_{start}_{end}.json"
            if path.is_file():
                record = json.loads(path.read_text(encoding="utf-8"))
                count = int(record["completed"])
                if not 0 <= count <= end - start:
                    raise ValueError("Invalid worker progress count")
                completed += count
                active_solver += float(record["solver_seconds"])
                active_retries += int(record["retry_count"])
        progress({"event": event, "completed": completed, "total": total,
                  "elapsed_seconds": old_elapsed + time.perf_counter() - started,
                  "solver_seconds": solver_seconds + active_solver,
                  "retry_count": retry_count + active_retries,
                  "build_seconds": setup_seconds, "workers": workers,
                  "retained_sequential_prefix": prefix, **sizes})

    report("ready")
    if blocks:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            pending = {pool.submit(_worker, agg.T, directions[start:end], region,
                                   str(output), start, end): (start, end)
                       for start, end in blocks}
            while pending:
                finished, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in finished:
                    start, end = pending.pop(future)
                    result = future.result()
                    expected = {"native_model_sha256": physical_hash,
                                "native_region_sha256": region_hash,
                                "native_directions_sha256": _array_hash(directions[start:end])}
                    if any(result[key] != value for key, value in expected.items()):
                        raise ValueError("Worker result model, region or direction signature differs")
                    for key, target in (("error_feas", feasibility), ("error_opt", optimality)):
                        values = np.asarray(result[key], dtype=float)
                        if (values.shape != (end - start,) or not np.isfinite(values).all()
                                or np.any(values < 0)):
                            raise ValueError(f"Invalid worker result {key}")
                        target[start:end] = values
                    solver_seconds += float(result["native_solver_seconds"])
                    solves += int(result["native_solves"])
                    retry_count += int(result["native_retry_count"])
                    retry_records.extend(_context_offset(json.loads(result["native_retry_records"]), start))
                    block_build_seconds.append(float(result["native_build_seconds"]))
                    done_blocks[start, end] = True
                report("progress" if pending else "completed")
    else:
        report("completed")
    elapsed_seconds = old_elapsed + time.perf_counter() - started
    _save_checkpoint(checkpoint, manifest, feasibility, optimality, total,
                     elapsed_seconds, solver_seconds, solves=solves,
                     retry_count=retry_count, retry_records=retry_records)
    return {
        "error_feas": feasibility, "error_opt": optimality,
        "mean_feas": float(feasibility.mean()), "mean_opt": float(optimality.mean()),
        "max_feas": float(feasibility.max()), "max_opt": float(optimality.max()),
        "count_feas_gt_1e4": int(np.count_nonzero(feasibility > 1e-4)),
        "num_sample": total, "direction_seed": None,
        **{"native_" + key: value for key, value in sizes.items()},
        "native_build_seconds": setup_seconds + max(block_build_seconds, default=0.),
        "native_solver_seconds": solver_seconds,
        "native_elapsed_seconds": elapsed_seconds,
        "native_solves": solves, "native_successful_solves": 4 * total,
        "native_retry_count": retry_count,
        "native_retry_policy": json.dumps(RETRY_POLICY, sort_keys=True),
        "native_retry_records": json.dumps(retry_records, sort_keys=True),
        "native_directions_sha256": manifest["directions_sha256"],
        "native_region_sha256": region_hash, "native_model_sha256": physical_hash,
        "native_workers": workers, "native_retained_sequential_prefix": prefix,
    }


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def metric_stats(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Invalid evaluated squared distances")
    return {"count": len(values), "mean": float(values.mean()),
            "max": float(values.max()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95))}


def main_evaluate(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the retained 500-device T24/T96 PEBs without retraining.")
    parser.add_argument("--horizon", type=int, choices=(24, 96), required=True)
    parser.add_argument("--t24-mode", choices=("exact", "approx"), default="approx",
                        help="Saved T24 gradient mode; T96 always uses its retained approximate-gradient model")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent direction batches; each Gurobi solve remains single-threaded")
    args = parser.parse_args(argv)
    from Simulator.runners.train_aggregation import build_aggregator, _main_scenario
    from Simulator.testers.case_methods import load_polytope, verify_scenario
    from Simulator.testers.polytope_evaluation import SOLVER_OPTIONS, evaluate_native

    if args.samples <= 0:
        parser.error("samples must be positive")
    if args.workers <= 0:
        parser.error("workers must be positive")
    horizon = args.horizon
    output = RESULTS_ROOT / "aggregation" / f"T{horizon}"
    output /= f"large_scale/{args.t24_mode}/peb" if horizon == 24 else "approx/peb"
    if not output.is_dir():
        raise FileNotFoundError(output)
    protected = {str(path.relative_to(output)): file_hash(path)
                 for path in output.rglob("*") if path.is_file()
                 and not path.name.startswith("evaluation_")}
    agg = build_aggregator(horizon, seed=0, counts=(100, 200, 200))
    agg.case_aggregator(build_only=True)
    if horizon == 24:
        training = json.loads((output / "training_config.json").read_text(encoding="utf-8"))
        if training.get("status") != "completed" or training.get("gradient_mode") != args.t24_mode:
            raise ValueError("Saved T24 training is incomplete or differs from the requested gradient mode")
        scenario_audit = verify_scenario(output, agg, 0, required=True)
    elif (output / "training_config.json").is_file():
        training = json.loads((output / "training_config.json").read_text(encoding="utf-8"))
        expected = {"status": "completed", "gradient_mode": "approx", "horizon": 96,
                    "counts": [100, 200, 200]}
        if any(training.get(key) != value for key, value in expected.items()):
            raise ValueError("Saved T96 training is incomplete or its mode/scene differs")
        scenario_audit = verify_scenario(output, agg, 0, required=True)
    else:
        # Historical case recovery must have independently verified its inputs,
        # generator/constraints and initial support bounds before evaluation.
        audit_path = output / "evaluation_historical_recovery.json"
        scenario_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if scenario_audit.get("status") != "passed":
            raise ValueError("Historical T96 recovery has not passed validation")
        if scenario_audit["reconstructed_scenario_sha256"] != _main_scenario(agg, 0)["scenario_sha256"]:
            raise ValueError("Reconstructed historical scene differs from recovery audit")
    scene = _main_scenario(agg, 0)
    region = load_polytope(output)
    directions = np.random.RandomState(args.seed).randn(args.samples, horizon)
    train_path = output / "training_directions.npy"
    if train_path.exists():
        held_out = {row.tobytes() for row in directions}
        if any(row.tobytes() in held_out for row in np.load(train_path)):
            raise ValueError("Training and evaluation directions overlap")
    np.save(output / "evaluation_directions.npy", directions)
    write_json(output / "evaluation_scenario.json", scene)
    source_files = [Path(__file__),
                    CODE_ROOT / "Simulator/testers/polytope_evaluation.py",
                    CODE_ROOT / "Simulator/cases/aggregation.py"]
    config = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "horizon": horizon, "counts": [100, 200, 200], "devices": 500,
              "gradient_mode": args.t24_mode if horizon == 24 else "approx",
              "model_directory": str(output), "samples": args.samples,
              "direction_seed": args.seed, "direction_distribution": "standard normal",
              "evaluation_workers": args.workers,
              "case_seed": 0, "metric": "sum of squared normalized-power distances",
              "metric_sqrt_applied": False, "metric_divided_by_T": False,
              "solver_options": dict(SOLVER_OPTIONS),
              "scenario_sha256": scene["scenario_sha256"], "scenario_audit": scenario_audit,
              "protected_training_file_sha256": protected,
              "source_sha256": {str(p.relative_to(CODE_ROOT)): file_hash(p) for p in source_files}}
    write_json(output / "evaluation_config.json", config)

    def progress(record):
        write_json(output / "evaluation_progress.json", record)
        print(json.dumps({"horizon": horizon, **record}), flush=True)

    try:
        if args.workers == 1:
            result = evaluate_native(agg, directions, region, progress=progress,
                                     checkpoint_path=output / "evaluation_checkpoint.npz")
        else:
            result = evaluate_parallel(agg, directions, region, output, progress=progress,
                                       workers=args.workers)
        result["direction_seed"] = args.seed
        np.savez(output / "evaluation_errors.npz", **result)
        stats = {"status": "completed", "horizon": horizon, "devices": 500,
                 "gradient_mode": config["gradient_mode"],
                 "samples": args.samples, "direction_seed": args.seed,
                 "metric": config["metric"],
                 "feasibility": metric_stats(result["error_feas"]),
                 "optimality": metric_stats(result["error_opt"]),
                 "evaluation_seconds": float(result["native_elapsed_seconds"]),
                 "scenario_sha256": scene["scenario_sha256"],
                 "errors_file_sha256": file_hash(output / "evaluation_errors.npz")}
        for name, digest in protected.items():
            if file_hash(output / name) != digest:
                raise RuntimeError(f"Training input changed during evaluation: {name}")
        stats["protected_training_files_unchanged"] = True
        write_json(output / "evaluation_summary.json", stats)
        config.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "evaluation_config.json", config)
        print(json.dumps(stats), flush=True)
    except BaseException as error:
        config.update(status="failed", error=repr(error))
        write_json(output / "evaluation_config.json", config)
        raise


def _scaling_load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _scaling_describe_training(training, horizon):
    stages = training["stages"]
    updates = "/".join(str(stage.get("epochs", stage["n_train"] + 1)) for stage in stages)
    rates = "/".join(f"{stage['lr']:g}" for stage in stages)
    return (f"T={horizon}采用{training['dtype'].removeprefix('torch.')}、"
            f"训练seed={training['training_seed']}，共{training['optimizer_updates']}次更新；"
            f"三阶段分别为{updates}次，初始学习率为{rates}。")


def summarize_scaling(results_root=RESULTS_ROOT, output_dir=None):
    cases = []
    for horizon, relative in ((24, "aggregation/T24/large_scale/approx/peb"),
                              (96, "aggregation/T96/approx/peb")):
        directory = results_root / relative
        summary = _scaling_load(directory / "evaluation_summary.json")
        evaluation = _scaling_load(directory / "evaluation_config.json")
        assert summary["status"] == evaluation["status"] == "completed"
        assert summary["samples"] == evaluation["samples"] == 5000
        assert summary["direction_seed"] == evaluation["direction_seed"] == 1
        assert summary["devices"] == 500 and summary["horizon"] == horizon
        assert summary["protected_training_files_unchanged"]
        model_path = directory / "final_polytope.npz"
        if not model_path.is_file():
            model_path = directory / "pretrainnet_weights.pth"
        assert hashlib.sha256(model_path.read_bytes()).hexdigest() == evaluation[
            "protected_training_file_sha256"][model_path.name]
        assert _scaling_load(directory / "evaluation_extrema_recheck.json")["status"] == "passed"
        np.testing.assert_array_equal(np.load(directory / "evaluation_directions.npy"),
                                      np.random.RandomState(1).randn(5000, horizon))
        with np.load(directory / "evaluation_errors.npz", allow_pickle=False) as data:
            assert int(data["num_sample"]) == 5000
            assert int(data["native_successful_solves"]) == 20000
            for label, key in (("feasibility", "error_feas"), ("optimality", "error_opt")):
                values = data[key]
                assert values.shape == (5000,) and np.isfinite(values).all() and np.all(values >= 0)
                assert float(values.mean()) == summary[label]["mean"]
                assert float(values.max()) == summary[label]["max"]
        cases.append(summary)

    training = _scaling_load(results_root / "aggregation/T24/large_scale/approx/peb/training_config.json")
    assert training["status"] == "completed" and training["gradient_mode"] == "approx"
    assert training["optimizer_updates"] == 1000 and training["dtype"] == "torch.float64"
    assert training["counts"] == [100, 200, 200]
    seconds24 = float(training["elapsed_seconds"])
    training96_path = results_root / "aggregation/T96/approx/peb/training_config.json"
    estimated96 = not training96_path.is_file()
    if not estimated96:
        training96 = _scaling_load(training96_path)
        assert training96["status"] == "completed" and training96["gradient_mode"] == "approx"
        assert training96["horizon"] == 96 and training96["counts"] == [100, 200, 200]
        seconds96 = float(training96["elapsed_seconds"])
        assert np.isfinite(seconds96) and seconds96 > 0
        timing_note = ("**训练计时。** 两个场景均采用本次实际训练循环计时，包含回调与快照保存，"
                       "不含前置准备和后续误差评估。T=96已使用重新训练记录的耗时，"
                       "没有继续采用历史文件时间戳估算值。")
        training96_text = _scaling_describe_training(training96, 96)
        source96 = ("- T=96：[训练配置](../aggregation/T96/approx/peb/training_config.json)、"
                    "[评估汇总](../aggregation/T96/approx/peb/evaluation_summary.json)、"
                    "[物理场景](../aggregation/T96/approx/peb/scenario.json)。")
    else:
        recovery = _scaling_load(results_root / "aggregation/T96/approx/peb/evaluation_historical_recovery.json")
        assert recovery["status"] == "passed" and recovery["recorded_updates"] == 1003
        weights = results_root / "aggregation/T96/approx/peb/pretrainnet_weights.pth"
        assert hashlib.sha256(weights.read_bytes()).hexdigest() == recovery["final_network_recovery"]["weights_sha256"]
        prefix = "results/aggregation/T96/approx/peb/"
        with zipfile.ZipFile(CODE_ROOT.parent / "code.zip") as archive:
            initial = datetime(*archive.getinfo(prefix + "snapshots/A_0.csv").date_time)
            finish = datetime(*archive.getinfo(prefix + "error_history.pkl").date_time)
        seconds96 = (finish - initial).total_seconds()
        timing_note = (f"* **T=96训练时间为估算值。** 历史结果未保存精确计时；归档初始快照时间为"
                       f"{initial:%Y-%m-%d %H:%M:%S}，最终误差记录和权重时间为{finish:%Y-%m-%d %H:%M:%S}，"
                       f"相差{seconds96:.0f}秒。该文件时间跨度不能作为正式训练计时或独占测速。"
                       "T=24为训练循环的实际计时，包含回调与快照保存，不含前置准备和后续误差评估。")
        training96_text = ("T=96采用历史最终权重，共1003次更新；归档入口的三阶段为301/201/501次，"
                           "初始学习率为0.01/0.05/0.1。T=96的输入文件、设备参数、物理约束和初始支撑边界已经恢复核验。")
        source96 = ("- T=96：[历史记录](../aggregation/T96/approx/peb/migration_info.json)、"
                    "[评估汇总](../aggregation/T96/approx/peb/evaluation_summary.json)、"
                    "[恢复核验](../aggregation/T96/approx/peb/evaluation_historical_recovery.json)；"
                    "时间戳来源为项目根目录`code.zip`。")

    lines = [
        "# 500台连续设备场景：训练时间与误差汇总", "",
        "| 场景 | 训练时间（秒 / 分钟） | 平均可行性误差 | 最大可行性误差 | 平均最优性误差 | 最大最优性误差 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for horizon, seconds, result in zip((24, 96), (seconds24, seconds96), cases):
        timing = f"{seconds:.2f} / {seconds / 60:.2f}"
        if horizon == 96 and estimated96:
            timing = f"约 {seconds:.0f} / {seconds / 60:.2f}*"
        lines.append(f"| T={horizon}，500台连续设备 | {timing} | "
                     f"{result['feasibility']['mean']:.8g} | {result['feasibility']['max']:.8g} | "
                     f"{result['optimality']['mean']:.8g} | {result['optimality']['max']:.8g} |")
    lines += [
        "",
        timing_note,
        "",
        "**训练设置。** 两者均采用PEB初始化、近似梯度，设备组成为100 CL、200 TCL、200 PV，"
        "设备场景seed=0。" + _scaling_describe_training(training, 24) + training96_text +
        "各阶段分别重新初始化SGD及StepLR调度器，阶段内每100次更新将学习率乘以0.98。",
        "",
        "**误差测试。** 每个场景使用`RandomState(1).randn(5000,T)`生成5000个标准正态方向。"
        "可行性误差是近似域支撑点到原始可行域的投影距离平方；最优性误差是原始域支撑点到近似域的投影距离平方。"
        "两项均在归一化功率坐标中计算平方欧氏距离，未开方、未除以T，不是百分比。"
        "T=24和T=96分别对24与96个分量求和。表中最大值为本次5000个方向的样本最大值。",
        "",
        "**数值核验。** 每方向实际求解两次LP、两次QP；Gurobi的Threads=1、Seed=0，"
        "FeasibilityTol=1e-9、OptimalityTol=1e-9、BarConvTol=1e-12。各场景的两项最大值对应方向均已通过原有Pyomo路径独立复核，"
        "误差未截断为零，评估期间训练文件哈希保持不变。",
        "",
        "**数据来源。**",
        "",
        "- T=24：[训练配置](../aggregation/T24/large_scale/approx/peb/training_config.json)、"
        "[评估汇总](../aggregation/T24/large_scale/approx/peb/evaluation_summary.json)。",
        source96,
        "- 两个模型目录的`evaluation_directions.npy`和`evaluation_errors.npz`分别保留全部方向与逐样本误差。"
        "可用[统一汇总入口](../../Simulator/testers/case_scaling.py)的 `summarize` 子命令重新生成本文件。",
    ]
    target = (Path(output_dir) if output_dir is not None else results_root / "figures") / "500_device_training_and_error_summary.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target

def main_summarize(argv=None):
    parser = argparse.ArgumentParser(description="Summarize the two saved 500-device continuous scaling cases.")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    print(summarize_scaling(args.results_root, args.output_dir))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "evaluate":
        return main_evaluate(argv[1:])
    if argv and argv[0] == "summarize":
        return main_summarize(argv[1:])
    if argv and argv[0] not in ("-h", "--help") and argv[0].startswith("-"):
        return main_evaluate(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("evaluate", help="Evaluate T24 or T96 in independent directions")
    commands.add_parser("summarize", help="Recompute the saved scaling summary without solving")
    if not argv:
        parser.print_help()
        return
    parser.parse_args(argv)


if __name__ == "__main__":
    main()
