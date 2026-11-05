"""Recalculate saved paper results through six explicit subcommands.

Examples from the repository root:
    python -m Simulator.testers.summarize_results paper-tables
    python -m Simulator.testers.summarize_results scaling
    python -m Simulator.testers.summarize_results nonconvexity --input-dir PATH
    python -m Simulator.testers.summarize_results scale3-percentage
    python -m Simulator.testers.summarize_results scale3-nonconvexity
    python -m Simulator.testers.summarize_results uptime-ablation

The calculations are retained from the former six summary entry points.
Scale3 calculations retain the source_snapshot/tmp provenance of
summarize_microgrid_scale3_percentage.py,
summarize_current_scale3_nonconvexity.py and
summarize_scale3_min_uptime_ablation.py. No training or optimization is run.
Only scale3-nonconvexity loads the physical-model module, and does so lazily.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from Simulator.paths import CODE_ROOT, RESULTS_ROOT


_PAPER_METHODS = (
    ("proposed_peb_exact", "Proposed method (PEB initialization)"),
    ("peb_inner", "PEB inner"),
    ("peb_outer", "PEB outer"),
    ("federated_peb", "Federated PEB"),
    ("proposed_pb_exact", "Proposed method (PB initialization)"),
    ("pb_inner", "PB inner"),
    ("cutting_plane", "Cutting plane (original)"),
    ("cutting_plane_modified", "Cutting plane (modified)"),
    ("pve", "PVE (original)"),
    ("pve_modified", "PVE (modified)"),
)


# Manuscript tables
def summarize_paper_tables(results: Path) -> dict:
    sources = {}

    def source(relative):
        path = results / relative
        sources[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read_json(relative):
        return json.loads(source(relative).read_text(encoding="utf-8"))

    def vector(relative):
        array = np.asarray(np.load(source(relative), allow_pickle=False), dtype=float)
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError(f"Expected finite vector: {relative}")
        return array

    def schedule(relative):
        config = read_json(relative)
        if config.get("status") != "completed":
            raise ValueError(f"Incomplete training: {relative}")
        rows, start = [], 1
        calls = config["train_calls"]
        if len(calls) != len(config["stages"]):
            raise ValueError(f"Stage and executed-call counts differ: {relative}")
        for i, (stage, call) in enumerate(zip(config["stages"], calls), 1):
            # A microgrid epoch contains multiple minibatch optimizer updates.
            # The executed per-stage counts, not raw epochs, define Table V.
            updates = int(call["optimizer_updates"])
            ratio = float(stage["rate_opt_feas"])
            rows.append({"stage": i, "first_update": start,
                         "last_update": start + updates - 1,
                         "updates": updates, "learning_rate": float(stage["lr"]),
                         "saved_optimality_to_feasibility_ratio": ratio,
                         "equivalent_lambda": 1.0 / (1.0 + ratio)})
            start += updates
        if (start - 1 != int(config["optimizer_updates"])
                or [row["last_update"] for row in rows] != config["phase_end_updates"]):
            raise ValueError(f"Executed stage counts disagree with saved totals: {relative}")
        return {"source": relative, "rows": rows,
                "optimizer_updates": int(config["optimizer_updates"]),
                "elapsed_seconds": float(config["elapsed_seconds"])}

    table_i = schedule("aggregation/T24/exact/peb/training_config.json")
    table_v = schedule("microgrid/training/training_config.json")
    folder = "aggregation/T24/comparisons"
    original = vector(f"{folder}/original_costs.npy")
    if len(original) != 365:
        raise ValueError("Table II requires 365 daily costs")
    saved = read_json(f"{folder}/summary.json")
    table_ii = []
    for key, name in _PAPER_METHODS:
        costs = vector(f"{folder}/{key}_costs.npy")
        if costs.shape != original.shape:
            raise ValueError(f"Daily cost shape mismatch: {key}")
        mean = float(np.mean(costs - original))
        np.testing.assert_allclose(mean, saved[key]["mean_cost_difference"], atol=1e-10, rtol=1e-10)
        table_ii.append({"method": name, "key": key, "days": len(costs),
                         "mean_cost_deviation_usd_day": mean})

    table_iii = []
    for horizon, relative in (
        (24, "aggregation/T24/large_scale/approx/peb"),
        (96, "aggregation/T96/approx/peb"),
    ):
        config = read_json(f"{relative}/training_config.json")
        if config.get("status") != "completed":
            raise ValueError(f"Incomplete training: {relative}")
        with np.load(source(f"{relative}/evaluation_errors.npz"), allow_pickle=False) as data:
            feas = np.asarray(data["error_feas"], dtype=float)
            opt = np.asarray(data["error_opt"], dtype=float)
        if feas.shape != (5000,) or opt.shape != (5000,) or not np.isfinite(feas).all() or not np.isfinite(opt).all():
            raise ValueError(f"Expected 5000 finite paired errors: {relative}")
        table_iii.append({"horizon": horizon, "devices": 500,
                          "training_seconds": float(config["elapsed_seconds"]),
                          "actual_optimizer_updates": int(config["optimizer_updates"]),
                          "samples": len(feas), "mean_feasibility": float(feas.mean()),
                          "max_feasibility": float(feas.max()),
                          "mean_optimality": float(opt.mean()), "max_optimality": float(opt.max())})

    table_iv = []
    for label, relative, sample_file, summary_file in (
        ("7 devices / 4 h", "microgrid/nonconvexity", "samples.csv", "summary.json"),
        ("21 devices / 4 h", "microgrid/scale3/nonconvexity", "samples.csv", "summary.json"),
        ("21 devices / removed", "microgrid/scale3/nonconvexity_no_min_uptime", "paired_samples.csv", "paired_summary.json"),
    ):
        with source(f"{relative}/{sample_file}").open(encoding="utf-8", newline="") as stream:
            samples = list(csv.DictReader(stream))
        lower = np.asarray([float(row["lower_percent"]) for row in samples])
        upper = np.asarray([float(row["upper_percent"]) for row in samples])
        if len(samples) != 5000 or not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError(f"Expected 5000 finite certified probe intervals: {relative}")
        old = read_json(f"{relative}/{summary_file}")["groups"]["overall"]
        np.testing.assert_allclose([lower.mean(), lower.max(), upper.mean(), upper.max()],
                                   [old["lower_percent"]["mean"], old["lower_percent"]["max"],
                                    old["upper_percent"]["mean"], old["upper_percent"]["max"]],
                                   atol=1e-12, rtol=1e-10)
        table_iv.append({"portfolio_minimum_on_time": label, "samples": len(samples),
                         "mean_gap_percent": float(upper.mean()), "max_gap_percent": float(upper.max()),
                         "mean_gap_lower_percent": float(lower.mean()),
                         "max_gap_lower_percent": float(lower.max()),
                         "sample_file": f"{relative}/{sample_file}"})

    expected_cost = [8.65, 9.82, -7.22, 56.08, 27.08, 65.29, -11.10, -5.77, 16.76, 34.37]
    if [round(row["mean_cost_deviation_usd_day"], 2) for row in table_ii] != expected_cost:
        raise ValueError("Table II differs from the manuscript at its reported precision")
    expected_gap = [(0.0351, 1.0849), (0.0113, 0.3671), (0.0001, 0.0480)]
    if [(round(row["mean_gap_percent"], 4), round(row["max_gap_percent"], 4)) for row in table_iv] != expected_gap:
        raise ValueError("Table IV differs from the manuscript at its reported precision")
    expected_scale = [
        (741.34, "2.66e-13", "1.33e-09", "0.2083", "0.5770"),
        (3262.06, "1.62e-08", "3.19e-05", "0.2800", "0.5411"),
    ]
    actual_scale = [(round(r["training_seconds"], 2), f'{r["mean_feasibility"]:.2e}',
                     f'{r["max_feasibility"]:.2e}', f'{r["mean_optimality"]:.4f}',
                     f'{r["max_optimality"]:.4f}') for r in table_iii]
    if actual_scale != expected_scale:
        raise ValueError("Table III differs from the manuscript at its reported precision")
    return {
        "scope": "Saved-data recalculation only; no solver or training calls",
        "table_I": table_i, "table_II": table_ii, "table_III": table_iii,
        "table_IV": table_iv, "table_V": table_v,
        "validation": {"tables_II_III_IV_match_manuscript_reported_precision": True},
        "notes": [
            "Cost is each method's saved native daily optimum minus the same day's original physical optimum.",
            "Table III errors are normalized squared Euclidean distances, not percentage or RMS distances.",
            "Table IV uses upper certified distances for reported values; lower interval endpoints are retained too.",
            "The removed-minimum-on-time row uses paired_samples.csv; generic samples.csv is a different experiment.",
            "Stage lambda is shown as 1/(1+saved_optimality_to_feasibility_ratio); the manuscript rounds these weights.",
            "The saved T96 run has 1003 updates. Historical counts, timings and array values are preserved.",
            "Manuscript gradient timings 197.215/195.594 seconds were author supplied; current retained run logs differ.",
            "Budget-study logs contain 1003/2006 updates and 225.768/461.017 seconds, versus manuscript 1000/2000 and 197.2/390.7.",
        ],
        "source_sha256": sources,
    }


def _paper_tabulations(data):
    blocks = []
    for key in ("I", "V"):
        rows = [[str(r["stage"]), f'{r["first_update"]}--{r["last_update"]}',
                 f'{r["equivalent_lambda"]:.8g}', f'{r["learning_rate"]:g}',
                 f'{r["saved_optimality_to_feasibility_ratio"]:g}'] for r in data[f"table_{key}"]["rows"]]
        blocks.append((f"Table {key}: saved training schedule", ["Stage", "Updates", "Equivalent lambda", "Learning rate", "Opt/feas ratio"], rows))
    blocks.insert(1, ("Table II: mean cost deviations", ["Method", "USD/day"],
                     [[r["method"], f'{r["mean_cost_deviation_usd_day"]:.2f}'] for r in data["table_II"]]))
    blocks.insert(2, ("Table III: 500 continuous devices", ["T", "Training (s)", "Mean feasibility", "Max feasibility", "Mean optimality", "Max optimality"],
                     [[str(r["horizon"]), f'{r["training_seconds"]:.2f}', f'{r["mean_feasibility"]:.2e}',
                       f'{r["max_feasibility"]:.2e}', f'{r["mean_optimality"]:.4f}', f'{r["max_optimality"]:.4f}'] for r in data["table_III"]]))
    blocks.insert(3, ("Table IV: sampled nonconvexity gaps", ["Portfolio / MT minimum on-time", "Mean gap (%)", "Max gap (%)"],
                     [[r["portfolio_minimum_on_time"], f'{r["mean_gap_percent"]:.4f}', f'{r["max_gap_percent"]:.4f}'] for r in data["table_IV"]]))
    return blocks


def _paper_write_outputs(data, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "paper_tables.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = ["# Manuscript tables recalculated from retained results", "", data["scope"], ""]
    latex = ["% Saved-data recalculation. Requires booktabs. See paper_tables.json for hashes and precision checks."]
    for title, headings, rows in _paper_tabulations(data):
        markdown += [f"## {title}", "", "| " + " | ".join(headings) + " |", "| " + " | ".join(["---"] * len(headings)) + " |"]
        markdown += ["| " + " | ".join(row) + " |" for row in rows] + [""]
        escape = lambda value: value.replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")
        latex += [r"\begin{table}[htbp]", r"\centering", r"\caption{" + escape(title) + "}",
                  r"\begin{tabular}{" + "l" * len(headings) + "}", r"\toprule",
                  " & ".join(map(escape, headings)) + r" \\", r"\midrule"]
        latex += [" & ".join(map(escape, row)) + r" \\" for row in rows]
        latex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    markdown += ["## Provenance notes", ""] + ["- " + note for note in data["notes"]]
    (output / "paper_tables.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (output / "paper_tables.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


# Continuous-resource scaling summary
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
        "可用[统一汇总入口](../../Simulator/testers/summarize_results.py)的 `scaling` 子命令重新生成本文件。",
    ]
    target = (Path(output_dir) if output_dir is not None else results_root / "figures") / "500_device_training_and_error_summary.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


# Nonconvexity reporting thresholds
def _nonconvexity_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nonconvexity_read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _nonconvexity_write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def summarize_nonconvexity(input_dir, threshold_percent=0.01):
    import pandas as pd
    from Simulator.testers._nonconvexity_sampling import build_summary

    folder = Path(input_dir)
    if not np.isfinite(threshold_percent) or threshold_percent <= 0:
        raise ValueError("Reporting threshold must be finite and positive")
    meta = _nonconvexity_read_json(folder / "metadata.json")
    previous = _nonconvexity_read_json(folder / "summary.json")
    checkpoint = _nonconvexity_read_json(folder / "checkpoint.json")
    if meta["status"] != "completed" or meta["completed"] != 5000:
        raise ValueError("A completed 5000-probe experiment is required")
    for name, expected in checkpoint["files"].items():
        if _nonconvexity_digest(folder / name) != expected:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
    immutable_names = ("samples.csv", "endpoints.npz", "probes.npz", "projections.npz",
                       "projection_diagnostics.json", "controls.npz", "controls.csv",
                       "controls_diagnostics.json")
    hashes = {name: _nonconvexity_digest(folder / name) for name in immutable_names}
    metadata_before = _nonconvexity_digest(folder / "metadata.json")
    rows = pd.read_csv(folder / "samples.csv").to_dict("records")
    for row in rows:
        row["classification"] = ("below_threshold" if row["upper_percent"] <= threshold_percent
                                 else "detected_gap" if row["lower_percent"] > threshold_percent
                                 else "undetermined")
    summary = {**previous, **build_summary(rows, threshold_percent=threshold_percent),
               "numerical_threshold_percent": meta["protocol"]["threshold_percent"],
               "reporting_threshold_percent": threshold_percent,
               "reporting_note": "Strict lower-bound exceedance; upper bounds at or below the reporting threshold are below_threshold, not numerical zeros. Original numerical controls and sample classifications are unchanged."}
    for side in ("lower_percent", "upper_percent"):
        for field in ("mean", "median", "p95", "max"):
            np.testing.assert_allclose(summary["groups"]["overall"][side][field],
                                       previous["groups"]["overall"][side][field], rtol=1e-12, atol=1e-14)
    statistics = []
    for name, group in summary["groups"].items():
        count = group["count"]
        detected = group["classification_counts"].get("detected_gap", 0)
        entry = {"group": name, "count": count, "detected_count": detected,
                 "detected_fraction": detected / count if count else None,
                 "undetermined_count": group["classification_counts"].get("undetermined", 0)}
        for statistic in ("mean", "median", "p95", "max"):
            for side in ("lower", "upper"):
                entry[f"{statistic}_{side}_percent"] = group[f"{side}_percent"][statistic]
        statistics.append(entry)
    nested = [{"sample_count": n,
               "max_lower_percent": max(r["lower_percent"] for r in rows[:n]),
               "max_upper_percent": max(r["upper_percent"] for r in rows[:n]),
               "detected_count": sum(r["classification"] == "detected_gap" for r in rows[:n])}
              for n in (500, 1000, 2000, 5000)]
    _nonconvexity_write_json(folder / "summary.json", summary)
    pd.DataFrame(statistics).to_csv(folder / "statistics.csv", index=False)
    pd.DataFrame(nested).to_csv(folder / "sample_count_comparison.csv", index=False)
    meta["reporting_threshold_percent"] = threshold_percent
    _nonconvexity_write_json(folder / "metadata.json", meta)
    checkpoint["files"] = {name: _nonconvexity_digest(folder / name) for name in checkpoint["files"]}
    _nonconvexity_write_json(folder / "checkpoint.json", checkpoint)
    if hashes != {name: _nonconvexity_digest(folder / name) for name in immutable_names}:
        raise RuntimeError("Original optimization results changed during summarization")
    _nonconvexity_write_json(folder / "reporting_statistics_update.json", {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "source_fingerprint": meta["fingerprint"], "metadata_before_sha256": metadata_before,
        "metadata_after_sha256": _nonconvexity_digest(folder / "metadata.json"),
        "previous_reporting_threshold_percent": previous["threshold_percent"],
        "reporting_threshold_percent": threshold_percent,
        "numerical_threshold_percent": meta["protocol"]["threshold_percent"],
        "optimization_rerun": False, "unchanged_raw_files_sha256": hashes,
        "classification_counts": summary["groups"]["overall"]["classification_counts"],
        "postprocessing_source_sha256": _nonconvexity_digest(Path(__file__))})
    return summary["groups"]["overall"]["classification_counts"]


# Scale3 relative disaggregation percentages
def summarize_scale3_percentage(args):
    from pathlib import Path
    import hashlib
    import json
    import pickle

    import numpy as np


    from Simulator.paths import CODE_ROOT
    ROOT = CODE_ROOT
    BASE = args.base_dir
    EVAL = BASE / "disaggregation"
    output = args.output_dir or EVAL
    output.mkdir(parents=True, exist_ok=True)
    sources = [EVAL / "results.pkl", EVAL / "metadata.json", EVAL / "errors.csv",
               BASE / "training/pretrainnet_weights.pth", BASE / "training/training_config.json"]
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    metadata = json.loads((EVAL / "metadata.json").read_text(encoding="utf-8"))
    with (EVAL / "results.pkl").open("rb") as stream:
        rows = pickle.load(stream)
    assert metadata["status"] == "completed" and len(rows) == 5000
    assert all(row["solver_status"] == "completed" for row in rows)
    command = np.asarray([row["power_curve"] for row in rows], dtype=float)
    projected = np.asarray([row["projected_power_curve"] for row in rows], dtype=float)
    physical_sse = np.asarray([row["feasibility_error"] for row in rows], dtype=float)
    assert command.shape == projected.shape == (5000, 24)
    assert np.isfinite(command).all() and np.isfinite(projected).all()
    trajectory_sse = np.sum((command - projected) ** 2, axis=1)
    np.testing.assert_allclose(physical_sse, trajectory_sse, rtol=1e-10, atol=1e-12)
    denominator = np.sum(command ** 2, axis=1)
    assert np.isfinite(physical_sse).all() and np.all(physical_sse >= 0)
    assert np.isfinite(denominator).all() and np.all(denominator > 0)
    percentages = 100 * np.sqrt(physical_sse / denominator)
    summary = {
        "status": "completed", "method": "postprocess_existing_trajectories",
        "additional_solver_calls": 0, "T": 24, "device_counts": metadata["device_counts"],
        "training_conservativeness": metadata["training_conservativeness"],
        "evaluation_conservativeness": metadata["evaluation_conservativeness"],
        "seed": metadata["seed"], "noise_scale": metadata["noise_scale"],
        "mean_shifts": {name: metadata[name] for name in ("pv_shift", "cl_shift", "temp_shift")},
        "metric": "relative_disaggregation_error_percent", "unit": "percent",
        "formula": "100*sqrt(sum((P_command-P_projected)**2)/sum(P_command**2))",
        "projection_definition": "Reuse the original projection minimizing squared distance in fixed nominal normalized power coordinates.",
        "valid_samples": int(len(percentages)), "failed_samples": metadata["failed_samples"],
        "zero_denominator_samples": int(np.count_nonzero(denominator == 0)),
        "mean": float(percentages.mean()), "max": float(percentages.max()),
        "max_sample_index": int(rows[int(np.argmax(percentages))]["sample_index"]),
        "physical_squared_error_recompute_max_abs_difference": float(np.max(np.abs(physical_sse - trajectory_sse))),
        "source_sha256": hashes,
    }
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
    (output / "relative_disaggregation_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("valid_samples", "mean", "max", "additional_solver_calls")}, indent=2))


# Scale3 nonconvexity and ESS ledgers
def summarize_scale3_nonconvexity(args):
    from pathlib import Path
    import json
    import sys
    import numpy as np
    import pandas as pd

    from Simulator.testers.evaluate_microgrid_nonconvexity import load_case, write_json

    folder = args.base_dir / 'nonconvexity'
    output = args.output_dir or folder
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    assert meta['status'] == 'completed' and meta['completed'] == 5000
    assert meta['reporting_threshold_percent'] == 0.01
    mg = load_case(args.base_dir / 'training', device_scale=3)
    with np.load(folder / 'endpoints.npz') as z:
        names = z['variable_names'].tolist()
        groups = {'raw_endpoints': z['raw_primal']}
    with np.load(folder / 'projections.npz') as z:
        groups['formal_projections'] = z['primal']
    with np.load(folder / 'controls.npz') as z:
        groups.update({key: z[key] for key in ('endpoint_primal', 'feasible_witnesses', 'primal')})
    checks = {}
    for label, primal in groups.items():
        maximum = 0.0
        for j, ess in enumerate(mg.ESS_list):
            ch = primal[:, [names.index(f'p_chg_ESS[{j},{t}]') for t in range(24)]]
            dis = primal[:, [names.index(f'p_dis_ESS[{j},{t}]') for t in range(24)]]
            soc = primal[:, [names.index(f'e_ESS[{j},{t}]') for t in range(25)]]
            expected = np.c_[np.full(len(primal), ess['init_SOC']),
                             ess['init_SOC'] + np.cumsum(
                                 mg.deltaT * (ess['eta_chg'] * ch - dis / ess['eta_dis'])
                                 / ess['E_rated'], axis=1)]
            maximum = max(maximum, float(np.max(np.abs(soc - expected))))
            assert np.min(soc) >= ess['SOC_min'] - 1e-8
            assert np.max(soc) <= ess['SOC_max'] + 1e-8
            assert np.min(soc[:, -1]) >= ess['init_SOC'] - 1e-8
        assert maximum < 1e-8
        checks[label] = {'passed': True, 'ESS_trajectories_checked': len(primal) * 3,
                         'max_cumulative_SOC_residual': maximum}
    write_json(output / 'independent_ESS_energy_verification.json',
               {'passed': True, 'physical_intervals': 24, 'state_boundaries': 25,
                'checks': checks, 'all_first_and_last_interval_powers_included': True})

    rows = []
    for key, label in [('local', '局部连线'), ('global', '全局连线'),
                       ('multi', '多点组合'), ('overall', '总体')]:
        group = summary['groups'][key]
        c = group['classification_counts']
        row = {'探针类型': label, '样本数': group['count']}
        for field, name in [('mean', '均值'), ('median', '中位数'), ('p95', 'P95'), ('max', '最大gap')]:
            row[f'{name}下界(%)'] = group['lower_percent'][field]
            row[f'{name}上界(%)'] = group['upper_percent'][field]
        row['超过0.01%的点数'] = c.get('detected_gap', 0)
        row['超过0.01%的比例(%)'] = 100 * row['超过0.01%的点数'] / group['count']
        row['阈值判定未确定数'] = c.get('undetermined', 0)
        row['最优性精度未达到数'] = group['convergence_counts']['not_converged']
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / 'gap_table.csv', index=False, encoding='utf-8-sig')

    def value(row, name):
        lo, hi = row[f'{name}下界(%)'], row[f'{name}上界(%)']
        low_text, high_text = f'{lo:.5f}', f'{hi:.5f}'
        return high_text if low_text == high_text else f'[{low_text}, {high_text}]'

    text = ['# 三倍设备场景非凸性诊断', '',
            '24时段；3 MT、3 PV、3 TL、3 CL、3 ESS、6 HP，共21台设备；360个二进制变量。',
            '单台参数与原始7设备算例完全一致；每台ESS为100 kW/250 kWh。固定名义场景，无随机扰动，conservativeness=0；不重新训练。',
            '使用修复MT最短运行时间、ESS全24时段能量递推后的现有设备模型。HP温度动态保持现有实现。',
            '1000个支撑方向(seed=0)生成端点，5000个凸组合探针(seed=1)，100个零间隙对照。',
            '距离单位为每时段固定功率跨度归一化的RMS百分比：100*sqrt(mean(((P-Q)/D)^2))。', '',
            '| 探针类型 | 样本数 | 均值(%) | 中位数(%) | P95(%) | 最大gap(%) | >0.01%点数 | 比例 | 未确定数 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['探针类型']} | {r['样本数']} | {value(r, '均值')} | {value(r, '中位数')} | {value(r, 'P95')} | {value(r, '最大gap')} | {r['超过0.01%的点数']} | {r['超过0.01%的比例(%)']:.2f}% | {r['阈值判定未确定数']} |")
    text += ['', '区间来自求解器目标上下界，非精确零值不修改为零。最大距离下界仅为全域非凸度的抽样下界，不是全域上界；检出比例仅描述本次抽样。',
             '完整原始整数变量在每次正式投影中均自由优化。独立原始约束、MT正向最短运行时间及ESS累计能量检查见同目录验证报告。',
             f"结果指纹：`{meta['fingerprint']}`。"]
    (output / 'gap_table.md').write_text('\n\n'.join(text[:8]) + '\n\n' + '\n'.join(text[8:]) + '\n', encoding='utf-8')
    print(json.dumps({'table': rows, 'ESS_validation': checks}, ensure_ascii=False))


# Minimum-uptime ablation
def summarize_uptime_ablation(args):
    from pathlib import Path
    import json
    import numpy as np
    import pandas as pd

    base = args.base_dir
    old, out = base / 'nonconvexity', base / 'nonconvexity_no_min_uptime'
    output = args.output_dir or out
    output.mkdir(parents=True, exist_ok=True)
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    baseline, paired, fresh = [read(path) for path in
                               (old / 'summary.json', out / 'paired_summary.json', out / 'summary.json')]
    assert read(out / 'metadata.json')['status'] == 'completed'
    assert read(out / 'independent_storage_verification.json')['passed']
    assert paired['physical_validation']['passed']
    previous_samples, paired_samples = [pd.read_csv(path) for path in
                                         (old / 'samples.csv', out / 'paired_samples.csv')]
    worst = int(previous_samples.lower_percent.idxmax())
    rows = []
    for strategy, sampling, data in [('保留4时段最短运行时间', '原5000条轨迹', baseline),
                                    ('移除最短运行时间', '原5000条轨迹（配对）', paired),
                                    ('移除最短运行时间', '重新支撑采样5000条轨迹', fresh)]:
        for group_name, label in [('overall', '总体'), ('local', '局部连线'),
                                  ('global', '全局连线'), ('multi', '多点组合')]:
            group = data['groups'][group_name]
            counts = group['classification_counts']
            row = {'策略': strategy, '测试轨迹': sampling, '组别': label, '样本数': group['count']}
            for field, title in [('mean', '均值'), ('median', '中位数'), ('p95', 'P95'), ('max', '最大gap')]:
                for side, side_label in [('lower', '下界'), ('upper', '上界')]:
                    row[f'{title}{side_label}(%)'] = group[f'{side}_percent'][field]
            row['超过0.01%点数'] = counts.get('detected_gap', 0)
            row['超过0.01%比例(%)'] = 100 * row['超过0.01%点数'] / group['count']
            row['未确定数'] = counts.get('undetermined', 0)
            row['最优性未达到数'] = group['convergence_counts']['not_converged']
            rows.append(row)
    pd.DataFrame(rows).to_csv(output / 'min_uptime_comparison.csv', index=False, encoding='utf-8-sig')

    def display(row, name):
        lo, hi = [row[f'{name}{side}(%)'] for side in ('下界', '上界')]
        l, u = f'{lo:.5f}', f'{hi:.5f}'
        return u if l == u else f'[{l}, {u}]'

    text = ['# 三倍场景：MT最短运行时间消融', '',
            '场景：3 MT、3 PV、3 TL、3 CL、3 ESS、6 HP，24时段，360个二进制变量。',
            '仅移除3台MT共72条最短连续运行时间约束；最低出力20 kW、最高出力100 kW、启停状态转移和全部其他设备约束保留。每台ESS仍为100 kW/250 kWh。',
            '名义场景、无随机扰动、conservativeness=0，原始功率跨度归一化；阈值为0.01%。不重新训练，不松弛整数变量。', '',
            '主对比复用上一轮完全相同的5000个凸组合目标，原端点均在移除约束后的模型中重新验证可行。补充试验重新查询1000个支撑方向(seed=0)，生成5000个凸组合(seed=1)。', '',
            '| 策略 | 测试轨迹 | 均值(%) | P95(%) | 最大gap(%) | >0.01% | 未确定 |',
            '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['组别'] != '总体':
            continue
        text.append(f"| {row['策略']} | {row['测试轨迹']} | {display(row,'均值')} | {display(row,'P95')} | {display(row,'最大gap')} | {row['超过0.01%点数']}/5000（{row['超过0.01%比例(%)']:.2f}%） | {row['未确定数']} |")
    original_mean = baseline['groups']['overall']['upper_percent']['mean']
    new_mean = paired['groups']['overall']['upper_percent']['mean']
    original_max = baseline['groups']['overall']['upper_percent']['max']
    new_max = paired['groups']['overall']['upper_percent']['max']
    text += ['', f'配对均值减少{100*(1-new_mean/original_mean):.2f}%，样本最大值减少{100*(1-new_max/original_max):.2f}%。',
             f'上一轮最大gap的样本索引（从0开始）为{worst}，原gap为{previous_samples.loc[worst,"lower_percent"]:.8f}%，移除约束后距离上界为{paired_samples.loc[worst,"upper_percent"]:.8f}%。',
             '新采样和配对结果不可混称同一批测试点。完整分组及上下界保存在CSV中；求解与数值残差检查见验证报告。',
             f"配对结果未达到最优性精度的点数：{paired['groups']['overall']['convergence_counts']['not_converged']}；新采样未达到最优性精度的点数：{fresh['groups']['overall']['convergence_counts']['not_converged']}。阈值判定未确定数另列于表中；达到时间上限的点仍保留距离上下界，未人为改为零。",
             '结论：移除最短运行时间显著减小本次抽样非凸间隙，但仍有正间隙，不能断言整个聚合域凸。最大距离下界仅为全域非凸度的抽样下界，检出比例不是实际运行不可行概率。',
             '设备模型沿用前一轮实现；HP温度动态未在本次消融中修改。']
    (output / 'min_uptime_comparison.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(json.dumps([r for r in rows if r['组别']=='总体'], ensure_ascii=False))

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    paper = commands.add_parser("paper-tables", help="Recalculate manuscript Tables I--V from saved inputs")
    paper.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    paper.add_argument("--output-dir", type=Path)
    scaling = commands.add_parser("scaling", help="Summarize the two 500-device continuous cases")
    scaling.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    scaling.add_argument("--output-dir", type=Path)
    nonconvexity = commands.add_parser("nonconvexity", help="Update reporting summaries in a saved nonconvexity experiment")
    nonconvexity.add_argument("--input-dir", type=Path, required=True)
    nonconvexity.add_argument("--threshold-percent", type=float, default=0.01)
    for name, description in (
        ("scale3-percentage", "Compute relative disaggregation percentages from saved trajectories"),
        ("scale3-nonconvexity", "Summarize scale3 gaps and verify saved ESS energy ledgers"),
        ("uptime-ablation", "Summarize paired and independent minimum-uptime ablations"),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("--base-dir", type=Path, default=RESULTS_ROOT / "microgrid" / "scale3",
                             help="Scale3 root containing training and saved evaluations")
        command.add_argument("--output-dir", type=Path,
                             help="Output directory; defaults to the original experiment directory")
    args = parser.parse_args(argv)
    if args.command == "paper-tables":
        results = args.results_root.absolute()
        output = args.output_dir or results / "figures"
        data = summarize_paper_tables(results)
        _paper_write_outputs(data, output)
        print(f"Tables I--V saved to {output}; Tables II--IV match manuscript values at reported precision.")
    elif args.command == "scaling":
        print(summarize_scaling(args.results_root, args.output_dir))
    elif args.command == "nonconvexity":
        print(json.dumps(summarize_nonconvexity(args.input_dir, args.threshold_percent)))
    elif args.command == "scale3-percentage":
        summarize_scale3_percentage(args)
    elif args.command == "scale3-nonconvexity":
        summarize_scale3_nonconvexity(args)
    elif args.command == "uptime-ablation":
        summarize_uptime_ablation(args)


if __name__ == "__main__":
    main()
