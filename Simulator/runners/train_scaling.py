"""Train the retained 500-device scaling experiment for T=24 or T=96.

T24 preserves the exact-reference scene, raw initialization, float64 and
300/200/500 updates. T96 preserves its archived raw initialization, float32
and 301/201/501 updates. Missing historical T96 inputs are never reconstructed
from normalized snapshots. Both branches keep their canonical result paths.
"""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from Simulator.paths import CODE_ROOT, DATA_ROOT, RESULTS_ROOT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


sha = digest


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n", encoding="utf-8")


def train_t24():
    import numpy as np
    import torch
    from Simulator.Approximator import PreTrainNet
    from Simulator.runners.train_aggregation import (
        _main_scenario, build_aggregator, prepare_main_output, train_case,
    )
    from Simulator.testers.case_methods import verify_scenario
    from Simulator.testers.polytope_evaluation import SOLVER_OPTIONS

    started = time.perf_counter()
    base = RESULTS_ROOT / "aggregation/T24/large_scale"
    reference = base / "exact/peb"
    config = json.loads((reference / "training_config.json").read_text(encoding="utf-8"))
    assert config["status"] == "completed" and config["optimizer_updates"] == 1000
    assert config["counts"] == [100, 200, 200] and config["dtype"] == "torch.float64"
    protected = {str(p.relative_to(reference)): digest(p)
                 for p in reference.rglob("*") if p.is_file()}
    stages = [(s["n_train"], s["lr"], s["rate_opt_feas"]) for s in config["stages"]]
    assert stages == [(299, .02, 1.), (199, .1, .01), (499, 1., 1e-6)]
    seed = config["training_seed"]
    agg = build_aggregator(24, config["case_seed"], counts=config["counts"])
    agg.case_aggregator(build_only=True)
    verify_scenario(reference, agg, config["case_seed"], required=True)
    scene = _main_scenario(agg, config["case_seed"])
    assert scene["scenario_sha256"] == config["scenario_sha256"]
    cumulative = np.tril(np.ones((24, 24)))
    cumulative /= cumulative.sum(axis=1, keepdims=True)
    raw_A = np.vstack((np.eye(24), -np.eye(24), cumulative[1:], -cumulative[1:]))
    old_A0 = np.loadtxt(reference / "snapshots/A_0.csv", delimiter=",")
    old_b0 = np.loadtxt(reference / "snapshots/b_0.csv", delimiter=",").reshape(-1)
    # ZeroInitLinear stores float32 biases, then the reference converts to double.
    # Undo only forward()'s row normalization; raw parameters set effective steps.
    raw_A32to64 = raw_A.astype(np.float32).astype(np.float64)
    raw_b = (old_b0 * (np.linalg.norm(raw_A32to64, axis=1) + 1e-8)).astype(np.float32)
    check = PreTrainNet(raw_A, raw_b, device="cpu").double()
    with torch.no_grad():
        initial_A, initial_b = check()
    np.testing.assert_array_equal(initial_A[0].numpy(), old_A0)
    np.testing.assert_array_equal(initial_b[0].numpy(), old_b0)
    case = agg.case_aggregator(A=raw_A, b=raw_b, gradient_mode="approx", template="peb")
    case["trainer_configure"] = {**case["trainer_configure"], "feas_tol": 1e-8, "opt_tol": 1e-8}
    calculator = case["errorcalculator"]
    for solver in (calculator.solver, calculator.cvx_solver):
        solver.options.update(SOLVER_OPTIONS)
    output = prepare_main_output(base / "approx/peb")
    # Evaluation checkpoints belong to the prior model; repeated training replaces them.
    for path in output.glob("evaluation_*"):
        if path.is_symlink() or not path.is_file() or path.resolve().parent != output.resolve():
            raise ValueError(f"Unsafe evaluation output: {path}")
        path.unlink()
    source_files = [Path(__file__), CODE_ROOT / "Simulator/runners/train_aggregation.py",
                    CODE_ROOT / "Simulator/Approximator.py", CODE_ROOT / "Simulator/cases/aggregation.py",
                    CODE_ROOT / "Simulator/cases/aggregation_case_T24.py"]
    source_sha = {str(p.relative_to(CODE_ROOT)): digest(p) for p in source_files}
    for path in source_files:
        destination = output / "source_snapshot" / path.relative_to(CODE_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    write_json(output / "scenario.json", scene)
    write_json(output / "setup_timing.json", {"setup_seconds": time.perf_counter() - started})
    recorded = []
    calculate = calculator.calculate

    def record_directions(*args, **kwargs):
        directions = np.asarray(kwargs["directions"], dtype=float)
        assert directions.shape == (1, 24) and np.isfinite(directions).all()
        recorded.append(directions[0].copy())
        return calculate(*args, **kwargs)

    calculator.calculate = record_directions
    context = {"horizon": 24, "template": "peb", "counts": config["counts"],
               "case_seed": config["case_seed"], "schedule": "three-stage", "stage_count": 3,
               "fourth_stage_used": False, "third_stage_extended": False,
               "scenario_sha256": scene["scenario_sha256"], "solver_options": dict(SOLVER_OPTIONS),
               "solver_threads": 1, "solver_seed": 0, "feas_tol": 1e-8, "opt_tol": 1e-8,
               "reference_run": str(reference), "only_changed_training_setting": "gradient_mode",
               "initial_forward_arrays_bit_identical": True, "source_sha256": source_sha,
               "output_policy": "fixed large_scale approximate run overwritten"}
    try:
        trainer = train_case(case, mode="approx", stages=stages, output_dir=output,
                             seed=seed, device="cpu", dtype=torch.float64, case_config=context)
    finally:
        np.save(output / "training_directions.npy", np.asarray(recorded).reshape(-1, 24))
        write_json(output / "projection_retries.json", {"records": calculator.projection_retry_records})
    np.testing.assert_array_equal(np.asarray(recorded), np.load(reference / "training_directions.npy"))
    assert trainer.optimizer_updates == 1000 and np.isfinite(np.asarray(trainer.grad_history)).all()
    with np.load(output / "final_polytope.npz", allow_pickle=False) as saved:
        region = {key: saved[key].copy() for key in ("A", "b")}
    np.savez(output / "final_polytope.npz", **region, T=24, case_seed=config["case_seed"],
             n_cl=100, n_tcl=200, n_pv=200, pmin=agg.pmin, pmax=agg.pmax,
             scenario_sha256=scene["scenario_sha256"], optimizer_updates=1000,
             training_seed=seed, method="peb", gradient_mode="approx")
    np.savetxt(output / "A_final.csv", region["A"], delimiter=",")
    np.savetxt(output / "b_final.csv", region["b"], delimiter=",")
    for name, expected in protected.items():
        assert digest(reference / name) == expected, f"Reference changed: {name}"
    final = json.loads((output / "training_config.json").read_text(encoding="utf-8"))
    final.update(actual_training_directions_verified=True, finite_gradient_norms_verified=True,
                 reference_training_files_unchanged=True,
                 final_polytope_sha256=digest(output / "final_polytope.npz"))
    write_json(output / "training_config.json", final)
    write_json(output / "training_validation.json", {
        "status": "passed", "gradient_mode": "approx", "dtype": "torch.float64",
        "horizon": 24, "total_resources": 500, "counts": config["counts"],
        "optimizer_updates": 1000, "phase_end_updates": final["phase_end_updates"],
        "scenario_and_initial_forward_match_reference": True,
        "training_directions_match_reference": True, "reference_training_files_unchanged": True,
        "reference_file_sha256": protected, "source_sha256": source_sha,
        "elapsed_training_seconds": final["elapsed_seconds"], "polytope_shape": list(region["A"].shape),
    })
    write_json(output / "run_summary.json", {"status": "completed", **context,
               "gradient_mode": "approx", "optimizer_updates": 1000,
               "phase_end_updates": final["phase_end_updates"],
               "elapsed_training_seconds": final["elapsed_seconds"]})
    print(json.dumps({"status": "completed", "output": str(output),
                      "elapsed_training_seconds": final["elapsed_seconds"]}), flush=True)


def train_t96(args):
    import numpy as np

    started = time.perf_counter()
    output = RESULTS_ROOT / "aggregation/T96/approx/peb"
    archive_path = args.initialization_archive
    old_audit_path = output / "evaluation_historical_recovery.json"
    if old_audit_path.exists():
        audit = json.loads(old_audit_path.read_text(encoding="utf-8"))
    else:
        audit = json.loads((output / "initialization_provenance.json").read_text(encoding="utf-8"))["original_recovery_audit"]
    assert audit["status"] == "passed"
    previous_evaluation = (json.loads((output / "evaluation_summary.json").read_text(encoding="utf-8"))
                           if (output / "evaluation_summary.json").exists() else None)
    prefix = "results/aggregation/T96/approx/peb/"
    if archive_path is not None:
        with zipfile.ZipFile(archive_path) as archive:
            raw_A_bytes = archive.read(prefix + "snapshots/A_0.csv")
            raw_b_bytes = archive.read(prefix + "snapshots/b_0.csv")
            for relative in ("profiles_data/profiles_data.npz", "aggregator_data/ZH_buildings.csv"):
                assert archive.read("data/" + relative) == (DATA_ROOT / relative).read_bytes()
        initialization_source = str(archive_path)
    else:
        initial_dir = args.raw_initialization
        if not all((initial_dir / name).is_file() for name in ("A_0.csv", "b_0.csv")):
            raise FileNotFoundError(
                "Historical raw T96 initialization is unavailable in this checkout. "
                "Saved final models and evaluations remain usable. Supply verified raw "
                "A_0.csv/b_0.csv with --raw-initialization, or the original code.zip "
                "with --initialization-archive. Current normalized snapshots are not raw inputs.")
        raw_A_bytes = (initial_dir / "A_0.csv").read_bytes()
        raw_b_bytes = (initial_dir / "b_0.csv").read_bytes()
        initialization_source = str(initial_dir)
    expected = json.loads((output / "initialization_provenance.json").read_text(encoding="utf-8"))
    for name, content in (("A", raw_A_bytes), ("b", raw_b_bytes)):
        if hashlib.sha256(content).hexdigest() != expected[f"raw_{name}_sha256"]:
            raise ValueError(f"Original raw T96 {name} initialization hash does not match")
    raw_A = np.loadtxt(io.BytesIO(raw_A_bytes), delimiter=",")
    raw_b = np.loadtxt(io.BytesIO(raw_b_bytes), delimiter=",").reshape(-1)
    cumulative = np.tril(np.ones((96, 96)))
    cumulative /= cumulative.sum(axis=1, keepdims=True)
    np.testing.assert_array_equal(raw_A, np.vstack((np.eye(96), -np.eye(96), cumulative, -cumulative)))
    assert raw_b.shape == (384,) and np.isfinite(raw_b).all()
    import torch
    from Simulator.Approximator import PreTrainNet
    from Simulator.runners.train_aggregation import (
        _main_scenario, build_aggregator, prepare_main_output, train_case,
    )
    from Simulator.testers.polytope_evaluation import SOLVER_OPTIONS

    agg = build_aggregator(96, seed=0, counts=(100, 200, 200))
    agg.case_aggregator(build_only=True)
    scene = _main_scenario(agg, 0)
    assert scene["scenario_sha256"] == audit["reconstructed_scenario_sha256"]
    # The old case saved raw support rows, before network row normalization.
    # Read them directly: the inverse normalization needed for T24 is inapplicable.
    check = PreTrainNet(raw_A, raw_b, device="cpu")
    with torch.no_grad():
        initial_A, initial_b = check()
    assert initial_A.dtype == initial_b.dtype == torch.float32
    case = agg.case_aggregator(A=raw_A, b=raw_b, gradient_mode="approx", template="peb")
    case["trainer_configure"] = {**case["trainer_configure"], "feas_tol": 1e-8, "opt_tol": 1e-8}
    calculator = case["errorcalculator"]
    for solver in (calculator.solver, calculator.cvx_solver):
        solver.options.update(SOLVER_OPTIONS)
    stages = [(300, .01, 1.), (200, .05, .01), (500, args.third_stage_lr, 1e-6)]
    expected_directions = np.random.RandomState(args.training_seed).randn(1003, 96)
    prepare_main_output(output)
    for path in output.glob("evaluation_*"):
        if path.is_symlink() or not path.is_file() or path.resolve().parent != output.resolve():
            raise ValueError(f"Unsafe evaluation output: {path}")
        path.unlink()
    sources = [Path(__file__), CODE_ROOT / "Simulator/runners/train_aggregation.py",
               CODE_ROOT / "Simulator/Approximator.py", CODE_ROOT / "Simulator/cases/aggregation.py",
               CODE_ROOT / "Simulator/cases/aggregation_case_T96.py"]
    source_sha = {str(p.relative_to(CODE_ROOT)): sha(p) for p in sources}
    for path in sources:
        destination = output / "source_snapshot" / path.relative_to(CODE_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    provenance = {"status": "passed", "initialization_source": initialization_source,
                  "raw_A_member": prefix + "snapshots/A_0.csv", "raw_b_member": prefix + "snapshots/b_0.csv",
                  "raw_A_sha256": hashlib.sha256(raw_A_bytes).hexdigest(),
                  "raw_b_sha256": hashlib.sha256(raw_b_bytes).hexdigest(),
                  "original_recovery_audit": audit,
                  "previous_evaluation": previous_evaluation,
                  "historical_training_directions_available": False,
                  "note": "Initialization and scene are recovered; new training uses an explicit reproducible seed."}
    write_json(output / "initialization_provenance.json", provenance)
    write_json(output / "scenario.json", scene)
    write_json(output / "setup_timing.json", {"setup_seconds": time.perf_counter() - started,
               "initial_support_bounds_reused": True})
    recorded = []
    calculate = calculator.calculate

    def record_directions(*positional, **kwargs):
        directions = np.asarray(kwargs["directions"], dtype=float)
        assert directions.shape == (1, 96) and np.isfinite(directions).all()
        recorded.append(directions[0].copy())
        return calculate(*positional, **kwargs)

    calculator.calculate = record_directions
    context = {"horizon": 96, "template": "peb", "counts": [100, 200, 200], "case_seed": 0,
               "schedule": "three-stage", "stage_count": 3,
               "third_stage_initial_lr": args.third_stage_lr,
               "optimizer": "SGD", "scheduler": {"type": "StepLR", "step_size": 100, "gamma": .98},
               "fourth_stage_used": False, "device": "cpu", "batch_size": 1,
               "scenario_sha256": scene["scenario_sha256"], "source_sha256": source_sha,
               "solver_options": dict(SOLVER_OPTIONS), "solver_threads": 1, "solver_seed": 0,
               "feas_tol": 1e-8, "opt_tol": 1e-8,
               "initialization": "archived raw 384-row PEB template and support bounds",
               "historical_training_directions_available": False,
               "output_policy": "fixed main T96 approximate run overwritten"}
    write_json(output / "training_validation.json", {"status": "running"})
    try:
        trainer = train_case(case, mode="approx", stages=stages, output_dir=output,
                             seed=args.training_seed, device="cpu", case_config=context)
    finally:
        np.save(output / "training_directions.npy", np.asarray(recorded).reshape(-1, 96))
        write_json(output / "projection_retries.json", {"records": calculator.projection_retry_records})
    np.testing.assert_array_equal(np.asarray(recorded), expected_directions)
    assert trainer.optimizer_updates == 1003 and np.isfinite(np.asarray(trainer.grad_history)).all()
    np.testing.assert_array_equal(np.loadtxt(output / "snapshots/A_0.csv", delimiter=","), initial_A[0].numpy())
    np.testing.assert_array_equal(np.loadtxt(output / "snapshots/b_0.csv", delimiter=",").reshape(-1), initial_b[0].numpy())
    with np.load(output / "final_polytope.npz", allow_pickle=False) as saved:
        region = {key: saved[key].copy() for key in ("A", "b")}
    np.savez(output / "final_polytope.npz", **region, T=96, case_seed=0, n_cl=100, n_tcl=200, n_pv=200,
             pmin=agg.pmin, pmax=agg.pmax, scenario_sha256=scene["scenario_sha256"],
             optimizer_updates=1003, training_seed=args.training_seed, method="peb", gradient_mode="approx")
    np.savetxt(output / "A_final.csv", region["A"], delimiter=",")
    np.savetxt(output / "b_final.csv", region["b"], delimiter=",")
    config = json.loads((output / "training_config.json").read_text(encoding="utf-8"))
    config.update(actual_training_directions_verified=True, finite_gradient_norms_verified=True,
                  raw_initialization_and_scenario_verified=True,
                  final_polytope_sha256=sha(output / "final_polytope.npz"),
                  final_learning_rates=[float(g["lr"]) for g in trainer.optimizer.param_groups])
    write_json(output / "training_config.json", config)
    write_json(output / "training_validation.json", {"status": "passed", "optimizer_updates": 1003,
               "phase_end_updates": config["phase_end_updates"], "scenario_sha256": scene["scenario_sha256"],
               "initial_forward_matches_archived_raw_initialization": True,
               "training_directions_match_seed": True, "finite_gradients": True,
               "polytope_shape": list(region["A"].shape), "dtype": config["dtype"],
               "elapsed_training_seconds": config["elapsed_seconds"]})
    write_json(output / "run_summary.json", {"status": "completed", **context,
               "optimizer_updates": 1003, "phase_end_updates": config["phase_end_updates"],
               "elapsed_training_seconds": config["elapsed_seconds"]})
    print(json.dumps({"status": "completed", "output": str(output),
                      "elapsed_training_seconds": config["elapsed_seconds"]}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, choices=(24, 96), default=24)
    parser.add_argument("--third-stage-lr", type=float,
                        help="T96 only: final-stage initial learning rate (default: 0.5)")
    parser.add_argument("--training-seed", type=int,
                        help="T96 only: training seed (default: 0)")
    parser.add_argument("--raw-initialization", type=Path,
                        help="T96 only: directory containing original raw A_0.csv/b_0.csv")
    parser.add_argument("--initialization-archive", type=Path,
                        help="T96 only: original code.zip containing verified raw initialization")
    args = parser.parse_args(argv)
    if args.horizon == 24:
        if any(getattr(args, name) is not None for name in
               ("third_stage_lr", "training_seed", "raw_initialization", "initialization_archive")):
            parser.error("T24 uses the saved reference protocol; additional initialization/seed/lr options require --horizon 96")
        return train_t24()
    args.third_stage_lr = .5 if args.third_stage_lr is None else args.third_stage_lr
    args.training_seed = 0 if args.training_seed is None else args.training_seed
    args.raw_initialization = args.raw_initialization or RESULTS_ROOT / "aggregation/T96/approx/peb/raw_initialization"
    if not math.isfinite(args.third_stage_lr) or args.third_stage_lr <= 0:
        parser.error("third-stage-lr must be finite and positive")
    return train_t96(args)


if __name__ == "__main__":
    main()
