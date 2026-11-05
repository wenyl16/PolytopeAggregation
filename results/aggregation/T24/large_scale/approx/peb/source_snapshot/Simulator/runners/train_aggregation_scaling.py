"""Retrain the retained T24 500-device case with approximate gradients only.

The exact reference provides its scene, stage lengths and raw initialization.
Retain float64 and the 94-row parameterization to isolate the gradient change.
Overwrite the same large_scale/approx/peb destination on repeated execution.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from Simulator.Approximator import PreTrainNet
from Simulator.paths import CODE_ROOT, RESULTS_ROOT
from Simulator.runners.train_aggregation import (
    _main_scenario, build_aggregator, prepare_main_output, train_case,
)
from Simulator.testers.compare_aggregation_methods import verify_scenario
from Simulator.testers.compute_polytope_errors_and_costs_gurobi import SOLVER_OPTIONS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n", encoding="utf-8")


def main():
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


if __name__ == "__main__":
    main()
