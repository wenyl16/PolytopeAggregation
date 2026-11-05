"""Train one PreTrainNet polytope over uncertain microgrid scenarios."""

import argparse
from copy import deepcopy
import json
import os
import hashlib
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Simulator.runners.train_aggregation import new_run_dir, train_case
from Simulator.cases.microgrid_case import MicroGrid
from Simulator.paths import DATA_ROOT, RESULTS_ROOT, comparison_output_path


DEVICE_COUNTS = {"MT": 1, "PV": 1, "TL": 1, "CL": 1, "ESS": 1, "HP": 2}


def build_microgrid(seed=42, device_scale=1):
    """Repeat the original portfolio, preserving each device's type and data.

    Calling gen_TL(5) instead would introduce chemical loads into the original
    EV-only portfolio. Independent deep copies keep this a size-only change.
    """
    if not isinstance(device_scale, int) or isinstance(device_scale, bool) or device_scale < 1:
        raise ValueError("device_scale must be a positive integer")
    mg = MicroGrid(seed=seed)
    for method, count in DEVICE_COUNTS.items():
        getattr(mg, f"gen_{method}")(count)
        devices = getattr(mg, f"{method}_list")
        setattr(mg, f"{method}_list", [deepcopy(device) for _ in range(device_scale)
                                      for device in devices])
    return mg


def training_output(requested, device_scale, overwrite=False):
    base = RESULTS_ROOT / "microgrid"
    if device_scale > 1:
        base = base / f"scale{device_scale}"
    target = comparison_output_path(requested or base / "training")
    if not target.exists() or not any(target.iterdir()):
        return new_run_dir(target, target)
    if all(path.name == "snapshots" and path.is_dir() and not any(path.iterdir())
           for path in target.iterdir()):
        return target
    config_path = target / "training_config.json"
    if not overwrite:
        raise FileExistsError(f"Use --overwrite to replace this training run: {target}")
    if not config_path.is_file():
        raise ValueError("Refusing to overwrite an unrecognized training directory")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("device_scale") != device_scale or config.get("portfolio_definition") != "repeat_original_devices":
        raise ValueError("Only a matching microgrid scale experiment can be overwritten")
    owned = {"metadata.pkl", "training_config.json", "error_history.pkl",
             "pretrainnet_weights.pth", "final_polytope.npz", "setup_timing.json", "solver.log",
             "solver_diagnostics.json", "solver_progress.json"}
    snapshots = target / "snapshots"
    for path in list(target.iterdir()):
        if path.is_file() and path.name in owned:
            path.unlink()
    if snapshots.exists():
        for pattern in ("A_*.csv", "b_*.csv"):
            for path in snapshots.glob(pattern):
                path.unlink()
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("approx", "exact"), default="approx")
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--case-seed", type=int, default=42)
    parser.add_argument("--device-scale", type=int, default=1,
                        help="Repeat every original device this many times (5 gives 35 devices)")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--noise-scale", type=float, default=0.05)
    parser.add_argument("--n-train", type=int, help="Single-stage override; executes n_train+1 epochs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--solver-log", action="store_true")
    parser.add_argument("--mip-focus", type=int, choices=(0, 1, 2, 3))
    parser.add_argument("--solver-symmetry", type=int, choices=(-1, 0, 1, 2))
    parser.add_argument("--count-branching", action=argparse.BooleanOptionalAction, default=None,
                        help="Branch on redundant integer device counts (automatic for enlarged portfolios)")
    parser.add_argument("--storage-headroom", action=argparse.BooleanOptionalAction, default=None,
                        help="Strengthen integer storage solves using implied SOC headroom bounds")
    args = parser.parse_args(argv)
    if args.solver_threads < 1 or args.device_scale < 1 or args.num_samples < 1 or args.batch_size < 1 or (args.n_train is not None and args.n_train < 0):
        parser.error("samples/batch-size must be positive and n-train nonnegative")
    output = training_output(args.output_dir, args.device_scale, args.overwrite)
    (output / "training_config.json").write_text(json.dumps({
        "device_scale": args.device_scale, "portfolio_definition": "repeat_original_devices",
        "status": "initializing"}, indent=2), encoding="utf-8")
    setup_started = time.perf_counter()
    mg = build_microgrid(args.case_seed, args.device_scale)
    solver_options = {"Threads": args.solver_threads, "Seed": 0}
    if args.mip_focus is not None:
        solver_options["MIPFocus"] = args.mip_focus
    if args.solver_symmetry is not None:
        solver_options["Symmetry"] = args.solver_symmetry
    if args.solver_log:
        solver_options["LogFile"] = os.path.relpath(output / "solver.log")
    case = mg.case_mg(conservativeness=0.0, device=args.device, gradient_mode=args.mode,
                      num_samples=args.num_samples, noise_scale=args.noise_scale, batch_size=args.batch_size,
                      solver_options=solver_options)
    count_branching = args.count_branching if args.count_branching is not None else args.device_scale > 1
    storage_headroom = args.storage_headroom if args.storage_headroom is not None else args.device_scale > 1
    if storage_headroom and not count_branching:
        parser.error("--storage-headroom requires --count-branching")
    physical_solver = None
    approximation_options = solver_options.copy()
    if count_branching:
        from Simulator.microgrid_count_solver import CountBranchingSolver
        solver_options = {**solver_options, "Presolve": 0}
        physical_solver = CountBranchingSolver(options=solver_options,
            storage_parameters=mg.ESS_list if storage_headroom else None, deltaT=mg.deltaT)
        case["errorcalculator"].solver = physical_solver
        last_progress = [0.0]

        def solver_progress(record, diagnostics):
            now = time.perf_counter()
            if now - last_progress[0] >= 10:
                (output / "solver_progress.json").write_text(json.dumps({
                    "status": "running", "diagnostics": diagnostics,
                    "last_solve": record}, indent=2, allow_nan=False), encoding="utf-8")
                last_progress[0] = now

        physical_solver.progress_callback = solver_progress
    source_paths = [Path(__file__), Path(__file__).with_name("train_aggregation.py"),
                    Path(__file__).parents[1] / "cases" / "microgrid_case.py",
                    Path(__file__).parents[1] / "Approximator.py"]
    if count_branching:
        source_paths.append(Path(__file__).parents[1] / "microgrid_count_solver.py")
    source_sha256 = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_paths}
    profiles_sha256 = hashlib.sha256((DATA_ROOT / "profiles_data" / "profiles_data.npz").read_bytes()).hexdigest()
    stages = [(29, 0.05, 1.0), (29, 0.1, 1e-2), (39, 0.2, 1e-6)]
    if args.n_train is not None:
        stages = [(args.n_train, 0.05, 1.0)]
    setup_seconds = time.perf_counter() - setup_started
    import pyomo.environ as pyo
    complexity = {"variables": sum(1 for _ in mg.model.component_data_objects(pyo.Var)),
                  "binary_variables": sum(v.is_binary() for v in mg.model.component_data_objects(pyo.Var)),
                  "constraints": sum(1 for _ in mg.model.component_data_objects(pyo.Constraint, active=True))}
    print(f"Device counts: { {name: len(getattr(mg, name + '_list')) for name in DEVICE_COUNTS} }; "
          f"model: {complexity}; setup: {setup_seconds:.3f} s", flush=True)
    (output / "setup_timing.json").write_text(json.dumps({"setup_seconds": setup_seconds,
        "includes_initial_outer_supports": True, "model_complexity": complexity}, indent=2), encoding="utf-8")
    try:
        train_case(case, mode=args.mode, stages=stages, output_dir=output,
                   seed=args.training_seed, device=args.device, uncertain=True,
                   case_config={"horizon": mg.T, "case_seed": args.case_seed,
                            "device_scale": args.device_scale,
                            "device_counts": {name: len(getattr(mg, name + '_list')) for name in DEVICE_COUNTS},
                            "portfolio_definition": "repeat_original_devices",
                            "model_complexity": complexity, "setup_seconds": setup_seconds,
                            "solver_options": solver_options,
                            "approximation_solver_options": approximation_options,
                            "count_branching": count_branching,
                            "storage_headroom": storage_headroom,
                            "physical_solver_backend": "gurobi_persistent_integer_counts" if count_branching else "pyomo_gurobi",
                            "source_sha256": source_sha256,
                            "profiles_sha256": profiles_sha256,
                            "num_samples": args.num_samples, "batch_size": args.batch_size,
                            "noise_scale": args.noise_scale})
    finally:
        if physical_solver is not None:
            (output / "solver_diagnostics.json").write_text(json.dumps({
                "diagnostics": physical_solver.diagnostics,
                "solves": physical_solver.solve_records}, indent=2, allow_nan=False), encoding="utf-8")
            (output / "solver_progress.json").write_text(json.dumps({
                "status": json.loads((output / "training_config.json").read_text(encoding="utf-8"))["status"],
                "diagnostics": physical_solver.diagnostics,
                "last_solve": physical_solver.last_solve}, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
