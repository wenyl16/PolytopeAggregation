"""Train the T24/T96 aggregation polytope with exact or approximate gradients.

Use --prepare-comparison for the explicit T24 verified PEB/PB preparation action.
"""

import argparse
import ast
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Preserve the existing Windows PyTorch/Gurobi runtime compatibility setting.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from Simulator.Approximator import ErrorCalculator, PreTrainNet, Trainer
from Simulator.paths import CODE_ROOT, DATA_ROOT, RESULTS_ROOT, aggregation_result_dir
from Simulator.baselines import build_cube


def build_aggregator(horizon=24, seed=0, counts=None):
    """Build a scene with independent time and CL/TCL/PV resource dimensions."""
    if horizon == 24:
        from Simulator.cases.aggregation_case_T24 import Aggregator
        default_counts = (5, 40, 5)
    elif horizon == 96:
        from Simulator.cases.aggregation_case_T96 import Aggregator
        default_counts = (100, 200, 200)
    else:
        raise ValueError("horizon must be 24 or 96")
    counts = default_counts if counts is None else tuple(counts)
    if (len(counts) != 3 or any(isinstance(number, (bool, np.bool_))
            or not isinstance(number, (int, np.integer)) or number <= 0 for number in counts)):
        raise ValueError("counts must contain three positive integers in CL/TCL/PV order")
    with (DATA_ROOT / "aggregator_data" / "ZH_buildings.csv").open(encoding="utf-8-sig", newline="") as stream:
        available_buildings = sum(1 for row in csv.DictReader(stream) if row)
    if counts[1] > available_buildings:
        raise ValueError(f"Requested {counts[1]} TCLs but only {available_buildings} building records are available")
    agg = Aggregator(seed=seed, discrete_rate=0.0)
    agg.gen_CL(counts[0])
    agg.gen_TCL(counts[1])
    agg.gen_PV(counts[2])
    return agg


def build_stages(horizon=24, schedule="three-stage", n_train=None, stage_multiplier=1):
    """Return Trainer calls, scaling actual n_train+1 updates in each stage."""
    if horizon not in (24, 96):
        raise ValueError("horizon must be 24 or 96")
    if schedule not in ("three-stage", "two-stage", "single-stage"):
        raise ValueError("schedule must be three-stage, two-stage or single-stage")
    if n_train is not None and (not isinstance(n_train, (int, np.integer)) or n_train < 0):
        raise ValueError("n_train must be a nonnegative integer")
    if not isinstance(stage_multiplier, (int, np.integer)) or stage_multiplier < 1:
        raise ValueError("stage_multiplier must be a positive integer")
    learning_rates = (0.02, 0.1, 1.0) if horizon == 24 else (0.01, 0.05, 0.1)
    stages = list(zip((300, 200, 500), learning_rates, (1.0, 1e-2, 1e-6)))
    if schedule == "single-stage":
        updates = sum(n + 1 for n, _, _ in stages)
        stages = [(updates - 1 if n_train is None else n_train, *stages[-1][1:])]
    elif schedule == "two-stage":
        if n_train is not None:
            raise ValueError("n_train override is not supported for the two-stage schedule")
        merged_updates = sum(n + 1 for n, _, _ in stages[1:])
        stages = [stages[0], (merged_updates - 1, *stages[-1][1:])]
    elif n_train is not None:
        stages = [(n_train, *stages[0][1:])]
    return [((n + 1) * stage_multiplier - 1, lr, weight) for n, lr, weight in stages]


def new_run_dir(base, requested=None):
    """Use a fresh run directory so retained experiments are never overwritten."""
    target = Path(requested) if requested else Path(base) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def prepare_main_output(base):
    """Overwrite the main run's known files, retaining canonical child runs.

    ``new_run_dir`` keeps its fresh-directory contract for existing consumers.
    Only this CLI's default destination uses replacement semantics.
    """
    base = Path(base)
    target = base.resolve()
    root = RESULTS_ROOT.resolve()
    if base.is_symlink() or not target.is_relative_to(root) or len(target.relative_to(root).parts) < 4:
        raise ValueError("Main training output must be a model directory inside results")
    target.mkdir(parents=True, exist_ok=True)
    owned = {target / name for name in (
        "training_config.json", "error_history.pkl", "metadata.pkl", "pretrainnet_weights.pth",
        "final_polytope.npz", "initial_polytope.npz", "training_directions.npy", "scenario.json",
        "A_final.csv", "b_final.csv", "run_summary.json", "training_pretrain.log", "migration_info.json",
        "projection_retries.json", "evaluation_errors.npz", "evaluation_directions.npy",
        "evaluation_progress.json", "evaluation_progress.tmp", "verification.json",
        "independent_recheck.json", "independent_coordinates.npz", "peb_checkpoint.npz", "pb_checkpoint.npz",
    )}
    snapshots = target / "snapshots"
    if snapshots.exists():
        if snapshots.is_symlink() or snapshots.resolve().parent != target:
            raise ValueError("Unsafe main training snapshots directory")
        owned.update(path for path in snapshots.glob("*.csv")
                     if path.stem.startswith(("A_", "b_")) and path.stem[2:].isdigit())
    figures = target / "figures"
    if figures.exists() and (figures.is_symlink() or figures.resolve().parent != target):
        raise ValueError("Unsafe main training figures directory")
    owned.update(figures / name for name in (
        "aggregation_gradient_comparison.pdf", "aggregation_gradient_comparison.png",
        "aggregation_training_trajectory.pdf",
    ))
    existing = [path for path in owned if path.exists()]
    for path in existing:
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(target):
            raise ValueError(f"Unsafe main training file: {path}")
    for path in existing:
        path.unlink()
    return target


def _main_scenario(agg, case_seed):
    """Full scene and input-file signature for the main CLI's actual case."""
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    record = {
        "aggregator_class": f"{type(agg).__module__}.{type(agg).__name__}",
        "case_seed": case_seed, "T": agg.T, "n_cl": len(agg.CL_list), "n_tcl": len(agg.TCL_list),
        "n_pv": len(agg.PV_list), "n_ev": len(agg.EV_list), "n_ess": len(agg.ESS_list),
        "discrete_rate": agg.discrete_rate, "deltaT": agg.deltaT,
        "pmin": np.asarray(agg.pmin), "pmax": np.asarray(agg.pmax),
        "CL_list": agg.CL_list, "TCL_list": agg.TCL_list, "PV_list": agg.PV_list,
        "EV_list": agg.EV_list, "ESS_list": agg.ESS_list, "theta_amb": agg.theta_amb,
        "pv_profiles": agg.pv_profiles, "load_profiles": agg.load_profiles,
        "input_sha256": {str(path.relative_to(DATA_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (DATA_ROOT / "profiles_data" / "profiles_data.npz",
                                       DATA_ROOT / "aggregator_data" / "ZH_buildings.csv")},
    }
    signature = hashlib.sha256(json.dumps(record, default=convert, sort_keys=True,
                               separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    return {**json.loads(json.dumps(record, default=convert)), "scenario_sha256": signature}


def _main_metric_stats(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "max": float(values.max()), "p95": float(np.quantile(values, .95)),
            "count_gt1e4": int(np.count_nonzero(values > 1e-4)), "count_threshold": 1e-4}


def train_case(case, *, mode, stages, output_dir, seed=0, device="cpu", uncertain=False,
               case_config=None):
    """Shared PreTrainNet training and output contract for aggregation/microgrid."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = PreTrainNet(case["A_hat"], case["b_hat"], device=device).to(device)
    if mode == "exact":
        model = model.double()
        for solver in (case["errorcalculator"].solver, case["errorcalculator"].cvx_solver):
            solver.options.update({"FeasibilityTol": 1e-9, "OptimalityTol": 1e-9, "BarConvTol": 1e-12})
    trainer = Trainer(model=model, error_calculator=case["errorcalculator"],
                      is_uncertain=uncertain, parallel=False, device=device,
                      params_data=case["params"])
    snapshots = output_dir / "snapshots"
    snapshots.mkdir(exist_ok=True)

    def snapshot(index):
        with torch.no_grad():
            A, b = model()
        np.savetxt(snapshots / f"A_{index}.csv", A[0].cpu().numpy(), delimiter=",")
        np.savetxt(snapshots / f"b_{index}.csv", b[0].cpu().numpy(), delimiter=",")

    def callback(_calculator, _iteration):
        updates = len(trainer.loss_history)
        if updates % 50 == 0:
            snapshot(updates)
            print(f"Update {updates}: feasibility={trainer.loss_history[-1][0]:.6g}, "
                  f"optimality={trainer.loss_history[-1][1]:.6g}", flush=True)

    snapshot(0)
    with (output_dir / "metadata.pkl").open("wb") as stream:
        pickle.dump(case.get("metadata", {}), stream)
    config = {**(case_config or {}), "gradient_mode": mode, "network": "PreTrainNet", "training_seed": seed,
              "dtype": str(next(model.parameters()).dtype), "is_uncertain": uncertain,
              "stages": [{"n_train": n, "epochs": n + 1, "lr": lr, "rate_opt_feas": weight}
                         for n, lr, weight in stages], "status": "running"}
    config_path = output_dir / "training_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    started = time.perf_counter()
    try:
        for n_train, lr, weight in stages:
            phase_started = time.perf_counter()
            trainer.configure(**case["trainer_configure"])
            trainer.configure(lr=lr, rate_opt_feas=weight, training_callback=callback, call_interval=1)
            trainer.initialize()
            trainer.train(n_train=n_train)
            trainer.train_calls[-1]["elapsed_seconds"] = time.perf_counter() - phase_started
        config["status"] = "completed"
    except Exception as exc:
        config.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        config.update(elapsed_seconds=time.perf_counter() - started,
                      train_calls=trainer.train_calls,
                      optimizer_updates=len(trainer.loss_history),
                      phase_end_updates=np.cumsum([call["optimizer_updates"]
                                                   for call in trainer.train_calls]).tolist())
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        history = np.asarray(trainer.loss_history, dtype=float).reshape(-1, 2)
        with (output_dir / "error_history.pkl").open("wb") as stream:
            pickle.dump({"feasibility_errors": history[:, 0], "optimality_errors": history[:, 1],
                         "objective_history": trainer.objective_history,
                         "surrogate_history": trainer.surrogate_history}, stream)
    snapshot(len(trainer.loss_history))
    torch.save(model.state_dict(), output_dir / "pretrainnet_weights.pth")
    with torch.no_grad():
        A, b = model()
    np.savez(output_dir / "final_polytope.npz", A=A[0].cpu().numpy(), b=b[0].cpu().numpy())
    print(f"Saved training results: {output_dir}")
    return trainer


def _train_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("exact", "approx"), required=True)
    parser.add_argument("--horizon", type=int, choices=(24, 96), default=24)
    parser.add_argument("--counts", type=int, nargs=3, metavar=("CL", "TCL", "PV"),
                        help="Explicit resource counts, independent of horizon; requires --output-dir")
    parser.add_argument("--template", choices=("peb", "pb"), default="peb")
    parser.add_argument("--box-dir", type=Path,
                        help="PB initialization directory containing A_cube.csv and b_cube.csv")
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--case-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--schedule", choices=("three-stage", "two-stage", "single-stage"), default="three-stage",
                        help="Two-stage merges phases 2/3 using final settings; single-stage uses final settings throughout")
    parser.add_argument("--n-train", type=int,
                        help="Single-call override using first/final settings for three/single-stage; unavailable for two-stage")
    parser.add_argument("--stage-multiplier", type=int, default=1,
                        help="Multiply actual optimizer updates in every selected stage (default: 1)")
    parser.add_argument("--third-stage-lr", type=float,
                        help="Third-stage initial learning rate; T24 schedules default to a final-stage learning rate of 1")
    parser.add_argument("--evaluation-samples", type=int,
                        help="Held-out directions after training (default 5000 for T24 exact, otherwise 0)")
    parser.add_argument("--evaluation-seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.counts is not None:
        if any(number <= 0 for number in args.counts):
            parser.error("--counts requires positive CL/TCL/PV counts")
        if args.output_dir is None:
            parser.error("--counts requires an explicit fresh --output-dir to protect the main results")
        if args.template == "pb" and args.box_dir is None:
            parser.error("Custom-count PB training requires its matching --box-dir")
    if args.n_train is not None and args.n_train < 0:
        parser.error("--n-train must be nonnegative")
    if args.stage_multiplier < 1:
        parser.error("--stage-multiplier must be positive")
    if args.schedule == "two-stage" and args.n_train is not None:
        parser.error("--n-train cannot be combined with --schedule two-stage")
    if args.box_dir is not None and args.template != "pb":
        parser.error("--box-dir requires --template pb")
    if args.third_stage_lr is not None and (not np.isfinite(args.third_stage_lr) or args.third_stage_lr <= 0):
        parser.error("--third-stage-lr must be finite and positive")
    samples = (5000 if args.horizon == 24 and args.mode == "exact" else 0) if args.evaluation_samples is None else args.evaluation_samples
    if samples < 0:
        parser.error("--evaluation-samples must be nonnegative")
    if args.third_stage_lr is not None and (args.schedule != "three-stage" or args.n_train is not None):
        parser.error("--third-stage-lr requires the full --schedule three-stage")
    stages = build_stages(args.horizon, args.schedule, args.n_train, args.stage_multiplier)
    # Preserve build_stages()'s historical 1003-update contract for comparison
    # runners. Only this main CLI defaults to exactly 300/200/500 for T24.
    if args.horizon == 24 and args.mode == "exact" and args.schedule == "three-stage" and args.n_train is None:
        third_lr = 1. if args.third_stage_lr is None else args.third_stage_lr
        stages = [(300 * args.stage_multiplier - 1, .02, 1.),
                  (200 * args.stage_multiplier - 1, .1, .01),
                  (500 * args.stage_multiplier - 1, third_lr, 1e-6)]
    elif args.third_stage_lr is not None:
        stages[2] = (stages[2][0], args.third_stage_lr, stages[2][2])
    expected_updates = sum(n + 1 for n, _lr, _weight in stages)
    evaluation_directions = np.random.RandomState(args.evaluation_seed).randn(samples, args.horizon)
    expected_training_directions = np.random.RandomState(args.training_seed).randn(expected_updates, args.horizon)
    evaluation_rows = {row.tobytes() for row in evaluation_directions}
    if any(row.tobytes() in evaluation_rows for row in expected_training_directions):
        parser.error("Training and held-out evaluation directions overlap; use different seeds")
    from Simulator.testers.compute_polytope_errors_and_costs_gurobi import SOLVER_OPTIONS, evaluate_native
    from Simulator.testers.compare_aggregation_methods import verify_scenario
    agg = build_aggregator(args.horizon, args.case_seed, counts=args.counts)
    agg.case_aggregator(build_only=True)
    scene = _main_scenario(agg, args.case_seed)
    A = b = None
    box = None
    if args.template == "pb":
        default_box = RESULTS_ROOT / "aggregation" / f"T{args.horizon}" / "baselines" / "box"
        if args.horizon == 24 and args.mode == "exact":
            default_box /= "verified_case0"
        box = args.box_dir or default_box
        if (args.horizon == 24 and args.mode == "exact") or args.counts is not None:
            verify_scenario(box, agg, args.case_seed, required=True)
        A = np.loadtxt(box / "A_cube.csv", delimiter=",")
        b = np.loadtxt(box / "b_cube.csv", delimiter=",").reshape(-1)
        if A.ndim != 2 or A.shape != (len(b), args.horizon):
            parser.error("box dimensions do not match the requested horizon")
    elif args.mode == "exact":
        cumulative = np.tril(np.ones((agg.T, agg.T)))
        cumulative /= cumulative.sum(axis=1, keepdims=True)
        A = np.vstack((np.eye(agg.T), -np.eye(agg.T), cumulative[1:], -cumulative[1:]))
    case = agg.case_aggregator(A=A, b=b, gradient_mode=args.mode, template=args.template)
    case["trainer_configure"] = {**case["trainer_configure"], "feas_tol": 1e-8, "opt_tol": 1e-8}
    calculator = case["errorcalculator"]
    for solver in (calculator.solver, calculator.cvx_solver):
        solver.options.update(SOLVER_OPTIONS)
    base = aggregation_result_dir(args.horizon, args.mode, args.template)
    output = (prepare_main_output(base) if args.output_dir is None else new_run_dir(base, args.output_dir))
    def write_json(path, value):
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    sources = [Path(__file__).resolve(), CODE_ROOT / "Simulator" / "Approximator.py",
               CODE_ROOT / "Simulator" / "testers/compute_polytope_errors_and_costs_gurobi.py", CODE_ROOT / "Simulator" / "cases" / "aggregation.py",
               CODE_ROOT / "Simulator" / "cases" / f"aggregation_case_T{args.horizon}.py"]
    source_sha256 = {str(path.relative_to(CODE_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    write_json(output / "scenario.json", scene)
    recorded = []
    calculate = calculator.calculate
    def calculate_with_directions(*positional, **kwargs):
        directions = np.asarray(kwargs.get("directions"), dtype=float)
        if directions.shape != (1, args.horizon) or not np.isfinite(directions).all():
            raise RuntimeError("Main training requires one explicit finite direction per update")
        recorded.append(directions[0].copy())
        return calculate(*positional, **kwargs)
    calculator.calculate = calculate_with_directions
    trainer = None
    summary = {"status": "running", "horizon": args.horizon, "template": args.template,
               "counts": [len(agg.CL_list), len(agg.TCL_list), len(agg.PV_list)],
               "resource_count_override": args.counts,
               "training_seed": args.training_seed, "evaluation_seed": args.evaluation_seed,
               "evaluation_samples": samples, "scenario_sha256": scene["scenario_sha256"],
               "source_sha256": source_sha256, "stages": [
                   {"n_train": n, "epochs": n + 1, "lr": lr, "rate_opt_feas": rate} for n, lr, rate in stages]}
    write_json(output / "run_summary.json", summary)
    try:
        try:
            trainer = train_case(case, mode=args.mode, stages=stages, output_dir=output,
                       seed=args.training_seed, device=args.device,
                       case_config={"horizon": args.horizon, "template": args.template,
                                    "case_seed": args.case_seed, "counts": summary["counts"],
                                    "resource_count_override": args.counts,
                                    "schedule": args.schedule, "n_train_override": args.n_train,
                                    "stage_multiplier": args.stage_multiplier,
                                    "box_dir": str(box.resolve()) if box is not None else None,
                                    "stage_count": len(stages), "fourth_stage_used": False,
                                    "third_stage_extended": False, "error_cutoff": 1e-8,
                                    "feas_tol": 1e-8, "opt_tol": 1e-8,
                                    "solver_options": dict(SOLVER_OPTIONS), "solver_threads": 1, "solver_seed": 0,
                                    "scenario_sha256": scene["scenario_sha256"], "source_sha256": source_sha256,
                                    "evaluation_samples": samples, "evaluation_seed": args.evaluation_seed,
                                    "output_policy": "fixed main result overwritten" if args.output_dir is None else "explicit fresh run"})
        finally:
            np.save(output / "training_directions.npy", np.asarray(recorded, dtype=float).reshape(-1, args.horizon))
            retries = calculator.projection_retry_records
            write_json(output / "projection_retries.json", {"status": "recorded", "records": retries,
                       "retry_count": sum(len(row["attempts"]) for row in retries)})
        np.testing.assert_array_equal(np.asarray(recorded), expected_training_directions)
        if trainer.optimizer_updates != expected_updates:
            raise RuntimeError("Actual optimizer updates differ from the main training schedule")
        gradients = np.asarray(trainer.grad_history, dtype=float)
        if gradients.shape != (expected_updates,) or not np.isfinite(gradients).all():
            raise RuntimeError("Actual main training gradient history is incomplete or nonfinite")
        with (output / "error_history.pkl").open("rb") as stream:
            history = pickle.load(stream)
        history["gradient_norms"] = gradients
        with (output / "error_history.pkl").open("wb") as stream:
            pickle.dump(history, stream)
        with np.load(output / "final_polytope.npz", allow_pickle=False) as saved:
            region = {key: saved[key].copy() for key in ("A", "b")}
        tags = {"method": args.template, "case_seed": args.case_seed, "T": args.horizon,
                "n_cl": len(agg.CL_list), "n_tcl": len(agg.TCL_list), "n_pv": len(agg.PV_list),
                "discrete_rate": agg.discrete_rate, "pmin": np.asarray(agg.pmin), "pmax": np.asarray(agg.pmax),
                "scenario_sha256": scene["scenario_sha256"], "training_seed": args.training_seed,
                "optimizer_updates": expected_updates}
        np.savez(output / "final_polytope.npz", **region, **tags)
        with np.load(output / "final_polytope.npz", allow_pickle=False) as saved:
            for key in region:
                np.testing.assert_array_equal(saved[key], region[key])
        np.savetxt(output / "A_final.csv", region["A"], delimiter=",")
        np.savetxt(output / "b_final.csv", region["b"], delimiter=",")
        config_path = output / "training_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.update(actual_training_directions_verified=True, finite_gradient_norms_verified=True,
                      scheduler_last_epoch=int(trainer.scheduler.last_epoch) if trainer.scheduler else None,
                      scheduler_step_count=int(trainer.scheduler.state_dict()["_step_count"]) if trainer.scheduler else None,
                      final_learning_rates=[float(group["lr"]) for group in trainer.optimizer.param_groups],
                      final_polytope_sha256=hashlib.sha256((output / "final_polytope.npz").read_bytes()).hexdigest(),
                      training_directions_sha256=hashlib.sha256((output / "training_directions.npy").read_bytes()).hexdigest())
        write_json(config_path, config)
        summary.update(optimizer_updates=expected_updates, phase_end_updates=config["phase_end_updates"],
                       scheduler_last_epoch=config["scheduler_last_epoch"], scheduler_step_count=config["scheduler_step_count"],
                       final_learning_rates=config["final_learning_rates"], elapsed_training_seconds=config["elapsed_seconds"],
                       polytope_shape=list(region["A"].shape),
                       training={"feasibility": _main_metric_stats(history["feasibility_errors"]),
                                 "optimality": _main_metric_stats(history["optimality_errors"])})
        if samples:
            np.save(output / "evaluation_directions.npy", evaluation_directions)
            def progress(record):
                write_json(output / "evaluation_progress.json", record)
                if record.get("completed", 0) % 1000 == 0 or record.get("completed") == samples:
                    print(f"{args.template} evaluation: {record.get('completed', 0)}/{samples}", flush=True)
            result = evaluate_native(agg, evaluation_directions, region, progress=progress,
                                     checkpoint_path=output / f"{args.template}_checkpoint.npz")
            np.savez(output / "evaluation_errors.npz", **{key: value for key, value in result.items() if value is not None})
            summary["evaluation"] = {"feasibility": _main_metric_stats(result["error_feas"]),
                                     "optimality": _main_metric_stats(result["error_opt"]),
                                     **{key: value for key, value in result.items() if key.startswith("native_")}}
            print(f"{args.template}: max feasibility={np.max(result['error_feas']):.12g}; "
                  f"count > 1e-4={np.count_nonzero(result['error_feas'] > 1e-4)}", flush=True)
        summary["status"] = "completed"
    except BaseException as exc:
        summary.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "run_summary.json", summary)
    print(f"Saved main training and evaluation results: {output}", flush=True)
    return output


# Fixed T24 comparison protocol; ordinary training keeps its existing defaults.
OPTIONS = {"FeasibilityTol": 1e-9, "OptimalityTol": 1e-9,
           "BarConvTol": 1e-12, "Threads": 1, "Seed": 0}
STAGES = [(299, .02, 1.), (199, .1, .01), (499, 1., 1e-6)]
BASE = RESULTS_ROOT / "aggregation" / "T24"

def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)

def write_json(path, value):
    path.write_text(json.dumps(value, default=json_value, indent=2,
                               ensure_ascii=False, allow_nan=False), encoding="utf-8")

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def configure(calculator):
    calculator.configure(feas_tol=0., opt_tol=0., active_tol=1e-5)
    for solver in (calculator.solver, calculator.cvx_solver):
        solver.options.update(OPTIONS)

def scenario(agg):
    """Include generated devices, input profiles, and physical coordinates."""
    record = {"aggregator_class": "Simulator.cases.aggregation_case_T24.Aggregator",
              "case_seed": 0, "T": 24, "n_cl": 5, "n_tcl": 40, "n_pv": 5,
              "n_ev": len(agg.EV_list), "n_ess": len(agg.ESS_list),
              "discrete_rate": 0., "deltaT": agg.deltaT,
              "pmin": np.asarray(agg.pmin), "pmax": np.asarray(agg.pmax),
              "CL_list": agg.CL_list, "TCL_list": agg.TCL_list,
              "PV_list": agg.PV_list, "EV_list": agg.EV_list,
              "ESS_list": agg.ESS_list, "theta_amb": agg.theta_amb,
              "pv_profiles": agg.pv_profiles, "load_profiles": agg.load_profiles,
              "input_sha256": {str(p.relative_to(DATA_ROOT)): digest(p) for p in (
                  DATA_ROOT / "profiles_data" / "profiles_data.npz",
                  DATA_ROOT / "aggregator_data" / "ZH_buildings.csv")}}
    payload = json.dumps(record, default=json_value, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    record["scenario_sha256"] = hashlib.sha256(payload).hexdigest()
    return record

def source_hashes():
    return {str(p.resolve().relative_to(CODE_ROOT.resolve())): digest(p) for p in (
        Path(__file__).resolve(), CODE_ROOT / "Simulator" / "Approximator.py",
        CODE_ROOT / "Simulator" / "baselines.py",
        CODE_ROOT / "Simulator" / "runners" / "train_aggregation.py",
        CODE_ROOT / "Simulator" / "cases" / "aggregation.py",
        CODE_ROOT / "Simulator" / "cases" / "aggregation_case_T24.py")}

def common_archive(agg, scene, method):
    return {"method": method, "case_seed": 0, "T": 24,
            "n_cl": 5, "n_tcl": 40, "n_pv": 5,
            "discrete_rate": 0., "pmin": np.asarray(agg.pmin),
            "pmax": np.asarray(agg.pmax), "scenario_sha256": scene["scenario_sha256"]}

def reuse(path, scene, *, training=False):
    if not path.exists():
        return False
    if not any(path.iterdir()):
        return False
    provenance_path = path / "provenance.json"
    if not provenance_path.is_file():
        raise FileExistsError(f"Refusing to overwrite an unverified directory: {path}")
    record = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (record.get("status") != "completed" or
            record.get("scenario_sha256") != scene["scenario_sha256"] or
            record.get("final_polytope_sha256") != digest(path / "final_polytope.npz")):
        raise ValueError(f"Existing output is incomplete or has a different scene: {path}")
    if training:
        expected_stages = [{"n_train": n, "epochs": n + 1, "lr": lr, "rate_opt_feas": rate}
                           for n, lr, rate in STAGES]
        config_path = path / "training_config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if (config.get("status") != "completed" or config.get("optimizer_updates") != 1000
                    or config.get("gradient_mode") != "exact" or config.get("training_seed") != 0
                    or config.get("stages") != expected_stages
                    or config.get("phase_end_updates") != [300, 500, 1000]):
                raise ValueError(f"Existing training does not match the final-lr-1 protocol: {path}")
        elif (record.get("optimizer_updates") != 1000
              or record.get("third_stage_learning_rate") != 1.
              or record.get("training_stages") != expected_stages):
            raise ValueError(f"Existing training has no verified final-lr-1 schedule: {path}")
    print(f"Reusing verified completed output: {path}", flush=True)
    return True

def verify_paired_execution_snapshot(source, recorded_hash, current_path):
    """Allow a CLI-only update only when the executed numerical AST is equal."""
    snapshot = source.parent / "executed_compare_gradients.py"
    if not snapshot.is_file() or snapshot.is_symlink() or digest(snapshot) != recorded_hash:
        raise ValueError("The paired source wrapper changed without its exact recorded execution snapshot")
    required = ("sample_errors", "take_step", "make_optimizer", "surrogate", "configure", "sync")
    def numerical_ast(path):
        module = ast.parse(path.read_text(encoding="utf-8-sig"))
        functions = {node.name: ast.dump(node, include_attributes=False) for node in module.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in required}
        constants = {}
        for node in module.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ("STAGES", "OPTIONS"):
                        constants[target.id] = ast.dump(node.value, include_attributes=False)
        if set(functions) != set(required) or set(constants) != {"STAGES", "OPTIONS"}:
            raise ValueError("The paired execution snapshot lacks a required numerical definition")
        return functions, constants
    executed, current = numerical_ast(snapshot), numerical_ast(current_path)
    if executed != current:
        raise ValueError("Paired numerical functions or STAGES/OPTIONS changed after training")
    return {"snapshot_verified": True, "execution_snapshot": str(snapshot.resolve()),
            "recorded_execution_sha256": recorded_hash, "current_cli_sha256": digest(current_path),
            "current_wrapper_differs": digest(current_path) != recorded_hash,
            "numerical_functions_ast_equal": list(required), "numerical_constants_ast_equal": ["STAGES", "OPTIONS"]}

def verify_peb(source, original):
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    for key, expected in {"status": "completed", "case_seed": 0,
                          "training_seed": 0, "iterations": 1000,
                          "polytope_shape": [94, 24]}.items():
        if metadata.get(key) != expected:
            raise ValueError(f"PEB source {key} differs: {metadata.get(key)!r}")
    if metadata.get("stages") != [[300, .02, 1.], [200, .1, .01], [500, 1., 1e-6]]:
        raise ValueError("PEB source must use the current final-stage learning rate 1")
    paths = {"compare_gradients.py": CODE_ROOT / "Simulator/runners/compare_gradients.py",
             "Approximator.py": CODE_ROOT / "Simulator/Approximator.py",
             "aggregation.py": CODE_ROOT / "Simulator/cases/aggregation.py",
             "aggregation_case_T24.py": CODE_ROOT / "Simulator/cases/aggregation_case_T24.py"}
    checks = {name: digest(path) == metadata.get("source_sha256", {}).get(name)
              for name, path in paths.items()}
    if not all(value for name, value in checks.items() if name != "compare_gradients.py"):
        raise ValueError(f"PEB numerical core source hashes differ from current implementation: {checks}")
    snapshot_proof = None
    if not checks["compare_gradients.py"]:
        snapshot_proof = verify_paired_execution_snapshot(
            source, metadata.get("source_sha256", {}).get("compare_gradients.py"), paths["compare_gradients.py"])
    directions = np.random.RandomState(0).randn(1000, 24)
    for name in ("training_directions.npy", "exact_used_directions.npy"):
        np.testing.assert_array_equal(np.load(source / name), directions)
    direction_hash = hashlib.sha256(directions.tobytes()).hexdigest()
    if direction_hash != metadata["training_directions_sha256"]:
        raise ValueError("PEB source direction hash differs")
    cumulative = np.tril(np.ones((24, 24)))
    cumulative /= cumulative.sum(axis=1, keepdims=True)
    initial_A = np.vstack((np.eye(24), -np.eye(24), cumulative[1:], -cumulative[1:]))
    calculator = ErrorCalculator({"model": original}, initial_A, np.zeros(94))
    configure(calculator)
    initial_b = np.array([row @ calculator.optimize_direction(-row) for row in initial_A])
    network = PreTrainNet(initial_A, initial_b, device="cpu").double()
    with torch.no_grad():
        A, b = network()
    with np.load(source / "initial_polytope.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(A[0].numpy(), saved["A"])
        np.testing.assert_array_equal(b[0].numpy(), saved["b"])
    with np.load(source / "exact_final.npz", allow_pickle=False) as saved:
        final = {name: saved[name].copy() for name in ("A", "b")}
    if final["A"].shape != (94, 24) or final["b"].shape != (94,):
        raise ValueError("Unexpected final PEB dimensions")
    print("PEB verified: source hashes, training directions, and current-case initialization match", flush=True)
    return final, {"source": str(source.resolve()), "source_hash_checks": checks,
                   "paired_execution_snapshot_proof": snapshot_proof,
                   "source_metadata_sha256": digest(source / "metadata.json"),
                   "source_polytope_sha256": digest(source / "exact_final.npz"),
                   "current_case_initial_polytope_exactly_equal": True,
                   "training_directions_sha256": direction_hash}

def prepare_comparison(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peb-source", required=True, type=Path,
                        help="Completed paired case-0/training-seed-0 PEB run")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    # build_aggregator explicitly imports aggregation_case_T24.Aggregator and
    # generates gen_CL(5), gen_TCL(40), gen_PV(5), in that order.
    agg = build_aggregator(24, 0)
    original = agg.case_aggregator(build_only=True)
    scene = scenario(agg)
    sources = source_hashes()
    peb, peb_checks = verify_peb(args.peb_source.resolve(), original)
    peb_output = BASE / "exact/peb/verified_case0_seed0"
    if not reuse(peb_output, scene, training=True):
        peb_output.mkdir(parents=True, exist_ok=True)
        np.savez(peb_output / "final_polytope.npz", **peb,
                 **common_archive(agg, scene, "peb_exact"), training_seed=0)
        write_json(peb_output / "scenario.json", scene)
        write_json(peb_output / "provenance.json", {
            "status": "completed", "scenario_sha256": scene["scenario_sha256"],
            "source_sha256": sources, **peb_checks,
            "optimizer_updates": 1000, "third_stage_learning_rate": 1.,
            "training_stages": [{"n_train": n, "epochs": n + 1, "lr": lr, "rate_opt_feas": rate}
                                for n, lr, rate in STAGES],
            "final_polytope_sha256": digest(peb_output / "final_polytope.npz")})

    box_output = BASE / "baselines/box/verified_case0"
    if reuse(box_output, scene):
        with np.load(box_output / "final_polytope.npz", allow_pickle=False) as saved:
            A_cube, b_cube = saved["A"].copy(), saved["b"].copy()
    else:
        (A_cube, b_cube), elapsed = build_cube(agg)
        box_output.mkdir(parents=True, exist_ok=True)
        np.savetxt(box_output / "A_cube.csv", A_cube, delimiter=",")
        np.savetxt(box_output / "b_cube.csv", b_cube, delimiter=",")
        np.savez(box_output / "final_polytope.npz", A=A_cube, b=b_cube,
                 **common_archive(agg, scene, "pb_inner"), comp_time=elapsed)
        old_box = BASE / "baselines/box"
        differences = {}
        for name, values in (("A_cube.csv", A_cube), ("b_cube.csv", b_cube)):
            old = np.loadtxt(old_box / name, delimiter=",")
            if name.startswith("b"):
                old = old.reshape(-1)
            differences[name] = {"old_shape": list(old.shape), "new_shape": list(values.shape),
                                 "max_abs_difference": (float(np.max(np.abs(old - values)))
                                                        if old.shape == values.shape else None),
                                 "exactly_equal": bool(np.array_equal(old, values)),
                                 "old_sha256": digest(old_box / name)}
        write_json(box_output / "scenario.json", scene)
        write_json(box_output / "provenance.json", {
            "status": "completed", "scenario_sha256": scene["scenario_sha256"],
            "source_sha256": sources, "builder": "Simulator.baselines.build_cube",
            "legacy_box_comparison": differences, "comp_time": elapsed,
            "final_polytope_sha256": digest(box_output / "final_polytope.npz")})
        print(f"PB inner recomputed: {differences}", flush=True)

    pb_output = BASE / "exact/pb/seed_0"
    if not reuse(pb_output, scene, training=True):
        pb_output = new_run_dir(pb_output, pb_output)
        case = agg.case_aggregator(A=A_cube, b=b_cube, gradient_mode="exact", template="pb")
        configure(case["errorcalculator"])
        case["trainer_configure"].update(feas_tol=0., opt_tol=0., active_tol=1e-5)
        directions = np.random.RandomState(0).randn(1000, 24)
        np.save(pb_output / "training_directions.npy", directions)
        write_json(pb_output / "scenario.json", scene)
        provenance = {"status": "running", "scenario_sha256": scene["scenario_sha256"],
                      "source_sha256": sources, "training_seed": 0,
                      "training_directions_sha256": hashlib.sha256(directions.tobytes()).hexdigest(),
                      "training_directions_note": "Trainer seeded with 0, n_cal=1; same RandomState(0) directions as paired PEB",
                      "solver_options": OPTIONS, "actual_stage_updates": [300, 200, 500],
                      "box_source": str(box_output), "peb_reference": str(peb_output)}
        write_json(pb_output / "provenance.json", provenance)
        print(f"Starting PB exact, 1000 updates: {pb_output}", flush=True)
        try:
            trainer = train_case(case, mode="exact", stages=STAGES, output_dir=pb_output,
                                 seed=0, device=args.device,
                                 case_config={"horizon": 24, "template": "pb", "case_seed": 0,
                                              "counts": [5, 40, 5], "solver_options": OPTIONS,
                                              "error_cutoff": 0., "active_tol": 1e-5,
                                              "scenario_sha256": scene["scenario_sha256"],
                                              "box_dir": str(box_output)})
            if len(trainer.loss_history) != 1000:
                raise RuntimeError("PB training did not perform exactly 1000 updates")
            # No NumPy draws occur between the last direction and train_case's
            # return; this checks the consumed direction sequence, not just seed.
            expected_rng = np.random.RandomState(0)
            expected_rng.randn(1000, 24)
            np.testing.assert_array_equal(np.random.randn(24), expected_rng.randn(24))
            with np.load(pb_output / "final_polytope.npz", allow_pickle=False) as saved:
                final = {key: saved[key].copy() for key in ("A", "b")}
            np.savez(pb_output / "final_polytope.npz", **final,
                     **common_archive(agg, scene, "pb_exact"), training_seed=0)
            provenance.update(status="completed", optimizer_updates=1000,
                              random_stream_consumption_verified=True,
                              final_polytope_sha256=digest(pb_output / "final_polytope.npz"))
        except Exception as exc:
            provenance.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            write_json(pb_output / "provenance.json", provenance)
    print(f"Prepared PEB exact: {peb_output}\nPrepared PB inner: {box_output}\nPrepared PB exact: {pb_output}", flush=True)

def main(argv=None):
    """Dispatch ordinary training or explicit verified-comparison preparation."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--prepare-comparison" in arguments:
        arguments.remove("--prepare-comparison")
        return prepare_comparison(arguments)
    return _train_main(arguments)


if __name__ == "__main__":
    main()
