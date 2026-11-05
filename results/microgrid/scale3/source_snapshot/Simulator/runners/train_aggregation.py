"""Train the T24/T96 aggregation polytope with exact or approximate gradients."""

import argparse
from datetime import datetime
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

from Simulator.Approximator import PreTrainNet, Trainer
from Simulator.paths import RESULTS_ROOT, aggregation_result_dir


def build_aggregator(horizon=24, seed=0):
    if horizon == 24:
        from Simulator.cases.aggregation_case_T24 import Aggregator
        counts = (5, 40, 5)
    elif horizon == 96:
        from Simulator.cases.aggregation_case_T96 import Aggregator
        counts = (100, 200, 200)
    else:
        raise ValueError("horizon must be 24 or 96")
    agg = Aggregator(seed=seed, discrete_rate=0.0)
    agg.gen_CL(counts[0])
    agg.gen_TCL(counts[1])
    agg.gen_PV(counts[2])
    return agg


def build_stages(horizon=24, schedule="three-stage", n_train=None, stage_multiplier=1):
    """Return Trainer calls, scaling actual n_train+1 updates in each stage."""
    if horizon not in (24, 96):
        raise ValueError("horizon must be 24 or 96")
    if schedule not in ("three-stage", "single-stage"):
        raise ValueError("schedule must be three-stage or single-stage")
    if n_train is not None and (not isinstance(n_train, (int, np.integer)) or n_train < 0):
        raise ValueError("n_train must be a nonnegative integer")
    if not isinstance(stage_multiplier, (int, np.integer)) or stage_multiplier < 1:
        raise ValueError("stage_multiplier must be a positive integer")
    learning_rates = (0.02, 0.1, 0.2) if horizon == 24 else (0.01, 0.05, 0.1)
    stages = list(zip((300, 200, 500), learning_rates, (1.0, 1e-2, 1e-6)))
    if schedule == "single-stage":
        updates = sum(n + 1 for n, _, _ in stages)
        stages = [(updates - 1 if n_train is None else n_train, *stages[-1][1:])]
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("exact", "approx"), required=True)
    parser.add_argument("--horizon", type=int, choices=(24, 96), default=24)
    parser.add_argument("--template", choices=("peb", "pb"), default="peb")
    parser.add_argument("--box-dir", type=Path,
                        help="PB initialization directory containing A_cube.csv and b_cube.csv")
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--case-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--schedule", choices=("three-stage", "single-stage"), default="three-stage",
                        help="Single-stage uses the final stage's lr/weight for the same total updates")
    parser.add_argument("--n-train", type=int,
                        help="Override with n_train+1 updates; uses first/final parameters for three/single-stage")
    parser.add_argument("--stage-multiplier", type=int, default=1,
                        help="Multiply actual optimizer updates in every selected stage (default: 1)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.n_train is not None and args.n_train < 0:
        parser.error("--n-train must be nonnegative")
    if args.stage_multiplier < 1:
        parser.error("--stage-multiplier must be positive")
    if args.box_dir is not None and args.template != "pb":
        parser.error("--box-dir requires --template pb")
    agg = build_aggregator(args.horizon, args.case_seed)
    A = b = None
    if args.template == "pb":
        box = args.box_dir or RESULTS_ROOT / "aggregation" / f"T{args.horizon}" / "baselines" / "box"
        A = np.loadtxt(box / "A_cube.csv", delimiter=",")
        b = np.loadtxt(box / "b_cube.csv", delimiter=",").reshape(-1)
        if A.ndim != 2 or A.shape != (len(b), args.horizon):
            parser.error("box dimensions do not match the requested horizon")
    elif args.mode == "exact":
        cumulative = np.tril(np.ones((agg.T, agg.T)))
        cumulative /= cumulative.sum(axis=1, keepdims=True)
        A = np.vstack((np.eye(agg.T), -np.eye(agg.T), cumulative[1:], -cumulative[1:]))
    case = agg.case_aggregator(A=A, b=b, gradient_mode=args.mode, template=args.template)
    stages = build_stages(args.horizon, args.schedule, args.n_train, args.stage_multiplier)
    output = new_run_dir(aggregation_result_dir(args.horizon, args.mode, args.template), args.output_dir)
    train_case(case, mode=args.mode, stages=stages, output_dir=output,
               seed=args.training_seed, device=args.device,
               case_config={"horizon": args.horizon, "template": args.template,
                            "case_seed": args.case_seed,
                            "schedule": args.schedule, "n_train_override": args.n_train,
                            "stage_multiplier": args.stage_multiplier,
                            "box_dir": str(box.resolve()) if args.template == "pb" else None})


if __name__ == "__main__":
    main()
