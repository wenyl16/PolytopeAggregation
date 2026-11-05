"""Evaluate out-of-sample microgrid disaggregation using saved devices and weights."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch

from Simulator.Approximator import PreTrainNet
from Simulator.cases.microgrid_case import MicroGrid
from Simulator.paths import DATA_ROOT, RESULTS_ROOT, comparison_output_path
from Simulator.runners.train_aggregation import new_run_dir


OWNED_OUTPUT_FILES = ("errors.csv", "results.pkl", "metadata.json", "summary.json", "solver_diagnostics.json")
METRIC_COLUMNS = ("feasibility_error", "feasibility_error_rated", "optimality_error_rated",
                  "objective_error", "objective_error_rated")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_summary(table):
    """Summarize finite raw values, retaining signed directional cost errors."""
    summary = {}
    for name in METRIC_COLUMNS:
        values = table[name].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        summary[name] = {
            "finite_count": int(len(finite)),
            "missing_or_nonfinite_count": int(len(values) - len(finite)),
            **{key: (float(operation(finite)) if len(finite) else None)
               for key, operation in (("mean", np.mean), ("median", np.median),
                                      ("p95", lambda x: np.percentile(x, 95)),
                                      ("max", np.max))},
        }
    return summary


def prepare_output(requested, overwrite, training_dir):
    if not overwrite:
        return new_run_dir(RESULTS_ROOT / "microgrid" / "disaggregation", requested)
    output = comparison_output_path(requested or RESULTS_ROOT / "microgrid" / "disaggregation")
    if output == training_dir.resolve():
        raise ValueError("Evaluation output must differ from the training directory")
    output.mkdir(parents=True, exist_ok=True)
    for name in OWNED_OUTPUT_FILES:
        path = output / name
        if not path.resolve().is_relative_to(output):
            raise ValueError(f"Output file resolves outside the evaluation directory: {path}")
        if path.exists():
            if not path.is_file():
                raise ValueError(f"Owned output path is not a file: {path}")
            path.unlink()
    return output


def load_polytope(training_dir, horizon):
    """Recover the trained forward result before building physical models.

    Passing these A/b to case_mg avoids recomputing initial bounds with one
    mixed-integer support solve per row.
    """
    state = torch.load(training_dir / "pretrainnet_weights.pth", map_location="cpu", weights_only=True)
    raw_b = state["b_net.0.bias"]
    raw_A = state["A_net.0.bias"].reshape(len(raw_b), horizon)
    model = PreTrainNet(raw_A.numpy(), raw_b.numpy(), device="cpu").to(dtype=raw_A.dtype)
    model.load_state_dict(state)
    with torch.no_grad():
        A_tensor, b_tensor = model()
    A, b = A_tensor[0].numpy().copy(), b_tensor[0].numpy().copy()
    if not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError("Saved weights produce nonfinite polytope coefficients")
    final_path = training_dir / "final_polytope.npz"
    if final_path.is_file():
        with np.load(final_path, allow_pickle=False) as final:
            np.testing.assert_array_equal(A, final["A"], err_msg="Weights and final A differ")
            np.testing.assert_array_equal(b, final["b"], err_msg="Weights and final b differ")
    return A, b, final_path.is_file()


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=RESULTS_ROOT / "microgrid" / "training")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-scale", type=float, default=0.05)
    parser.add_argument("--pv-shift", type=float, default=0.0)
    parser.add_argument("--cl-shift", type=float, default=0.0)
    parser.add_argument("--temp-shift", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace only this evaluator's owned files in a fixed output directory")
    parser.add_argument("--count-branching", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--storage-headroom", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--solver-threads", type=int)
    args = parser.parse_args(argv)
    if args.num_samples < 1 or not np.isfinite(args.noise_scale) or args.noise_scale < 0:
        parser.error("--num-samples must be positive and --noise-scale finite and nonnegative")
    if not np.isfinite([args.pv_shift, args.cl_shift, args.temp_shift]).all():
        parser.error("Perturbation shifts must be finite")
    if args.solver_threads is not None and args.solver_threads < 1:
        parser.error("--solver-threads must be positive")
    args.training_dir = args.training_dir.resolve()
    weight_path = args.training_dir / "pretrainnet_weights.pth"
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    load_started = time.perf_counter()
    metadata_path = args.training_dir / "metadata.pkl"
    source_paths = [weight_path, metadata_path]
    training_config_path = args.training_dir / "training_config.json"
    training_config = {}
    if training_config_path.is_file():
        training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
        source_paths.append(training_config_path)
    profiles_path = DATA_ROOT / "profiles_data" / "profiles_data.npz"
    source_paths.append(profiles_path)
    if training_config.get("profiles_sha256") and training_config["profiles_sha256"] != file_hash(profiles_path):
        raise ValueError("Physical profiles differ from those used during training")
    final_path = args.training_dir / "final_polytope.npz"
    if final_path.is_file():
        source_paths.append(final_path)
    protected_hashes = {str(path): file_hash(path) for path in source_paths}
    with metadata_path.open("rb") as stream:
        metadata = pickle.load(stream)
    horizon = int(metadata.get("T", 24))
    A, b, final_checked = load_polytope(args.training_dir, horizon)
    mg = MicroGrid(seed=args.seed, T=horizon, data=metadata)
    count_branching = (args.count_branching if args.count_branching is not None
                       else training_config.get("count_branching", False))
    storage_headroom = (args.storage_headroom if args.storage_headroom is not None
                        else training_config.get("storage_headroom", False))
    if storage_headroom and not count_branching:
        parser.error("--storage-headroom requires --count-branching")
    solver_options = {"Threads": args.solver_threads or (8 if count_branching else 1), "Seed": 0}
    if count_branching:
        solver_options.update(MIPFocus=2, Symmetry=2)
    mg.case_mg(A=A, b=b, conservativeness=0.0, device="cpu", is_uncertain=False,
               solver_options=solver_options)
    physical_solver = None
    if count_branching:
        from Simulator.microgrid_count_solver import CountBranchingSolver
        solver_options = {**solver_options, "Presolve": 0}
        physical_solver = CountBranchingSolver(options=solver_options,
            storage_parameters=mg.ESS_list if storage_headroom else None, deltaT=mg.deltaT)
        mg.errorcalculator.solver = physical_solver
    load_seconds = time.perf_counter() - load_started
    output = prepare_output(args.output_dir, args.overwrite, args.training_dir)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        output_dir=str(output.resolve()), status="running", T=horizon,
        device_counts={name: len(metadata[name + "_list"])
                       for name in ("MT", "PV", "TL", "CL", "ESS", "HP")},
        training_source_sha256=protected_hashes, final_polytope_checked=final_checked,
        backend="gurobi_persistent_integer_counts" if count_branching else "pyomo", solver="gurobi", solver_options=solver_options,
        count_branching=bool(count_branching),
        storage_headroom=bool(storage_headroom),
        physical_integer_constraints="unrelaxed",
        case_source_sha256=file_hash(Path(__file__).parents[1] / "cases" / "microgrid_case.py"),
        evaluator_source_sha256=file_hash(Path(__file__)),
        load_initialization_seconds=load_seconds,
        direction_definition="c = -numpy.random.rand(T), independently for each sample",
        feasibility_error_rated_definition=(
            "Squared Euclidean distance of the approximate support point to the perturbed physical region "
            "in normalized power coordinates"),
        optimality_error_rated_definition=(
            "Squared Euclidean distance of the nominal physical support point to the learned approximate "
            "region in normalized power coordinates"),
        objective_error_definition=(
            "Signed physical-power directional support cost difference: approximate minus nominal physical "
            "region; uses the shared normalized-power direction and is not the geometric optimality error"),
        objective_error_rated_definition="objective_error / abs(nominal directional physical-power cost)",
        recorded_samples=0, completed_samples=0, failed_samples=0,
    )
    (output / "metadata.json").write_text(json.dumps(config, indent=2, allow_nan=False), encoding="utf-8")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    evaluation_started = time.perf_counter()
    partial_results = []

    def progress(row, processed_count, total_count):
        partial_results.append(row)
        if processed_count % 10 == 0 or processed_count == total_count:
            completed = sum(item.get("solver_status") == "completed" for item in partial_results)
            config.update(recorded_samples=len(partial_results), completed_samples=completed,
                          failed_samples=len(partial_results) - completed,
                          evaluation_elapsed_seconds=time.perf_counter() - evaluation_started,
                          elapsed_seconds=time.perf_counter() - started)
            (output / "metadata.json").write_text(json.dumps(config, indent=2, allow_nan=False), encoding="utf-8")
            print(f"Evaluation {processed_count}/{total_count}: completed={completed}, "
                  f"failed={len(partial_results) - completed}, "
                  f"elapsed={config['evaluation_elapsed_seconds']:.1f}s", flush=True)

    failure = None
    try:
        results = mg.test_disaggregation_error(n_test=args.num_samples, noise_scale=args.noise_scale,
                                               pv_shift=args.pv_shift, cl_shift=args.cl_shift,
                                               temp_shift=args.temp_shift, progress=progress)
    except Exception as exc:
        results = partial_results
        failure = exc
        config.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
    config["evaluation_elapsed_seconds"] = time.perf_counter() - evaluation_started
    if physical_solver is not None:
        (output / "solver_diagnostics.json").write_text(json.dumps({
            "diagnostics": physical_solver.diagnostics,
            "solves": physical_solver.solve_records}, indent=2, allow_nan=False), encoding="utf-8")
        config["count_solver_source_sha256"] = file_hash(Path(__file__).parents[1] / "microgrid_count_solver.py")
    columns = ("sample_index", "solver_status", "failure", *METRIC_COLUMNS)
    # Preserve one row per direction, including failures; never filter columns separately.
    table = pd.DataFrame([{key: row.get(key) for key in columns} for row in results], columns=columns)
    for name in METRIC_COLUMNS:
        table[name] = pd.to_numeric(table[name], errors="raise")
    table.to_csv(output / "errors.csv", index=False)
    with (output / "results.pkl").open("wb") as stream:
        pickle.dump(results, stream)
    config.update(recorded_samples=len(results),
                  completed_samples=sum(row.get("solver_status") == "completed" for row in results))
    config["failed_samples"] = len(results) - config["completed_samples"]
    config["solver_status_counts"] = dict(Counter(row.get("solver_status", "missing") for row in results))
    if failure is None and len(results) != args.num_samples:
        failure = RuntimeError(f"Expected {args.num_samples} result rows, received {len(results)}")
        config.update(status="failed", failure=str(failure))
    changed_sources = [path for path, digest in protected_hashes.items() if file_hash(path) != digest]
    config["training_sources_unchanged"] = not changed_sources
    if changed_sources:
        failure = RuntimeError(f"Training source files changed during evaluation: {changed_sources}")
        config.update(status="failed", failure=str(failure))
    if failure is None:
        config["status"] = "completed_with_failures" if config["failed_samples"] else "completed"
    config["elapsed_seconds"] = time.perf_counter() - started
    summary = {key: config[key] for key in ("status", "T", "device_counts", "num_samples", "seed",
                                            "noise_scale", "recorded_samples", "completed_samples",
                                            "failed_samples", "solver_status_counts", "elapsed_seconds",
                                            "load_initialization_seconds", "evaluation_elapsed_seconds",
                                            "training_sources_unchanged")}
    summary["metrics"] = metric_summary(table)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (output / "metadata.json").write_text(json.dumps(config, indent=2, allow_nan=False), encoding="utf-8")
    if failure is not None:
        raise failure
    print(table.describe().to_string())
    print(f"Saved disaggregation results: {output}")


if __name__ == "__main__":
    main()
