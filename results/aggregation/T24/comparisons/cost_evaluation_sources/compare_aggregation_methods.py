"""Compare aggregation regions on shared directions and/or physical linear costs."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import pickle
import shutil
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pyomo.environ as pyo
from scipy.io import loadmat
import torch

from Simulator.Approximator import PreTrainNet
from Simulator.testers.compute_polytope_errors_pyomo import build_error_calculator, evaluate_calculator
from Simulator.paths import CODE_ROOT, RESULTS_ROOT, comparison_output_path, prepare_comparison_output
from Simulator.runners.train_aggregation import build_aggregator


SOLVER_OPTIONS = {"FeasibilityTol": 1e-9, "OptimalityTol": 1e-9,
                  "BarConvTol": 1e-12, "Threads": 1, "Seed": 0}


def numpy_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def default_methods(base):
    """Saved regions, including an available federated PEB baseline.

    PEB outer is rebuilt from the same physical case by the caller.
    """
    methods = [
        ("proposed_peb_exact", base / "exact" / "peb" / "verified_case0_seed0"),
        ("proposed_pb_exact", base / "exact" / "pb" / "seed_0"),
        ("peb_inner", base / "baselines" / "matlab" / "verified_case0"),
        ("pb_inner", base / "baselines" / "box" / "verified_case0"),
        ("cutting_plane", base / "baselines" / "cutting plane" / "cutting_plane.npz"),
        ("cutting_plane_modified", base / "baselines" / "cutting plane" / "cutting_plane_modified.npz"),
        ("pve", base / "baselines" / "pve" / "pve.npz"),
        ("pve_modified", base / "baselines" / "pve" / "pve_modified.npz"),
    ]
    federated = base / "baselines" / "federated_peb"
    if (federated / "final_polytope.npz").is_file():
        provenance_path = federated / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8-sig")) if provenance_path.is_file() else {}
        if (provenance.get("status") == "completed" and
                provenance.get("paper_core_protocol") in ("original_equations_12_14_15_17_19_36_38_algorithm1",
                                                         "user_authorized_lexicographic_peb_variant_v1")):
            methods.append(("federated_peb", federated))
    return methods


def verify_scenario(path, agg, case_seed, required=False):
    """Reject a mismatched case before comparing normalized coordinates."""
    archive = path / "final_polytope.npz" if path.is_dir() else path
    record = {"source": str(path.resolve()), "scenario_verified": False}
    if archive.suffix != ".npz" or not archive.is_file():
        if required:
            raise ValueError(f"A scenario-labelled final_polytope.npz is required: {path}")
        return record
    record["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    with np.load(archive, allow_pickle=False) as data:
        expected = {"T": agg.T, "case_seed": case_seed, "n_cl": len(agg.CL_list),
                    "n_tcl": len(agg.TCL_list), "n_pv": len(agg.PV_list)}
        missing = [key for key in (*expected, "pmin", "pmax") if key not in data]
        if missing and required:
            raise ValueError(f"Missing scenario fields {missing}: {archive}")
        for key, value in expected.items():
            if key in data and data[key].item() != value:
                raise ValueError(f"Scenario mismatch for {key}: {archive}")
        for key in ("pmin", "pmax"):
            if key in data and not np.array_equal(data[key], getattr(agg, key)):
                raise ValueError(f"Physical normalization mismatch for {key}: {archive}")
        record["scenario_verified"] = not missing
    provenance = path / "provenance.json" if path.is_dir() else None
    if provenance is not None and provenance.is_file():
        record["provenance"] = json.loads(provenance.read_text(encoding="utf-8-sig"))
        proof = record["provenance"]
        if proof.get("status") != "completed" or proof.get("final_polytope_sha256") != record["archive_sha256"]:
            raise ValueError(f"Result is incomplete or differs from its provenance: {path}")
    scene_path = path / "scenario.json" if path.is_dir() else None
    if scene_path is not None and scene_path.is_file():
        scene = json.loads(scene_path.read_text(encoding="utf-8-sig"))
        actual = {"aggregator_class": f"{type(agg).__module__}.{type(agg).__name__}",
                  "case_seed": case_seed, "T": agg.T, "deltaT": agg.deltaT,
                  "discrete_rate": agg.discrete_rate, "n_cl": len(agg.CL_list),
                  "n_tcl": len(agg.TCL_list), "n_pv": len(agg.PV_list),
                  "n_ev": len(agg.EV_list), "n_ess": len(agg.ESS_list)}
        if any(scene.get(key) != value for key, value in actual.items()):
            raise ValueError(f"Scenario generator or device configuration differs: {path}")
        for key in ("CL_list", "TCL_list", "PV_list", "EV_list", "ESS_list",
                    "theta_amb", "pv_profiles", "load_profiles"):
            current = json.dumps(getattr(agg, key), default=numpy_json, sort_keys=True)
            if json.dumps(scene[key], sort_keys=True) != current:
                raise ValueError(f"Device/input parameters differ for {key}: {path}")
        for relative, expected_hash in scene.get("input_sha256", {}).items():
            if hashlib.sha256((CODE_ROOT / "data" / relative).read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"Scenario input file differs: {relative}")
        payload = {key: value for key, value in scene.items() if key != "scenario_sha256"}
        signature = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                             separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        if signature != scene["scenario_sha256"]:
            raise ValueError(f"Scenario signature is invalid: {path}")
        with np.load(archive, allow_pickle=False) as data:
            if data["scenario_sha256"].item() != signature:
                raise ValueError(f"Archive and device-parameter signatures differ: {path}")
        record.update(full_device_parameters_verified=True, scenario_sha256=signature)
    return record


def evaluate_region(agg, directions, region, *, backend="pyomo", progress=None, checkpoint_path=None):
    if backend == "native":
        from Simulator.testers.compute_polytope_errors_and_costs_gurobi import evaluate_native
        return evaluate_native(agg, directions, region, progress=progress,
                               checkpoint_path=checkpoint_path)
    calculator = build_error_calculator(agg, **region)
    for solver in (calculator.solver, calculator.cvx_solver):
        solver.options.update(SOLVER_OPTIONS)
    return evaluate_calculator(calculator, directions)


def evaluate_method(task):
    """A process owns its solver models and its method-specific checkpoints."""
    name, horizon, seed, directions, region, backend, output = task
    output = Path(output)
    agg = build_aggregator(horizon, seed)

    def progress(record):
        record = {"method": name, **record}
        temporary = output / f"{name}_progress.tmp"
        temporary.write_text(json.dumps(record, default=numpy_json, indent=2), encoding="utf-8")
        temporary.replace(output / f"{name}_progress.json")
        completed = record.get("completed", 0)
        if completed in (0, 1, len(directions)) or completed % 1000 == 0:
            print(f"{name}: {completed}/{len(directions)} directions", flush=True)

    values = evaluate_region(agg, directions, region, backend=backend, progress=progress,
                             checkpoint_path=output / f"{name}_checkpoint.npz")
    return name, values


def reuse_costs(source, output, metadata, regions, coefficients):
    """Reuse the exact earlier 100 cost samples only after checking all inputs."""
    source = Path(source)
    previous = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if previous.get("status") != "completed":
        raise ValueError("The source cost evaluation is not completed")
    for key in ("case_seed", "horizon", "aggregator_class", "counts", "solver_options", "method_order"):
        if previous.get(key) != metadata.get(key):
            raise ValueError(f"Source cost protocol differs: {key}")
    for key in ("pmin", "pmax"):
        np.testing.assert_array_equal(previous[key], metadata[key])
    for name in regions:
        if name == "peb_outer":
            with np.load(source / "peb_outer.npz", allow_pickle=False) as old:
                for key in ("A", "b"):
                    np.testing.assert_array_equal(old[key], regions[name][key])
        elif previous["scenario_audit"][name].get("archive_sha256") != metadata["scenario_audit"][name].get("archive_sha256"):
            raise ValueError(f"The saved approximation changed since the cost evaluation: {name}")
    np.testing.assert_array_equal(np.load(source / "cost_coefficients.npy", allow_pickle=False), coefficients)
    with (source / "cost_comparison.pkl").open("rb") as stream:
        costs = pickle.load(stream)
    if list(costs) != ["original", *regions]:
        raise ValueError("Source costs do not contain the identical ordered methods")
    files = ["cost_coefficients.npy", "cost_comparison.pkl", "cost_samples.csv"]
    for name, values in costs.items():
        if np.asarray(values).shape != (len(coefficients),) or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid source cost samples: {name}")
        filename = f"{name}_costs.npy"
        np.testing.assert_array_equal(np.load(source / filename, allow_pickle=False), values)
        files.append(filename)
    for filename in files:
        shutil.copyfile(source / filename, output / filename)
    metadata["reused_cost_evaluation"] = {
        "source": str(source.resolve()),
        "source_metadata_sha256": hashlib.sha256((source / "metadata.json").read_bytes()).hexdigest(),
        "bitwise_preserved_files": {filename: hashlib.sha256((output / filename).read_bytes()).hexdigest()
                                     for filename in files},
    }
    print(f"Reused {len(coefficients)} identical physical-cost samples from {source}", flush=True)
    return costs


def load_polytope(path):
    """Read an H/V archive or a training directory; prefer its actual final weights."""
    path = Path(path)
    if path.is_dir():
        archive = path / "final_polytope.npz"
        if archive.is_file():
            return load_polytope(archive)
        weight_path = path / "pretrainnet_weights.pth"
        if weight_path.is_file():
            state = torch.load(weight_path, map_location="cpu", weights_only=True)
            raw_A, raw_b = state["A_net.0.bias"], state["b_net.0.bias"]
            rows = raw_b.numel()
            model = PreTrainNet(raw_A.reshape(rows, -1).numpy(), raw_b.numpy(), device="cpu").to(dtype=raw_A.dtype)
            model.load_state_dict(state)
            with torch.no_grad():
                A, b = model()
            return {"A": A[0].numpy(), "b": b[0].numpy()}
        if (path / "A_cube.csv").is_file():
            return {"A": np.loadtxt(path / "A_cube.csv", delimiter=","),
                    "b": np.loadtxt(path / "b_cube.csv", delimiter=",").reshape(-1)}
        snapshots = path / "snapshots"
        files = [p for p in snapshots.glob("A_*.csv") if p.stem[2:].isdigit()]
        if files:
            A_path = max(files, key=lambda p: int(p.stem[2:]))
            return {"A": np.loadtxt(A_path, delimiter=","),
                    "b": np.loadtxt(A_path.with_name("b" + A_path.name[1:]), delimiter=",").reshape(-1)}
        raise FileNotFoundError(f"No polytope data in {path}")
    if path.suffix == ".mat":
        data = loadmat(path)["polytope"]
        A, b = data["A"][0][0], data["b"][0][0].reshape(-1)
        norms = np.maximum(np.linalg.norm(A, axis=1), 1e-10)
        return {"A": A / norms[:, None], "b": b / norms}
    with np.load(path, allow_pickle=False) as data:
        if "V" in data:
            return {"V": np.array(data["V"])}
        return {"A": np.array(data["A"]), "b": np.array(data["b"]).reshape(-1)}


def solve_cost(c, scale, offset, *, original=None, A=None, b=None, V=None):
    """Return min c @ physical_power; apply the same normalization to every model."""
    if V is not None:
        return float(np.min((np.asarray(V) * scale + offset) @ c))
    if original is not None:
        model = original.clone()
        for objective in model.component_objects(pyo.Objective, active=True):
            objective.deactivate()
        power = model.var_proj
    else:
        model = pyo.ConcreteModel()
        model.power = pyo.Var(range(len(c)))
        power = model.power
        model.region = pyo.Constraint(range(len(b)), rule=lambda m, i:
            sum(float(A[i, t]) * power[t] for t in range(len(c))) <= float(b[i]))
    model.evaluation_cost = pyo.Objective(expr=sum(float(c[t]) *
        (float(scale[t]) * power[t] + float(offset[t])) for t in range(len(c))))
    solver = pyo.SolverFactory("gurobi")
    solver.options["OutputFlag"] = 0
    solver.options.update(SOLVER_OPTIONS)
    result = solver.solve(model)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"Cost solve failed: {result.solver.termination_condition}")
    return float(pyo.value(model.evaluation_cost))


ERROR_SUMMARY_KEYS = ("mean_feas", "max_feas", "mean_opt", "max_opt", "count_feas_gt_1e4")


def verify_retained_context(previous, current, regions, output):
    """Keep the other metric only when it describes these exact same regions."""
    for key in ("horizon", "case_seed", "aggregator_class", "counts", "method_order",
                "solver_options", "model_sources_sha256"):
        if previous.get(key) != current.get(key):
            raise ValueError(f"Retained comparison uses a different scene/protocol: {key}")
    for key in ("pmin", "pmax"):
        np.testing.assert_array_equal(previous[key], current[key])
    for name in regions:
        if name == "peb_outer":
            with np.load(output / "peb_outer.npz", allow_pickle=False) as saved:
                for key in ("A", "b"):
                    np.testing.assert_array_equal(saved[key], regions[name][key])
        elif (previous["scenario_audit"][name].get("archive_sha256")
              != current["scenario_audit"][name].get("archive_sha256")):
            raise ValueError(f"Retained comparison uses a different region: {name}")


def cost_metadata_field(key):
    return (key.startswith(("cost_", "price_")) or key in (
        "num_cost_samples", "native_cost_diagnostics", "original_mean_cost", "daily_prices_sha256",
        "negative_price_handling", "deltaT_hours", "power_convention", "reused_cost_evaluation"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=("error", "cost", "both"), default="error")
    parser.add_argument("--horizon", type=int, choices=(24, 96), default=24)
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Shared error-evaluation directions per method (default: 1000)")
    parser.add_argument("--num-cost-samples", type=int, default=100,
                        help="Shared physical-cost coefficient vectors (default: 100)")
    parser.add_argument("--error-backend", choices=("native", "pyomo"), default="native")
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent method processes for error evaluation")
    parser.add_argument("--resume", action="store_true", help="Resume matching native checkpoints in --output-dir")
    parser.add_argument("--reuse-costs", type=Path, help="Reuse a completed identical cost evaluation")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--case-seed", type=int, default=0)
    parser.add_argument("--method", action="append", metavar="NAME=PATH",
                        help="Repeat to compare selected directories or .npz/.mat polytopes")
    parser.add_argument("--output-dir", type=Path,
                        help="Overwrite requested metric here (default: T24/comparisons); keep the other metric")
    args = parser.parse_args(argv)
    if args.num_samples < 1 or args.num_cost_samples < 1 or args.workers < 1:
        parser.error("Sample counts and workers must be positive")
    if args.resume and (args.error_backend != "native" or args.metric == "cost"):
        parser.error("--resume requires an error evaluation with --error-backend native")
    if not args.method and (args.horizon != 24 or args.case_seed != 0):
        parser.error("The prepared nine-method suite uses T24/case-seed 0; use --method for other scenarios")
    base = RESULTS_ROOT / "aggregation" / f"T{args.horizon}"
    agg = build_aggregator(args.horizon, args.case_seed)
    case = agg.case_aggregator()
    specs = default_methods(base)
    if args.method:
        specs = []
        for item in args.method:
            if "=" not in item:
                parser.error("--method must be NAME=PATH")
            name, path = item.split("=", 1)
            if not name or name in ("outer", "peb_outer", "original", "summary"):
                parser.error("method name is empty or reserved")
            specs.append((name, Path(path)))
        if len({name for name, _ in specs}) != len(specs):
            parser.error("method names must be unique")
    audit = {name: verify_scenario(path, agg, args.case_seed, required=not args.method)
             for name, path in specs}
    regions = {name: load_polytope(path) for name, path in specs}
    regions["peb_outer"] = {"A": case["A_hat"], "b": case["b_hat"]}
    audit["peb_outer"] = {"scenario_verified": True, "source": "current Aggregator support LPs"}
    output = comparison_output_path(args.output_dir or base / "comparisons")
    previous_metadata = None
    if (output / "metadata.json").is_file():
        previous_metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    elif args.resume:
        raise FileNotFoundError(f"No comparison to resume: {output}")
    want_error = args.metric in ("error", "both")
    want_cost = args.metric in ("cost", "both")
    keep_error = not want_error and (output / "error_comparison.pkl").is_file()
    keep_cost = not want_cost and (output / "cost_comparison.pkl").is_file()
    if (keep_error or keep_cost) and previous_metadata is None:
        raise ValueError("Retained metric files need their metadata.json before overwriting the other metric")
    for name, retain in (("error", keep_error), ("cost", keep_cost)):
        if not retain:
            continue
        if previous_metadata.get(f"{name}_status", previous_metadata.get("status")) != "completed":
            raise ValueError(f"Cannot retain an incomplete {name} evaluation")
        for filename in (f"{name}_samples.csv", "summary.json"):
            if not (output / filename).is_file():
                raise ValueError(f"Retained {name} evaluation is missing {filename}")
    if args.reuse_costs and args.reuse_costs.resolve() == output and want_cost:
        raise ValueError("Use --metric error to retain the costs already in this comparison directory")
    directions = np.random.RandomState(args.seed).normal(size=(args.num_samples, args.horizon))
    if args.resume:
        np.testing.assert_array_equal(np.load(output / "directions.npy", allow_pickle=False), directions)
        with np.load(output / "peb_outer.npz", allow_pickle=False) as old_outer:
            for key in ("A", "b"):
                np.testing.assert_array_equal(old_outer[key], regions["peb_outer"][key])
    metadata = {
        "status": "running", "metric": args.metric, "horizon": args.horizon,
        "case_seed": args.case_seed, "direction_seed": args.seed, "num_samples": args.num_samples,
        "num_error_samples": args.num_samples if want_error else 0,
        "num_cost_samples": args.num_cost_samples if want_cost else 0,
        "error_backend": args.error_backend, "error_workers": args.workers,
        "completed_error_methods": [],
        "aggregator_class": f"{type(agg).__module__}.{type(agg).__name__}",
        "counts": {"CL": len(agg.CL_list), "TCL": len(agg.TCL_list), "PV": len(agg.PV_list),
                   "EV": len(agg.EV_list), "ESS": len(agg.ESS_list)},
        "pmin": agg.pmin.tolist(), "pmax": agg.pmax.tolist(),
        "methods": {name: str(path.resolve()) for name, path in specs},
        "method_order": list(regions), "scenario_audit": audit, "solver_options": SOLVER_OPTIONS,
        "model_sources_sha256": {name: hashlib.sha256((CODE_ROOT / name).read_bytes()).hexdigest()
                                 for name in ("Simulator/cases/aggregation.py",
                                              f"Simulator/cases/aggregation_case_T{args.horizon}.py")},
        "error_definition": "Squared Euclidean distance in the common normalized power coordinates",
        "cost_coordinates": "physical: normalized_power * (pmax-pmin) + pmin",
        "peb_inner_model_note": "The existing PEB inner algorithm uses cumulative-energy device boundary proxies; all methods are evaluated against the same original thermal-dynamics model.",
    }
    evaluation_sources = ["Simulator/testers/compare_aggregation_methods.py", "Simulator/testers/compute_polytope_errors_pyomo.py"]
    if args.error_backend == "native":
        evaluation_sources.append("Simulator/testers/compute_polytope_errors_and_costs_gurobi.py")
    metadata["evaluation_sources_sha256"] = {
        name: hashlib.sha256((CODE_ROOT / name).read_bytes()).hexdigest() for name in evaluation_sources}
    current_evaluation_sources = dict(metadata["evaluation_sources_sha256"])
    if "peb_inner" in audit:
        metadata["legacy_baseline_model_audit"] = audit["peb_inner"].get("provenance", {}).get(
            "legacy_200_second_baseline_audit", {})
    if args.resume:
        for key in ("horizon", "case_seed", "direction_seed", "num_samples", "num_cost_samples",
                    "error_backend", "solver_options", "method_order", "model_sources_sha256"):
            if key == "num_cost_samples" and keep_cost:
                continue
            if previous_metadata.get(key) != metadata.get(key):
                raise ValueError(f"Resume protocol differs: {key}")
        for name in regions:
            if previous_metadata["scenario_audit"][name].get("archive_sha256") != audit[name].get("archive_sha256"):
                raise ValueError(f"Cannot resume a different region: {name}")
        history = list(previous_metadata.get("resume_history", []))
        backup_name = f"metadata_before_resume_{len(history) + 1:03d}.json"
        backup = output / backup_name
        if backup.exists():
            raise FileExistsError(f"Resume provenance backup already exists: {backup}")
        shutil.copyfile(output / "metadata.json", backup)
        history.append({"metadata_backup": backup_name,
                        "metadata_sha256": hashlib.sha256(backup.read_bytes()).hexdigest(),
                        "previous_status": previous_metadata.get("status"),
                        "evaluation_sources_sha256": previous_metadata.get("evaluation_sources_sha256", {}),
                        "completed_error_methods": previous_metadata.get("completed_error_methods", [])})
        metadata["resume_history"] = history
    elif keep_error or keep_cost:
        verify_retained_context(previous_metadata, metadata, regions, output)
    if keep_error:
        for key in ("num_samples", "num_error_samples", "direction_seed", "directions_sha256",
                    "error_backend", "error_workers", "completed_error_methods", "error_definition",
                    "evaluation_sources_sha256", "reused_error_evaluation", "error_completed_at_utc"):
            if key in previous_metadata:
                metadata[key] = previous_metadata[key]
    if keep_cost:
        metadata.update({key: value for key, value in previous_metadata.items() if cost_metadata_field(key)})
    metadata.update(
        metric="both" if (want_error or keep_error) and (want_cost or keep_cost) else args.metric,
        error_status="running" if want_error else "completed" if keep_error else "not_run",
        cost_status="running" if want_cost else "completed" if keep_cost else "not_run",
        last_updated_metric=args.metric, updated_at_utc=datetime.now(timezone.utc).isoformat(),
        result_policy="One current comparison; requested metrics overwrite in place",
    )
    if want_cost:
        metadata.update(cost_source="Independent negative uniform random coefficients",
                        cost_backend="pyomo", cost_sampling=f"default_rng({args.seed}).uniform, negated",
                        cost_evaluation_sources_sha256=current_evaluation_sources)
    old_summary = {}
    if (keep_error or keep_cost) and (output / "summary.json").is_file():
        old_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if not args.resume:
        prepare_comparison_output(output, args.metric)
        if want_error:
            np.save(output / "directions.npy", directions)
        np.savez(output / "peb_outer.npz", **regions["peb_outer"], T=agg.T, case_seed=args.case_seed,
                 n_cl=len(agg.CL_list), n_tcl=len(agg.TCL_list), n_pv=len(agg.PV_list),
                 pmin=agg.pmin, pmax=agg.pmax, method="peb_outer")
    if want_error:
        metadata["directions_sha256"] = hashlib.sha256((output / "directions.npy").read_bytes()).hexdigest()
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {name: {key: value for key, value in old_summary.get(name, {}).items()
                      if (keep_error and key in ERROR_SUMMARY_KEYS)
                      or (keep_cost and key not in ERROR_SUMMARY_KEYS)} for name in regions}
    prepared_costs = None
    if args.metric in ("cost", "both") and args.reuse_costs:
        coefficients = -np.random.default_rng(args.seed).uniform(size=(args.num_cost_samples, args.horizon))
        prepared_costs = reuse_costs(args.reuse_costs, output, metadata, regions, coefficients)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.metric in ("error", "both"):
        errors = {}

        def save_error_result(name, values):
            errors[name] = values
            np.savez(output / f"{name}_errors.npz", **{k: v for k, v in values.items()
                                                       if v is not None})
            summary[name].update({key: values[key] for key in
                                 ("mean_feas", "max_feas", "mean_opt", "max_opt", "count_feas_gt_1e4")})
            metadata["completed_error_methods"].append(name)
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            print(name, "mean feasibility:", values["mean_feas"], "mean optimality:", values["mean_opt"],
                  "feasibility > 1e-4:", values["count_feas_gt_1e4"], flush=True)
        tasks = [(name, args.horizon, args.case_seed, directions, region,
                  args.error_backend, str(output.resolve())) for name, region in regions.items()]
        try:
            if args.workers == 1:
                for task in tasks:
                    save_error_result(*evaluate_method(task))
            else:
                context = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks)), mp_context=context) as executor:
                    futures = [executor.submit(evaluate_method, task) for task in tasks]
                    for future in as_completed(futures):
                        save_error_result(*future.result())
        except BaseException as exc:
            metadata.update(status="failed", error_status="failed", failure=f"{type(exc).__name__}: {exc}")
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
        errors = {name: errors[name] for name in regions}
        with (output / "error_comparison.pkl").open("wb") as stream:
            pickle.dump(errors, stream)
        with (output / "error_samples.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(("method", "sample", "feasibility_squared_distance", "optimality_squared_distance"))
            for name, values in errors.items():
                writer.writerows((name, i, f, o) for i, (f, o) in enumerate(
                    zip(values["error_feas"], values["error_opt"])))
        metadata.update(error_status="completed", error_completed_at_utc=datetime.now(timezone.utc).isoformat())
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.metric in ("cost", "both"):
        try:
            coefficients = -np.random.default_rng(args.seed).uniform(size=(args.num_cost_samples, args.horizon))
            if prepared_costs is None:
                np.save(output / "cost_coefficients.npy", coefficients)
            scale, offset = agg.pmax - agg.pmin, agg.pmin
            if prepared_costs is not None:
                costs = prepared_costs
            else:
                print("Evaluating physical costs: original", flush=True)
                costs = {"original": np.array([solve_cost(c, scale, offset, original=agg.model) for c in coefficients])}
                np.save(output / "original_costs.npy", costs["original"])
                for name, region in regions.items():
                    print(f"Evaluating physical costs: {name}", flush=True)
                    costs[name] = np.array([solve_cost(c, scale, offset, **region) for c in coefficients])
                    np.save(output / f"{name}_costs.npy", costs[name])
            for name in regions:
                difference = costs[name] - costs["original"]
                summary[name].update(mean_cost_difference=float(difference.mean()),
                                     min_cost_difference=float(difference.min()),
                                     max_cost_difference=float(difference.max()))
            if prepared_costs is None:
                with (output / "cost_comparison.pkl").open("wb") as stream:
                    pickle.dump(costs, stream)
                with (output / "cost_samples.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(("method", "sample", "original_cost", "method_cost", "difference"))
                    for name in regions:
                        writer.writerows((name, i, reference, value, value - reference)
                                         for i, (reference, value) in enumerate(zip(costs["original"], costs[name])))
            metadata["cost_coefficients_sha256"] = hashlib.sha256((output / "cost_coefficients.npy").read_bytes()).hexdigest()
            metadata.update(cost_status="completed", cost_completed_at_utc=datetime.now(timezone.utc).isoformat())
        except BaseException as exc:
            metadata.update(status="failed", cost_status="failed", failure=f"{type(exc).__name__}: {exc}")
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metadata["status"] = "completed"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved comparison: {output}")


if __name__ == "__main__":
    main()
