"""T24 method-comparison case: shared-direction errors and real-price market costs."""

# Shared physical scenario, region loading and directional evaluation.

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
from Simulator.testers.polytope_evaluation import build_error_calculator, evaluate_calculator
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
        from Simulator.testers.polytope_evaluation import evaluate_native
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


def evaluate_main(argv=None):
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
    evaluation_sources = ["Simulator/testers/case_methods.py", "Simulator/testers/polytope_evaluation.py"]
    if args.error_backend == "native":
        evaluation_sources.append("Simulator/testers/polytope_evaluation.py")
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




# Saved real-price evaluation and canonical model refresh.


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from Simulator.testers.polytope_evaluation import evaluate_native_costs
from Simulator.paths import CODE_ROOT, RESULTS_ROOT, prepare_comparison_output, resolve_recorded_code_path


BASE = RESULTS_ROOT / "aggregation" / "T24"
PRICE_DATA = CODE_ROOT / "data" / "market_prices" / "caiso_poole_spot"
ERROR_RESULTS = OUTPUT = BASE / "comparisons"
PRICE_AREA_CURRENCIES = {
    "DK1": "EUR", "DE-LU": "EUR", "FR": "EUR",
    "GB-ELEXON": "GBP",
    **{zone: "EUR" for zone in ("FI", "SE1", "SE2", "SE3", "SE4", "NO1", "NO2", "NO3", "NO4", "NO5", "DK2", "EE", "LV", "LT")},
    **{f"AEMO-{zone}": "AUD" for zone in ("SA1", "VIC1", "QLD1", "NSW1", "TAS1")},
    **{f"NYISO-{zone}": "USD" for zone in
       ("WEST", "GENESE", "CENTRL", "NORTH", "MHK-VL", "CAPITL", "HUD-VL",
        "MILLWD", "DUNWOD", "NYC", "LONGIL")},
    **{f"CAISO-{zone}": "USD" for zone in ("NP15", "SP15", "ZP26")},
    **{f"ERCOT-{zone}": "USD" for zone in ("NORTH", "SOUTH", "WEST", "HOUSTON")},
    **{f"ISONE-{zone}": "USD" for zone in
       ("HUB", "ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA")},
    **{f"MISO-{zone}-HUB": "USD" for zone in
       ("ARKANSAS", "ILLINOIS", "INDIANA", "LOUISIANA", "MICHIGAN", "MINN", "MS", "TEXAS")},
    **{f"MISO-{zone}": "USD" for zone in (
        "MDU.ELLNDL_LD", "OTP.MDU", "MEC.HCPD.ABRN", "NSP.KASOTA",
        "ALTW.CMMPA.WIN", "ALTW.WRTHNGT", "ALTW.MRES", "NSP.HCPD.SXFLS")},
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def price_currency(source):
    area = source["price_area"]
    if area == "FI":
        if (source.get("provider") == "ENTSO-E Transparency Platform"
                and source.get("country") == "FI"
                and source.get("area_code") == "BZN|10YFI-1--------U"
                and source.get("public_read_only_endpoint") == "https://transparency.entsoe.eu/market/energyPrices/load"
                and source.get("price_product_type") == "day_ahead_energy"
                and source.get("contract_type") == "A01"
                and source.get("classification_sequence") == "NIL"
                and source.get("reserve_or_imbalance_price") is False
                and source.get("capacity_price") is False
                and source.get("unit") == "EUR/MWh"
                and source.get("currency") == "EUR"):
            return "EUR"
        raise ValueError("FI costs require verified day-ahead electricity spot prices (A01); reserve and imbalance prices are not accepted")
    if (source.get("price_product_type") in ("balancing_energy", "reserve_capacity", "imbalance_settlement")
            or source.get("reserve_or_imbalance_price") is True or source.get("capacity_price") is True):
        raise ValueError("Use electricity spot-market prices, not reserve or imbalance products")
    if (isinstance(source.get("pricing_node"), str)
            and area == "CAISO-" + source["pricing_node"]):
        if (source.get("provider") == "California Independent System Operator (CAISO)"
                and source.get("market_run_id") == "DAM"
                and source.get("query_name") == "PRC_LMP"
                and source.get("price_product_type") == "day_ahead_energy"
                and source.get("price_component") == "LMP"
                and source.get("source_value_field") == "MW"
                and source.get("reserve_or_imbalance_price") is False
                and source.get("capacity_price") is False
                and source.get("unit") == "USD/MWh"
                and source.get("currency") == "USD"):
            return "USD"
        raise ValueError("CAISO node costs require actual DAM electricity LMPs in USD/MWh")
    if area in PRICE_AREA_CURRENCIES:
        return PRICE_AREA_CURRENCIES[area]
    # Additional official MISO pricing nodes are prepared from the same checked
    # raw reports. Keep their exact source node labels instead of inventing
    # state-wide prices or maintaining hundreds of hard-coded node names.
    node = source.get("pricing_node")
    nodes = source.get("nodes", [])
    areas = source.get("price_areas", [])
    if (isinstance(node, str) and node in nodes and len(nodes) == len(areas)
            and area == "MISO-" + node and areas[nodes.index(node)] == area
            and source.get("pricing_node_type") in ("Hub", "Loadzone", "Gennode")
            and source.get("node_types", {}).get(node) == source["pricing_node_type"]
            and source.get("provider") == "Midcontinent Independent System Operator (MISO)"
            and source.get("market") in ("Day Ahead Market ExPost LMPs", "Realtime Market LMPs")
            and source.get("report_url_root") == "https://docs.misoenergy.org/marketreports/"
            and source.get("source_value_field") == "LMP" and source.get("currency") == "USD"):
        return "USD"
    raise ValueError(f"Unsupported price area: {area}")


def read_prices(directory, delta_t, expected_area=None):
    """Require complete 365-by-24 prices and the checksummed source preparation."""
    directory = Path(directory)
    proof = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    if proof.get("schema_version") != 1:
        raise ValueError("Expected price provenance schema_version 1")
    source = proof["source"]
    area = source["price_area"]
    currency = price_currency(source)
    if expected_area is not None and area != expected_area:
        raise ValueError(f"Expected price area {expected_area}, found {area}")
    for key in ("provider", "market", "dataset"):
        if not isinstance(source.get(key), str) or not source[key].strip():
            raise ValueError(f"Price provenance needs source.{key}")
    preparation = proof["preparation"]
    if preparation.get("price_currency", currency) != currency:
        raise ValueError(f"Price currency does not match area {area}: {preparation.get('price_currency')}")
    if source.get("currency", currency) != currency:
        raise ValueError(f"Source currency does not match area {area}: {source.get('currency')}")
    price_field = f"daily_prices_{currency.lower()}_mwh"
    for key, expected in (("original_price_unit", f"{currency}/MWh"),
                          ("hourly_price_unit", f"{currency}/MWh"),
                          ("cost_coefficient_unit", f"{currency}/kWh")):
        if preparation.get(key) != expected:
            raise ValueError(f"Unsupported price unit for {key}: {preparation.get(key)}")
    hashes = proof["files_sha256"]
    if not isinstance(hashes, dict) or "prices.npz" not in hashes:
        raise ValueError("Price provenance must include the prices.npz checksum")
    for filename, expected in hashes.items():
        path = (directory / filename).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f"Unsafe or missing price input: {filename}")
        if sha256(path) != expected:
            raise ValueError(f"Price input differs from its provenance: {filename}")
    window = proof["window"]
    if window.get("scenario_timezone") != "UTC":
        raise ValueError("Daily price scenarios must use UTC")
    boundaries = []
    for key in ("start_utc", "end_utc_exclusive"):
        value = datetime.fromisoformat(window[key].replace("Z", "+00:00"))
        if (value.utcoffset() is None or value.utcoffset().total_seconds() != 0
                or any((value.hour, value.minute, value.second, value.microsecond))):
            raise ValueError(f"Price window {key} must be midnight UTC")
        boundaries.append(value)
    if (boundaries[1] - boundaries[0]).total_seconds() != 365 * 24 * 3600:
        raise ValueError("Price window must contain 365 complete consecutive UTC days")
    expected_dates = np.arange(np.datetime64(boundaries[0].date()),
                               np.datetime64(boundaries[1].date()))
    for key, expected in (("first_utc_delivery_date", str(expected_dates[0])),
                          ("last_utc_delivery_date", str(expected_dates[-1]))):
        if key in window and window[key] != expected:
            raise ValueError(f"Price window date differs: {key}")
    with np.load(directory / "prices.npz", allow_pickle=False) as data:
        dates = data["dates"].copy()
        starts = data["utc_day_starts"].copy()
        ends = data["utc_day_ends"].copy()
        if price_field not in data:
            raise ValueError(f"Price archive is missing its currency-labelled field: {price_field}")
        prices = data[price_field].copy()
        coefficients = data["cost_coefficients"].copy()
    if (prices.shape != (365, 24) or coefficients.shape != (365, 24)
            or not np.all(np.isfinite(prices)) or not np.all(np.isfinite(coefficients)) or delta_t != 1):
        raise ValueError("Expected 365 complete 24-hour price profiles and a one-hour model")
    np.testing.assert_array_equal(dates, expected_dates.astype(str))
    np.testing.assert_array_equal(starts, np.char.add(dates, "T00:00:00Z"))
    np.testing.assert_array_equal(ends, np.char.add((expected_dates + 1).astype(str), "T00:00:00Z"))
    np.testing.assert_array_equal(coefficients, -prices * delta_t / 1000)
    return dates, starts, ends, prices, coefficients, proof


def prepare_error_reuse(source, agg, regions, audit):
    """Verify source errors without relying solely on directory names."""
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    # A failed cost update does not invalidate a completed error metric.
    if metadata.get("error_status", metadata.get("status")) != "completed":
        raise ValueError("The source error evaluation is not completed")
    count = metadata.get("num_error_samples")
    seed = metadata.get("direction_seed")
    if (isinstance(count, bool) or not isinstance(count, int) or count < 1
            or isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("Source error metadata needs an integer direction seed and positive sample count")
    expected = {
        "horizon": 24, "case_seed": 0, "error_backend": "native",
        "aggregator_class": "Simulator.cases.aggregation_case_T24.Aggregator",
        "counts": {"CL": 5, "TCL": 40, "PV": 5, "EV": 0, "ESS": 0},
        "solver_options": SOLVER_OPTIONS, "method_order": list(regions),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Source error protocol differs: {key}")
    for key in ("pmin", "pmax"):
        np.testing.assert_array_equal(metadata[key], getattr(agg, key))
    for filename, expected_hash in metadata["model_sources_sha256"].items():
        if sha256(CODE_ROOT / filename) != expected_hash:
            raise ValueError(f"Physical model source changed: {filename}")
    directions = np.load(source / "directions.npy", allow_pickle=False)
    np.testing.assert_array_equal(directions, np.random.RandomState(seed).normal(size=(count, 24)))
    if sha256(source / "directions.npy") != metadata["directions_sha256"]:
        raise ValueError("Source direction hash differs")
    with np.load(source / "peb_outer.npz", allow_pickle=False) as saved:
        for key in ("A", "b"):
            np.testing.assert_array_equal(saved[key], regions["peb_outer"][key])
    with (source / "error_comparison.pkl").open("rb") as stream:
        errors = pickle.load(stream)
    if list(errors) != list(regions):
        raise ValueError("Source error methods differ")
    if (set(metadata.get("completed_error_methods", ())) != set(regions)
            or len(metadata.get("completed_error_methods", ())) != len(regions)):
        raise ValueError("Source error evaluation has missing or duplicate completed methods")
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    array = np.ascontiguousarray(directions, dtype="<f8")
    digest = hashlib.sha256(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    expected_direction_hash = digest.hexdigest()
    physical_hashes = set()
    for name in regions:
        if name != "peb_outer" and (metadata["scenario_audit"][name]["archive_sha256"]
                                    != audit[name]["archive_sha256"]):
            raise ValueError(f"Saved approximation changed: {name}")
        with np.load(source / f"{name}_errors.npz", allow_pickle=False) as saved:
            if saved["num_sample"].item() != count or errors[name]["num_sample"] != count:
                raise ValueError(f"Source error sample count differs: {name}")
            if (saved["native_successful_solves"].item() != 4 * count
                    or errors[name]["native_successful_solves"] != 4 * count):
                raise ValueError(f"Source error solves are incomplete: {name}")
            if (saved["native_directions_sha256"].item() != expected_direction_hash
                    or errors[name]["native_directions_sha256"] != expected_direction_hash):
                raise ValueError(f"Source error directions differ: {name}")
            for key in ("error_feas", "error_opt"):
                values = np.asarray(errors[name][key])
                if (values.shape != (count,) or not np.all(np.isfinite(values))
                        or np.any(values < 0)):
                    raise ValueError(f"Invalid source errors: {name}/{key}")
                np.testing.assert_array_equal(saved[key], values)
            region_signature = {}
            for key, value in regions[name].items():
                array = np.ascontiguousarray(value, dtype="<f8")
                digest = hashlib.sha256(str(array.shape).encode("ascii"))
                digest.update(array.tobytes())
                region_signature[key] = digest.hexdigest()
            expected_region_hash = hashlib.sha256(json.dumps(
                region_signature, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode("utf-8")).hexdigest()
            if (saved["native_region_sha256"].item() != expected_region_hash
                    or errors[name]["native_region_sha256"] != expected_region_hash):
                raise ValueError(f"Source error approximation hash differs: {name}")
            physical_hash = saved["native_model_sha256"].item()
            if errors[name]["native_model_sha256"] != physical_hash:
                raise ValueError(f"Source error physical model hash differs: {name}")
            physical_hashes.add(physical_hash)
            expected_statistics = {
                "mean_feas": float(np.mean(errors[name]["error_feas"])),
                "max_feas": float(np.max(errors[name]["error_feas"])),
                "mean_opt": float(np.mean(errors[name]["error_opt"])),
                "max_opt": float(np.max(errors[name]["error_opt"])),
                "count_feas_gt_1e4": int(np.count_nonzero(errors[name]["error_feas"] > 1e-4)),
            }
            for key, value in expected_statistics.items():
                for data in (errors[name], saved, summary[name]):
                    if not np.isclose(data[key], value, atol=1e-12, rtol=1e-12):
                        raise ValueError(f"Source error statistic differs: {name}/{key}")
    if len(physical_hashes) != 1:
        raise ValueError("Source error methods used different original physical models")
    return metadata, errors


def compatible_price_crosscheck(source, output, price_proof):
    """Retain an official crosscheck only for the identical area and raw data."""
    source = Path(source).resolve()
    if source != Path(output).resolve():
        return None
    crosscheck = source / "nordpool_crosscheck.json"
    old_provenance = source / "price_provenance.json"
    if not crosscheck.is_file() or not old_provenance.is_file():
        return None
    raw_hash = price_proof.get("files_sha256", {}).get("raw.json")
    if not isinstance(raw_hash, str) or len(raw_hash) != 64:
        return None
    try:
        previous = json.loads(old_provenance.read_text(encoding="utf-8"))
        evidence_bytes = crosscheck.read_bytes()
        evidence = json.loads(evidence_bytes)
    except (ValueError, UnicodeError):
        return None
    area = price_proof.get("source", {}).get("price_area")
    if (previous.get("source", {}).get("price_area") != area
            or evidence.get("comparison", {}).get("area") != area
            or previous.get("files_sha256", {}).get("raw.json") != raw_hash
            or evidence.get("status") != "passed"
            or evidence.get("energinet", {}).get("raw_sha256") != raw_hash):
        return None
    return evidence_bytes


def evaluate_costs(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-data", type=Path, default=PRICE_DATA)
    parser.add_argument("--price-area",
                        help="Require this price area in the input provenance")
    parser.add_argument("--error-results", type=Path, default=ERROR_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    agg = build_aggregator(24, 0)
    dates, starts, ends, prices, coefficients, price_proof = read_prices(
        args.price_data, agg.deltaT, args.price_area)
    price_source = price_proof["source"]
    price_area = price_source["price_area"]
    currency = price_currency(price_source)
    price_field = f"daily_prices_{currency.lower()}_mwh"
    price_filename = f"{price_field}.npy"
    print("Checking the fixed T24 scene and saved approximations", flush=True)
    case = agg.case_aggregator()
    specs = default_methods(BASE)
    audit = {name: verify_scenario(path, agg, 0, required=True) for name, path in specs}
    regions = {name: load_polytope(path) for name, path in specs}
    regions["peb_outer"] = {"A": case["A_hat"], "b": case["b_hat"]}
    audit["peb_outer"] = {"scenario_verified": True, "source": "current Aggregator support LPs"}
    previous, errors = prepare_error_reuse(args.error_results, agg, regions, audit)
    # Read the complete source before cleaning cost-owned output files. Source
    # and output are the same directory by default, including metadata.json.
    source = args.error_results.resolve()
    source_metadata = (source / "metadata.json").read_bytes()
    source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    files = ["directions.npy", "error_comparison.pkl", "error_samples.csv", "peb_outer.npz",
             *[f"{name}_errors.npz" for name in regions]]
    files.extend(filename for name in regions for filename in
                 (f"{name}_checkpoint.npz", f"{name}_progress.json")
                 if (source / filename).is_file())
    source_hashes = {filename: sha256(source / filename) for filename in files}
    price_crosscheck = compatible_price_crosscheck(source, args.output_dir, price_proof)
    output = prepare_comparison_output(args.output_dir, "cost")
    if price_crosscheck is not None:
        (output / "nordpool_crosscheck.json").write_bytes(price_crosscheck)
    if source != output:
        for filename in files:
            shutil.copyfile(source / filename, output / filename)
    for filename, expected_hash in source_hashes.items():
        if sha256(output / filename) != expected_hash:
            raise ValueError(f"Retained error evidence changed: {filename}")
    error_reuse = {
        "source": str(source),
        "source_metadata_sha256": hashlib.sha256(source_metadata).hexdigest(),
        "direction_seed": previous["direction_seed"],
        "num_error_samples": previous["num_error_samples"],
        "original_evaluation_sources_sha256": previous["evaluation_sources_sha256"],
        "bitwise_preserved_files": source_hashes,
    }
    write_json(output / "error_reuse.json", error_reuse)
    (output / "source_error_metadata.json").write_bytes(source_metadata)
    shutil.copyfile(args.price_data / "provenance.json", output / "price_provenance.json")
    np.save(output / price_filename, prices)
    np.save(output / "cost_coefficients.npy", coefficients)
    with (output / "cost_scenarios.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample", "utc_day", "start_utc", "end_utc_exclusive"))
        writer.writerows(zip(range(len(dates)), dates, starts, ends))
    metadata = dict(previous)
    for key in ("failure", "native_cost_diagnostics", "original_mean_cost", "cost_completed_at_utc",
                "publication", "evaluated_price_data_directory", "temporary_exact_source_registration",
                "diagnostic_only", "inactive_published_zero_days", "inactive_published_zero_quarters",
                "source_selection_based_on_observed_results"):
        metadata.pop(key, None)
    now = datetime.now(timezone.utc).isoformat()
    metadata.setdefault("created_at_utc", now)
    metadata.setdefault("error_completed_at_utc", previous.get("completed_at_utc"))
    metadata.update(
        status="running", metric="both", num_samples=previous["num_error_samples"],
        num_error_samples=previous["num_error_samples"], num_cost_samples=len(dates),
        error_status="completed", cost_status="running", last_updated_metric="cost",
        cost_started_at_utc=now,
        method_order=list(regions), scenario_audit=audit,
        methods={name: str(path.resolve()) for name, path in specs},
        solver_options=SOLVER_OPTIONS, error_backend=previous["error_backend"],
        completed_error_methods=list(regions), reused_error_evaluation=error_reuse,
        cost_backend="native",
        cost_source=f"{price_source['market']} {price_area} via {price_source['provider']}",
        price_area=price_area, cost_price_area=price_area,
        price_label=price_source.get("display_name", "Finland FI day-ahead spot"
                                     if price_area == "FI" else price_area),
        price_dataset=price_source["dataset"], cost_dataset=price_source["dataset"],
        cost_currency=currency, cost_unit=f"{currency} per 24 h", deltaT_hours=agg.deltaT,
        cost_coordinates="physical: normalized_power * (pmax-pmin) + pmin",
        power_convention="kW, net generation positive; net consumption negative",
        cost_coefficient_formula=f"-{price_field} * deltaT_hours / 1000",
        cost_objective=f"min sum(cost_coefficients * physical_power), linear energy-price objective in {currency}",
        price_product_type=price_source.get("price_product_type"),
        price_source_value_field=price_source.get("source_value_field"),
        capacity_price=price_source.get("capacity_price"),
        complete_market_invoice_modelled=False,
        price_diagnostic_only=bool(price_proof.get("diagnostic_only", False)),
        cost_sampling="All 365 complete UTC-day profiles in chronological order; no random selection",
        cost_scene_note="The same device, weather and load scenario is used for every price profile.",
        negative_price_handling="Preserved without clipping, absolute values or normalization",
        price_window=price_proof["window"],
        price_data_directory=str(args.price_data.resolve()),
        price_provenance={"path": "price_provenance.json", "sha256": sha256(output / "price_provenance.json")},
        cost_coefficients_sha256=sha256(output / "cost_coefficients.npy"),
        daily_prices_file=price_filename,
        daily_prices_sha256=sha256(output / price_filename),
        cost_evaluation_sources_sha256={name: sha256(CODE_ROOT / name) for name in (
            "Simulator/testers/case_methods.py", "Simulator/testers/polytope_evaluation.py",
            "Simulator/testers/case_methods.py")},
    )
    write_json(output / "metadata.json", metadata)
    try:
        print(f"Solving 365 shared {price_area} real-price profiles for the original model and {len(regions)} methods", flush=True)
        result = evaluate_native_costs(agg, coefficients, regions)
        costs, diagnostics = result["costs"], result["diagnostics"]
        for name in regions:
            if errors[name]["native_model_sha256"] != diagnostics["model_sha256"]:
                raise ValueError(f"Cost and error physical models differ: {name}")
            if errors[name]["native_region_sha256"] != diagnostics["region_sha256"][name]:
                raise ValueError(f"Cost and error regions differ: {name}")
        scale, offset = agg.pmax - agg.pmin, agg.pmin
        for name, values in costs.items():
            np.save(output / f"{name}_costs.npy", values)
            points = result["dispatch_normalized"][name]
            np.savez(output / f"{name}_cost_dispatch.npz", normalized_power=points,
                     physical_power=points * scale + offset, utc_day=dates)
        with (output / "cost_comparison.pkl").open("wb") as stream:
            pickle.dump(costs, stream)
        with (output / "cost_samples.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(("method", "sample", "utc_day", "original_cost", "method_cost", "difference"))
            for name in regions:
                writer.writerows((name, i, dates[i], reference, value, value - reference)
                                 for i, (reference, value) in enumerate(zip(costs["original"], costs[name])))
        summary = {name: dict(source_summary.get(name, {})) for name in regions}
        for name in regions:
            difference = costs[name] - costs["original"]
            summary[name].update({key: errors[name][key] for key in
                                 ("mean_feas", "max_feas", "mean_opt", "max_opt", "count_feas_gt_1e4")})
            summary[name].update(
                mean_cost_difference=float(difference.mean()),
                median_cost_difference=float(np.median(difference)),
                min_cost_difference=float(difference.min()), max_cost_difference=float(difference.max()),
                mean_cost=float(costs[name].mean()), num_cost_samples=len(dates), cost_unit=f"{currency} per 24 h",
            )
        write_json(output / "summary.json", summary)
        write_json(output / "cost_solver_diagnostics.json", diagnostics)
        finished = datetime.now(timezone.utc).isoformat()
        metadata.update(status="completed", cost_status="completed", native_cost_diagnostics=diagnostics,
                        original_mean_cost=float(costs["original"].mean()),
                        cost_completed_at_utc=finished, completed_at_utc=finished)
    except BaseException as exc:
        metadata.update(status="failed", cost_status="failed", failure=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "metadata.json", metadata)
    print(f"Saved real-price comparison: {output}", flush=True)



def refresh_comparison(argv=None, *, price_data=PRICE_DATA, price_area=None):
    """Publish completed formal models and refresh all nine methods, without training."""
    def evaluate_market_costs(arguments=None):
        options = [*(arguments or []), "--price-data", str(price_data)]
        if price_area is not None:
            options += ["--price-area", price_area]
        return evaluate_costs(options)

    import argparse

    from concurrent.futures import ProcessPoolExecutor, as_completed

    import csv

    from datetime import datetime, timezone

    import json

    import multiprocessing

    import os

    from pathlib import Path

    import pickle

    import shutil

    import sys

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    import numpy as np

    from Simulator.paths import CODE_ROOT, RESULTS_ROOT, prepare_comparison_output

    from Simulator.testers.case_training_budgets import check_training_files

    from Simulator.runners.train_aggregation import digest, scenario, write_json

    from Simulator.runners.train_aggregation import build_aggregator


    from Simulator.drawers.plot_comparison import main as plot_comparison

    BASE = RESULTS_ROOT / "aggregation/T24"

    STAGES = [(299, .02, 1.), (199, .1, .01), (499, 1., 1e-6)]

    PAYLOAD = ("final_polytope.npz", "pretrainnet_weights.pth", "training_config.json",
               "training_directions.npy", "error_history.pkl",
               "metadata.pkl", "scenario.json", "projection_retries.json")

    def _refresh_main(argv=None):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--source", type=Path, default=BASE / "exact")
        parser.add_argument("--workers", type=int, default=3)
        args = parser.parse_args(argv)
        if args.workers < 1:
            parser.error("--workers must be positive")
        source, output = args.source.resolve(), (BASE / "comparisons").resolve()
        directions = np.random.RandomState(1).randn(5000, 24)
        np.testing.assert_array_equal(np.load(source / "evaluation_directions.npy"), directions)
        agg = build_aggregator(24, 0)
        case = agg.case_aggregator()
        scene = scenario(agg)
        specs = default_methods(BASE)
        destinations = dict(specs)
        previous = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        report = {"status": "validating", "source": str(source), "training_reused": True,
                  "third_stage_learning_rate": 1., "training_updates_each": 1000,
                  "training_seed": 0, "evaluation_seed": 1, "evaluation_samples": 5000,
                  "previous_summary": json.loads((output / "summary.json").read_text(encoding="utf-8")),
                  "previous_metadata_sha256": digest(output / "metadata.json"), "models": {}}
        # Validate every source and both full training payloads before replacing files.
        errors = {}
        for template in ("peb", "pb"):
            name = f"proposed_{template}_exact"
            directory = source / template
            config, region, _history = check_training_files(directory, STAGES, template)
            expected = {"training_seed": 0, "gradient_mode": "exact", "fourth_stage_used": False,
                        "third_stage_extended": False,
                        "stages": [{"n_train": n, "epochs": n + 1, "lr": lr, "rate_opt_feas": w}
                                   for n, lr, w in STAGES]}
            for key, value in expected.items():
                if config.get(key) != value:
                    raise ValueError(f"Saved training protocol differs: {template}/{key}")
            verify_scenario(directory, agg, 0, required=True)
            np.testing.assert_array_equal(np.load(directory / "training_directions.npy"),
                                          np.random.RandomState(0).randn(1000, 24))
            with np.load(directory / "evaluation_errors.npz", allow_pickle=False) as saved:
                values = {key: saved[key].copy() for key in saved.files}
            from Simulator.testers.polytope_evaluation import _array_hash, _json_hash
            region_hash = _json_hash({key: _array_hash(value) for key, value in region.items()})
            if (values["num_sample"].item() != 5000 or values["native_successful_solves"].item() != 20000
                    or values["native_region_sha256"].item() != region_hash
                    or values["native_directions_sha256"].item() != _array_hash(directions)):
                raise ValueError(f"Saved evaluation does not describe this model/direction set: {template}")
            for key in ("error_feas", "error_opt"):
                if values[key].shape != (5000,) or not np.isfinite(values[key]).all() or np.any(values[key] < 0):
                    raise ValueError(f"Invalid raw evaluation: {template}/{key}")
            # NPZ metadata loads as 0-D arrays; keep the evaluator's scalar contract
            # so the market-cost summary remains directly JSON serializable.
            values = {key: value.item() if value.ndim == 0 else value for key, value in values.items()}
            errors[name] = values
            report["models"][name] = {
                "source": str(directory), "destination": str(destinations[name].resolve()),
                "source_files_sha256": {filename: digest(directory / filename) for filename in PAYLOAD},
                "evaluation_sha256": digest(directory / "evaluation_errors.npz"),
                "previous_polytope_sha256": digest(destinations[name] / "final_polytope.npz"),
            }
        for name, path in specs[2:]:
            verify_scenario(path, agg, 0, required=True)
        # Keep the pre-existing original price data; this refresh does not download or resample it.
        price_files = ["cost_coefficients.npy", "price_provenance.json", "cost_scenarios.csv",
                       *[path.name for path in output.glob("daily_prices_*_mwh.npy")]]
        if (output / "nordpool_crosscheck.json").exists():
            price_files.append("nordpool_crosscheck.json")
        price_data = resolve_recorded_code_path(previous["price_data_directory"])
        price_area = previous["price_area"]
        report["price_files_before"] = {name: digest(output / name) for name in price_files}
        report["baseline_archives_before"] = {name: digest(path / "final_polytope.npz" if path.is_dir() else path)
                                               for name, path in specs[2:]}
        for name, record in report["models"].items():
            directory, destination = Path(record["source"]), Path(record["destination"]).resolve()
            if not destination.is_relative_to((BASE / "exact").resolve()) or destination == directory:
                raise ValueError("Publication target must be a separate existing exact-model directory")
            for filename in PAYLOAD:
                shutil.copyfile(directory / filename, destination / filename)
            # Copy the complete formal training trajectory, including step zero.
            snapshots = destination / "snapshots"
            snapshots.mkdir(exist_ok=True)
            for path in snapshots.glob("*.csv"):
                if path.stem[:2] in ("A_", "b_") and path.stem[2:].isdigit():
                    if not path.resolve().is_relative_to(destination):
                        raise ValueError("Snapshot cleanup escaped the publication directory")
                    path.unlink()
            for path in (directory / "snapshots").glob("*.csv"):
                shutil.copyfile(path, snapshots / path.name)
            initial = {key: np.loadtxt(snapshots / f"{key}_0.csv", delimiter=",") for key in ("A", "b")}
            from Simulator.runners.train_aggregation import common_archive
            np.savez(destination / "initial_polytope.npz", **initial,
                     **common_archive(agg, scene, name + "_initial"), optimizer_updates=0)
            for key in ("A", "b"):
                shutil.copyfile(directory / f"{key}_final.csv", destination / f"{key}_final.csv")
            write_json(destination / "provenance.json", {
                "status": "completed", "scenario_sha256": scene["scenario_sha256"],
                "source": str(directory), "training_reused": True, "training_seed": 0,
                "optimizer_updates": 1000, "actual_stage_updates": [300, 200, 500],
                "third_stage_learning_rate": 1., "fourth_stage_used": False,
                "initial_polytope_optimizer_updates": 0,
                "initial_polytope_note": "Full-training initialization reconstructed from the formal step-zero snapshot",
                "source_files_sha256": record["source_files_sha256"],
                "final_polytope_sha256": digest(destination / "final_polytope.npz")})
            for filename, expected_hash in record["source_files_sha256"].items():
                if digest(destination / filename) != expected_hash:
                    raise RuntimeError(f"Published file differs from its source: {name}/{filename}")
        audit = {name: verify_scenario(path, agg, 0, required=True) for name, path in specs}
        regions = {name: load_polytope(path) for name, path in specs}
        regions["peb_outer"] = {"A": case["A_hat"], "b": case["b_hat"]}
        audit["peb_outer"] = {"scenario_verified": True, "source": "current Aggregator support LPs"}
        # Unchanged baseline regions already have these exact 5000 held-out samples.
        # Validate their region and direction hashes before reusing the raw arrays.
        reused_baselines, baseline_cache_files = [], {}
        for name, region in regions.items():
            saved_path = output / f"{name}_errors.npz"
            if name in errors or not saved_path.is_file():
                continue
            with np.load(saved_path, allow_pickle=False) as saved:
                values = {key: saved[key].item() if saved[key].ndim == 0 else saved[key].copy()
                          for key in saved.files}
            expected_hash = _json_hash({key: _array_hash(value) for key, value in region.items()})
            if (values.get("native_region_sha256") == expected_hash
                    and values.get("native_directions_sha256") == _array_hash(directions)
                    and values.get("native_successful_solves") == 20000
                    and all(np.asarray(values[key]).shape == (5000,)
                            and np.isfinite(values[key]).all() and np.all(values[key] >= 0)
                            for key in ("error_feas", "error_opt"))):
                errors[name] = values
                reused_baselines.append(name)
                checkpoint = output / f"{name}_checkpoint.npz"
                if checkpoint.is_file():
                    with np.load(checkpoint, allow_pickle=False) as saved:
                        manifest = json.loads(saved["manifest"].item())
                        if (saved["completed"].item() != 5000
                                or manifest["region_sha256"] != expected_hash
                                or manifest["directions_sha256"] != _array_hash(directions)):
                            raise ValueError(f"The reusable baseline checkpoint differs: {name}")
                        for key in ("error_feas", "error_opt"):
                            np.testing.assert_array_equal(saved[key], values[key])
                    baseline_cache_files[checkpoint.name] = checkpoint.read_bytes()
                    progress = output / f"{name}_progress.json"
                    if progress.is_file():
                        baseline_cache_files[progress.name] = progress.read_bytes()
        report["unchanged_baseline_evaluations_reused"] = reused_baselines
        prepare_comparison_output(output, "error")
        for filename, payload in baseline_cache_files.items():
            (output / filename).write_bytes(payload)
        np.save(output / "directions.npy", directions)
        np.savez(output / "peb_outer.npz", **regions["peb_outer"], T=24, case_seed=0,
                 n_cl=5, n_tcl=40, n_pv=5, pmin=agg.pmin, pmax=agg.pmax, method="peb_outer")
        metadata = {key: previous[key] for key in ("horizon", "case_seed", "aggregator_class", "counts",
                     "error_definition", "peb_inner_model_note", "legacy_baseline_model_audit") if key in previous}
        metadata.update(status="running", metric="both", error_status="running", cost_status="pending_update",
            direction_seed=1, num_samples=5000, num_error_samples=5000, num_cost_samples=365,
            error_backend="native", error_workers=args.workers, completed_error_methods=[],
            pmin=agg.pmin, pmax=agg.pmax, method_order=list(regions), scenario_audit=audit,
            methods={name: str(path.resolve()) for name, path in specs}, solver_options=SOLVER_OPTIONS,
            model_sources_sha256={name: digest(CODE_ROOT / name) for name in
                                  ("Simulator/cases/aggregation.py", "Simulator/cases/aggregation_case_T24.py")},
            evaluation_sources_sha256={name: digest(CODE_ROOT / name) for name in
                ("Simulator/testers/case_methods.py", "Simulator/testers/polytope_evaluation.py", "Simulator/testers/polytope_evaluation.py")},
            directions_sha256=digest(output / "directions.npy"),
            result_policy="One current comparison; overwrite in place",
            model_update={"training_reused": True, "third_stage_learning_rate": 1.,
                          "source": str(source), "report": "model_update.json"})
        report["status"] = "evaluating"
        write_json(output / "model_update.json", report)
        write_json(output / "metadata.json", metadata)
        try:
            def save_error(name, values):
                errors[name] = values
                np.savez(output / f"{name}_errors.npz", **{k: v for k, v in values.items() if v is not None})
                metadata["completed_error_methods"].append(name)
                write_json(output / "metadata.json", metadata)
                print(f"{name}: max feasibility={float(values['max_feas']):.9g}; "
                      f">1e-4={int(values['count_feas_gt_1e4'])}", flush=True)
            for template in ("peb", "pb"):
                name = f"proposed_{template}_exact"
                shutil.copyfile(source / template / f"{template}_checkpoint.npz", output / f"{name}_checkpoint.npz")
                progress = json.loads((source / template / "evaluation_progress.json").read_text(encoding="utf-8"))
                progress.update(method=name, reused_from=str(source / template), reused_without_new_solves=True)
                write_json(output / f"{name}_progress.json", progress)
                save_error(name, errors[name])
            for name in reused_baselines:
                save_error(name, errors[name])
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
                futures = [executor.submit(evaluate_method, (name, 24, 0, directions, region, "native", str(output)))
                           for name, region in regions.items() if name not in errors]
                for future in as_completed(futures):
                    save_error(*future.result())
            errors = {name: errors[name] for name in regions}
            if len({str(values["native_model_sha256"]) for values in errors.values()}) != 1:
                raise RuntimeError("Reused and newly evaluated results use different physical models")
            with (output / "error_comparison.pkl").open("wb") as stream:
                pickle.dump(errors, stream)
            with (output / "error_samples.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.writer(stream)
                writer.writerow(("method", "sample", "feasibility_squared_distance", "optimality_squared_distance"))
                for name, values in errors.items():
                    writer.writerows((name, i, f, o) for i, (f, o) in enumerate(zip(values["error_feas"], values["error_opt"])))
            summary = {name: {key: values[key] for key in
                       ("mean_feas", "max_feas", "mean_opt", "max_opt", "count_feas_gt_1e4")}
                       for name, values in errors.items()}
            write_json(output / "summary.json", summary)
            metadata.update(error_status="completed", error_completed_at_utc=datetime.now(timezone.utc).isoformat())
            write_json(output / "metadata.json", metadata)
            evaluate_market_costs(["--error-results", str(output), "--output-dir", str(output),
                                   "--price-data", str(price_data), "--price-area", price_area])
            plot_comparison([str(output)])
            for name, record in report["models"].items():
                directory, destination = Path(record["source"]), Path(record["destination"])
                for filename, expected_hash in record["source_files_sha256"].items():
                    if digest(directory / filename) != expected_hash or digest(destination / filename) != expected_hash:
                        raise RuntimeError(f"Original or published training payload changed: {name}/{filename}")
                with np.load(directory / "evaluation_errors.npz", allow_pickle=False) as saved:
                    for key in ("error_feas", "error_opt"):
                        np.testing.assert_array_equal(saved[key], errors[name][key])
            for filename, expected_hash in report["price_files_before"].items():
                if digest(output / filename) != expected_hash:
                    raise RuntimeError(f"Price data changed: {filename}")
            for name, path in specs[2:]:
                if digest(path / "final_polytope.npz" if path.is_dir() else path) != report["baseline_archives_before"][name]:
                    raise RuntimeError(f"Baseline model changed: {name}")
            report.update(status="completed", source_payloads_copied_bitwise=True,
                          source_payloads_unchanged=True, reused_raw_error_arrays_bitwise_equal=True,
                          prices_unchanged=True, baseline_models_unchanged=True,
                          completed_at_utc=datetime.now(timezone.utc).isoformat())
        except BaseException as exc:
            report.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            write_json(output / "model_update.json", report)
    return _refresh_main(argv)


def market_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-data", type=Path, default=PRICE_DATA)
    parser.add_argument("--price-area", help="Require this area in the price provenance")
    parser.add_argument("--error-results", type=Path, default=ERROR_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--refresh-models", action="store_true", help="Reuse formal PEB/PB training and refresh all nine errors, real-price costs and PDF plots")
    parser.add_argument("--source", type=Path, default=BASE / "exact", help="Formal PEB/PB source for --refresh-models")
    parser.add_argument("--workers", type=int, default=3, help="Error-evaluation workers for --refresh-models")
    args = parser.parse_args(argv)
    if args.refresh_models:
        if args.error_results.resolve() != ERROR_RESULTS.resolve() or args.output_dir.resolve() != OUTPUT.resolve():
            parser.error("--refresh-models updates the fixed canonical comparison directory")
        return refresh_comparison(["--source", str(args.source), "--workers", str(args.workers)],
                                  price_data=args.price_data, price_area=args.price_area)
    options = ["--price-data", str(args.price_data), "--error-results", str(args.error_results),
               "--output-dir", str(args.output_dir)]
    if args.price_area is not None:
        options += ["--price-area", args.price_area]
    return evaluate_costs(options)



def main(argv=None):
    """Evaluate saved methods, evaluate market prices, or refresh formal models."""
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = ("evaluate", "market", "refresh")
    if not argv or argv[0] in ("-h", "--help"):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("command", choices=commands, help="Use COMMAND --help for its original options")
        parser.parse_args(argv)
        return 0
    if argv[0] == "evaluate":
        return evaluate_main(argv[1:])
    if argv[0] == "market":
        return market_main(argv[1:])
    if argv[0] == "refresh":
        return market_main(["--refresh-models", *argv[1:]])
    if argv[0].startswith("-"):
        return evaluate_main(argv)
    raise SystemExit("Unknown method-comparison command: " + argv[0])


if __name__ == "__main__":
    raise SystemExit(main())
