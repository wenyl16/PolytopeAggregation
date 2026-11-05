"""Reproducible convex-combination diagnostic of a frozen nominal microgrid.

No trained polytope, relaxed binary model, or perturbed scenario supplies probes.
The reported maximum lower bound is a sampled lower bound, never a global upper
bound on nonconvexity. All percentages use the saved fixed power spans.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from Simulator.cases.microgrid_case import MicroGrid
from Simulator.paths import DATA_ROOT, RESULTS_ROOT, comparison_output_path
from Simulator.testers._nonconvexity_domain import PhysicalDomain
from Simulator.testers._nonconvexity_sampling import (
    build_summary, make_directions, make_probes, unique_points,
)


PROTOCOL = {
    "version": 1, "T": 24, "case_seed": 42, "endpoint_seed": 0,
    "probe_seed": 1, "control_seed": 2, "solver_seed": 0,
    "device_counts": {"MT": 1, "PV": 1, "TL": 1, "CL": 1, "ESS": 1, "HP": 2},
    "conservativeness": 0.0, "uncertainty": False,
    "support_directions": 1000, "formal_probes": 5000, "zero_controls": 100,
    "local_pairs": 500, "global_pairs": 300, "nearest_neighbors": 20,
    "lambdas": [0.1, 0.25, 0.5, 0.75, 0.9],
    "three_point_probes": 500, "five_point_probes": 500, "dirichlet_alpha": 0.5,
    "duplicate_linf_tolerance": 1e-8, "threshold_percent": 0.001,
    "metric": "100 * sqrt(min sum(((P-Q)/(p_max-p_min))**2) / 24)",
    "solver_options": {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-8,
                       "IntFeasTol": 1e-9, "OptimalityTol": 1e-9,
                       "MIPGap": 0.0, "MIPGapAbs": 1e-10},
    "first_time_limit_seconds": 30, "retry_time_limit_seconds": 300,
}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def progress(output, phase, completed, total, **extra):
    state = {"phase": phase, "completed": completed, "total": total,
             "updated_utc": datetime.now(timezone.utc).isoformat(), **extra}
    write_json(output / "progress.json", state)
    print(json.dumps(clean(state), ensure_ascii=True), flush=True)


def effective_counts(device_scale=1, device_counts_override=None):
    if device_counts_override is None:
        return {name: count * device_scale for name, count in PROTOCOL["device_counts"].items()}
    if set(device_counts_override) != set(PROTOCOL["device_counts"]):
        raise ValueError("Specify all six device types: MT, PV, TL, CL, ESS, HP")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1
           for n in device_counts_override.values()):
        raise ValueError("Device counts must be positive integers")
    return dict(device_counts_override)


def expected_model_size(counts, T=24):
    variables = T * (4 * counts["MT"] + counts["PV"] + 2 * counts["TL"]
                     + counts["CL"] + 4 * counts["ESS"] + 2 * counts["HP"] + 1) + counts["ESS"]
    binaries = T * (3 * counts["MT"] + counts["TL"] + counts["ESS"])
    return variables, binaries


def load_case(training_dir, device_scale=1, storage_energy_kwh=None, device_counts_override=None,
              disable_mt_min_uptime=False):
    metadata_path = training_dir / "metadata.pkl"
    with metadata_path.open("rb") as stream:
        saved = pickle.load(stream)
    if int(saved["T"]) != 24:
        raise ValueError("The retained T=24 microgrid is required")
    for name, count in PROTOCOL["device_counts"].items():
        if len(saved[name + "_list"]) != count * device_scale:
            raise ValueError(f"Unexpected {name} count; refusing another case")
    selected = deepcopy(saved)
    counts = effective_counts(device_scale, device_counts_override)
    source_indices = {}
    if device_counts_override is not None:
        for name, count in counts.items():
            originals = saved[name + "_list"]
            source_indices[name] = [i % len(originals) for i in range(count)]
            selected[name + "_list"] = [deepcopy(originals[i]) for i in source_indices[name]]
    changes = []
    if storage_energy_kwh is not None:
        if not np.isfinite(storage_energy_kwh) or storage_energy_kwh <= 0:
            raise ValueError("Storage energy must be positive and finite")
        for i, storage in enumerate(selected["ESS_list"]):
            changes.append({"device": "ESS", "index": i, "parameter": "E_rated",
                            "before": float(storage["E_rated"]), "after": float(storage_energy_kwh)})
            storage["E_rated"] = float(storage_energy_kwh)
    mg = MicroGrid(seed=42, T=24, data=selected)
    mg.build_mg_constraints(conservativeness=0.0)
    removed = []
    if disable_mt_min_uptime:
        for component in list(mg.model.component_objects(pyo.Constraint, active=True)):
            if component.local_name.startswith(("mt_min_uptime_", "mt_initial_uptime_")):
                removed.append({"component": component.local_name, "rows": len(component)})
                component.deactivate()
        if sum(item["rows"] for item in removed) < 24 * counts["MT"]:
            raise ValueError("Expected all MT minimum-uptime constraints to be removed")
    mg.nonconvexity_removed_constraints = removed
    # Changing counts requires the new portfolio's power spans. Energy-only
    # changes preserve the saved spans exactly.
    if device_counts_override is None:
        np.testing.assert_array_equal(mg.pmin, saved["p_min"])
        np.testing.assert_array_equal(mg.pmax, saved["p_max"])
    if not np.all(mg.pmax > mg.pmin):
        raise ValueError("The fixed normalization must have positive spans")
    variables = list(mg.model.component_data_objects(pyo.Var))
    if (len(variables), sum(v.is_binary() for v in variables)) != expected_model_size(counts):
        raise ValueError("The original variable and binary counts must scale with the portfolio")
    mg.nonconvexity_parameter_changes = changes
    mg.nonconvexity_source_indices = source_indices
    return mg


def provenance(training_dir, device_scale=1, storage_energy_kwh=None, device_counts_override=None,
               disable_mt_min_uptime=False):
    source_dir = Path(__file__).resolve().parent
    inputs = [training_dir / "metadata.pkl",
              DATA_ROOT / "profiles_data" / "profiles_data.npz"]
    sources = [Path(__file__), source_dir / "_nonconvexity_domain.py",
               source_dir / "_nonconvexity_sampling.py",
               source_dir.parent / "cases" / "microgrid_case.py"]
    protocol = deepcopy(PROTOCOL)
    protocol.update(device_scale=device_scale, storage_energy_kwh=storage_energy_kwh,
                    device_counts=effective_counts(device_scale, device_counts_override),
                    device_counts_override=device_counts_override)
    if disable_mt_min_uptime:
        protocol["disable_mt_min_uptime"] = True
    result = {"inputs": {str(p.resolve()): digest(p) for p in inputs},
              "sources": {str(p.resolve()): digest(p) for p in sources},
              "protocol": protocol}
    result["fingerprint"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode("utf-8")).hexdigest()
    return result


def require_witness(domain, result, context):
    primal = result.get("primal")
    if primal is None or result.get("point") is None:
        raise RuntimeError(f"No feasible witness for {context}: {clean(result)}")
    validation = domain.validate_primal(primal)
    if not validation["valid"]:
        raise RuntimeError(f"Invalid physical witness for {context}: {validation}")
    return validation


def make_endpoint_pool(domain, mg, output, resume):
    path = output / "endpoints.npz"
    directions, labels = make_directions(seed=0, T=24)
    if resume and path.exists():
        with np.load(path, allow_pickle=False) as data:
            endpoint = {key: data[key] for key in data.files}
        np.testing.assert_array_equal(endpoint["directions"], directions)
        np.testing.assert_array_equal(endpoint["p_min"], mg.pmin)
        np.testing.assert_array_equal(endpoint["p_max"], mg.pmax)
        np.testing.assert_array_equal(endpoint["points"], endpoint["raw_points"][endpoint["unique_indices"]])
        np.testing.assert_array_equal(endpoint["primal"], endpoint["raw_primal"][endpoint["unique_indices"]])
        for i, primal in enumerate(endpoint["raw_primal"]):
            if not domain.validate_primal(primal)["valid"]:
                raise RuntimeError(f"Cached endpoint {i} is invalid")
            np.testing.assert_allclose(endpoint["raw_points"][i], domain.x_from_primal(primal), atol=1e-14, rtol=0)
        return endpoint
    points, primals, diagnostics = [], [], []
    for i, direction in enumerate(directions):
        answer = domain.support(direction)
        validation = require_witness(domain, answer, f"support {i}")
        points.append(answer["point"])
        primals.append(answer["primal"])
        diagnostics.append({"direction_index": i, "direction_kind": str(labels[i]),
                            **answer["diagnostics"], "validation": validation})
        if (i + 1) % 100 == 0:
            progress(output, "endpoint_pool", i + 1, 1000)
    raw_points, raw_primal = np.asarray(points), np.asarray(primals)
    indices = np.asarray(unique_points(raw_points, tol=1e-8), dtype=int)
    if len(indices) < 5:
        raise RuntimeError("Coverage insufficient: fewer than five unique endpoints")
    endpoint = dict(directions=directions, direction_kind=np.asarray(labels),
                    raw_points=raw_points, raw_primal=raw_primal,
                    unique_indices=indices, points=raw_points[indices],
                    primal=raw_primal[indices], p_min=mg.pmin, p_max=mg.pmax,
                    variable_names=np.asarray(domain.names),
                    binary_indices=np.asarray(domain.binary_indices, dtype=int),
                    x_indices=np.asarray(domain.x_indices, dtype=int))
    write_npz(path, **endpoint)
    write_json(output / "endpoint_diagnostics.json", diagnostics)
    return endpoint


def sample_row(index, kind, answer, same_binary_mode=None):
    d = answer["diagnostics"]
    lower = d.get("lower_squared")
    upper = d.get("upper_squared")
    if upper is None or not np.isfinite(upper):
        raise RuntimeError("A verified warm start must supply a finite upper bound")
    if lower is None or not np.isfinite(lower):
        lower = 0.0  # The sum-of-squares objective is nonnegative.
    lower = max(0.0, float(lower))
    upper = max(0.0, float(upper))
    if lower > upper:
        raise RuntimeError(f"Inconsistent projection bounds: {lower} > {upper}")
    lp, up = 100 * np.sqrt(lower / 24), 100 * np.sqrt(upper / 24)
    threshold = PROTOCOL["threshold_percent"]
    classification = ("numerical_zero" if up <= threshold else
                      "detected_gap" if lp > threshold else "undetermined")
    row = {"sample_index": index, "kind": kind, "lower_squared": lower,
           "upper_squared": upper, "lower_percent": lp, "upper_percent": up,
           "classification": classification, "gap_converged": bool(d.get("gap_converged", False)),
           "status_name": d.get("status_name"), "objective": d.get("objective"),
           "best_bound": d.get("best_bound"), "recomputed_objective": d.get("recomputed_objective"),
           "runtime": d.get("runtime"), "wall_seconds": d.get("wall_seconds"),
           "same_binary_mode": same_binary_mode}
    return row


def run_controls(domain, endpoint, output, resume):
    if resume and (output / "controls_verification.json").exists():
        previous = json.loads((output / "controls_verification.json").read_text(encoding="utf-8"))
        if previous.get("passed"):
            with np.load(output / "controls.npz", allow_pickle=False) as cache:
                if len(cache["targets"]) != 100:
                    raise ValueError("Expected 100 cached zero-gap controls")
                for i in range(100):
                    for key in ("feasible_witnesses", "primal"):
                        if not domain.validate_primal(cache[key][i])["valid"]:
                            raise ValueError(f"Invalid cached control {i}: {key}")
                    np.testing.assert_allclose(cache["targets"][i], domain.x_from_primal(cache["feasible_witnesses"][i]), atol=1e-14, rtol=0)
                    np.testing.assert_allclose(cache["points"][i], domain.x_from_primal(cache["primal"][i]), atol=1e-14, rtol=0)
                distances = 100 * np.sqrt(np.mean((cache["targets"] - cache["points"]) ** 2, axis=1))
                if np.max(distances) > PROTOCOL["threshold_percent"]:
                    raise ValueError("Cached zero-gap control is above the declared tolerance")
            return previous
    rng = np.random.default_rng(2)
    anchor = endpoint["primal"][0]
    fixed = np.rint(anchor[domain.binary_indices])
    control_endpoints, targets, witnesses, indices, weights = [], [], [], [], []
    for j in range(20):
        vector = rng.normal(size=24)
        vector /= np.linalg.norm(vector)
        pair = []
        for sign in (1, -1):
            answer = domain.support(sign * vector, start=anchor, fixed_binary=fixed)
            require_witness(domain, answer, f"control endpoint {j}:{sign}")
            if not np.array_equal(np.rint(answer["primal"][domain.binary_indices]), fixed):
                raise RuntimeError("Fixed-mode endpoint changed its binary schedule")
            pair.append(answer["primal"])
            control_endpoints.append(answer["primal"])
        for weight in PROTOCOL["lambdas"]:
            witness = weight * pair[0] + (1 - weight) * pair[1]
            if not domain.validate_primal(witness)["valid"]:
                raise RuntimeError("Known-feasible fixed-mode convex combination failed")
            witnesses.append(witness)
            targets.append(domain.x_from_primal(witness))
            indices.append([2 * j, 2 * j + 1])
            weights.append([weight, 1 - weight])
    rows, diagnostics, points, primals = [], [], [], []
    for i, (target, witness) in enumerate(zip(targets, witnesses)):
        answer = domain.project(target, start=witness)
        require_witness(domain, answer, f"control projection {i}")
        rows.append(sample_row(i, "fixed_mode_control", answer, True))
        diagnostics.append(answer["diagnostics"])
        points.append(answer["point"])
        primals.append(answer["primal"])
    verification = {"passed": all(r["classification"] == "numerical_zero" for r in rows),
                    "count": len(rows), "max_upper_percent": max(r["upper_percent"] for r in rows),
                    "gap_converged_count": sum(r["gap_converged"] for r in rows),
                    "construction": "20 opposite LP endpoint pairs with one fixed feasible binary schedule; five weights per pair",
                    "projection": "Original mixed-integer region; all original binary bounds restored"}
    write_csv(output / "controls.csv", rows)
    write_json(output / "controls_diagnostics.json", diagnostics)
    write_json(output / "controls_verification.json", verification)
    write_npz(output / "controls.npz", targets=np.asarray(targets), points=np.asarray(points),
              primal=np.asarray(primals), feasible_witnesses=np.asarray(witnesses),
              endpoint_primal=np.asarray(control_endpoints), indices=np.asarray(indices),
              weights=np.asarray(weights), fixed_binary=fixed)
    if not verification["passed"]:
        raise RuntimeError("Zero-gap controls failed; formal probes are blocked")
    progress(output, "zero_controls", 100, 100, max_upper_percent=verification["max_upper_percent"])
    return verification


def save_samples(output, rows, diagnostics, points, primals, controls):
    write_csv(output / "samples.csv", rows)
    write_json(output / "projection_diagnostics.json", diagnostics)
    write_npz(output / "projections.npz", points=points, primal=primals)
    summary = build_summary(rows, threshold_percent=PROTOCOL["threshold_percent"])
    summary.update(threshold_percent=PROTOCOL["threshold_percent"], formal_probes_expected=5000,
                   completed=len(rows), controls=controls,
                   interpretation="The maximum lower distance is a sampled lower bound on global nonconvexity. Maximum upper distance only bounds the finite-sample maximum; it is not a global upper bound.",
                   metric=PROTOCOL["metric"])
    write_json(output / "summary.json", summary)
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
    write_csv(output / "statistics.csv", statistics)
    nested = []
    for size in (500, 1000, 2000, 5000):
        if len(rows) >= size:
            prefix = rows[:size]
            nested.append({"sample_count": size,
                           "max_lower_percent": max(r["lower_percent"] for r in prefix),
                           "max_upper_percent": max(r["upper_percent"] for r in prefix),
                           "detected_count": sum(r["classification"] == "detected_gap" for r in prefix)})
    write_csv(output / "sample_count_comparison.csv", nested)
    checkpoint_files = ("samples.csv", "projection_diagnostics.json", "projections.npz", "summary.json",
                        "statistics.csv", "sample_count_comparison.csv")
    write_json(output / "checkpoint.json", {"completed": len(rows),
               "files": {name: digest(output / name) for name in checkpoint_files}})
    return summary


def run(args):
    started = time.perf_counter()
    output = comparison_output_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts_override = getattr(args, "device_counts_override", None)
    counts = effective_counts(args.device_scale, counts_override)
    disable_uptime = getattr(args, "disable_mt_min_uptime", False)
    evidence = provenance(args.training_dir, args.device_scale, args.storage_energy_kwh,
                          counts_override, disable_uptime)
    metadata_path = output / "metadata.json"
    if args.resume:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous["fingerprint"] != evidence["fingerprint"]:
            raise ValueError("Resume refused: source, inputs or protocol changed")
        metadata = previous
    else:
        # Retain a compact original-scenario reference, not another full run.
        if metadata_path.exists() and (output / "summary.json").exists():
            prior = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (prior.get("status") == "completed"
                    and prior.get("storage_energy_kwh") is None
                    and prior.get("device_counts_override") is None
                    and not (output / "baseline_reference.json").exists()):
                write_json(output / "baseline_reference.json", {
                    "metadata": prior,
                    "summary": json.loads((output / "summary.json").read_text(encoding="utf-8")),
                    "note": "Original unchanged-energy scenario retained for before/after comparison; active arrays are overwritten by the new run."})
        for name in ("independent_verification.json", "fresh_solver_recheck.json", "figure_validation.json",
                     "independent_storage_verification.json", "worst_gap_explanation.json", "gap_explanation.json",
                     "storage_scale_comparison.json", "storage_scale_comparison.csv",
                     "reporting_statistics_update.json"):
            owned = output / name
            if owned.exists():
                owned.unlink()
        metadata = {**evidence, "started_utc": datetime.now(timezone.utc).isoformat()}
    metadata["status"] = "running"
    metadata.update(device_scale=args.device_scale, storage_energy_kwh=args.storage_energy_kwh,
                    device_counts_override=counts_override,
                    disable_mt_min_uptime=disable_uptime,
                    training_directory=str(args.training_dir.resolve()))
    metadata["environment"] = {"python": sys.version, "platform": platform.platform(),
                               "packages": {name: version(name) for name in
                                            ("gurobipy", "pyomo", "numpy", "pandas")}}
    write_json(metadata_path, metadata)
    mg = load_case(args.training_dir, args.device_scale, args.storage_energy_kwh,
                   counts_override, disable_uptime)
    domain = PhysicalDomain(mg.model, options=PROTOCOL["solver_options"])
    try:
        if (domain.shape[0], len(domain.binary_indices)) != expected_model_size(counts):
            raise ValueError("Native physical model did not preserve all original variables")
        metadata.update(original_shape=domain.shape, binary_variables=len(domain.binary_indices),
                        p_min=mg.pmin, p_max=mg.pmax, power_span=mg.pmax - mg.pmin,
                        parameter_changes=mg.nonconvexity_parameter_changes,
                        removed_constraints=mg.nonconvexity_removed_constraints,
                        device_source_indices=mg.nonconvexity_source_indices,
                        storage_parameters=mg.ESS_list,
                        normalization_note=("Power spans recomputed from the declared device counts and unchanged per-device parameters."
                                            if counts_override is not None else
                                            "Saved original power spans remain identical after changing storage energy; larger portfolios use their own original aggregate spans."))
        write_json(metadata_path, metadata)
        endpoint = make_endpoint_pool(domain, mg, output, args.resume)
        metadata["unique_endpoints"] = len(endpoint["points"])
        controls = run_controls(domain, endpoint, output, args.resume)
        probes = make_probes(endpoint["points"], seed=1)
        write_npz(output / "probes.npz", **probes)
        rows, diagnostics = [], []
        points = np.full((5000, 24), np.nan)
        primals = np.full((5000, len(domain.names)), np.nan)
        if args.resume and (output / "samples.csv").exists():
            checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
            if any(digest(output / name) != expected for name, expected in checkpoint["files"].items()):
                raise ValueError("Checkpoint files belong to different generations or were modified")
            rows = pd.read_csv(output / "samples.csv").to_dict("records")
            diagnostics = json.loads((output / "projection_diagnostics.json").read_text(encoding="utf-8"))
            with np.load(output / "projections.npz") as cache:
                points, primals = cache["points"], cache["primal"]
            if len(rows) != len(diagnostics):
                raise ValueError("Checkpoint row count mismatch")
            if len(rows) != checkpoint["completed"] or [r["sample_index"] for r in rows] != list(range(len(rows))):
                raise ValueError("Checkpoint sample sequence mismatch")
            if points.shape != (5000, 24) or primals.shape != (5000, len(domain.names)):
                raise ValueError("Checkpoint array shape mismatch")
            for i, row in enumerate(rows):
                if row["kind"] != str(probes["kind"][i]) or not domain.validate_primal(primals[i])["valid"]:
                    raise ValueError(f"Invalid checkpoint witness or probe kind at {i}")
                np.testing.assert_allclose(points[i], domain.x_from_primal(primals[i]), rtol=0, atol=1e-14)
                q = float(np.sum((points[i] - probes["targets"][i]) ** 2))
                np.testing.assert_allclose(q, row["upper_squared"], atol=1e-12, rtol=1e-10)
        limit = 100 if args.pilot else 5000
        binary = np.rint(endpoint["primal"][:, domain.binary_indices]).astype(np.int8)
        for i in range(len(rows), limit):
            idx = probes["indices"][i]
            idx = idx[idx >= 0]
            target = probes["targets"][i]
            mixture = (probes["weights"][i, :len(idx), None] * endpoint["points"][idx]).sum(axis=0)
            np.testing.assert_allclose(target, mixture, rtol=0, atol=2e-14)
            nearest = idx[np.argmin(np.sum((endpoint["points"][idx] - target) ** 2, axis=1))]
            answer = domain.project(target, start=endpoint["primal"][nearest])
            validation = require_witness(domain, answer, f"formal projection {i}")
            same_mode = bool(np.all(binary[idx] == binary[idx[0]]))
            rows.append(sample_row(i, str(probes["kind"][i]), answer, same_mode))
            diagnostics.append({"sample_index": i, **answer["diagnostics"], "validation": validation})
            points[i], primals[i] = answer["point"], answer["primal"]
            if (i + 1) % 100 == 0 or i + 1 == limit:
                save_samples(output, rows, diagnostics, points, primals, controls)
                progress(output, "formal_projections", i + 1, 5000,
                         detected=sum(r["classification"] == "detected_gap" for r in rows),
                         numerical_zero=sum(r["classification"] == "numerical_zero" for r in rows),
                         undetermined=sum(r["classification"] == "undetermined" for r in rows),
                         gap_converged=sum(bool(r["gap_converged"]) for r in rows),
                         max_lower_percent=max(r["lower_percent"] for r in rows),
                         max_upper_percent=max(r["upper_percent"] for r in rows))
            if i == 99:
                pilot = {"passed": True, "samples": 100, "controls_passed": controls["passed"],
                         "all_primal_valid": True, "all_bounds_consistent": True,
                         "gap_converged_count": sum(r["gap_converged"] for r in rows),
                         "mean_wall_seconds": float(np.mean([r["wall_seconds"] for r in rows])),
                         "scope": "Numerical correctness gate, not a requirement to find or exclude nonconvexity"}
                write_json(output / "pilot_verification.json", pilot)
        summary = save_samples(output, rows, diagnostics, points, primals, controls)
        metadata.update(status="completed" if len(rows) == 5000 else "pilot_completed",
                        completed=len(rows), last_run_seconds=time.perf_counter() - started,
                        finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(metadata_path, metadata)
        if len(rows) == 5000 and not args.no_plot:
            # Torch/Gurobi and the plotting stack can load incompatible OpenMP
            # runtimes on Windows. A clean process avoids duplicate-runtime hacks.
            plotter = Path(__file__).resolve().parents[1] / "drawers" / "plot_microgrid_nonconvexity.py"
            subprocess.run([sys.executable, str(plotter), "--input-dir", str(output),
                            "--output", str(args.figure_output)], check=True)
        progress(output, metadata["status"], len(rows), 5000)
        return summary
    except Exception as exc:
        metadata.update(status="failed", failure=f"{type(exc).__name__}: {exc}",
                        last_run_seconds=time.perf_counter() - started)
        write_json(metadata_path, metadata)
        raise
    finally:
        domain.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device-scale", type=int, choices=(1, 3), default=1)
    parser.add_argument("--device-counts", nargs=6, metavar="TYPE=N",
                        help="Repeat frozen source devices to these counts, e.g. MT=3 PV=5 TL=3 CL=5 ESS=3 HP=8")
    parser.add_argument("--storage-energy-kwh", type=float,
                        help="Override only each ESS energy rating; power, efficiencies and SOC limits stay frozen")
    parser.add_argument("--disable-mt-min-uptime", action="store_true",
                        help="Diagnostic ablation only: remove MT minimum-uptime rows; keep all binaries and power limits")
    parser.add_argument("--figure-output", type=Path)
    parser.add_argument("--pilot", action="store_true", help="Run the controls and first 100 formal probes")
    parser.add_argument("--resume", action="store_true", help="Resume only an identical saved protocol and input fingerprint")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    args.device_counts_override = None
    if args.device_counts is not None:
        try:
            pairs = [value.split("=") for value in args.device_counts]
            if any(len(pair) != 2 for pair in pairs) or len({pair[0] for pair in pairs}) != 6:
                raise ValueError("Each device type must occur once as TYPE=N")
            args.device_counts_override = effective_counts(
                args.device_scale, {name: int(count) for name, count in pairs})
        except ValueError as exc:
            parser.error(str(exc))
    base = RESULTS_ROOT / "microgrid"
    if args.device_scale != 1:
        base = base / f"scale{args.device_scale}"
    args.training_dir = args.training_dir or base / "training"
    args.output_dir = args.output_dir or base / (
        "nonconvexity_no_min_uptime" if args.disable_mt_min_uptime else "nonconvexity")
    args.figure_output = args.figure_output or (
        RESULTS_ROOT / "figures_summary" / "07_nonconvexity_diagnostic.pdf" if args.device_scale == 1
        else args.output_dir / "figures" / "nonconvexity_diagnostic.pdf")
    run(args)


if __name__ == "__main__":
    main()
