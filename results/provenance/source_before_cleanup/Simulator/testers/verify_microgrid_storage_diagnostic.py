"""Independently verify a saved portfolio/storage nonconvexity diagnostic.

This entry point does not optimize, retrain, repair, or change saved samples.
It reconstructs the original nominal Pyomo model from the retained training
metadata, optionally repeats each device type to explicit counts, and optionally
changes only each ESS's E_rated. Without --report it writes no files. The
feasibility checks compile the original Pyomo rows independently of the native
optimization bridge used by the diagnostic runner.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from Simulator.cases.microgrid_case import MicroGrid
from Simulator.testers._nonconvexity_sampling import (
    build_summary, make_directions, make_probes, unique_points,
)
from Simulator.testers.verify_microgrid_nonconvexity import physical_validator


COUNTS = {"MT": 1, "PV": 1, "TL": 1, "CL": 1, "ESS": 1, "HP": 2}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _assert_equal(actual, expected, context):
    """Recursively compare metadata without array truth-value ambiguities."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Metadata keys changed: {context}")
        for key in expected:
            _assert_equal(actual[key], expected[key], f"{context}.{key}")
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError(f"Metadata list changed: {context}")
        for index, value in enumerate(expected):
            _assert_equal(actual[index], value, f"{context}[{index}]")
    elif isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected, err_msg=context)
    elif actual != expected:
        raise ValueError(f"Metadata value changed: {context}: {actual!r} != {expected!r}")


def _bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(f"Expected an explicit boolean, received {value!r}")


def _effective_counts(scale, override):
    if override is None:
        return {name: count * scale for name, count in COUNTS.items()}
    if not isinstance(override, dict) or set(override) != set(COUNTS):
        raise ValueError("The count override must specify exactly MT, PV, TL, CL, ESS, HP")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in override.values()):
        raise ValueError("Every custom device count must be a positive integer")
    return dict(override)


def _model_size(counts):
    variables = 24 * (4 * counts["MT"] + counts["PV"] + 2 * counts["TL"] +
                      counts["CL"] + 4 * counts["ESS"] + 2 * counts["HP"] + 1) + counts["ESS"]
    binaries = 24 * (3 * counts["MT"] + counts["TL"] + counts["ESS"])
    return variables, binaries


def _power_spans(data):
    """Independently total the unperturbed physical device power bounds."""
    lower, upper = np.zeros(24), np.zeros(24)
    for device in data["MT_list"]:
        upper += device["P_max"]
    for device in data["PV_list"]:
        upper += device["output_profile"]
    for device in data["TL_list"]:
        lower[device["start_window"]:device["end_window"]] -= device["P_rated"]
    for device in data["CL_list"]:
        lower -= device["p_max_profile"]
        upper -= device["p_max_profile"] * (1 - device["curt_ratio"])
    for device in data["ESS_list"]:
        lower -= device["Pchg"]
        upper += device["Pdis"]
    for device in data["HP_list"]:
        lower -= device["Qmax"] / device["COP"]
    return lower, upper


def rebuild_case(training_dir, scale, energy, device_counts_override=None):
    """Rebuild from frozen physical parameters, independent of runner.load_case."""
    with (training_dir / "metadata.pkl").open("rb") as stream:
        saved = pickle.load(stream)
    if saved["T"] != 24 or saved["deltaT"] != 1.0:
        raise ValueError("Expected the retained T=24, one-hour microgrid")
    for name, count in COUNTS.items():
        devices = saved[name + "_list"]
        if len(devices) != count * scale:
            raise ValueError(f"Unexpected {name} device count")
        for index, device in enumerate(devices):
            _assert_equal(device, devices[index % count], f"repeated {name}[{index}]")
    counts = _effective_counts(scale, device_counts_override)
    if device_counts_override is not None and scale != 1:
        raise ValueError("Custom counts must use the original seven-device source (device_scale=1)")
    original_snapshot = deepcopy(saved)
    portfolio = deepcopy(saved)
    if device_counts_override is not None:
        for name, count in counts.items():
            originals = saved[name + "_list"]
            portfolio[name + "_list"] = [deepcopy(originals[index % len(originals)])
                                         for index in range(count)]
    changed = deepcopy(portfolio)
    changes = []
    for index, ess in enumerate(changed["ESS_list"]):
        if ess["Pchg"] != 100.0 or ess["Pdis"] != 100.0:
            raise ValueError("Storage diagnostic must retain 100 kW charging/discharging")
        if portfolio["ESS_list"][index]["E_rated"] != 250.0:
            raise ValueError("The original source storage energy must remain 250 kWh")
        if energy is not None:
            if not np.isfinite(energy) or energy <= 0:
                raise ValueError("Storage energy override must be finite and positive")
            changes.append({"device": "ESS", "index": index, "parameter": "E_rated",
                            "before": float(ess["E_rated"]), "after": float(energy)})
            ess["E_rated"] = float(energy)
    unchanged_view = deepcopy(changed)
    for index, ess in enumerate(unchanged_view["ESS_list"]):
        ess["E_rated"] = portfolio["ESS_list"][index]["E_rated"]
    _assert_equal(unchanged_view, portfolio, "only_E_rated_may_change")
    mg = MicroGrid(seed=42, T=24, data=changed)
    mg.build_mg_constraints(conservativeness=0.0)
    _assert_equal(saved, original_snapshot, "training_metadata_not_mutated")
    for name in COUNTS:
        _assert_equal(getattr(mg, name + "_list"), changed[name + "_list"], name)
    expected_lower, expected_upper = _power_spans(portfolio)
    np.testing.assert_array_equal(mg.pmin, expected_lower)
    np.testing.assert_array_equal(mg.pmax, expected_upper)
    if device_counts_override is None:
        np.testing.assert_array_equal(mg.pmin, saved["p_min"])
        np.testing.assert_array_equal(mg.pmax, saved["p_max"])
    if not np.all(mg.pmax > mg.pmin):
        raise ValueError("The declared portfolio must have positive fixed power spans")
    variables = list(mg.model.component_data_objects(pyo.Var))
    if (len(variables), sum(v.is_binary() for v in variables)) != _model_size(counts):
        raise ValueError("Rebuilt model did not preserve all original variables/binaries")
    return mg, saved, changes


def _check_rows(rows, diagnostics, points, targets, threshold, binary_count):
    if len(rows) != len(diagnostics):
        raise ValueError("Projection diagnostics and sample rows have different lengths")
    np.testing.assert_array_equal(rows.sample_index.to_numpy(), np.arange(len(rows)))
    lower = rows.lower_squared.to_numpy(dtype=float)
    upper = rows.upper_squared.to_numpy(dtype=float)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Nonfinite distance bounds")
    if np.any(lower < 0) or np.any(upper < 0) or np.any(lower > upper):
        raise ValueError("Invalid squared-distance interval")
    q = np.sum((points - targets) ** 2, axis=1)
    np.testing.assert_allclose(q, upper, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(100 * np.sqrt(lower / 24), rows.lower_percent,
                               rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(100 * np.sqrt(q / 24), rows.upper_percent,
                               rtol=1e-10, atol=1e-12)
    classifications = np.where(rows.upper_percent <= threshold, "numerical_zero",
                               np.where(rows.lower_percent > threshold, "detected_gap", "undetermined"))
    np.testing.assert_array_equal(classifications, rows.classification)
    for index, (row, diagnostic) in enumerate(zip(rows.to_dict("records"), diagnostics)):
        if diagnostic.get("query") != "projection" or diagnostic.get("integer_relaxation") is not False:
            raise ValueError(f"Projection {index} was not declared an original integer query")
        if diagnostic.get("base_binary_variables") != binary_count:
            raise ValueError(f"Projection {index} has a different number of physical binaries")
        if diagnostic.get("bound_inconsistent"):
            raise ValueError(f"Projection {index} reported inconsistent solver bounds")
        np.testing.assert_allclose(diagnostic["lower_squared"], row["lower_squared"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(diagnostic["upper_squared"], row["upper_squared"], rtol=1e-10, atol=1e-12)
        if _bool(row["gap_converged"]) != _bool(diagnostic["gap_converged"]):
            raise ValueError(f"Convergence flag changed for projection {index}")
        if _bool(row["gap_converged"]):
            tolerance = float(diagnostic["absolute_gap_tolerance"])
            if upper[index] - lower[index] > tolerance + 1e-12:
                raise ValueError(f"Projection {index} claims convergence outside its tolerance")
    return q


def _reporting_rows(rows, threshold):
    """Reclassify detached records; retain all stored numerical rows verbatim."""
    records = rows.to_dict("records")
    for row in records:
        row["classification"] = ("detected_gap" if row["lower_percent"] > threshold
                                 else "below_threshold" if row["upper_percent"] <= threshold
                                 else "undetermined")
    return records


def _check_summary(saved, expected):
    def check_groups(actual_groups, expected_groups, context):
        if set(actual_groups) != set(expected_groups):
            raise ValueError(f"Saved summary groups differ: {context}")
        for name, group in expected_groups.items():
            actual = actual_groups[name]
            for key in ("count", "classification_counts", "detected_nonconvex_count"):
                _assert_equal(actual[key], group[key], f"{context}.{name}.{key}")
            for side in ("lower", "upper"):
                for statistic in ("mean", "median", "p95", "max"):
                    value = group[side + "_percent"][statistic]
                    stored = actual[side + "_percent"][statistic]
                    if value is None:
                        if stored is not None:
                            raise ValueError(f"Unexpected statistic for empty group: {context}.{name}")
                    else:
                        np.testing.assert_allclose(stored, value, rtol=1e-10, atol=1e-12)
    check_groups(saved["groups"], expected["groups"], "overall")
    if set(saved["prefixes"]) != set(expected["prefixes"]):
        raise ValueError("Saved nested-prefix sizes differ")
    for size, prefix in expected["prefixes"].items():
        for key in ("requested_count", "available_count", "complete"):
            _assert_equal(saved["prefixes"][size][key], prefix[key], f"prefix{size}.{key}")
        check_groups(saved["prefixes"][size]["groups"], prefix["groups"], f"prefix{size}")


def _check_forward_mt_schedules(primals, names, devices, context, tolerance=1e-9,
                                enforce_min_uptime=True):
    """Check saved MT runs from transitions, without consulting model rows.

    This oracle looks FORWARD from each observed 0->1 transition. It never
    reconstructs or evaluates the model's backward rolling-startup inequality.
    The ending is truncated to visible periods, and initially-online devices
    follow the declared conservative requirement to remain on for min(U,T).
    """
    values = np.asarray(primals, dtype=float)
    if (values.ndim != 2 or values.shape[1] != len(names) or not len(values)
            or not np.isfinite(values).all()):
        raise ValueError(f"Missing or invalid complete primal rows for {context}")
    position = {str(name): index for index, name in enumerate(names)}
    if len(position) != len(names):
        raise ValueError(f"Duplicate physical-variable names for {context}")
    T = 24
    startups = truncated = initial_windows = 0
    maximum_binary_deviation = 0.0
    for device_index, device in enumerate(devices):
        uptime, initial = device["min_on_time"], device["initial_status"]
        if isinstance(uptime, (bool, np.bool_)) or not isinstance(uptime, (int, np.integer)) or uptime < 1:
            raise ValueError(f"Invalid original MT minimum uptime in {context}")
        if initial not in (0, 1):
            raise ValueError(f"Invalid original MT initial status in {context}")
        sequences = {}
        for field in ("u_MT", "y_MT", "z_MT"):
            required = [f"{field}[{device_index},{t}]" for t in range(T)]
            missing = [name for name in required if name not in position]
            if missing:
                raise ValueError(f"Missing MT trajectory variables in {context}: {missing}")
            raw = values[:, [position[name] for name in required]]
            rounded = np.rint(raw)
            deviation = float(np.max(np.abs(raw - rounded)))
            maximum_binary_deviation = max(maximum_binary_deviation, deviation)
            if deviation > tolerance or np.any((rounded < 0) | (rounded > 1)):
                raise ValueError(f"Nonbinary {field} trajectory in {context}, MT {device_index}")
            sequences[field] = rounded.astype(np.int8)
        online = sequences["u_MT"]
        previous = np.column_stack((np.full(len(values), initial, dtype=np.int8), online[:, :-1]))
        observed_start = (online == 1) & (previous == 0)
        observed_stop = (online == 0) & (previous == 1)
        for field, expected in (("y_MT", observed_start), ("z_MT", observed_stop)):
            wrong = np.argwhere(sequences[field] != expected)
            if len(wrong):
                sample, t = (int(index) for index in wrong[0])
                raise ValueError(f"Saved {field} differs from the state transition in {context}, "
                                 f"row {sample}, MT {device_index}, period {t + 1}")
        if initial and enforce_min_uptime:
            initial_windows += len(values)
            wrong = np.argwhere(online[:, :min(int(uptime), T)] != 1)
            if len(wrong):
                sample, t = (int(index) for index in wrong[0])
                raise ValueError(f"Conservative initial-online requirement failed in {context}, "
                                 f"row {sample}, MT {device_index}, period {t + 1}")
        for start in range(T):
            rows = np.flatnonzero(observed_start[:, start])
            startups += len(rows)
            end = min(start + int(uptime), T) if enforce_min_uptime else start + 1
            if enforce_min_uptime and start + uptime > T:
                truncated += len(rows)
            wrong = np.argwhere(online[rows, start:end] != 1)
            if len(wrong):
                relative_row, offset = (int(index) for index in wrong[0])
                raise ValueError(f"Independent forward uptime check failed in {context}, "
                                 f"row {int(rows[relative_row])}, MT {device_index}: "
                                 f"startup period {start + 1}, offline period {start + offset + 1}, "
                                 f"required uptime {uptime}")
    return {"passed": True, "primal_rows": len(values), "MT_devices": len(devices),
            "minimum_uptime_enforced": enforce_min_uptime,
            "MT_trajectories_checked": len(values) * len(devices),
            "observed_startups_checked": int(startups),
            "terminal_truncated_startups_checked": int(truncated),
            "conservative_initial_online_windows_checked": int(initial_windows),
            "startup_shutdown_states_match_observed_transitions": True,
            "maximum_binary_deviation": maximum_binary_deviation,
            "violations": 0}


def verify(input_dir, training_dir):
    input_dir, training_dir = Path(input_dir).resolve(), Path(training_dir).resolve()
    meta = _json(input_dir / "metadata.json")
    scale = meta["device_scale"]
    energy = None if meta["storage_energy_kwh"] is None else float(meta["storage_energy_kwh"])
    counts_override = meta.get("device_counts_override")
    if isinstance(scale, bool) or not isinstance(scale, int) or scale not in (1, 3):
        raise ValueError("Expected device_scale 1 or 3")
    if energy is not None and (not np.isfinite(energy) or energy <= 0):
        raise ValueError("storage_energy_kwh must be finite and positive")
    if Path(meta["training_directory"]).resolve() != training_dir:
        raise ValueError("The declared training_directory differs from --training-dir")
    protocol = meta["protocol"]
    disable_uptime = _bool(meta.get("disable_mt_min_uptime", False))
    if _bool(protocol.get("disable_mt_min_uptime", False)) != disable_uptime:
        raise ValueError("Minimum-uptime ablation disagrees with the recorded protocol")
    expected_counts = _effective_counts(scale, counts_override)
    variable_count, binary_count = _model_size(expected_counts)
    if protocol["device_counts"] != expected_counts:
        raise ValueError("Protocol device counts differ from the requested original portfolio")
    _assert_equal(protocol.get("device_counts_override"), counts_override,
                  "protocol.device_counts_override")
    if protocol["conservativeness"] != 0.0 or protocol["uncertainty"] is not False:
        raise ValueError("This verification requires the unshrunk nominal physical region")
    for key, value in (("T", 24), ("case_seed", 42), ("endpoint_seed", 0), ("probe_seed", 1),
                       ("support_directions", 1000), ("formal_probes", 5000), ("zero_controls", 100)):
        if protocol[key] != value:
            raise ValueError(f"Unexpected protocol value for {key}")
    if protocol.get("device_scale", scale) != scale or protocol.get("storage_energy_kwh", energy) != energy:
        raise ValueError("Protocol scale/storage energy disagrees with metadata")
    threshold = float(protocol["threshold_percent"])
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid classification threshold")
    reporting_override = "reporting_threshold_percent" in meta
    reporting_threshold = float(meta.get("reporting_threshold_percent", threshold))
    if not np.isfinite(reporting_threshold) or reporting_threshold < 0:
        raise ValueError("Invalid reporting threshold")
    if "numerical_threshold_percent" in meta and meta["numerical_threshold_percent"] != threshold:
        raise ValueError("Metadata numerical threshold differs from the solver protocol")
    recorded = {Path(path).resolve(): value for group in ("inputs", "sources")
                for path, value in meta[group].items()}
    if recorded.get(training_dir / "metadata.pkl") != _digest(training_dir / "metadata.pkl"):
        raise ValueError("Training parameters do not match the saved input hash")
    for path, expected in recorded.items():
        if _digest(path) != expected:
            raise ValueError(f"Input/source hash changed: {path}")
    checkpoint = _json(input_dir / "checkpoint.json")
    if checkpoint["completed"] != 5000 or meta.get("completed") != 5000:
        raise ValueError("Verification requires the complete 5000-probe experiment")
    for name, expected in checkpoint["files"].items():
        if _digest(input_dir / name) != expected:
            raise ValueError(f"Checkpoint file changed: {name}")
    mg, original, changes = rebuild_case(training_dir, scale, energy, counts_override)
    if disable_uptime:
        removed = []
        for i in mg.model.MTs:
            component = mg.model.component(f"mt_min_uptime_{i}")
            removed.append({"component": component.local_name, "rows": len(component)})
            component.deactivate()
            for t in range(24):
                component = mg.model.component(f"mt_initial_uptime_{i}_{t}")
                if component is not None:
                    removed.append({"component": component.local_name, "rows": len(component)})
                    component.deactivate()
        _assert_equal(meta["removed_constraints"], removed, "removed_minimum_uptime_rows")
    _assert_equal(meta["parameter_changes"], changes, "declared_parameter_changes")
    expected_source_indices = {} if counts_override is None else {
        name: [index % len(original[name + "_list"]) for index in range(count)]
        for name, count in expected_counts.items()}
    if counts_override is not None or "device_source_indices" in meta:
        _assert_equal(meta["device_source_indices"], expected_source_indices,
                      "declared_device_source_indices")
    endpoints = _arrays(input_dir / "endpoints.npz")
    probes = _arrays(input_dir / "probes.npz")
    projected = _arrays(input_dir / "projections.npz")
    controls = _arrays(input_dir / "controls.npz")
    rows = pd.read_csv(input_dir / "samples.csv")
    control_rows = pd.read_csv(input_dir / "controls.csv")
    if len(rows) != 5000 or len(control_rows) != 100 or len(endpoints["raw_points"]) != 1000:
        raise ValueError("Expected 1000 raw endpoints, 100 controls, and 5000 formal projections")
    names = endpoints["variable_names"].tolist()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate saved physical-variable names")
    variable_map = {v.name: v for v in mg.model.component_data_objects(pyo.Var)}
    expected_binary = np.asarray([i for i, name in enumerate(names) if variable_map[name].is_binary()])
    expected_x = np.asarray([names.index(mg.model.var_proj[t].name) for t in range(24)])
    np.testing.assert_array_equal(endpoints["binary_indices"], expected_binary)
    np.testing.assert_array_equal(endpoints["x_indices"], expected_x)
    for key, expected in (("p_min", mg.pmin), ("p_max", mg.pmax)):
        np.testing.assert_array_equal(endpoints[key], expected)
        if key in meta:
            np.testing.assert_array_equal(meta[key], expected)
    np.testing.assert_array_equal(endpoints["directions"], make_directions(seed=0, T=24)[0])
    np.testing.assert_array_equal(endpoints["unique_indices"], unique_points(endpoints["raw_points"], tol=1e-8))
    np.testing.assert_array_equal(endpoints["points"], endpoints["raw_points"][endpoints["unique_indices"]])
    np.testing.assert_array_equal(endpoints["primal"], endpoints["raw_primal"][endpoints["unique_indices"]])
    np.testing.assert_array_equal(endpoints["raw_points"], endpoints["raw_primal"][:, expected_x])
    np.testing.assert_array_equal(projected["points"], projected["primal"][:, expected_x])
    if projected["points"].shape != (5000, 24) or projected["primal"].shape != (5000, len(names)):
        raise ValueError("Unexpected formal-projection arrays")
    validate = physical_validator(mg.model, names)
    validations = {"raw_endpoints": validate(endpoints["raw_primal"]),
                   "formal_projections": validate(projected["primal"]),
                   "control_endpoints": validate(controls["endpoint_primal"]),
                   "control_feasible_witnesses": validate(controls["feasible_witnesses"]),
                   "control_projections": validate(controls["primal"])}
    primal_groups = {"raw_endpoints": (endpoints["raw_primal"], 1000),
                     "control_endpoints": (controls["endpoint_primal"], 40),
                     "control_feasible_witnesses": (controls["feasible_witnesses"], 100),
                     "control_projections": (controls["primal"], 100),
                     "formal_projections": (projected["primal"], 5000)}
    uptime_checks = {}
    for group_name, (primals, expected_rows) in primal_groups.items():
        if len(primals) != expected_rows:
            raise ValueError(f"Missing original primal rows in {group_name}")
        uptime_checks[group_name] = _check_forward_mt_schedules(
            primals, names, mg.MT_list, group_name, enforce_min_uptime=not disable_uptime)
    regenerated = make_probes(endpoints["points"], seed=1)
    for key in ("targets", "indices", "weights", "kind", "pair_id", "lambda", "sample_index"):
        np.testing.assert_array_equal(probes[key], regenerated[key])
    if np.any(probes["weights"] < 0):
        raise ValueError("Negative mixture weights")
    np.testing.assert_allclose(probes["weights"].sum(axis=1), 1, rtol=0, atol=1e-14)
    mixture = np.sum(probes["weights"][:, :, None] *
                     endpoints["points"][np.maximum(probes["indices"], 0)], axis=1)
    np.testing.assert_allclose(probes["targets"], mixture, atol=2e-14, rtol=0)
    np.testing.assert_array_equal(rows.kind, probes["kind"])
    binary = np.rint(endpoints["primal"][:, expected_binary]).astype(np.int8)
    same_mode = [bool(np.all(binary[index[index >= 0]] == binary[index[index >= 0]][0]))
                 for index in probes["indices"]]
    np.testing.assert_array_equal([_bool(x) for x in rows.same_binary_mode], same_mode)
    np.testing.assert_allclose(controls["targets"], controls["feasible_witnesses"][:, expected_x], atol=1e-14, rtol=0)
    np.testing.assert_array_equal(controls["points"], controls["primal"][:, expected_x])
    witness = np.sum(controls["weights"][:, :, None] *
                     controls["endpoint_primal"][controls["indices"]], axis=1)
    np.testing.assert_allclose(witness, controls["feasible_witnesses"], atol=1e-12, rtol=0)
    fixed = controls["fixed_binary"]
    np.testing.assert_array_equal(np.rint(controls["endpoint_primal"][:, expected_binary]),
                                  np.broadcast_to(fixed, (len(controls["endpoint_primal"]), len(fixed))))
    diagnostics = _json(input_dir / "projection_diagnostics.json")
    q = _check_rows(rows, diagnostics, projected["points"], probes["targets"], threshold, binary_count)
    cq = _check_rows(control_rows, _json(input_dir / "controls_diagnostics.json"),
                     controls["points"], controls["targets"], threshold, binary_count)
    if np.max(100 * np.sqrt(cq / 24)) > threshold:
        raise ValueError("Known-feasible fixed-mode control exceeds the threshold")
    numerical_summary = build_summary(rows.to_dict("records"), threshold_percent=threshold)
    reporting_summary = build_summary(_reporting_rows(rows, reporting_threshold),
                                      threshold_percent=reporting_threshold)
    # Legacy runs have only numerical classifications. Explicit reporting
    # overrides use separate labels and never overwrite samples.csv.
    summary = reporting_summary if reporting_override else numerical_summary
    saved_summary = _json(input_dir / "summary.json")
    if saved_summary["threshold_percent"] != (reporting_threshold if reporting_override else threshold):
        raise ValueError("Saved summary threshold differs from its metadata declaration")
    for key, expected in (("numerical_threshold_percent", threshold),
                          ("reporting_threshold_percent", reporting_threshold)):
        if reporting_override or key in saved_summary:
            if saved_summary.get(key) != expected:
                raise ValueError(f"Saved summary has inconsistent {key}")
    _check_summary(saved_summary, summary)
    endpoint_diagnostics = _json(input_dir / "endpoint_diagnostics.json")
    if len(endpoint_diagnostics) != 1000 or any(d.get("integer_relaxation") is not False or
            d.get("base_binary_variables") != binary_count for d in endpoint_diagnostics):
        raise ValueError("Support diagnostics do not preserve the original integer model")
    for path, expected in recorded.items():
        if _digest(path) != expected:
            raise ValueError(f"Input/source changed during verification: {path}")
    return {"passed": True, "completed": 5000, "device_scale": scale,
            "disable_mt_min_uptime": disable_uptime,
            "device_counts": expected_counts, "storage_energy_kwh": energy,
            "device_counts_override": counts_override,
            "device_source_indices": expected_source_indices,
            "parameter_changes": changes,
            "only_storage_energy_changed": counts_override is None and energy is not None,
            "only_device_counts_changed": counts_override is not None and energy is None,
            "all_per_device_parameters_unchanged": energy is None,
            "source_storage_energy_kwh": 250.0,
            "fixed_original_normalization_unchanged": counts_override is None,
            "fixed_portfolio_power_normalization_verified": True,
            "normalization_reference": ("Original retained training power spans" if counts_override is None
                                        else "Independent power sums of the declared counts with unchanged source parameters"),
            "physical_validation": validations, "original_variables": len(names),
            "expected_variables": variable_count, "original_binary_variables": binary_count,
            "independent_forward_MT_uptime_validation": {
                "passed": True, "groups": uptime_checks,
                "minimum_uptime_enforced": not disable_uptime,
                "total_MT_trajectories_checked": sum(
                    group["MT_trajectories_checked"] for group in uptime_checks.values()),
                "total_observed_startups_checked": sum(
                    group["observed_startups_checked"] for group in uptime_checks.values()),
                "validation_basis": ("Observed MT startup/shutdown states only; minimum uptime explicitly removed"
                                     if disable_uptime else
                                     "Forward checking from observed u state transitions, independent of Pyomo rolling-uptime expressions"),
                "initial_online_semantics": ("No minimum-uptime requirement" if disable_uptime else
                                             "First min(U,T) periods remain online"),
                "terminal_semantics": ("No minimum-uptime requirement" if disable_uptime else
                                       "Only the visible portion of each startup run is required")},
            "reproducible_directions": True, "reproducible_probes": True,
            "mixture_membership_verified": True, "distance_fields_verified": True,
            "numerical_threshold_percent": threshold,
            "reporting_threshold_percent": reporting_threshold,
            "stored_sample_classifications_use_numerical_threshold": True,
            "numerical_classification_counts": numerical_summary["groups"]["overall"]["classification_counts"],
            "reporting_classification_counts": reporting_summary["groups"]["overall"]["classification_counts"],
            "classification_counts": summary["groups"]["overall"]["classification_counts"],
            "max_lower_percent": float(rows.lower_percent.max()),
            "max_upper_percent": float(100 * np.sqrt(np.max(q) / 24)),
            "max_control_percent": float(100 * np.sqrt(np.max(cq) / 24)),
            "recorded_input_source_hashes_match": True,
            "validation_basis": "Original nominal Pyomo expressions; no optimizer or training used",
            "scope": "Finite convex-combination probes, not a global upper bound or proof of convexity",
            "caveat": "Feasibility and solver bounds use declared numerical tolerances, not exact arithmetic"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Optionally write the independent JSON report")
    args = parser.parse_args(argv)
    result = verify(args.input_dir, args.training_dir)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.report is not None:
        args.report.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
