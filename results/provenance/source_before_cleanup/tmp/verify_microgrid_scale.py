"""Independent checks for the retained full Microgrid scale experiment.

Run with the project Python from any working directory. --skip-solves checks
all serialized data without a Gurobi license; the default additionally repeats
the extreme feasibility/optimality subproblems through the Pyomo path.
"""

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn
import torch

from Simulator.cases.microgrid_case import MicroGrid
from Simulator.runners.train_microgrid import build_microgrid


DEVICE_NAMES = ("MT", "PV", "TL", "CL", "ESS", "HP")
def device_counts(scale):
    return dict(zip(DEVICE_NAMES, (scale, scale, scale, scale, scale, 2 * scale)))


def physical_complexity(model):
    variables = list(model.component_data_objects(pyo.Var))
    return {"variables": len(variables),
            "binary_variables": sum(v.is_binary() for v in variables),
            "constraints": sum(1 for _ in model.component_data_objects(pyo.Constraint, active=True))}


def native_layout(physical):
    model = physical.model
    shape = physical_complexity(model)
    ranges = sum(c.has_lb() and c.has_ub() and not c.equality
                 for c in model.component_data_objects(pyo.Constraint, active=True))
    original_integers = sum(v.is_binary() or v.is_integer()
                            for v in model.component_data_objects(pyo.Var))
    counts = len(model.T) * sum(bool(len(getattr(model, name))) for name in ("MTs", "TLs", "ESSs"))
    headroom = 2 * len(model.ESSs) * (len(model.T) - 1)
    return {"original_variables": shape["variables"], "original_binary_variables": shape["binary_variables"],
            "original_integer_variables": original_integers, "original_constraints": shape["constraints"],
            "range_slack_variables": ranges, "native_original_variables": shape["variables"] + ranges,
            "native_original_constraints": shape["constraints"], "added_counts": counts,
            "added_headroom": headroom, "native_variables": shape["variables"] + ranges + counts,
            "native_integer_variables": original_integers + counts,
            "native_constraints": shape["constraints"] + counts + headroom}
METRICS = ("feasibility_error", "feasibility_error_rated",
           "optimality_error_rated", "objective_error", "objective_error_rated")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def close(actual, expected, *, atol=1e-10, rtol=1e-10, name="value"):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol,
                               equal_nan=True, err_msg=name)


def equal_tree(actual, expected, path=""):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            equal_tree(actual[key], expected[key], path + "." + str(key))
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected), path
        for i, (left, right) in enumerate(zip(actual, expected)):
            equal_tree(left, right, path + f"[{i}]")
    else:
        assert actual == expected, (path, actual, expected)


def mutable_ids(value):
    if isinstance(value, dict):
        return {id(value)}.union(*(mutable_ids(v) for v in value.values()))
    if isinstance(value, (tuple, list)):
        own = {id(value)} if isinstance(value, list) else set()
        return own.union(*(mutable_ids(v) for v in value))
    return {id(value)} if isinstance(value, np.ndarray) else set()


def portfolio_checks(metadata, config, scale):
    counts = device_counts(scale)
    assert config["device_scale"] == scale
    base = build_microgrid(config["case_seed"], 1)
    scaled = build_microgrid(config["case_seed"], scale)
    assert base.T == scaled.T == metadata["T"] == 24
    seen = set()
    checked_devices = 0
    for kind in DEVICE_NAMES:
        original = getattr(base, kind + "_list")
        actual = getattr(scaled, kind + "_list")
        saved = metadata[kind + "_list"]
        assert len(actual) == len(saved) == counts[kind]
        for i, device in enumerate(actual):
            equal_tree(device, original[i % len(original)], f"factory.{kind}[{i}]")
            equal_tree(saved[i], device, f"metadata.{kind}[{i}]")
            ids = mutable_ids(device)
            assert not seen.intersection(ids), f"Mutable data shared across devices: {kind}[{i}]"
            seen.update(ids)
            checked_devices += 1
        # The saved pickle must also preserve separation between devices.
        seen_saved = set()
        for device in saved:
            ids = mutable_ids(device)
            assert not seen_saved.intersection(ids), f"Saved devices alias: {kind}"
            seen_saved.update(ids)
    assert checked_devices == sum(counts.values())
    assert {d["type"] for d in scaled.TL_list} == {d["type"] for d in base.TL_list}
    for bad in (0, -1, True, 1.5):
        try:
            build_microgrid(42, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid scale accepted: {bad!r}")
    # Physical construction requires no optimization or license.
    base.build_mg_constraints()
    scaled.build_mg_constraints()
    for field in ("pmin", "pmax", "baseline"):
        close(getattr(scaled, field), scale * getattr(base, field), name=field)
    close(metadata["p_min"], scaled.pmin, name="saved p_min")
    close(metadata["p_max"], scaled.pmax, name="saved p_max")
    close(metadata["baseline"], scaled.baseline, name="saved baseline")
    assert np.all(scaled.pmax > scaled.pmin)
    actual_complexity, base_complexity = physical_complexity(scaled.model), physical_complexity(base.model)
    # Only the shared aggregate power variables and balance rows are not copied.
    expected_complexity = {
        "variables": base.T + scale * (base_complexity["variables"] - base.T),
        "binary_variables": scale * base_complexity["binary_variables"],
        "constraints": base.T + scale * (base_complexity["constraints"] - base.T)}
    assert actual_complexity == config["model_complexity"] == expected_complexity
    return {"device_scale": scale, "device_counts": counts, "devices_checked": checked_devices,
            "TL_types": sorted({d["type"] for d in scaled.TL_list}), "independent_deep_copies": True,
            "physical_bounds_and_baseline_scaled": True,
            "model_complexity": actual_complexity, "native_layout": native_layout(scaled)}, scaled


def storage_proof_checks(physical, metadata):
    """Check the original algebra that implies both storage headroom cuts.

    For u=1 discharge is zero, and e_t <= SOC_max gives the charge cut.
    For u=0 charge is zero, and e_t >= SOC_min gives the discharge cut.
    The opposite mode's cut follows from nonnegativity and e_(t-1) bounds.
    No equation links period-zero power to SOC; period zero must be omitted.
    """
    model = physical.model
    dt = float(metadata["deltaT"])
    assert dt == physical.deltaT > 0

    def linear_coefficients(constraint):
        rep = generate_standard_repn(constraint.body)
        assert rep.is_linear()
        return {v.name: float(c) for v, c in zip(rep.linear_vars, rep.linear_coefs)}, float(rep.constant)

    checked = 0
    parameters = []
    for i, data in enumerate(metadata["ESS_list"]):
        energy, eta_c, eta_d = (float(data[k]) for k in ("E_rated", "eta_chg", "eta_dis"))
        soc_min, soc_max = float(data["SOC_min"]), float(data["SOC_max"])
        assert np.isfinite([energy, eta_c, eta_d, soc_min, soc_max]).all()
        assert energy > 0 and eta_c > 0 and eta_d > 0 and soc_min <= soc_max
        dynamics = getattr(model, f"ess_energy_dynamics_{i}")
        assert 0 not in dynamics
        for t in model.T:
            u, charge, discharge = model.u_ESS[i, t], model.p_chg_ESS[i, t], model.p_dis_ESS[i, t]
            assert u.is_binary() and charge.lb == discharge.lb == 0
            soc_bound = getattr(model, f"ess_soc_limit_{i}")[t]
            assert pyo.value(soc_bound.lower) == soc_min and pyo.value(soc_bound.upper) == soc_max
            charge_bound = getattr(model, f"ess_charging_power_limit_{i}")[t]
            discharge_bound = getattr(model, f"ess_discharging_power_limit_{i}")[t]
            coeff, constant = linear_coefficients(charge_bound)
            assert coeff.keys() == {charge.name, u.name}
            close([coeff[charge.name], coeff[u.name], constant, pyo.value(charge_bound.upper)],
                  [1., -data["Pchg"], 0., 0.])
            coeff, constant = linear_coefficients(discharge_bound)
            assert coeff.keys() == {discharge.name, u.name}
            close([coeff[discharge.name], coeff[u.name], constant, pyo.value(discharge_bound.upper)],
                  [1., data["Pdis"], -data["Pdis"], 0.])
            if t == 0:
                continue
            coefficients, constant = linear_coefficients(dynamics[t])
            expected = {model.e_ESS[i, t].name: 1., model.e_ESS[i, t - 1].name: -1.,
                        charge.name: -dt * eta_c / energy,
                        discharge.name: dt / (energy * eta_d)}
            assert coefficients.keys() == expected.keys()
            close([coefficients[name] for name in expected], list(expected.values()))
            close([constant, pyo.value(dynamics[t].lower), pyo.value(dynamics[t].upper)], [0., 0., 0.])
            checked += 2
        parameters.append({"E_rated": energy, "eta_chg": eta_c, "eta_dis": eta_d,
                           "SOC_min": soc_min, "SOC_max": soc_max})
    assert checked == 2 * len(metadata["ESS_list"]) * (len(model.T) - 1)
    return {"status": "passed", "implied_headroom_inequalities": checked,
            "deltaT": dt, "storage_parameters": parameters, "period_zero_omitted": True,
            "derivation": "Binary charging mode removes the opposite flow; original SOC dynamics and previous/current SOC bounds imply each headroom inequality"}


def runner_factory_facts(source):
    """Check the current factory without accepting changed physical imports."""
    tree = ast.parse(source)
    factory = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == "build_microgrid")
    counts = next(n for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "DEVICE_COUNTS" for t in n.targets))
    imports = ast.Module(body=[n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))],
                         type_ignores=[])
    return {name: hashlib.sha256(ast.dump(node, include_attributes=False).encode("utf-8")).hexdigest()
            for name, node in (("build_microgrid", factory), ("DEVICE_COUNTS", counts), ("imports", imports))}


def shared_runner_used_facts(source):
    tree = ast.parse(source)
    used = {node.name: node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in ("train_case", "new_run_dir")}
    assert set(used) == {"train_case", "new_run_dir"}
    used["imports"] = ast.Module(body=[n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))],
                                 type_ignores=[])
    # The observed two-stage extension changes only aggregate build_stages/main.
    # Keep other statements strict, including globals used by the shared loop.
    used["top_level_excluding_unused_aggregate_schedule"] = ast.Module(
        body=[n for n in tree.body if not (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                                            and n.name in ("build_stages", "main"))], type_ignores=[])
    return {name: hashlib.sha256(ast.dump(node, include_attributes=False).encode("utf-8")).hexdigest()
            for name, node in used.items()}


def training_source_checks(training_dir, config):
    """Keep immutable execution hashes even if an external CLI edit occurs.

    Current physics/algorithm files must still match the launch-time hashes.
    The two observed runner edits may use exact executed-source snapshots when
    their current reviewed hashes and used factory/function/import AST agree. This
    records a real working-tree change; it does not relabel the changed file
    as the code loaded by the running training process.
    """
    result_dir = Path(training_dir).parent
    code = ROOT / "code"
    runner = code / "Simulator/runners/train_microgrid.py"
    shared_runner = code / "Simulator/runners/train_aggregation.py"
    strict_core = {code / name for name in (
        "Simulator/cases/microgrid_case.py", "Simulator/Approximator.py", "Simulator/microgrid_count_solver.py")}
    recorded = {Path(name): expected for name, expected in config["source_sha256"].items()}
    assert {runner, shared_runner, *strict_core}.issubset(recorded), "Missing recorded training source hashes"
    provenance_path = result_dir / "source_provenance.json"
    provenance = read_json(provenance_path) if provenance_path.is_file() else None
    entries, changed, factory_facts, shared_facts = [], [], None, None
    for path, expected in recorded.items():
        current_hash = digest(path)
        snapshot = result_dir / "source_snapshot" / path.relative_to(code)
        snapshot_hash = digest(snapshot) if snapshot.is_file() else None
        snapshot_verified = snapshot_hash == expected
        if path not in (runner, shared_runner):
            assert current_hash == expected, f"Current physical/core source changed: {path}"
        elif current_hash != expected:
            assert snapshot_verified, "Changed runner requires an exact recorded executed-source snapshot"
            assert provenance and provenance["executed_source_snapshot_verified"] is True
            assert provenance["training_config_unmodified"] is True
            archived_record = next(row for row in provenance["files"] if Path(row["source_path"]) == path)
            assert archived_record["recorded_executed_sha256"] == archived_record["snapshot_sha256"] == expected
            assert current_hash == archived_record["current_source_sha256"], "Unreviewed current runner edit"
            if path == runner:
                factory_facts = runner_factory_facts(path.read_text(encoding="utf-8-sig"))
                assert factory_facts == runner_factory_facts(snapshot.read_text(encoding="utf-8-sig")), (
                    "Changed runner factory, DEVICE_COUNTS or imports cannot be used for independent validation")
            else:
                shared_facts = shared_runner_used_facts(path.read_text(encoding="utf-8-sig"))
                assert shared_facts == shared_runner_used_facts(snapshot.read_text(encoding="utf-8-sig")), (
                    "Changed train_case/new_run_dir/imports or other shared globals cannot be accepted")
        if snapshot_hash is not None:
            assert snapshot_verified, f"Executed-source snapshot differs from recorded SHA: {snapshot}"
        if current_hash != expected:
            changed.append(str(path))
        entries.append({"source_path": str(path), "recorded_executed_sha256": expected,
                        "current_source_sha256": current_hash, "current_source_changed": current_hash != expected,
                        "snapshot_path": str(snapshot) if snapshot.is_file() else None,
                        "snapshot_sha256": snapshot_hash, "snapshot_verified": snapshot_verified,
                        "executed_source_verified": current_hash == expected or snapshot_verified})
    return {"status": "passed", "executed_source_verified": True,
            "physics_and_core_current_sources_match": True, "current_source_changed": bool(changed),
            "current_changed_source_count": len(changed),
            "strict_current_source_paths": sorted(map(str, strict_core)),
            "changed_current_source_paths": changed, "files": entries,
            "all_recorded_sources_have_exact_snapshots": all(row["snapshot_verified"] for row in entries),
            "current_runner_factory_AST_unchanged": True,
            "runner_factory_AST": factory_facts,
            "used_shared_functions_unchanged": True,
            "shared_runner_used_AST": shared_facts,
            "provenance_path": str(provenance_path) if provenance_path.is_file() else None}


def training_checks(training_dir, config, scale, source_audit=None):
    assert config["status"] == "completed"
    assert config["device_scale"] == scale
    assert config["device_counts"] == device_counts(scale)
    assert config["portfolio_definition"] == "repeat_original_devices"
    assert config["gradient_mode"] == "approx" and config["is_uncertain"]
    assert config["case_seed"] == config["training_seed"] == 42
    assert config["num_samples"] == 100 and config["batch_size"] == 5
    assert config["noise_scale"] == 0.05
    assert config["solver_options"]["Threads"] == 8
    assert config["solver_options"]["Seed"] == 0
    assert config["solver_options"]["MIPFocus"] == 2
    assert config["solver_options"]["Symmetry"] == 2
    assert config["solver_options"]["Presolve"] == 0
    assert config["count_branching"] is True
    assert config["storage_headroom"] is True
    assert config["physical_solver_backend"] == "gurobi_persistent_integer_counts"
    source_audit = source_audit or training_source_checks(training_dir, config)
    assert digest(ROOT / "code/data/profiles_data/profiles_data.npz") == config["profiles_sha256"]
    assert config["optimizer_updates"] == 2000
    assert config["phase_end_updates"] == [600, 1200, 2000]
    stages = config["stages"]
    assert [s["n_train"] for s in stages] == [29, 29, 39]
    assert [s["epochs"] for s in stages] == [30, 30, 40]
    close([s["lr"] for s in stages], [.05, .1, .2], atol=0, rtol=0)
    close([s["rate_opt_feas"] for s in stages], [1, .01, 1e-6], atol=0, rtol=0)
    calls = config["train_calls"]
    assert len(calls) == 3
    assert [c["optimizer_updates"] for c in calls] == [600, 600, 800]
    assert [c["epochs"] for c in calls] == [30, 30, 40]
    phase_seconds = [c["elapsed_seconds"] for c in calls]
    assert all(math.isfinite(t) and t > 0 for t in phase_seconds)
    assert 0 <= config["elapsed_seconds"] - sum(phase_seconds) < 1
    assert config["setup_seconds"] > 0
    setup = read_json(training_dir / "setup_timing.json")
    close(setup["setup_seconds"], config["setup_seconds"], atol=0, rtol=0)
    assert setup["includes_initial_outer_supports"]
    assert setup["model_complexity"] == config["model_complexity"]
    with (training_dir / "error_history.pkl").open("rb") as stream:
        history = pickle.load(stream)
    for name in ("feasibility_errors", "optimality_errors", "objective_history", "surrogate_history"):
        array = np.asarray(history[name])
        assert len(array) == 2000 and np.isfinite(array).all(), name
    assert np.min(history["feasibility_errors"]) >= 0
    assert np.min(history["optimality_errors"]) >= 0
    surrogate = np.asarray(history["surrogate_history"])
    assert surrogate.shape == (2000, 2)
    weights = np.repeat([1., .01, 1e-6], [600, 600, 800])
    close(history["objective_history"], surrogate[:, 0] + weights * surrogate[:, 1],
          rtol=2e-6, atol=2e-7, name="stored weighted approximate objective")
    state = torch.load(training_dir / "pretrainnet_weights.pth", map_location="cpu", weights_only=True)
    raw_b = state["b_net.0.bias"]
    raw_A = state["A_net.0.bias"].reshape(len(raw_b), 24)
    assert tuple(raw_A.shape) == (96, 24)
    assert state["A_net.0.weight"].numel() == state["b_net.0.weight"].numel() == 0
    # Reconstruct directly, without using the production PreTrainNet/evaluator.
    norms = torch.norm(raw_A.unsqueeze(0), dim=2, keepdim=True)
    A = (raw_A.unsqueeze(0) / (norms + 1e-8))[0].numpy()
    b = (raw_b / (norms.squeeze(2) + 1e-8))[0].numpy()
    assert np.isfinite(A).all() and np.isfinite(b).all()
    with np.load(training_dir / "final_polytope.npz", allow_pickle=False) as final:
        np.testing.assert_array_equal(A, final["A"])
        np.testing.assert_array_equal(b, final["b"])
    close(np.loadtxt(training_dir / "snapshots" / "A_2000.csv", delimiter=","), A, atol=0, rtol=0)
    close(np.loadtxt(training_dir / "snapshots" / "b_2000.csv", delimiter=","), b, atol=0, rtol=0)
    assert np.loadtxt(training_dir / "snapshots" / "A_0.csv", delimiter=",").shape == (96, 24)
    return {"optimizer_updates": 2000, "phase_end_updates": [600, 1200, 2000],
            "training_scenario_directions_inferred_from_schedule": 10000,
            "training_seconds": config["elapsed_seconds"], "phase_seconds": phase_seconds,
            "setup_seconds": config["setup_seconds"], "weights_npz_snapshot_equal": True,
            "history_and_weight_schedule_consistent": True,
            "physical_solver_options": config["solver_options"],
            "physical_solver_backend": config["physical_solver_backend"],
            "source_audit": source_audit}, A, b


def evaluation_checks(output_dir, metadata, physical, scale):
    config = read_json(output_dir / "metadata.json")
    summary = read_json(output_dir / "summary.json")
    with (output_dir / "results.pkl").open("rb") as stream:
        rows = pickle.load(stream)
    table = pd.read_csv(output_dir / "errors.csv", keep_default_na=False)
    assert config["status"] == "completed"
    assert config["num_samples"] == config["recorded_samples"] == len(rows) == len(table) == 5000
    assert config["completed_samples"] == 5000 and config["failed_samples"] == 0
    assert config["seed"] == 1 and config["noise_scale"] == .05
    assert [config[name] for name in ("pv_shift", "cl_shift", "temp_shift")] == [0., 0., 0.]
    assert config["device_counts"] == device_counts(scale)
    assert config["T"] == 24 and config["final_polytope_checked"]
    assert config["backend"] == "gurobi_persistent_integer_counts"
    assert config["count_branching"] is True
    assert config["storage_headroom"] is True
    assert config["physical_integer_constraints"] == "unrelaxed"
    for name, expected in {"Threads": 8, "Seed": 0, "Presolve": 0,
                           "MIPFocus": 2, "Symmetry": 2}.items():
        assert config["solver_options"][name] == expected, name
    assert config["training_sources_unchanged"]
    for path, expected in config["training_source_sha256"].items():
        assert digest(path) == expected, path
    assert digest(ROOT / "code/Simulator/cases/microgrid_case.py") == config["case_source_sha256"]
    assert digest(ROOT / "code/Simulator/testers/evaluate_microgrid_disaggregation.py") == config["evaluator_source_sha256"]
    assert digest(ROOT / "code/Simulator/microgrid_count_solver.py") == config["count_solver_source_sha256"]
    directions = -np.random.RandomState(1).rand(5000, 24)
    generator = torch.Generator(device="cpu").manual_seed(1)
    pv_base = torch.tensor(np.array([d["output_profile"] for d in metadata["PV_list"]]), dtype=torch.float32)
    cl_base = torch.tensor(np.array([d["p_max_profile"] for d in metadata["CL_list"]]), dtype=torch.float32)
    temp_base = torch.tensor(physical.temp_amb, dtype=torch.float32)
    nominal = {"p_PV_rated": pv_base.numpy().astype(float),
               "p_CL_rated": cl_base.numpy().astype(float),
               "temp_amb": np.array(physical.temp_amb)}
    # Original parameter arrays retain float64 baselines, not their float32
    # sampling copies. Use these originals when adding float32 noise.
    nominal["p_PV_rated"] = np.array([d["output_profile"] for d in metadata["PV_list"]])
    nominal["p_CL_rated"] = np.array([d["p_max_profile"] for d in metadata["CL_list"]])
    pmin, width = np.asarray(metadata["p_min"]), np.asarray(metadata["p_max"] - metadata["p_min"])
    maxima = {name: 0. for name in METRICS}
    sample_times = []
    for i, row in enumerate(rows):
        assert row["sample_index"] == i == int(table.iloc[i]["sample_index"])
        assert row["solver_status"] == table.iloc[i]["solver_status"] == "completed"
        assert not row["failure"] and not table.iloc[i]["failure"]
        np.testing.assert_array_equal(row["direction"], directions[i])
        delta = {
            "p_PV_rated": torch.randn(pv_base.shape, generator=generator) * .05 * pv_base,
            "p_CL_rated": torch.randn(cl_base.shape, generator=generator) * .05 * cl_base,
            "temp_amb": torch.randn(temp_base.shape, generator=generator) * .05 * temp_base,
        }
        assert row["perturbed_parameters"].keys() == nominal.keys()
        for name in nominal:
            np.testing.assert_array_equal(row["perturbed_parameters"][name],
                                          nominal[name] + delta[name].numpy(),
                                          err_msg=f"sample {i} {name} perturbation")
        curves = {name: np.asarray(row[name], dtype=float) for name in (
            "power_curve", "projected_power_curve", "nominal_power_curve",
            "approx_projected_nominal_power_curve")}
        assert all(c.shape == (24,) and np.isfinite(c).all() for c in curves.values())
        p, q, z, w = (curves[k] for k in curves)
        cost, cost0 = directions[i] @ p, directions[i] @ z
        recalculated = {
            "feasibility_error": float(np.sum((p - q) ** 2)),
            "feasibility_error_rated": float(np.sum(((p - pmin) / width - (q - pmin) / width) ** 2)),
            "optimality_error_rated": float(np.sum(((z - pmin) / width - (w - pmin) / width) ** 2)),
            "objective_error": float(cost - cost0),
            "objective_error_rated": float((cost - cost0) / abs(cost0)) if cost0 else np.nan,
        }
        for name in METRICS:
            close(row[name], float(table.iloc[i][name]), rtol=2e-12, atol=2e-12,
                  name=f"CSV/pickle {name} sample {i}")
            close(row[name], recalculated[name], rtol=2e-9, atol=2e-10,
                  name=f"trajectory-derived {name} sample {i}")
            maxima[name] = max(maxima[name], abs(float(row[name]) - recalculated[name]))
        assert row["elapsed_seconds"] > 0
        sample_times.append(row["elapsed_seconds"])
    for key, expected in summary.items():
        if key != "metrics":
            equal_tree(config[key], expected, "summary." + key)
    assert config["solver_status_counts"] == dict(Counter(r["solver_status"] for r in rows))
    for name in METRICS:
        values = np.array([r[name] for r in rows], dtype=float)
        finite = values[np.isfinite(values)]
        expected = {"finite_count": len(finite), "missing_or_nonfinite_count": len(values) - len(finite),
                    "mean": np.mean(finite), "median": np.median(finite),
                    "p95": np.percentile(finite, 95), "max": np.max(finite)}
        for key, value in expected.items():
            close(summary["metrics"][name][key], value, rtol=1e-13, atol=1e-13,
                  name=f"summary {name}.{key}")
    assert sum(sample_times) <= config["evaluation_elapsed_seconds"] + .1
    return {"samples": 5000, "direction_seed": 1, "noise_seed": 1,
            "directions_and_perturbations_exact": True, "CSV_pickle_summary_consistent": True,
            "trajectory_metric_max_absolute_differences": maxima,
            "evaluation_seconds": config["evaluation_elapsed_seconds"],
            "sample_time_mean_seconds": float(np.mean(sample_times)),
            "sample_time_p95_seconds": float(np.percentile(sample_times, 95)),
            "sample_time_max_seconds": float(np.max(sample_times))}, rows, config


def count_backend_checks(directory, expected_successful_solves, metadata, layout):
    """Audit the exact redundant extension, not just the backend's label.

    For each original integral solution K is uniquely the sum of its binary
    statuses and lies in [0,n]. Conversely, deleting K from any extended
    solution leaves all original constraints/variables intact. The original
    device-level feasible set, aggregate projection and objective all agree.
    This proof does not require device homogeneity.
    """
    data = read_json(directory / "solver_diagnostics.json")
    diagnostics, solves = data["diagnostics"], data["solves"]
    assert diagnostics["backend"] == "gurobi_persistent_redundant_integer_count_branching"
    assert diagnostics["fresh_native_model_per_solve"] is True
    assert diagnostics["native_model_closed_after_solve"] is True
    assert diagnostics["physical_integer_constraints"] == "unrelaxed"
    assert diagnostics["count_branch_priority"] == 100
    assert diagnostics["original_branch_priority"] == 0
    assert diagnostics["storage_headroom"] is True
    assert diagnostics["storage_headroom_t0_skipped"] is True
    assert diagnostics["storage_parameters_verified_against_original_model"] is True
    assert isinstance(diagnostics["storage_headroom_proof"], str) and diagnostics["storage_headroom_proof"]
    assert diagnostics["deltaT"] == metadata["deltaT"]
    storage_keys = ("E_rated", "eta_chg", "eta_dis", "SOC_min", "SOC_max", "Pchg", "Pdis")
    expected_storage = [{key: float(device[key]) for key in storage_keys}
                        for device in metadata["ESS_list"]]
    equal_tree(diagnostics["storage_parameters"], expected_storage, "native storage parameters")
    assert diagnostics["solve_count"] == len(solves)
    assert diagnostics["optimal_solves"] == expected_successful_solves
    assert diagnostics["exception_count"] == 0
    assert len(solves) == expected_successful_solves + diagnostics["nonoptimal_solves"]
    default_tolerances = {"MIPGap": 1e-4, "MIPGapAbs": 1e-10,
                          "FeasibilityTol": 1e-6, "IntFeasTol": 1e-5, "OptimalityTol": 1e-6}
    assert diagnostics["actual_tolerances"] == default_tolerances
    status_counts = Counter()
    native_shapes = Counter()
    max_equality_residual = max_integrality_residual = max_gap = 0.
    max_binary_residual = max_constraint_violation = max_bound_violation = 0.
    max_storage_headroom_violation = 0.
    for i, row in enumerate(solves):
        assert row["solve_index"] == i + 1
        assert row["original_variables"] == layout["original_variables"]
        assert row["original_binary_variables"] == layout["original_binary_variables"]
        assert row["original_constraints"] == layout["original_constraints"]
        assert row["native_original_integer_variables"] == layout["original_integer_variables"]
        assert row["native_original_binary_variables"] == layout["original_binary_variables"]
        assert row["native_original_variables"] == layout["native_original_variables"]
        assert row["native_original_constraints"] == layout["native_original_constraints"]
        assert row["storage_headroom"] is True
        assert row["storage_headroom_t0_skipped"] is True
        assert row["added_storage_headroom_cuts"] == layout["added_headroom"]
        assert row["added_integer_counts"] == row["added_count_equalities"] == layout["added_counts"]
        assert row["native_variables"] - row["native_original_variables"] == layout["added_counts"]
        assert row["native_integer_variables"] == layout["native_integer_variables"]
        assert row["native_binary_variables"] == row["native_original_binary_variables"]
        assert row["native_constraints"] - row["native_original_constraints"] == layout["added_counts"] + layout["added_headroom"]
        assert row["native_variables"] == layout["native_variables"] and row["native_constraints"] == layout["native_constraints"]
        assert row["native_original_quadratic_constraints"] == 0
        assert row["native_original_quadratic_objective_terms"] in (0, 24)
        for key, value in {"presolve": 0, "threads": 8, "seed": 0, "mip_focus": 2}.items():
            assert row[key] == value, (i, key)
        if "symmetry" in row:
            assert row["symmetry"] == 2
        if "actual_tolerances" in row:
            assert row["actual_tolerances"] == default_tolerances
        if "native_model_closed" in row:
            assert row["native_model_closed"] is True
        assert row["wall_seconds"] >= row["optimizer_runtime_seconds"] - .1
        assert row["build_seconds"] >= 0
        status_counts[row["termination_condition"]] += 1
        native_shapes[str((row["native_original_variables"], row["native_original_constraints"],
                           row["native_variables"], row["native_constraints"]))] += 1
        if row["termination_condition"] == "optimal":
            assert row["status"] == 2 and row["solution_count"] > 0
            assert np.isfinite([row["objective"], row["bound"]]).all()
            absolute_gap = abs(row["objective"] - row["bound"])
            relative_gap = row["relative_gap"]
            if relative_gap is not None:
                assert np.isfinite(relative_gap)
            if "absolute_gap" in row:
                close(row["absolute_gap"], absolute_gap, rtol=1e-13, atol=1e-13)
            assert ((relative_gap is not None and relative_gap <= default_tolerances["MIPGap"] + 1e-10)
                    or absolute_gap <= default_tolerances["MIPGapAbs"] + 1e-10)
            assert row["max_count_equality_residual"] <= default_tolerances["FeasibilityTol"] * 1.1
            assert row["max_count_integrality_residual"] <= default_tolerances["IntFeasTol"] * 1.1
            max_equality_residual = max(max_equality_residual, row["max_count_equality_residual"])
            max_integrality_residual = max(max_integrality_residual, row["max_count_integrality_residual"])
            assert np.isfinite(row["max_storage_headroom_violation"])
            assert 0 <= row["max_storage_headroom_violation"] <= default_tolerances["FeasibilityTol"] * 1.1
            max_storage_headroom_violation = max(max_storage_headroom_violation,
                                                row["max_storage_headroom_violation"])
            if "max_original_binary_integrality_residual" in row:
                assert row["max_original_binary_integrality_residual"] <= default_tolerances["IntFeasTol"] * 1.1
                max_binary_residual = max(max_binary_residual, row["max_original_binary_integrality_residual"])
            for field in ("constraint_violation", "bound_violation"):
                if field in row:
                    assert row[field] is not None and np.isfinite(row[field]) and row[field] >= 0
            max_constraint_violation = max(max_constraint_violation, row.get("constraint_violation", 0))
            max_bound_violation = max(max_bound_violation, row.get("bound_violation", 0))
            if relative_gap is not None:
                max_gap = max(max_gap, relative_gap)
    assert status_counts["optimal"] == expected_successful_solves
    close(diagnostics["wall_seconds"], sum(row["wall_seconds"] for row in solves), atol=1e-7)
    close(diagnostics["optimizer_runtime_seconds"], sum(row["optimizer_runtime_seconds"] for row in solves), atol=1e-7)
    return {"status": "passed", "successful_original_physical_solves": expected_successful_solves,
            "total_attempts": len(solves), "termination_counts": dict(status_counts),
            "original_physical_variables": layout["original_variables"], "original_binary_variables": layout["original_binary_variables"],
            "original_physical_constraints": layout["original_constraints"], "redundant_integer_counts_per_solve": layout["added_counts"],
            "redundant_equalities_per_solve": layout["added_counts"], "implied_storage_headroom_cuts_per_solve": layout["added_headroom"],
            "headroom_period_zero_omitted": True, "storage_parameters": expected_storage,
            "deltaT": metadata["deltaT"], "native_shape_counts": dict(native_shapes),
            "actual_tolerances": default_tolerances, "maximum_relative_gap": max_gap,
            "maximum_count_equality_residual": max_equality_residual,
            "maximum_count_integrality_residual": max_integrality_residual,
            "maximum_original_binary_integrality_residual": max_binary_residual,
            "maximum_native_constraint_violation": max_constraint_violation,
            "maximum_native_bound_violation": max_bound_violation,
            "maximum_storage_headroom_violation": max_storage_headroom_violation,
            "equivalence": "K is uniquely determined by original binary sums; storage headroom cuts follow from binary charge mode, original SOC dynamics/bounds and nonnegative power; original integer feasible set is unchanged"}


def fixed_target_attainability(original_model, target, metadata, *, plain_pyomo=False):
    """Certify a saved physical point directly, avoiding a near-zero MIQP.

    Default QP optimality tolerances can leave a nonzero distance even when
    the target is attainable. A constant-objective MILP with all target
    coordinates fixed instead supplies a physical witness. The stricter
    tolerances below apply only to this independent audit, not the experiment.
    """
    model = original_model.clone()
    original_shape = physical_complexity(model)
    for objective in model.component_data_objects(pyo.Objective, active=True):
        objective.deactivate()
    model.audit_feasibility_objective = pyo.Objective(expr=0.0, sense=pyo.minimize)
    for t, value in enumerate(target):
        model.var_proj[t].fix(float(value))
    assert physical_complexity(model) == original_shape
    tolerance = 1e-9
    options = {"Threads": 8, "Seed": 0, "MIPFocus": 2, "Symmetry": 2,
               "FeasibilityTol": tolerance, "IntFeasTol": tolerance,
               "OptimalityTol": tolerance, "TimeLimit": 60}
    if plain_pyomo:
        solver = pyo.SolverFactory("gurobi")
        solver.options.update(options)
    else:
        from Simulator.microgrid_count_solver import CountBranchingSolver
        options["Presolve"] = 0
        solver = CountBranchingSolver(options=options,
                                      storage_parameters=metadata["ESS_list"],
                                      deltaT=metadata["deltaT"])
    result = solver.solve(model)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal
    actual = np.asarray([pyo.value(model.var_proj[t]) for t in range(len(target))])
    distance = float(np.sum((actual - target) ** 2))
    assert np.isfinite(actual).all() and distance == 0.0
    maximum_constraint_violation = 0.0
    maximum_bound_violation = 0.0
    maximum_integer_residual = 0.0
    for constraint in model.component_data_objects(pyo.Constraint, active=True):
        body = float(pyo.value(constraint.body))
        assert math.isfinite(body)
        if constraint.has_lb():
            maximum_constraint_violation = max(maximum_constraint_violation,
                                               float(pyo.value(constraint.lower)) - body)
        if constraint.has_ub():
            maximum_constraint_violation = max(maximum_constraint_violation,
                                               body - float(pyo.value(constraint.upper)))
    for variable in model.component_data_objects(pyo.Var):
        value = float(pyo.value(variable))
        assert math.isfinite(value)
        if variable.has_lb():
            maximum_bound_violation = max(maximum_bound_violation,
                                          float(pyo.value(variable.lb)) - value)
        if variable.has_ub():
            maximum_bound_violation = max(maximum_bound_violation,
                                          value - float(pyo.value(variable.ub)))
        if variable.is_binary() or variable.is_integer():
            maximum_integer_residual = max(maximum_integer_residual, abs(value - round(value)))
    residuals = {"original_constraint_violation": maximum_constraint_violation,
                 "original_bound_violation": maximum_bound_violation,
                 "original_integer_residual": maximum_integer_residual}
    assert all(value <= tolerance for value in residuals.values()), residuals
    native = None if plain_pyomo else solver.last_solve
    if native is not None:
        assert native["status"] == 2 and native["solution_count"] > 0
        assert native["original_variables"] == original_shape["variables"]
        assert native["original_binary_variables"] == original_shape["binary_variables"]
        assert native["original_constraints"] == original_shape["constraints"]
        assert native["native_original_quadratic_objective_terms"] == 0
        for name in ("constraint_violation", "bound_violation", "max_count_equality_residual",
                     "max_count_integrality_residual", "max_original_binary_integrality_residual",
                     "max_storage_headroom_violation"):
            assert native[name] is not None and native[name] <= tolerance, (name, native[name])
    return {"status": "passed", "method": "fixed_target_constant_objective_MILP",
            "target_squared_distance": distance, "physical_shape_unchanged": True,
            "original_physical_shape": original_shape, "integer_domains_preserved": True,
            "audit_only_solver_options": options, "residuals": residuals, "native_solve": native}


def independent_extremes(rows, metadata, A, b, config, *, plain_pyomo=False):
    """Repeat solves with fresh models; permit nonunique support coordinates.

    Projection checks fix the stored support target. A tied support optimum can
    otherwise select a different point whose geometric projection error differs.
    """
    mg = MicroGrid(seed=config["seed"], T=24, data=metadata)
    solver_options = {"Threads": 8, "Seed": 0, "MIPFocus": 2, "Symmetry": 2}
    mg.case_mg(A=A, b=b, is_uncertain=False, solver_options=solver_options)
    calculator = mg.errorcalculator
    physical_solver = None
    if not plain_pyomo:
        from Simulator.microgrid_count_solver import CountBranchingSolver
        solver_options = {**solver_options, "Presolve": 0}
        physical_solver = CountBranchingSolver(options=solver_options,
                                               storage_parameters=metadata["ESS_list"],
                                               deltaT=metadata["deltaT"])
        calculator.solver = physical_solver
    nominal_parameters = {
        "p_PV_rated": np.array([d["output_profile"] for d in metadata["PV_list"]]),
        "p_CL_rated": np.array([d["p_max_profile"] for d in metadata["CL_list"]]),
        "temp_amb": np.asarray(mg.temp_amb),
    }
    pmin = np.asarray(metadata["p_min"])
    width = np.asarray(metadata["p_max"] - metadata["p_min"])
    indices = {name: int(np.argmax([r[name] for r in rows]))
               for name in ("feasibility_error_rated", "optimality_error_rated")}
    records = []
    for name, index in indices.items():
        row = rows[index]
        c = np.asarray(row["direction"])
        x = (row["power_curve"] - pmin) / width
        y = (row["nominal_power_curve"] - pmin) / width
        calculator.update_parameters(nominal_parameters)
        support_approx = calculator.optimize_direction(c, in_approx=True)
        support_original = calculator.optimize_direction(c, in_approx=False)
        assert support_approx is not None and support_original is not None
        support_differences = {"approximate": float(c @ support_approx - c @ x),
                               "original": float(c @ support_original - c @ y)}
        # The physical mixed-integer solver's default relative MIPGap is 1e-4.
        # Respect the same certified accuracy rather than tightening the test.
        close(c @ support_approx, c @ x, rtol=1e-7, atol=1e-7, name="support approximation")
        close(c @ support_original, c @ y, rtol=1.1e-4, atol=2e-7, name="support physical MIP")
        assert np.max(A.astype(float) @ x - b) <= 2e-5
        assert np.max(A.astype(float) @ ((row["approx_projected_nominal_power_curve"] - pmin) / width) - b) <= 2e-5
        # Check the saved original support point is physically attainable.
        attainability = fixed_target_attainability(calculator.original_model, y, metadata,
                                                  plain_pyomo=plain_pyomo)
        projection_approx = calculator.project(y, to_approx=True)
        assert projection_approx is not None
        recomputed_optimality = float(np.sum((projection_approx - y) ** 2))
        calculator.update_parameters(row["perturbed_parameters"])
        projection_original = calculator.project(x, to_approx=False)
        assert projection_original is not None
        recomputed_feasibility = float(np.sum((projection_original - x) ** 2))
        close(recomputed_optimality, row["optimality_error_rated"], rtol=1.1e-4, atol=2e-7,
              name="independent geometric optimality")
        close(recomputed_feasibility, row["feasibility_error_rated"], rtol=1.1e-4, atol=2e-7,
              name="independent perturbed physical feasibility")
        records.append({"selected_for": name, "sample_index": index,
                        "stored_feasibility_error_rated": float(row["feasibility_error_rated"]),
                        "recomputed_feasibility_error_rated": recomputed_feasibility,
                        "stored_optimality_error_rated": float(row["optimality_error_rated"]),
                        "recomputed_optimality_error_rated": recomputed_optimality,
                        "support_objective_differences": support_differences,
                        "nominal_support_attainability_error": attainability["target_squared_distance"],
                        "nominal_support_attainability_certificate": attainability})
    return {"status": "passed", "fresh_Pyomo_models": True,
            "integer_variables_preserved": True, "solver_options": solver_options,
            "physical_backend": "pyomo_gurobi_original" if plain_pyomo else "gurobi_persistent_integer_counts",
            "physical_solver_diagnostics": physical_solver.diagnostics if physical_solver else None,
            "projection_comparison_atol": 2e-7, "projection_comparison_rtol": 1.1e-4,
            "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-solves", action="store_true")
    parser.add_argument("--plain-pyomo-extremes", action="store_true",
                        help="Use the original unaugmented Pyomo path for extreme checks; may be slow")
    parser.add_argument("--device-scale", type=int, choices=(3, 5), default=3)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--protected-manifest", type=Path)
    args = parser.parse_args()
    result_dir = (args.result_dir or ROOT / f"code/results/microgrid/scale{args.device_scale}").resolve()
    training_dir, output_dir = result_dir / "training", result_dir / "disaggregation"
    report = {"status": "running", "device_scale": args.device_scale,
              "checked_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        manifest = args.protected_manifest or ROOT / f"tmp/microgrid_scale_protected_inputs_scale{args.device_scale}.json"
        if not manifest.exists() and args.device_scale == 5 and args.protected_manifest is None:
            manifest = ROOT / "tmp/microgrid_scale5_protected_inputs.json"
        protected = read_json(manifest)
        required = protected.get("required_unchanged_files", protected.get("files", {}))
        captured = protected.get("captured_files", required)
        assert required, "Protection baseline must be captured before the experiment"
        changed = [name for name, expected in required.items()
                   if not (ROOT / name).is_file() or digest(ROOT / name) != expected]
        assert not changed, f"Protected original microgrid training inputs changed: {changed}"
        reference_changed = [name for name, expected in captured.items() if name not in required
                             and (not (ROOT / name).is_file() or digest(ROOT / name) != expected)]
        report["protected_inputs"] = {
            "manifest": str(manifest), "count": len(required), "unchanged": True,
            "captured_reference_files": len(captured), "reference_only_changed_files": reference_changed,
            "assertion_scope": protected.get("assertion_scope", "Legacy original Microgrid inputs"),
            "captured_at_utc": protected["captured_at_utc"]}
        config = read_json(training_dir / "training_config.json")
        report["source_audit"] = training_source_checks(training_dir, config)
        with (training_dir / "metadata.pkl").open("rb") as stream:
            metadata = pickle.load(stream)
        report["portfolio"], physical = portfolio_checks(metadata, config, args.device_scale)
        report["storage_headroom_original_algebra"] = storage_proof_checks(physical, metadata)
        report["training"], A, b = training_checks(training_dir, config, args.device_scale, report["source_audit"])
        report["evaluation"], rows, evaluation_config = evaluation_checks(output_dir, metadata, physical, args.device_scale)
        report["training_count_backend"] = count_backend_checks(training_dir, 20000, metadata, native_layout(physical))
        report["evaluation_count_backend"] = count_backend_checks(output_dir, 10000, metadata, native_layout(physical))
        if args.skip_solves:
            report["independent_extremes"] = {"status": "skipped", "reason": "--skip-solves"}
            report["status"] = "passed_basic_checks"
        else:
            report["independent_extremes"] = independent_extremes(
                rows, metadata, A, b, evaluation_config, plain_pyomo=args.plain_pyomo_extremes)
            report["status"] = "passed"
        artifacts = [*training_dir.glob("*.json"), *training_dir.glob("*.pkl"),
                     training_dir / "final_polytope.npz", training_dir / "pretrainnet_weights.pth",
                     *output_dir.glob("*.json"), *output_dir.glob("*.pkl"), *output_dir.glob("*.csv")]
        if (result_dir / "source_provenance.json").is_file():
            artifacts.extend([result_dir / "source_provenance.json", *(result_dir / "source_snapshot").rglob("*.py")])
        artifacts.extend(path for name in ("attainability_recheck.json", "attainability_recheck.npz",
                                            "projection_recheck.json")
                         if (path := result_dir / name).is_file())
        report["verified_artifact_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in artifacts}
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    except Exception as exc:
        report.update(status="failed", failure=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        raise
    finally:
        (result_dir / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()


