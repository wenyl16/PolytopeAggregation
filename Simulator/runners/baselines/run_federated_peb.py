"""Train the paper's federated PEB baseline and append its shared comparisons."""

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import zipfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import gurobipy as gp
import numpy as np

from Simulator.runners.baselines.federated_training import (
    PaperAlgorithmFailure, train_federated_peb,
    peb_volume_and_gradient, peb_volume_and_gradient_modified,
)
from Simulator.paths import CODE_ROOT, RESULTS_ROOT
from Simulator.runners.train_aggregation import build_aggregator, common_archive, digest, scenario, write_json
from Simulator.testers.case_methods import default_methods, load_polytope, verify_scenario
from Simulator.testers.polytope_evaluation import (
    SOLVER_OPTIONS, _array_hash, _build_approximation, evaluate_native,
)

NAME = "federated_peb"
BASE = RESULTS_ROOT / "aggregation/T24"
OUTPUT = BASE / "baselines" / NAME
COMPARISON = BASE / "comparisons"
EXPERIMENT = "t24-federated-full-affine-peb-200s-v1"
REFERENCE_PAPER_SHA256 = "5832deb9f83c0cca8ed8b3a9c4b354ba27b0fe91d5529af07475298a9e031b41"  # DOI 10.1109/TSG.2026.3676867
PAPER_PROTOCOL = "original_equations_12_14_15_17_19_36_38_algorithm1"
MODIFIED_PROTOCOL = "user_authorized_lexicographic_peb_variant_v1"


def archive_strict_evidence():
    """Keep the prior proof as one audit archive, not a second active result."""
    proof_names = ("paper_fidelity_check.json", "paper_fidelity_verification.json", "kkt_inverse_confirmation.json")
    if not any((OUTPUT / name).exists() for name in proof_names):
        return None
    archive = OUTPUT / "evidence/strict_initial_evidence.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    paths = [OUTPUT / name for name in (*proof_names, "algorithm_notes.md", "certificate.npz",
             "scenario.json", "provenance.json", "training_diagnostics.json", "training_progress.json")]
    paths.extend((OUTPUT / "devices").glob("device_*.npz"))
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as saved:
        for path in paths:
            if path.is_file():
                saved.write(path, str(path.relative_to(OUTPUT)))
    for name in proof_names:
        (OUTPUT / name).unlink(missing_ok=True)
    return {"path": str(archive.resolve()), "sha256": digest(archive), "meaning": "Prior strict-paper failure evidence; not an active trained model"}


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].item() if saved[key].ndim == 0 else saved[key].copy() for key in saved.files}


def protected_inputs():
    metadata = json.loads((COMPARISON / "metadata.json").read_text(encoding="utf-8-sig"))
    if metadata.get("error_status") != "completed" or metadata.get("cost_status") != "completed":
        raise ValueError("A completed existing methods comparison is required")
    old_names = [name for name in metadata["method_order"] if name != NAME]
    paths = {COMPARISON / name for name in ("directions.npy", "cost_coefficients.npy", "cost_scenarios.csv",
             "price_provenance.json", "original_costs.npy", "original_cost_dispatch.npz", metadata["daily_prices_file"])}
    for name in old_names:
        paths.update(COMPARISON / f"{name}{suffix}" for suffix in ("_errors.npz", "_costs.npy", "_cost_dispatch.npz"))
    for name, directory in default_methods(BASE):
        if name != NAME:
            paths.add(directory / "final_polytope.npz" if directory.is_dir() else directory)
    paths.add(COMPARISON / "peb_outer.npz")
    return {str(path.resolve()): digest(path) for path in paths}, metadata, old_names


def assert_protected(protected):
    for path, expected in protected.items():
        if digest(path) != expected:
            raise RuntimeError(f"An existing benchmark input changed: {path}")


def withdraw_nonpaper_comparison(protected, reason):
    """Remove the superseded auxiliary-L1 variant, preserving the nine controls."""
    assert_protected(protected)
    metadata = json.loads((COMPARISON / "metadata.json").read_text(encoding="utf-8-sig"))
    names = [name for name in metadata["method_order"] if name != NAME]
    for filename in ("error_comparison.pkl", "cost_comparison.pkl"):
        with (COMPARISON / filename).open("rb") as stream:
            data = pickle.load(stream)
        data.pop(NAME, None)
        with (COMPARISON / filename).open("wb") as stream:
            pickle.dump(data, stream)
    for filename in ("error_samples.csv", "cost_samples.csv"):
        path = COMPARISON / filename
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerows(row for row in rows if row and row[0] != NAME)
    summary = json.loads((COMPARISON / "summary.json").read_text(encoding="utf-8-sig"))
    summary.pop(NAME, None)
    write_json(COMPARISON / "summary.json", summary)
    cost = metadata["native_cost_diagnostics"]
    removed = cost["models"].pop(NAME, None)
    if removed:
        for key in ("solves", "successful_solves", "solver_seconds"):
            cost[key] -= removed[key]
    for key in ("region_sha256", "model_sha256", "scale_sha256", "offset_sha256"):
        if isinstance(cost.get(key), dict):
            cost[key].pop(NAME, None)
    cost["method_order"] = ["original", *names]
    cost.pop("incremental_update", None)
    metadata.update(method_order=names, completed_error_methods=names)
    for key in ("methods", "scenario_audit", "model_sources_sha256"):
        metadata.get(key, {}).pop(NAME, None)
    metadata.pop("federated_baseline_update", None)
    metadata["excluded_methods"] = {NAME: {"reason": reason,
        "paper_core_protocol": PAPER_PROTOCOL,
        "superseded_result": "The previous auxiliary-L1/near-optimal-trace variant is not the paper algorithm"}}
    metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(COMPARISON / "cost_solver_diagnostics.json", cost)
    write_json(COMPARISON / "metadata.json", metadata)
    for suffix in ("_errors.npz", "_costs.npy", "_cost_dispatch.npz"):
        (COMPARISON / f"{NAME}{suffix}").unlink(missing_ok=True)
    assert_protected(protected)
    from Simulator.drawers.plot_comparison import main as plot_comparison
    plot_comparison([str(COMPARISON), "--sync-summary"])


def save_local_certificates(trained):
    H0, h, s, S = (trained[key] for key in ("H0", "h", "s", "S"))
    np.savez(OUTPUT / "certificate.npz", H0=H0, h=h, raw_h=trained["raw_h"], s=s, S=S,
             singular_values=trained["singular_values"],
             state="initial_untrained" if trained["diagnostics"]["status"] != "completed" else "trained")
    devices = OUTPUT / "devices"
    devices.mkdir(exist_ok=True)
    for index, (device, local) in enumerate(zip(trained["devices"], trained["local_results"])):
        values = {key: local[key] for key in ("gamma", "Gamma", "Lambda", "primary_trace_optimum",
            "primary_optimality_equality_residual", "primary_trace_gap", "primary_value_gradient",
            "primary_dual_normals", "primary_dual_containment", "primary_dual_nonnegativity",
            "primary_dual_objective", "primary_gamma", "primary_Gamma", "primary_Lambda",
            "primary_trace_tolerance", "secondary_trace_target", "secondary_trace_target_residual",
            "secondary_offdiagonal_l1", "secondary_dual_normals", "secondary_dual_containment",
            "secondary_dual_nonnegativity", "secondary_dual_offdiagonal_plus", "secondary_dual_offdiagonal_minus",
            "secondary_dual_trace", "secondary_dual_objective", "trace_target_derivative_scale") if key in local}
        np.savez(devices / f"device_{index:03d}.npz", A=device["A"], b=device["b"],
                 id=device["id"], family=device["family"], **values)


def costs_for_new_region(agg, coefficients, region):
    """Only the new region is solved; the original and previous nine stay cached."""
    started = time.perf_counter()
    with gp.Env(empty=True) as env:
        for key, value in SOLVER_OPTIONS.items():
            env.setParam(key, value)
        env.start()
        native, signature = _build_approximation(agg.T, region, env)
        try:
            points = np.asarray([native.point(row * (agg.pmax - agg.pmin)) for row in coefficients])
            physical = points * (agg.pmax - agg.pmin) + agg.pmin
            costs = np.sum(coefficients * physical, axis=1)
            diagnostics = {"variables": native.model.NumVars, "rows": native.model.NumConstrs,
                           "solves": native.solves, "successful_solves": len(coefficients),
                           "solver_seconds": native.solver_seconds, "elapsed_seconds": time.perf_counter()-started,
                           "termination_status": "OPTIMAL", "region_sha256": signature}
        finally:
            native.model.dispose()
    return costs, points, physical, diagnostics


def merge_comparisons(agg, scene, region, result, new_costs, normalized, physical, diagnostics, protected):
    assert_protected(protected)
    metadata = json.loads((COMPARISON / "metadata.json").read_text(encoding="utf-8-sig"))
    summary = json.loads((COMPARISON / "summary.json").read_text(encoding="utf-8-sig"))
    with (COMPARISON / "error_comparison.pkl").open("rb") as stream:
        previous_errors = pickle.load(stream)
    with (COMPARISON / "cost_comparison.pkl").open("rb") as stream:
        previous_costs = pickle.load(stream)
    old_array_hashes = {name: {key: _array_hash(values[key]) for key in ("error_feas", "error_opt")}
                        for name, values in previous_errors.items() if name != NAME}
    old_cost_hashes = {name: _array_hash(values) for name, values in previous_costs.items() if name != NAME}
    names = [name for name, _path in default_methods(BASE)] + ["peb_outer"]
    previous_errors[NAME] = result
    errors = {name: previous_errors[name] for name in names}
    previous_costs[NAME] = new_costs
    costs = {name: previous_costs[name] for name in ("original", *names)}
    if any(values["native_directions_sha256"] != result["native_directions_sha256"] or
           values["native_model_sha256"] != result["native_model_sha256"] for values in errors.values()):
        raise ValueError("The new method must use the same full physical model and evaluation directions")
    with np.load(COMPARISON / "original_cost_dispatch.npz", allow_pickle=False) as saved:
        dates = saved["utc_day"].copy()
    np.savez(COMPARISON / f"{NAME}_errors.npz", **{key: value for key, value in result.items() if value is not None})
    np.save(COMPARISON / f"{NAME}_costs.npy", new_costs)
    np.savez(COMPARISON / f"{NAME}_cost_dispatch.npz", normalized_power=normalized,
             physical_power=physical, utc_day=dates)
    with (COMPARISON / "error_comparison.pkl").open("wb") as stream:
        pickle.dump(errors, stream)
    with (COMPARISON / "cost_comparison.pkl").open("wb") as stream:
        pickle.dump(costs, stream)
    with (COMPARISON / "error_samples.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("method", "sample", "feasibility_squared_distance", "optimality_squared_distance"))
        for name in names:
            writer.writerows((name, i, f, o) for i, (f, o) in enumerate(zip(errors[name]["error_feas"], errors[name]["error_opt"])))
    with (COMPARISON / "cost_samples.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("method", "sample", "utc_day", "original_cost", "method_cost", "difference"))
        for name in names:
            writer.writerows((name, i, dates[i], reference, value, value-reference)
                             for i, (reference, value) in enumerate(zip(costs["original"], costs[name])))
    difference = new_costs - costs["original"]
    summary[NAME] = {key: result[key] for key in ("mean_feas", "max_feas", "mean_opt", "max_opt", "count_feas_gt_1e4")}
    summary[NAME].update(mean_cost_difference=float(difference.mean()), median_cost_difference=float(np.median(difference)),
                         min_cost_difference=float(difference.min()), max_cost_difference=float(difference.max()),
                         mean_cost=float(new_costs.mean()), num_cost_samples=365, cost_unit="AUD per 24 h")
    cost_diagnostics = copy.deepcopy(metadata["native_cost_diagnostics"])
    old_record = cost_diagnostics["models"].get(NAME)
    if old_record is not None:
        cost_diagnostics["solves"] -= old_record["solves"]
        cost_diagnostics["successful_solves"] -= old_record["successful_solves"]
        cost_diagnostics["solver_seconds"] -= old_record["solver_seconds"]
    cost_diagnostics["models"][NAME] = diagnostics
    cost_diagnostics["region_sha256"][NAME] = diagnostics["region_sha256"]
    cost_diagnostics["method_order"] = ["original", *names]
    for key in ("solves", "successful_solves", "solver_seconds"):
        cost_diagnostics[key] += diagnostics[key]
    cost_diagnostics["incremental_update"] = {"method": NAME, "new_solves": 365,
        "existing_models_and_price_data_reused": True, "original_costs_recomputed": False}
    metadata.update(status="completed", method_order=names, completed_error_methods=names,
                    num_samples=5000, num_error_samples=5000, num_cost_samples=365,
                    error_status="completed", cost_status="completed", native_cost_diagnostics=cost_diagnostics,
                    last_updated_metric="both", completed_at_utc=datetime.now(timezone.utc).isoformat())
    metadata["methods"][NAME] = str(OUTPUT.resolve())
    metadata["scenario_audit"][NAME] = verify_scenario(OUTPUT, agg, 0, required=True)
    metadata.get("excluded_methods", {}).pop(NAME, None)
    current_provenance = json.loads((OUTPUT / "provenance.json").read_text(encoding="utf-8"))
    metadata["federated_baseline_update"] = {"method": NAME, "source": str(OUTPUT.resolve()),
        "training_budget_seconds": 200, "evaluation_directions": 5000, "cost_days": 365,
        "old_error_array_hashes": old_array_hashes, "old_cost_array_hashes": old_cost_hashes,
        "protected_inputs_sha256": protected, "publication_source": str(Path(__file__).resolve())}
    metadata["federated_baseline_update"].update(
        implementation_protocol=current_provenance["paper_core_protocol"],
        implementation_variant=current_provenance.get("implementation_variant"),
        paper_identical_implementation=current_provenance.get("paper_identical_implementation", True))
    write_json(COMPARISON / "cost_solver_diagnostics.json", cost_diagnostics)
    write_json(COMPARISON / "summary.json", summary)
    write_json(COMPARISON / "metadata.json", metadata)
    assert_protected(protected)
    from Simulator.drawers.plot_comparison import main as plot_comparison
    plot_comparison([str(COMPARISON), "--sync-summary"])
    return old_array_hashes, old_cost_hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-limit", type=float, default=200.)
    parser.add_argument("--grid-points", type=int, default=1025)
    parser.add_argument("--evaluate-only", action="store_true", help="Publish an already completed fixed baseline without retraining")
    parser.add_argument("--degeneracy-treatment", choices=("lexicographic", "strict"), default="lexicographic",
                        help="User-authorized auxiliary numerical selection (default), or strict original-paper diagnostics")
    args = parser.parse_args(argv)
    modified = args.degeneracy_treatment == "lexicographic"
    protocol = MODIFIED_PROTOCOL if modified else PAPER_PROTOCOL
    if args.time_limit != 200.:
        parser.error("The published comparison uses exactly a 200-second training budget")
    protected, old_metadata, old_names = protected_inputs()
    if len(old_names) != 9 or old_metadata.get("cost_price_area", old_metadata.get("price_area")) != "AEMO-VIC1":
        raise ValueError("Expected the nine-method AEMO-VIC1 base comparison")
    directions = np.load(COMPARISON / "directions.npy")
    coefficients = np.load(COMPARISON / "cost_coefficients.npy")
    np.testing.assert_array_equal(directions, np.random.RandomState(1).randn(5000, 24))
    if coefficients.shape != (365, 24):
        raise ValueError("Expected all 365 existing daily price profiles")
    agg = build_aggregator(24, 0)
    agg.case_aggregator(build_only=True)
    scene = scenario(agg)
    if OUTPUT.is_symlink() or not OUTPUT.resolve().is_relative_to((BASE / "baselines").resolve()):
        raise ValueError("Invalid fixed baseline destination")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    marker = OUTPUT / "provenance.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8"))["experiment"] != EXPERIMENT:
        raise ValueError("An unrelated experiment occupies the baseline destination")
    write_json(OUTPUT / "scenario.json", scene)
    if not args.evaluate_only:
        strict_archive = archive_strict_evidence() if modified else None
        earlier_attempts = []
        if marker.is_file():
            previous = json.loads(marker.read_text(encoding="utf-8"))
            earlier_attempts = previous.get("earlier_implementation_attempts", [])
            earlier_attempts.append({"status": previous.get("status"), "failure": previous.get("failure"),
                "source_sha256": previous.get("source_sha256"), "weights_reused": False})
        provenance = {"experiment": EXPERIMENT, "status": "training", "method": NAME,
            "paper_core_protocol": protocol,
            "implementation_variant": "modified: user-authorized near-optimal-trace L1 selection" if modified else "strict paper equations",
            "paper_identical_implementation": not modified,
            "degeneracy_treatment": args.degeneracy_treatment,
            "prior_strict_evidence": strict_archive,
            "paper_doi": "10.1109/TSG.2026.3676867", "paper_sha256": REFERENCE_PAPER_SHA256,
            "scenario_sha256": scene["scenario_sha256"], "case_seed": 0, "counts": [5,40,5],
            "training_budget_seconds": 200., "protected_inputs_sha256": protected,
            "earlier_implementation_attempts": earlier_attempts,
            "source_sha256": {str(path.relative_to(CODE_ROOT)): digest(path) for path in (
                Path(__file__).resolve(), CODE_ROOT / "Simulator/runners/baselines/federated_training.py",
                CODE_ROOT / "Simulator/runners/baselines/federated_aggregation.py",
                CODE_ROOT / "Simulator/cases/aggregation.py", CODE_ROOT / "Simulator/cases/aggregation_case_T24.py")}}
        write_json(marker, provenance)
        def progress(record):
            write_json(OUTPUT / "training_progress.json", record)
            if record.get("event") != "client_round" or record.get("completed_clients") == 50:
                print(json.dumps(record, default=lambda x: x.item() if isinstance(x,np.generic) else x), flush=True)
        try:
            trained = train_federated_peb(agg, timelimit=200., grid_points=args.grid_points, progress=progress,
                                          mode=args.degeneracy_treatment)
        except PaperAlgorithmFailure as exc:
            write_json(OUTPUT / "training_diagnostics.json", exc.diagnostics)
            if exc.initial is not None:
                save_local_certificates(exc.initial)
            for filename in ("final_polytope.npz", "evaluation_errors.npz", "evaluation_directions.npy",
                             f"{NAME}_checkpoint.npz", "evaluation_progress.json",
                             "independent_recheck.json", "independent_coordinates.npz"):
                (OUTPUT / filename).unlink(missing_ok=True)
            provenance.update(status="unsupported_paper_regularity", failure=str(exc),
                evaluation_status="not_run_no_valid_trained_model", old_results_unchanged=True,
                training_diagnostics_sha256=digest(OUTPUT / "training_diagnostics.json"))
            write_json(marker, provenance)
            withdraw_nonpaper_comparison(protected, str(exc))
            print(f"Original paper algorithm stopped: {exc}. Superseded nonpaper result withdrawn.", flush=True)
            return
        except BaseException as exc:
            provenance.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
            write_json(marker, provenance)
            raise
        H0, h, s, S = (trained[key] for key in ("H0", "h", "s", "S"))
        transformed = np.linalg.solve(S.T, H0.T).T
        region = {"A": transformed * (agg.pmax-agg.pmin)[None,:], "b": h + transformed @ (s-agg.pmin)}
        save_local_certificates(trained)
        volume_function = peb_volume_and_gradient_modified if modified else peb_volume_and_gradient
        fine = volume_function(h, 24, grid_points=2*args.grid_points-1, delta_t=agg.deltaT)
        diagnostic = trained["diagnostics"]
        diagnostic["final_volume_grid_refinement"] = {"coarse_grid": args.grid_points,
            "fine_grid": 2*args.grid_points-1,
            "relative_volume_change": float(abs(np.expm1(fine["log_volume"]-trained["volume"]["log_volume"]))),
            "relative_gradient_change": float(np.linalg.norm(fine["log_gradient"]-trained["volume"]["log_gradient"])/max(1.,np.linalg.norm(fine["log_gradient"])))}
        write_json(OUTPUT / "training_diagnostics.json", diagnostic)
        np.savez(OUTPUT / "final_polytope.npz", **region, **common_archive(agg, scene, NAME),
                 timelimit=200., training_seed=0, comp_time=diagnostic["training_seconds"],
                 implementation_protocol=protocol, paper_identical_implementation=not modified)
        provenance.update(status="completed", final_polytope_sha256=digest(OUTPUT / "final_polytope.npz"),
                          training_diagnostics_sha256=digest(OUTPUT / "training_diagnostics.json"),
                          training_diagnostics_summary={key: value for key, value in diagnostic.items() if key != "history"})
        write_json(marker, provenance)
    else:
        provenance = json.loads(marker.read_text(encoding="utf-8"))
        if (provenance.get("status") != "completed" or
                provenance.get("paper_core_protocol") not in (PAPER_PROTOCOL, MODIFIED_PROTOCOL)):
            raise ValueError("Only a completed declared original/modified model can be evaluated/published")
        verify_scenario(OUTPUT, agg, 0, required=True)
        region = load_polytope(OUTPUT)
    for filename in (f"{NAME}_checkpoint.npz", "evaluation_errors.npz"):
        (OUTPUT / filename).unlink(missing_ok=True)
    def error_progress(record):
        write_json(OUTPUT / "evaluation_progress.json", record)
        if record.get("completed",0) % 1000 == 0:
            print(f"{NAME}: {record.get('completed',0)}/5000 evaluation directions", flush=True)
    result = evaluate_native(agg, directions, region, progress=error_progress,
                             checkpoint_path=OUTPUT / f"{NAME}_checkpoint.npz")
    np.save(OUTPUT / "evaluation_directions.npy", directions)
    np.savez(OUTPUT / "evaluation_errors.npz", **{key:value for key,value in result.items() if value is not None})
    new_costs, normalized, physical, diagnostics = costs_for_new_region(agg, coefficients, region)
    if diagnostics["region_sha256"] != result["native_region_sha256"]:
        raise ValueError("Cost and error approximations differ")
    provenance.update(evaluation_status="completed", evaluation_seed=1, evaluation_samples=5000, cost_days=365,
                      old_results_unchanged=True, publication_started_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(marker, provenance)
    merge_comparisons(agg, scene, region, result, new_costs, normalized, physical, diagnostics, protected)
    print(f"Published {NAME}: mean feasibility={result['mean_feas']:.9g}; mean optimality={result['mean_opt']:.9g}", flush=True)


if __name__ == "__main__":
    main()
