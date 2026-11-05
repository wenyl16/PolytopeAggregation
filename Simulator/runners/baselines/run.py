"""Build box, cutting-plane and vertex baselines, then evaluate shared directions.

Use --prepare-inner for the original MATLAB PEB-inner boundary-surrogate action.
"""

import argparse
import ast
import hashlib
import subprocess
import time
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from Simulator.runners.baselines.polytope import build_cube, build_cutting_plane, build_vertices
from Simulator.testers.polytope_evaluation import evaluate_errors
from Simulator.paths import CODE_ROOT, MATLAB_ROOT, RESULTS_ROOT
from Simulator.runners.train_aggregation import (
    build_aggregator, new_run_dir, common_archive, digest, scenario, write_json, json_value,
)


METHODS = ("box", "cutting_plane", "cutting_plane_modified", "pve", "pve_modified")




def _baseline_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    parser.add_argument("--time-limit", type=float, default=200)
    parser.add_argument("--horizon", type=int, choices=(24, 96), default=24)
    parser.add_argument("--case-seed", type=int, default=0)
    parser.add_argument("--direction-seed", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.time_limit <= 0 or args.num_samples < 1:
        parser.error("time-limit and num-samples must be positive")
    base = RESULTS_ROOT / "aggregation" / f"T{args.horizon}" / "baselines"
    output = new_run_dir(base, args.output_dir)
    directions = np.random.RandomState(args.direction_seed).normal(size=(args.num_samples, args.horizon))
    np.save(output / "directions.npy", directions)
    for method in dict.fromkeys(args.methods):
        agg = build_aggregator(args.horizon, args.case_seed)
        modified = method.endswith("_modified")
        if method == "box":
            approximation, seconds = build_cube(agg)
            info = {"status": "completed"}
        elif method.startswith("cutting_plane"):
            approximation, seconds = build_cutting_plane(agg, timelimit=args.time_limit, modified=modified)
            info = getattr(agg, "cutting_plane_modified_info" if modified else "cutting_plane_info")
        else:
            approximation, seconds = build_vertices(agg, timelimit=args.time_limit, modified=modified)
            info = getattr(agg, "vertex_enumeration_modified_info" if modified else "vertex_enumeration_info")
        region = {"V": approximation} if method.startswith("pve") else dict(zip(("A", "b"), approximation))
        if "V" in region and np.asarray(region["V"]).size == 0:
            # A legal time limit can expire before the first vertex is found.
            # Keep that algorithm outcome without inventing evaluation errors.
            errors = {}
            evaluation = {"evaluation_status": "skipped",
                          "evaluation_reason": "No vertices returned; the empty region cannot be evaluated."}
        else:
            errors = evaluate_errors(agg, directions=directions, **region)
            evaluation = {"evaluation_status": "completed", "evaluation_reason": ""}
        result = {"method": method, "comp_time": seconds, "status": info["status"],
                  "case_seed": args.case_seed, "direction_seed": args.direction_seed,
                  "timelimit": args.time_limit, "T": agg.T, "pmin": agg.pmin, "pmax": agg.pmax,
                  **region, **errors, **evaluation}
        # Explicit directions are saved above; their seed remains known to this runner.
        result["direction_seed"] = args.direction_seed
        np.savez(output / f"{method}.npz", **result)
        (output / f"{method}_details.json").write_text(
            json.dumps({**info, **evaluation}, default=json_value, ensure_ascii=False, indent=2), encoding="utf-8")
        if method == "box":
            np.savetxt(output / "A_cube.csv", region["A"], delimiter=",")
            np.savetxt(output / "b_cube.csv", region["b"], delimiter=",")
        if evaluation["evaluation_status"] == "completed":
            print(f"{method}: {seconds:.3f} s, {info['status']}, "
                  f"feas={errors['mean_feas']:.6g}, opt={errors['mean_opt']:.6g}")
        else:
            print(f"{method}: {seconds:.3f} s, {info['status']}, "
                  f"evaluation skipped: {evaluation['evaluation_reason']}")
    print(f"Saved baselines: {output}")


LIMITATION = (
    "The original MATLAB PEB-inner method operates on per-device power and "
    "ordinary cumulative-energy envelopes from gen_boundaries(). For TCLs "
    "these envelopes are a surrogate, not equivalent to the original "
    "exponentially decaying temperature dynamics in case_aggregator(). "
    "gen_boundaries advances temperature at t=0 while case_aggregator fixes "
    "the initial temperature and starts transitions at t=1. Consequently "
    "inner describes the surrogate approximation; feasibility against the "
    "original device model must be measured separately."
)

def array_digest(value):
    value = np.ascontiguousarray(value, dtype=np.float64)
    return hashlib.sha256(value.tobytes()).hexdigest()

def legacy_model_audit(agg):
    """Record semantic source checks for the retained 200-second baselines."""
    old_path = CODE_ROOT / "Simulator/runners/baselines/legacy_model_reference.py"
    new_path = CODE_ROOT / "Simulator/cases/aggregation.py"
    def functions(path):
        module = ast.parse(path.read_text(encoding="utf-8-sig"))
        return {node.name: node for cls in module.body if isinstance(cls, ast.ClassDef)
                and cls.name == "Aggregator" for node in cls.body if isinstance(node, ast.FunctionDef)}
    old, current = functions(old_path), functions(new_path)
    checks = {}
    for name in ("gen_CL", "gen_PV"):
        checks[name + "_ast_equal"] = ast.dump(old[name], include_attributes=False) == ast.dump(
            current[name], include_attributes=False)
    def model_statements(node):
        values = []
        for statement in node.body:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) \
                    and isinstance(statement.value.value, str):
                continue
            values.append(ast.dump(statement, include_attributes=False))
            if isinstance(statement, ast.Assign) and any(
                    isinstance(target, ast.Attribute) and target.attr == "model"
                    for target in statement.targets):
                break
        return values
    checks["device_model_ast_equal"] = model_statements(old["case_aggregator"]) == model_statements(
        current["case_aggregator"])
    # gen_TCL differs solely in the location-independent input path expression.
    for nodes in (old, current):
        nodes["gen_TCL"].body[0] = ast.Pass()
    checks["gen_TCL_after_removing_input_path_ast_equal"] = ast.dump(
        old["gen_TCL"], include_attributes=False) == ast.dump(current["gen_TCL"], include_attributes=False)
    if not all(checks.values()):
        raise ValueError(f"Legacy baseline model does not match: {checks}")
    records = {}
    for name, folder in (("cutting_plane", "cutting plane"),
                         ("cutting_plane_modified", "cutting plane"),
                         ("pve", "pve"), ("pve_modified", "pve")):
        path = RESULTS_ROOT / "aggregation/T24/baselines" / folder / (name + ".npz")
        with np.load(path, allow_pickle=False) as saved:
            for key, expected in {"T": 24, "case_seed": 0, "n_cl": 5, "n_tcl": 40,
                                  "n_pv": 5, "timelimit": 200}.items():
                if saved[key].item() != expected:
                    raise ValueError(f"{name}: unexpected {key}")
            np.testing.assert_array_equal(saved["pmin"], agg.pmin)
            np.testing.assert_array_equal(saved["pmax"], agg.pmax)
        records[name] = {"path": str(path.resolve()), "sha256": digest(path),
                         "metadata_and_full_physical_coordinate_checks": "passed"}
    return {"source_ast_checks": checks,
            "source_sha256": {str(path.relative_to(CODE_ROOT)): digest(path)
                              for path in (old_path, new_path)}, "archives": records}

def matlab_string(value):
    return "'" + str(value).replace("\\", "/").replace("'", "''") + "'"

def prepare_inner(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab", type=Path, default=Path("matlab"))
    parser.add_argument("--output-dir", type=Path,
                        default=RESULTS_ROOT / "aggregation/T24/baselines/matlab/verified_case0")
    parser.add_argument("--resume-prepared", action="store_true",
                        help="Resume only a prepared directory before MATLAB has started")
    args = parser.parse_args(argv)
    from scipy.io import loadmat, savemat
    from Simulator.testers.case_methods import load_polytope
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    agg = build_aggregator(24, 0)
    agg.case_aggregator(build_only=True)
    scene = scenario(agg)
    if args.resume_prepared:
        output = args.output_dir
        saved = json.loads((output / "scenario.json").read_text(encoding="utf-8"))
        if saved["scenario_sha256"] != scene["scenario_sha256"] or any(
                (output / name).exists() for name in ("matlab.log", "polytope_params.mat", "final_polytope.npz")):
            raise ValueError("Only an identical scene not yet started in MATLAB may resume preparation")
    else:
        output = new_run_dir(args.output_dir, args.output_dir)
    write_json(output / "scenario.json", scene)
    grouped = agg.gen_boundaries()
    boundaries = {key: np.concatenate([
        np.asarray(grouped[resource][key.title()], dtype=float).T
        for resource in ("PV", "CL", "TCL")], axis=1)
        for key in ("pmin", "pmax", "emin", "emax")}
    for value in boundaries.values():
        if value.shape != (24, 50) or not np.all(np.isfinite(value)):
            raise ValueError("Invalid current-case boundary matrix")
    boundary_file = output / "boundaries.mat"
    savemat(boundary_file, boundaries)
    source_paths = [Path(__file__).resolve(), CODE_ROOT / "Simulator/runners/train_aggregation.py",
                    CODE_ROOT / "Simulator/cases/aggregation.py",
                    CODE_ROOT / "Simulator/cases/aggregation_case_T24.py",
                    MATLAB_ROOT / "run_baseline.m", MATLAB_ROOT / "data/Aggs_data.mat"]
    source_paths.extend(sorted((MATLAB_ROOT / "@Aggregator").glob("*.m")))
    provenance = {"status": "running", "method": "peb_inner", "template_id": 1,
                  "scenario_sha256": scene["scenario_sha256"],
                  "case": {"aggregator_class": scene["aggregator_class"], "case_seed": 0,
                           "T": 24, "counts": [5, 40, 5], "discrete_rate": 0.},
                  "boundary_column_order": ["PV", "CL", "TCL"],
                  "boundary_file_sha256": digest(boundary_file),
                  "boundary_matrix_sha256": {key: array_digest(value) for key, value in boundaries.items()},
                  "source_sha256": {str(path.relative_to(CODE_ROOT)): digest(path) for path in source_paths},
                  "algorithm": "Original MATLAB run_baseline(template_id=1) and innerapprox_mine; CPLEX; original 20-round cap and 1e-4 stopping tolerance",
                  "model_limitation": LIMITATION,
                  "legacy_200_second_baseline_audit": legacy_model_audit(agg)}
    write_json(output / "provenance.json", provenance)
    mat_file = output / "polytope_params.mat"
    expression = "; ".join([
        "disp(version)", "fprintf('PEB_INNER_STARTED %s\\n',datestr(now,31))",
        "addpath(" + matlab_string(MATLAB_ROOT) + ")",
        "[polytope,diagnostics]=run_baseline(1," + matlab_string(boundary_file) + "," + matlab_string(mat_file) + ")",
        "disp(diagnostics)", "fprintf('PEB_INNER_FINISHED %s\\n',datestr(now,31))"])
    command = [str(args.matlab), "-batch", expression]
    provenance["matlab_command"] = command
    write_json(output / "provenance.json", provenance)
    started = time.perf_counter()
    print(f"Running original MATLAB PEB inner; output={output}", flush=True)
    try:
        with (output / "matlab.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=str(CODE_ROOT), stdout=log, stderr=subprocess.STDOUT)
            previous_size = 0
            while process.poll() is None:
                time.sleep(10)
                size = (output / "matlab.log").stat().st_size
                print(f"MATLAB running: {time.perf_counter()-started:.0f}s; log={size} bytes", flush=True)
                if size != previous_size:
                    previous_size = size
                    tail = (output / "matlab.log").read_text(encoding="utf-8", errors="replace").splitlines()[-4:]
                    print("\n".join(tail), flush=True)
            if process.returncode:
                raise RuntimeError(f"MATLAB exited with status {process.returncode}; see matlab.log")
        region = load_polytope(mat_file)
        A, b = region["A"], region["b"]
        if A.shape != (96, 24) or b.shape != (96,) or not all(
                np.all(np.isfinite(value)) for value in (A, b)):
            raise ValueError("MATLAB did not return a finite 96-by-24 PEB approximation")
        diagnostics = loadmat(mat_file, simplify_cells=True)["diagnostics"]
        raw = diagnostics["aggregators"]
        records = [raw] if isinstance(raw, dict) else list(np.asarray(raw, dtype=object).reshape(-1))
        summaries = []
        for record in records:
            errors = np.asarray(record["errors"], dtype=float).reshape(-1)
            if not errors.size or not np.all(np.isfinite(errors)):
                raise ValueError("MATLAB did not return a finite stopping-error history")
            summaries.append({"errors": errors.tolist(), "iterations": int(record["iterations"]),
                              "converged": bool(record["converged"]),
                              "solvertime": float(record["solvertime"])})
        converged = all(record["converged"] for record in summaries)
        archive = output / "final_polytope.npz"
        np.savez(archive, A=A, b=b, **common_archive(agg, scene, "peb_inner"),
                 algorithm_converged=converged, model_type="cumulative_energy_boundary_surrogate")
        reloaded = load_polytope(output)
        np.testing.assert_array_equal(reloaded["A"], A)
        np.testing.assert_array_equal(reloaded["b"], b)
        provenance.update(status="completed", elapsed_seconds=time.perf_counter()-started,
                          algorithm_converged=converged,
                          termination="stopping_tolerance" if converged else "original_20_round_limit",
                          matlab_diagnostics=summaries, final_polytope_sha256=digest(archive),
                          matlab_result_sha256=digest(mat_file), matlab_log_sha256=digest(output / "matlab.log"),
                          final_shape=list(A.shape), validation="finite; exact archive reload; existing .mat loader row normalization")
        write_json(output / "provenance.json", provenance)
        print(f"Completed PEB inner: converged={converged}; elapsed={provenance['elapsed_seconds']:.1f}s", flush=True)
    except BaseException as exc:
        provenance.update(status="failed", elapsed_seconds=time.perf_counter()-started,
                          failure=f"{type(exc).__name__}: {exc}")
        write_json(output / "provenance.json", provenance)
        raise

def main(argv=None):
    """Dispatch ordinary baselines or the explicit original MATLAB inner action."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--prepare-inner" in arguments:
        arguments.remove("--prepare-inner")
        return prepare_inner(arguments)
    return _baseline_main(arguments)


if __name__ == "__main__":
    main()
