"""Deadline-controlled Taheri et al. (TSG 2022) reproduction on the T24 microgrid.

All new implementation modules live in runners. Checkpoints are written before
expensive stages. A supervising process enforces a wall deadline even if a native
solver does not return. No incumbent is called a completed paper model unless
all four paper stages and the containment checks have succeeded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                     allow_nan=False, default=str), encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 19: raise
            time.sleep(.05)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def stage(output, name, **details):
    record = {"stage": name, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **details}
    dump(output / "progress.json", record)
    with (output / "stages.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(record, ensure_ascii=False, default=str), flush=True)


def remaining(deadline):
    return max(0.0, deadline - time.monotonic())


def classifier_worker(args):
    """Eq.11 as a convex SDP, including EXACT physical-coordinate regularizer."""
    import numpy as np
    import cvxpy as cp

    output = args.output_dir
    data = np.load(output / "dataset.npz")
    X, y = np.asarray(data["X"], float), np.asarray(data["y"], int)
    mapping = read_json(output / "coordinate_map.json")
    offset = np.asarray(mapping["offset"], float)
    width = np.asarray(mapping["width"], float)
    keep = np.isin(y, [-1, 1])
    X, y = X[keep], y[keep]
    if len(X) < 8 or len(np.unique(y)) != 2:
        raise ValueError("SDP needs at least 8 labeled points and both classes")
    rng = np.random.default_rng(args.seed + 101)
    permutation = rng.permutation(len(y))
    cut = max(1, int(.8 * len(y)))
    train, validation = permutation[:cut], permutation[cut:]
    T = X.shape[1]
    Q = cp.Variable((T, T), symmetric=True)
    q, c = cp.Variable(T), cp.Variable()
    # p = offset + diag(width) x. This change of numerical coordinates leaves
    # the classifier and original regularization on physical Qp,qp unchanged.
    Qp = cp.multiply(1.0 / np.outer(width, width), Q)
    qp = cp.multiply(1.0 / width, q) - 2 * Qp @ offset
    values = cp.sum(cp.multiply(X[train] @ Q, X[train]), axis=1) + X[train] @ q + c
    objective = cp.sum(cp.pos(1 - cp.multiply(y[train], values))) / len(train)
    objective += args.regularization * (cp.sum_squares(Qp) + cp.sum_squares(qp))
    problem = cp.Problem(cp.Minimize(objective), [Q >> 0])
    started = time.monotonic()
    options = {"max_iter": 300, "time_limit": args.classifier_seconds,
               "tol_gap_abs": 1e-7, "tol_feas": 1e-7, "tol_gap_rel": 1e-7}
    problem.solve(solver="CLARABEL", verbose=False, **options)
    if Q.value is None:
        raise RuntimeError(f"No SDP incumbent: {problem.status}")
    Qn = (np.asarray(Q.value) + np.asarray(Q.value).T) / 2
    qn, cn = np.asarray(q.value), float(c.value)
    eig = np.linalg.eigvalsh(Qn)
    # Never silently add a diagonal or clip eigenvalues to manufacture an ellipse.
    physical_Q = Qn / np.outer(width, width)
    physical_q = qn / width - 2 * physical_Q @ offset
    physical_c = float(cn + offset @ physical_Q @ offset - (qn / width) @ offset)
    scores = np.einsum("ni,ij,nj->n", X, Qn, X) + X @ qn + cn
    prediction = np.where(scores <= 0, -1, 1)
    validation_positive = prediction[validation] == -1
    validation_false_accept = validation_positive & (y[validation] == 1)
    info = {"status": problem.status, "solver": "CLARABEL", "objective": float(problem.value),
            "elapsed_seconds": time.monotonic() - started,
            "regularization": args.regularization, "regularization_coordinate": "physical_power",
            "training_points": int(len(train)), "validation_points": int(len(validation)),
            "train_accuracy": float(np.mean(prediction[train] == y[train])),
            "validation_accuracy": float(np.mean(prediction[validation] == y[validation])) if len(validation) else None,
            "validation_predicted_feasible":int(np.sum(validation_positive)),
            "validation_false_feasible":int(np.sum(validation_false_accept)),
            "minimum_eigenvalue_normalized": float(eig.min()),
            "maximum_eigenvalue_normalized": float(eig.max()),
            "strictly_positive_definite": bool(eig.min() > max(1e-10, eig.max()*1e-10)),
            "split": "random_80_20", "psd_violation": float(max(0, -eig.min())),
            "original_data_target": 7244, "actual_labeled_points": int(len(X)),
            "lambda_cross_validated": False,
            "physical_regularizer_equivalence": "exact_affine_coefficient_transformation",
            "intercept_regularized": False}
    np.savez_compressed(output / "ellipsoid.npz", Q=physical_Q, q=physical_q, c=physical_c,
                        Q_normalized=Qn, q_normalized=qn, c_normalized=cn,
                        train_indices=train, validation_indices=validation, scores=scores)
    dump(output / "classifier.json", info)


def geometry_worker(args):
    import numpy as np
    from Simulator.runners import taheri_geometry as geometry
    output = args.output_dir
    deadline = time.monotonic() + max(0.0, args.deadline_unix - time.time() - 3)
    started = time.monotonic()
    A, b, n_aux, metadata = geometry.lifted_ball(24, args.delta)
    metadata.update({"dimension": 24, "auxiliary_variables": n_aux,
                     "lifted_dimension": int(A.shape[1]), "initial_rows": int(A.shape[0]),
                     "geometry_independent_of_training_data": True})
    dump(output / "lifted_geometry.json", metadata)
    np.savez_compressed(output / "lifted_ball.npz", A=A, b=b, n_keep=24)
    result = geometry.fourier_motzkin(A, b, n_keep=24, deadline=deadline,
                                     max_rows=args.max_fm_rows, checkpoint_dir=output / "fm")
    matrix, rhs = result.pop("A", None), result.pop("b", None)
    result["elapsed_seconds"] = time.monotonic() - started
    if matrix is not None and rhs is not None:
        np.savez_compressed(output / "fm_intermediate.npz", A=matrix, b=rhs)
        result["remaining_columns"] = int(matrix.shape[1])
        result["rows"] = int(matrix.shape[0])
        if matrix.shape[1] == 24 and result.get("status") == "complete":
            np.savez_compressed(output / "ball_polytope.npz", E=matrix, d=rhs)
    dump(output / "geometry_result.json", result)


def wait_subprocess(process, deadline):
    while process.poll() is None and remaining(deadline) > 0:
        time.sleep(min(.25, remaining(deadline)))
    if process.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
        else:
            process.kill()
        process.wait(timeout=5)
        return False
    return process.returncode == 0


def run_child(args):
    import numpy as np
    from Simulator.runners.taheri_data import ScenarioOracle
    from Simulator.runners import taheri_geometry as geometry

    output = args.output_dir
    started = time.monotonic()
    deadline = started + max(0, args.deadline_unix - time.time() - 5)
    geometry_log = (output / "geometry.log").open("w", encoding="utf-8")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-geometry",
               "--output-dir", str(output), "--deadline-unix", str(args.deadline_unix),
               "--delta", str(args.delta), "--max-fm-rows", str(args.max_fm_rows)]
    gp = subprocess.Popen(command, stdout=geometry_log, stderr=subprocess.STDOUT)
    try:
        stage(output, "building_physical_oracle")
        oracle = ScenarioOracle(seed=args.case_seed, K=25, epsilon=args.epsilon,
                                noise_scale=.05, conservativeness=.02,
                                device_scale=1, solver_threads=1)
        dump(output / "oracle_metadata.json", oracle.metadata)
        dump(output / "coordinate_map.json", {"offset": oracle.pmin.tolist(),
                                              "width": oracle.width.tolist(),
                                              "normalized": True})
        # A staged run must leave time for the SDP and geometry, rather than spend
        # all 1000 s labeling and discard every usable intermediate model.
        data_seconds = min(args.data_seconds, max(5, .48 * remaining(deadline)))
        data_deadline = min(deadline - 60, time.monotonic() + data_seconds)
        stage(output, "labeling", data_seconds=data_seconds, target_points=7244, K=25)
        try:
            X, y, metadata = oracle.generate(output, data_deadline, target_points=7244,
                                             per_solve_seconds=args.per_solve_seconds,
                                             kappa=.2, progress_callback=None)
        except Exception as error:
            dump(output/"labeling_error.json",{"error":str(error),"traceback":traceback.format_exc()})
            if not (output/"dataset.npz").exists(): raise
            with np.load(output/"dataset.npz") as checkpoint:
                X,y=checkpoint["X"].copy(),checkpoint["y"].copy()
            metadata={**oracle.metadata,"sampling_stopped_on_error":str(error)}
        dump(output / "dataset_result.json", {**metadata,"labeled_points":len(y),
              "feasible":int(np.sum(np.asarray(y)==-1)),"infeasible":int(np.sum(np.asarray(y)==1)),
              "solve_count":oracle.solve_count,"label_attempts":oracle.label_count})
        stage(output, "data_checkpoint", samples=len(X), feasible=int(np.sum(np.asarray(y)==-1)),
              infeasible=int(np.sum(np.asarray(y)==1)), remaining_seconds=remaining(deadline))
        labels = np.asarray(y)
        if len(X) >= 8 and len(np.unique(labels[np.isin(labels, [-1,1])])) == 2 and remaining(deadline) > 30:
            fit_seconds = min(args.classifier_seconds, max(10, remaining(deadline)-30))
            command = [str(args.classifier_python), str(Path(__file__).resolve()), "--worker-classifier",
                       "--output-dir", str(output), "--seed", str(args.seed),
                       "--classifier-seconds", str(fit_seconds),
                       "--regularization", str(args.regularization)]
            stage(output, "fitting_sdp", time_limit=fit_seconds)
            with (output / "classifier.log").open("w", encoding="utf-8") as handle:
                cp = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
                wait_subprocess(cp, min(deadline-20, time.monotonic()+fit_seconds+10))
        else:
            dump(output / "classifier.json", {"status": "not_run_insufficient_labeled_classes_or_time"})
        # The expensive ball projection is independent of the learned ellipse;
        # its exact affine image can be applied after the SDP, without changing S3.
        if gp.poll() is not None and (output / "ball_polytope.npz").exists() and (output / "ellipsoid.npz").exists():
            ellipse = np.load(output / "ellipsoid.npz")
            ball = geometry.ellipsoid_to_ball(ellipse["Q"], ellipse["q"], float(ellipse["c"]))
            polytope = np.load(output / "ball_polytope.npz")
            E, d = polytope["E"], polytope["d"]
            # ellipsoid_to_ball returns y = transform @ p + shift, ||y|| <= radius.
            transform, shift, radius = ball["S"], ball["offset"], ball["radius"]
            Ep, dp = E @ transform / radius, d - E @ shift / radius
            np.savez_compressed(output / "ellipse_polytope.npz", E=Ep, d=dp)
            G = geometry.market_template(24, dt=1.)
            feasible = np.asarray(X)[np.asarray(y)==-1] * oracle.width + oracle.pmin
            stage(output, "fitting_homothet", rows=int(len(dp)))
            fit = geometry.fit_homothet(G, feasible, Ep, dp, deadline=deadline-10)
            arrays = {key: value for key,value in fit.items() if isinstance(value,np.ndarray)}
            if arrays:
                np.savez_compressed(output / "homothet_arrays.npz", **arrays)
            dump(output / "homothet_result.json", {key:value for key,value in fit.items() if key not in arrays})
            certificate_ok=(float(fit.get("certificate_equality_residual",math.inf))<=1e-7 and
                            float(fit.get("certificate_inequality_violation",math.inf))<=1e-7)
            if "A" in arrays and "b" in arrays and certificate_ok:
                np.savez_compressed(output / "final_polytope.npz", A=arrays["A"], b=arrays["b"])
        else:
            stage(output, "geometry_or_sdp_incomplete", geometry=read_json(output / "geometry_result.json").get("status"),
                  classifier=read_json(output / "classifier.json").get("status"))
        if (output / "ellipsoid.npz").exists() and remaining(deadline)>25:
            try:
                ellipse=np.load(output / "ellipsoid.npz")
                ball=geometry.ellipsoid_to_ball(ellipse["Q"],ellipse["q"],float(ellipse["c"]))
                rng=np.random.default_rng(args.seed+5000)
                records=[]
                stage(output,"evaluating_intermediate_ellipsoid")
                eval_deadline=min(deadline-10,time.monotonic()+45)
                for i in range(100):
                    if remaining(eval_deadline)<1: break
                    direction=rng.normal(size=24); direction/=np.linalg.norm(direction)
                    radius=1.0 if i%2==0 else float(rng.random()**(1/24))
                    physical=ball["center"]+np.linalg.solve(ball["S"],direction*ball["radius"]*radius)
                    normalized=(physical-oracle.pmin)/oracle.width
                    label=oracle.label(normalized,eval_deadline,args.per_solve_seconds)
                    records.append({"point_kind":"boundary" if i%2==0 else "interior",
                        "label":label["label"],"successes":label["successes"],
                        "failures":label["failures"],"unknown":label["unknown"],
                        "scenarios_observed":label["scenarios_observed"]})
                    dump(output/"ellipsoid_physical_evaluation.json",{"scope":"intermediate_ellipsoid_not_final_paper_polytope",
                         "epsilon":args.epsilon,"K":25,"points":records,
                         "feasible_points":sum(r["label"]==-1 for r in records),
                         "infeasible_points":sum(r["label"]==1 for r in records),
                         "unknown_points":sum(r["label"]==0 for r in records)})
            except Exception as error:
                dump(output/"ellipsoid_evaluation_error.json",{"error":str(error)})
        # Do not make a certificate from a model with residual auxiliary variables.
        if gp.poll() is None:
            wait_subprocess(gp, deadline-3)
        stage(output, "finished", elapsed_seconds=time.monotonic()-started,
              final_polytope_available=(output / "final_polytope.npz").exists())
    finally:
        if gp.poll() is None:
            gp.kill(); gp.wait(timeout=5)
        geometry_log.close()
        if "oracle" in locals() and hasattr(oracle, "close"):
            oracle.close()


def supervisor(args):
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    args.output_dir = output
    started = time.monotonic()
    wall_started = time.time()
    wall_deadline = min(wall_started + args.budget_seconds,
                        args.deadline_unix if args.deadline_unix else math.inf)
    args.deadline_unix = wall_deadline
    dump(output / "run_config.json", {**vars(args), "output_dir": str(output),
         "wall_start_unix": wall_started, "wall_deadline_unix": wall_deadline,
         "effective_compute_budget_seconds": max(0,wall_deadline-wall_started),
         "paper": "10.1109/TSG.2022.3185532", "K":25, "kappa":.2,
         "T":24, "case_seed":args.case_seed,"noise_scale":.05,"conservativeness":.02,
         "paper_N":7244,"paper_lambda":1e-6,
         "scenario": "current_project_microgrid_7_devices_120_binaries_not_original_200_devices",
         "python":sys.executable})
    hashes = {}
    for name in ("run_taheri2022.py","taheri_data.py","taheri_geometry.py"):
        path = Path(__file__).with_name(name)
        if path.exists(): hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    dump(output / "source_sha256.json",hashes)
    command = [sys.executable,str(Path(__file__).resolve()),"--worker",
               "--output-dir",str(output),"--deadline-unix",str(wall_deadline),
               "--seed",str(args.seed),"--case-seed",str(args.case_seed),
               "--data-seconds",str(args.data_seconds),"--classifier-seconds",str(args.classifier_seconds),
               "--per-solve-seconds",str(args.per_solve_seconds),"--epsilon",str(args.epsilon),
               "--delta",str(args.delta),"--max-fm-rows",str(args.max_fm_rows),
               "--regularization",str(args.regularization)]
    command.extend(["--classifier-python",str(args.classifier_python)])
    with (output / "run.log").open("w",encoding="utf-8") as handle:
        child = subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT)
        # Reserve a few seconds for the final intermediate manifest.
        finished = wait_subprocess(child,started+max(0,wall_deadline-wall_started-3))
    summary = {"status":"returned_within_budget", "worker_completed":finished,
               "worker_returncode":child.returncode,"elapsed_seconds":time.monotonic()-started,
               "effective_budget_seconds":wall_deadline-wall_started,
               "last_stage":read_json(output / "progress.json"),
               "dataset":read_json(output / "dataset_result.json"),
               "labeling":read_json(output / "labeling_progress.json"),
               "classifier":read_json(output / "classifier.json"),
               "geometry":read_json(output / "geometry_result.json"),
               "final_polytope_available":(output / "final_polytope.npz").exists(),
               "files":[str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()]}
    dump(output / "intermediate_result.json",summary)
    print(json.dumps({"status":summary["status"],"elapsed_seconds":summary["elapsed_seconds"],
        "worker_returncode":summary["worker_returncode"],"last_stage":summary["last_stage"],
        "dataset_points":summary["dataset"].get("labeled_points"),
        "classifier_status":summary["classifier"].get("status"),
        "geometry_status":summary["geometry"].get("status"),
        "final_polytope_available":summary["final_polytope_available"]},ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--budget-seconds",type=float,default=1000)
    parser.add_argument("--deadline-unix",type=float)
    parser.add_argument("--data-seconds",type=float,default=300)
    parser.add_argument("--classifier-seconds",type=float,default=120)
    parser.add_argument("--per-solve-seconds",type=float,default=2)
    parser.add_argument("--max-fm-rows",type=int,default=500000)
    parser.add_argument("--epsilon",type=float,default=.04)
    parser.add_argument("--delta",type=float,default=.1)
    parser.add_argument("--regularization",type=float,default=1e-6)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--case-seed",type=int,default=42)
    parser.add_argument("--classifier-python",type=Path,
        default=Path(__file__).resolve().parents[3]/"tmp"/"gradient_review"/"cvx_env"/"Scripts"/"python.exe")
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--worker-classifier",action="store_true")
    parser.add_argument("--worker-geometry",action="store_true")
    args=parser.parse_args()
    try:
        if args.worker_classifier: classifier_worker(args)
        elif args.worker_geometry: geometry_worker(args)
        elif args.worker: run_child(args)
        else: supervisor(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        name="classifier_error.json" if args.worker_classifier else "geometry_error.json" if args.worker_geometry else "error.json"
        dump(args.output_dir/name,{"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()})
        traceback.print_exc()
        sys.exit(1)


if __name__=="__main__":
    main()
