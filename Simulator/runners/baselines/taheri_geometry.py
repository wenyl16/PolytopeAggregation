"""Taheri et al. (2022), geometric stages, with explicit fidelity metadata.

The classifier-to-ball algebra corrects the missing 1/2 and square-root factors
in the accepted manuscript.  The lifted SOC construction is the standard
Ben-Tal--Nemirovski/Glineur rotation-and-folding tree.  Its level schedule uses
Kim--Soltani--Giannakis (2013), equations (13)--(16), the extended version of
Taheri's reference [23]. It does not reproduce the reported 23 auxiliaries.
Callers must retain
the construction metadata and must not claim a byte-for-byte reconstruction.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np


def market_template(T, dt=1.0):
    """Equation (4), power, cumulative energy, and adjacent ramps."""
    eye = np.eye(int(T))
    cumulative = np.tril(np.ones((int(T), int(T)))) * float(dt)
    ramps = np.diff(eye, axis=0)
    return np.vstack((eye, -eye, cumulative, -cumulative, ramps, -ramps))


def ellipsoid_to_ball(Q, q, c, eigen_tolerance=1e-10):
    """Return y=S p+offset, ||y|| <= radius, without silently regularizing Q."""
    Q = np.asarray(Q, dtype=float)
    q = np.asarray(q, dtype=float)
    Q = (Q + Q.T) / 2
    values, vectors = np.linalg.eigh(Q)
    # Relative eigenvalue test is invariant to power units / multiplying the
    # entire classifier by a positive scalar. An absolute floor of 1 here
    # would wrongly reject otherwise identical ellipsoids in kW vs MW.
    scale = max(np.finfo(float).tiny, float(np.max(np.abs(values))))
    if values[0] <= eigen_tolerance * scale:
        raise ValueError("Classifier is not positive definite; bounded ellipsoid unavailable")
    S = (vectors * np.sqrt(values)) @ vectors.T
    inv_q = np.linalg.solve(Q, q)
    center = -0.5 * inv_q
    offset = -S @ center
    radius_squared = float(0.25 * q @ inv_q - c)
    if not math.isfinite(radius_squared) or radius_squared <= 0:
        raise ValueError("Classifier has no nonempty full-dimensional ellipsoid")
    return {"S": S, "offset": offset, "center": center,
            "radius": math.sqrt(radius_squared), "radius_squared": radius_squared,
            "eigenvalues": values, "condition_number": float(values[-1] / values[0]),
            "formula_correction": "center=-0.5 Q^-1 q; r²=0.25 q'Q^-1q-c"}


def lifted_ball(T, delta=0.1):
    """Build an inner lifted polyhedral approximation of the unit T-ball.

    A row represents A @ [y,aux] <= b.  Pairwise cone outputs form a
    balanced norm tree. The level schedule follows equation (16) of
    Kim--Soltani--Giannakis (2013); root radius is 1/(1+delta), as their
    inner approximation equation (17). Thus ball(1/(1+delta)) subset P
    subset ball(1).
    α equalities are substituted symbolically; the final β is eliminated by
    its two affine lower bounds.  This reduces each cone to s-1 auxiliaries.
    """
    T = int(T)
    if T < 2 or not 0 < delta < 1:
        raise ValueError("Require T>=2 and 0<delta<1")
    depth = int(math.ceil(math.log2(T)))
    precision_offset = math.floor(math.log((16/(9*math.pi**2))*math.log1p(delta), 4))
    level_stages = [max(3, math.ceil((ell+1)/2)-precision_offset)
                    for ell in range(1, depth+1)]
    inflation = math.prod(1/math.cos(math.pi/2**s) for s in level_stages)
    gamma = 1/(1+delta)
    if inflation > 1+delta+1e-12:
        raise ArithmeticError("Referenced SOC schedule does not meet requested global delta")
    expressions = [{j: 1.0} for j in range(T)]
    rows = []
    bounds = []
    next_index = T
    cone_count = 0
    output_count = 0
    level_cones = []

    def add(left, right, a=1.0, b=1.0):
        out = {k: a * v for k, v in left.items()}
        for k, v in right.items():
            out[k] = out.get(k, 0.0) + b * v
        return {k: v for k, v in out.items() if abs(v) > 1e-15}

    def fresh():
        nonlocal next_index
        result = {next_index: 1.0}
        next_index += 1
        return result

    def inequality(expr, rhs=0.0):
        rows.append(expr)
        bounds.append(float(rhs))

    def cone(a, b, output, root, stages):
        nonlocal cone_count
        cone_count += 1
        alpha = {k: -v for k, v in a.items()}
        beta = fresh()
        inequality(add(b, beta, 1, -1))
        inequality(add(b, beta, -1, -1))
        for i in range(1, stages):
            angle = math.pi / 2 ** i
            ca, sa = math.cos(angle), math.sin(angle)
            next_alpha = add(alpha, beta, ca, sa)
            fold = add(beta, alpha, ca, -sa)
            if i == stages - 1:
                phi = math.pi / 2 ** stages
                for sign in (-1, 1):
                    terminal = add(next_alpha, fold, math.cos(phi), sign * math.sin(phi))
                    if root:
                        inequality(terminal, gamma)
                    else:
                        inequality(add(terminal, output, 1, -1))
            else:
                next_beta = fresh()
                inequality(add(fold, next_beta, 1, -1))
                inequality(add(fold, next_beta, -1, -1))
                alpha, beta = next_alpha, next_beta
        if not root:
            inequality({k: -v for k, v in output.items()})

    level = 0
    while len(expressions) > 1:
        new = []
        count_this = 0
        is_last = len(expressions) == 2
        for j in range(0, len(expressions) - 1, 2):
            output = {} if is_last else fresh()
            output_count += int(not is_last)
            cone(expressions[j], expressions[j + 1], output, is_last, level_stages[level])
            new.append(output)
            count_this += 1
        if len(expressions) % 2:
            new.append(expressions[-1])
        expressions = new
        level_cones.append(count_this)
        level += 1
    A = np.zeros((len(rows), next_index))
    for i, row in enumerate(rows):
        for j, v in row.items():
            A[i, j] = v
    metadata = {"construction": "BenTalNemirovski_Glineur_balanced_folded_SOC_tree",
        "delta_requested": float(delta), "inner_radius_bound": float(gamma),
        "outer_radius_bound": float(inflation*gamma), "error_bound_verified": bool(inflation <= 1+delta+1e-14),
        "depth": depth, "rotation_stages_by_level": level_stages,
        "cone_count": cone_count, "cone_counts_by_level": level_cones,
        "norm_output_auxiliaries": output_count, "auxiliaries": next_index-T,
        "paper_reported_T8_delta01_auxiliaries": 23,
        "paper_exact_auxiliary_schedule_recovered": False,
        "fidelity_note": "Same cited lifted-SOC family and inner-error guarantee; manuscript's 23-variable schedule not specified/recovered",
        "primary_construction_reference": "https://userpages.cs.umbc.edu/sjkim/papers/KimSoltaniGiannakisTWC13.pdf, equations13-17",
        "schedule_note": "Extended 2013 version of Taheri reference23; exact published T8 23-variable implementation not recovered"}
    return A, np.asarray(bounds), next_index-T, metadata


def lifted_ellipsoid(Q, q, c, delta=0.1):
    ball = ellipsoid_to_ball(Q, q, c)
    A, b, n_aux, metadata = lifted_ball(len(q), delta)
    left = A[:, :len(q)].copy()
    A[:, :len(q)] = left @ ball["S"] / ball["radius"]
    b = b - left @ ball["offset"] / ball["radius"]
    metadata.update(ellipsoid_condition_number=ball["condition_number"],
                    ball_radius=ball["radius"], formula_correction=ball["formula_correction"])
    return A, b, n_aux, metadata


def _deduplicate(A, b, decimals=11, tol=1e-12):
    """Remove positive row duplicates and tautologies, never arbitrary cuts."""
    if not len(b):
        return A, b
    norms = np.max(np.abs(A), axis=1)
    zero = norms <= tol
    if np.any(zero & (b < -tol)):
        raise ValueError("Elimination produced contradiction")
    A, b, norms = A[~zero], b[~zero], norms[~zero]
    A = A / norms[:, None]
    b = b / norms
    # Quantization is used only to identify proportional coefficient vectors;
    # retain the actual floating row with smallest normalized RHS.
    packed = np.round(A, decimals=decimals)
    _, inverse = np.unique(packed, axis=0, return_inverse=True)
    order = np.lexsort((b, inverse))
    picked = order[np.r_[True, inverse[order][1:] != inverse[order][:-1]]]
    return A[picked], b[picked]


def fourier_motzkin(A, b, n_keep, deadline, max_rows=200000,
                   checkpoint_dir=None, zero_tolerance=1e-12):
    """Numerical exact projection, with explicit stop instead of dropping rows.

    deadline is an absolute time.monotonic() deadline.  A stopped checkpoint
    still has remaining auxiliaries and MUST NOT be used as an H-polytope in p.
    """
    A, b = np.asarray(A, dtype=float).copy(), np.asarray(b, dtype=float).copy()
    n_keep = int(n_keep)
    original_columns = list(range(A.shape[1]))
    history = []
    started = time.monotonic()
    status = "running"
    output = None if checkpoint_dir is None else Path(checkpoint_dir)
    if output:
        output.mkdir(parents=True, exist_ok=True)

    def save():
        if output:
            np.savez_compressed(output / "fm_checkpoint.npz", A=A, b=b,
                original_columns=np.asarray(original_columns), n_keep=n_keep)
            (output / "fm_progress.json").write_text(json.dumps({"status": status,
                "history": history, "remaining_auxiliaries": A.shape[1]-n_keep,
                "rows": len(b), "elapsed_seconds": time.monotonic()-started}, indent=2), encoding="utf-8")

    while A.shape[1] > n_keep:
        if time.monotonic() >= deadline:
            status = "deadline_before_elimination"
            break
        counts = []
        for j in range(n_keep, A.shape[1]):
            positive = int(np.count_nonzero(A[:, j] > zero_tolerance))
            negative = int(np.count_nonzero(A[:, j] < -zero_tolerance))
            counts.append((positive * negative + len(b)-positive-negative, j, positive, negative))
        predicted, column, np_, nn_ = min(counts)
        record = {"step": len(history), "original_column": original_columns[column],
            "rows_before": len(b), "positive_rows": np_, "negative_rows": nn_,
            "predicted_raw_rows_after": predicted,
            "remaining_auxiliaries_before": A.shape[1]-n_keep}
        if predicted > int(max_rows):
            record["stopped_before_allocation"] = True
            history.append(record)
            status = "row_guard"
            break
        coefficients = A[:, column]
        pos = np.flatnonzero(coefficients > zero_tolerance)
        neg = np.flatnonzero(coefficients < -zero_tolerance)
        zero = np.flatnonzero(np.abs(coefficients) <= zero_tolerance)
        kept = np.delete(np.arange(A.shape[1]), column)
        C = np.empty((predicted, len(kept)))
        d = np.empty(predicted)
        C[:len(zero)] = A[np.ix_(zero, kept)]
        d[:len(zero)] = b[zero]
        offset = len(zero)
        abandoned = False
        for i in pos:
            if time.monotonic() >= deadline:
                abandoned = True
                break
            length = len(neg)
            C[offset:offset+length] = (A[i, kept] / coefficients[i]
                - A[np.ix_(neg, kept)] / coefficients[neg, None])
            d[offset:offset+length] = b[i] / coefficients[i] - b[neg] / coefficients[neg]
            offset += length
        if abandoned:
            status = "deadline_during_elimination"
            record["partial_step_discarded"] = True
            history.append(record)
            break
        A, b = _deduplicate(C, d, tol=zero_tolerance)
        original_columns.pop(column)
        record["rows_after_deduplication"] = len(b)
        record["elapsed_seconds"] = time.monotonic()-started
        history.append(record)
        save()
    if A.shape[1] == n_keep:
        status = "complete"
    save()
    return {"status": status, "A": A, "b": b, "history": history,
        "remaining_auxiliaries": A.shape[1]-n_keep,
        "original_columns": original_columns, "elapsed_seconds": time.monotonic()-started,
        "numerical_note": "floating-point FM; 1e-12 coefficient zero test and 11-decimal proportional-row keys"}


def fit_homothet(G, feasible, E, d, deadline, threads=2, max_variables=3000000):
    """Equations (19)--(21), without changing the prototype or Farkas LP."""
    import gurobipy as gp
    from gurobipy import GRB

    G, E, d = np.asarray(G), np.asarray(E), np.asarray(d)
    feasible = np.asarray(feasible)
    if feasible.ndim != 2 or not len(feasible):
        raise ValueError("At least one labelled feasible point is needed")
    lower = np.max(feasible @ G.T, axis=0)
    xbar = np.maximum(lower, 0.0)  # exact separable solution of equation (21)
    m, k = len(d), len(G)
    result = {"prototype": xbar, "raw_multiplier_variables": m*k,
        "prototype_formula": "xbar_j=max(0,max_feasible (G p)_j), exact eq21 minimizer"}
    if m*k > max_variables:
        result.update(status="lp_variable_guard", max_variables=max_variables)
        return result
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        result.update(status="deadline_before_lp")
        return result
    with gp.Model("taheri_equation19") as model:
        model.Params.OutputFlag = 0
        model.Params.Threads = int(threads)
        model.Params.TimeLimit = max(0.01, remaining)
        F = model.addMVar((m, k), lb=0, name="F")
        z = model.addMVar(G.shape[1], lb=-GRB.INFINITY, name="z")
        # A 0-dimensional MVar keeps beta*d inside Gurobi's matrix API;
        # scalar Var*d would create a NumPy object array of LinExpr objects.
        beta = model.addMVar((), lb=0, name="beta")
        model.addConstr(F @ G == E)
        model.addConstr(F @ xbar <= E @ z + beta * d)
        model.setObjective(beta, GRB.MINIMIZE)
        model.update()
        model.Params.TimeLimit = max(0.01, deadline-time.monotonic())
        model.optimize()
        result.update(status=int(model.Status), solver_runtime=float(model.Runtime),
            solution_count=int(model.SolCount), num_variables=model.NumVars,
            num_constraints=model.NumConstrs)
        if model.SolCount and beta.X > 1e-12:
            fval, zval, beta_val = F.X, z.X, float(beta.X)
            result.update(beta=beta_val, z=zval, A=G.copy(),
                b=(xbar-G@zval)/beta_val,
                certificate_equality_residual=float(np.max(np.abs(fval@G-E))),
                certificate_inequality_violation=float(np.max(fval@xbar-E@zval-beta_val*d)))
    return result


def geometry_self_check():
    """Independent algebra, projection, and unit-ball containment checks."""
    from scipy.optimize import linprog
    rng = np.random.default_rng(42)
    T = 2
    B = rng.normal(size=(T, T))
    Q = B.T@B + np.eye(T)
    q = rng.normal(size=T)
    c = -2.0
    transform = ellipsoid_to_ball(Q, q, c)
    points = rng.normal(size=(100, T))
    lhs = np.einsum("ni,ij,nj->n", points, Q, points)+points@q+c
    rhs = np.sum((points@transform["S"].T+transform["offset"])**2, axis=1)-transform["radius_squared"]
    A,b,n_aux,metadata = lifted_ball(T, .1)
    fm = fourier_motzkin(A,b,T,time.monotonic()+20,100000)
    if fm["status"] != "complete":
        raise AssertionError("T2 FM did not finish")
    max_norm = 0.0
    for angle in np.linspace(0, 2*math.pi, 64, endpoint=False):
        direction = np.array([math.cos(angle),math.sin(angle)])
        solved = linprog(-direction,A_ub=fm["A"],b_ub=fm["b"],bounds=[(None,None)]*T,method="highs")
        if not solved.success:
            raise AssertionError("T2 projected polytope LP failed")
        max_norm = max(max_norm,float(np.linalg.norm(solved.x)))
    guaranteed_inner = metadata["inner_radius_bound"]
    samples = rng.normal(size=(100, T))
    samples *= guaranteed_inner/np.linalg.norm(samples,axis=1)[:,None]
    inner_violation = float(np.max(samples@fm["A"].T-fm["b"]))
    result = {"completion_square_max_error":float(np.max(np.abs(lhs-rhs))),
        "T2_auxiliaries":n_aux,"T2_projected_rows":len(fm["b"]),
        "T2_max_vertex_norm":max_norm,"guaranteed_inner_ball_violation":inner_violation,
        "market_T24_shape":list(market_template(24).shape)}
    if result["completion_square_max_error"]>1e-10 or max_norm>1+1e-9 or inner_violation>1e-9:
        raise AssertionError(result)
    return result
