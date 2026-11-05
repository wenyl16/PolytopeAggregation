"""Persistent Gurobi implementation of the common directional-distance metric.

The physical model is translated from the actual Pyomo constraints once. Every
direction still performs both support LPs and both projection QPs; no inclusion
assumption is used to replace an error by zero. Only the aggregate coordinates
enter the squared Euclidean distance, exactly as in ``evaluation.py``.
"""

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pyomo.environ as pyo
from pyomo.repn.standard_repn import generate_standard_repn


SOLVER_OPTIONS = {
    "FeasibilityTol": 1e-9, "OptimalityTol": 1e-9, "BarConvTol": 1e-12,
    "Threads": 1, "Seed": 0, "OutputFlag": 0,
}
_VERSION = "native-directional-squared-distance-v1"
RETRY_POLICY = {
    "projection_only": True, "primary_method": "Gurobi default",
    "retry_methods": [1, 0], "require_optimal": True,
    "same_objective_constraints_and_tolerances": True,
}


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _array_hash(value):
    array = np.ascontiguousarray(value, dtype="<f8")
    digest = hashlib.sha256(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _finite_value(value, description):
    value = float(pyo.value(value))
    if not np.isfinite(value):
        raise ValueError(f"Nonfinite {description}")
    return value


def _translate_pyomo(pyomo_model, env):
    """Translate every active linear row, variable bound and fixed variable.

    Fixed variables may be folded into row constants by Pyomo's standard
    representation. They are also retained with equal native bounds, so their
    values remain available if an aggregate variable happens to be fixed.
    Integer variables and nonlinear rows are rejected rather than relaxed.
    """
    import gurobipy as gp

    model = gp.Model("original_device_model", env=env)
    variables = list(pyomo_model.component_data_objects(pyo.Var, active=True))
    mapping, native_variables, variable_records = {}, [], []
    for variable in variables:
        if not variable.is_continuous():
            raise ValueError(f"Native evaluation requires continuous variables: {variable.name}")
        lower = None if variable.lb is None else _finite_value(variable.lb, variable.name + " lower bound")
        upper = None if variable.ub is None else _finite_value(variable.ub, variable.name + " upper bound")
        if variable.fixed:
            lower = upper = _finite_value(variable.value, variable.name + " fixed value")
        native = model.addVar(lb=-gp.GRB.INFINITY if lower is None else lower,
                              ub=gp.GRB.INFINITY if upper is None else upper,
                              name=variable.name)
        mapping[id(variable)] = (len(native_variables), native)
        native_variables.append(native)
        variable_records.append((variable.name, lower, upper, bool(variable.fixed)))
    row_records = []
    for constraint in pyomo_model.component_data_objects(pyo.Constraint, active=True):
        repn = generate_standard_repn(constraint.body, compute_values=True)
        if not repn.is_linear():
            raise ValueError(f"Nonlinear physical constraint: {constraint.name}")
        constant = _finite_value(repn.constant or 0.0, constraint.name + " constant")
        indices, coefficients, row_variables = [], [], []
        for variable, coefficient in zip(repn.linear_vars, repn.linear_coefs):
            if id(variable) not in mapping:
                raise ValueError(f"Unmapped physical variable: {variable.name}")
            index, native = mapping[id(variable)]
            indices.append(index)
            coefficients.append(_finite_value(coefficient, constraint.name + " coefficient"))
            row_variables.append(native)
        expression = gp.LinExpr(coefficients, row_variables)
        if constraint.equality:
            rhs = _finite_value(constraint.lower, constraint.name + " equality") - constant
            model.addLConstr(expression, gp.GRB.EQUAL, rhs, name=constraint.name)
            row_records.append((constraint.name, indices, coefficients, "=", rhs))
        else:
            if constraint.has_lb():
                rhs = _finite_value(constraint.lower, constraint.name + " lower bound") - constant
                model.addLConstr(expression, gp.GRB.GREATER_EQUAL, rhs, name=constraint.name + "_lb")
                row_records.append((constraint.name, indices, coefficients, ">=", rhs))
            if constraint.has_ub():
                rhs = _finite_value(constraint.upper, constraint.name + " upper bound") - constant
                model.addLConstr(expression, gp.GRB.LESS_EQUAL, rhs, name=constraint.name + "_ub")
                row_records.append((constraint.name, indices, coefficients, "<=", rhs))
    aggregate = [mapping[id(variable)][1] for variable in pyomo_model.var_proj.values()]
    aggregate_indices = [mapping[id(variable)][0] for variable in pyomo_model.var_proj.values()]
    model.update()
    signature = _json_hash({"variables": variable_records, "rows": row_records,
                            "aggregate_indices": aggregate_indices})
    return _NativeModel(model, aggregate), signature


def _build_approximation(dim, region, env):
    import gurobipy as gp

    model = gp.Model("approximate_region", env=env)
    aggregate = list(model.addVars(dim, lb=-gp.GRB.INFINITY, name="var_proj").values())
    if "V" in region:
        if set(region) != {"V"}:
            raise ValueError("Supply either A/b or V, not both")
        vertices = np.asarray(region["V"], dtype=float)
        if (vertices.ndim != 2 or vertices.shape[1] != dim or not len(vertices)
                or not np.all(np.isfinite(vertices))):
            raise ValueError("V must be a finite nonempty (vertices, dim) matrix")
        alpha = list(model.addVars(len(vertices), lb=0.0, name="alpha").values())
        model.addLConstr(gp.LinExpr([1.0] * len(alpha), alpha), gp.GRB.EQUAL, 1.0,
                         name="alpha_sum")
        for column in range(dim):
            model.addLConstr(gp.LinExpr(vertices[:, column].tolist(), alpha) - aggregate[column],
                             gp.GRB.EQUAL, 0.0, name=f"vertex_projection_{column}")
        signature = _json_hash({"V": _array_hash(vertices)})
    else:
        if set(region) != {"A", "b"}:
            raise ValueError("A and b are required for an inequality representation")
        matrix, bounds = np.asarray(region["A"], dtype=float), np.asarray(region["b"], dtype=float)
        if (matrix.ndim != 2 or matrix.shape[1] != dim or bounds.shape != (len(matrix),)
                or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(bounds))):
            raise ValueError("A/b must be finite arrays with shapes (rows, dim)/(rows,)")
        for index, (row, bound) in enumerate(zip(matrix, bounds)):
            model.addLConstr(gp.LinExpr(row.tolist(), aggregate), gp.GRB.LESS_EQUAL,
                             float(bound), name=f"region_{index}")
        signature = _json_hash({"A": _array_hash(matrix), "b": _array_hash(bounds)})
    model.update()
    return _NativeModel(model, aggregate), signature


class _NativeModel:
    def __init__(self, model, aggregate):
        import gurobipy as gp

        self.model, self.aggregate = model, aggregate
        self.quadratic = gp.QuadExpr()
        for variable in aggregate:
            self.quadratic.add(variable * variable)
        self.solves, self.solver_seconds = 0, 0.0
        self.retry_count, self.retry_records = 0, []

    def point(self, vector, projection=False, context=None):
        import gurobipy as gp

        vector = np.asarray(vector, dtype=float)
        if vector.shape != (len(self.aggregate),) or not np.all(np.isfinite(vector)):
            raise ValueError("Direction/target must be a finite vector with length dim")
        if projection:
            objective = self.quadratic + gp.LinExpr((-2.0 * vector).tolist(), self.aggregate)
            # The constant is kept to represent the exact same squared norm.
            objective += float(vector @ vector)
        else:
            objective = gp.LinExpr(vector.tolist(), self.aggregate)
        self.model.setObjective(objective, gp.GRB.MINIMIZE)
        # The legacy Pyomo LP/QP writer creates a fresh solver problem each time.
        # Clear prior basis/solution information while retaining the native model.
        def optimize():
            self.model.reset()
            self.model.optimize()
            self.solves += 1
            self.solver_seconds += float(self.model.Runtime)

        optimize()
        if projection and self.model.Status != gp.GRB.OPTIMAL:
            # A convex projection can occasionally stop with SUBOPTIMAL under
            # automatic barrier selection. Re-solve the identical QP with
            # simplex; neither its constraints nor tolerances are relaxed.
            # The projected aggregate is unique, unlike a support LP vertex.
            record = {"initial_status": int(self.model.Status), "attempts": []}
            if context is not None:
                record["context"] = context
            original_method = self.model.Params.Method
            try:
                for method in RETRY_POLICY["retry_methods"]:
                    self.model.setParam("Method", method)
                    self.retry_count += 1
                    optimize()
                    record["attempts"].append({"method": method, "status": int(self.model.Status)})
                    if self.model.Status == gp.GRB.OPTIMAL:
                        break
            finally:
                self.model.setParam("Method", original_method)
                self.retry_records.append(record)
        if self.model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"Native {'projection' if projection else 'support'} solve "
                               f"failed: Gurobi status {self.model.Status}")
        point = np.asarray(self.model.getAttr("X", self.aggregate), dtype=float)
        if not np.all(np.isfinite(point)):
            raise RuntimeError("Nonfinite aggregate point returned by native Gurobi")
        return point


def _save_checkpoint(path, manifest, feasibility, optimality, completed, elapsed, solver_seconds,
                     solves=0, retry_count=0, retry_records=None):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, manifest=json.dumps(manifest, sort_keys=True), completed=completed,
                 error_feas=feasibility[:completed], error_opt=optimality[:completed],
                 elapsed_seconds=elapsed, solver_seconds=solver_seconds,
                 solves=solves, retry_count=retry_count,
                 retry_policy=json.dumps(RETRY_POLICY, sort_keys=True),
                 retry_records=json.dumps(retry_records or [], sort_keys=True))
    os.replace(temporary, path)


def evaluate_native(agg, directions, region, progress=None, checkpoint_path=None):
    """Evaluate every direction using persistent native LP/QP models.

    ``progress`` receives a dictionary at setup and every 100 completed samples.
    A checkpoint can resume only with bit-identical directions/region and an
    identical translated physical model, solver version and solver parameters.
    Returned metric keys match ``evaluate_calculator``; ``native_*`` scalar keys
    report reproducibility signatures, model sizes and elapsed/solver time.
    """
    import gurobipy as gp

    started = time.perf_counter()
    directions = np.asarray(directions, dtype=float)
    if (directions.ndim != 2 or directions.shape[1] != agg.T or not len(directions)
            or not np.all(np.isfinite(directions))):
        raise ValueError("directions must be a finite nonempty (samples, dim) matrix")
    feasibility = np.empty(len(directions), dtype=float)
    optimality = np.empty(len(directions), dtype=float)
    completed, old_elapsed, old_solver_seconds = 0, 0.0, 0.0
    old_solves, old_retry_count, old_retry_records = 0, 0, []
    with gp.Env(empty=True) as env:
        for key, value in SOLVER_OPTIONS.items():
            env.setParam(key, value)
        env.start()
        original, physical_hash = _translate_pyomo(agg.case_aggregator(build_only=True), env)
        approximation, region_hash = _build_approximation(agg.T, region, env)
        try:
            manifest = {"version": _VERSION, "directions_sha256": _array_hash(directions),
                        "region_sha256": region_hash, "model_sha256": physical_hash,
                        "solver_options": SOLVER_OPTIONS, "gurobi_version": list(gp.gurobi.version())}
            if checkpoint_path is not None and Path(checkpoint_path).is_file():
                with np.load(checkpoint_path, allow_pickle=False) as saved:
                    if json.loads(str(saved["manifest"].item())) != manifest:
                        raise ValueError("Checkpoint directions, region, physical model or solver differs")
                    completed = int(saved["completed"].item())
                    if not 0 <= completed <= len(directions):
                        raise ValueError("Invalid checkpoint completed count")
                    for key, target in (("error_feas", feasibility), ("error_opt", optimality)):
                        values = saved[key]
                        if values.shape != (completed,) or not np.all(np.isfinite(values)) or np.any(values < 0):
                            raise ValueError(f"Invalid checkpoint {key}")
                        target[:completed] = values
                    old_elapsed = float(saved["elapsed_seconds"].item())
                    old_solver_seconds = float(saved["solver_seconds"].item())
                    old_solves = int(saved["solves"].item()) if "solves" in saved else 4 * completed
                    old_retry_count = int(saved["retry_count"].item()) if "retry_count" in saved else 0
                    old_retry_records = json.loads(str(saved["retry_records"].item())) if "retry_records" in saved else []
                    if (not np.isfinite(old_elapsed) or old_elapsed < 0
                            or not np.isfinite(old_solver_seconds) or old_solver_seconds < 0
                            or old_solves < 4 * completed or old_retry_count < 0
                            or not isinstance(old_retry_records, list)):
                        raise ValueError("Invalid checkpoint timing")
            sizes = {"original_variables": original.model.NumVars,
                     "original_rows": original.model.NumConstrs,
                     "approximation_variables": approximation.model.NumVars,
                     "approximation_rows": approximation.model.NumConstrs}
            build_seconds = time.perf_counter() - started

            def report(event):
                if progress is not None:
                    progress({"event": event, "completed": completed, "total": len(directions),
                              "elapsed_seconds": old_elapsed + time.perf_counter() - started,
                              "solver_seconds": old_solver_seconds + original.solver_seconds + approximation.solver_seconds,
                              "retry_count": old_retry_count + original.retry_count + approximation.retry_count,
                              "build_seconds": build_seconds, **sizes})

            def save_checkpoint():
                records = old_retry_records + [
                    {"model": name, **record}
                    for name, native in (("original", original), ("approximation", approximation))
                    for record in native.retry_records]
                _save_checkpoint(checkpoint_path, manifest, feasibility, optimality, completed,
                                 old_elapsed + time.perf_counter() - started,
                                 old_solver_seconds + original.solver_seconds + approximation.solver_seconds,
                                 old_solves + original.solves + approximation.solves,
                                 old_retry_count + original.retry_count + approximation.retry_count, records)

            report("ready")
            try:
                for index in range(completed, len(directions)):
                    direction = directions[index]
                    x_approximation = approximation.point(direction)
                    x_original = original.point(x_approximation, projection=True,
                                                context={"direction_index": index, "metric": "feasibility"})
                    feasibility[index] = float(np.sum((x_approximation - x_original) ** 2))
                    x_original = original.point(direction)
                    x_approximation = approximation.point(x_original, projection=True,
                                                          context={"direction_index": index, "metric": "optimality"})
                    optimality[index] = float(np.sum((x_approximation - x_original) ** 2))
                    if not np.isfinite(feasibility[index]) or not np.isfinite(optimality[index]):
                        raise RuntimeError(f"Nonfinite error at direction {index}")
                    completed = index + 1
                    if completed % 100 == 0 or completed == len(directions):
                        save_checkpoint()
                        report("progress" if completed < len(directions) else "completed")
            except BaseException:
                save_checkpoint()
                raise
            return {
                "error_feas": feasibility, "error_opt": optimality,
                "mean_feas": float(feasibility.mean()), "mean_opt": float(optimality.mean()),
                "max_feas": float(feasibility.max()), "max_opt": float(optimality.max()),
                "count_feas_gt_1e4": int(np.count_nonzero(feasibility > 1e-4)),
                "num_sample": len(directions), "direction_seed": None,
                "native_original_variables": sizes["original_variables"],
                "native_original_rows": sizes["original_rows"],
                "native_approximation_variables": sizes["approximation_variables"],
                "native_approximation_rows": sizes["approximation_rows"],
                "native_build_seconds": build_seconds,
                "native_solver_seconds": old_solver_seconds + original.solver_seconds + approximation.solver_seconds,
                "native_elapsed_seconds": old_elapsed + time.perf_counter() - started,
                "native_solves": old_solves + original.solves + approximation.solves,
                "native_successful_solves": 4 * len(directions),
                "native_retry_count": old_retry_count + original.retry_count + approximation.retry_count,
                "native_retry_policy": json.dumps(RETRY_POLICY, sort_keys=True),
                "native_retry_records": json.dumps(old_retry_records + [
                    {"model": name, **record}
                    for name, native in (("original", original), ("approximation", approximation))
                    for record in native.retry_records], sort_keys=True),
                "native_directions_sha256": manifest["directions_sha256"],
                "native_region_sha256": region_hash, "native_model_sha256": physical_hash,
            }
        finally:
            original.model.dispose()
            approximation.model.dispose()


def evaluate_native_costs(agg, coefficients, regions, progress=None):
    """Minimize shared physical linear costs over the original and saved regions.

    ``coefficients`` is a finite, nonempty ``(samples, agg.T)`` matrix. Each
    row already includes the desired sign, currency/power conversion and slot
    duration; it is never rescaled or clipped here. For net generation in kW,
    a price in EUR/MWh gives ``c = -price * deltaT / 1000`` and an objective in
    EUR. ``regions`` maps ordered method names to either ``{"A", "b"}`` or
    ``{"V"}`` dictionaries in the common normalized power coordinates.

    Returns ``costs`` (ordered method-to-vector dictionaries, starting with
    ``original``), ``dispatch_normalized`` (the corresponding samples-by-time
    optimal aggregate points), and JSON-serializable ``diagnostics``. Models
    are built once, and every LP must terminate OPTIMAL. Progress receives a
    ``ready`` event and one ``method_completed`` event per model.
    """
    import gurobipy as gp

    started = time.perf_counter()
    coefficients = np.asarray(coefficients, dtype=float)
    if (coefficients.ndim != 2 or coefficients.shape[1] != agg.T
            or not len(coefficients) or not np.all(np.isfinite(coefficients))):
        raise ValueError("coefficients must be a finite nonempty (samples, agg.T) matrix")
    if not hasattr(regions, "items"):
        raise ValueError("regions must map ordered method names to H/V dictionaries")
    if any(not isinstance(name, str) or not name or name == "original" for name in regions):
        raise ValueError("Region names must be nonempty strings other than 'original'")
    costs, dispatch, natives, region_hashes = {}, {}, {}, {}
    with gp.Env(empty=True) as env:
        for key, value in SOLVER_OPTIONS.items():
            env.setParam(key, value)
        env.start()
        try:
            natives["original"], physical_hash = _translate_pyomo(
                agg.case_aggregator(build_only=True), env)
            scale = np.asarray(agg.pmax, dtype=float) - np.asarray(agg.pmin, dtype=float)
            offset = np.asarray(agg.pmin, dtype=float)
            if (scale.shape != (agg.T,) or offset.shape != (agg.T,)
                    or not np.all(np.isfinite(scale)) or not np.all(np.isfinite(offset))
                    or np.any(scale <= 0)):
                raise ValueError("Physical normalization must have finite pmin and strictly positive pmax-pmin")
            weights = coefficients * scale[None, :]
            if not np.all(np.isfinite(weights)):
                raise ValueError("Nonfinite normalized objective coefficients")
            for name, region in regions.items():
                natives[name], region_hashes[name] = _build_approximation(agg.T, region, env)
            build_seconds = time.perf_counter() - started
            diagnostics = {
                "version": "native-physical-linear-cost-v1",
                "num_samples": len(coefficients), "horizon": agg.T,
                "method_order": list(natives),
                "coefficients_sha256": _array_hash(coefficients),
                "model_sha256": physical_hash, "region_sha256": region_hashes,
                "scale_sha256": _array_hash(scale), "offset_sha256": _array_hash(offset),
                "physical_coordinates": "normalized_power * (pmax-pmin) + pmin",
                "objective": "min coefficients @ physical_power; coefficients include units and slot duration",
                "solver_options": dict(SOLVER_OPTIONS),
                "gurobi_version": list(gp.gurobi.version()),
                "require_optimal": True, "build_seconds": build_seconds, "models": {},
            }
            if progress is not None:
                progress({"event": "ready", "num_samples": len(coefficients),
                          "method_order": list(natives), "build_seconds": build_seconds})
            for name, native in natives.items():
                method_started = time.perf_counter()
                points = np.empty_like(coefficients)
                values = np.empty(len(coefficients), dtype=float)
                for index, (c, weight) in enumerate(zip(coefficients, weights)):
                    try:
                        point = native.point(weight)
                    except RuntimeError as exc:
                        raise RuntimeError(f"Native cost solve failed for {name}, sample {index}") from exc
                    points[index] = point
                    values[index] = float(c @ (point * scale + offset))
                    if not np.isfinite(values[index]):
                        raise RuntimeError(f"Nonfinite physical cost for {name}, sample {index}")
                costs[name], dispatch[name] = values, points
                record = {
                    "variables": native.model.NumVars, "rows": native.model.NumConstrs,
                    "solves": native.solves, "successful_solves": len(coefficients),
                    "solver_seconds": native.solver_seconds,
                    "elapsed_seconds": time.perf_counter() - method_started,
                    "termination_status": "OPTIMAL",
                }
                diagnostics["models"][name] = record
                print(f"Native costs: {name}: {len(coefficients)}/{len(coefficients)} samples", flush=True)
                if progress is not None:
                    progress({"event": "method_completed", "method": name,
                              "completed": len(coefficients), "total": len(coefficients), **record})
            diagnostics.update(
                elapsed_seconds=time.perf_counter() - started,
                solver_seconds=sum(native.solver_seconds for native in natives.values()),
                solves=sum(native.solves for native in natives.values()),
                successful_solves=len(coefficients) * len(natives),
            )
            return {"costs": costs, "dispatch_normalized": dispatch, "diagnostics": diagnostics}
        finally:
            for native in natives.values():
                native.model.dispose()
