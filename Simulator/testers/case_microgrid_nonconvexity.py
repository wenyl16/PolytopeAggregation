"""Microgrid nonconvexity dataset: evaluation, ablation, checks and summaries.

Commands: evaluate, ablation, verify, verify-storage, summarize, scale3-summary,
and ablation-summary. Help and saved-data statistics do not require Gurobi.
Actual support/projection solves require gurobipy and a working Gurobi setup.

The physical domain, deterministic probes, solver tolerances, bound-aware
statistics and independent constraint checks are retained from the original
six nonconvexity modules. Saved metadata and historical source hashes are not
rewritten. New-run provenance hashes this complete case module and its device
model. Strict verification may reject an older run whose source fingerprint
does not match the current implementation; historical snapshots retain those
original sources. No compatibility modules or embedded source loaders are used.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import importlib
import json
import math
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
from pyomo.repn.standard_repn import generate_standard_repn
from scipy.sparse import coo_matrix

from Simulator.paths import DATA_ROOT, RESULTS_ROOT, comparison_output_path, resolve_recorded_code_path

gp = None


def _require_gurobi():
    """Load the solver only for actual domain construction or solver commands."""
    global gp
    if gp is None:
        try:
            gp = importlib.import_module("gurobipy")
        except ModuleNotFoundError as exc:
            if exc.name != "gurobipy":
                raise
            raise RuntimeError("Nonconvexity evaluation and ablation require gurobipy. "
                               "Help and saved-data summary commands remain available without it.") from exc
    return gp


# Persistent physical-domain queries
DEFAULT_OPTIONS = {
    "FeasibilityTol": 1e-8,
    "IntFeasTol": 1e-9,
    "OptimalityTol": 1e-9,
    "MIPGap": 0.0,
    "MIPGapAbs": 1e-10,
    "Threads": 1,
    "Seed": 0,
    "OutputFlag": 0,
}


def _finite_attribute(model, name, integer=False):
    try:
        value = float(getattr(model, name))
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return int(value) if integer else value


def _status_name(status):
    if status is None:
        return "NOT_SOLVED"
    for name in (
        "LOADED", "OPTIMAL", "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED",
        "CUTOFF", "ITERATION_LIMIT", "NODE_LIMIT", "TIME_LIMIT",
        "SOLUTION_LIMIT", "INTERRUPTED", "NUMERIC", "SUBOPTIMAL",
        "INPROGRESS", "USER_OBJ_LIMIT", "WORK_LIMIT", "MEM_LIMIT",
    ):
        if getattr(gp.GRB, name, None) == status:
            return name
    return str(status)


class PhysicalDomain:
    """Support MILPs and globally bounded projection MIQPs of one fixed region.

    ``fixed_binary`` is a 0/1 vector in ``binary_indices`` order and is used
    only by ``support``.  Its bounds are restored in a ``finally`` block.
    ``start`` is a full physical primal vector in ``names`` order.

    ``TimeLimit`` (default 30 seconds) and ``RetryTimeLimit`` (default 300)
    in ``options`` control projection's at-most-two attempts.  The latter is
    a helper option, not a Gurobi parameter.  Support makes one attempt and
    may return a checked feasible incumbent without asserting optimality.
    """

    def __init__(self, source, options=None):
        _require_gurobi()
        supplied = dict(options or {})
        self.initial_time_limit = float(supplied.pop("TimeLimit", 30.0))
        self.retry_time_limit = float(supplied.pop("RetryTimeLimit", 300.0))
        if not all(math.isfinite(t) and t > 0 for t in
                   (self.initial_time_limit, self.retry_time_limit)):
            raise ValueError("Query time limits must be finite and positive")
        self.options = {**DEFAULT_OPTIONS, **supplied}
        self.feasibility_tolerance = float(self.options["FeasibilityTol"])
        self.integrality_tolerance = float(self.options["IntFeasTol"])
        self.absolute_gap_tolerance = float(self.options["MIPGapAbs"])
        self.bound_consistency_tolerance = max(self.absolute_gap_tolerance, 1e-10)
        self._closed = False
        self.bridge = None
        self.model = None
        self.source = source.clone()
        self.query_count = 0
        self.attempt_count = 0
        self.total_runtime = 0.0
        self.total_wall_seconds = 0.0

        try:
            for objective in self.source.component_data_objects(pyo.Objective, active=True):
                objective.deactivate()
            # Pyomo's persistent plugin can otherwise introduce native range
            # slack variables.  Split the exact same ranged rows explicitly.
            split = pyo.ConstraintList()
            component_name = "_nonconvexity_range_sides"
            while self.source.component(component_name) is not None:
                component_name += "_"
            self.source.add_component(component_name, split)
            rows = list(self.source.component_data_objects(pyo.Constraint, active=True))
            for row in rows:
                if not generate_standard_repn(row.body, compute_values=True).is_linear():
                    raise ValueError(f"PhysicalDomain requires linear physical constraints: {row.name}")
                if row.has_lb() and row.has_ub() and not row.equality:
                    split.add((row.lower, row.body, None))
                    split.add((None, row.body, row.upper))
                    row.deactivate()
            self.bridge = pyo.SolverFactory("gurobi_persistent")
            self.bridge.set_instance(self.source)
            self.model = self.bridge._solver_model
            for name, value in self.options.items():
                self.model.setParam(name, value)
            self.model.update()
            if self.model.NumQConstrs:
                raise ValueError("A nonlinear physical constraint was not translated as a linear row")

            self._variables = list(self.model.getVars())
            self._base_variable_count = len(self._variables)
            self.shape = (self._base_variable_count, int(self.model.NumConstrs))
            mapping = self.bridge._pyomo_var_to_solver_var_map
            names = [variable.VarName for variable in self._variables]
            for variable, native in mapping.items():
                names[native.index] = variable.name
            self.names = tuple(names)
            types = np.asarray(self.model.getAttr("VType", self._variables))
            if not np.isin(types, ["C", "B", "I"]).all():
                raise ValueError("Unsupported semicontinuous/semi-integer physical variable")
            self.binary_indices = np.flatnonzero(types == "B")
            self.integer_indices = np.flatnonzero(np.isin(types, ["B", "I"]))
            self.x_indices = np.asarray(
                [mapping[self.source.var_proj[j]].index for j in range(len(self.source.var_proj))],
                dtype=int,
            )
            self._x = [self._variables[int(index)] for index in self.x_indices]
            self._lower = np.asarray(self.model.getAttr("LB", self._variables), dtype=float)
            self._upper = np.asarray(self.model.getAttr("UB", self._variables), dtype=float)
            self._matrix = self.model.getA().tocsr()
            self._rhs = np.asarray(self.model.getAttr("RHS", self.model.getConstrs()), dtype=float)
            self._senses = np.asarray(self.model.getAttr("Sense", self.model.getConstrs()))

            # These equations are a harmless extended formulation: for every
            # original x there is exactly one free residual r=x-z.
            self._residuals = [self.model.addVar(lb=-gp.GRB.INFINITY,
                name=f"_nonconvexity_residual_{j}") for j in range(len(self._x))]
            self.model.update()
            self._residual_rows = [self.model.addConstr(x - residual == 0.0,
                name=f"_nonconvexity_residual_definition_{j}")
                for j, (x, residual) in enumerate(zip(self._x, self._residuals))]
            self._quadratic = gp.quicksum(residual * residual for residual in self._residuals)
            self.model.update()
            self.query_shape = (int(self.model.NumVars), int(self.model.NumConstrs))
        except Exception:
            self.close()
            raise

    def _check_open(self):
        if self._closed:
            raise RuntimeError("PhysicalDomain is closed")

    def x_from_primal(self, primal):
        array = np.asarray(primal, dtype=float)
        if array.shape != (self._base_variable_count,) or not np.isfinite(array).all():
            raise ValueError("Expected a finite complete physical primal vector")
        return array[self.x_indices].copy()

    def validate_primal(self, primal):
        """Check the frozen ORIGINAL rows, bounds, and integer domains."""
        try:
            array = np.asarray(primal, dtype=float)
        except Exception as exc:
            return {"valid": False, "failure": f"Invalid primal array: {exc}"}
        if array.shape != (self._base_variable_count,) or not np.isfinite(array).all():
            return {"valid": False, "failure": "Nonfinite or incorrectly shaped physical primal",
                    "expected_variables": self._base_variable_count, "actual_shape": list(array.shape)}
        activity = self._matrix @ array
        if not np.isfinite(activity).all():
            return {"valid": False, "failure": "Nonfinite original constraint activity"}
        violation = np.zeros(len(self._rhs))
        less = self._senses == "<"
        greater = self._senses == ">"
        equal = self._senses == "="
        violation[less] = np.maximum(activity[less] - self._rhs[less], 0.0)
        violation[greater] = np.maximum(self._rhs[greater] - activity[greater], 0.0)
        violation[equal] = np.abs(activity[equal] - self._rhs[equal])
        row_violation = float(np.max(violation, initial=0.0))
        bound_violation = float(max(0.0, np.max(self._lower - array, initial=0.0),
                                    np.max(array - self._upper, initial=0.0)))
        integrality = float(np.max(np.abs(array[self.integer_indices] -
                                          np.rint(array[self.integer_indices])), initial=0.0))
        return {"valid": bool(row_violation <= self.feasibility_tolerance and
                              bound_violation <= self.feasibility_tolerance and
                              integrality <= self.integrality_tolerance),
                "max_constraint_violation": row_violation,
                "max_bound_violation": bound_violation,
                "max_integrality_violation": integrality,
                "max_violation": max(row_violation, bound_violation, integrality),
                "feasibility_tolerance": self.feasibility_tolerance,
                "integrality_tolerance": self.integrality_tolerance}

    def _start_candidate(self, start, vector, projection, fixed_binary=None):
        if start is None:
            return None, {"valid": False, "provided": False}
        validation = self.validate_primal(start)
        validation["provided"] = True
        if validation["valid"] and fixed_binary is not None:
            delta = float(np.max(np.abs(np.asarray(start)[self.binary_indices] - fixed_binary), initial=0.0))
            validation["fixed_binary_violation"] = delta
            validation["valid"] = bool(delta <= self.feasibility_tolerance)
        if not validation["valid"]:
            return None, validation
        primal = np.asarray(start, dtype=float).copy()
        point = self.x_from_primal(primal)
        value = float(np.sum((point - vector) ** 2) if projection else vector @ point)
        return {"point": point, "primal": primal, "value": value,
                "validation": validation, "source": "validated_warm_start"}, validation

    def _prepare_objective(self, vector, projection):
        self.model.setAttr("RHS", self._residual_rows,
                           vector.tolist() if projection else [0.0] * len(self._x))
        objective = self._quadratic if projection else gp.LinExpr(vector.tolist(), self._x)
        self.model.setObjective(objective, gp.GRB.MINIMIZE if projection else gp.GRB.MAXIMIZE)
        self.model.update()

    def _attempt(self, vector, projection, limit, warm, fixed_binary=None):
        started = time.perf_counter()
        diagnostics = {"status": None, "status_name": "NOT_SOLVED", "runtime": 0.0,
            "objective": None, "best_bound": None, "recomputed_objective": None,
            "time_limit_seconds": limit, "solution_count": 0,
            "validation": {"valid": False}, "is_mip": bool(self.model.IsMIP)}
        candidate = None
        try:
            self.model.reset(0)
            self.model.NumStart = 0
            if warm is not None:
                self.model.NumStart = 1
                self.model.Params.StartNumber = 0
                self.model.setAttr("Start", self._variables, warm["primal"].tolist())
                residual = warm["point"] - vector if projection else warm["point"]
                self.model.setAttr("Start", self._residuals, residual.tolist())
            self.model.setParam("TimeLimit", limit)
            self.model.optimize()
            status = int(self.model.Status)
            diagnostics.update(status=status, status_name=_status_name(status),
                runtime=float(self.model.Runtime), solution_count=int(self.model.SolCount),
                best_bound=(_finite_attribute(self.model, "ObjBound") if self.model.IsMIP else None),
                node_count=_finite_attribute(self.model, "NodeCount"))
            if self.model.SolCount:
                diagnostics.update(objective=_finite_attribute(self.model, "ObjVal"),
                    mip_gap=(_finite_attribute(self.model, "MIPGap") if self.model.IsMIP else None),
                    native_constraint_violation=_finite_attribute(self.model, "ConstrVio"),
                    native_bound_violation=_finite_attribute(self.model, "BoundVio"),
                    native_integer_violation=(_finite_attribute(self.model, "IntVio") if self.model.IsMIP else 0.0))
                primal = np.asarray(self.model.getAttr("X", self._variables), dtype=float)
                validation = self.validate_primal(primal)
                if validation["valid"] and fixed_binary is not None:
                    delta = float(np.max(np.abs(primal[self.binary_indices] - fixed_binary), initial=0.0))
                    validation["fixed_binary_violation"] = delta
                    validation["valid"] = bool(delta <= self.feasibility_tolerance)
                diagnostics["validation"] = validation
                if validation["valid"]:
                    point = self.x_from_primal(primal)
                    value = float(np.sum((point - vector) ** 2) if projection else vector @ point)
                    diagnostics["recomputed_objective"] = value
                    diagnostics["objective_recomputation_difference"] = (
                        abs(value - diagnostics["objective"]) if diagnostics["objective"] is not None else None)
                    candidate = {"point": point, "primal": primal, "value": value,
                                 "validation": validation, "source": "solver_incumbent"}
                if not self.model.IsMIP and status == gp.GRB.OPTIMAL:
                    diagnostics["best_bound"] = diagnostics["objective"]
        except Exception as exc:
            diagnostics["failure"] = f"{type(exc).__name__}: {exc}"
            diagnostics["status_name"] = "EXCEPTION"
        finally:
            diagnostics["wall_seconds"] = time.perf_counter() - started
            self.attempt_count += 1
            self.total_runtime += diagnostics["runtime"]
        return candidate, diagnostics

    def _valid_bound(self, diagnostics):
        valid_statuses = {getattr(gp.GRB, name) for name in
                         ("OPTIMAL", "TIME_LIMIT", "NODE_LIMIT", "ITERATION_LIMIT",
                          "SOLUTION_LIMIT", "INTERRUPTED", "USER_OBJ_LIMIT")}
        bound = diagnostics.get("best_bound")
        if diagnostics.get("status") not in valid_statuses or bound is None:
            return None
        return float(bound)

    def _summary(self, vector, projection, attempts, chosen, warm_validation, started):
        last = attempts[-1]
        bounds = [bound for attempt in attempts if (bound := self._valid_bound(attempt)) is not None]
        bound = (max(bounds) if projection else min(bounds)) if bounds else None
        lower_squared = upper_squared = None
        converged = False
        inconsistent = False
        if projection:
            lower_squared = max(0.0, bound) if bound is not None else 0.0
            if chosen is not None:
                upper_squared = chosen["value"]
                if lower_squared > upper_squared + self.bound_consistency_tolerance:
                    inconsistent = True
                    lower_squared = 0.0
                else:
                    lower_squared = min(lower_squared, upper_squared)
                converged = bool(not inconsistent and
                                 upper_squared - lower_squared <= self.absolute_gap_tolerance)
        elif chosen is not None and bound is not None:
            inconsistent = bool(bound < chosen["value"] - self.bound_consistency_tolerance)
            converged = bool(not inconsistent and max(0.0, bound - chosen["value"]) <= self.absolute_gap_tolerance)
        wall = time.perf_counter() - started
        diagnostics = {"status": last.get("status"), "status_name": last["status_name"],
            "runtime": float(sum(attempt["runtime"] for attempt in attempts)),
            "wall_seconds": wall, "objective": last.get("objective"), "best_bound": bound,
            "recomputed_objective": None if chosen is None else chosen["value"],
            "lower_squared": lower_squared, "upper_squared": upper_squared,
            "gap_converged": converged, "bound_inconsistent": inconsistent,
            "absolute_gap": (None if upper_squared is None else upper_squared - lower_squared),
            "absolute_gap_tolerance": self.absolute_gap_tolerance,
            "validation": {"valid": False} if chosen is None else chosen["validation"],
            "incumbent_source": None if chosen is None else chosen["source"],
            "warm_start_validation": warm_validation, "attempts": attempts,
            "query": "projection" if projection else "max_support",
            "base_shape": list(self.shape), "query_shape": list(self.query_shape),
            "base_binary_variables": int(len(self.binary_indices)),
            "integer_relaxation": False, "projection_objective": "sum(residual**2), residual=x-target",
            "bounds_scope": "solver bounds for this query only, not a global nonconvexity upper bound"}
        self.query_count += 1
        self.total_wall_seconds += wall
        return {"point": None if chosen is None else chosen["point"].copy(),
                "primal": None if chosen is None else chosen["primal"].copy(),
                "diagnostics": diagnostics}

    def support(self, v, start=None, fixed_binary=None):
        """Maximize v@x; a valid returned incumbent need not be support-optimal."""
        self._check_open()
        vector = np.asarray(v, dtype=float)
        if vector.shape != (len(self._x),) or not np.isfinite(vector).all():
            raise ValueError("Expected a finite support direction with length T")
        fixed = None
        if fixed_binary is not None:
            fixed = np.asarray(fixed_binary, dtype=float)
            if (fixed.shape != self.binary_indices.shape or not np.isfinite(fixed).all()
                    or np.any(np.abs(fixed - np.rint(fixed)) > self.integrality_tolerance)
                    or np.any((np.rint(fixed) < 0) | (np.rint(fixed) > 1))):
                raise ValueError("fixed_binary must follow binary_indices and contain only zero/one")
            fixed = np.rint(fixed)
            if (np.any(fixed < self._lower[self.binary_indices]) or
                    np.any(fixed > self._upper[self.binary_indices])):
                raise ValueError("fixed_binary conflicts with the original physical variable bounds")
        started = time.perf_counter()
        try:
            if fixed is not None:
                binaries = [self._variables[int(index)] for index in self.binary_indices]
                self.model.setAttr("LB", binaries, fixed.tolist())
                self.model.setAttr("UB", binaries, fixed.tolist())
                self.model.update()
            self._prepare_objective(vector, projection=False)
            warm, warm_validation = self._start_candidate(start, vector, False, fixed)
            candidate, diagnostics = self._attempt(vector, False, self.initial_time_limit, warm, fixed)
            chosen = candidate if warm is None or (candidate is not None and candidate["value"] >= warm["value"]) else warm
            result = self._summary(vector, False, [diagnostics], chosen, warm_validation, started)
            result["diagnostics"]["fixed_binary"] = fixed is not None
            return result
        finally:
            if fixed is not None:
                binaries = [self._variables[int(index)] for index in self.binary_indices]
                self.model.setAttr("LB", binaries, self._lower[self.binary_indices].tolist())
                self.model.setAttr("UB", binaries, self._upper[self.binary_indices].tolist())
                self.model.update()

    def project(self, z, start):
        """Project onto the ORIGINAL free-binary region, retaining every bound."""
        self._check_open()
        vector = np.asarray(z, dtype=float)
        if vector.shape != (len(self._x),) or not np.isfinite(vector).all():
            raise ValueError("Expected a finite projection target with length T")
        started = time.perf_counter()
        self._prepare_objective(vector, projection=True)
        chosen, warm_validation = self._start_candidate(start, vector, True)
        attempts = []
        for limit in (self.initial_time_limit, self.retry_time_limit):
            candidate, diagnostics = self._attempt(vector, True, limit, chosen)
            attempts.append(diagnostics)
            if candidate is not None and (chosen is None or candidate["value"] <= chosen["value"]):
                chosen = candidate
            bounds = [bound for attempt in attempts if (bound := self._valid_bound(attempt)) is not None]
            lower = max(0.0, max(bounds)) if bounds else 0.0
            if chosen is not None and lower <= chosen["value"] + self.bound_consistency_tolerance:
                if max(0.0, chosen["value"] - lower) <= self.absolute_gap_tolerance:
                    break
        return self._summary(vector, True, attempts, chosen, warm_validation, started)

    def close(self):
        if not self._closed:
            self._closed = True
            if self.bridge is not None:
                self.bridge.close()
            elif self.model is not None:
                self.model.dispose()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# Deterministic sampling and certified-interval statistics
class CoverageError(ValueError):
    """The endpoint bank cannot support the declared sampling protocol."""


CHORD_LAMBDAS = np.array((0.1, 0.25, 0.5, 0.75, 0.9), dtype=float)


PREFIX_COUNTS = (500, 1000, 2000, 5000)


def _generator(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return np.random.default_rng(int(seed))


def _points(points):
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] < 1 or not np.isfinite(array).all():
        raise ValueError("points must be a finite two-dimensional array")
    return array


def make_directions(seed=0, T=24):
    """Return 1000 unit directions and their ASCII provenance labels.

    For T=24: 48 signed axes, 88 signed contiguous length-2/4 blocks,
    and 432 Gaussian directions paired with their negatives (864 rows).
    For another positive T, the Gaussian count fills the same 1000-row bank.
    """
    if isinstance(T, (bool, np.bool_)) or not isinstance(T, (int, np.integer)) or T < 1:
        raise ValueError("T must be a positive integer")
    T = int(T)
    rng = _generator(seed)
    rows, labels = [], []

    def pair(vector, label):
        unit = vector / np.linalg.norm(vector)
        rows.extend((unit, -unit))
        labels.extend((label + "_pos", label + "_neg"))

    for index in range(T):
        row = np.zeros(T)
        row[index] = 1.
        pair(row, f"axis_{index:02d}")
    for length in (2, 4):
        for start in range(max(0, T - length + 1)):
            row = np.zeros(T)
            row[start:start + length] = 1.
            pair(row, f"block{length}_{start:02d}")
    remainder = 1000 - len(rows)
    if remainder < 0 or remainder % 2:
        raise CoverageError("Structured direction bank leaves no valid 1000-row paired budget")
    for index, row in enumerate(rng.normal(size=(remainder // 2, T))):
        if np.linalg.norm(row) == 0:
            raise CoverageError("A zero Gaussian direction was drawn; choose another seed")
        pair(row, f"gaussian_{index:03d}")
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=str)


def unique_points(points, tol=1e-8):
    """Greedily retain the first representative of each Linf-near duplicate.

    A candidate is compared with already retained representatives. This is not
    a transitive clustering operation; every retained pair is farther than tol.
    """
    array = _points(points)
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    retained = []
    for index, point in enumerate(array):
        if retained and np.any(np.max(np.abs(array[retained] - point), axis=1) <= tol):
            continue
        retained.append(index)
    return np.asarray(retained, dtype=np.int64)


def _nearest_pairs(points, count=20):
    n = len(points)
    distances = np.empty((n, n), dtype=float)
    # Difference-based distances avoid cancellation for nearby support points.
    for start in range(0, n, 64):
        difference = points[start:start + 64, None, :] - points[None, :, :]
        distances[start:start + 64] = np.einsum("ijk,ijk->ij", difference, difference)
    np.fill_diagonal(distances, np.inf)
    neighbours = np.argsort(distances, axis=1, kind="stable")[:, :count]
    sources = np.repeat(np.arange(n), count)
    destinations = neighbours.ravel()
    pairs = np.column_stack((np.minimum(sources, destinations), np.maximum(sources, destinations)))
    return np.unique(pairs, axis=0)


def _chords(pairs, first_id):
    n = len(pairs) * len(CHORD_LAMBDAS)
    indices = np.full((n, 5), -1, dtype=np.int64)
    weights = np.zeros((n, 5), dtype=float)
    values = np.tile(CHORD_LAMBDAS, len(pairs))
    indices[:, :2] = np.repeat(pairs, len(CHORD_LAMBDAS), axis=0)
    weights[:, 0], weights[:, 1] = 1. - values, values
    ids = np.repeat(np.arange(first_id, first_id + len(pairs)), len(CHORD_LAMBDAS))
    return indices, weights, ids, values


def _multipoint(rng, n, count):
    indices = np.full((500, 5), -1, dtype=np.int64)
    weights = np.zeros((500, 5), dtype=float)
    indices[:, :count] = np.asarray([rng.choice(n, size=count, replace=False) for _ in range(500)])
    weights[:, :count] = rng.dirichlet(np.full(count, 0.5), size=500)
    permutation = rng.permutation(500)
    return indices[permutation], weights[permutation]


def make_probes(points_unique, seed=1):
    """Return the fixed 5000-probe bank with balanced nested prefixes.

    Local pairs are sampled without replacement from the undirected union of
    each endpoint's 20 Euclidean nearest neighbours. Global pairs are sampled
    from all undirected pairs except the 500 selected local pairs. Each chosen
    pair has five lambda values, with lambda the second endpoint's weight.

    Every ten rows contain five local chords, three global chords, one
    three-point mixture and one five-point mixture. Prefixes 500/1000/2000/5000
    therefore have the exact 50/30/10/10 percent composition. At these prefixes
    all included chord pairs have their complete five-lambda sweep.

    Pair IDs are local 0..499, global 500..799, and -1 for multipoint probes.
    The lambda array has NaN only for non-chord rows, where it is inapplicable.
    Targets and weights are always finite. Padded endpoint indices/weights are
    -1/0 and all active endpoints within a probe are distinct.
    """
    points = _points(points_unique)
    n, dim = points.shape
    if n <= 20 or n * (n - 1) // 2 < 800:
        raise CoverageError(f"Need at least 800 distinct undirected endpoint pairs; got {n} endpoints")
    rng = _generator(seed)
    local_candidates = _nearest_pairs(points)
    if len(local_candidates) < 500:
        raise CoverageError(f"20-neighbour graph has {len(local_candidates)} pairs; 500 local pairs required")
    local_pairs = local_candidates[rng.choice(len(local_candidates), size=500, replace=False)]
    first, second = np.triu_indices(n, k=1)
    all_pairs = np.column_stack((first, second))
    local_codes = local_pairs[:, 0] * n + local_pairs[:, 1]
    available = all_pairs[~np.isin(first * n + second, local_codes)]
    if len(available) < 300:
        raise CoverageError(f"Only {len(available)} global pairs remain; 300 required")
    global_pairs = available[rng.choice(len(available), size=300, replace=False)]
    local = _chords(local_pairs, 0)
    global_ = _chords(global_pairs, 500)
    three, five = _multipoint(rng, n, 3), _multipoint(rng, n, 5)

    indices = np.full((500, 10, 5), -1, dtype=np.int64)
    weights = np.zeros((500, 10, 5), dtype=float)
    pair_ids = np.full((500, 10), -1, dtype=np.int64)
    lambdas = np.full((500, 10), np.nan, dtype=float)
    kinds = np.empty((500, 10), dtype="U12")
    for selected, slots, width, kind in ((local, slice(0, 5), 5, "local_chord"),
                                         (global_, slice(5, 8), 3, "global_chord")):
        indices[:, slots] = selected[0].reshape(500, width, 5)
        weights[:, slots] = selected[1].reshape(500, width, 5)
        pair_ids[:, slots] = selected[2].reshape(500, width)
        lambdas[:, slots] = selected[3].reshape(500, width)
        kinds[:, slots] = kind
    indices[:, 8], weights[:, 8], kinds[:, 8] = three[0], three[1], "three_point"
    indices[:, 9], weights[:, 9], kinds[:, 9] = five[0], five[1], "five_point"
    indices, weights = indices.reshape(5000, 5), weights.reshape(5000, 5)
    targets = np.einsum("ij,ijk->ik", weights, points[np.where(indices >= 0, indices, 0)])
    if not np.isfinite(targets).all() or targets.shape != (5000, dim):
        raise ValueError("Nonfinite convex combinations; inspect endpoint magnitudes")
    return {"targets": targets, "indices": indices, "weights": weights,
            "kind": kinds.ravel(), "pair_id": pair_ids.ravel(), "lambda": lambdas.ravel(),
            "sample_index": np.arange(5000, dtype=np.int64)}


def _number(value, field):
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result):
        return None
    if result < 0:
        raise ValueError(f"{field} must be nonnegative; received {result}")
    return result


def _flag(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    raise ValueError(f"Expected a boolean or missing flag, received {value!r}")


def _statistics(values, total):
    finite = np.asarray([value for value in values if value is not None], dtype=float)
    result = {"count": len(finite), "missing_count": total - len(finite)}
    result.update({key: None for key in ("mean", "median", "p95", "max")})
    if len(finite):
        result.update(mean=float(finite.mean()), median=float(np.median(finite)),
                      p95=float(np.percentile(finite, 95)), max=float(finite.max()))
    return result


def _probe_record(row):
    if row is None:
        return None
    return {key: row[key] for key in ("sample_index", "kind", "lower_percent", "upper_percent",
                                     "classification", "gap_converged", "same_binary_mode")}


def _group_summary(rows, threshold):
    n = len(rows)
    lower = _statistics([row["lower_percent"] for row in rows], n)
    upper = _statistics([row["upper_percent"] for row in rows], n)
    finite_lower = [row for row in rows if row["lower_percent"] is not None]
    detected = [row for row in finite_lower if row["lower_percent"] > threshold]
    largest = max(finite_lower, key=lambda row: row["lower_percent"], default=None)
    largest_detected = max(detected, key=lambda row: row["lower_percent"], default=None)
    intervals = {}
    for statistic in ("mean", "median", "p95", "max"):
        low = lower[statistic] if lower["count"] == n and n else None
        if statistic == "max" and finite_lower:
            low = lower["max"]
        high = upper[statistic] if upper["count"] == n and n else None
        intervals[statistic] = {"lower": low, "upper": high}
    flags = [row["gap_converged"] for row in rows]
    same = [row["same_binary_mode"] for row in rows]
    return {"count": n, "lower_percent": lower, "upper_percent": upper,
            "statistic_intervals": intervals,
            "classification_counts": dict(sorted(Counter(row["classification"] for row in rows).items())),
            "convergence_counts": {"converged": sum(value is True for value in flags),
                                   "not_converged": sum(value is False for value in flags),
                                   "unknown": sum(value is None for value in flags)},
            "same_binary_count": sum(value is True for value in same),
            "different_binary_count": sum(value is False for value in same),
            "unknown_binary_mode_count": sum(value is None for value in same),
            "detected_nonconvex_count": len(detected),
            "max_lower_bound_probe": _probe_record(largest),
            "max_detected_lower_bound_probe": _probe_record(largest_detected)}


def _summaries(rows, threshold):
    groups = {"overall": rows}
    for label, kind in (("local", "local_chord"), ("global", "global_chord"),
                         ("three", "three_point"), ("five", "five_point")):
        groups[label] = [row for row in rows if row["kind"] == kind]
    groups["multi"] = [row for row in rows if row["kind"] in ("three_point", "five_point")]
    if any(row["same_binary_mode"] is not None for row in rows):
        groups["same_binary_mode"] = [row for row in rows if row["same_binary_mode"] is True]
        groups["different_binary_mode"] = [row for row in rows if row["same_binary_mode"] is False]
        for label in ("local", "global"):
            groups[label + "_different_binary_mode"] = [row for row in groups[label]
                                                         if row["same_binary_mode"] is False]
    return {name: _group_summary(selected, threshold) for name, selected in groups.items()}


def build_summary(rows, threshold_percent=0.001):
    """Summarize certified distance intervals without hiding incomplete solves.

    Input sample_index is zero-based. Rows may arrive out of order and are
    sorted by that index; incomplete prefixes remain explicitly incomplete.
    Distribution summaries with missing values state the available count.
    An all-probe upper statistic is unavailable if any upper bound is missing;
    it is never replaced with a finite-subset upper bound. Maximum finite lower
    bounds remain valid lower estimates even when other probes are missing.
    Bounds concern these finite probes, not a global nonconvexity upper bound.
    """
    if not np.isfinite(threshold_percent) or threshold_percent < 0:
        raise ValueError("threshold_percent must be finite and nonnegative")
    normalized = []
    seen = set()
    required = ("lower_percent", "upper_percent", "classification", "gap_converged", "kind", "sample_index")
    for original in rows:
        missing = [key for key in required if key not in original]
        if missing:
            raise ValueError(f"Missing summary fields: {missing}")
        index = original["sample_index"]
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or index < 0:
            raise ValueError("sample_index must be a nonnegative integer")
        index = int(index)
        if index in seen:
            raise ValueError(f"Duplicate sample_index: {index}")
        seen.add(index)
        low = _number(original["lower_percent"], "lower_percent")
        high = _number(original["upper_percent"], "upper_percent")
        if low is not None and high is not None and low > high:
            raise ValueError(f"Reversed certified interval at sample {index}: {low}>{high}")
        normalized.append({"sample_index": index, "kind": str(original["kind"]),
                           "lower_percent": low, "upper_percent": high,
                           "classification": str(original["classification"]),
                           "gap_converged": _flag(original["gap_converged"]),
                           "same_binary_mode": _flag(original.get("same_binary_mode"))})
    normalized.sort(key=lambda row: row["sample_index"])
    threshold = float(threshold_percent)
    groups = _summaries(normalized, threshold)
    prefixes = {}
    for count in PREFIX_COUNTS:
        selected = [row for row in normalized if row["sample_index"] < count]
        prefixes[str(count)] = {"requested_count": count, "available_count": len(selected),
                                "complete": len(selected) == count,
                                "groups": _summaries(selected, threshold)}
    return {"threshold_percent": threshold, "sample_count": len(normalized),
            "scope": "Finite convex-combination probes; no global nonconvexity upper bound",
            "groups": groups, "prefixes": prefixes,
            "max_lower_bound_probe": groups["overall"]["max_lower_bound_probe"],
            "max_detected_lower_bound_probe": groups["overall"]["max_detected_lower_bound_probe"]}


# Evaluation and original-run provenance
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
    from Simulator.cases.microgrid_case import MicroGrid

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
    sources = [Path(__file__), source_dir.parent / "cases" / "microgrid_case.py"]
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


def run_evaluation(args):
    _require_gurobi()
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


def evaluate_main(argv=None):
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
        RESULTS_ROOT / "figures" / "07_nonconvexity_diagnostic.pdf" if args.device_scale == 1
        else RESULTS_ROOT / "figures" / ("07_nonconvexity_" + args.output_dir.name + "_scale" + str(args.device_scale) + ".pdf"))
    run_evaluation(args)


# Paired MT minimum-uptime ablation
def run_ablation(args):
    _require_gurobi()
    from pathlib import Path
    import sys
    import json
    import time
    import numpy as np
    import pandas as pd


    base = args.base_dir
    old = base / 'nonconvexity'
    out = args.output_dir or base / 'nonconvexity_no_min_uptime'
    out.mkdir(parents=True, exist_ok=True)
    mg = load_case(base / 'training', device_scale=3, disable_mt_min_uptime=True)
    def arrays(path):
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    endpoint, probes, saved = [arrays(old / name) for name in
                              ('endpoints.npz', 'probes.npz', 'projections.npz')]
    old_rows = pd.read_csv(old / 'samples.csv')
    checkpoint = json.loads((old / 'checkpoint.json').read_text(encoding='utf-8'))
    assert checkpoint['completed'] == 5000
    for name, sha in checkpoint['files'].items():
        assert digest(old / name) == sha
    inputs = {name: digest(old / name) for name in
              ('metadata.json', 'endpoints.npz', 'probes.npz', 'projections.npz',
               'samples.csv', 'independent_storage_verification.json')}
    rows, diagnostics, points, primals = [], [], [], []
    started = time.perf_counter()
    with PhysicalDomain(mg.model) as domain:
        assert list(domain.names) == endpoint['variable_names'].tolist()
        np.testing.assert_array_equal(endpoint['p_min'], mg.pmin)
        np.testing.assert_array_equal(endpoint['p_max'], mg.pmax)
        validate = physical_validator(mg.model, domain.names)
        assert validate(endpoint['raw_primal'])['passed']
        assert validate(saved['primal'])['passed']
        for i in range(5000):
            answer = domain.project(probes['targets'][i], start=saved['primal'][i])
            require_witness(domain, answer, f'paired probe {i}')
            row = sample_row(i, str(probes['kind'][i]), answer,
                             bool(old_rows.iloc[i]['same_binary_mode']))
            assert row['upper_squared'] <= old_rows.iloc[i]['upper_squared'] + 1e-10
            row['classification'] = ('below_threshold' if row['upper_percent'] <= .01 else
                                     'detected_gap' if row['lower_percent'] > .01 else 'undetermined')
            rows.append(row)
            diagnostics.append(answer['diagnostics'])
            points.append(answer['point'])
            primals.append(answer['primal'])
            if (i + 1) % 500 == 0:
                print(json.dumps({'paired_completed': i + 1,
                                  'max_upper_percent': max(r['upper_percent'] for r in rows)}), flush=True)
        primals, points = np.asarray(primals), np.asarray(points)
        verification = validate(primals)
        assert verification['passed']
        ledger_error = 0.
        for j, ess in enumerate(mg.ESS_list):
            names = list(domain.names)
            ch = primals[:, [names.index(f'p_chg_ESS[{j},{t}]') for t in range(24)]]
            dis = primals[:, [names.index(f'p_dis_ESS[{j},{t}]') for t in range(24)]]
            soc = primals[:, [names.index(f'e_ESS[{j},{t}]') for t in range(25)]]
            expected = np.c_[np.full(5000, ess['init_SOC']), ess['init_SOC'] +
                             np.cumsum(mg.deltaT * (ess['eta_chg'] * ch - dis / ess['eta_dis']) /
                                       ess['E_rated'], axis=1)]
            ledger_error = max(ledger_error, float(np.max(np.abs(soc - expected))))
        assert ledger_error < 1e-8
    summary = build_summary(rows, threshold_percent=.01)
    summary.update(completed=5000, scope='Exactly the baseline targets and normalization; only MT minimum uptime removed',
                   original_inputs_sha256=inputs, physical_validation=verification,
                   max_cumulative_ESS_SOC_residual=ledger_error,
                   all_360_original_binary_variables_free=True,
                   removed_constraints=mg.nonconvexity_removed_constraints,
                   wall_seconds=time.perf_counter() - started,
                   runner_sha256=digest(Path(__file__)))
    assert inputs == {name: digest(old / name) for name in inputs}
    pd.DataFrame(rows).to_csv(out / 'paired_samples.csv', index=False)
    write_npz(out / 'paired_projections.npz', points=points, primal=primals)
    write_json(out / 'paired_projection_diagnostics.json', diagnostics)
    write_json(out / 'paired_summary.json', summary)
    print(json.dumps(summary['groups']['overall']), flush=True)


def ablation_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path,
                        default=RESULTS_ROOT / "microgrid" / "scale3",
                        help="Scale3 root containing training and saved evaluations")
    parser.add_argument("--output-dir", type=Path,
                        help="Output directory; defaults to the original experiment directory")
    args = parser.parse_args(argv)
    run_ablation(args)


# Independent physical-row verification
def physical_validator(model, names):
    variables = {v.name: v for v in model.component_data_objects(pyo.Var)}
    if set(variables) != set(names):
        raise ValueError("Saved variable names differ from the original physical model")
    ordered = [variables[str(name)] for name in names]
    index = {id(v): i for i, v in enumerate(ordered)}
    row, col, values, lower, upper = [], [], [], [], []
    for r, constraint in enumerate(model.component_data_objects(pyo.Constraint, active=True)):
        repn = generate_standard_repn(constraint.body, compute_values=True)
        if not repn.is_linear():
            raise ValueError("This audit expects the retained linear mixed-integer model")
        constant = float(repn.constant or 0.0)
        for var, coefficient in zip(repn.linear_vars, repn.linear_coefs):
            row.append(r)
            col.append(index[id(var)])
            values.append(float(coefficient))
        lower.append(float(pyo.value(constraint.lower)) - constant if constraint.has_lb() else -np.inf)
        upper.append(float(pyo.value(constraint.upper)) - constant if constraint.has_ub() else np.inf)
    matrix = coo_matrix((values, (row, col)), shape=(len(lower), len(ordered))).tocsr()
    lower, upper = np.asarray(lower), np.asarray(upper)
    lb = np.asarray([float(pyo.value(v.lb)) if v.has_lb() else -np.inf for v in ordered])
    ub = np.asarray([float(pyo.value(v.ub)) if v.has_ub() else np.inf for v in ordered])
    integers = np.asarray([i for i, v in enumerate(ordered) if v.is_integer()], dtype=int)

    def validate(primals):
        primals = np.asarray(primals)
        if primals.ndim != 2 or primals.shape[1] != len(ordered) or not np.isfinite(primals).all():
            raise ValueError("Invalid saved full-primal array")
        violation = bound_violation = integer_violation = 0.0
        for start in range(0, len(primals), 250):
            chunk = primals[start:start + 250]
            lhs = (matrix @ chunk.T).T
            violation = max(violation, float(np.max(lower - lhs)), float(np.max(lhs - upper)))
            bound_violation = max(bound_violation, float(np.max(lb - chunk)), float(np.max(chunk - ub)))
            integer_violation = max(integer_violation, float(np.max(np.abs(chunk[:, integers] - np.rint(chunk[:, integers])))))
        result = dict(count=len(primals), constraint_violation=max(0.0, violation),
                      bound_violation=max(0.0, bound_violation), integer_violation=integer_violation)
        result["passed"] = (violation <= 1.00001e-8 and bound_violation <= 1.00001e-8
                            and integer_violation <= 1.00001e-9)
        if not result["passed"]:
            raise ValueError(f"Independent physical validation failed: {result}")
        return result

    return validate


def verify_nonconvexity(input_dir, training_dir):
    meta = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    if meta["fingerprint"] != provenance(training_dir)["fingerprint"]:
        raise ValueError("Input/source provenance changed after the saved run")
    mg = load_case(training_dir)
    with np.load(input_dir / "endpoints.npz", allow_pickle=False) as data:
        endpoints = {key: data[key] for key in data.files}
    with np.load(input_dir / "probes.npz", allow_pickle=False) as data:
        probes = {key: data[key] for key in data.files}
    with np.load(input_dir / "projections.npz", allow_pickle=False) as data:
        projected = {key: data[key] for key in data.files}
    with np.load(input_dir / "controls.npz", allow_pickle=False) as data:
        controls = {key: data[key] for key in data.files}
    rows = pd.read_csv(input_dir / "samples.csv")
    n = len(rows)
    if n not in (100, 5000):
        raise ValueError("Verification requires a completed pilot or formal batch")
    np.testing.assert_array_equal(rows.sample_index, np.arange(n))
    checkpoint = json.loads((input_dir / "checkpoint.json").read_text(encoding="utf-8"))
    for name, expected in checkpoint["files"].items():
        if digest(input_dir / name) != expected:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
    validate = physical_validator(mg.model, endpoints["variable_names"])
    validations = {"raw_endpoints": validate(endpoints["raw_primal"]),
                   "formal_projections": validate(projected["primal"][:n]),
                   "control_endpoints": validate(controls["endpoint_primal"]),
                   "control_feasible_witnesses": validate(controls["feasible_witnesses"]),
                   "control_projections": validate(controls["primal"])}
    np.testing.assert_array_equal(endpoints["directions"], make_directions(seed=0, T=24)[0])
    regenerated = make_probes(endpoints["points"], seed=1)
    for key in ("targets", "indices", "weights", "kind", "pair_id"):
        np.testing.assert_array_equal(probes[key], regenerated[key])
    if np.any(probes["weights"] < 0):
        raise ValueError("Negative mixture weights")
    np.testing.assert_allclose(probes["weights"].sum(axis=1), 1, rtol=0, atol=1e-14)
    xidx = endpoints["x_indices"]
    np.testing.assert_array_equal(endpoints["raw_points"], endpoints["raw_primal"][:, xidx])
    np.testing.assert_array_equal(projected["points"][:n], projected["primal"][:n, xidx])
    np.testing.assert_allclose(controls["targets"], controls["feasible_witnesses"][:, xidx], atol=1e-14, rtol=0)
    reconstructed_controls = np.sum(controls["weights"][:, :, None]
                                   * controls["endpoint_primal"][controls["indices"]], axis=1)
    np.testing.assert_allclose(reconstructed_controls, controls["feasible_witnesses"], atol=1e-12, rtol=0)
    q = np.sum((projected["points"][:n] - probes["targets"][:n]) ** 2, axis=1)
    np.testing.assert_allclose(q, rows.upper_squared, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(100 * np.sqrt(q / 24), rows.upper_percent, rtol=1e-10, atol=1e-12)
    lower = rows.lower_percent.to_numpy()
    upper = rows.upper_percent.to_numpy()
    if np.any(lower > upper):
        raise ValueError("An inverted distance interval was saved")
    threshold = PROTOCOL["threshold_percent"]
    classification = np.where(upper <= threshold, "numerical_zero",
                              np.where(lower > threshold, "detected_gap", "undetermined"))
    np.testing.assert_array_equal(classification, rows.classification)
    control_upper = 100 * np.sqrt(np.mean((controls["targets"] - controls["points"]) ** 2, axis=1))
    if np.max(control_upper) > threshold:
        raise ValueError("An independently checked zero control exceeds the threshold")
    result = dict(passed=True, completed=n, physical_validation=validations,
                  reproducible_directions=True, reproducible_probes=True,
                  membership_weights_verified=True, objective_and_classifications_verified=True,
                  max_control_percent=float(np.max(control_upper)),
                  original_variables=len(endpoints["variable_names"]),
                  original_binary_variables=len(endpoints["binary_indices"]),
                  validation_basis="Original nominal Pyomo expressions, independently compiled into sparse rows; no optimizer needed",
                  caveat="Feasibility and global solver bounds are interpreted within declared numerical tolerances, not exact-arithmetic certificates")
    write_json(input_dir / "independent_verification.json", result)
    print(json.dumps(result, indent=2))
    return result


def verify_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=RESULTS_ROOT / "microgrid" / "nonconvexity")
    parser.add_argument("--training-dir", type=Path, default=RESULTS_ROOT / "microgrid" / "training")
    args = parser.parse_args(argv)
    verify_nonconvexity(args.input_dir, args.training_dir)


# Independent portfolio/storage verification
COUNTS = {"MT": 1, "PV": 1, "TL": 1, "CL": 1, "ESS": 1, "HP": 2}


def _storage_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _storage_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _storage_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _storage_assert_equal(actual, expected, context):
    """Recursively compare metadata without array truth-value ambiguities."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Metadata keys changed: {context}")
        for key in expected:
            _storage_assert_equal(actual[key], expected[key], f"{context}.{key}")
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError(f"Metadata list changed: {context}")
        for index, value in enumerate(expected):
            _storage_assert_equal(actual[index], value, f"{context}[{index}]")
    elif isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected, err_msg=context)
    elif actual != expected:
        raise ValueError(f"Metadata value changed: {context}: {actual!r} != {expected!r}")


def _storage_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(f"Expected an explicit boolean, received {value!r}")


def _storage_effective_counts(scale, override):
    if override is None:
        return {name: count * scale for name, count in COUNTS.items()}
    if not isinstance(override, dict) or set(override) != set(COUNTS):
        raise ValueError("The count override must specify exactly MT, PV, TL, CL, ESS, HP")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in override.values()):
        raise ValueError("Every custom device count must be a positive integer")
    return dict(override)


def _storage_model_size(counts):
    variables = 24 * (4 * counts["MT"] + counts["PV"] + 2 * counts["TL"] +
                      counts["CL"] + 4 * counts["ESS"] + 2 * counts["HP"] + 1) + counts["ESS"]
    binaries = 24 * (3 * counts["MT"] + counts["TL"] + counts["ESS"])
    return variables, binaries


def _storage_power_spans(data):
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
    from Simulator.cases.microgrid_case import MicroGrid

    with (training_dir / "metadata.pkl").open("rb") as stream:
        saved = pickle.load(stream)
    if saved["T"] != 24 or saved["deltaT"] != 1.0:
        raise ValueError("Expected the retained T=24, one-hour microgrid")
    for name, count in COUNTS.items():
        devices = saved[name + "_list"]
        if len(devices) != count * scale:
            raise ValueError(f"Unexpected {name} device count")
        for index, device in enumerate(devices):
            _storage_assert_equal(device, devices[index % count], f"repeated {name}[{index}]")
    counts = _storage_effective_counts(scale, device_counts_override)
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
    _storage_assert_equal(unchanged_view, portfolio, "only_E_rated_may_change")
    mg = MicroGrid(seed=42, T=24, data=changed)
    mg.build_mg_constraints(conservativeness=0.0)
    _storage_assert_equal(saved, original_snapshot, "training_metadata_not_mutated")
    for name in COUNTS:
        _storage_assert_equal(getattr(mg, name + "_list"), changed[name + "_list"], name)
    expected_lower, expected_upper = _storage_power_spans(portfolio)
    np.testing.assert_array_equal(mg.pmin, expected_lower)
    np.testing.assert_array_equal(mg.pmax, expected_upper)
    if device_counts_override is None:
        np.testing.assert_array_equal(mg.pmin, saved["p_min"])
        np.testing.assert_array_equal(mg.pmax, saved["p_max"])
    if not np.all(mg.pmax > mg.pmin):
        raise ValueError("The declared portfolio must have positive fixed power spans")
    variables = list(mg.model.component_data_objects(pyo.Var))
    if (len(variables), sum(v.is_binary() for v in variables)) != _storage_model_size(counts):
        raise ValueError("Rebuilt model did not preserve all original variables/binaries")
    return mg, saved, changes


def _storage_check_rows(rows, diagnostics, points, targets, threshold, binary_count):
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
        if _storage_bool(row["gap_converged"]) != _storage_bool(diagnostic["gap_converged"]):
            raise ValueError(f"Convergence flag changed for projection {index}")
        if _storage_bool(row["gap_converged"]):
            tolerance = float(diagnostic["absolute_gap_tolerance"])
            if upper[index] - lower[index] > tolerance + 1e-12:
                raise ValueError(f"Projection {index} claims convergence outside its tolerance")
    return q


def _storage_reporting_rows(rows, threshold):
    """Reclassify detached records; retain all stored numerical rows verbatim."""
    records = rows.to_dict("records")
    for row in records:
        row["classification"] = ("detected_gap" if row["lower_percent"] > threshold
                                 else "below_threshold" if row["upper_percent"] <= threshold
                                 else "undetermined")
    return records


def _storage_check_summary(saved, expected):
    def check_groups(actual_groups, expected_groups, context):
        if set(actual_groups) != set(expected_groups):
            raise ValueError(f"Saved summary groups differ: {context}")
        for name, group in expected_groups.items():
            actual = actual_groups[name]
            for key in ("count", "classification_counts", "detected_nonconvex_count"):
                _storage_assert_equal(actual[key], group[key], f"{context}.{name}.{key}")
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
            _storage_assert_equal(saved["prefixes"][size][key], prefix[key], f"prefix{size}.{key}")
        check_groups(saved["prefixes"][size]["groups"], prefix["groups"], f"prefix{size}")


def _storage_check_forward_mt_schedules(primals, names, devices, context, tolerance=1e-9,
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


def verify_storage(input_dir, training_dir):
    input_dir, training_dir = Path(input_dir).resolve(), Path(training_dir).resolve()
    meta = _storage_json(input_dir / "metadata.json")
    scale = meta["device_scale"]
    energy = None if meta["storage_energy_kwh"] is None else float(meta["storage_energy_kwh"])
    counts_override = meta.get("device_counts_override")
    if isinstance(scale, bool) or not isinstance(scale, int) or scale not in (1, 3):
        raise ValueError("Expected device_scale 1 or 3")
    if energy is not None and (not np.isfinite(energy) or energy <= 0):
        raise ValueError("storage_energy_kwh must be finite and positive")
    if resolve_recorded_code_path(meta["training_directory"]).resolve() != training_dir:
        raise ValueError("The declared training_directory differs from --training-dir")
    protocol = meta["protocol"]
    disable_uptime = _storage_bool(meta.get("disable_mt_min_uptime", False))
    if _storage_bool(protocol.get("disable_mt_min_uptime", False)) != disable_uptime:
        raise ValueError("Minimum-uptime ablation disagrees with the recorded protocol")
    expected_counts = _storage_effective_counts(scale, counts_override)
    variable_count, binary_count = _storage_model_size(expected_counts)
    if protocol["device_counts"] != expected_counts:
        raise ValueError("Protocol device counts differ from the requested original portfolio")
    _storage_assert_equal(protocol.get("device_counts_override"), counts_override,
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
    recorded = {resolve_recorded_code_path(path).resolve(): value for group in ("inputs", "sources")
                for path, value in meta[group].items()}
    if recorded.get(training_dir / "metadata.pkl") != _storage_digest(training_dir / "metadata.pkl"):
        raise ValueError("Training parameters do not match the saved input hash")
    for path, expected in recorded.items():
        if _storage_digest(path) != expected:
            raise ValueError(f"Input/source hash changed: {path}")
    checkpoint = _storage_json(input_dir / "checkpoint.json")
    if checkpoint["completed"] != 5000 or meta.get("completed") != 5000:
        raise ValueError("Verification requires the complete 5000-probe experiment")
    for name, expected in checkpoint["files"].items():
        if _storage_digest(input_dir / name) != expected:
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
        _storage_assert_equal(meta["removed_constraints"], removed, "removed_minimum_uptime_rows")
    _storage_assert_equal(meta["parameter_changes"], changes, "declared_parameter_changes")
    expected_source_indices = {} if counts_override is None else {
        name: [index % len(original[name + "_list"]) for index in range(count)]
        for name, count in expected_counts.items()}
    if counts_override is not None or "device_source_indices" in meta:
        _storage_assert_equal(meta["device_source_indices"], expected_source_indices,
                      "declared_device_source_indices")
    endpoints = _storage_arrays(input_dir / "endpoints.npz")
    probes = _storage_arrays(input_dir / "probes.npz")
    projected = _storage_arrays(input_dir / "projections.npz")
    controls = _storage_arrays(input_dir / "controls.npz")
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
        uptime_checks[group_name] = _storage_check_forward_mt_schedules(
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
    np.testing.assert_array_equal([_storage_bool(x) for x in rows.same_binary_mode], same_mode)
    np.testing.assert_allclose(controls["targets"], controls["feasible_witnesses"][:, expected_x], atol=1e-14, rtol=0)
    np.testing.assert_array_equal(controls["points"], controls["primal"][:, expected_x])
    witness = np.sum(controls["weights"][:, :, None] *
                     controls["endpoint_primal"][controls["indices"]], axis=1)
    np.testing.assert_allclose(witness, controls["feasible_witnesses"], atol=1e-12, rtol=0)
    fixed = controls["fixed_binary"]
    np.testing.assert_array_equal(np.rint(controls["endpoint_primal"][:, expected_binary]),
                                  np.broadcast_to(fixed, (len(controls["endpoint_primal"]), len(fixed))))
    diagnostics = _storage_json(input_dir / "projection_diagnostics.json")
    q = _storage_check_rows(rows, diagnostics, projected["points"], probes["targets"], threshold, binary_count)
    cq = _storage_check_rows(control_rows, _storage_json(input_dir / "controls_diagnostics.json"),
                     controls["points"], controls["targets"], threshold, binary_count)
    if np.max(100 * np.sqrt(cq / 24)) > threshold:
        raise ValueError("Known-feasible fixed-mode control exceeds the threshold")
    numerical_summary = build_summary(rows.to_dict("records"), threshold_percent=threshold)
    reporting_summary = build_summary(_storage_reporting_rows(rows, reporting_threshold),
                                      threshold_percent=reporting_threshold)
    # Legacy runs have only numerical classifications. Explicit reporting
    # overrides use separate labels and never overwrite samples.csv.
    summary = reporting_summary if reporting_override else numerical_summary
    saved_summary = _storage_json(input_dir / "summary.json")
    if saved_summary["threshold_percent"] != (reporting_threshold if reporting_override else threshold):
        raise ValueError("Saved summary threshold differs from its metadata declaration")
    for key, expected in (("numerical_threshold_percent", threshold),
                          ("reporting_threshold_percent", reporting_threshold)):
        if reporting_override or key in saved_summary:
            if saved_summary.get(key) != expected:
                raise ValueError(f"Saved summary has inconsistent {key}")
    _storage_check_summary(saved_summary, summary)
    endpoint_diagnostics = _storage_json(input_dir / "endpoint_diagnostics.json")
    if len(endpoint_diagnostics) != 1000 or any(d.get("integer_relaxation") is not False or
            d.get("base_binary_variables") != binary_count for d in endpoint_diagnostics):
        raise ValueError("Support diagnostics do not preserve the original integer model")
    for path, expected in recorded.items():
        if _storage_digest(path) != expected:
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


def verify_storage_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Optionally write the independent JSON report")
    args = parser.parse_args(argv)
    result = verify_storage(args.input_dir, args.training_dir)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.report is not None:
        args.report.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


# Saved-data summaries
def _nonconvexity_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nonconvexity_read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _nonconvexity_write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def summarize_nonconvexity(input_dir, threshold_percent=0.01):
    import pandas as pd

    folder = Path(input_dir)
    if not np.isfinite(threshold_percent) or threshold_percent <= 0:
        raise ValueError("Reporting threshold must be finite and positive")
    meta = _nonconvexity_read_json(folder / "metadata.json")
    previous = _nonconvexity_read_json(folder / "summary.json")
    checkpoint = _nonconvexity_read_json(folder / "checkpoint.json")
    if meta["status"] != "completed" or meta["completed"] != 5000:
        raise ValueError("A completed 5000-probe experiment is required")
    for name, expected in checkpoint["files"].items():
        if _nonconvexity_digest(folder / name) != expected:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
    immutable_names = ("samples.csv", "endpoints.npz", "probes.npz", "projections.npz",
                       "projection_diagnostics.json", "controls.npz", "controls.csv",
                       "controls_diagnostics.json")
    hashes = {name: _nonconvexity_digest(folder / name) for name in immutable_names}
    metadata_before = _nonconvexity_digest(folder / "metadata.json")
    rows = pd.read_csv(folder / "samples.csv").to_dict("records")
    for row in rows:
        row["classification"] = ("below_threshold" if row["upper_percent"] <= threshold_percent
                                 else "detected_gap" if row["lower_percent"] > threshold_percent
                                 else "undetermined")
    summary = {**previous, **build_summary(rows, threshold_percent=threshold_percent),
               "numerical_threshold_percent": meta["protocol"]["threshold_percent"],
               "reporting_threshold_percent": threshold_percent,
               "reporting_note": "Strict lower-bound exceedance; upper bounds at or below the reporting threshold are below_threshold, not numerical zeros. Original numerical controls and sample classifications are unchanged."}
    for side in ("lower_percent", "upper_percent"):
        for field in ("mean", "median", "p95", "max"):
            np.testing.assert_allclose(summary["groups"]["overall"][side][field],
                                       previous["groups"]["overall"][side][field], rtol=1e-12, atol=1e-14)
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
    nested = [{"sample_count": n,
               "max_lower_percent": max(r["lower_percent"] for r in rows[:n]),
               "max_upper_percent": max(r["upper_percent"] for r in rows[:n]),
               "detected_count": sum(r["classification"] == "detected_gap" for r in rows[:n])}
              for n in (500, 1000, 2000, 5000)]
    _nonconvexity_write_json(folder / "summary.json", summary)
    pd.DataFrame(statistics).to_csv(folder / "statistics.csv", index=False)
    pd.DataFrame(nested).to_csv(folder / "sample_count_comparison.csv", index=False)
    meta["reporting_threshold_percent"] = threshold_percent
    _nonconvexity_write_json(folder / "metadata.json", meta)
    checkpoint["files"] = {name: _nonconvexity_digest(folder / name) for name in checkpoint["files"]}
    _nonconvexity_write_json(folder / "checkpoint.json", checkpoint)
    if hashes != {name: _nonconvexity_digest(folder / name) for name in immutable_names}:
        raise RuntimeError("Original optimization results changed during summarization")
    _nonconvexity_write_json(folder / "reporting_statistics_update.json", {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "source_fingerprint": meta["fingerprint"], "metadata_before_sha256": metadata_before,
        "metadata_after_sha256": _nonconvexity_digest(folder / "metadata.json"),
        "previous_reporting_threshold_percent": previous["threshold_percent"],
        "reporting_threshold_percent": threshold_percent,
        "numerical_threshold_percent": meta["protocol"]["threshold_percent"],
        "optimization_rerun": False, "unchanged_raw_files_sha256": hashes,
        "classification_counts": summary["groups"]["overall"]["classification_counts"],
        "postprocessing_source_sha256": _nonconvexity_digest(Path(__file__))})
    return summary["groups"]["overall"]["classification_counts"]


def summarize_scale3_nonconvexity(args):
    from pathlib import Path
    import json
    import sys
    import numpy as np
    import pandas as pd


    folder = args.base_dir / 'nonconvexity'
    output = args.output_dir or folder
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    assert meta['status'] == 'completed' and meta['completed'] == 5000
    assert meta['reporting_threshold_percent'] == 0.01
    mg = load_case(args.base_dir / 'training', device_scale=3)
    with np.load(folder / 'endpoints.npz') as z:
        names = z['variable_names'].tolist()
        groups = {'raw_endpoints': z['raw_primal']}
    with np.load(folder / 'projections.npz') as z:
        groups['formal_projections'] = z['primal']
    with np.load(folder / 'controls.npz') as z:
        groups.update({key: z[key] for key in ('endpoint_primal', 'feasible_witnesses', 'primal')})
    checks = {}
    for label, primal in groups.items():
        maximum = 0.0
        for j, ess in enumerate(mg.ESS_list):
            ch = primal[:, [names.index(f'p_chg_ESS[{j},{t}]') for t in range(24)]]
            dis = primal[:, [names.index(f'p_dis_ESS[{j},{t}]') for t in range(24)]]
            soc = primal[:, [names.index(f'e_ESS[{j},{t}]') for t in range(25)]]
            expected = np.c_[np.full(len(primal), ess['init_SOC']),
                             ess['init_SOC'] + np.cumsum(
                                 mg.deltaT * (ess['eta_chg'] * ch - dis / ess['eta_dis'])
                                 / ess['E_rated'], axis=1)]
            maximum = max(maximum, float(np.max(np.abs(soc - expected))))
            assert np.min(soc) >= ess['SOC_min'] - 1e-8
            assert np.max(soc) <= ess['SOC_max'] + 1e-8
            assert np.min(soc[:, -1]) >= ess['init_SOC'] - 1e-8
        assert maximum < 1e-8
        checks[label] = {'passed': True, 'ESS_trajectories_checked': len(primal) * 3,
                         'max_cumulative_SOC_residual': maximum}
    write_json(output / 'independent_ESS_energy_verification.json',
               {'passed': True, 'physical_intervals': 24, 'state_boundaries': 25,
                'checks': checks, 'all_first_and_last_interval_powers_included': True})

    rows = []
    for key, label in [('local', '局部连线'), ('global', '全局连线'),
                       ('multi', '多点组合'), ('overall', '总体')]:
        group = summary['groups'][key]
        c = group['classification_counts']
        row = {'探针类型': label, '样本数': group['count']}
        for field, name in [('mean', '均值'), ('median', '中位数'), ('p95', 'P95'), ('max', '最大gap')]:
            row[f'{name}下界(%)'] = group['lower_percent'][field]
            row[f'{name}上界(%)'] = group['upper_percent'][field]
        row['超过0.01%的点数'] = c.get('detected_gap', 0)
        row['超过0.01%的比例(%)'] = 100 * row['超过0.01%的点数'] / group['count']
        row['阈值判定未确定数'] = c.get('undetermined', 0)
        row['最优性精度未达到数'] = group['convergence_counts']['not_converged']
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / 'gap_table.csv', index=False, encoding='utf-8-sig')

    def value(row, name):
        lo, hi = row[f'{name}下界(%)'], row[f'{name}上界(%)']
        low_text, high_text = f'{lo:.5f}', f'{hi:.5f}'
        return high_text if low_text == high_text else f'[{low_text}, {high_text}]'

    text = ['# 三倍设备场景非凸性诊断', '',
            '24时段；3 MT、3 PV、3 TL、3 CL、3 ESS、6 HP，共21台设备；360个二进制变量。',
            '单台参数与原始7设备算例完全一致；每台ESS为100 kW/250 kWh。固定名义场景，无随机扰动，conservativeness=0；不重新训练。',
            '使用修复MT最短运行时间、ESS全24时段能量递推后的现有设备模型。HP温度动态保持现有实现。',
            '1000个支撑方向(seed=0)生成端点，5000个凸组合探针(seed=1)，100个零间隙对照。',
            '距离单位为每时段固定功率跨度归一化的RMS百分比：100*sqrt(mean(((P-Q)/D)^2))。', '',
            '| 探针类型 | 样本数 | 均值(%) | 中位数(%) | P95(%) | 最大gap(%) | >0.01%点数 | 比例 | 未确定数 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['探针类型']} | {r['样本数']} | {value(r, '均值')} | {value(r, '中位数')} | {value(r, 'P95')} | {value(r, '最大gap')} | {r['超过0.01%的点数']} | {r['超过0.01%的比例(%)']:.2f}% | {r['阈值判定未确定数']} |")
    text += ['', '区间来自求解器目标上下界，非精确零值不修改为零。最大距离下界仅为全域非凸度的抽样下界，不是全域上界；检出比例仅描述本次抽样。',
             '完整原始整数变量在每次正式投影中均自由优化。独立原始约束、MT正向最短运行时间及ESS累计能量检查见同目录验证报告。',
             f"结果指纹：`{meta['fingerprint']}`。"]
    (output / 'gap_table.md').write_text('\n\n'.join(text[:8]) + '\n\n' + '\n'.join(text[8:]) + '\n', encoding='utf-8')
    print(json.dumps({'table': rows, 'ESS_validation': checks}, ensure_ascii=False))


def summarize_uptime_ablation(args):
    from pathlib import Path
    import json
    import numpy as np
    import pandas as pd

    base = args.base_dir
    old, out = base / 'nonconvexity', base / 'nonconvexity_no_min_uptime'
    output = args.output_dir or out
    output.mkdir(parents=True, exist_ok=True)
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    baseline, paired, fresh = [read(path) for path in
                               (old / 'summary.json', out / 'paired_summary.json', out / 'summary.json')]
    assert read(out / 'metadata.json')['status'] == 'completed'
    assert read(out / 'independent_storage_verification.json')['passed']
    assert paired['physical_validation']['passed']
    previous_samples, paired_samples = [pd.read_csv(path) for path in
                                         (old / 'samples.csv', out / 'paired_samples.csv')]
    worst = int(previous_samples.lower_percent.idxmax())
    rows = []
    for strategy, sampling, data in [('保留4时段最短运行时间', '原5000条轨迹', baseline),
                                    ('移除最短运行时间', '原5000条轨迹（配对）', paired),
                                    ('移除最短运行时间', '重新支撑采样5000条轨迹', fresh)]:
        for group_name, label in [('overall', '总体'), ('local', '局部连线'),
                                  ('global', '全局连线'), ('multi', '多点组合')]:
            group = data['groups'][group_name]
            counts = group['classification_counts']
            row = {'策略': strategy, '测试轨迹': sampling, '组别': label, '样本数': group['count']}
            for field, title in [('mean', '均值'), ('median', '中位数'), ('p95', 'P95'), ('max', '最大gap')]:
                for side, side_label in [('lower', '下界'), ('upper', '上界')]:
                    row[f'{title}{side_label}(%)'] = group[f'{side}_percent'][field]
            row['超过0.01%点数'] = counts.get('detected_gap', 0)
            row['超过0.01%比例(%)'] = 100 * row['超过0.01%点数'] / group['count']
            row['未确定数'] = counts.get('undetermined', 0)
            row['最优性未达到数'] = group['convergence_counts']['not_converged']
            rows.append(row)
    pd.DataFrame(rows).to_csv(output / 'min_uptime_comparison.csv', index=False, encoding='utf-8-sig')

    def display(row, name):
        lo, hi = [row[f'{name}{side}(%)'] for side in ('下界', '上界')]
        l, u = f'{lo:.5f}', f'{hi:.5f}'
        return u if l == u else f'[{l}, {u}]'

    text = ['# 三倍场景：MT最短运行时间消融', '',
            '场景：3 MT、3 PV、3 TL、3 CL、3 ESS、6 HP，24时段，360个二进制变量。',
            '仅移除3台MT共72条最短连续运行时间约束；最低出力20 kW、最高出力100 kW、启停状态转移和全部其他设备约束保留。每台ESS仍为100 kW/250 kWh。',
            '名义场景、无随机扰动、conservativeness=0，原始功率跨度归一化；阈值为0.01%。不重新训练，不松弛整数变量。', '',
            '主对比复用上一轮完全相同的5000个凸组合目标，原端点均在移除约束后的模型中重新验证可行。补充试验重新查询1000个支撑方向(seed=0)，生成5000个凸组合(seed=1)。', '',
            '| 策略 | 测试轨迹 | 均值(%) | P95(%) | 最大gap(%) | >0.01% | 未确定 |',
            '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['组别'] != '总体':
            continue
        text.append(f"| {row['策略']} | {row['测试轨迹']} | {display(row,'均值')} | {display(row,'P95')} | {display(row,'最大gap')} | {row['超过0.01%点数']}/5000（{row['超过0.01%比例(%)']:.2f}%） | {row['未确定数']} |")
    original_mean = baseline['groups']['overall']['upper_percent']['mean']
    new_mean = paired['groups']['overall']['upper_percent']['mean']
    original_max = baseline['groups']['overall']['upper_percent']['max']
    new_max = paired['groups']['overall']['upper_percent']['max']
    text += ['', f'配对均值减少{100*(1-new_mean/original_mean):.2f}%，样本最大值减少{100*(1-new_max/original_max):.2f}%。',
             f'上一轮最大gap的样本索引（从0开始）为{worst}，原gap为{previous_samples.loc[worst,"lower_percent"]:.8f}%，移除约束后距离上界为{paired_samples.loc[worst,"upper_percent"]:.8f}%。',
             '新采样和配对结果不可混称同一批测试点。完整分组及上下界保存在CSV中；求解与数值残差检查见验证报告。',
             f"配对结果未达到最优性精度的点数：{paired['groups']['overall']['convergence_counts']['not_converged']}；新采样未达到最优性精度的点数：{fresh['groups']['overall']['convergence_counts']['not_converged']}。阈值判定未确定数另列于表中；达到时间上限的点仍保留距离上下界，未人为改为零。",
             '结论：移除最短运行时间显著减小本次抽样非凸间隙，但仍有正间隙，不能断言整个聚合域凸。最大距离下界仅为全域非凸度的抽样下界，检出比例不是实际运行不可行概率。',
             '设备模型沿用前一轮实现；HP温度动态未在本次消融中修改。']
    (output / 'min_uptime_comparison.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(json.dumps([r for r in rows if r['组别']=='总体'], ensure_ascii=False))

def summarize_main(argv=None):
    parser = argparse.ArgumentParser(description="Update nonconvexity reporting summaries from saved intervals")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--threshold-percent", type=float, default=0.01)
    args = parser.parse_args(argv)
    print(json.dumps(summarize_nonconvexity(args.input_dir, args.threshold_percent)))


def scale3_summary_main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize scale3 nonconvexity and check saved ESS ledgers")
    parser.add_argument("--base-dir", type=Path, default=RESULTS_ROOT / "microgrid" / "scale3")
    parser.add_argument("--output-dir", type=Path)
    summarize_scale3_nonconvexity(parser.parse_args(argv))


def ablation_summary_main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize paired and fresh minimum-uptime ablations")
    parser.add_argument("--base-dir", type=Path, default=RESULTS_ROOT / "microgrid" / "scale3")
    parser.add_argument("--output-dir", type=Path)
    summarize_uptime_ablation(parser.parse_args(argv))


_COMMANDS = {
    "evaluate": evaluate_main,
    "ablation": ablation_main,
    "verify": verify_main,
    "verify-storage": verify_storage_main,
    "summarize": summarize_main,
    "scale3-summary": scale3_summary_main,
    "ablation-summary": ablation_summary_main,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in _COMMANDS:
        subparsers.add_parser(name, add_help=False, help=name.replace("-", " "))
    args, remaining = parser.parse_known_args(argv)
    return _COMMANDS[args.command](remaining)


if __name__ == "__main__":
    main()
