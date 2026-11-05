"""Solve the unchanged integer microgrid with redundant count branching.

Each native integer K[group,t] equals the sum of existing binary statuses.
These equalities preserve the original feasible set and objective. Disabling
presolve retains the redundant counts so their branching priorities take effect.
A fresh persistent bridge on every call captures current mutable parameters and
the active objective, including its constant, without retaining stale range rows.
"""

import math
import time
import warnings

import gurobipy as gp
import pyomo.environ as pyo
from pyomo.repn.standard_repn import generate_standard_repn


def _finite_attribute(model, name, *, integer=False):
    """Diagnostics must not mask an optimizer/setup failure or emit JSON NaN."""
    try:
        number = float(getattr(model, name))
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return int(number) if integer else number


def _check_linear_constraint(constraint, terms, constant, lower, upper):
    """Check the actual physical algebra before deriving a redundant cut."""
    if not constraint.active:
        raise ValueError(f"Required storage constraint is inactive: {constraint.name}")
    representation = generate_standard_repn(constraint.body, compute_values=True)
    if not representation.is_linear():
        raise ValueError(f"Storage constraint is not linear: {constraint.name}")
    actual = {id(var): float(coefficient) for var, coefficient in
              zip(representation.linear_vars, representation.linear_coefs) if float(coefficient) != 0.0}
    expected = {id(var): float(coefficient) for var, coefficient in terms if float(coefficient) != 0.0}
    if actual.keys() != expected.keys() or not all(
            math.isfinite(value) and math.isclose(value, expected[key], rel_tol=1e-12, abs_tol=1e-12)
            for key, value in actual.items()):
        raise ValueError(f"Storage coefficients disagree with supplied parameters: {constraint.name}")
    actual_constant = float(representation.constant or 0.0)
    if not math.isfinite(actual_constant) or not math.isclose(actual_constant, constant, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"Storage constant disagrees with supplied parameters: {constraint.name}")
    for expression, expected_bound in ((constraint.lower, lower), (constraint.upper, upper)):
        if expected_bound is None:
            if expression is not None:
                raise ValueError(f"Unexpected storage constraint bound: {constraint.name}")
        elif expression is None or not math.isclose(float(pyo.value(expression)), expected_bound,
                                                    rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Storage bound disagrees with supplied parameters: {constraint.name}")


class CountBranchingSolver:
    """Small SolverFactory-compatible adapter for original microgrid solves.

    ``options`` is intentionally mutable: ErrorCalculator's projection retries
    temporarily change ``Method``. No feasibility, integrality or optimality
    tolerance, and no MIP gap, is changed by this adapter.
    """

    DEFAULT_OPTIONS = {"Presolve": 0, "Threads": 8, "Seed": 0, "MIPFocus": 2}
    GROUPS = (("u_MT", "MTs"), ("I_TL", "TLs"), ("u_ESS", "ESSs"))
    TOLERANCE_NAMES = ("MIPGap", "MIPGapAbs", "FeasibilityTol", "IntFeasTol", "OptimalityTol")

    def __init__(self, options=None, storage_parameters=None, deltaT=1.0):
        self.options = {**self.DEFAULT_OPTIONS, **(options or {})}
        self.deltaT = float(deltaT)
        if not math.isfinite(self.deltaT) or self.deltaT <= 0:
            raise ValueError("Storage deltaT must be finite and positive")
        self.storage_parameters = None
        if storage_parameters is not None:
            fields = ("E_rated", "eta_chg", "eta_dis", "SOC_min", "SOC_max", "Pchg", "Pdis")
            self.storage_parameters = [{name: float(device[name]) for name in fields}
                                       for device in storage_parameters]
            for parameters in self.storage_parameters:
                if not all(math.isfinite(value) for value in parameters.values()):
                    raise ValueError("Storage parameters must be finite")
                if any(parameters[name] <= 0 for name in ("E_rated", "eta_chg", "eta_dis")):
                    raise ValueError("Storage energy and efficiencies must be positive")
                if parameters["SOC_min"] >= parameters["SOC_max"]:
                    raise ValueError("Storage SOC_min must be smaller than SOC_max")
                if parameters["Pchg"] < 0 or parameters["Pdis"] < 0:
                    raise ValueError("Storage charging/discharging power limits must be nonnegative")
                try:
                    energy, eta_chg, eta_dis = (parameters[name] for name in ("E_rated", "eta_chg", "eta_dis"))
                    factors = (energy / (self.deltaT * eta_chg), energy * eta_dis / self.deltaT,
                               self.deltaT * eta_chg / energy, self.deltaT / (energy * eta_dis))
                except (ZeroDivisionError, OverflowError) as exc:
                    raise ValueError("Derived storage headroom factors must be finite and positive") from exc
                if not all(math.isfinite(value) and value > 0 for value in factors):
                    raise ValueError("Derived storage headroom factors must be finite and positive")
        self.solve_records = []
        self.progress_callback = None
        self.diagnostics = {
            "backend": "gurobi_persistent_redundant_integer_count_branching",
            "fresh_native_model_per_solve": True,
            "native_model_closed_after_solve": True,
            "physical_integer_constraints": "unrelaxed",
            "count_definition": "K[group,t] = sum of original binary status variables over devices",
            "count_branch_priority": 100,
            "original_branch_priority": 0,
            "storage_headroom": self.storage_parameters is not None,
            "deltaT": self.deltaT,
            "storage_parameters": self.storage_parameters,
            "storage_headroom_t0_skipped": False,
            "storage_headroom_proof": (
                "For every power interval t=0..T-1, the transition from SOC[t] to SOC[t+1] "
                "and SOC bounds imply charging/discharging headroom bounds in each binary mode; "
                "the other mode has zero corresponding power. SOC[0] is the initial state."),
            "solve_count": 0,
            "optimal_solves": 0,
            "nonoptimal_solves": 0,
            "exception_count": 0,
            "optimizer_runtime_seconds": 0.0,
            "wall_seconds": 0.0,
        }

    @property
    def last_solve(self):
        return self.solve_records[-1] if self.solve_records else None

    def _storage_specs(self, model):
        if self.storage_parameters is None:
            return []
        devices = tuple(model.ESSs)
        times = tuple(model.T)
        if devices != tuple(range(len(self.storage_parameters))) or times != tuple(range(len(times))):
            raise ValueError("Storage parameters must match the model's consecutive device/time indices")
        if tuple(model.ESS_state_times) != tuple(range(len(times) + 1)):
            raise ValueError("Storage requires T+1 boundary SOC states for T power intervals")
        specs = []
        for i, parameters in enumerate(self.storage_parameters):
            energy, eta_chg, eta_dis = (parameters[key] for key in ("E_rated", "eta_chg", "eta_dis"))
            minimum, maximum = parameters["SOC_min"], parameters["SOC_max"]
            for boundary in model.ESS_state_times:
                _check_linear_constraint(getattr(model, f"ess_soc_limit_{i}")[boundary],
                                         [(model.e_ESS[i, boundary], 1.0)], 0.0, minimum, maximum)
            for t in times:
                previous, state = model.e_ESS[i, t], model.e_ESS[i, t + 1]
                charge, discharge, mode = (getattr(model, name)[i, t]
                                          for name in ("p_chg_ESS", "p_dis_ESS", "u_ESS"))
                if not mode.is_binary() or any(var.lb is None or float(pyo.value(var.lb)) < 0
                                               for var in (charge, discharge)):
                    raise ValueError("Storage headroom proof requires binary modes and nonnegative powers")
                _check_linear_constraint(getattr(model, f"ess_charging_power_limit_{i}")[t],
                                         [(charge, 1.0), (mode, -parameters["Pchg"])], 0.0, None, 0.0)
                _check_linear_constraint(getattr(model, f"ess_discharging_power_limit_{i}")[t],
                                         [(discharge, 1.0), (mode, parameters["Pdis"])],
                                         -parameters["Pdis"], None, 0.0)
                _check_linear_constraint(getattr(model, f"ess_energy_dynamics_{i}")[t],
                                         [(state, 1.0), (previous, -1.0),
                                          (charge, -self.deltaT * eta_chg / energy),
                                          (discharge, self.deltaT / (energy * eta_dis))], 0.0, 0.0, 0.0)
                specs.append((i, t, previous, charge, discharge, minimum, maximum,
                              energy / (self.deltaT * eta_chg), energy * eta_dis / self.deltaT))
        self.diagnostics["storage_parameters_verified_against_original_model"] = True
        return specs

    def solve(self, model, **kwargs):
        started = time.perf_counter()
        bridge = pyo.SolverFactory("gurobi_persistent")
        native = None
        count_rows = []
        storage_cut_rows = []
        original_binary_vars = []
        record = {"solve_index": len(self.solve_records) + 1,
                  "termination_condition": "exception"}
        result = None
        try:
            effective_options = dict(self.options)
            effective_options.update(kwargs.pop("options", {}))
            if "options_string" in kwargs:
                effective_options.update(bridge._options_string_to_dict(kwargs.pop("options_string")))
            # Count variables must survive presolve for the intended branching.
            if any(str(key).lower() == "presolve" and int(value) != 0
                   for key, value in effective_options.items()):
                raise ValueError("Count branching requires Presolve=0")
            effective_options["Presolve"] = 0
            bridge.options.update(effective_options)
            pyomo_variables = list(model.component_data_objects(pyo.Var, active=True))
            record.update(original_variables=len(pyomo_variables),
                          original_binary_variables=sum(var.is_binary() for var in pyomo_variables),
                          original_constraints=sum(1 for _ in model.component_data_objects(pyo.Constraint, active=True)))
            storage_specs = self._storage_specs(model)
            bridge.set_instance(model)
            native = bridge._solver_model
            native.setParam("OutputFlag", 0)
            native.update()
            original_native_shape = (native.NumVars, native.NumIntVars, native.NumBinVars, native.NumConstrs)
            record.update(native_original_variables=native.NumVars,
                          native_original_integer_variables=native.NumIntVars,
                          native_original_binary_variables=native.NumBinVars,
                          native_original_constraints=native.NumConstrs,
                          native_original_quadratic_constraints=native.NumQConstrs,
                          native_original_quadratic_objective_terms=native.NumQNZs)
            for var in native.getVars():
                var.BranchPriority = 0
            native_map = bridge._pyomo_var_to_solver_var_map
            original_binary_vars = [native_map[var] for var in pyomo_variables if var.is_binary()]
            for component_name, device_set_name in self.GROUPS:
                component = getattr(model, component_name)
                devices = tuple(getattr(model, device_set_name))
                if not devices:
                    continue
                for t in model.T:
                    original_binaries = [component[i, t] for i in devices]
                    if not all(var.is_binary() for var in original_binaries):
                        raise ValueError(f"{component_name} contains a nonbinary status variable")
                    native_binaries = [native_map[var] for var in original_binaries]
                    count = native.addVar(lb=0, ub=len(devices), vtype=gp.GRB.INTEGER,
                                          name=f"_branch_count_{component_name}_{t}")
                    count.BranchPriority = 100
                    native.addConstr(count == gp.quicksum(native_binaries),
                                     name=f"_branch_count_equality_{component_name}_{t}")
                    count_rows.append((count, native_binaries))
            for i, t, previous, charge, discharge, minimum, maximum, charge_factor, discharge_factor in storage_specs:
                previous_native = native_map[previous]
                storage_cut_rows.append(native.addConstr(
                    native_map[charge] + charge_factor * previous_native <= charge_factor * maximum,
                    name=f"_storage_charge_headroom_{i}_{t}"))
                storage_cut_rows.append(native.addConstr(
                    native_map[discharge] - discharge_factor * previous_native <= -discharge_factor * minimum,
                    name=f"_storage_discharge_headroom_{i}_{t}"))
            native.update()
            added = len(count_rows)
            added_storage = len(storage_cut_rows)
            if ((native.NumVars - original_native_shape[0],
                 native.NumIntVars - original_native_shape[1],
                 native.NumBinVars - original_native_shape[2],
                 native.NumConstrs - original_native_shape[3]) != (added, added, 0, added + added_storage)):
                raise RuntimeError("Unexpected variable or constraint change while adding redundant counts")
            record.update(added_integer_counts=added, added_count_equalities=added,
                          storage_headroom=self.storage_parameters is not None,
                          storage_headroom_t0_skipped=False,
                          added_storage_headroom_cuts=added_storage,
                          native_variables=native.NumVars,
                          native_integer_variables=native.NumIntVars,
                          native_binary_variables=native.NumBinVars,
                          native_constraints=native.NumConstrs)
            for key, value in effective_options.items():
                native.setParam(key, value)
            record["build_seconds"] = time.perf_counter() - started
            record["method"] = int(native.Params.Method)
            record["presolve"] = int(native.Params.Presolve)
            record["threads"] = int(native.Params.Threads)
            record["seed"] = int(native.Params.Seed)
            record["mip_focus"] = int(native.Params.MIPFocus)
            record["symmetry"] = int(native.Params.Symmetry)
            tolerances = {name: float(getattr(native.Params, name)) for name in self.TOLERANCE_NAMES}
            self.diagnostics.setdefault("actual_tolerances", tolerances)
            if tolerances != self.diagnostics["actual_tolerances"]:
                record["actual_tolerances"] = tolerances
            # The plugin loads only mapped original Pyomo variables; native-only
            # K variables do not enter the physical model or its saved solution.
            result = bridge.solve(model, **kwargs)
            record["termination_condition"] = str(result.solver.termination_condition)
            return result
        except Exception as exc:
            record["failure"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if native is not None:
                record["status"] = _finite_attribute(native, "Status", integer=True)
                record["optimizer_runtime_seconds"] = _finite_attribute(native, "Runtime")
                record["solution_count"] = _finite_attribute(native, "SolCount", integer=True)
                record["bound"] = _finite_attribute(native, "ObjBound")
                record["node_count"] = _finite_attribute(native, "NodeCount")
                if record["solution_count"]:
                    record["objective"] = _finite_attribute(native, "ObjVal")
                    record["relative_gap"] = _finite_attribute(native, "MIPGap")
                    record["absolute_gap"] = (abs(record["objective"] - record["bound"])
                                               if record["objective"] is not None and record["bound"] is not None
                                               else None)
                    record["constraint_violation"] = _finite_attribute(native, "ConstrVio")
                    record["bound_violation"] = _finite_attribute(native, "BoundVio")
                    try:
                        residuals = {
                            "max_count_equality_residual": max(
                                (abs(count.X - sum(var.X for var in binaries)) for count, binaries in count_rows), default=0.0),
                            "max_count_integrality_residual": max(
                                (abs(count.X - round(count.X)) for count, _ in count_rows), default=0.0),
                            "max_original_binary_integrality_residual": max(
                                (abs(var.X - round(var.X)) for var in original_binary_vars), default=0.0),
                            "max_storage_headroom_violation": max(
                                (max(0.0, -constraint.Slack) for constraint in storage_cut_rows), default=0.0),
                        }
                        record.update({key: value if math.isfinite(value) else None
                                       for key, value in residuals.items()})
                    except Exception as exc:
                        record["solution_diagnostics_failure"] = f"{type(exc).__name__}: {exc}"
            close_error = None
            try:
                bridge.close()
                record["native_model_closed"] = True
            except Exception as exc:
                record["native_model_closed"] = False
                record["close_failure"] = f"{type(exc).__name__}: {exc}"
                self.diagnostics["native_model_closed_after_solve"] = False
                if "failure" not in record:
                    close_error = exc
                    record["failure"] = record["close_failure"]
            record["wall_seconds"] = time.perf_counter() - started
            self.solve_records.append(record)
            self.diagnostics["solve_count"] += 1
            if record["termination_condition"] == "optimal":
                self.diagnostics["optimal_solves"] += 1
            else:
                self.diagnostics["nonoptimal_solves"] += 1
            if "failure" in record:
                self.diagnostics["exception_count"] += 1
            self.diagnostics["optimizer_runtime_seconds"] += record.get("optimizer_runtime_seconds") or 0.0
            self.diagnostics["wall_seconds"] += record["wall_seconds"]
            if self.progress_callback is not None:
                try:
                    self.progress_callback(record, self.diagnostics)
                except Exception as exc:
                    record["progress_callback_failure"] = f"{type(exc).__name__}: {exc}"
                    try:
                        warnings.warn(f"Count solver progress callback failed: {exc}", RuntimeWarning, stacklevel=2)
                    except Exception:
                        # Even a warnings-as-errors policy must not replace the
                        # actual optimizer result or the original solve error.
                        pass
            if close_error is not None:
                raise close_error
