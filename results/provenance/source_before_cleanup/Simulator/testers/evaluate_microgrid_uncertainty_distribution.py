"""Paired mean/variance-shift evaluation of one frozen microgrid polytope.

Each sample pairs one standard-normal direction with one random resource
realization. Six conditions share cached aggregate commands and Gaussian
innovations. No training, clipping, resampling or nominal-cost comparison is
performed.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.core.expr.visitor import identify_mutable_parameters
import torch

from Simulator.Approximator import PreTrainNet, pyomo_params_to_numpy
from Simulator.cases.microgrid_case import MicroGrid
from Simulator.paths import CODE_ROOT, DATA_ROOT, RESULTS_ROOT, comparison_output_path


CONDITIONS = (
    ("A", 0.0, 0.05), ("B", -0.05, 0.05), ("C", 0.05, 0.05),
    ("D", 0.0, 0.10), ("E", -0.05, 0.10), ("F", 0.05, 0.10),
)
PARAMETERS = ("p_PV_rated", "p_CL_rated", "temp_amb")
PREFIX = "distribution_shift"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def replace_file(temporary, target):
    """Allow brief Windows sync/read locks without losing a completed snapshot."""
    for attempt in range(6):
        try:
            temporary.replace(target)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.1 * 2 ** attempt)


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    replace_file(temporary, path)


def load_frozen_microgrid(training_dir, conservativeness=0.0):
    """Load learned A/b before building the case, avoiding bound initialization."""
    with (training_dir / "metadata.pkl").open("rb") as stream:
        metadata = pickle.load(stream)
    if int(metadata.get("T", -1)) != 24:
        raise ValueError("This experiment requires the retained T=24 microgrid")
    expected = {"MT_list": 1, "PV_list": 1, "TL_list": 1,
                "CL_list": 1, "ESS_list": 1, "HP_list": 2}
    if any(len(metadata.get(name, [])) != count for name, count in expected.items()):
        raise ValueError("Training metadata has a different microgrid device portfolio")
    state = torch.load(training_dir / "pretrainnet_weights.pth",
                       map_location="cpu", weights_only=True)
    bounds = state["b_net.0.bias"]
    matrix = state["A_net.0.bias"].reshape(len(bounds), metadata["T"])
    net = PreTrainNet(matrix.numpy(), bounds.numpy(), device="cpu").to(dtype=matrix.dtype)
    net.load_state_dict(state)
    with torch.no_grad():
        A, b = net()
    A, b = A[0].numpy().copy(), b[0].numpy().copy()
    mg = MicroGrid(seed=42, T=metadata["T"], data=metadata)
    mg.case_mg(A=A, b=b, conservativeness=conservativeness, is_uncertain=False, device="cpu")
    for key in ("p_min", "p_max"):
        np.testing.assert_array_equal(getattr(mg, key.replace("_", "")), metadata[key])
    return mg, A, b


class _Domain:
    def __init__(self, source):
        import gurobipy as gp

        self.source = source.clone()
        for objective in self.source.component_data_objects(pyo.Objective, active=True):
            objective.deactivate()
        sides = pyo.ConstraintList()
        self.source.add_component('_persistent_range_sides', sides)
        ranges = [constraint for constraint in
                  self.source.component_data_objects(pyo.Constraint, active=True)
                  if constraint.has_lb() and constraint.has_ub() and not constraint.equality]
        for constraint in ranges:
            sides.add((constraint.lower, constraint.body, None))
            sides.add((None, constraint.body, constraint.upper))
            constraint.deactivate()
        self.bridge = pyo.SolverFactory('gurobi_persistent')
        self.bridge.set_instance(self.source)
        self.model = self.bridge._solver_model
        for key, value in {'OutputFlag': 0, 'Threads': 1, 'Seed': 0}.items():
            self.model.setParam(key, value)
        self.x = [self.bridge._pyomo_var_to_solver_var_map[self.source.var_proj[j]]
                  for j in range(len(self.source.var_proj))]
        self.quadratic = gp.quicksum(variable * variable for variable in self.x)
        self.model.update()
        self.shape = (self.model.NumVars, self.model.NumConstrs)
        self.options = {name: float(getattr(self.model.Params, name)) for name in
                        ('MIPGap', 'MIPGapAbs', 'FeasibilityTol', 'OptimalityTol', 'IntFeasTol')}
        self.options.update(Threads=1, Seed=0, OutputFlag=0)

    def solve(self, vector, projection):
        import gurobipy as gp

        started = time.perf_counter()
        diagnostics = {'status': None, 'status_name': 'not_solved', 'runtime': 0.0,
                       'mip_gap': None, 'best_bound': None, 'objective': None,
                       'constraint_violation': None, 'bound_violation': None,
                       'integer_violation': None, 'max_violation': None,
                       'solution_count': 0, 'is_mip': bool(self.model.IsMIP)}
        try:
            vector = np.asarray(vector, dtype=float)
            if vector.shape != (len(self.x),) or not np.all(np.isfinite(vector)):
                raise ValueError('Expected a finite direction/target of length T')
            if projection:
                objective = self.quadratic + gp.LinExpr((-2.0 * vector).tolist(), self.x)
                objective += float(vector @ vector)
            else:
                objective = gp.LinExpr(vector.tolist(), self.x)
            self.model.setObjective(objective, gp.GRB.MINIMIZE)
            # Avoid carrying a prior scenario's incumbent or basis into a pair.
            self.model.reset(0)
            self.model.optimize()
            status = int(self.model.Status)
            status_names = {int(getattr(gp.GRB, name)): name for name in
                            ('LOADED', 'OPTIMAL', 'INFEASIBLE', 'INF_OR_UNBD', 'UNBOUNDED',
                             'CUTOFF', 'ITERATION_LIMIT', 'NODE_LIMIT', 'TIME_LIMIT',
                             'SOLUTION_LIMIT', 'INTERRUPTED', 'NUMERIC', 'SUBOPTIMAL',
                             'INPROGRESS', 'USER_OBJ_LIMIT')}
            diagnostics.update(status=status, status_name=status_names.get(status, str(status)),
                               runtime=float(self.model.Runtime),
                               solution_count=int(self.model.SolCount))
            if self.model.IsMIP:
                bound = float(self.model.ObjBound)
                diagnostics['best_bound'] = bound if np.isfinite(bound) else None
            if self.model.SolCount:
                objective_value = float(self.model.ObjVal)
                diagnostics['objective'] = objective_value if np.isfinite(objective_value) else None
                if self.model.IsMIP:
                    gap = float(self.model.MIPGap)
                    diagnostics['mip_gap'] = gap if np.isfinite(gap) else None
                elif status == gp.GRB.OPTIMAL:
                    diagnostics['best_bound'] = diagnostics['objective']
                violations = [float(self.model.ConstrVio), float(self.model.BoundVio),
                              float(self.model.IntVio) if self.model.IsMIP else 0.0]
                diagnostics.update(constraint_violation=violations[0], bound_violation=violations[1],
                                   integer_violation=violations[2], max_violation=max(violations))
            if status != gp.GRB.OPTIMAL:
                return None, diagnostics
            point = np.asarray(self.model.getAttr('X', self.x), dtype=float)
            if not np.all(np.isfinite(point)):
                raise RuntimeError('Gurobi returned nonfinite aggregate coordinates')
            return point, diagnostics
        except Exception as exc:
            diagnostics.update(failure=f'{type(exc).__name__}: {exc}')
            if diagnostics['status'] is None:
                diagnostics['status_name'] = 'exception'
            return None, diagnostics
        finally:
            diagnostics['wall_seconds'] = time.perf_counter() - started

    def close(self):
        self.model.dispose()


class PersistentShiftSolver:
    """Freeze the fitted polytope; restore nominal physical rows after each call.

    ``mg`` must already contain its fitted A/b and the nominal resource model.
    Native models use Gurobi's default numerical tolerances and MIP gaps, with
    one thread and a fixed solver seed. No integer variables are relaxed.
    """

    def __init__(self, mg):
        import gurobipy as gp

        self.original = _Domain(mg.errorcalculator.original_model)
        try:
            self.approximation = _Domain(mg.errorcalculator.approx_model)
        except Exception:
            self.original.close()
            raise
        self._closed = False
        self._parameters = {param.name: param for param in
                            self.original.source.component_objects(pyo.Param, descend_into=True)
                            if param.mutable}
        self._nominal = {name: ({index: float(pyo.value(param[index])) for index in param}
                                if param.is_indexed() else float(pyo.value(param)))
                         for name, param in self._parameters.items()}
        self._parameter_rows = {name: [] for name in self._parameters}
        self._all_rows = list(self.original.source.component_data_objects(pyo.Constraint, active=True))
        for constraint in self._all_rows:
            names = set()
            for expression in (constraint.body, constraint.lower, constraint.upper):
                if expression is not None:
                    names.update(param.parent_component().name
                                 for param in identify_mutable_parameters(expression))
            for name in names:
                self._parameter_rows[name].append(constraint)
        self.solver_metadata = {
            'backend': 'gurobi_persistent_remove_add_mutable_rows',
            'gurobi_version': list(gp.gurobi.version()),
            'original_options': self.original.options,
            'approximation_options': self.approximation.options,
            'original_variables': int(self.original.model.NumVars),
            'original_rows': int(self.original.model.NumConstrs),
            'original_binary_variables': int(self.original.model.NumBinVars),
            'original_integer_variables': int(self.original.model.NumIntVars),
            'projection_objective': 'sum((normalized_power - target_normalized_power)**2)',
            'projection_objective_includes_constant': True,
            'integer_relaxation': False,
            'range_constraints_split_into_equivalent_scalar_rows': True,
            'parameter_clipping': False,
            'require_status': 'OPTIMAL',
        }

    def support(self, direction):
        if self._closed:
            raise RuntimeError('PersistentShiftSolver is closed')
        return self.approximation.solve(direction, projection=False)

    @staticmethod
    def _set_parameter(param, values):
        if not param.is_indexed():
            value = float(np.asarray(values))
            if not np.isfinite(value):
                raise ValueError(f'Nonfinite parameter {param.name}')
            param.set_value(value)
        elif isinstance(values, dict):
            for index, number in values.items():
                number = float(number)
                if not np.isfinite(number):
                    raise ValueError(f'Nonfinite parameter {param.name}[{index}]')
                param[index].set_value(number)
        else:
            array = np.asarray(values, dtype=float)
            indices = list(param.index_set())
            if array.size != len(indices) or not np.all(np.isfinite(array)):
                raise ValueError(f'Invalid values for parameter {param.name}')
            for index, number in zip(indices, array.flat):
                param[index].set_value(float(number))

    def _remove_rows(self, rows):
        mapping = self.original.bridge._pyomo_con_to_solver_con_map
        for constraint in rows:
            if constraint in mapping:
                self.original.bridge.remove_constraint(constraint)

    def project(self, target_normalized, updates):
        if self._closed:
            raise RuntimeError('PersistentShiftSolver is closed')
        started = time.perf_counter()
        names = list(updates)
        unknown = set(names) - self._parameters.keys()
        if unknown:
            return None, {'status': None, 'status_name': 'invalid_parameter',
                          'failure': f'Unknown mutable parameters: {sorted(unknown)}',
                          'wall_seconds': time.perf_counter() - started}
        affected_ids = {id(row) for name in names for row in self._parameter_rows[name]}
        rows = [row for row in self._all_rows if id(row) in affected_ids]
        point, diagnostics = None, {}
        try:
            self._remove_rows(rows)
            for name in names:
                self._set_parameter(self._parameters[name], updates[name])
            for constraint in rows:
                self.original.bridge.add_constraint(constraint)
            point, diagnostics = self.original.solve(target_normalized, projection=True)
        except Exception as exc:
            diagnostics.update(status=None, status_name='parameter_update_failed',
                               failure=f'{type(exc).__name__}: {exc}')
        finally:
            try:
                self._remove_rows(rows)
                for name in names:
                    self._set_parameter(self._parameters[name], self._nominal[name])
                for constraint in rows:
                    self.original.bridge.add_constraint(constraint)
                self.original.model.update()
                shape = (self.original.model.NumVars, self.original.model.NumConstrs)
                if shape != self.original.shape:
                    raise RuntimeError(f'Persistent model size changed: {self.original.shape} -> {shape}')
                diagnostics['nominal_restored'] = True
            except Exception as exc:
                point = None
                diagnostics.update(nominal_restored=False, restore_failure=f'{type(exc).__name__}: {exc}')
            diagnostics['updated_constraint_objects'] = len(rows)
            diagnostics['total_wall_seconds'] = time.perf_counter() - started
        return point, diagnostics

    def close(self):
        if not self._closed:
            self.original.close()
            self.approximation.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class PyomoShiftSolver:
    """Unchanged reference solver, with nominal parameters restored each call."""

    def __init__(self, mg):
        self.calculator = mg.errorcalculator
        parameters, _ = pyomo_params_to_numpy(mg.model)
        self.nominal = {name: np.array(parameters[name]["initial_value"], copy=True)
                        for name in PARAMETERS}
        for solver in (self.calculator.solver, self.calculator.cvx_solver):
            solver.options.update({"Threads": 1, "Seed": 0})
        self.solver_metadata = {"backend": "pyomo", "Threads": 1, "Seed": 0,
                                "tolerances": "Gurobi defaults"}

    def support(self, direction):
        start = time.perf_counter()
        point = self.calculator.optimize_direction(direction, in_approx=True)
        return point, {"runtime": time.perf_counter() - start,
                       "status_name": "OPTIMAL" if point is not None else "FAILED"}

    def project(self, target, updates):
        start = time.perf_counter()
        try:
            self.calculator.update_parameters(updates)
            point = self.calculator.project(target, to_approx=False)
            return point, {"runtime": time.perf_counter() - start,
                           "status_name": "OPTIMAL" if point is not None else "FAILED"}
        finally:
            self.calculator.update_parameters(self.nominal)

    def close(self):
        self.calculator.update_parameters(self.nominal)


def generate_inputs(mg, count, seed):
    """Pair each randn direction with independent physical Gaussian innovations."""
    streams = np.random.SeedSequence(seed).spawn(4)
    bases = {
        "p_PV_rated": np.asarray([d["output_profile"] for d in mg.PV_list]),
        "p_CL_rated": np.asarray([d["p_max_profile"] for d in mg.CL_list]),
        "temp_amb": np.asarray(mg.temp_amb),
    }
    # A local randn generator samples every sign combination without changing
    # global RNG state or the three existing resource-innovation streams.
    direction_rng = np.random.RandomState(streams[0].generate_state(1)[0])
    arrays = {"directions": direction_rng.randn(count, mg.T)}
    for name, stream in zip(PARAMETERS, streams[1:]):
        arrays["base_" + name] = bases[name]
        arrays["z_" + name] = np.random.default_rng(stream).standard_normal(
            (count, *bases[name].shape))
    arrays.update(A=mg.errorcalculator.A_hat, b=mg.errorcalculator.b_hat,
                  pmin=mg.pmin, pmax=mg.pmax,
                  pv_capacity=np.asarray([d["P_max"] for d in mg.PV_list]))
    return arrays


def scenario_parameters(inputs, index, delta, sigma):
    return {name: inputs["base_" + name] *
            (1.0 + delta + sigma * inputs["z_" + name][index]) for name in PARAMETERS}


def parameter_flags(updates, capacities):
    pv, cl = updates["p_PV_rated"], updates["p_CL_rated"]
    return {
        "negative_pv_values": int(np.count_nonzero(pv < 0)),
        "negative_cl_values": int(np.count_nonzero(cl < 0)),
        "pv_above_capacity_values": int(np.count_nonzero(pv > capacities[:, None])),
        "nonfinite_parameter_values": int(sum(np.count_nonzero(~np.isfinite(v))
                                               for v in updates.values())),
    }


def relative_error_percent(physical_squared_error, power_squared_sum):
    """Relative physical L2 error; a zero denominator is undefined, not zero."""
    if not np.isfinite(power_squared_sum) or power_squared_sum <= 0:
        return None
    if not np.isfinite(physical_squared_error) or physical_squared_error < 0:
        raise ValueError("Physical squared disaggregation error must be finite and nonnegative")
    return float(100.0 * np.sqrt(physical_squared_error / power_squared_sum))


def perturbation_statistics(inputs, count):
    statistics = {}
    for condition, delta, sigma in CONDITIONS:
        statistics[condition] = {}
        for name in PARAMETERS:
            z = inputs["z_" + name].reshape(-1)
            residual = delta + sigma * z
            n = len(z)
            mean, std = float(residual.mean()), float(residual.std(ddof=1))
            mean_tolerance = 5 * sigma / np.sqrt(n)
            std_tolerance = 5 * sigma / np.sqrt(2 * (n - 1)) if n > 1 else None
            statistics[condition][name] = {
                "relative_mean": mean, "relative_std_ddof1": std,
                "expected_relative_mean": delta, "expected_relative_std": sigma,
                "variance_multiplier": (sigma / 0.05) ** 2,
                "five_standard_error_mean_tolerance": float(mean_tolerance),
                "five_standard_error_std_tolerance": None if std_tolerance is None else float(std_tolerance),
                "monte_carlo_moment_check": bool(abs(mean - delta) <= mean_tolerance and
                    (std_tolerance is None or abs(std - sigma) <= std_tolerance)),
            }
    return statistics


def summarize(samples):
    summaries = []
    for condition, delta, sigma in CONDITIONS:
        rows = samples[samples.condition == condition]
        completed = rows.solver_status.eq("completed")
        percentages = pd.to_numeric(rows["relative_disaggregation_error_percent"], errors="coerce")
        valid = completed & np.isfinite(percentages)
        values = percentages[valid].to_numpy(dtype=float)
        row = {"condition": condition, "mean_shift": delta, "noise_scale": sigma,
               "variance_multiplier": (sigma / 0.05) ** 2,
               "recorded_samples": len(rows), "valid_samples": len(values),
               "is_original_baseline": condition == "A", "metric_unit": "percent",
               "solver_completed_samples": int(completed.sum()),
               "failed_samples": int((~completed).sum()),
               "undefined_relative_error_samples": int((completed & ~valid).sum()),
               "solver_failures": int(rows.solver_status.eq("projection_failed").sum()),
               "invalid_parameter_samples": int(rows.solver_status.eq("invalid_parameters").sum()),
               "support_failures": int(rows.solver_status.eq("approximate_support_failed").sum())}
        for name, function in (("mean", np.mean), ("median", np.median),
                               ("p95", lambda x: np.quantile(x, .95)), ("max", np.max)):
            row[name] = float(function(values)) if len(values) else np.nan
            for prefix, column in (("normalized_squared", "feasibility_error_rated"),
                                   ("physical_squared", "feasibility_error")):
                raw = rows.loc[completed, column].to_numpy(dtype=float)
                row[f"{prefix}_{name}"] = float(function(raw)) if len(raw) else np.nan
        for key in ("negative_pv_values", "negative_cl_values", "pv_above_capacity_values",
                    "nonfinite_parameter_values"):
            row[key] = int(rows[key].sum())
            row[key.replace("_values", "_samples")] = int(rows[key].gt(0).sum())
        summaries.append(row)
    return pd.DataFrame(summaries)


def paired_comparisons(samples):
    results = {}
    for first, second in (("A", "B"), ("A", "C"), ("A", "D"), ("B", "E"), ("C", "F")):
        metric = "relative_disaggregation_error_percent"
        columns = ["sample_index", metric]
        valid = samples.solver_status.eq("completed") & np.isfinite(
            pd.to_numeric(samples[metric], errors="coerce"))
        a = samples[(samples.condition == first) & valid][columns]
        b = samples[(samples.condition == second) & valid][columns]
        paired = a.merge(b, on="sample_index", suffixes=("_first", "_second"))
        change = paired[metric + "_second"] - paired[metric + "_first"]
        results[first + "_to_" + second] = {
            "unit": "percentage_points",
            "valid_pairs": len(change),
            "mean_change": float(change.mean()) if len(change) else None,
            "median_change": float(change.median()) if len(change) else None,
            "p95_change": float(change.quantile(.95)) if len(change) else None,
        }
    return results


def evaluate(args):
    output = comparison_output_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    training = args.training_dir.resolve()
    training_config_path = training / "training_config.json"
    training_config = {}
    if training_config_path.is_file():
        training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
        if training_config.get("status") != "completed":
            raise ValueError("The selected training run is not completed")
    protected = {"training_weights": file_hash(training / "pretrainnet_weights.pth"),
                 "training_metadata": file_hash(training / "metadata.pkl"),
                 "physical_profiles": file_hash(DATA_ROOT / "profiles_data" / "profiles_data.npz")}
    if training_config:
        protected["training_config"] = file_hash(training_config_path)
    backend_file = Path(__file__).resolve()
    protocol = {"num_samples": args.num_samples, "seed": args.seed,
                "direction_distribution": "standard_normal",
                "direction_rng": "RandomState(SeedSequence(seed).spawn(4)[0].generate_state(1)[0]).randn(num_samples,T)",
                "sampling_design": "One direction paired with one resource realization per sample_index, not a Cartesian product",
                "evaluation_conservativeness": args.conservativeness,
                "conditions": [dict(condition=c, mean_shift=d, noise_scale=s)
                               for c, d, s in CONDITIONS], "backend": args.backend,
                "input_sha256": protected, "runner_sha256": file_hash(__file__),
                "backend_sha256": file_hash(backend_file) if args.backend == "persistent" else None,
                "case_source_sha256": file_hash(Path(__file__).parents[1] / "cases" / "microgrid_case.py")}
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode("utf-8")).hexdigest()
    archive = output / f"{PREFIX}_data.npz"
    meta_path = output / f"{PREFIX}_metadata.json"
    old = None
    if args.resume:
        if not archive.is_file() or not meta_path.is_file():
            raise FileNotFoundError("--resume requires this experiment's data and metadata")
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        if previous["protocol_signature"] != signature:
            raise ValueError(
                "Cannot resume: inputs, settings or evaluation source changed. "
                "Keep historical provenance unchanged; replot completed data with "
                "python -m Simulator.drawers.plot_microgrid_shift <result-directory>."
            )
        with np.load(archive, allow_pickle=False) as data:
            old = {key: data[key] for key in data.files}
    else:
        # Only owned fixed-name exports are replaced; legacy results are preserved.
        # Replace the PDF only when the new evaluations are ready to plot.
        # An open viewer must not prevent the numerical experiment from running.
        for path in (archive, meta_path, output / f"{PREFIX}_samples.csv",
                     output / f"{PREFIX}_summary.csv", output / f"{PREFIX}_verification.json"):
            if path.is_file():
                path.unlink()

    mg, A, b = load_frozen_microgrid(training, args.conservativeness)
    inputs = generate_inputs(mg, args.num_samples, args.seed)
    n, T = args.num_samples, mg.T
    commands = np.full((n, T), np.nan)
    support_done = np.zeros(n, dtype=bool)
    projected = np.full((6, n, T), np.nan)
    done = np.zeros((6, n), dtype=bool)
    rows = []
    if old is not None:
        for name, value in inputs.items():
            np.testing.assert_array_equal(value, old[name])
        commands, support_done = old["commands"], old["support_done"]
        projected, done = old["projected_curves"], old["done"]
        rows = json.loads(old["rows_json"].item())
        if len(rows) != np.count_nonzero(done):
            raise ValueError("Checkpoint sample rows do not match completed evaluations")
    solver = None
    start = time.perf_counter()
    metadata = {"status": "running", "protocol_signature": signature, **protocol,
                "training_conservativeness": training_config.get("conservativeness"),
                "training_margin_note": "Training margin is distinct from physical evaluation conservativeness",
                "training_directory": str(training), "output_directory": str(output),
                "horizon": T, "device_counts": {name: len(getattr(mg, name + "_list"))
                    for name in ("MT", "PV", "TL", "CL", "ESS", "HP")},
                "direction_definition": "Independent N(0,I_T) vectors from randn; no sign restriction, normalization or radial scaling",
                "direction_statistics": {
                    "mean": float(inputs["directions"].mean()),
                    "std_ddof1": float(inputs["directions"].std(ddof=1)),
                    "positive_components": int(np.count_nonzero(inputs["directions"] > 0)),
                    "negative_components": int(np.count_nonzero(inputs["directions"] < 0)),
                    "mixed_sign_vectors": int(np.count_nonzero(
                        np.any(inputs["directions"] > 0, axis=1) &
                        np.any(inputs["directions"] < 0, axis=1))),
                },
                "pairing": "One cached command and one set of base Gaussian arrays per sample_index; shared by A-F",
                "error_definition": "100*sqrt(sum((P_command-P_projected)**2)/sum(P_command**2)) in physical power coordinates",
                "summary_metric": "relative_disaggregation_error_percent",
                "percentage_denominator": "Sum of squares of the cached commanded physical aggregate power trajectory",
                "zero_denominator_policy": "Relative error undefined (missing), raw squared errors and solver status retained",
                "legacy_normalized_error_definition": "Sum of squared aggregate differences in the fixed nominal normalized coordinates",
                "physical_error_definition": "Physical-power squared difference for that same normalized-distance projection",
                "original_baseline": {"condition": "A", "mean_shift": 0.0, "noise_scale": 0.05,
                    "source": "Original microgrid test defaults; re-evaluated together with all five shifted conditions",
                    "legacy_spreadsheet_not_concatenated": "Historical file lacks power-trajectory denominators and has fewer evaluations"},
                "figure_definition": {"kind": "violin", "y_scale": "linear", "unit": "percent",
                                      "raw_values_not_floored": True},
                "perturbation_definition": "x=base*(1+mean_shift+noise_scale*z), with no clipping or resampling",
                "temperature_note": "Negative Celsius baseline: positive relative mean shift is colder",
                "scope": "Empirical disaggregation of sampled commands; no containment or all-directions guarantee",
                "perturbation_statistics": perturbation_statistics(inputs, n),
                "array_sha256": {name: array_hash(value) for name, value in inputs.items()}}

    def save(status):
        temporary = archive.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **inputs, commands=commands, support_done=support_done,
                                projected_curves=projected, done=done,
                                power_squared_sum=np.sum(commands ** 2, axis=1),
                                rows_json=np.asarray(json.dumps(rows, allow_nan=False)))
        replace_file(temporary, archive)
        if rows:
            samples = pd.DataFrame(rows)
            samples.to_csv(output / f"{PREFIX}_samples.csv", index=False)
            summarize(samples).to_csv(output / f"{PREFIX}_summary.csv", index=False)
        metadata.update(status=status, support_queries_recorded=int(support_done.sum()),
                        recorded_samples=len(rows),
                        completed_samples=sum(r["solver_status"] == "completed" for r in rows),
                        seconds_this_invocation=time.perf_counter() - start,
                        updated_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(meta_path, metadata)

    try:
        if args.backend == "persistent":
            solver = PersistentShiftSolver(mg)
        else:
            solver = PyomoShiftSolver(mg)
        metadata["solver"] = solver.solver_metadata
        for i, direction in enumerate(inputs["directions"]):
            if support_done[i]:
                continue
            point, info = solver.support(direction)
            if point is not None and np.all(np.isfinite(point)):
                commands[i] = point * (mg.pmax - mg.pmin) + mg.pmin
            support_done[i] = True
            if (i + 1) % args.checkpoint_every == 0 or i + 1 == n:
                save("generating_commands")
                print(f"Cached commands: {i + 1}/{n}", flush=True)
        for ci, (condition, delta, sigma) in enumerate(CONDITIONS):
            for i in range(n):
                if done[ci, i]:
                    continue
                updates = scenario_parameters(inputs, i, delta, sigma)
                flags = parameter_flags(updates, inputs["pv_capacity"])
                power_squared_sum = float(np.sum(commands[i] ** 2)) if np.all(
                    np.isfinite(commands[i])) else None
                row = {"condition": condition, "sample_index": i, "mean_shift": delta,
                       "noise_scale": sigma, "variance_multiplier": (sigma / .05) ** 2,
                       "is_original_baseline": condition == "A",
                       "solver_status": "pending", "failure": "",
                       "feasibility_error": None, "feasibility_error_rated": None,
                       "power_squared_sum": power_squared_sum,
                       "relative_disaggregation_error_percent": None,
                       "relative_error_status": "not_evaluated",
                       "solve_seconds": None, "solver_diagnostics": "", **flags}
                if not np.all(np.isfinite(commands[i])):
                    row.update(solver_status="approximate_support_failed", failure="Cached support query failed")
                elif flags["negative_pv_values"] or flags["negative_cl_values"] or flags["nonfinite_parameter_values"]:
                    row.update(solver_status="invalid_parameters", failure="Unclipped realization violates parameter domain")
                else:
                    target = (commands[i] - mg.pmin) / (mg.pmax - mg.pmin)
                    try:
                        point, info = solver.project(target, updates)
                        row["solve_seconds"] = info.get("runtime")
                        row["solver_diagnostics"] = json.dumps(info, allow_nan=False)
                        if point is None or not np.all(np.isfinite(point)):
                            row.update(solver_status="projection_failed", failure=str(info.get("status_name", "Failed projection")))
                        else:
                            restored = point * (mg.pmax - mg.pmin) + mg.pmin
                            projected[ci, i] = restored
                            physical_error = float(np.sum((commands[i] - restored) ** 2))
                            percentage = relative_error_percent(physical_error, power_squared_sum)
                            row.update(solver_status="completed",
                                       feasibility_error=physical_error,
                                       feasibility_error_rated=float(np.sum((target - point) ** 2)),
                                       relative_disaggregation_error_percent=percentage,
                                       relative_error_status="completed" if percentage is not None
                                           else "undefined_zero_power")
                    except Exception as exc:
                        row.update(solver_status="projection_failed", failure=f"{type(exc).__name__}: {exc}")
                rows.append(row)
                done[ci, i] = True
                if (i + 1) % args.checkpoint_every == 0 or i + 1 == n:
                    save("evaluating")
                    print(f"{condition} (mean={delta:+.2f}, std={sigma:.2f}): {i + 1}/{n}; "
                          f"total {len(rows)}/{6*n}", flush=True)
        samples = pd.DataFrame(rows)
        if len(samples) != 6*n or samples.duplicated(["condition", "sample_index"]).any():
            raise RuntimeError("Incomplete or duplicate paired evaluation records")
        protected_after = {"training_weights": file_hash(training / "pretrainnet_weights.pth"),
                           "training_metadata": file_hash(training / "metadata.pkl"),
                           "physical_profiles": file_hash(DATA_ROOT / "profiles_data" / "profiles_data.npz")}
        if training_config:
            protected_after["training_config"] = file_hash(training_config_path)
        if protected_after != protected:
            raise RuntimeError("Protected training/input data changed during evaluation")
        metadata.update(protected_inputs_unchanged=True,
                        cached_commands_sha256=array_hash(commands),
                        paired_comparisons=paired_comparisons(samples))
        # Isolate plotting from Torch/Gurobi's OpenMP runtimes on Windows.
        subprocess.run([sys.executable, "-m", "Simulator.drawers.plot_microgrid_shift",
                        str(output)], cwd=CODE_ROOT, check=True)
        figure = output / "figures" / f"{PREFIX}.pdf"
        metadata["figure"] = {"path": str(figure), "sha256": file_hash(figure)}
        save("completed")
        print(summarize(samples).to_string(index=False), flush=True)
        print(f"Saved six-condition evaluation: {output}", flush=True)
    except BaseException:
        save("interrupted")
        raise
    finally:
        if solver is not None:
            solver.close()
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=RESULTS_ROOT / "microgrid" / "training")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT / "microgrid" / "disaggregation")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--conservativeness", type=float, default=0.0,
                        help="Physical evaluation constraint shrinkage; keep 0 to evaluate training margins against true ranges")
    parser.add_argument("--backend", choices=("persistent", "pyomo"), default="persistent")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true",
                        help="Require the same inputs, settings and source hashes; replot completed historical data with Simulator.drawers.plot_microgrid_shift")
    args = parser.parse_args(argv)
    if args.num_samples < 1 or args.checkpoint_every < 1:
        parser.error("--num-samples and --checkpoint-every must be positive")
    if not 0 <= args.conservativeness < 0.2:
        parser.error("--conservativeness must be in [0, 0.2) for the original CL portfolio")
    evaluate(args)


if __name__ == "__main__":
    main()
