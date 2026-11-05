"""Train one normalized polytope using approximate or exact error gradients.

The exact feasibility adjoint solves B.T p = 2(x-q): a direct solve for T
active rows, thin QR for more than T rows. Its derivatives are -p x.T and p.
For optimality the full-square projection objective gives -Pi z.T and Pi,
using Gurobi's RHS duals. Nonregular sensitivities are labelled separately;
a minimum-norm adjoint does not certify differentiability at a degenerate point.
The original region must have a unique nearest aggregate point for the distance
argument; a mixed-integer solver return alone does not establish uniqueness.

PyTorch propagates these output derivatives through row normalization. The
approximate mode retains the original active-residual surrogate instead.
"""
from concurrent.futures import ThreadPoolExecutor
import threading
import warnings

import numpy as np
from scipy.linalg import solve_triangular
import torch
from torch import nn, optim
from pyomo.environ import (
    ConcreteModel, Constraint, Objective, Param, SolverFactory, Suffix,
    TerminationCondition, Var, minimize,
)

# Legacy Pyomo solver plugins share process-wide output/tempfile machinery.
_PYOMO_SOLVE_LOCK = threading.RLock()


class NonDifferentiableGradientError(RuntimeError):
    """The active system cannot provide the requested checked sensitivity."""


class ErrorCalculator:
    """Solve directional distance problems for a fixed or mutable original region."""

    def __init__(self, original_model, A_hat, b_hat=None, solver='gurobi',
                 cvx_solver='gurobi', gradient_mode='approx', solver_options=None):
        if gradient_mode not in ('exact', 'approx'):
            raise ValueError("gradient_mode must be 'exact' or 'approx'")
        self.gradient_mode = gradient_mode
        self.baseline = original_model.get('baseline')
        self.original_model_dict = original_model
        self.original_model = original_model['model'].clone()
        self.dim = len(self.original_model.var_proj)
        self.solver_str, self.cvx_solver_str = solver, cvx_solver
        self.solver = SolverFactory(solver)
        self.cvx_solver = SolverFactory(cvx_solver)
        for optimizer in (self.solver, self.cvx_solver):
            optimizer.options.update(solver_options or {})
        self.params = dict(active_tol=1e-5, dual_tol=1e-8, kkt_tol=1e-5,
                           max_active_condition=1e10, nondifferentiable='raise',
                           feas_tol=1e-8, opt_tol=1e-8)
        self.gradient_failures = {'feas': 0, 'opt': 0}
        self.training_history = {'feas': [], 'opt': []}
        self.projection_retry_records = []
        self._iter = 0
        self._last_approx_solve = None
        for objective in self.original_model.component_objects(Objective, active=True):
            objective.deactivate()
        model = self.original_model
        model.x_apx = Param(range(self.dim), initialize=0., mutable=True)
        model.direction = Param(range(self.dim), initialize=0., mutable=True)
        self._add_objectives(model, model.x_apx)

        self.A_hat = np.asarray(A_hat, dtype=float).copy()
        if (self.A_hat.ndim != 2 or self.A_hat.shape[1] != self.dim
                or not np.all(np.isfinite(self.A_hat))):
            raise ValueError('A_hat must be a finite (rows, dim) matrix')
        if b_hat is None:
            bounds = []
            for row in self.A_hat:
                point = self.optimize_direction(-row)
                if point is None:
                    raise RuntimeError('Failed to initialize an outer-polytope bound')
                bounds.append(row @ point)
            b_hat = bounds
        self.b_hat = np.asarray(b_hat, dtype=float).copy()
        if self.b_hat.shape != (len(self.A_hat),) or not np.all(np.isfinite(self.b_hat)):
            raise ValueError('b_hat must be a finite vector with one entry per row')

        model = self.approx_model = ConcreteModel()
        model.var_proj = Var(range(self.dim), initialize=0.)
        model.x_org = Param(range(self.dim), initialize=0., mutable=True)
        model.direction = Param(range(self.dim), initialize=0., mutable=True)
        self._add_objectives(model, model.x_org)
        rows = range(len(self.A_hat))
        model.A = Param(rows, range(self.dim), mutable=True,
                        initialize=lambda m, i, j: self.A_hat[i, j])
        model.b = Param(rows, mutable=True, initialize=lambda m, i: self.b_hat[i])
        model.constraints = Constraint(rows, rule=lambda m, i:
            sum(m.A[i, j] * m.var_proj[j] for j in range(self.dim)) <= m.b[i])
        if gradient_mode == 'exact':
            model.dual = Suffix(direction=Suffix.IMPORT)

    def _add_objectives(self, model, target):
        model.min_direction = Objective(expr=sum(
            model.direction[j] * model.var_proj[j] for j in range(self.dim)), sense=minimize)
        model.min_error = Objective(expr=sum(
            (model.var_proj[j] - target[j]) ** 2 for j in range(self.dim)), sense=minimize)
        model.min_error.deactivate()

    def configure(self, **kwargs):
        if 'active_basis_policy' in kwargs:
            raise ValueError('Active-policy selection was removed: T rows use direct solve; more rows use QR')
        if kwargs.get('nondifferentiable', 'raise') not in ('raise', 'skip'):
            raise ValueError("nondifferentiable must be 'raise' or 'skip'")
        self.params.update(kwargs)

    @staticmethod
    def _solve_model(solver, model):
        with _PYOMO_SOLVE_LOCK:
            return solver.solve(model)

    def _solve_point(self, vector, in_approx, projection):
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.dim,) or not np.all(np.isfinite(vector)):
            raise ValueError('Direction/target must be a finite vector with length dim')
        model = self.approx_model if in_approx else self.original_model
        solver = self.cvx_solver if in_approx else self.solver
        if in_approx:
            self._last_approx_solve = None
            if hasattr(model, 'dual'):
                model.dual.clear()
        if projection:
            model.min_direction.deactivate()
            model.min_error.activate()
            param = model.x_org if in_approx else model.x_apx
        else:
            model.min_error.deactivate()
            model.min_direction.activate()
            param = model.direction
        for j, number in enumerate(vector):
            param[j] = float(number)
        try:
            result = self._solve_model(solver, model)
        except Exception as exc:
            warnings.warn(f'Solver failed: {exc}', RuntimeWarning, stacklevel=2)
            return None
        solver_name = self.cvx_solver_str if in_approx else self.solver_str
        if (projection and str(solver_name).startswith('gurobi')
                and result.solver.termination_condition != TerminationCondition.optimal):
            # Re-solve the identical convex projection after a numerical failure.
            # Neither constraints nor feasibility/optimality tolerances change.
            record = {'calculation_index': self._iter, 'in_approx': in_approx,
                      'initial_termination': str(result.solver.termination_condition),
                      'attempts': []}
            method_key = next((key for key in solver.options if str(key).lower() == 'method'), 'Method')
            had_method, previous_method = method_key in solver.options, solver.options.get(method_key)
            try:
                for method in (1, 0):
                    solver.options[method_key] = method
                    if in_approx and hasattr(model, 'dual'):
                        model.dual.clear()
                    try:
                        result = self._solve_model(solver, model)
                    except Exception as exc:
                        record['attempts'].append({'method': method, 'failure': str(exc)})
                        continue
                    record['attempts'].append({
                        'method': method, 'termination': str(result.solver.termination_condition)})
                    if result.solver.termination_condition == TerminationCondition.optimal:
                        break
            finally:
                if had_method:
                    solver.options[method_key] = previous_method
                else:
                    solver.options.pop(method_key, None)
                self.projection_retry_records.append(record)
        if result.solver.termination_condition != TerminationCondition.optimal:
            warnings.warn(f'Solver termination: {result.solver.termination_condition}',
                          RuntimeWarning, stacklevel=2)
            return None
        if in_approx:
            self._last_approx_solve = 'projection' if projection else 'direction'
        return np.array([model.var_proj[j].value for j in range(self.dim)], dtype=float)

    def project(self, target, to_approx=False):
        return self._solve_point(target, to_approx, projection=True)

    def optimize_direction(self, direction, in_approx=False):
        return self._solve_point(direction, in_approx, projection=False)

    def _find_active(self, x_apx):
        return np.flatnonzero(np.abs(self.A_hat @ x_apx - self.b_hat) < self.params['active_tol'])

    def _checked_duals(self, x_apx, objective_gradient, solve_kind, preserve_all=False):
        """Read fresh Gurobi Pi and check the LP/QP KKT conditions."""
        if self._last_approx_solve != solve_kind:
            raise RuntimeError(f"Exact gradient requires the latest {solve_kind} solve")
        pi = np.array([
            self.approx_model.dual.get(self.approx_model.constraints[i], np.nan)
            for i in range(len(self.b_hat))
        ], dtype=float)
        if not np.all(np.isfinite(pi)):
            raise RuntimeError("Missing constraint duals: use a solver supporting Pyomo dual IMPORT for LP/QP")
        residual = self.A_hat @ x_apx - self.b_hat
        tol = self.params['kkt_tol']
        if np.max(residual, initial=0.) > tol:
            raise RuntimeError("Approximate-polytope solve failed primal feasibility check")
        if np.max(pi, initial=0.) > tol:
            raise RuntimeError("Unexpected dual sign: minimization with <= rows requires Pi <= 0")
        if np.max(np.abs(pi * residual), initial=0.) > tol * (1. + np.linalg.norm(objective_gradient)):
            raise RuntimeError("Approximate-polytope solve failed complementary slackness check")
        # Inactive rows have exactly zero sensitivity; remove solver roundoff.
        if not preserve_all:
            pi[abs(residual) >= self.params['active_tol']] = 0.
        stationarity = objective_gradient - self.A_hat.T @ pi
        if np.linalg.norm(stationarity, ord=np.inf) > tol * (1. + np.linalg.norm(objective_gradient, ord=np.inf)):
            raise RuntimeError("Approximate-polytope solve failed stationarity check; tighten solver tolerances")
        return pi

    def _active_matrix(self, x_apx, square=False):
        active = self._find_active(x_apx)
        C = np.asarray(self.A_hat[active], dtype=float)
        if square and C.shape != (self.dim, self.dim):
            raise NonDifferentiableGradientError(
                f"Support LP has {len(active)} active rows in dimension {self.dim}; "
                "an independent, stable vertex is required. Remove redundant rows or tie their parameters."
            )
        if len(active):
            singular_values = np.linalg.svd(C, compute_uv=False)
            if (len(active) > self.dim or singular_values[-1] == 0. or
                    singular_values[0] / singular_values[-1] > self.params['max_active_condition']):
                raise NonDifferentiableGradientError(
                    "Active constraints are dependent or ill-conditioned; no unique exact gradient is certified"
                )
        return active, C

    def _qr_minimum_norm_adjoint(self, B, rhs):
        """Solve B.T p = rhs by thin QR only when B has more than T rows.

        For full-column-rank B = Q R, p = Q solve(R.T, rhs) lies in
        range(B), orthogonal to null(B.T), so it has minimum Euclidean norm.
        Singular values of the small R only certify numerical rank and
        conditioning; they are not used to compute p. No rows are selected,
        no normal equations are formed, and no alternative solve is used.
        """
        B = np.asarray(B, dtype=float)
        rhs = np.asarray(rhs, dtype=float)
        if B.ndim != 2 or B.shape[1] != self.dim or rhs.shape != (self.dim,):
            raise ValueError("Expected B=(active_rows, dim) and rhs=(dim,)")
        if not np.all(np.isfinite(B)) or not np.all(np.isfinite(rhs)):
            raise ValueError("QR adjoint requires finite B and rhs")
        if B.shape[0] <= self.dim:
            raise NonDifferentiableGradientError(
                f"QR adjoint requires more than {self.dim} active rows; got {B.shape[0]}"
            )
        Q, R = np.linalg.qr(B, mode='reduced')
        singular = np.linalg.svd(R, compute_uv=False)
        cutoff = singular[0] / self.params['max_active_condition']
        active_rank = int(np.sum(singular > cutoff))
        if active_rank != self.dim:
            raise NonDifferentiableGradientError(
                f"QR adjoint requires full column rank {self.dim} within the condition limit; "
                f"got {B.shape[0]} active rows with numerical rank {active_rank}. "
                "No alternate solve is used."
            )
        condition = float(singular[0] / singular[-1])
        y = solve_triangular(R, rhs, trans='T', lower=False, check_finite=False)
        p = Q @ y
        if not np.all(np.isfinite(p)):
            raise NonDifferentiableGradientError("QR adjoint returned a non-finite result")
        return p, active_rank, condition

    def _feasibility_gradient(self, x_apx, x_org, direction):
        """Use every active row; QR is used only for an underdetermined adjoint."""
        direction = np.asarray(direction, dtype=float)
        active = self._find_active(x_apx)
        pi = self._checked_duals(x_apx, direction, 'direction')
        rhs = 2. * (x_apx - x_org)
        used_qr = len(active) > self.dim
        if used_qr:
            B = self.A_hat[active]
            p, _, condition = self._qr_minimum_norm_adjoint(B, rhs)
        else:
            active, B = self._active_matrix(x_apx, square=True)
            condition = float(np.linalg.cond(B))
            p = np.linalg.solve(B.T, rhs)
        multipliers = -pi[active]
        regular = len(active) == self.dim and np.all(multipliers > self.params['dual_tol'])
        residual = float(np.linalg.norm(B.T @ p - rhs, ord=np.inf))
        if residual > self.params['kkt_tol'] * (1. + np.linalg.norm(rhs, ord=np.inf)):
            raise NonDifferentiableGradientError('Adjoint solve failed its residual check')
        self._last_gradient_info = {
            'gradient_status': ('ok' if regular else 'qr_minimum_norm_sensitivity'
                                if used_qr else 'square_branch_sensitivity'),
            'adjoint_solver': 'qr' if used_qr else 'solve',
            'gradient_active_indices': active.copy(), 'gradient_basis_count': len(active),
            'gradient_active_count': len(active), 'active_matrix_rank': self.dim,
            'active_matrix_condition': condition, 'adjoint_residual_inf': residual,
            'active_constraint_residual_inf': float(np.max(np.abs(B @ x_apx - self.b_hat[active]))),
            'linearization_incompatibility_dimension': len(active) - self.dim,
        }
        if not used_qr:
            self._last_gradient_info.update(
                basis_condition=condition, basis_min_multiplier=float(np.min(multipliers)),
                basis_stationarity_residual_inf=float(np.linalg.norm(
                    direction + B.T @ multipliers, ord=np.inf)))
        grad_A, grad_b = np.zeros_like(self.A_hat), np.zeros_like(self.b_hat)
        grad_A[active], grad_b[active] = -np.outer(p, x_apx), p
        return grad_A, grad_b

    def _optimality_gradient(self, x_apx, x_org):
        """Projection value gradient: full-square objective gives g_b=Pi.

        Gurobi Pi=-mu for <= rows, hence g_A=-Pi z.T. There is NO
        extra factor of two, and no projection Jacobian is needed.
        """
        active = self._find_active(x_apx)
        C = self.A_hat[active]
        pi = self._checked_duals(x_apx, 2. * (x_apx - x_org), 'projection',
                                preserve_all=True)
        # Retain every imported multiplier: an arbitrary independent QP subset
        # need not represent the projection value sensitivity. Dependent active
        # rows signal that this is a solver-selected dual sensitivity.
        independent = not len(active) or (len(active) <= self.dim and
            np.linalg.matrix_rank(C) == len(active) and
            np.linalg.cond(C) <= self.params['max_active_condition'])
        used = np.flatnonzero(pi != 0.)
        self._last_gradient_info = {
            'gradient_status': 'ok' if independent else 'dual_selection',
            'gradient_active_indices': used, 'gradient_basis_count': len(used),
        }
        return -np.outer(pi, x_apx), pi.copy()

    def _error_result(self, x_apx, x_org, kind, direction):
        result = {
            'is_valid': False,
            'error': float(np.sum((x_apx - x_org) ** 2)),
            'active_indices': self._find_active(x_apx),
            'x_apx': x_apx.copy(),
            'x_org': x_org.copy(),
            'gradient_status': 'ok',
        }
        if self.gradient_mode == 'approx':
            result['is_valid'] = result['error'] > self.params[kind + '_tol']
            result['gradient_status'] = 'surrogate' if result['is_valid'] else 'thresholded'
            return result
        self._last_gradient_info = {}
        try:
            if kind == 'feas':
                g_A, g_b = self._feasibility_gradient(x_apx, x_org, direction)
            else:
                g_A, g_b = self._optimality_gradient(x_apx, x_org)
        except NonDifferentiableGradientError as exc:
            self.gradient_failures[kind] += 1
            if self.params['nondifferentiable'] == 'raise':
                raise NonDifferentiableGradientError(f"{kind} error: {exc}") from exc
            # Opt-in filtering changes the sampled training objective. Preserve
            # the real error in diagnostics; never label a fallback as exact.
            result['gradient_status'] = 'nondifferentiable'
            result['gradient_message'] = str(exc)
            warnings.warn(f"Skipping {kind} gradient: {exc}", RuntimeWarning, stacklevel=2)
            return result
        result.update(is_valid=True, gradient_A=g_A, gradient_b=g_b, **self._last_gradient_info)
        return result

    def calculate(self, n_cal=1, cal_feas=True, cal_opt=True, directions=None):
        """Evaluate shared directions and build the selected mode's gradient samples.

        Exact mode includes all errors, including zero; approximate mode retains
        its original error thresholds and active-residual surrogate convention.
        Failed solves raise in either mode rather than masquerading as zero error.
        """
        if n_cal < 1:
            raise ValueError("n_cal must be positive")
        if directions is not None:
            directions = np.asarray(directions, dtype=float)
            if directions.shape != (n_cal, self.dim) or not np.all(np.isfinite(directions)):
                raise ValueError("directions must be finite with shape (n_cal, dim)")

        def disabled():
            return {'is_valid': False, 'error': 0., 'gradient_status': 'disabled',
                    'active_indices': np.array([], dtype=int),
                    'x_org': np.zeros(self.dim), 'x_apx': np.zeros(self.dim)}

        feas_results, opt_results = [], []
        for i in range(n_cal):
            c = np.random.randn(self.dim) if directions is None else directions[i]
            f, o = disabled(), disabled()
            if cal_feas:
                x_apx = self.optimize_direction(c, in_approx=True)
                x_org = self.project(x_apx) if x_apx is not None else None
                if x_apx is None or x_org is None:
                    raise RuntimeError("Feasibility-error solve failed; refusing a zero/missing gradient")
                # Consume the support LP duals before the next approximate QP.
                f = self._error_result(x_apx, x_org, 'feas', c)
            if cal_opt:
                x_org = self.optimize_direction(c)
                x_apx = self.project(x_org, to_approx=True) if x_org is not None else None
                if x_apx is None or x_org is None:
                    raise RuntimeError("Optimality-error solve failed; refusing a zero/missing gradient")
                o = self._error_result(x_apx, x_org, 'opt', c)
            feas_results.append(f)
            opt_results.append(o)

        self.training_history['feas'].append(np.mean([r['error'] for r in feas_results]))
        self.training_history['opt'].append(np.mean([r['error'] for r in opt_results]))
        self._iter += 1
        return feas_results, opt_results

    def update_polytope(self, A_hat=None, b_hat=None):
        self._last_approx_solve = None
        if A_hat is not None:
            A_hat = np.asarray(A_hat, dtype=float)
            if A_hat.shape != self.A_hat.shape or not np.all(np.isfinite(A_hat)):
                raise ValueError('Updated A_hat must retain its finite matrix shape')
            self.A_hat = A_hat.copy()
            for i, j in np.ndindex(self.A_hat.shape):
                self.approx_model.A[i, j] = self.A_hat[i, j]
        if b_hat is not None:
            b_hat = np.asarray(b_hat, dtype=float)
            if b_hat.shape != self.b_hat.shape or not np.all(np.isfinite(b_hat)):
                raise ValueError('Updated b_hat must retain its finite vector shape')
            self.b_hat = b_hat.copy()
            for i in range(len(self.b_hat)):
                self.approx_model.b[i] = self.b_hat[i]

    def update_parameters(self, param_dict):
        """Update mutable original-region parameters with scalar, array or mapping values."""
        for name, values in param_dict.items():
            param = getattr(self.original_model, name)
            if not param.is_indexed():
                param.set_value(float(np.asarray(values)))
            elif isinstance(values, dict):
                for index, number in values.items():
                    param[index].set_value(float(number))
            else:
                array = np.asarray(values)
                indices = list(param.index_set())
                if array.size != len(indices):
                    raise ValueError(f'Parameter {name} has {len(indices)} entries, got {array.size}')
                for index, number in zip(indices, array.flat):
                    param[index].set_value(float(number))

    def copy(self):
        calculator = ErrorCalculator(self.original_model_dict, self.A_hat, self.b_hat,
                                     self.solver_str, self.cvx_solver_str, self.gradient_mode)
        calculator.configure(**self.params)
        calculator.solver.options.update(self.solver.options)
        calculator.cvx_solver.options.update(self.cvx_solver.options)
        return calculator


class PreTrainNet(nn.Module):
    def __init__(self,A_init,b_init,device = 'cpu'):
        super().__init__()
        self.nrows,self.dim_x = A_init.shape
        self.A_net = nn.Sequential(
            ZeroInitLinear(0, self.nrows*self.dim_x, init_bias=np.array(A_init).flatten(),device = device)
        )
        self.b_net = nn.Sequential(
            ZeroInitLinear(0, self.nrows, init_bias=np.array(b_init), device=device)
        )

    def forward(self):
        # 生成A矩阵
        A_flat = self.A_net(self.A_net[0].bias.new_empty(0))
        A = A_flat.view(-1, self.nrows, self.dim_x)
        # 归一化A的行向量
        row_norms = torch.norm(A, dim=2, keepdim=True)  # (batch, rows, 1)
        A_normalized = A / (row_norms + 1e-8)  # Add small epsilon to avoid division by zero

        # 生成b向量
        b = self.b_net(self.b_net[0].bias.new_empty(0))

        # 归一化b向量（除以对应的A行范数）
        b_normalized = b / (row_norms.squeeze(2) + 1e-8)  # (batch, rows)

        return A_normalized, b_normalized


class ZeroInitLinear(nn.Module):
    """自定义全连接层：权重初始为0，bias初始化为给定值"""

    def __init__(self, in_features, out_features, init_bias,device='cpu'):
        super().__init__()
        self.device = device
        self.weight = nn.Parameter(torch.zeros(out_features, in_features,device=device))
        self.bias = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32,device=device))

    def forward(self, x):
        return torch.matmul(x.to(device=self.weight.device, dtype=self.weight.dtype), self.weight.t()) + self.bias


class _ExactErrorValue(torch.autograd.Function):
    """Real solver value in forward, analytic first derivative in backward.

    This is not a residual surrogate. At regular points the backward tensors
    are exact derivatives; QR/dual sensitivities
    at nonregular points are explicitly labelled in the sample metadata and
    are not asserted to be ordinary derivatives there.
    Higher-order derivatives through the solver are intentionally unsupported.
    """

    @staticmethod
    def forward(ctx, A, b, error, grad_A, grad_b):
        ctx.save_for_backward(grad_A, grad_b)
        return A.new_tensor(error)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        grad_A, grad_b = ctx.saved_tensors
        return grad_output * grad_A, grad_output * grad_b, None, None, None


def compute_exact_error(A, b, batch_data):
    """Mean real error with exact regular or labelled nonregular sensitivities.

    A: (batch, rows, dim), b: (batch, rows). batch_data must come from
    ErrorCalculator.calculate at these very A/b values. All valid samples,
    including labelled sensitivities and zero/small errors,
    enter the denominator. Disabled paths return a connected zero. Skipped
    nonregular samples
    are excluded; their real errors remain in ErrorCalculator's history.

    Autograd propagates these A/b derivatives through the actual network,
    including BOTH appearances of the row norm in normalized A and b.
    """
    if A.ndim != 3 or b.shape != A.shape[:2]:
        raise ValueError("Expected A=(batch, rows, dim), b=(batch, rows)")
    if len(batch_data) != A.shape[0]:
        raise ValueError("batch_data length must match the A/b batch")
    grad_A, grad_b = torch.zeros_like(A), torch.zeros_like(b)
    total_error, count, skipped = 0., 0, 0
    for batch_idx, samples in enumerate(batch_data):
        for sample in samples:
            if not sample['is_valid']:
                status = sample.get('gradient_status')
                if status == 'nondifferentiable':
                    skipped += 1
                elif status != 'disabled':
                    raise RuntimeError("Missing exact gradient; use this module's ErrorCalculator")
                continue
            g_A = torch.as_tensor(sample['gradient_A'], dtype=A.dtype, device=A.device)
            g_b = torch.as_tensor(sample['gradient_b'], dtype=b.dtype, device=b.device)
            if g_A.shape != A.shape[1:] or g_b.shape != b.shape[1:]:
                raise ValueError("Solver gradient shapes do not match the network output")
            error = float(sample['error'])
            if not np.isfinite(error) or not torch.isfinite(g_A).all() or not torch.isfinite(g_b).all():
                raise RuntimeError("Non-finite exact error/gradient")
            grad_A[batch_idx] += g_A
            grad_b[batch_idx] += g_b
            total_error += error
            count += 1
    if not count and skipped:
        raise NonDifferentiableGradientError(
            "All samples in this error branch are nonregular; cannot take an exact-gradient training step"
        )
    denominator = max(count, 1)
    return _ExactErrorValue.apply(A, b, total_error / denominator,
                                 grad_A / denominator, grad_b / denominator)


def compute_surrogate_error(A, b, batch_data):
    """Original active-residual surrogate, with the network's dtype and device.

    Divide by the number of valid directions, not the number of active rows.
    This value is a training surrogate, not a squared geometric distance.
    """
    if A.ndim != 3 or b.shape != A.shape[:2] or len(batch_data) != len(A):
        raise ValueError('Expected A=(batch, rows, dim), b=(batch, rows), matching samples')
    total, count = (A.sum() + b.sum()) * 0., 0
    for i, samples in enumerate(batch_data):
        for sample in samples:
            if not sample['is_valid']:
                continue
            active = torch.as_tensor(sample['active_indices'], dtype=torch.long, device=A.device)
            target = torch.as_tensor(sample['x_org'], dtype=A.dtype, device=A.device)
            total = total + (A[i, active] @ target - b[i, active]).square().sum()
            count += 1
    return total / max(count, 1)


def compute_error_loss(A, b, batch_data, gradient_mode='approx'):
    """Dispatch the training objective without confusing surrogate values with errors."""
    if gradient_mode == 'exact':
        return compute_exact_error(A, b, batch_data)
    if gradient_mode == 'approx':
        return compute_surrogate_error(A, b, batch_data)
    raise ValueError("gradient_mode must be 'exact' or 'approx'")


class Trainer:
    """Shared PreTrainNet loop; train(n_train) always executes n_train + 1 epochs.

    Under uncertainty each DataLoader batch updates the original Pyomo model,
    while all scenarios share one learned A/b. The final short batch is kept.
    loss_history stores unweighted real (feasibility, optimality) errors;
    surrogate_history separately stores approximate-mode training values.
    """

    def __init__(self, model, error_calculator, params=None, is_uncertain=False,
                 device='cpu', params_data=None, parallel=False):
        if not isinstance(model, PreTrainNet):
            raise TypeError('Trainer supports PreTrainNet')
        self.model, self.error_calculator = model, error_calculator
        self.is_uncertain, self.parallel = is_uncertain, parallel
        self.params_data = {} if params_data is None else params_data
        self.params = dict(optimizer='SGD', lr=0.5, scheduler=None, batch_size=1,
                           n_cal=1, cal_feas=True, cal_opt=True, rate_opt_feas=1.,
                           feas_tol=1e-8, opt_tol=1e-8, call_interval=20,
                           training_callback=None, max_workers=5)
        self.params.update(params or {})
        self.loss_history, self.surrogate_history = [], []
        self.objective_history, self.grad_history, self.train_calls = [], [], []
        self.optimizer_updates = 0

    def configure(self, **kwargs):
        self.params.update(kwargs)

    def initialize(self):
        self.error_calculator.configure(**{
            name: self.params[name] for name in self.error_calculator.params
            if name in self.params
        })
        lr = self.params['lr']
        groups = [dict(params=self.model.A_net.parameters(), lr=self.params.get('lr_A', lr)),
                  dict(params=self.model.b_net.parameters(), lr=self.params.get('lr_b', lr))]
        kind = self.params['optimizer'].lower()
        if kind not in ('sgd', 'adam'):
            raise ValueError(f'Unsupported optimizer: {kind}')
        self.optimizer = (optim.SGD if kind == 'sgd' else optim.Adam)(groups)
        self.scheduler = None
        config = self.params['scheduler']
        if config:
            if config['type'] != 'StepLR':
                raise ValueError(f"Unsupported scheduler: {config['type']}")
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=config.get('step_size', 100), gamma=config.get('gamma', .95))

    def _batch_size(self, batch_data):
        if batch_data:
            sizes = {len(data) for data in batch_data.values()}
            if len(sizes) != 1:
                raise ValueError('All parameter tensors must have the same batch size')
            return sizes.pop()
        return self.params['batch_size']

    def _scenario_parameters(self, batch_data, index):
        if not batch_data:
            return None
        updates = {}
        for name, data in batch_data.items():
            delta = data[index]
            if torch.is_tensor(delta):
                delta = delta.detach().cpu().numpy()
            updates[name] = self.params_data['params_dict'][name]['initial_value'] + delta
        return updates

    def _calculate(self, calculator, A, b, updates, directions):
        calculator.update_polytope(A, b)
        if updates is not None:
            calculator.update_parameters(updates)
        return calculator.calculate(n_cal=len(directions), directions=directions,
                                    cal_feas=self.params['cal_feas'], cal_opt=self.params['cal_opt'])

    def train(self, n_train=1):
        if not isinstance(n_train, (int, np.integer)) or n_train < 0:
            raise ValueError('n_train must be a nonnegative integer; zero executes one epoch')
        if not hasattr(self, 'optimizer'):
            raise RuntimeError('Call initialize() before train()')
        batches = self.params_data['dataloader'] if self.is_uncertain else [None]
        calculators = []
        before_updates = self.optimizer_updates
        local_updates = 0
        executor = ThreadPoolExecutor(max_workers=self.params['max_workers']) if self.parallel else None
        try:
            for _ in range(n_train + 1):
                for batch in batches:
                    size = self._batch_size(batch)
                    if size < 1:
                        raise ValueError('A training batch cannot be empty')
                    self.optimizer.zero_grad(set_to_none=True)
                    A, b = self.model()
                    A, b = A.repeat(size, 1, 1), b.repeat(size, 1)
                    # Generate in the caller thread for reproducible worker ordering.
                    directions = [np.random.randn(self.params['n_cal'], self.error_calculator.dim)
                                  for _ in range(size)]
                    updates = [self._scenario_parameters(batch, i) for i in range(size)]
                    if self.parallel:
                        while len(calculators) < size:
                            calculators.append(self.error_calculator.copy())
                        jobs = [executor.submit(self._calculate, calculators[i],
                                A[i].detach().cpu().numpy(), b[i].detach().cpu().numpy(),
                                updates[i], directions[i]) for i in range(size)]
                        results = [job.result() for job in jobs]
                        callback_calculator = calculators[0]
                        # Preserve complete scenario histories on the public calculator.
                        for feas, opt in results:
                            self.error_calculator.training_history['feas'].append(np.mean([s['error'] for s in feas]))
                            self.error_calculator.training_history['opt'].append(np.mean([s['error'] for s in opt]))
                            self.error_calculator._iter += 1
                    else:
                        results = [self._calculate(self.error_calculator,
                                   A[i].detach().cpu().numpy(), b[i].detach().cpu().numpy(),
                                   updates[i], directions[i]) for i in range(size)]
                        callback_calculator = self.error_calculator
                    feasibility, optimality = zip(*results)
                    mode = self.error_calculator.gradient_mode
                    loss_feas = compute_error_loss(A, b, feasibility, mode)
                    loss_opt = compute_error_loss(A, b, optimality, mode)
                    objective = loss_feas + self.params['rate_opt_feas'] * loss_opt
                    objective.backward()
                    self.grad_history.append(max((p.grad.detach().abs().max().item()
                        for p in self.model.parameters() if p.grad is not None and p.grad.numel()), default=0.))
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    self.loss_history.append(tuple(float(np.mean([
                        sample['error'] for samples in branch for sample in samples]))
                        for branch in (feasibility, optimality)))
                    self.objective_history.append(float(objective.detach()))
                    if mode == 'approx':
                        self.surrogate_history.append((float(loss_feas.detach()), float(loss_opt.detach())))
                    callback = self.params['training_callback']
                    interval = self.params['call_interval']
                    if callback and interval > 0 and local_updates % interval == 0:
                        callback(callback_calculator, local_updates)
                    self.optimizer_updates += 1
                    local_updates += 1
        finally:
            if executor:
                executor.shutdown(wait=True)
        self.train_calls.append(dict(n_train=int(n_train), epochs=int(n_train + 1),
                                     optimizer_updates=self.optimizer_updates - before_updates))


def pyomo_params_to_numpy(model):
    params_dict = {}
    param_count = 0

    def _get_matrix_dimensions(matrix_param):
        rows = set()
        cols = set()
        for (i, j) in matrix_param.index_set():
            rows.add(i)
            cols.add(j)
        return len(rows), len(cols)

    for param in model.component_objects(Param, descend_into=True):
        if not param.mutable:
            continue
        param_name = param.name
        if not param.is_indexed():  # 0维参数（标量）
            value = param.value
            size = 1
            param_count+=size
        else:  # 多维参数
            # 获取参数维度
            dim = param.dim()
            if dim == 1:  # 1维参数（向量）
                values = [param[idx].value for idx in param.index_set()]
                value = np.array(values)
                size = len(value)
                param_count+=size
            elif dim == 2:  # 2维参数（矩阵）
                nrow,ncol = _get_matrix_dimensions(param)

                # 创建矩阵并填充值
                value = np.zeros((nrow, ncol))
                for count, idx in enumerate(param.index_set()):
                    i,j = count//ncol,count%ncol
                    value[i, j] = param[idx[0], idx[1]].value
                size = (nrow,ncol)
                param_count+=size[0]*size[1]
            else:
                raise ValueError(f"参数 {param.name} 的维度 {dim} 超过2维，本代码不支持")
        # 将参数信息存入字典
        params_dict[param_name] = {
            'initial_value': value,
            'size': size
        }
    return params_dict, param_count
