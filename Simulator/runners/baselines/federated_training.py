"""Time-budgeted federated PEB base-set optimization from Dong et al. (2026)."""

from __future__ import annotations

import copy
import time

import numpy as np

_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# PEB volume integration (formerly federated_volume.py).

def _prefix(values):
    return np.r_[0., np.cumsum(values, dtype=float)]


def _range_sum(prefix, left, right):
    """Unweighted sum over [left,right), including the empty interval."""
    return np.where(right >= left, prefix[right]-prefix[left], 0.)


def _cell_grid(lower, upper, delta_x):
    """Midpoints of complete delta_x cells; omit the final fractional cell."""
    ratio = (upper-lower)/delta_x
    # Correct only a few floating-point ulps at an intended integer count.
    nearest = np.rint(ratio)
    if abs(ratio-nearest) <= 8*np.finfo(float).eps*max(1., abs(ratio)):
        ratio = nearest
    count = int(np.floor(ratio))
    if count < 1:
        raise ArithmeticError("The fixed Riemann delta_x is too coarse for a state interval")
    return lower+(np.arange(count, dtype=float)+.5)*delta_x


def peb_volume_and_gradient(h, horizon, *, grid_points=513, delta_t=1., delta_x=None):
    """Estimate PEB volume and its continuous boundary gradient using (44)-(46).

    delta_x remains fixed across all dimensions and calls. If omitted, it is
    2*delta_t/(grid_points-1), a reference-grid choice independent of h.
    grid_points therefore controls a reference spacing, not each K_t.
    Each I_t uses midpoint nodes of floor(|I_t|/delta_x) complete cells, all
    carrying weight delta_x. A fractional rightmost cell is omitted.

    The gradient quadratures follow Appendix C: analytically integrate each
    Dirac boundary term, then apply the same Riemann recursion to the remaining
    dimensions. This is an estimate of the continuous-volume derivative, not
    an automatic derivative of the finite indicator-sum program. At coincident
    facets it does not assert a unique classical gradient; closed-indicator
    boundary values and an explicit tie diagnostic are returned.

    Equations (40)-(46) integrate cumulative states. Division by delta_t**T
    converts that integral and its RHS derivatives into power-space volume.
    The benchmark uses delta_t=1, where the factor is exactly one.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    n = int(horizon)
    h = np.asarray(h, dtype=float)
    if h.shape != (4*n,) or not np.isfinite(h).all():
        raise ValueError("Expected finite 4*T RHS for H0=[L;-L;I;-I]")
    if not np.isfinite(delta_t) or delta_t <= 0:
        raise ValueError("delta_t must be finite and positive")
    if isinstance(grid_points, bool) or not isinstance(grid_points, (int, np.integer)) or grid_points < 3:
        raise ValueError("grid_points must be an integer >=3")
    delta_t = float(delta_t)
    reference_step = delta_x is None
    delta_x = 2*delta_t/(int(grid_points)-1) if reference_step else float(delta_x)
    if not np.isfinite(delta_x) or delta_x <= 0:
        raise ValueError("delta_x must be finite and positive")
    energy_upper, energy_lower = h[:n], -h[n:2*n]
    power_upper, power_lower = h[2*n:3*n], -h[3*n:]
    if np.any(energy_upper <= energy_lower) or np.any(power_upper <= power_lower):
        raise ValueError("The base set must have a nonempty interior")

    # In (44), the summation index is i; the printed u(t) inside that sum
    # must mean u(i) for a time-varying bound to enclose every feasible state.
    lower = np.maximum(energy_lower, np.cumsum(power_lower)*delta_t)
    upper = np.minimum(energy_upper, np.cumsum(power_upper)*delta_t)
    if np.any(upper <= lower):
        raise ValueError("The base state intervals are empty")
    grids = [_cell_grid(lo, hi, delta_x) for lo, hi in zip(lower, upper)]
    forward, forward_prefix = [], []
    # Shift the previous nodes, rather than subtracting the bound from a
    # query, so floating-point endpoint comparisons equal the literal (45).
    shifted_previous_lower, shifted_previous_upper = [None]*n, [None]*n

    def incoming(t, points):
        points = np.asarray(points, dtype=float)
        if t == 0:
            return ((points >= power_lower[0]*delta_t) &
                    (points <= power_upper[0]*delta_t)).astype(float)
        left = np.searchsorted(shifted_previous_upper[t], points, side="left")
        right = np.searchsorted(shifted_previous_lower[t], points, side="right")
        return delta_x*_range_sum(forward_prefix[t-1], left, right)

    def forward_value(t, points):
        points = np.asarray(points, dtype=float)
        inside_energy = (points >= energy_lower[t]) & (points <= energy_upper[t])
        return incoming(t, points)*inside_energy

    for t, grid in enumerate(grids):
        if t:
            shifted_previous_lower[t] = grids[t-1]+power_lower[t]*delta_t
            shifted_previous_upper[t] = grids[t-1]+power_upper[t]*delta_t
        values = forward_value(t, grid)
        forward.append(values)
        forward_prefix.append(_prefix(values))
    state_volume = float(delta_x*np.sum(forward[-1]))
    if not np.isfinite(state_volume) or state_volume <= 0:
        raise ArithmeticError("Zero/nonfinite Riemann mass: refine delta_x or check base feasibility")

    backward, backward_prefix = [None]*n, [None]*n

    def continuation(t, points):
        points = np.asarray(points, dtype=float)
        if t == n-1:
            return np.ones_like(points)
        lo = points+power_lower[t+1]*delta_t
        hi = points+power_upper[t+1]*delta_t
        left = np.searchsorted(grids[t+1], lo, side="left")
        right = np.searchsorted(grids[t+1], hi, side="right")
        return delta_x*_range_sum(backward_prefix[t+1], left, right)

    for t in range(n-1, -1, -1):
        backward[t] = continuation(t, grids[t])
        backward_prefix[t] = _prefix(backward[t])

    gradient_state = np.zeros(4*n)
    for t, grid in enumerate(grids):
        for target, bound in ((t, energy_upper[t]), (n+t, energy_lower[t])):
            gradient_state[target] = float(incoming(t, bound)*continuation(t, bound))
        for target, bound in ((2*n+t, power_upper[t]), (3*n+t, power_lower[t])):
            if t == 0:
                point = bound*delta_t
                inside_energy = energy_lower[0] <= point <= energy_upper[0]
                value = delta_t*float(continuation(0, point))*inside_energy
            else:
                value = delta_t*delta_x*float(np.sum(
                    forward_value(t-1, grid-bound*delta_t)*backward[t]))
            gradient_state[target] = value

    conversion = delta_t**n
    volume = state_volume/conversion
    gradient = gradient_state/conversion
    consistency = np.asarray([delta_x*np.sum(f*g) for f, g in zip(forward, backward)])
    scale = max(1., *(abs(value) for value in
        (energy_upper[0], energy_lower[0], power_upper[0]*delta_t, power_lower[0]*delta_t)))
    upper_tie = abs(energy_upper[0]-power_upper[0]*delta_t) <= 1e-12*scale
    lower_tie = abs(energy_lower[0]-power_lower[0]*delta_t) <= 1e-12*scale
    pairs = ([[0, 2*n]] if upper_tie else [])+([[n, 3*n]] if lower_tie else [])
    return {
        "volume": float(volume), "state_volume": state_volume,
        "log_volume": float(np.log(state_volume)-n*np.log(delta_t)),
        "gradient": gradient, "log_gradient": gradient_state/state_volume,
        "grid_points": int(grid_points), "delta_x": delta_x,
        "delta_x_source": "2*delta_t/(grid_points-1), independent of h" if reference_step else "explicit fixed delta_x",
        "grid_counts": [len(grid) for grid in grids],
        "grid_intervals": [[float(lo), float(hi)] for lo, hi in zip(lower, upper)],
        "omitted_right_tail_widths": [float(max(0., hi-lo-len(grid)*delta_x)) for lo, hi, grid in zip(lower, upper, grids)],
        "grid_rule": "midpoint nodes of floor(|I_t|/delta_x) complete cells; one common delta_x",
        "quadrature": "Appendix D (44)-(46) closed-indicator Riemann sums, accelerated by prefix sums",
        "gradient_rule": "Appendix C analytic Dirac-boundary elimination, then the same Riemann sums",
        "boundary_indicator": "closed intervals; no interpolation or half-endpoint density replacement",
        "coincident_first_period_rhs_pairs": pairs,
        "unique_classical_gradient_asserted": not bool(pairs),
        "gradient_regular_point_note": "A reported coincident facet has no unique classical RHS gradient; absence of this diagnostic is not a general regularity proof",
        "state_to_power_volume_factor": float(1/conversion),
        "forward_backward_relative_discrepancy": float(np.max(np.abs(consistency/state_volume-1.))),
        "minimum_density_roundoff": 0.,
    }


class _ModifiedIntegral:
    """Piecewise-linear-density CDF used only by the authorized variant."""

    def __init__(self, grid, density):
        self.grid = np.asarray(grid, dtype=float)
        self.density = np.asarray(density, dtype=float)
        self.width = np.diff(self.grid)
        self.mass = np.r_[0., np.cumsum(
            .5*(self.density[:-1]+self.density[1:])*self.width)]

    def __call__(self, points):
        points = np.asarray(points, dtype=float)
        clipped = np.clip(points, self.grid[0], self.grid[-1])
        index = np.clip(np.searchsorted(self.grid, clipped, side="right")-1,
                        0, len(self.grid)-2)
        distance = clipped-self.grid[index]
        slope = (self.density[index+1]-self.density[index])/self.width[index]
        return self.mass[index]+self.density[index]*distance+.5*slope*distance**2


def peb_volume_and_gradient_modified(h, horizon, *, grid_points=513, delta_t=1.):
    """User-authorized 94-row CDF/trapezoid variant, separate from Algorithm 1.

    This restores the earlier modified numerical scheme. The RHS follows
    [I;-I;L[1:];-L[1:]], of size 4*T-2; duplicate first-period energy rows are
    omitted. Reachable-state intervals are clipped recursively, each receives
    grid_points nodes, and piecewise-linear CDFs/trapezoids evaluate the forward,
    backward and boundary integrals. These choices intentionally differ from
    the strict complete-template Riemann entry point above.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    n = int(horizon)
    h = np.asarray(h, dtype=float)
    if h.shape != (4*n-2,) or not np.isfinite(h).all():
        raise ValueError("Expected a finite RHS for the nonduplicate modified PEB template")
    if (isinstance(grid_points, bool) or not isinstance(grid_points, (int, np.integer))
            or grid_points < 17 or grid_points % 2 != 1):
        raise ValueError("grid_points must be an odd integer >=17")
    if not np.isfinite(delta_t) or delta_t <= 0:
        raise ValueError("delta_t must be finite and positive")
    grid_points, delta_t = int(grid_points), float(delta_t)
    upper_power, lower_power = h[:n], -h[n:2*n]
    upper_energy = np.r_[np.inf, h[2*n:3*n-1]]
    lower_energy = np.r_[-np.inf, -h[3*n-1:]]
    if np.any(upper_power <= lower_power):
        raise ValueError("The base set must have a nonempty interior")
    grids, forwards, integrals = [], [], []
    previous_lower = previous_upper = 0.
    negative_roundoff = 0.
    for t in range(n):
        lo = max(lower_energy[t], previous_lower+lower_power[t]*delta_t)
        hi = min(upper_energy[t], previous_upper+upper_power[t]*delta_t)
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            raise ValueError("The cumulative-state reachable interval is empty")
        grid = np.linspace(lo, hi, grid_points)
        if t == 0:
            density = np.ones(grid_points)
        else:
            previous_cdf = integrals[-1]
            density = (previous_cdf(grid-lower_power[t]*delta_t)
                       - previous_cdf(grid-upper_power[t]*delta_t))
            negative_roundoff = min(negative_roundoff, float(density.min()))
            if density.min() < -1e-10*max(1., previous_cdf.mass[-1]):
                raise ArithmeticError("Nonmonotone numerical integration")
            density = np.maximum(density, 0.)
        grids.append(grid)
        forwards.append(density)
        integrals.append(_ModifiedIntegral(grid, density))
        previous_lower, previous_upper = lo, hi
    state_volume = float(integrals[-1].mass[-1])
    if not np.isfinite(state_volume) or state_volume <= 0:
        raise ArithmeticError("The volume estimate underflowed or is nonpositive")
    backwards = [None]*n
    backwards[-1] = np.ones(grid_points)
    for t in range(n-2, -1, -1):
        cdf = _ModifiedIntegral(grids[t+1], backwards[t+1])
        backwards[t] = (cdf(grids[t]+upper_power[t+1]*delta_t)
                        - cdf(grids[t]+lower_power[t+1]*delta_t))
        backwards[t] = np.maximum(backwards[t], 0.)

    gradient_state = np.zeros_like(h)
    gradient_state[0] = delta_t*backwards[0][-1]
    gradient_state[n] = delta_t*backwards[0][0]
    for t in range(1, n):
        grid, continuation = grids[t], backwards[t]
        previous_grid, previous_density = grids[t-1], forwards[t-1]
        for target, bound in ((t, upper_power[t]), (n+t, lower_power[t])):
            query = grid-bound*delta_t
            boundary_density = np.interp(query, previous_grid, previous_density,
                                         left=0., right=0.)
            # This is the earlier variant's symmetric value at a density jump.
            # It is not the closed-indicator convention of strict Algorithm 1.
            tolerance = 1e-12*max(1., float(previous_grid[-1]-previous_grid[0]))
            at_lower = np.abs(query-previous_grid[0]) < tolerance
            at_upper = np.abs(query-previous_grid[-1]) < tolerance
            boundary_density[at_lower] = .5*previous_density[0]
            boundary_density[at_upper] = .5*previous_density[-1]
            gradient_state[target] = delta_t*_trapezoid(boundary_density*continuation, grid)
        cdf = integrals[t-1]
        for target, bound in ((2*n+t-1, upper_energy[t]), (3*n+t-2, lower_energy[t])):
            incoming = float(cdf(bound-lower_power[t]*delta_t)
                             - cdf(bound-upper_power[t]*delta_t))
            remaining = float(np.interp(bound, grid, continuation, left=0., right=0.))
            gradient_state[target] = max(0., incoming)*remaining
    consistency = np.asarray([_trapezoid(f*g, x) for x, f, g in zip(grids, forwards, backwards)])
    return {
        "volume": float(state_volume/delta_t**n),
        "log_volume": float(np.log(state_volume)-n*np.log(delta_t)),
        "gradient": gradient_state/delta_t**n,
        "log_gradient": gradient_state/state_volume,
        "grid_points": grid_points,
        "quadrature": "piecewise-linear CDF and trapezoidal boundary integrals",
        "grid_counts": [grid_points]*n,
        "grid_spacings": [float(grid[1]-grid[0]) for grid in grids],
        "template_rows": 4*n-2,
        "template_order": "[I;-I;L[1:];-L[1:]]",
        "implementation_variant": "user_authorized_modified",
        "paper_algorithm1_literal": False,
        "forward_backward_relative_discrepancy": float(np.max(np.abs(consistency/state_volume-1.))),
        "minimum_density_roundoff": negative_roundoff,
    }



class _BudgetExpired(RuntimeError):
    pass


class PaperAlgorithmFailure(RuntimeError):
    """The paper's required inverse/derivative is undefined for this run."""

    def __init__(self, message, diagnostics, initial=None):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.initial = initial


def train_federated_peb(agg, *, timelimit=200., grid_points=1025, epsilon=1e-5,
                        initial_step=1., backtracking_factor=.5, armijo=1e-4,
                        gradient_tolerance=1e-6, max_backtracking=40, progress=None,
                        mode="strict"):
    """Learn the PEB RHS; each client solves the full affine max-trace LP.

    Input generation and pure persistent-model construction are timed separately.
    The 200-second training clock starts before the first projection/local solve,
    and includes all client solves, sensitivities, integration and rejected trials.
    Evaluation and independent certificates are downstream, outside this clock.
    """
    # Volume integration remains usable without the optional optimization stack.
    import gurobipy as gp
    from Simulator.runners.baselines.federated_aggregation import (
        KKTRegularityError, LocalAffineLP, extract_device_polytopes, peb_template, project_base,
    )
    from Simulator.testers.polytope_evaluation import SOLVER_OPTIONS

    if timelimit <= 0 or not 0 < backtracking_factor < 1 or not 0 < armijo < 1:
        raise ValueError("Invalid training budget or line-search parameters")
    if mode not in ("strict", "lexicographic"):
        raise ValueError("Choose strict paper execution or the user-authorized lexicographic variant")
    modified = mode == "lexicographic"
    setup_started = time.perf_counter()
    devices = extract_device_polytopes(agg)
    H0 = peb_template(agg.T, agg.deltaT)
    if modified:
        cumulative = agg.deltaT*np.tril(np.ones((agg.T, agg.T)))
        H0 = np.vstack((np.eye(agg.T), -np.eye(agg.T), cumulative[1:], -cumulative[1:]))
    volume_function = peb_volume_and_gradient_modified if modified else peb_volume_and_gradient
    history, clients = [], []
    accepted = None
    total_local_solves = 0
    candidate_evaluations = 0
    stop_reason = "time_limit"
    gradient_count = 0
    gradient_diagnostics = []
    invalid_candidates = []
    regularity_failure = None
    env = gp.Env(empty=True)
    for key, value in SOLVER_OPTIONS.items():
        env.setParam(key, value)
    env.start()
    training_started = deadline = None

    def event(**record):
        if progress is not None:
            record.update(elapsed_training_seconds=0. if training_started is None else time.perf_counter() - training_started,
                          candidate_evaluations=candidate_evaluations, local_lp_solves=total_local_solves)
            progress(record)

    def evaluate(raw_h, iteration, trial):
        nonlocal candidate_evaluations, total_local_solves
        if time.perf_counter() >= deadline:
            raise _BudgetExpired("Training clock expired")
        try:
            h, projection_jacobian, projection_info = project_base(H0, raw_h, eps=epsilon, env=env, deadline=deadline)
        except RuntimeError:
            if time.perf_counter() >= deadline:
                raise _BudgetExpired("Projection reached the common budget")
            raise
        volume = volume_function(h, agg.T, grid_points=grid_points, delta_t=agg.deltaT)
        local = []
        for index, client in enumerate(clients):
            if time.perf_counter() >= deadline:
                raise _BudgetExpired("Incomplete client round discarded")
            try:
                result = client.solve(h, deadline=deadline)
            except RuntimeError as exc:
                if time.perf_counter() >= deadline:
                    raise _BudgetExpired("A trial solver reached the common budget")
                invalid_candidates.append({"iteration": iteration, "trial": trial,
                    "reason": "local numerical solver/certificate failure", "client": devices[index]["id"],
                    "detail": str(exc), "elapsed_seconds": time.perf_counter()-training_started})
                return None
            total_local_solves += 1
            if not result.get("optimal", False):
                if time.perf_counter() >= deadline or result.get("status") in (gp.GRB.TIME_LIMIT, "time_limit"):
                    raise _BudgetExpired("Time-limited local solve discarded")
                invalid_candidates.append({"iteration": iteration, "trial": trial,
                    "reason": "nonoptimal local numerical trial", "client": devices[index]["id"],
                    "solver_status": result.get("status"), "elapsed_seconds": time.perf_counter()-training_started})
                return None
            # Offline audit evidence: a dual-feasible primary LP upper bound
            # certifies the max-trace value independently of the tie selection.
            primary_rows = client.primary_model.getConstrs()
            primary_dual = np.asarray(client.primary_model.getAttr("Pi", primary_rows))
            result["primary_dual_normals"] = primary_dual[:client.q*client.n].reshape(client.q, client.n).copy()
            result["primary_dual_containment"] = primary_dual[client.q*client.n:client.q*client.n+client.q].copy()
            result["primary_dual_objective"] = float(devices[index]["b"] @ result["primary_dual_containment"])
            local.append(copy.deepcopy(result))
            if (index + 1) % 10 == 0 or index + 1 == len(clients):
                event(event="client_round", iteration=iteration, trial=trial,
                      completed_clients=index+1, total_clients=len(clients))
        s = sum(value["gamma"] for value in local)
        S = sum(value["Gamma"] for value in local)
        candidate_evaluations += 1
        singular_values = np.linalg.svd(S, compute_uv=False)
        sign, logdet = np.linalg.slogdet(S)
        rank_threshold = 1e-12 if modified else agg.T*np.finfo(float).eps
        full_rank = bool(sign != 0 and singular_values[-1] > rank_threshold*singular_values[0])
        if not full_rank:
            invalid_candidates.append({"iteration": iteration, "trial": trial,
                "reason": "singular aggregate affine map", "singular_values": singular_values.copy(),
                "raw_h": np.asarray(raw_h).copy(), "projected_h": h.copy(),
                "elapsed_seconds": time.perf_counter()-training_started})
        return {"raw_h": np.asarray(raw_h).copy(), "h": h.copy(), "projection_jacobian": projection_jacobian,
                "projection_info": projection_info, "volume": volume, "s": s, "S": S,
                "local_results": local, "log_determinant": float(logdet),
                "determinant_sign": float(sign), "singular_values": singular_values,
                "aggregate_full_rank": full_rank,
                "condition_number": float(singular_values[0]/singular_values[-1]),
                "log_objective": float(logdet + volume["log_volume"]) if full_rank else -np.inf}

    try:
        for index, device in enumerate(devices):
            clients.append(LocalAffineLP(H0, device, env=env, mode=mode))
            if (index + 1) % 10 == 0 or index + 1 == len(devices):
                event(event="model_preparation", completed_clients=index+1, total_clients=len(devices))
        setup_seconds = time.perf_counter() - setup_started
        training_started = time.perf_counter()
        deadline = training_started + timelimit
        initial = evaluate(np.ones(len(H0)), 0, 0)
        if initial is None:
            raise RuntimeError("All clients must solve and certify the initial base before training")
        if time.perf_counter() >= deadline:
            raise _BudgetExpired("No complete initial iterate within the training budget")
        accepted = initial
        history.append({"iteration": 0, "trial": 0, "accepted": True,
                        "log_objective": accepted["log_objective"], "log_volume": accepted["volume"]["log_volume"],
                        "condition_number": accepted["condition_number"],
                        "elapsed_seconds": time.perf_counter()-training_started})
        event(event="accepted", **history[-1])
        iteration = 0
        next_step = initial_step
        while time.perf_counter() < deadline:
            iteration += 1
            if not accepted["aggregate_full_rank"]:
                raise KKTRegularityError("Equation (19) requires an invertible aggregate matrix", {
                    "failure": "singular_aggregate_affine_map",
                    "singular_values": accepted["singular_values"].tolist()})
            upstream = np.linalg.solve(accepted["S"].T, np.eye(agg.T))
            grad = accepted["volume"]["log_gradient"].copy()
            for client in clients:
                if time.perf_counter() >= deadline:
                    raise _BudgetExpired("Gradient round ended at the budget")
                grad += client.vjp(upstream)
            if not modified and not accepted["volume"].get("unique_classical_gradient_asserted", True):
                raise KKTRegularityError("Appendix-C classical volume derivative is not defined at coincident first-period bounds", {
                    "failure": "coincident_volume_boundaries",
                    "coincident_first_period_rhs_pairs": accepted["volume"].get("coincident_first_period_rhs_pairs", [])})
            branch_records = [getattr(client, "last_vjp_metadata", {}) for client in clients]
            gradient_diagnostics.append({
                "iteration": iteration,
                "degenerate_client_count": sum(not item.get("classical_regularity_conditions_verified", False) for item in branch_records),
                "maximum_adjoint_residual_inf": max(float(item.get("adjoint_residual_inf", 0.)) for item in branch_records),
                "classical_jacobian_certified_for_all_clients": all(item.get("classical_regularity_conditions_verified", False) for item in branch_records),
                "kkt_dimensions": [item.get("kkt_dimension") for item in branch_records]})
            if gradient_diagnostics[-1]["maximum_adjoint_residual_inf"] > 1e-7:
                raise ArithmeticError("The original KKT adjoint failed its residual check")
            grad = accepted["projection_jacobian"].T @ grad
            if not np.isfinite(grad).all():
                raise ArithmeticError("Nonfinite outer derivative")
            gradient_count += 1
            norm = float(np.linalg.norm(grad))
            if norm <= gradient_tolerance:
                stop_reason = "gradient_tolerance"
                break
            direction = grad / max(1., norm)
            derivative = float(grad @ direction)
            alpha, improved = next_step, False
            for trial in range(max_backtracking):
                if time.perf_counter() >= deadline:
                    raise _BudgetExpired("Backtracking ended at the budget")
                raw_trial = accepted["raw_h"] + alpha * direction
                try:
                    candidate = evaluate(raw_trial, iteration, trial)
                except ArithmeticError as exc:
                    if "singular aggregate" not in str(exc):
                        raise
                    candidate = None
                if time.perf_counter() >= deadline:
                    raise _BudgetExpired("Unfinished candidate discarded at the budget")
                # Armijo ascent on the ORIGINAL paper objective J=volume*|det S|.
                # Dividing by J(h) gives this stable, algebraically identical
                # log-ratio comparison; this is not Armijo applied to log J.
                sufficient = (candidate is not None and candidate["log_objective"] >=
                              accepted["log_objective"] + np.log1p(armijo * alpha * derivative))
                history.append({"iteration": iteration, "trial": trial, "step": alpha,
                                "gradient_norm": norm, "directional_derivative": derivative,
                                "gradient_scale_identity_residual": float(accepted["h"] @ grad),
                                "gradient_translation_residual": float(np.max(np.abs(H0.T @ grad))),
                                "accepted": bool(sufficient),
                                "log_objective": None if candidate is None else candidate["log_objective"],
                                "elapsed_seconds": time.perf_counter()-training_started})
                if sufficient:
                    accepted = candidate
                    improved = True
                    if modified:
                        next_step = min(initial_step, 2.*alpha)
                    event(event="accepted", **history[-1])
                    break
                event(event="backtracking", **history[-1])
                alpha *= backtracking_factor
            if not improved:
                stop_reason = "line_search_stalled"
                break
        else:
            stop_reason = "time_limit"
    except KKTRegularityError as exc:
        stop_reason = "paper_derivative_or_inverse_undefined"
        regularity_failure = {"message": str(exc),
            "metadata": getattr(exc, "metadata", getattr(exc, "diagnostics", {}))}
    except (_BudgetExpired, TimeoutError):
        stop_reason = "time_limit"
        if accepted is None:
            raise RuntimeError("The budget expired before all clients produced an initial certified aggregate")
    finally:
        elapsed = 0. if training_started is None else time.perf_counter() - training_started
        for client in clients:
            if hasattr(client, "close"):
                client.close()
            elif hasattr(client, "model"):
                client.model.dispose()
        env.dispose()
    if accepted is None:
        raise RuntimeError("No complete feasible federated solution")
    accepted.update(H0=H0, devices=devices)
    accepted["diagnostics"] = {
        "status": "unsupported_paper_regularity" if regularity_failure else "completed",
        "paper_core_protocol": ("user_authorized_lexicographic_peb_variant_v1" if modified else
            "original_equations_12_14_15_17_19_36_38_algorithm1"),
        "degeneracy_treatment": mode,
        "paper_identical_implementation": not modified,
        "algorithm": "full-affine federated PEB base-set optimization",
        "paper_equations": [9, 12, 14, 15, 17, 18, 19, 21, 22, 39, 40, 41, 42, 43, 45, 46],
        "budget_seconds": float(timelimit), "training_seconds": elapsed,
        "pure_model_preparation_seconds": setup_seconds, "budget_overrun_seconds": max(0., elapsed-timelimit),
        "budget_start": "Before the first projection/local LP; all initialization solves and rejected trials included",
        "budget_excludes": ["shared physical-case generation", "pure sparse model assembly", "independent evaluation and certificate audit"],
        "stop_reason": stop_reason, "gradient_rounds": gradient_count,
        "candidate_evaluations": candidate_evaluations, "local_lp_solves": total_local_solves,
        "accepted_updates": sum(value["accepted"] for value in history)-1,
        "last_accepted_training_seconds": max(value["elapsed_seconds"] for value in history if value["accepted"]),
        "initialization": ("all-one RHS of 94-row PEB with duplicate first-period rows merged" if modified else
            "all-one RHS of original 4T-row PEB [L;-L;I;-I]"), "template_rows": len(H0),
        "grid_points": grid_points, "epsilon": epsilon, "initial_step": initial_step,
        "gradient_tolerance": gradient_tolerance, "armijo": armijo, "backtracking_factor": backtracking_factor,
        "gradient_diagnostics": gradient_diagnostics,
        "max_backtracking": max_backtracking,
        "invalid_candidates": invalid_candidates,
        "objective": "volume(base)*abs(det(sum Gamma)); original-J Armijo evaluated by a stable log ratio",
        "local_objective": ("max trace(Gamma), then min L1(offdiagonal Gamma) at an explicit near-optimal trace target" if modified else
            "max trace(Gamma), exactly the single original LP in equation (12)"),
        "local_selection_note": ("User-authorized numerical variant: trace target=max(0,p-1e-8-1e-9*abs(p)); full Gamma remains unrestricted" if modified else
            "Solver solution of the original LP; no secondary objective, trace target, regularization or selected-basis derivative"),
        "gradient": ("selected secondary-simplex-basis VJP including the primary trace-target derivative; not a certified unique classical derivative" if modified else
            "original primal-dual KKT adjoint, equations (36)-(38), plus Appendix C/D recursive Riemann boundary integrals"),
        "line_search_initial_step_policy": "min(initial_step,2*last_accepted_step)" if modified else "initial_step each round",
        "execution": "single-machine deterministic simulation of client/aggregator computations",
        "training_seed": 0, "solver_options": dict(SOLVER_OPTIONS),
        "volume_discretization_relative_consistency": accepted["volume"]["forward_backward_relative_discrepancy"],
        "volume_quadrature": accepted["volume"]["quadrature"],
        "volume_grid_spacing": accepted["volume"].get("delta_x"),
        "volume_derivative_classical_at_initial_point": accepted["volume"].get("unique_classical_gradient_asserted"),
        "coincident_first_period_rhs_pairs": accepted["volume"].get("coincident_first_period_rhs_pairs", []),
        "history": history,
        "regularity_failure": regularity_failure,
    }
    initial_log_objective = history[0]["log_objective"]
    accepted["diagnostics"].update(initial_log_objective=initial_log_objective,
        final_log_objective=accepted["log_objective"],
        log_objective_gain=accepted["log_objective"]-initial_log_objective)
    if regularity_failure:
        raise PaperAlgorithmFailure(regularity_failure["message"], accepted["diagnostics"], accepted)
    return accepted
