"""Polyhedral/affine core of Dong--Chen--Qin's federated aggregation method.

This implements the full-matrix local LP (12), Farkas containment certificates
(9), projection (14), and the original primal-dual KKT system (36)--(38).
The explicit ``mode="strict"`` retains the unmodified LP/KKT route and raises
KKTRegularityError at singular systems. The separately labelled, user-authorized
``mode="lexicographic"`` uses a near-optimal max-trace face, a secondary min-L1
off-diagonal selection and its selected-simplex-basis VJP. That numerical variant
is not claimed to be identical to the paper or to have a unique classical
derivative. Neither mode uses a ridge or pseudoinverse. The caller owns the outer
volume/backtracking loop and its total wall-clock budget.
"""

import time
import numpy as np
import gurobipy as gp
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import structural_rank

from Simulator.testers.polytope_evaluation import SOLVER_OPTIONS


def _array(value, shape=None, name="array"):
    result = np.asarray(value, dtype=np.float64)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError(f"Invalid finite {name}: {result.shape}, expected {shape}")
    return result.copy()


def peb_template(T, delta_t=1.):
    """Paper equation (3): raw [L,-L,I,-I], retaining all 4T RHS parameters."""
    if isinstance(T, bool) or not isinstance(T, (int, np.integer)) or T < 1:
        raise ValueError("T must be a positive integer")
    if not np.isfinite(delta_t) or delta_t <= 0:
        raise ValueError("delta_t must be positive")
    cumulative = float(delta_t) * np.tril(np.ones((T, T)))
    return np.vstack((cumulative, -cumulative, np.eye(T), -np.eye(T)))


def extract_device_polytopes(agg):
    """Exact per-resource H-polytopes in physical net-injection kW.

    This T24 benchmark has continuous CL/TCL/PV only. TCL temperatures are
    eliminated analytically using the actual model's t>0 recurrence; p(0) does
    not enter that recurrence. No thermal or temporal constraint is discarded.
    """
    T = int(agg.T)
    if len(agg.EV_list) or len(agg.ESS_list):
        raise ValueError("This extraction supports the benchmark's CL/TCL/PV population only")
    if any(bool(device.get("is_discrete", False)) for device in agg.TCL_list):
        raise ValueError("Discrete TCLs do not admit this exact convex H representation")
    I = np.eye(T)
    devices = []
    for index, device in enumerate(agg.CL_list):
        maximum = _array(device["p_max_profile"], (T,), "CL power profile")
        if np.any(maximum < 0):
            raise ValueError("CL upper consumption bounds cannot be negative")
        minimum = np.maximum(0., (1. - float(device["curt_ratio"])) * maximum)
        if np.any(minimum > maximum):
            raise ValueError("CL consumption bounds are inconsistent")
        devices.append({"id": f"cl_{index:03d}", "family": "cl", "resource_index": index,
                        "A": np.vstack((I, -I)), "b": np.r_[-minimum, maximum],
                        "power_lower": -maximum, "power_upper": -minimum,
                        "coordinate_system": "physical_net_injection_kw"})
    ambient = _array(agg.theta_amb, (T,), "ambient temperature")
    for index, device in enumerate(agg.TCL_list):
        H, C, COP, Q = (float(device[key]) for key in ("H", "C", "COP", "Qmax"))
        if not np.isfinite([H, C, COP, Q]).all() or min(H, C, COP) <= 0 or Q < 0:
            raise ValueError("Invalid TCL physical coefficients")
        lower, upper, initial = (float(device[key]) for key in ("temp_min", "temp_max", "temp_set"))
        if not np.isfinite([lower, upper, initial]).all() or not lower <= initial <= upper:
            raise ValueError("TCL initial temperature violates its actual model bounds")
        alpha = float(np.exp(-float(agg.deltaT) * H / C))
        gain = (1. - alpha) * COP / H
        drift, B = np.empty(T), np.zeros((T, T))
        drift[0] = initial
        for t in range(1, T):
            drift[t] = alpha * drift[t-1] + (1. - alpha) * ambient[t]
            B[t, 1:t+1] = gain * alpha ** np.arange(t-1, -1, -1)
        # z=-p_TCL, theta=drift-Bz; p_TCL<=Qmax/COP (not Qmax).
        maximum = Q / COP
        devices.append({"id": f"tcl_{index:03d}", "family": "tcl", "resource_index": index,
                        "A": np.vstack((I, -I, -B[1:], B[1:])),
                        "b": np.r_[np.zeros(T), np.full(T, maximum), upper-drift[1:], drift[1:]-lower],
                        "power_lower": np.full(T, -maximum), "power_upper": np.zeros(T),
                        "temperature_drift": drift, "temperature_net_power_matrix": -B,
                        "temperature_lower": lower, "temperature_upper": upper,
                        "alpha": alpha, "time_zero_power_enters_temperature": False,
                        "coordinate_system": "physical_net_injection_kw"})
    for index, device in enumerate(agg.PV_list):
        maximum = _array(device["output_profile"], (T,), "PV output profile")
        if np.any(maximum < 0):
            raise ValueError("PV upper output bounds cannot be negative")
        devices.append({"id": f"pv_{index:03d}", "family": "pv", "resource_index": index,
                        "A": np.vstack((I, -I)), "b": np.r_[maximum, np.zeros(T)],
                        "power_lower": np.zeros(T), "power_upper": maximum,
                        "fixed_zero_coordinates": np.flatnonzero(maximum == 0),
                        "coordinate_system": "physical_net_injection_kw"})
    return devices


def _coordinate_box(H, h, tol=1e-12):
    n = H.shape[1]
    lower, upper = np.full(n, -np.inf), np.full(n, np.inf)
    for row, bound in zip(H, h):
        nonzero = np.flatnonzero(np.abs(row) > tol)
        if len(nonzero) == 1:
            j = int(nonzero[0])
            if row[j] > 0:
                upper[j] = min(upper[j], bound / row[j])
            else:
                lower[j] = max(lower[j], bound / row[j])
    return (lower, upper) if np.isfinite(lower).all() and np.isfinite(upper).all() else None


def check_affine_certificate(H0, h, device, solution, tol=1e-7):
    """Audit (9), plus a residual-aware sufficient bound using the base box."""
    H0, h = _array(H0), _array(h)
    A, b = _array(device["A"]), _array(device["b"])
    n, m, q = H0.shape[1], H0.shape[0], A.shape[0]
    gamma = _array(solution["gamma"], (n,), "gamma")
    Gamma = _array(solution["Gamma"], (n, n), "Gamma")
    Lambda = _array(solution["Lambda"], (q, m), "Lambda")
    equality = Lambda @ H0 - A @ Gamma
    containment = Lambda @ h + A @ gamma - b
    nonnegative_violation = max(0., float(-Lambda.min()))
    report = {"passed": False, "absolute_tolerance": float(tol),
              "minimum_lambda": float(Lambda.min()), "nonnegative_violation": nonnegative_violation,
              "maximum_absolute_equality_residual": float(np.max(np.abs(equality))),
              "maximum_rhs_violation": float(np.max(containment)),
              "farkas_form": "Lambda>=0; Lambda H0=Ai Gamma; Lambda h+Ai gamma<=bi"}
    box = _coordinate_box(H0, h)
    if box is not None:
        lower, upper = box
        if np.any(lower > upper + tol):
            raise ValueError("The supplied base has inconsistent coordinate bounds")
        positive = np.maximum(Lambda, 0.)
        residual = A @ Gamma - positive @ H0
        support = np.maximum(residual, 0.) @ upper + np.minimum(residual, 0.) @ lower
        robust = A @ gamma + positive @ h + support - b
        report.update(base_coordinate_lower=lower, base_coordinate_upper=upper,
                      maximum_residual_aware_containment_violation=float(robust.max()))
    else:
        report["maximum_residual_aware_containment_violation"] = None
    report["passed"] = bool(nonnegative_violation <= tol
        and report["maximum_absolute_equality_residual"] <= tol and report["maximum_rhs_violation"] <= tol
        and (box is None or report["maximum_residual_aware_containment_violation"] <= tol))
    return report


def _environment(env):
    if env is not None:
        return env, False
    owned = gp.Env(empty=True)
    owned.setParam("OutputFlag", 0)
    owned.start()
    return owned, True


def _deadline(model, deadline):
    remaining = gp.GRB.INFINITY if deadline is None else float(deadline)-time.perf_counter()
    if remaining <= 0:
        raise TimeoutError("The shared aggregation deadline has elapsed")
    model.setParam("TimeLimit", max(0.001, remaining))


class KKTRegularityError(ArithmeticError):
    """The original paper's implicit linear system is not regularly solvable.

    ``diagnostics`` is JSON-compatible and is also available as ``metadata``.
    The caller must report this failure; this exception never triggers an
    auxiliary objective, pseudoinverse, regularization, or alternative LP solve.
    """
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.metadata = diagnostics


class LocalAffineLP:
    """Original max-trace LP with an explicit, auditable degeneracy mode.

    The unrestricted full Gamma, gamma and nonnegative Lambda solve exactly
    equation (12). ``vjp`` applies equation (38) through the transposed *full*
    primal-dual KKT Jacobian. Row-major storage is only a permutation of the
    paper's column-major convention. No simplex-basis derivative is substituted
    when that full Jacobian is singular in ``mode="strict"``. The optional
    ``mode="lexicographic"`` is a numerical variant with a separately solved
    near-optimal secondary LP and a labelled selected-basis sensitivity.
    """
    def __init__(self, H0, device, env=None, *, mode="strict"):
        started = time.perf_counter()
        if mode not in ("strict", "lexicographic"):
            raise ValueError("mode must be 'strict' or 'lexicographic'")
        self.mode = mode
        self.H0 = _array(H0)
        if self.H0.ndim != 2:
            raise ValueError("H0 must be a matrix")
        self.n, self.m = self.H0.shape[1], self.H0.shape[0]
        self.device = device
        self.A = _array(device["A"])
        self.b = _array(device["b"], (len(self.A),), "device RHS")
        if self.A.shape[1] != self.n:
            raise ValueError("Base and device coordinate dimensions differ")
        self.q = len(self.A)
        self.env, self._owns_env = _environment(env)
        self.model = gp.Model("federated_" + str(device.get("id", "device")), env=self.env)
        self.primary_model = self.model  # Compatibility alias: there is one LP.
        for key, value in SOLVER_OPTIONS.items():
            self.model.setParam(key, value)
        self.model.setParam("Method", 1)
        self.gamma_offset, self.Gamma_offset = 0, self.n
        self.Lambda_offset = self.n+self.n*self.n
        count = self.Lambda_offset+self.q*self.m
        lower = np.r_[np.full(self.Lambda_offset, -gp.GRB.INFINITY), np.zeros(self.q*self.m)]
        self.x = self.model.addMVar(count, lb=lower, name="affine")
        I = sp.eye(self.n, format="csr")
        equalities = sp.hstack((sp.csr_matrix((self.q*self.n, self.n)),
                               -sp.kron(sp.csr_matrix(self.A), I),
                               sp.kron(sp.eye(self.q), sp.csr_matrix(self.H0.T))), format="csr")
        self.model.addMConstr(equalities, self.x, "=", np.zeros(self.q*self.n), name="normal_map")
        self._contain_start = self.q*self.n
        containment = sp.hstack((sp.csr_matrix(self.A), sp.csr_matrix((self.q, self.n*self.n)),
                                sp.kron(sp.eye(self.q), np.ones((1, self.m)))), format="csr")
        self.model.addMConstr(containment, self.x, "<", self.b, name="containment")
        objective = np.zeros(count)
        objective[self.Gamma_offset+np.arange(self.n)*(self.n+1)] = 1.
        self.model.setObjective(objective @ self.x, gp.GRB.MAXIMIZE)
        self.model.update()
        self._variables = self.model.getVars()
        self._normal_rows = self.model.getConstrs()[:self._contain_start]
        self._contain_rows = self.model.getConstrs()[self._contain_start:]
        self._lambda_variables = self._variables[self.Lambda_offset:]
        self._current_h = np.ones(self.m)
        self.last_solution = self.last_h = self.last_vjp_metadata = None
        self._kkt_cache = None
        self._primary_solution = None
        self._secondary_basis_cache = None
        self.last_deadline = None
        self.secondary_model = None
        if self.mode == "lexicographic":
            self.model.setParam("NumericFocus", 2)
            self.model.update()
            self.secondary_model = self.model.copy(env=self.env)
            self.secondary_model.ModelName = "federated_numerical_lexicographic_" + str(device.get("id", "device"))
            secondary_variables = self.secondary_model.getVars()
            offdiagonal = [self.Gamma_offset+i*self.n+j for i in range(self.n)
                           for j in range(self.n) if i != j]
            absolute = self.secondary_model.addVars(len(offdiagonal),lb=0.,name="offdiagonal_absolute")
            for index,variable_index in enumerate(offdiagonal):
                self.secondary_model.addLConstr(secondary_variables[variable_index]-absolute[index]<=0.,
                                               name=f"offdiagonal_plus_{index}")
                self.secondary_model.addLConstr(-secondary_variables[variable_index]-absolute[index]<=0.,
                                               name=f"offdiagonal_minus_{index}")
            self._secondary_trace_constraint = self.secondary_model.addLConstr(
                gp.LinExpr([1.]*self.n,
                    [secondary_variables[self.Gamma_offset+i*(self.n+1)] for i in range(self.n)]),
                gp.GRB.EQUAL,0.,name="numerical_primary_trace_target")
            self.secondary_model.setObjective(gp.quicksum(absolute.values()),gp.GRB.MINIMIZE)
            self.secondary_model.update()
            self._secondary_variables = self.secondary_model.getVars()
            self._secondary_contain_rows = self.secondary_model.getConstrs()[self._contain_start:self._contain_start+self.q]
            self._secondary_lambda_variables = self._secondary_variables[self.Lambda_offset:self.Lambda_offset+self.q*self.m]
            self._secondary_trace_row = self.secondary_model.NumConstrs-1
            self._secondary_current_h = np.ones(self.m)
        self.build_seconds = time.perf_counter()-started

    def _check_deadline(self, deadline=None):
        deadline = self.last_deadline if deadline is None else deadline
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("The shared aggregation deadline has elapsed during KKT differentiation")

    def solve(self, h, deadline=None):
        """Solve the chosen mode; no automatic fallback from strict to variant."""
        started = time.perf_counter()
        primary = self._solve_primary(h,deadline)
        self._primary_solution = primary
        if self.mode == "strict" or not primary["optimal"]:
            return primary
        return self._solve_secondary(primary,started,deadline)

    def _solve_primary(self, h, deadline=None):
        h = _array(h, (self.m,), "base RHS")
        self.last_deadline = deadline
        _deadline(self.model, deadline)
        started = time.perf_counter()
        changed = np.flatnonzero(h != self._current_h)
        for i, row in enumerate(self._contain_rows):
            offset = i*self.m
            for j in changed:
                self.model.chgCoeff(row, self._lambda_variables[offset+int(j)], float(h[j]))
        self._current_h = h.copy()
        _deadline(self.model, deadline)
        self.model.optimize()
        status = int(self.model.Status)
        report = {"status": status, "optimal": status == gp.GRB.OPTIMAL,
                  "has_solution": self.model.SolCount > 0,
                  "solver_seconds": float(self.model.Runtime),
                  "elapsed_seconds": time.perf_counter()-started,
                  "build_seconds": self.build_seconds,
                  "device_id": str(self.device.get("id", "device")),
                  "full_matrix_affine_map": True,
                  "local_objective": "original equation (12): max trace(Gamma)",
                  "mode":self.mode,"paper_exact":self.mode=="strict",
                  "auxiliary_objectives": False, "trace_target_shift": False}
        self._kkt_cache, self.last_vjp_metadata = None, None
        self._secondary_basis_cache = None
        self.last_h = h
        if self.model.SolCount:
            values = np.asarray(self.x.X, dtype=float)
            report.update(gamma=values[:self.n].copy(),
                          Gamma=values[self.Gamma_offset:self.Lambda_offset].reshape(self.n,self.n).copy(),
                          Lambda=values[self.Lambda_offset:].reshape(self.q,self.m).copy(),
                          trace=float(self.model.ObjVal), primary_trace_optimum=float(self.model.ObjVal))
            report["certificate"] = check_affine_certificate(self.H0,h,self.device,report)
            if report["optimal"]:
                Y = np.asarray(self.model.getAttr("Pi",self._normal_rows)).reshape(self.q,self.n)
                mu = np.asarray(self.model.getAttr("Pi",self._contain_rows))
                nu = -np.asarray(self.model.getAttr("RC",self._lambda_variables)).reshape(self.q,self.m)
                report.update(primary_dual_normals=Y.copy(), primary_dual_containment=mu.copy(),
                              primary_dual_nonnegativity=nu.copy(),
                              primary_dual_objective=float(self.b@mu),
                              primary_value_gradient=-report["Lambda"].T@mu)
                if not report["certificate"]["passed"]:
                    raise RuntimeError(f"Optimal local solve failed its Farkas certificate: {report['device_id']}")
        self.last_solution = report
        return report

    def _solve_secondary(self, primary, started, deadline):
        """User-authorized numerical selection; never advertised as paper-exact."""
        h = self.last_h
        changed = np.flatnonzero(h!=self._secondary_current_h)
        for i,row in enumerate(self._secondary_contain_rows):
            offset=i*self.m
            for j in changed:
                self.secondary_model.chgCoeff(row,self._secondary_lambda_variables[offset+int(j)],float(h[j]))
        self._secondary_current_h = h.copy()
        optimum = primary["primary_trace_optimum"]
        tolerance = 1e-8+1e-9*abs(optimum)
        if optimum < -tolerance:
            raise ArithmeticError("A negative maximum trace contradicts the feasible zero affine map")
        target = max(0.,optimum-tolerance)
        self._trace_target_derivative_scale = (1.-1e-9*np.sign(optimum)) if target>0. else 0.
        self._secondary_trace_constraint.RHS=target
        _deadline(self.secondary_model,deadline)
        self.secondary_model.optimize()
        status=int(self.secondary_model.Status)
        report={**primary,"status":status,"optimal":status==gp.GRB.OPTIMAL,
            "has_solution":self.secondary_model.SolCount>0,
            "primary_status":primary["status"],"secondary_status":status,
            "primary_solver_seconds":primary["solver_seconds"],
            "secondary_solver_seconds":float(self.secondary_model.Runtime),
            "solver_seconds":primary["solver_seconds"]+float(self.secondary_model.Runtime),
            "elapsed_seconds":time.perf_counter()-started,
            "primary_trace_tolerance":tolerance,"secondary_trace_target":target,
            "trace_target_derivative_scale":self._trace_target_derivative_scale,
            "mode":"lexicographic","auxiliary_objectives":True,"trace_target_shift":True,
            "paper_exact":False,"local_numeric_focus":2,
            "selection":"user-authorized numerical variant: min L1(offdiagonal Gamma) at an explicitly near-optimal trace target",
            "primary_gamma":primary["gamma"].copy(),"primary_Gamma":primary["Gamma"].copy(),
            "primary_Lambda":primary["Lambda"].copy()}
        # Never retain a primary primal point under a secondary status if that
        # secondary solve did not return its own incumbent.
        for key in ("gamma","Gamma","Lambda","trace","certificate"):
            report.pop(key,None)
        if self.secondary_model.SolCount:
            values=np.asarray(self.secondary_model.getAttr("X",self._secondary_variables))
            selected=values[self.Gamma_offset:self.Lambda_offset].reshape(self.n,self.n)
            report.update(gamma=values[:self.n].copy(),Gamma=selected.copy(),
                Lambda=values[self.Lambda_offset:self.Lambda_offset+self.q*self.m].reshape(self.q,self.m).copy(),
                trace=float(np.trace(selected)),secondary_offdiagonal_l1=float(self.secondary_model.ObjVal))
            report["primary_trace_gap"]=optimum-report["trace"]
            report["primary_optimality_equality_residual"]=abs(report["primary_trace_gap"])
            report["secondary_trace_target_residual"]=abs(report["trace"]-target)
            if report["optimal"] and (report["secondary_trace_target_residual"]>1e-8
                    or report["primary_trace_gap"]>tolerance+1e-8 or report["primary_trace_gap"] < -1e-8):
                raise ArithmeticError("Numerical selection exceeded its explicit primary trace tolerance")
            report["certificate"]=check_affine_certificate(self.H0,h,self.device,report)
            if report["optimal"] and not report["certificate"]["passed"]:
                raise ArithmeticError("Numerical selection failed the unchanged physical containment certificate")
            if report["optimal"]:
                rows=self.secondary_model.getConstrs()
                dual=np.asarray(self.secondary_model.getAttr("Pi",rows))
                reduced=np.asarray(self.secondary_model.getAttr("RC",self._secondary_variables))
                offdiagonal_count=self.n*(self.n-1)
                offdiagonal_dual=dual[self._contain_start+self.q:self._secondary_trace_row].reshape(offdiagonal_count,2)
                report.update(
                    secondary_dual_normals=dual[:self._contain_start].reshape(self.q,self.n).copy(),
                    secondary_dual_containment=dual[self._contain_start:self._contain_start+self.q].copy(),
                    secondary_dual_nonnegativity=reduced[self.Lambda_offset:self.Lambda_offset+self.q*self.m].reshape(self.q,self.m).copy(),
                    secondary_dual_offdiagonal_plus=offdiagonal_dual[:,0].copy(),
                    secondary_dual_offdiagonal_minus=offdiagonal_dual[:,1].copy(),
                    secondary_dual_trace=float(dual[self._secondary_trace_row]),
                    secondary_dual_objective=float(self.b@dual[self._contain_start:self._contain_start+self.q]+target*dual[self._secondary_trace_row]),
                    secondary_dual_convention="Gurobi minimization: Pi<=0 for <= rows; RC>=0 at nonnegative lower bounds")
        self.last_solution=report
        return report

    def _original_kkt(self, deadline=None):
        """Build G_z from (36)--(38), including both complementarity blocks."""
        result = self._primary_solution or self.last_solution
        if result is None or not result["optimal"]:
            raise RuntimeError("A current OPTIMAL original max-trace solve is required")
        self._check_deadline(deadline)
        if self._kkt_cache is not None:
            return self._kkt_cache
        started = time.perf_counter()
        # The strict diagnostic always audits the original primary primal/dual,
        # including when a separate secondary point is used by the variant.
        n,m,q = self.n,self.m,self.q
        H,A,h = self.H0,self.A,self.last_h
        Gamma,Lambda,gamma = result["Gamma"],result["Lambda"],result["gamma"]
        Y,mu,nu = (result[key] for key in ("primary_dual_normals","primary_dual_containment","primary_dual_nonnegativity"))
        g = Lambda@h+A@gamma-self.b
        Iq,In = sp.eye(q,format="csr"),sp.eye(n,format="csr")
        # Columns: gamma, vec_r(Gamma), vec_r(Lambda), vec_r(Y), mu, vec_r(nu).
        # Rows: stationarity_gamma/Gamma/Lambda, normals, mu*g, nu*Lambda.
        J = sp.bmat([
            [None,None,None,None,sp.csr_matrix(A.T),None],
            [None,None,None,-sp.kron(sp.csr_matrix(A.T),In),None,None],
            [None,None,None,sp.kron(Iq,sp.csr_matrix(H)),
             sp.kron(Iq,sp.csr_matrix(h[:,None])),-sp.eye(q*m)],
            [None,-sp.kron(sp.csr_matrix(A),In),sp.kron(Iq,sp.csr_matrix(H.T)),None,None,None],
            [sp.diags(mu)@sp.csr_matrix(A),None,
             sp.diags(mu)@sp.kron(Iq,sp.csr_matrix(h[None,:])),None,sp.diags(g),None],
            [None,None,sp.diags(nu.ravel()),None,None,sp.diags(Lambda.ravel())]
        ],format="csc")
        J.eliminate_zeros()
        residuals = [A.T@mu,-np.eye(n)-A.T@Y,
                     Y@H.T+mu[:,None]*h[None,:]-nu,
                     Lambda@H-A@Gamma,mu*g,nu*Lambda]
        residual = float(max(np.max(np.abs(value)) for value in residuals))
        weak_inequalities = np.flatnonzero((np.abs(mu)<=1e-8)&(np.abs(g)<=1e-8))
        weak_lower = np.flatnonzero((np.abs(Lambda.ravel())<=1e-8)&(np.abs(nu.ravel())<=1e-8))
        row_nnz = np.diff(J.tocsr().indptr)
        zero_rows = np.flatnonzero(row_nnz==0)
        structural = int(structural_rank(J))
        self._check_deadline(deadline)
        metadata = {"derivative_method":"original_full_primal_dual_KKT_equations_36_38",
                    "audited_model":"original primary max-trace LP",
                    "device_id":result["device_id"],"kkt_dimension":J.shape[0],"kkt_nnz":J.nnz,
                    "kkt_structural_rank":structural,"zero_jacobian_rows":int(len(zero_rows)),
                    "first_zero_jacobian_rows":zero_rows[:16].tolist(),
                    "weak_containment_complementarity_pairs":int(len(weak_inequalities)),
                    "weak_nonnegativity_complementarity_pairs":int(len(weak_lower)),
                    "primal_dual_kkt_residual_inf":residual,
                    "dual_nonnegative_violation":float(max(0.,-mu.min(),-nu.min())),
                    "classical_regularity_conditions_verified":False,
                    "singular":bool(structural<J.shape[0]),
                    "auxiliary_objectives":False,"trace_target_shift":False,
                    "ridge":False,"pseudoinverse":False,"selected_lp_basis":False,
                    "assembly_seconds":time.perf_counter()-started,
                    "vectorization":"row-major simultaneous permutation of Appendix B column-major system"}
        self._kkt_cache = (J,mu,Lambda,metadata)
        return self._kkt_cache

    def kkt_diagnostics(self, deadline=None):
        """Read-only audit of the original KKT system, without another LP solve."""
        _,_,_,metadata = self._original_kkt(deadline)
        return dict(metadata)

    def vjp(self, W, deadline=None):
        """Differentiate only the explicitly chosen strict or numerical mode."""
        if self.mode=="lexicographic":
            return self._lexicographic_vjp(W,deadline)
        return self._strict_vjp(W,deadline)

    def _secondary_basis(self, deadline=None):
        """A selected simplex branch of the authorized numerical secondary LP."""
        if self.last_solution is None or not self.last_solution["optimal"]:
            raise RuntimeError("A current OPTIMAL secondary numerical solve is required")
        self._check_deadline(deadline)
        if self._secondary_basis_cache is not None:
            return self._secondary_basis_cache
        started=time.perf_counter()
        model=self.secondary_model
        matrix=model.getA().tocsr()
        variables=self._secondary_variables
        rows=model.getConstrs()
        vb=np.asarray(model.getAttr("VBasis",variables),dtype=int)
        cb=np.asarray(model.getAttr("CBasis",rows),dtype=int)
        basic=np.flatnonzero(vb==0)
        slacks=np.flatnonzero(cb==0)
        if len(basic)+len(slacks)!=matrix.shape[0]:
            raise KKTRegularityError("Secondary solver did not expose a square simplex basis",
                                     {"mode":self.mode,"basis_rows":matrix.shape[0],"basic_columns":len(basic)+len(slacks)})
        B=sp.hstack((matrix[:,basic],sp.eye(matrix.shape[0],format="csr")[:,slacks]),format="csc")
        try:
            lu=splu(B)
        except RuntimeError as exc:
            raise KKTRegularityError("Selected numerical secondary basis is singular",
                                     {"mode":self.mode,"factorization_error":str(exc)}) from exc
        values=np.asarray(model.getAttr("X",variables))
        lower=np.asarray(model.getAttr("LB",variables));upper=np.asarray(model.getAttr("UB",variables))
        at_bound=basic[((np.abs(values[basic]-lower[basic])<=1e-8)
                       |(np.abs(values[basic]-upper[basic])<=1e-8))]
        slack=np.asarray(model.getAttr("Slack",rows))
        zero_slacks=slacks[np.abs(slack[slacks])<=1e-8]
        reduced=np.asarray(model.getAttr("RC",variables))
        zero_reduced=np.flatnonzero((vb!=0)&(np.abs(reduced)<=1e-8))
        equality_slacks=slacks[(slacks<self._contain_start)|(slacks==self._secondary_trace_row)]
        degenerate=bool(len(at_bound) or len(zero_slacks) or len(zero_reduced) or len(equality_slacks))
        metadata={"mode":"lexicographic","paper_exact":False,
            "derivative_method":"user-authorized numerical secondary selected-simplex-basis VJP",
            "classical_regularity_conditions_verified":False,
            "degenerate_or_nonunique_basis":degenerate,
            "selected_lp_basis":True,"ridge":False,"pseudoinverse":False,
            "primary_value_chain_rule_included":True,
            "basis_dimension":B.shape[0],"basis_nnz":B.nnz,
            "basic_variables":basic,"basic_slack_rows":slacks,
            "basic_variables_at_bounds":at_bound,"degenerate_basic_slack_rows":zero_slacks,
            "nonbasic_zero_reduced_cost_variables":zero_reduced,
            "basic_equality_slack_rows":equality_slacks,
            "factor_seconds":time.perf_counter()-started,
            "note":"A selected numerical branch at degeneracy; no unique classical Jacobian or branch feasibility is certified."}
        self._check_deadline(deadline)
        self._secondary_basis_cache=(lu,basic,B,metadata)
        return self._secondary_basis_cache

    def _lexicographic_vjp(self, W, deadline=None):
        W=_array(W,(self.n,self.n),"Gamma VJP weight")
        lu,basic,B,metadata=self._secondary_basis(deadline)
        rhs=np.zeros(B.shape[0])
        where=np.flatnonzero((basic>=self.Gamma_offset)&(basic<self.Lambda_offset))
        rhs[where]=W.ravel()[basic[where]-self.Gamma_offset]
        adjoint=lu.solve(rhs,trans="T")
        residual=float(np.max(np.abs(B.T@adjoint-rhs)))
        self.last_vjp_metadata={**metadata,"adjoint_residual_inf":residual,
            "primary_trace_adjoint":float(adjoint[self._secondary_trace_row]),
            "trace_target_derivative_scale":self._trace_target_derivative_scale}
        self._check_deadline(deadline)
        if not np.isfinite(adjoint).all() or residual>1e-7:
            raise KKTRegularityError("Numerical selected-basis adjoint failed its residual audit",self.last_vjp_metadata)
        gradient=(-self.last_solution["Lambda"].T@adjoint[self._contain_start:self._contain_start+self.q]
            +adjoint[self._secondary_trace_row]*self._trace_target_derivative_scale*self._primary_solution["primary_value_gradient"])
        if not np.isfinite(gradient).all():
            raise KKTRegularityError("Numerical selected-basis VJP is nonfinite",self.last_vjp_metadata)
        return gradient

    def _strict_vjp(self, W, deadline=None):
        """Equation (38) adjoint; raise rather than invent a singular derivative."""
        W = _array(W,(self.n,self.n),"Gamma VJP weight")
        J,mu,Lambda,metadata = self._original_kkt(deadline)
        self.last_vjp_metadata = dict(metadata)
        if metadata["primal_dual_kkt_residual_inf"]>1e-7 or metadata["dual_nonnegative_violation"]>1e-7:
            raise KKTRegularityError("Original LP primal-dual solution failed the KKT residual audit",self.last_vjp_metadata)
        if metadata["singular"]:
            raise KKTRegularityError("Original Appendix-B KKT Jacobian is structurally singular; equation (38) is undefined",self.last_vjp_metadata)
        if metadata["weak_containment_complementarity_pairs"] or metadata["weak_nonnegativity_complementarity_pairs"]:
            raise KKTRegularityError("Original LP lacks numerically separated complementarity; no regular Jacobian is certified",self.last_vjp_metadata)
        self._check_deadline(deadline)
        started = time.perf_counter()
        try:
            lu = splu(J)
        except RuntimeError as exc:
            self.last_vjp_metadata.update(singular=True,factorization_error=str(exc))
            raise KKTRegularityError("Original Appendix-B KKT Jacobian has a singular numerical factorization",self.last_vjp_metadata) from exc
        pivots = np.abs(lu.U.diagonal())
        pivot_ratio = float(pivots.min()/pivots.max()) if pivots.max()>0 else 0.
        self.last_vjp_metadata.update(lu_pivot_magnitude_ratio=pivot_ratio)
        if pivot_ratio<=np.finfo(float).eps*J.shape[0]:
            self.last_vjp_metadata.update(singular=True)
            raise KKTRegularityError("Original Appendix-B KKT Jacobian is numerically singular at working precision",self.last_vjp_metadata)
        rhs = np.zeros(J.shape[0])
        rhs[self.n:self.n+self.n*self.n] = W.ravel()
        adjoint = lu.solve(rhs,trans="T")
        adjoint_residual = float(np.max(np.abs(J.T@adjoint-rhs)))
        self.last_vjp_metadata.update(adjoint_residual_inf=adjoint_residual,
                                     factor_and_solve_seconds=time.perf_counter()-started)
        self._check_deadline(deadline)
        if not np.isfinite(adjoint).all() or adjoint_residual>1e-7:
            raise KKTRegularityError("Original Appendix-B KKT adjoint failed its strict linear-system residual audit",self.last_vjp_metadata)
        start_third = self.n+self.n*self.n
        a3 = adjoint[start_third:start_third+self.q*self.m].reshape(self.q,self.m)
        start_fifth = start_third+self.q*self.m+self.q*self.n
        a5 = adjoint[start_fifth:start_fifth+self.q]
        # -G_h.T a: G3_h[dh]=mu dh.T; G5_h[dh]=mu*(Lambda dh).
        gradient = -a3.T@mu-Lambda.T@(mu*a5)
        if not np.isfinite(gradient).all():
            raise KKTRegularityError("Original Appendix-B KKT derivative is nonfinite",self.last_vjp_metadata)
        self.last_vjp_metadata.update(classical_regularity_conditions_verified=True,
                                     original_kkt_linear_system_solved=True,singular=False)
        return gradient

    def close(self):
        if self.secondary_model is not None:
            self.secondary_model.dispose()
        self.model.dispose()
        if self._owns_env:
            self.env.dispose()

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()


def project_base(H0, h, eps=1e-5, env=None, deadline=None, active_tol=1e-7):
    """Projection (14), with the unique RHS projection's active-branch Jacobian."""
    H0 = _array(H0)
    m, n = H0.shape
    h = _array(h, (m,), "unprojected base RHS")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("The nonempty-interior margin must be positive")
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("The shared aggregation deadline has elapsed")
    if np.all(h > eps+active_tol):
        # u=0 strictly satisfies (14b) with h'=h. The nonnegative squared
        # objective is exactly zero, proving this projection is globally optimal.
        return h.copy(), np.eye(m), {
            "status": "optimal", "epsilon": float(eps), "interior_witness": np.zeros(n),
            "minimum_margin": float(h.min()), "squared_projection_distance": 0.,
            "active_rows": np.empty(0, dtype=int), "weakly_active_rows": np.empty(0, dtype=int),
            "active_normal_rank": 0, "classical_regularity_conditions_verified": True,
            "jacobian_method": "identity at a certified strict-interior zero-distance projection",
            "projection_solved_analytically": True, "maximum_primal_violation": 0.,
            "stationarity_h_inf": 0., "stationarity_u_inf": 0., "solver_seconds": 0., "elapsed_seconds": 0.}
    environment, owned = _environment(env)
    model = gp.Model("federated_base_projection", env=environment)
    try:
        for key, value in SOLVER_OPTIONS.items():
            model.setParam(key, value)
        hp = model.addMVar(m, lb=-gp.GRB.INFINITY, name="projected_rhs")
        u = model.addMVar(n, lb=-gp.GRB.INFINITY, name="interior_witness")
        constraints = model.addConstr(H0 @ u + eps <= hp, name="strict_base")
        model.setObjective((hp-h) @ (hp-h), gp.GRB.MINIMIZE)
        _deadline(model, deadline)
        started = time.perf_counter()
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"Base projection was not OPTIMAL: {model.Status}")
        projected, witness = np.asarray(hp.X), np.asarray(u.X)
        slack = projected-H0@witness-eps
        dual = -np.asarray(constraints.Pi)
        active = np.flatnonzero(slack <= active_tol)
        weak = active[dual[active] <= active_tol]
        Jac = np.eye(m)
        if len(active):
            U, singular, _ = np.linalg.svd(H0[active], full_matrices=False)
            rank = int(np.count_nonzero(singular > max(H0[active].shape)*np.finfo(float).eps*singular[0])) if len(singular) else 0
            Jac[np.ix_(active, active)] = U[:, :rank] @ U[:, :rank].T
        else:
            rank = 0
        metadata = {"status": "optimal", "epsilon": float(eps), "interior_witness": witness,
                    "minimum_margin": float(np.min(projected-H0@witness)),
                    "squared_projection_distance": float(np.sum((projected-h)**2)),
                    "active_rows": active, "weakly_active_rows": weak, "active_normal_rank": rank,
                    "classical_regularity_conditions_verified": len(weak) == 0,
                    "regularity_scope": "unique projected RHS after eliminating nonunique witness directions",
                    "witness_nullity": int(n-rank),
                    "jacobian_method": "active-normal-column-space orthogonal projector",
                    "maximum_primal_violation": float(max(0., -slack.min())),
                    "stationarity_h_inf": float(np.max(np.abs(2*(projected-h)-dual))),
                    "stationarity_u_inf": float(np.max(np.abs(H0.T@dual))),
                    "solver_seconds": float(model.Runtime), "elapsed_seconds": time.perf_counter()-started}
        if metadata["maximum_primal_violation"] > 1e-7:
            raise RuntimeError("Projection failed its strict-interior witness audit")
        if len(weak):
            raise KKTRegularityError(
                "Equation (14) has weakly active projection constraints; a unique classical RHS Jacobian is not certified",
                {**metadata, "weakly_active_rows": weak.tolist(), "active_rows": active.tolist()})
        return projected.copy(), Jac, metadata
    finally:
        model.dispose()
        if owned:
            environment.dispose()


def aggregate_affine_maps(H0, h, solutions, pmin=None, pmax=None):
    """Sum full affine maps; never invert a rank-deficient aggregate map."""
    H0, h = _array(H0), _array(h)
    n = H0.shape[1]
    if not solutions:
        raise ValueError("At least one certified local affine map is required")
    if any(not solution.get("optimal") or not solution.get("certificate", {}).get("passed") for solution in solutions):
        raise ValueError("Only optimal locally certified affine maps may be aggregated")
    gamma = sum((_array(solution["gamma"], (n,)) for solution in solutions), np.zeros(n))
    Gamma = sum((_array(solution["Gamma"], (n, n)) for solution in solutions), np.zeros((n, n)))
    rank = int(np.linalg.matrix_rank(Gamma))
    condition = float(np.linalg.cond(Gamma))
    result = {"gamma_sum": gamma, "Gamma_sum": Gamma, "rank": rank,
              "full_rank": rank == n, "base_H": H0, "base_h": h,
              "representation": "affine_image", "condition_number": condition if np.isfinite(condition) else None}
    if rank == n:
        inverse_map = np.linalg.solve(Gamma, np.eye(n))
        physical_A = H0 @ inverse_map
        result.update(physical_A=physical_A, physical_b=h+physical_A@gamma)
        if pmin is not None or pmax is not None:
            lower, upper = _array(pmin, (n,)), _array(pmax, (n,))
            scale = upper-lower
            if np.any(scale <= 0):
                raise ValueError("Physical aggregate normalization must have positive widths")
            result.update(A=physical_A*scale[None, :], b=h+physical_A@(gamma-lower), representation="h_polytope")
        else:
            result.update(A=physical_A, b=result["physical_b"], representation="h_polytope")
    return result
