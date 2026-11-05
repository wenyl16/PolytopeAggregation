"""Paper-faithful chance labels and appendix triplets for Taheri et al. (2022).

The device portfolio is the existing project microgrid, rather than the paper's
200-device example. Every label draws K NEW independent externality scenarios.
Only RHS values change between scenarios; the unrelaxed integer model is reused.
Power points are returned in the project's normalized coordinates. The L1
objective is weighted by pmax-pmin, hence is EXACTLY the raw-kW L1 objective.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import gurobipy as gp
import numpy as np
import pyomo.environ as pyo
from pyomo.repn.standard_repn import generate_standard_repn

from Simulator.runners.train_microgrid import build_microgrid


def _json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    # Windows/OneDrive readers can transiently hold the target without delete
    # sharing. A bounded retry keeps an unrelated reader from aborting the run.
    for attempt in range(20):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05)


class ScenarioOracle:
    """Original Eq.6 L1 MILPs and Eq.7 empirical chance labels.

    Labels: -1=certified feasible in at least ceil(K*(1-epsilon)) samples;
    +1=certified infeasible in more than floor(K*epsilon) samples;
    0=unknown (a timeout or deadline never becomes a negative training label).
    """

    def __init__(self, seed=42, K=25, epsilon=.04, noise_scale=.05,
                 conservativeness=.02, device_scale=1, solver_threads=1,
                 tolerance=1e-5, sampling_seed=None):
        if K < 1 or not 0 <= epsilon < 1:
            raise ValueError("K must be positive; epsilon must be in [0,1)")
        self.K, self.epsilon = int(K), float(epsilon)
        self.rng = np.random.default_rng(seed if sampling_seed is None else sampling_seed)
        self.noise_scale, self.tolerance = float(noise_scale), float(tolerance)
        self.mg = build_microgrid(seed, device_scale)
        self.mg.build_mg_constraints(conservativeness=conservativeness)
        self.source = self.mg.model
        self.T = self.mg.T
        self.pmin, self.pmax = self.mg.pmin.copy(), self.mg.pmax.copy()
        self.width = self.pmax-self.pmin
        if np.any(self.width <= 0):
            raise ValueError("Normalization widths must be positive")
        self.base_pv = np.asarray([d['output_profile'] for d in self.mg.PV_list])
        self.base_cl = np.asarray([d['p_max_profile'] for d in self.mg.CL_list])
        self.base_temp = np.asarray(self.mg.temp_amb)
        # A sampling window, NOT a probability guarantee for unbounded Gaussian
        # noise. The extra 20% accommodates scenario-varying physical bounds.
        self.box_lower = np.full(self.T, -.2)
        self.box_upper = np.full(self.T, 1.2)
        # Split ranges before translation, so mutable RHS updates introduce no
        # hidden native range slacks whose bounds could otherwise become stale.
        self.source._taheri_sides = pyo.ConstraintList()
        for row in list(self.source.component_data_objects(pyo.Constraint, active=True)):
            if row.has_lb() and row.has_ub() and not row.equality:
                self.source._taheri_sides.add((row.lower, row.body, None))
                self.source._taheri_sides.add((None, row.body, row.upper))
                row.deactivate()
        for obj in self.source.component_data_objects(pyo.Objective, active=True):
            obj.deactivate()
        self.bridge = pyo.SolverFactory('gurobi_persistent')
        self.bridge.set_instance(self.source)
        self.model = self.bridge._solver_model
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('Threads', int(solver_threads))
        self.model.setParam('Seed', 0)
        self.model.setParam('FeasibilityTol', 1e-8)
        self.model.setParam('IntFeasTol', 1e-9)
        self.model.setParam('MIPGap', 0.)
        self.model.setParam('MIPGapAbs', 1e-8)
        self.model.update()
        varmap = self.bridge._pyomo_var_to_solver_var_map
        self.x = [varmap[self.source.var_proj[t]] for t in range(self.T)]
        self.rows, self.rhs_expressions = [], []
        conmap = self.bridge._pyomo_con_to_solver_con_map
        for row in self.source.component_data_objects(pyo.Constraint, active=True):
            repn = generate_standard_repn(row.body, compute_values=False)
            if not repn.is_linear():
                raise ValueError(f"Nonlinear original physical row: {row.name}")
            # Present microgrid mutable parameters occur only in bounds/constants;
            # verify coefficients remain fixed before relying on RHS-only reuse.
            if any(not pyo.is_constant(c) for c in repn.linear_coefs):
                raise ValueError(f"Mutable matrix coefficient requires explicit update: {row.name}")
            bound = row.upper if row.has_ub() else row.lower
            self.rows.append(conmap[row])
            constant = 0. if repn.constant is None else repn.constant
            self.rhs_expressions.append(bound - constant)
        self.original_variables = self.model.getVars()
        self.physical_binary_count = sum(v.VType == gp.GRB.BINARY for v in self.original_variables)
        self.r = self.model.addVars(self.T, lb=0., name='_taheri_l1')
        self.pos = [self.model.addConstr(self.x[t]-self.r[t] <= 0.) for t in range(self.T)]
        self.neg = [self.model.addConstr(-self.x[t]-self.r[t] <= 0.) for t in range(self.T)]
        self.model.setObjective(gp.quicksum(self.width[t]*self.r[t] for t in range(self.T)), gp.GRB.MINIMIZE)
        self.model.update()
        self.solve_count = 0
        self.label_count = 0
        self.scenario_hashes = []
        self.trace_path = None
        self.metadata = dict(case_seed=seed, sampling_seed=seed if sampling_seed is None else sampling_seed,
            horizon=self.T, K=self.K, epsilon=self.epsilon, noise_scale=self.noise_scale,
            conservativeness=conservativeness, device_scale=device_scale,
            device_counts={k:len(getattr(self.mg,k+'_list')) for k in ['MT','PV','TL','CL','ESS','HP']},
            physical_binary_variables=self.physical_binary_count,
            physical_variables=len(self.original_variables), physical_constraints=len(self.rows),
            coordinates='project normalized; raw power = x*width+pmin',
            objective='raw-kW L1 = sum(width[t]*abs(x[t]-target[t]))',
            scenario_policy='fresh independent K Gaussian scenarios for every candidate label',
            clipping='none; reject invalid negative PV/CL parameter draw and redraw',
            sampling_window=[-.2,1.2], sampling_window_is_global_bound=False,
            adaptations=['existing T24 seven-device microgrid, not paper 200-device T8',
                'Gaussian 5% existing project externality distribution',
                'finite heuristic sampling window because Gaussian support is unbounded'])

    def _scenario(self):
        # Match existing base*(1+sigma*z), while respecting the original
        # NonNegativeReals parameter domain. Redraw rather than silently clip.
        while True:
            pv = self.base_pv*(1+self.noise_scale*self.rng.standard_normal(self.base_pv.shape))
            cl = self.base_cl*(1+self.noise_scale*self.rng.standard_normal(self.base_cl.shape))
            if np.all(pv >= 0) and np.all(cl >= 0):
                break
        temp = self.base_temp*(1+self.noise_scale*self.rng.standard_normal(self.base_temp.shape))
        for i in range(len(pv)):
            for t in range(self.T): self.source.p_PV_rated[i,t] = float(pv[i,t])
        for i in range(len(cl)):
            for t in range(self.T): self.source.p_CL_rated[i,t] = float(cl[i,t])
        for t in range(self.T): self.source.temp_amb[t] = float(temp[t])
        rhs = [float(pyo.value(expr)) for expr in self.rhs_expressions]
        self.model.setAttr('RHS', self.rows, rhs)
        digest = hashlib.sha256(pv.tobytes()+cl.tobytes()+temp.tobytes()).hexdigest()
        return digest

    def solve(self, point, deadline_monotonic, per_solve_seconds=2.):
        remaining = deadline_monotonic-time.monotonic()
        if remaining <= .02:
            return dict(state='unknown', reason='deadline', objective=None, lower_bound=None, point=None)
        p = np.asarray(point,dtype=float)
        if p.shape != (self.T,) or not np.isfinite(p).all():
            raise ValueError("Expected finite T-vector")
        digest = self._scenario()
        self.model.setAttr('RHS', self.pos, p.tolist())
        self.model.setAttr('RHS', self.neg, (-p).tolist())
        self.model.setParam('TimeLimit', max(.01,min(float(per_solve_seconds),remaining-.01)))
        self.model.optimize()
        self.solve_count += 1
        status, count = int(self.model.Status), int(self.model.SolCount)
        objective = float(self.model.ObjVal) if count else None
        bound = float(self.model.ObjBound) if math.isfinite(self.model.ObjBound) else None
        projection = np.asarray([x.X for x in self.x]) if count else None
        # Successful MILP incumbent itself is a feasibility certificate; a
        # positive incumbent alone is never an infeasibility certificate.
        if count and objective <= self.tolerance:
            state = 'feasible'
        elif status == gp.GRB.INFEASIBLE or (bound is not None and bound > self.tolerance):
            state = 'infeasible'
        else:
            state = 'unknown'
        projection_optimal = count > 0 and (status == gp.GRB.OPTIMAL or
            (bound is not None and objective-bound <= self.tolerance))
        record=dict(solve=self.solve_count,status=status,state=state,objective=objective,
            lower_bound=bound,runtime=float(self.model.Runtime),scenario_sha256=digest,
            mip_gap=float(self.model.MIPGap) if count and math.isfinite(self.model.MIPGap) else None,
            projection_is_optimal=bool(projection_optimal),
            original_integer_constraints=True)
        if self.trace_path:
            with self.trace_path.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,allow_nan=False)+'\n')
        return {**record,'point':projection}

    def label(self, point, deadline_monotonic, per_solve_seconds=2., all_scenarios=True):
        records, projections = [], []
        successes=failures=unknown=0
        required = int(math.ceil(self.K*(1-self.epsilon)-1e-12))
        for k in range(self.K):
            result=self.solve(point,deadline_monotonic,per_solve_seconds)
            records.append({k:v for k,v in result.items() if k!='point'})
            successes += result['state']=='feasible'
            failures += result['state']=='infeasible'
            unknown += result['state']=='unknown'
            if result.get('point') is not None and result.get('projection_is_optimal'):
                projections.append((result['objective'],result['point']))
            if time.monotonic() >= deadline_monotonic-.02:
                break
            if not all_scenarios and (successes>=required or failures>self.K-required):
                break
        observed=len(records)
        missing=self.K-observed
        if successes>=required: label=-1
        elif failures>self.K-required: label=1
        else: label=0
        self.label_count += 1
        return dict(label=label,successes=successes,failures=failures,
            unknown=unknown+missing,scenarios_observed=observed,required_successes=required,
            projections=projections,records=records)

    def generate(self, output_dir, deadline_monotonic, target_points=128,
                 per_solve_seconds=2., kappa=.2, progress_callback=None):
        output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
        self.trace_path=output/'labeling_solves.jsonl'
        X=[]; y=[]; kinds=[]; details=[]
        started=time.monotonic()
        def save(status='running'):
            array=np.asarray(X,dtype=float).reshape((-1,self.T))
            np.savez_compressed(output/'dataset.npz',X=array,y=np.asarray(y,dtype=int),
                kinds=np.asarray(kinds),pmin=self.pmin,pmax=self.pmax)
            state=dict(status=status,elapsed_seconds=time.monotonic()-started,
                labeled_points=len(y),feasible=sum(v==-1 for v in y),infeasible=sum(v==1 for v in y),
                label_attempts=self.label_count,solve_count=self.solve_count,
                metadata=self.metadata,candidates=details)
            _json(output/'labeling_progress.json',state)
            if progress_callback: progress_callback(state)
        def classify(p,kind):
            result=self.label(p,deadline_monotonic,per_solve_seconds)
            details.append(dict(kind=kind,label=result['label'],successes=result['successes'],
                failures=result['failures'],unknown=result['unknown'],scenarios_observed=result['scenarios_observed']))
            if result['label']:
                X.append(np.asarray(p).copy()); y.append(result['label']); kinds.append(kind)
            save()
            return result
        while len(y)<target_points and time.monotonic()<deadline_monotonic-.05:
            p1=self.rng.uniform(self.box_lower,self.box_upper)
            r1=classify(p1,'D1_uniform')
            if r1['label']==-1: continue
            if not r1['projections']: continue
            # Appendix's g(phat;omega) is zero by definition; intended quantity
            # is the largest original g(p1;omega), i.e. farthest projection.
            p2=max(r1['projections'],key=lambda pair:pair[0])[1]
            r2=classify(p2,'D2_farthest_projection')
            # The appendix recursively replaces a chance-infeasible p2 by its
            # farthest projection. Its reported 1--2 iterations are empirical,
            # not a mathematical stopping rule (T24 often needs more retries).
            for retry in range(25):
                if r2['label']==-1 or not r2['projections'] or time.monotonic()>=deadline_monotonic-.05: break
                p2=max(r2['projections'],key=lambda pair:pair[0])[1]
                r2=classify(p2,'D2_projection_retry')
            if r2['label']!=-1: continue
            p3=kappa*p1+(1-kappa)*p2
            for retry in range(10):
                if time.monotonic()>=deadline_monotonic-.05 or len(y)>=target_points: break
                r3=classify(p3,'D3_boundary_interpolation')
                if r3['label']!=-1: break
                p3=kappa*p1+(1-kappa)*p3
            # Appendix adds interior convex combinations to balance the classes.
            feasible=[X[i] for i,value in enumerate(y) if value==-1]
            while sum(v==-1 for v in y)<sum(v==1 for v in y) and len(feasible)>=2:
                if time.monotonic()>=deadline_monotonic-.05 or len(y)>=target_points: break
                selected=self.rng.choice(len(feasible),size=min(4,len(feasible)),replace=False)
                weights=self.rng.dirichlet(np.ones(len(selected)))
                interior=np.sum([weight*feasible[index] for weight,index in zip(weights,selected)],axis=0)
                ri=classify(interior,'interior_convex_combination')
                if ri['label']!=-1: break
                feasible.append(interior)
        save('deadline' if time.monotonic()>=deadline_monotonic-.05 else 'target_reached')
        return np.asarray(X,dtype=float).reshape((-1,self.T)), np.asarray(y,dtype=int), self.metadata

    def close(self):
        self.model.dispose()

