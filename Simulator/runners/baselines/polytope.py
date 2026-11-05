"""Box, cutting-plane and vertex-enumeration aggregation baselines.

All coordinates match ``Aggregator.case_aggregator``. Builders return the
polytope and elapsed algorithm time, and attach diagnostic records to ``agg``.
"""

import numpy as np
import pyomo.environ as pyo


def build_cube(agg):
    model = pyo.ConcreteModel()
    model.T = pyo.Set(initialize=range(agg.T))
    model.P_max = pyo.Var(model.T, within=pyo.Reals)
    model.P_min = pyo.Var(model.T, within=pyo.Reals)
    pmax = np.zeros(agg.T)
    pmin = np.zeros(agg.T)

    model.TCLs = pyo.Set(initialize=range(len(agg.TCL_list)))
    model.temp_TCL_u = pyo.Var(model.TCLs, model.T, within=pyo.Reals)
    model.p_TCL_u = pyo.Var(model.TCLs, model.T, within=pyo.NonNegativeReals)
    model.temp_TCL_l = pyo.Var(model.TCLs, model.T, within=pyo.Reals)
    model.p_TCL_l = pyo.Var(model.TCLs, model.T, within=pyo.NonNegativeReals)

    # 定义EV集合
    model.EVs = pyo.Set(initialize=range(len(agg.EV_list)))
    model.e_EV_u = pyo.Var(model.EVs, model.T, within=pyo.NonNegativeReals)  # 每个EV的功率变量
    model.p_EV_u = pyo.Var(model.EVs, model.T, within=pyo.NonNegativeReals)  # 每个EV的功率变量
    model.e_EV_l = pyo.Var(model.EVs, model.T, within=pyo.NonNegativeReals)  # 每个EV的功率变量
    model.p_EV_l = pyo.Var(model.EVs, model.T, within=pyo.NonNegativeReals)  # 每个EV的功率变量

    # 定义PV集合
    model.PVs = pyo.Set(initialize=range(len(agg.PV_list)))
    model.p_PV_u = pyo.Var(model.PVs, model.T, within=pyo.NonNegativeReals)  # PV出力上限
    model.p_PV_l = pyo.Var(model.PVs, model.T, within=pyo.NonNegativeReals)  # PV出力下限

    # 定义CL集合
    model.CLs = pyo.Set(initialize=range(len(agg.CL_list)))
    model.p_CL_u = pyo.Var(model.CLs, model.T, within=pyo.NonNegativeReals)  # CL削减量上限
    model.p_CL_l = pyo.Var(model.CLs, model.T, within=pyo.NonNegativeReals)  # CL削减量下限

    for i in model.TCLs:
        tcl_data = agg.TCL_list[i]
        pmin -= tcl_data['Qmax'] / tcl_data['COP']
        def power_upper_limit_rule(m, t):
            return pyo.inequality(0, m.p_TCL_u[i, t], tcl_data['Qmax'] / tcl_data['COP'])

        model.add_component(f"tcl_power_upper_limit{i}", pyo.Constraint(model.T, rule=power_upper_limit_rule))
        def power_lower_limit_rule(m, t):
            return pyo.inequality(0, m.p_TCL_l[i, t], tcl_data['Qmax'] / tcl_data['COP'])

        model.add_component(f"tcl_power_lower_limit{i}", pyo.Constraint(model.T, rule=power_lower_limit_rule))

        def tcl_lower_upper_rule(m, t):
            return m.p_TCL_l[i, t]<=m.p_TCL_u[i, t]
        model.add_component(f"tcl_lower_upper{i}", pyo.Constraint(model.T, rule=tcl_lower_upper_rule))

        alpha = np.exp(-agg.deltaT * tcl_data['H'] / tcl_data['C'])

        # 动态温度更新公式
        def temperature_u_dynamics_rule(m, t):
            if t > 0:
                return m.temp_TCL_u[i, t] == (alpha * m.temp_TCL_u[i, t - 1] + (1 - alpha) * (
                        agg.theta_amb[t] + tcl_data['COP'] / tcl_data['H'] * model.p_TCL_u[i, t]))
            return pyo.Constraint.Skip

        model.add_component(f"temperature_u_dynamics{i}", pyo.Constraint(model.T, rule=temperature_u_dynamics_rule))
        # 动态温度更新公式
        def temperature_l_dynamics_rule(m, t):
            if t > 0:
                return m.temp_TCL_l[i, t] == (alpha * m.temp_TCL_l[i, t - 1] + (1 - alpha) * (
                        agg.theta_amb[t] + tcl_data['COP'] / tcl_data['H'] * model.p_TCL_l[i, t]))
            return pyo.Constraint.Skip

        model.add_component(f"temperature_l_dynamics{i}", pyo.Constraint(model.T, rule=temperature_l_dynamics_rule))

        # 温度限制约束：确保温度在上下限之间
        def temperature_u_limit_rule(m, t):
            return pyo.inequality(tcl_data['temp_min'], m.temp_TCL_l[i, t], tcl_data['temp_max'])

        model.add_component(f"temperature_u_limit{i}", pyo.Constraint(model.T, rule=temperature_u_limit_rule))

        # 温度限制约束：确保温度在上下限之间
        def temperature_l_limit_rule(m, t):
            return pyo.inequality(tcl_data['temp_min'], m.temp_TCL_l[i, t], tcl_data['temp_max'])

        model.add_component(f"temperature_l_limit{i}", pyo.Constraint(model.T, rule=temperature_l_limit_rule))

        def initial_temperature_u_condition_rule(m):
            return m.temp_TCL_u[i, 0] == tcl_data['temp_set']

        model.add_component(f"initial_temperature_u_condition{i}",
                            pyo.Constraint(rule=initial_temperature_u_condition_rule))
        def initial_temperature_l_condition_rule(m):
            return m.temp_TCL_l[i, 0] == tcl_data['temp_set']

        model.add_component(f"initial_temperature_l_condition{i}",
                            pyo.Constraint(rule=initial_temperature_l_condition_rule))
    for i in model.EVs:
        ev_data = agg.EV_list[i]  # 默认使用第一个EV的数据
        # 计算离散化的到达和离开时间索引
        ta_idx = int(np.floor(ev_data['ta'] / agg.deltaT))
        td_idx = int(np.ceil(ev_data['td'] / agg.deltaT))
        pmin[ta_idx:td_idx] -= ev_data['P_chg']

        # 约束1: 功率上下限约束
        def power_u_limit_rule(m, t):
            if ta_idx <= t < td_idx:
                return (0, m.p_EV_u[i, t], ev_data['P_chg'])
            else:
                return m.p_EV_u[i, t] == 0
        model.add_component(f"ev_power_u_limit{i}", pyo.Constraint(model.T, rule=power_u_limit_rule))
        # 约束1: 功率上下限约束
        def power_l_limit_rule(m, t):
            if ta_idx <= t < td_idx:
                return (0, m.p_EV_l[i, t], ev_data['P_chg'])
            else:
                return m.p_EV_l[i, t] == 0
        model.add_component(f"ev_power_l_limit{i}", pyo.Constraint(model.T, rule=power_l_limit_rule))

        def ev_lower_upper_rule(m, t):
            return m.p_EV_l[i, t]<=m.p_EV_u[i, t]
        model.add_component(f"ev_lower_upper{i}", pyo.Constraint(model.T, rule=ev_lower_upper_rule))

        # 约束2: 初始能量状态
        def initial_energy_u_rule(m):
            return m.e_EV_u[i, ta_idx] == ev_data['SOCa'] * ev_data['B']

        model.add_component(f"initial_energy_u{i}", pyo.Constraint(rule=initial_energy_u_rule))
        # 约束2: 初始能量状态
        def initial_energy_l_rule(m):
            return m.e_EV_l[i, ta_idx] == ev_data['SOCa'] * ev_data['B']

        model.add_component(f"initial_energy_l{i}", pyo.Constraint(rule=initial_energy_l_rule))

        # 约束3: 能量动态更新
        def energy_u_dynamics_rule(m, t):
            if ta_idx <= t < td_idx:
                return m.e_EV_u[i, t + 1] == m.e_EV_u[i, t] + m.p_EV_u[i, t] * ev_data['eta_chg'] * agg.deltaT
            else:
                return pyo.Constraint.Skip
        model.add_component(f"energy_u_dynamics{i}", pyo.Constraint(model.T, rule=energy_u_dynamics_rule))

        # 约束3: 能量动态更新
        def energy_l_dynamics_rule(m, t):
            if ta_idx <= t < td_idx:
                return m.e_EV_l[i, t + 1] == m.e_EV_l[i, t] + m.p_EV_l[i, t] * ev_data['eta_chg'] * agg.deltaT
            else:
                return pyo.Constraint.Skip

        model.add_component(f"energy_l_dynamics{i}", pyo.Constraint(model.T, rule=energy_l_dynamics_rule))

        # 约束4: 能量状态上下限
        def energy_u_limit_rule(m, t):
            if ta_idx <= t <= td_idx:
                return (ev_data['SOCa'] * ev_data['B'], m.e_EV_u[i, t], ev_data['SOCmax'] * ev_data['B'])
            else:
                return pyo.Constraint.Skip

        model.add_component(f"energy_u_limit{i}", pyo.Constraint(model.T, rule=energy_u_limit_rule))
        # 约束4: 能量状态上下限
        def energy_l_limit_rule(m, t):
            if ta_idx <= t <= td_idx:
                return (ev_data['SOCa'] * ev_data['B'], m.e_EV_l[i, t], ev_data['SOCmax'] * ev_data['B'])
            else:
                return pyo.Constraint.Skip

        model.add_component(f"energy_l_limit{i}", pyo.Constraint(model.T, rule=energy_l_limit_rule))

        # 约束5: 离开时最小能量要求
        def departure_energy_u_rule(m):
            return m.e_EV_u[i, td_idx] >= ev_data['SOCd'] * ev_data['B']

        model.add_component(f"departure_energy_u{i}", pyo.Constraint(rule=departure_energy_u_rule))
        # 约束5: 离开时最小能量要求
        def departure_energy_l_rule(m):
            return m.e_EV_l[i, td_idx] >= ev_data['SOCd'] * ev_data['B']

        model.add_component(f"departure_energy_l{i}", pyo.Constraint(rule=departure_energy_l_rule))

    # PV约束 - 光伏
    for i in model.PVs:
        pv_data = agg.PV_list[i]
        pmax += pv_data['output_profile']  # PV增加发电能力上限

        # PV出力上限限制
        def pv_power_u_limit_rule(m, t):
            return pyo.inequality(0, m.p_PV_u[i, t], pv_data['output_profile'][t])

        model.add_component(f"pv_power_u_limit_{i}", pyo.Constraint(model.T, rule=pv_power_u_limit_rule))

        # PV出力下限限制
        def pv_power_l_limit_rule(m, t):
            return pyo.inequality(0, m.p_PV_l[i, t], pv_data['output_profile'][t])

        model.add_component(f"pv_power_l_limit_{i}", pyo.Constraint(model.T, rule=pv_power_l_limit_rule))

        # 确保下限不超过上限
        def pv_lower_upper_rule(m, t):
            return m.p_PV_l[i, t] <= m.p_PV_u[i, t]

        model.add_component(f"pv_lower_upper_{i}", pyo.Constraint(model.T, rule=pv_lower_upper_rule))

    # CL约束 - 可中断负荷
    for i in model.CLs:
        cl_data = agg.CL_list[i]
        pmin -= (cl_data['p_max_profile'])  # CL增加负荷能力（负值）
        pmax -= (cl_data['p_max_profile']) * (1 - cl_data['curt_ratio'])  # CL增加负荷能力（负值）

        # CL削减量上限限制
        def cl_curtailment_u_limit_rule(m, t):
            return pyo.inequality((1 - cl_data['curt_ratio']) * cl_data['p_max_profile'][t],
                                 m.p_CL_u[i, t], cl_data['p_max_profile'][t])

        model.add_component(f"cl_curtailment_u_limit_{i}", pyo.Constraint(model.T, rule=cl_curtailment_u_limit_rule))

        # CL削减量下限限制
        def cl_curtailment_l_limit_rule(m, t):
            return pyo.inequality((1 - cl_data['curt_ratio']) * cl_data['p_max_profile'][t],
                                 m.p_CL_l[i, t], cl_data['p_max_profile'][t])

        model.add_component(f"cl_curtailment_l_limit_{i}", pyo.Constraint(model.T, rule=cl_curtailment_l_limit_rule))

        # 确保下限不超过上限
        def cl_lower_upper_rule(m, t):
            return m.p_CL_l[i, t] <= m.p_CL_u[i, t]

        model.add_component(f"cl_lower_upper_{i}", pyo.Constraint(model.T, rule=cl_lower_upper_rule))

    def agg_power_u_rule(m, t):
        generation = (sum(m.p_PV_u[i, t] for i in m.PVs))
        consumption = (sum(m.p_EV_u[i, t] for i in m.EVs) +
                       sum(m.p_TCL_u[i, t] for i in m.TCLs) +
                      sum(m.p_CL_u[i, t] for i in m.CLs))
        return m.P_max[t] * (pmax[t] - pmin[t]) + pmin[t] == generation - consumption

    model.agg_power_u_constraint = pyo.Constraint(model.T, rule=agg_power_u_rule)

    def agg_power_l_rule(m, t):
        generation = (sum(m.p_PV_l[i, t] for i in m.PVs))
        consumption = (sum(m.p_EV_l[i, t] for i in m.EVs) +
                     sum(m.p_TCL_l[i, t] for i in m.TCLs) +
                      sum(m.p_CL_l[i, t] for i in m.CLs))
        return m.P_min[t] * (pmax[t] - pmin[t]) + pmin[t] == generation - consumption

    model.agg_power_l_constraint = pyo.Constraint(model.T, rule=agg_power_l_rule)

    model.obj = pyo.Objective(expr=-sum(model.P_max[t]-model.P_min[t] for t in model.T))
    solver = pyo.SolverFactory('gurobi')
    import time
    start_t = time.time()
    solver.solve(model)
    end_t = time.time()
    comp_time = end_t-start_t
    P_max = np.array([pyo.value(model.P_max[i]) for i in model.P_max])
    P_min = np.array([pyo.value(model.P_min[i]) for i in model.P_min])

    A_hat = np.vstack([np.eye(agg.T), -np.eye(agg.T)])
    b_hat = np.hstack([P_max, -P_min])
    return (A_hat, b_hat), comp_time


def build_cutting_plane(agg, timelimit, modified=False):
    """两种割平面策略共用的建模、数值校验及限时实现。"""
    import time
    import gurobipy as gp
    from scipy.sparse import csr_matrix
    from pyomo.repn.standard_repn import generate_standard_repn

    timelimit = float(timelimit)
    if not np.isfinite(timelimit) or timelimit <= 0:
        raise ValueError('timelimit 必须是正的有限秒数。')

    total_started = time.perf_counter()
    tolerance = 1e-7
    solver_tolerance = 1e-9
    info = {
        'status': 'initializing', 'iterations': 0, 'n_cuts': 0,
        'tolerance': tolerance, 'worst_violation_bound': np.inf,
        'inner_point': None, 'solve_counts': {'lp': 0, 'milp': 0},
        'history': [], 'oracle_history': [], 'timelimit': timelimit,
        'separation_strategy': ('first_positive_incumbent'
                                if modified else 'global_worst'),
        'separation_retries': 0, 'retry_history': [],
    }
    info_attribute = ('cutting_plane_modified_info' if modified
                      else 'cutting_plane_info')
    setattr(agg, info_attribute, info)
    model = agg.case_aggregator(build_only=True)
    projected_variables = [model.var_proj[t] for t in model.T]
    projected_ids = {id(v) for v in projected_variables}
    variables = [v for v in model.component_data_objects(pyo.Var, active=True)
                 if not v.fixed or id(v) in projected_ids]
    if any((v.is_binary() or v.is_integer()) and not v.fixed for v in variables):
        raise ValueError('割平面法依赖 LP 强对偶，不能直接处理未固定的整数设备变量。')
    variable_index = {id(v): j for j, v in enumerate(variables)}
    projected_indices = np.array([variable_index[id(v)]
                                  for v in projected_variables], dtype=int)
    projected_set = set(projected_indices)
    internal_indices = [j for j in range(len(variables)) if j not in projected_set]

    # 将所有约束及变量界写成 D x <= d，再分块为 B y + C P <= d。
    # 等式和范围约束拆成两行；固定变量由 standard_repn 代入常数。
    row_indices, column_indices, coefficients, rhs = [], [], [], []

    def append_row(terms, bound):
        terms = [(j, float(a)) for j, a in terms if a != 0]
        bound = float(bound)
        if not np.isfinite(bound) or any(not np.isfinite(a) for _, a in terms):
            raise ValueError('原模型包含非有限的约束系数或边界。')
        if not terms:
            if bound < -solver_tolerance:
                raise ValueError('原模型含不可行的常数约束。')
            return
        scale = np.linalg.norm([a for _, a in terms])
        row = len(rhs)
        for j, a in terms:
            row_indices.append(row)
            column_indices.append(j)
            coefficients.append(a / scale)
        rhs.append(bound / scale)

    for constraint in model.component_data_objects(pyo.Constraint, active=True):
        repn = generate_standard_repn(constraint.body, compute_values=True)
        if not repn.is_linear():
            raise ValueError(f'割平面法仅支持线性约束：{constraint.name}')
        terms = [(variable_index[id(v)], pyo.value(a))
                 for v, a in zip(repn.linear_vars, repn.linear_coefs)]
        constant = pyo.value(repn.constant)
        if constraint.has_ub():
            append_row(terms, pyo.value(constraint.upper) - constant)
        if constraint.has_lb():
            append_row([(j, -a) for j, a in terms],
                       constant - pyo.value(constraint.lower))
    for j, variable in enumerate(variables):
        if variable.fixed:
            # repn 已将固定坐标代入约束常数，但它仍需要作为投影坐标
            # 返回；显式添加其固定值，避免缺失索引或无界的虚拟坐标。
            append_row([(j, 1.0)], pyo.value(variable))
            append_row([(j, -1.0)], -pyo.value(variable))
        if variable.has_ub():
            append_row([(j, 1.0)], pyo.value(variable.ub))
        if variable.has_lb():
            append_row([(j, -1.0)], -pyo.value(variable.lb))

    D = csr_matrix((coefficients, (row_indices, column_indices)),
                   shape=(len(rhs), len(variables)))
    d = np.asarray(rhs)
    B, C = D[:, internal_indices], D[:, projected_indices]
    dim = len(projected_indices)

    class CalculationTimeLimit(Exception):
        pass

    deadline = None
    force_global_oracle = False

    def retry_separation(reason):
        """同一外包最多重试一次全局搜索，避免反复接受数值无效候选。"""
        nonlocal force_global_oracle
        if (not modified or force_global_oracle
                or time.perf_counter() >= deadline):
            return False
        force_global_oracle = True
        info['separation_retries'] += 1
        info['retry_history'].append({
            'iteration': info['iterations'], 'reason': reason,
            'elapsed': time.perf_counter() - cutting_started,
        })
        return True

    def configure(solver_model):
        solver_model.Params.OutputFlag = 0
        solver_model.Params.Threads = 1
        solver_model.Params.FeasibilityTol = solver_tolerance
        solver_model.Params.OptimalityTol = solver_tolerance
        solver_model.Params.DualReductions = 0

    def optimize(solver_model, kind='lp', limited=True):
        if limited:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise CalculationTimeLimit
            solver_model.Params.TimeLimit = remaining
        info['solve_counts'][kind] += 1
        solver_model.optimize()

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model('aggregation_support', env=env) as joint, \
                gp.Model('aggregation_slack_dual', env=env) as dual:
            configure(joint)
            configure(dual)
            x = joint.addMVar(len(variables), lb=-gp.GRB.INFINITY, name='x')
            joint.addMConstr(D, x, gp.GRB.LESS_EQUAL, d)
            P = x[projected_indices]
            # q(P)=min 1^T s, B y+C P<=d+s, s>=0.
            # 对偶：max h^T(d-C P), B^T h=0, -1<=h<=0。
            h = dual.addMVar(len(d), lb=-1.0, ub=0.0, name='h')
            if internal_indices:
                dual.addConstr(B.T @ h == np.zeros(len(internal_indices)))

            def support(direction, limited=True):
                joint.setObjective(np.asarray(direction) @ P, gp.GRB.MAXIMIZE)
                optimize(joint, limited=limited)
                if joint.Status == gp.GRB.TIME_LIMIT:
                    raise CalculationTimeLimit
                if joint.Status == gp.GRB.INFEASIBLE:
                    raise ValueError('原始设备模型不可行。')
                if joint.Status == gp.GRB.UNBOUNDED:
                    raise ValueError('聚合投影无界；请检查功率归一化范围和设备约束。')
                if joint.Status != gp.GRB.OPTIMAL:
                    raise RuntimeError(f'支持 LP 求解失败，Gurobi 状态 {joint.Status}。')
                return max(joint.ObjVal, joint.ObjBound), P.X.copy()

            def violation(point):
                dual.setObjective((d - C @ point) @ h, gp.GRB.MAXIMIZE)
                optimize(dual)
                if dual.Status == gp.GRB.TIME_LIMIT:
                    raise CalculationTimeLimit
                if dual.Status != gp.GRB.OPTIMAL:
                    raise RuntimeError(f'可行性 LP 求解失败，Gurobi 状态 {dual.Status}。')
                return max(0.0, dual.ObjVal), h.X.copy()

            # 用原 LP 的真实支持值建外包，不能默认 var_proj 属于 [0,1]^T。
            upper, lower, feasible_points = [], [], []
            for axis in np.eye(dim):
                maximum, point = support(axis, limited=False)
                upper.append(maximum + 1e-8 * max(1.0, abs(maximum)))
                feasible_points.append(point)
                negative_minimum, point = support(-axis, limited=False)
                lower.append(-negative_minimum - 1e-8 * max(1.0, abs(negative_minimum)))
                feasible_points.append(point)
            A_hat = np.vstack([np.eye(dim), -np.eye(dim)])
            b_hat = np.r_[upper, -np.asarray(lower)]
            # 连续凸模型的可行支持点均值仍可行，无需假设内部变量为零。
            inner_point = np.mean(feasible_points, axis=0)
            info['inner_point'] = inner_point.copy()

            # 计时恰从首轮割平面（含最坏点 MILP 的构建）开始。
            cutting_started = time.perf_counter()
            deadline = cutting_started + timelimit
            info['initialization_time'] = cutting_started - total_started
            info['status'] = 'running'
            try:
                while time.perf_counter() < deadline:
                    info['iterations'] += 1
                    with gp.Model('worst_infeasibility', env=env) as worst:
                        configure(worst)
                        worst.Params.IntFeasTol = solver_tolerance
                        worst.Params.MIPGap = 0.0
                        worst.Params.MIPGapAbs = solver_tolerance
                        early_stop_enabled = modified and not force_global_oracle
                        if early_stop_enabled:
                            # 正 incumbent 只用于提出候选；仍须由下面的
                            # 独立 LP 与原模型支撑值校验后才能写入割。
                            worst.Params.BestObjStop = tolerance + 10 * solver_tolerance
                        z = worst.addMVar(dim, lb=-gp.GRB.INFINITY, name='P')
                        hd = worst.addMVar(len(d), lb=-1.0, ub=0.0, name='h')
                        mu = worst.addMVar(len(b_hat), lb=0.0, name='mu')
                        slack = worst.addMVar(len(b_hat), lb=0.0, name='outer_slack')
                        active = worst.addMVar(len(b_hat), vtype=gp.GRB.BINARY,
                                              name='complementarity')
                        if internal_indices:
                            worst.addConstr(B.T @ hd == np.zeros(len(internal_indices)))
                        worst.addConstr(A_hat.T @ mu + C.T @ hd == np.zeros(dim))
                        worst.addConstr(slack == b_hat - A_hat @ z)
                        for j in range(len(b_hat)):
                            worst.addGenConstrIndicator(active[j], 0, slack[j] == 0)
                            worst.addGenConstrIndicator(active[j], 1, mu[j] == 0)
                        worst.setObjective(d @ hd + b_hat @ mu, gp.GRB.MAXIMIZE)
                        oracle_started = time.perf_counter()
                        optimize(worst, kind='milp')
                        info['oracle_history'].append({
                            'iteration': info['iterations'],
                            'runtime': time.perf_counter() - oracle_started,
                            'status': worst.Status, 'n_solutions': worst.SolCount,
                            'objective': float(worst.ObjVal) if worst.SolCount else None,
                            'bound': float(worst.ObjBound), 'nodes': worst.NodeCount,
                            'early_stop_enabled': early_stop_enabled,
                            'elapsed': time.perf_counter() - cutting_started,
                        })
                        allowed_statuses = (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT)
                        if modified:
                            allowed_statuses += (gp.GRB.USER_OBJ_LIMIT,)
                        if worst.Status not in allowed_statuses:
                            raise RuntimeError(f'最坏不可行点 MILP 求解失败，状态 {worst.Status}。')
                        bound = float(worst.ObjBound)
                        info['worst_violation_bound'] = max(0.0, bound)
                        if bound <= tolerance:
                            info['status'] = 'converged'
                            break
                        if (worst.SolCount == 0 or
                                (not modified and worst.Status == gp.GRB.TIME_LIMIT)):
                            raise CalculationTimeLimit
                        bad_point = z.X.copy()

                    q_bad, outer_dual = violation(bad_point)
                    if q_bad <= tolerance:
                        if retry_separation('incumbent_not_infeasible'):
                            continue
                        info['status'] = 'stalled'
                        info['message'] = 'MILP 上界尚未闭合，当前解未给出可分离的不可行点。'
                        break
                    low, high = 0.0, 1.0
                    boundary_point, q_boundary = bad_point.copy(), q_bad
                    # 保留微小不可行的外端对偶，避免边界处 h=0 的退化解。
                    while high - low > 1e-7:
                        fraction = (low + high) / 2
                        point = inner_point + fraction * (bad_point - inner_point)
                        q, candidate_dual = violation(point)
                        if q > tolerance / 10:
                            high = fraction
                            boundary_point, q_boundary, outer_dual = point, q, candidate_dual
                        else:
                            low = fraction

                    normal = -np.asarray(C.T @ outer_dual).ravel()
                    boundary = -float(d @ outer_dual)
                    norm = np.linalg.norm(normal)
                    if norm <= 1e-12:
                        if retry_separation('zero_cut_normal'):
                            continue
                        info['status'] = 'stalled'
                        info['message'] = '对偶证书未生成有效的割法向。'
                        break
                    normal, boundary = normal / norm, boundary / norm
                    # 用原 LP 支持值校验每个割，并向外留数值余量。
                    support_bound, _ = support(normal)
                    boundary = max(boundary, support_bound) + 1e-8
                    repeated = any(np.allclose(row, normal, rtol=0, atol=1e-8)
                                   and old_bound <= boundary + solver_tolerance
                                   for row, old_bound in zip(A_hat, b_hat))
                    if normal @ bad_point - boundary <= solver_tolerance or repeated:
                        if retry_separation('duplicate_cut' if repeated else
                                            'insufficient_separation'):
                            continue
                        info['status'] = 'stalled'
                        info['message'] = '数值容差下无法生成新的有效分离割。'
                        break
                    A_hat = np.vstack([A_hat, normal])
                    b_hat = np.r_[b_hat, boundary]
                    info['n_cuts'] += 1
                    force_global_oracle = False
                    info['history'].append({
                        'iteration': info['iterations'], 'normal': normal.copy(),
                        'boundary': boundary, 'bad_point': bad_point.copy(),
                        'boundary_point': boundary_point.copy(),
                        'violation': q_bad, 'boundary_violation': q_boundary,
                        'worst_violation_bound': info['worst_violation_bound'],
                        'elapsed': time.perf_counter() - cutting_started,
                    })
                else:
                    info['status'] = 'time_limit'
            except CalculationTimeLimit:
                info['status'] = 'time_limit'
            comp_time = time.perf_counter() - cutting_started

    info['computation_time'] = comp_time
    info['total_time'] = time.perf_counter() - total_started
    info['A_hat'], info['b_hat'] = A_hat.copy(), b_hat.copy()
    return (A_hat, b_hat), comp_time


def build_vertices(agg, timelimit, modified=False):
    """Shared continuous-LP extraction and certified support solver."""
    import time
    import gurobipy as gp
    from scipy.linalg import qr
    from scipy.sparse import csr_matrix
    from scipy.spatial import ConvexHull, QhullError
    from pyomo.repn.standard_repn import generate_standard_repn

    timelimit = float(timelimit)
    if not np.isfinite(timelimit) or timelimit <= 0:
        raise ValueError('timelimit 必须是正的有限秒数。')
    total_started = time.perf_counter()
    tolerance, solver_tolerance = 1e-7, 1e-9
    rank_tolerance, duplicate_tolerance = 1e-9, 1e-8
    info = {
        'status': 'initializing', 'iterations': 0, 'n_vertices': 0,
        'n_support_lps': 0, 'n_support_queries': 0,
        'n_lexicographic_lps': 0, 'tolerance': tolerance,
        'n_numerical_retries': 0, 'numerical_retry_history': [],
        'n_certificate_lps': 0, 'n_vertex_certificates': 0,
        'n_certificate_rejections': 0, 'failed_supports': [],
        'vertex_tolerance': duplicate_tolerance,
        'support_strategy': 'dual_rhs_face_with_vertex_certificate',
        'max_dual_residual': 0.0, 'max_dual_gap': 0.0,
        'timelimit': timelimit, 'initialization_time': 0.0,
        'vertex_initialization_time': 0.0, 'computation_time': 0.0,
        'total_time': 0.0, 'history': [],
        'n_working_vertices': 0, 'hull_history': [], 'vertex_elapsed': [],
        'initialization_strategy': 'affinely_independent_simplex',
        'hull_status': 'not_computed', 'projection_dimension': None,
        'affine_hull_verified': False,
        'last_complete_max_improvement_ratio': np.inf,
    }
    if modified:
        agg.vertex_enumeration_modified_info = info
        info.update({
            'direction_seed': 0,
            'direction_distribution': 'standard_normal',
            'initialization_strategy': 'random_directions',
            'hull_status': 'skipped_random_directions',
        })
    else:
        agg.vertex_enumeration_info = info
    source = agg.case_aggregator(build_only=True)
    variables = [v for v in source.component_data_objects(pyo.Var, active=True)
                 if not v.fixed]
    if any(v.is_binary() or v.is_integer() for v in variables):
        raise ValueError('PVE 的凸包内近似保证不能直接用于未固定的整数设备变量。')
    variable_index = {id(v): j for j, v in enumerate(variables)}
    projected_variables = [source.var_proj[t] for t in source.T]
    dim = len(projected_variables)
    if dim == 0:
        raise ValueError('PVE 至少需要一个聚合功率坐标。')

    # 保留全部变量域界、约束常数和等式两侧；固定变量由 repn 代入。
    row_indices, column_indices, coefficients, rhs = [], [], [], []

    def append_row(terms, bound):
        terms = [(j, float(a)) for j, a in terms if a != 0]
        bound = float(bound)
        if not np.isfinite(bound) or any(not np.isfinite(a) for _, a in terms):
            raise ValueError('原模型包含非有限的约束系数或边界。')
        if not terms:
            if bound < -solver_tolerance:
                raise ValueError('原模型含不可行的常数约束。')
            return
        scale = np.linalg.norm([a for _, a in terms])
        row = len(rhs)
        for j, a in terms:
            row_indices.append(row)
            column_indices.append(j)
            coefficients.append(a / scale)
        rhs.append(bound / scale)

    for constraint in source.component_data_objects(pyo.Constraint, active=True):
        repn = generate_standard_repn(constraint.body, compute_values=True)
        if not repn.is_linear():
            raise ValueError(f'PVE 仅支持线性约束：{constraint.name}')
        terms = [(variable_index[id(v)], pyo.value(a))
                 for v, a in zip(repn.linear_vars, repn.linear_coefs)]
        constant = pyo.value(repn.constant)
        if constraint.has_ub():
            append_row(terms, pyo.value(constraint.upper) - constant)
        if constraint.has_lb():
            append_row([(j, -a) for j, a in terms],
                       constant - pyo.value(constraint.lower))
    for j, variable in enumerate(variables):
        if variable.has_ub():
            append_row([(j, 1.0)], pyo.value(variable.ub))
        if variable.has_lb():
            append_row([(j, -1.0)], -pyo.value(variable.lb))
    D = csr_matrix((coefficients, (row_indices, column_indices)),
                   shape=(len(rhs), len(variables)))
    d = np.asarray(rhs, dtype=float)
    vertices = []
    working_vertices = []
    cached_hull = None
    vertex_started = deadline = None
    phase = 'axis_initialization'

    class CalculationTimeLimit(Exception):
        pass

    class VertexNumericalFailure(Exception):
        pass

    def append_vertex(point):
        if not np.all(np.isfinite(point)):
            raise RuntimeError('支持 LP 返回了非有限的投影顶点。')
        if any(np.max(np.abs(point - old)) <= duplicate_tolerance
               for old in vertices):
            return False
        vertices.append(point.copy())
        info['vertex_elapsed'].append(time.perf_counter() - vertex_started)
        info['n_vertices'] = len(vertices)
        return True

    def append_working_vertex(point):
        # archive 里已经存在的轴向顶点仍可能是工作凸包的新顶点。
        if any(np.max(np.abs(point - old)) <= duplicate_tolerance
               for old in working_vertices):
            return False
        working_vertices.append(point.copy())
        info['n_working_vertices'] = len(working_vertices)
        return True

    def affine_basis():
        points = np.asarray(vertices, dtype=float)
        center = points.mean(axis=0)
        _, singular_values, right_vectors = np.linalg.svd(
            points - center, full_matrices=True
        )
        scale = max(1.0, singular_values[0] if len(singular_values) else 0.0)
        rank = int(np.sum(singular_values > rank_tolerance * scale))
        return center, right_vectors[:rank].T, right_vectors[rank:].T

    def compute_working_hull(center, basis):
        """仅计算原 PVE 搜索下一轮方向所需的相对面法向与支撑值。"""
        points = np.asarray(working_vertices, dtype=float)
        relative = (points - center) @ basis
        rank = basis.shape[1]
        if rank == 0:
            normals = np.empty((0, 0))
            bounds = np.empty(0)
        elif rank == 1:
            normals = np.array([[1.0], [-1.0]])
            bounds = np.array([relative[:, 0].max(), -relative[:, 0].min()])
        else:
            if rank >= 5:
                # 高维 Qhull 不能在主线程中中断；其构建及去重也必须
                # 服从剩余时间，到时仍可返回已经生成的全部顶点。
                import pickle
                import subprocess
                import sys
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise CalculationTimeLimit
                worker = (
                    'import pickle,sys,numpy as np\n'
                    'from scipy.spatial import ConvexHull\n'
                    'points=pickle.loads(sys.stdin.buffer.read())\n'
                    'equations=ConvexHull(points).equations\n'
                    '_,indices=np.unique(np.round(equations,10),axis=0,return_index=True)\n'
                    'pickle.dump(equations[indices],sys.stdout.buffer,protocol=4)\n'
                )
                try:
                    result = subprocess.run(
                        [sys.executable, '-c', worker],
                        input=pickle.dumps(relative, protocol=4),
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        timeout=remaining,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                    )
                except subprocess.TimeoutExpired:
                    raise CalculationTimeLimit from None
                if result.returncode:
                    raise QhullError(result.stderr.decode('utf-8', errors='replace')[-1000:])
                if time.perf_counter() >= deadline:
                    raise CalculationTimeLimit
                equations = pickle.loads(result.stdout)
            else:
                equations = ConvexHull(relative).equations
                # Qhull 的三角剖分可能给出多个相同的多面体面。
                _, indices = np.unique(np.round(equations, 10), axis=0,
                                       return_index=True)
                equations = equations[indices]
            normals, bounds = equations[:, :-1], -equations[:, -1]
        return {
            'n_vertices': len(working_vertices), 'relative_normals': normals,
            'relative_bounds': bounds,
        }

    def construct_hull(center, basis):
        nonlocal phase
        previous_phase = phase
        phase = 'hull_construction'
        started = time.perf_counter()
        record = {
            'phase': phase, 'context': previous_phase,
            'n_working_vertices': len(working_vertices),
            'n_vertices': len(vertices), 'dimension': basis.shape[1],
            'status': 'running', 'duration': 0.0,
        }
        info['hull_history'].append(record)
        info['hull_status'] = 'running'
        try:
            hull = compute_working_hull(center, basis)
        except CalculationTimeLimit:
            record['status'] = 'time_limit'
            info['hull_status'] = 'time_limit'
            raise
        except QhullError:
            record['status'] = 'failed'
            info['hull_status'] = 'failed'
            raise
        else:
            record['status'] = 'complete'
            info['hull_status'] = 'complete'
            phase = previous_phase
            return hull
        finally:
            record['duration'] = time.perf_counter() - started
            record['elapsed'] = time.perf_counter() - vertex_started

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model('aggregation_progressive_vertex_enumeration', env=env) as joint, \
                gp.Model('aggregation_vertex_certificate', env=env) as certificate:
            joint.Params.OutputFlag = 0
            joint.Params.Threads = 1
            joint.Params.Method = 1
            joint.Params.FeasibilityTol = solver_tolerance
            joint.Params.OptimalityTol = solver_tolerance
            joint.Params.DualReductions = 0
            x = joint.addMVar(len(variables), lb=-gp.GRB.INFINITY, name='x')
            row_constraints = None
            if len(d):
                row_constraints = joint.addMConstr(D, x, gp.GRB.LESS_EQUAL, d)
            # 独立坐标变量也支持 var_proj 被显式固定的退化投影。
            P = joint.addMVar(dim, lb=-gp.GRB.INFINITY, name='P')
            for j, variable in enumerate(projected_variables):
                value = pyo.value(variable) if variable.fixed else x[variable_index[id(variable)]]
                if variable.fixed and not np.isfinite(value):
                    raise ValueError('固定的聚合坐标必须是有限数。')
                joint.addConstr(P[j] == value)
            joint.update()

            # 两份完整原模型检验投影极点：P+ + P- = 2v。
            # 对称性使逐坐标 max(P+_j-v_j)=0 足以排除一切非零弦。
            certificate.Params.OutputFlag = 0
            certificate.Params.Threads = 1
            certificate.Params.Method = 1
            certificate.Params.FeasibilityTol = solver_tolerance
            certificate.Params.OptimalityTol = solver_tolerance
            certificate.Params.DualReductions = 0
            certificate_points = []
            for label in ('plus', 'minus'):
                xc = certificate.addMVar(len(variables), lb=-gp.GRB.INFINITY, name='x_' + label)
                if len(d):
                    certificate.addMConstr(D, xc, gp.GRB.LESS_EQUAL, d)
                pc = certificate.addMVar(dim, lb=-gp.GRB.INFINITY, name='P_' + label)
                for j, variable in enumerate(projected_variables):
                    value = pyo.value(variable) if variable.fixed else xc[variable_index[id(variable)]]
                    certificate.addConstr(pc[j] == value)
                certificate_points.append(pc)
            midpoint = certificate.addConstr(
                certificate_points[0] + certificate_points[1] == np.zeros(dim)
            )
            certificate.update()
            joint_matrix = joint.getA()
            joint_rhs = np.asarray(joint.getAttr('RHS'))
            original_senses = None if row_constraints is None else row_constraints.Sense.copy()

            def optimize(solver_model=joint, lexicographic=False, certification=False):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise CalculationTimeLimit
                solver_model.Params.TimeLimit = remaining
                info['n_support_lps'] += 1
                if lexicographic:
                    info['n_lexicographic_lps'] += 1
                if certification:
                    info['n_certificate_lps'] += 1
                solver_model.optimize()
                if solver_model.Status == gp.GRB.TIME_LIMIT:
                    raise CalculationTimeLimit
                return solver_model.Status

            def vertex_certificate(point):
                # 检验对象是完整原区域，不是工作凸包或本次最优面。
                info['n_vertex_certificates'] += 1
                midpoint.RHS = 2 * point
                certificate.update()
                certificate.Params.Method = 1
                certificate.Params.Presolve = -1
                try:
                    for coordinate in range(dim):
                        certificate.setObjective(certificate_points[0][coordinate], gp.GRB.MAXIMIZE)
                        status = optimize(certificate, certification=True)
                        if status != gp.GRB.OPTIMAL:
                            info['n_numerical_retries'] += 1
                            certificate.Params.Method = 0
                            certificate.Params.Presolve = 0
                            certificate.reset()
                            status = optimize(certificate, certification=True)
                        if status != gp.GRB.OPTIMAL:
                            return False
                        if certificate.ConstrVio > solver_tolerance + 1e-12:
                            return False
                        radius_bound = max(certificate.ObjVal, certificate.ObjBound) - point[coordinate]
                        if radius_bound > duplicate_tolerance:
                            info['n_certificate_rejections'] += 1
                            return False
                    return True
                finally:
                    certificate.Params.Method = 1
                    certificate.Params.Presolve = -1

            def check_dual():
                # 保留 EQ 任意符号的乘子；只在驻点和 gap 达到精度时
                # 将原 LE 行按互补性改成 EQ。RHS 始终是原始 d。
                pi = np.asarray(joint.getAttr('Pi'))
                objective = np.asarray(joint.getAttr('Obj'))
                residual = float(np.max(np.abs(joint_matrix.T @ pi - objective)))
                gap = abs(float(joint_rhs @ pi) - float(joint.ObjVal))
                info['max_dual_residual'] = max(info['max_dual_residual'], residual)
                info['max_dual_gap'] = max(info['max_dual_gap'], gap)
                if residual > solver_tolerance + 1e-12 or gap > 1e-8 * max(1.0, abs(joint.ObjVal)):
                    raise VertexNumericalFailure('支持 LP 对偶精度不足，不能锁定最优面。')
                if row_constraints is not None:
                    current_pi = row_constraints.Pi.copy()
                    less_equal = row_constraints.Sense == gp.GRB.LESS_EQUAL
                    if np.any(current_pi[less_equal] < -solver_tolerance - 1e-12):
                        raise VertexNumericalFailure('支持 LP 对偶符号未达到原求解精度。')
                    return less_equal & (current_pi > 0)
                return np.zeros(0, dtype=bool)

            def support(direction):
                """原支持值用于 IR；返回点必须独立通过投影极点检验。"""
                direction = np.asarray(direction, dtype=float)
                direction_norm = np.linalg.norm(direction)
                if direction_norm <= 0 or not np.isfinite(direction_norm):
                    raise ValueError('支持方向必须是非零有限向量。')
                unit_direction = direction / direction_norm
                info['n_support_queries'] += 1
                last_message = '未取得可认证的投影顶点。'
                for attempt in range(2):
                    try:
                        if attempt:
                            info['n_numerical_retries'] += 1
                            info['numerical_retry_history'].append({
                                'phase': phase, 'method': 0,
                                'elapsed': time.perf_counter() - vertex_started,
                            })
                            joint.Params.Method = 0
                            joint.Params.Presolve = 0
                            joint.Params.NumericFocus = 3
                            joint.reset()
                        joint.setObjective(unit_direction @ P, gp.GRB.MAXIMIZE)
                        status = optimize()
                        if status == gp.GRB.UNBOUNDED:
                            raise ValueError('聚合投影无界；PVE 需要有界的聚合区域。')
                        if status != gp.GRB.OPTIMAL:
                            raise VertexNumericalFailure(f'支持 LP 未求得最优解，状态 {status}。')
                        support_bound = max(float(joint.ObjVal), float(joint.ObjBound))
                        candidate = P.X.copy()
                        if vertex_certificate(candidate):
                            return support_bound * direction_norm, candidate

                        # 用正乘子的原 LE 行定义最优面；不锁浮点目标值。
                        for coordinate in range(dim):
                            lock_rows = check_dual()
                            if np.any(lock_rows):
                                row_constraints[lock_rows].Sense = gp.GRB.EQUAL
                            joint.setObjective(P[coordinate], gp.GRB.MAXIMIZE)
                            status = optimize(lexicographic=True)
                            if status != gp.GRB.OPTIMAL:
                                raise VertexNumericalFailure(f'原约束最优面 LP 求解失败，状态 {status}。')
                            candidate = P.X.copy()
                        check_dual()
                        primary_gap = support_bound - float(unit_direction @ candidate)
                        if abs(primary_gap) > 1e-8 * max(1.0, abs(support_bound)):
                            raise VertexNumericalFailure('最优面选点未保留原支持值。')
                        if vertex_certificate(candidate):
                            return support_bound * direction_norm, candidate
                        last_message = '最优面候选未通过独立极点检验。'
                    except VertexNumericalFailure as exc:
                        last_message = str(exc)
                    finally:
                        if row_constraints is not None:
                            row_constraints.Sense = original_senses
                        joint.Params.Method = 1
                        joint.Params.Presolve = -1
                        joint.Params.NumericFocus = 0
                        joint.update()
                raise VertexNumericalFailure(last_message)

            def record_failed_support(direction, message):
                info['failed_supports'].append({
                    'phase': phase, 'direction': np.asarray(direction).copy(),
                    'message': str(message),
                    'elapsed': time.perf_counter() - vertex_started,
                })

            # axis 初始化也占用 timelimit；准备模型和环境的时间单列。
            vertex_started = time.perf_counter()
            deadline = vertex_started + timelimit
            info['initialization_time'] = vertex_started - total_started
            info['status'] = 'running'
            try:
                if modified:
                    phase = 'random_directions'
                    direction_rng = np.random.default_rng(info['direction_seed'])
                    while True:
                        if time.perf_counter() >= deadline:
                            raise CalculationTimeLimit
                        direction = direction_rng.standard_normal(dim)
                        info['iterations'] += 1
                        try:
                            _, point = support(direction)
                        except CalculationTimeLimit:
                            # A solver can report its local time limit
                            # just before the shared wall-clock deadline.
                            if time.perf_counter() >= deadline:
                                raise
                            record_failed_support(direction, 'Support query reached its solver time limit.')
                            continue
                        except VertexNumericalFailure as exc:
                            record_failed_support(direction, exc)
                            continue
                        # Retain every unique completed certified vertex;
                        # duplicate queries still count towards the budget.
                        if append_vertex(point):
                            info['history'].append({
                                'phase': phase, 'direction': direction.copy(),
                                'vertex': point.copy(), 'added': True,
                                'query': info['n_support_queries'],
                                'elapsed': time.perf_counter() - vertex_started,
                            })

                for axis in np.eye(dim):
                    for direction in (axis, -axis):
                        try:
                            _, point = support(direction)
                        except VertexNumericalFailure as exc:
                            record_failed_support(direction, exc)
                            continue
                        added = append_vertex(point)
                        info['history'].append({
                            'phase': phase, 'direction': direction.copy(),
                            'vertex': point.copy(), 'added': added,
                            'elapsed': time.perf_counter() - vertex_started,
                        })

                # 极端数值情形也不提前丢弃预算；重试原初始化方向。
                while not vertices:
                    for axis in np.eye(dim):
                        for direction in (axis, -axis):
                            try:
                                _, point = support(direction)
                            except VertexNumericalFailure as exc:
                                record_failed_support(direction, exc)
                                continue
                            append_vertex(point)

                # ±坐标轴的极值点可能共线，不能据此断言真实区域低维。
                # 沿当前仿射空间的法向查询；发现外点则补秩，否则验证降维。
                phase = 'affine_initialization'
                while True:
                    center, basis, null_basis = affine_basis()
                    if basis.shape[1] == dim:
                        break
                    expanded = False
                    failed_affine_probe = False
                    max_affine_violation = 0.0
                    for normal in null_basis.T:
                        for direction in (normal, -normal):
                            try:
                                support_bound, point = support(direction)
                            except VertexNumericalFailure as exc:
                                record_failed_support(direction, exc)
                                failed_affine_probe = True
                                continue
                            added = append_vertex(point)
                            violation = support_bound - float(direction @ center)
                            max_affine_violation = max(max_affine_violation, violation)
                            info['history'].append({
                                'phase': phase, 'direction': direction.copy(),
                                'vertex': point.copy(), 'added': added,
                                'affine_violation': violation,
                                'elapsed': time.perf_counter() - vertex_started,
                            })
                            if abs(float(normal @ (point - center))) > rank_tolerance:
                                _, new_basis, _ = affine_basis()
                                if new_basis.shape[1] > basis.shape[1]:
                                    expanded = True
                                    break
                        if expanded:
                            break
                    if not expanded:
                        if failed_affine_probe:
                            continue
                        info['affine_residual_bound'] = max_affine_violation
                        if max_affine_violation > 10 * rank_tolerance:
                            info['message'] = '支持证书与仿射秩的数值精度不一致。'
                            continue
                        center, basis, null_basis = affine_basis()
                        break
                info['vertex_initialization_time'] = time.perf_counter() - vertex_started
                if info['status'] != 'stalled':
                    # 原文仅要求 rank+1 个仿射独立初始顶点。主元 QR
                    # 避免顺序选点误取近数值相关点，且不额外调用支持 LP。
                    phase = 'simplex_initialization'
                    archive = np.asarray(vertices, dtype=float)
                    rank = basis.shape[1]
                    simplex_indices = [0]
                    if rank:
                        differences = (archive[1:] - archive[0]) @ basis
                        _, _, pivots = qr(differences.T, mode='economic', pivoting=True)
                        simplex_indices.extend((pivots[:rank] + 1).tolist())
                    for index in simplex_indices:
                        append_working_vertex(archive[index])
                    center = np.asarray(working_vertices).mean(axis=0)
                    info['initial_simplex_indices'] = np.asarray(simplex_indices, dtype=int)
                    info['vertex_initialization_time'] = time.perf_counter() - vertex_started
                    info['affine_hull_verified'] = True
                    info['projection_dimension'] = basis.shape[1]
                    info['inner_point'] = center.copy()
                    info['affine_basis'] = basis.copy()
                    phase = 'expansion'
                    while True:
                        if time.perf_counter() >= deadline:
                            raise CalculationTimeLimit
                        if cached_hull is None or cached_hull['n_vertices'] != len(working_vertices):
                            cached_hull = construct_hull(center, basis)
                        normals = cached_hull['relative_normals']
                        bounds = cached_hull['relative_bounds']
                        if not len(normals):
                            info['last_complete_max_improvement_ratio'] = 0.0
                            info['status'] = 'converged'
                            break
                        if np.any(bounds <= 0):
                            info['status'] = 'stalled'
                            info['message'] = '初始顶点均值未形成数值上严格的相对内点。'
                            break
                        info['iterations'] += 1
                        record = {
                            'phase': phase, 'iteration': info['iterations'],
                            'n_vertices_before': len(vertices),
                            'n_vertices_after': len(vertices),
                            'n_working_vertices_before': len(working_vertices),
                            'n_working_vertices_after': len(working_vertices),
                            'facet_count': len(normals), 'facets_completed': 0,
                            'max_improvement_ratio': 0.0, 'complete': False,
                            'failed_facets': 0,
                            'new_vertices': [], 'elapsed': 0.0,
                        }
                        info['history'].append(record)
                        # 当前 round 的面在循环开始时固定；本轮新增点不改变扫描顺序。
                        for normal, boundary in zip(normals, bounds):
                            direction = basis @ normal
                            try:
                                support_bound, point = support(direction)
                            except VertexNumericalFailure as exc:
                                record_failed_support(direction, exc)
                                record['failed_facets'] += 1
                                continue
                            height = support_bound - float(direction @ center)
                            improvement = max(0.0, (height - boundary) / boundary)
                            record['max_improvement_ratio'] = max(
                                record['max_improvement_ratio'], improvement
                            )
                            record['facets_completed'] += 1
                            # 用户要求返回所有已完成的支持顶点；与原文 C1
                            # 相比也保留 IR=0 的新顶点，仍为安全的凸包内近似。
                            append_vertex(point)
                            if append_working_vertex(point):
                                record['new_vertices'].append(point.copy())
                            record['n_vertices_after'] = len(vertices)
                            record['n_working_vertices_after'] = len(working_vertices)
                            record['elapsed'] = time.perf_counter() - vertex_started
                        record['complete'] = record['failed_facets'] == 0
                        if record['complete']:
                            info['last_complete_max_improvement_ratio'] = record['max_improvement_ratio']
                        if record['complete'] and record['max_improvement_ratio'] <= tolerance:
                            info['status'] = 'converged'
                            break
                        if not record['new_vertices']:
                            # 未证明闭合时继续求解/重试，不能因为一轮
                            # 未找到新点或某个面数值失败而提前结束预算。
                            info['message'] = '本轮未闭合，继续重试原凸包的面。'
            except CalculationTimeLimit:
                info['status'] = 'time_limit'
                info['interrupted_phase'] = phase
            except QhullError as exc:
                info['status'] = 'stalled'
                info['message'] = f'未扰动顶点的凸包构建失败：{exc}'

            if phase in ('axis_initialization', 'affine_initialization',
                         'simplex_initialization'):
                info['vertex_initialization_time'] = time.perf_counter() - vertex_started
            comp_time = time.perf_counter() - vertex_started

    V = np.asarray(vertices, dtype=float).reshape((-1, dim))
    info['n_vertices'] = len(V)
    info['computation_time'] = comp_time
    info['total_time'] = time.perf_counter() - total_started
    return V, comp_time

