"""Shared continuous/discrete resource models for the T24 and T96 cases."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from Simulator.paths import DATA_ROOT, RESULTS_ROOT
from Simulator.Approximator import ErrorCalculator


class Aggregator:
    def __init__(self, seed=None,T = 24, data = None, discrete_rate = 0.0):
        """
        初始化设备参数生成器

        参数:
            seed (int, optional): 随机种子，用于复现结果
        """

        self.rng = np.random.default_rng(seed)
        self.discrete_EV = 0
        self.discrete_TCL = 0

        if data is None:
            self.EV_list = []
            self.TCL_list = []
            self.ESS_list = []
            self.PV_list = []
            self.CL_list = []
        else:
            self.EV_list = data['EV_list']
            self.discrete_EV = sum(self.EV_list[i]['is_discrete'] for i in range(len(self.EV_list)))

            self.TCL_list = data['TCL_list']
            self.discrete_TCL = sum(self.TCL_list[i]['is_discrete'] for i in range(len(self.TCL_list)))

            self.ESS_list = data['ESS_list']
            self.PV_list = data['PV_list']
            self.CL_list = data['CL_list']
        self.T = T
        self.deltaT = 24/self.T

        self.data_path = DATA_ROOT
        data = np.load(self.data_path / 'profiles_data' / 'profiles_data.npz')
        data_deltaT = 5/60
        self.discrete_rate = discrete_rate
        self.theta_amb = data['temp_data'][0:int(self.T*self.deltaT/data_deltaT):int(self.deltaT/data_deltaT)]
        self.pv_profiles = data['pv_data'][0:int(self.T*self.deltaT/data_deltaT):int(self.deltaT/data_deltaT)]
        self.load_profiles = data['load_data'][0:int(self.T*self.deltaT/data_deltaT):int(self.deltaT/data_deltaT)]
        self.pmax = []
        self.pmin = []

    def gen_ESS(self, N):
        for i in range(N):
            self.ESS_list.append({
                'Pchg': self.rng.choice([25.,50.],p=[0.5,0.5]),
                'init_SOC': self.rng.uniform(0.3,0.7),
                'eta_chg': self.rng.uniform(0.95,0.98),
                'eta_dis': self.rng.uniform(0.96, 0.98),
                'B':self.rng.choice([100.,200.],p=[0.5,0.5]),
            })
        print('ESS generated')

    def gen_TCL(self, N):
        hp_df = pd.read_csv(self.data_path / 'aggregator_data' / 'ZH_buildings.csv').head(N)
        COP = 3.85
        for index, row in hp_df.iterrows():
            # 提取必要的字段
            HBLD = row['HBLD']
            CBLD = row['CBLD']
            PRT = row['PRT']
            is_discrete= self.rng.choice([0, 1], p=[1-self.discrete_rate, self.discrete_rate])
            self.discrete_TCL += is_discrete
            # 创建字典并添加到列表
            self.TCL_list.append({
                'H': HBLD,  # 热容
                'C': CBLD,  # 热阻
                'temp_min':18.0,
                'temp_max':22.0,
                'temp_set': 20.0,
                'Qmax': PRT,  # 热功率
                'COP': COP,  # COP系数
                'is_discrete': is_discrete,
            })
        print('TCL generated')

    def gen_EV(self, N):
        """
        生成N个电动汽车(EV)的充电需求参数
        完全按照MATLAB版本逻辑实现

        参数:
            N (int): 要生成的电动汽车数量
        """
        P_chg = 7  # 充电功率 (kW)
        B = 50  # 电池容量 (kWh)
        eta_chg = 0.95  # 充电效率

        i = 0
        while i < N:
            is_discrete= self.rng.choice([0, 1], p=[1-self.discrete_rate, self.discrete_rate])
            self.discrete_EV += is_discrete
            # 白天充电 (不跨天)
            start_time = min(10, max(7, self.rng.normal(8.5, 1.5)))
            end_time = min(20, start_time + self.rng.uniform(8, 12))
            end_SOC = 1-P_chg*self.deltaT/B
            start_SOC = min(1-2*P_chg*self.deltaT/B, max(self.rng.normal(0.5, 0.2), 0.2))

            # 检查充电功率是否足够
            if (end_SOC - start_SOC) * B / (end_time - start_time) > P_chg:
                pass  # 原MATLAB代码中这里只是占位

            self.EV_list.append({
                'ta': start_time,
                'td': end_time,
                'SOCa': start_SOC,
                'SOCd': end_SOC,
                'SOCmax': 1,
                'P_chg': P_chg,
                'B': B,
                'eta_chg': eta_chg,
                'is_discrete': is_discrete,
            })
            i += 1

    def gen_PV(self, N):
        """
        生成N个光伏(PV)的参数
        每个PV的出力曲线在原始数据的上下20%范围内随机
        """
        for i in range(N):
            # 生成该PV的随机缩放因子（在0.8到1.2之间）
            scale_factor = self.rng.uniform(0.8, 1.2)

            # 基于数据文件中的光伏曲线，添加随机变化
            pv_output_profile = np.zeros(self.T)
            capacity = 30.0 * scale_factor  # 额定容量也在20%范围内随机

            for t in range(self.T):
                # 每个时刻的出力也在原始数据的80%-120%之间随机
                time_variation = self.rng.uniform(0.8, 1.2)
                pv_output_profile[t] = self.pv_profiles[t] * capacity * time_variation

            self.PV_list.append({
                'P_max': capacity,  # 最大出力
                'output_profile': pv_output_profile,  # 出力时间序列
                'baseline': pv_output_profile
            })
        print(f'PV (Photovoltaic) generated: {N} units with ±20% variation')

    def gen_CL(self, N):
        """
        生成N个可中断负荷(CL)的参数
        每个CL的负荷曲线在原始数据的上下20%范围内随机
        """
        for i in range(N):
            # 生成该CL的整体随机缩放因子（在0.8到1.2之间）
            scale_factor = self.rng.uniform(0.8, 1.2)

            # 生成时变最大可削减功率，基于负荷基准值，每个时刻添加20%随机变化
            p_max_profile = np.zeros(self.T)
            for t in range(self.T):
                # 每个时刻的负荷值在原始数据的80%-120%之间随机
                time_variation = self.rng.uniform(0.8, 1.2)
                p_max_profile[t] = 30 * self.load_profiles[t] * scale_factor * time_variation

            # 生成时变成本
            cost_profile = np.zeros(self.T)
            for t in range(self.T):
                hour = t * self.deltaT
                if 18 <= hour <= 22:  # 晚上高峰时段
                    cost_profile[t] = 0.2
                elif 8 <= hour <= 17:  # 工作时段
                    cost_profile[t] = 0.15
                else:  # 夜间时段
                    cost_profile[t] = 0.1

            self.CL_list.append({
                'p_max_profile': p_max_profile,  # 时变最大可削减功率（已添加随机变化）
                'curt_ratio': 0.4,  # 削减比例曲线
                'cost_profile':cost_profile,
                'baseline':p_max_profile,
            })
        print(f'CL (Interruptible Load) generated: {N} units with ±20% variation')

    def case_aggregator(self, A=None, b=None, template=None, gradient_mode="approx",
                        build_only=False, output_dir=None):
        """Build the device model and training inputs without writing files.

        ``build_only=True`` returns the original Pyomo model; otherwise the
        returned case dictionary contains the calculator and training defaults.
        """
        model = pyo.ConcreteModel()
        model.T = pyo.Set(initialize=range(self.T))
        model.var_proj = pyo.Var(model.T)  # 聚合功率变量
        p_max = np.zeros(self.T)
        p_min = np.zeros(self.T)

        model.ESSs= pyo.Set(initialize=range(len(self.ESS_list)))
        model.e_ESS= pyo.Var(model.ESSs, model.T, within=pyo.Reals)
        model.p_chg_ESS = pyo.Var(model.ESSs, model.T, within=pyo.NonNegativeReals)
        model.p_dis_ESS = pyo.Var(model.ESSs, model.T, within=pyo.NonNegativeReals)
        model.u_ESS = pyo.Var(model.ESSs, model.T, within=pyo.Binary)


        model.TCLs= pyo.Set(initialize=range(len(self.TCL_list)))
        model.discete_TCLs = pyo.Set(initialize=range(self.discrete_TCL))
        model.temp_TCL= pyo.Var(model.TCLs, model.T, within=pyo.Reals)
        model.p_TCL = pyo.Var(model.TCLs, model.T, within=pyo.NonNegativeReals)
        model.u_TCL = pyo.Var(model.discete_TCLs, model.T, within=pyo.Binary)

        # 定义EV集合
        model.EVs = pyo.Set(initialize=range(len(self.EV_list)))
        model.discete_EVs = pyo.Set(initialize=range(self.discrete_EV))
        # 定义变量
        model.e_EV = pyo.Var(model.EVs, model.T, within=pyo.NonNegativeReals)  # 每个EV的功率变量
        model.p_EV = pyo.Var(model.EVs, model.T, within=pyo.NonNegativeReals)  # 每个EV的功率变量
        model.u_EV = pyo.Var(model.discete_EVs, model.T, within=pyo.Binary)  # u_EV为0-1变量

        # 定义PV集合
        model.PVs = pyo.Set(initialize=range(len(self.PV_list)))
        # 定义PV变量
        model.p_PV = pyo.Var(model.PVs, model.T, within=pyo.NonNegativeReals)  # PV出力

        # 定义CL集合
        model.CLs = pyo.Set(initialize=range(len(self.CL_list)))
        # 定义CL变量
        model.p_CL = pyo.Var(model.CLs, model.T, within=pyo.NonNegativeReals)  # 负荷削减量

        # 初始化基线
        baseline = np.zeros(self.T)

        # ESS能量动态方程
        for i in model.ESSs:
            ess_data = self.ESS_list[i]
            p_max += ess_data['Pchg']
            p_min -= ess_data['Pchg']
            def energy_dynamics_rule(m, t):
                if t > 0:
                    return m.e_ESS[i, t] == m.e_ESS[i, t - 1] + (self.deltaT * (
                                m.p_chg_ESS[i, t] * ess_data['eta_chg'] - m.p_dis_ESS[i, t] / ess_data['eta_dis']))
                return pyo.Constraint.Skip

            model.add_component(f"ess_energy_dynamics_{i}", pyo.Constraint(model.T, rule=energy_dynamics_rule))

            def energy_limit_rule(m, t):
                return pyo.inequality(0, m.e_ESS[i, t], ess_data['B'])

            model.add_component(f"ess_energy_limit_{i}", pyo.Constraint(model.T, rule=energy_limit_rule))

            def charging_power_limit_rule(m, t):
                return m.p_chg_ESS[i, t]<=m.u_ESS[i,t]*ess_data['Pchg']

            model.add_component(f"ess_charging_power_limit_{i}", pyo.Constraint(model.T, rule=charging_power_limit_rule))

            def discharging_power_limit_rule(m, t):
                return m.p_dis_ESS[i, t]<=(1-m.u_ESS[i,t])*ess_data['Pchg']

            model.add_component(f"ess_discharging_power_limit_{i}",
                                pyo.Constraint(model.T, rule=discharging_power_limit_rule))

            def initial_energy_condition_rule(m):
                return m.e_ESS[i, 0] == ess_data['init_SOC'] * ess_data['B']

            model.add_component(f"ess_initial_energy_condition_{i}", pyo.Constraint(rule=initial_energy_condition_rule))

            def final_energy_condition_rule(m):
                return m.e_ESS[i, self.T - 1] >= m.e_ESS[i, 0]

            model.add_component(f"ess_final_energy_condition_{i}", pyo.Constraint(rule=final_energy_condition_rule))

        i_discrete = 0
        for i in model.TCLs:
            tcl_data = self.TCL_list[i]  # 默认使用第一个EV的数据
            p_min-=tcl_data['Qmax'] / tcl_data['COP']
            if tcl_data['is_discrete']:
                def power_limit_rule(m, t):
                    return m.p_TCL[i, t] == m.u_TCL[i_discrete, t] * tcl_data['Qmax']/tcl_data['COP']
                model.add_component(f"tcl_power_limit_discrete{i}", pyo.Constraint(model.T, rule=power_limit_rule))
                i_discrete+=1
            else:
                def power_limit_rule(m, t):
                    return pyo.inequality(0, m.p_TCL[i, t], tcl_data['Qmax']/tcl_data['COP'])
                model.add_component(f"tcl_power_limit{i}", pyo.Constraint(model.T, rule=power_limit_rule))

            alpha = np.exp(-self.deltaT*tcl_data['H']/tcl_data['C'])
            # 动态温度更新公式
            def temperature_dynamics_rule(m, t):
                if t > 0:
                    return m.temp_TCL[i, t] == (alpha * m.temp_TCL[i, t - 1] +(1 - alpha) * (self.theta_amb[t] +tcl_data['COP']/tcl_data['H']*model.p_TCL[i, t]))
                return pyo.Constraint.Skip

            model.add_component(f"temperature_dynamics{i}", pyo.Constraint(model.T, rule=temperature_dynamics_rule))

            # 温度限制约束：确保温度在上下限之间
            def temperature_limit_rule(m, t):
                return pyo.inequality(tcl_data['temp_min'], m.temp_TCL[i, t], tcl_data['temp_max'])

            model.add_component(f"temperature_limit{i}", pyo.Constraint(model.T, rule=temperature_limit_rule))

            def initial_temperature_condition_rule(m):
                return m.temp_TCL[i, 0] == tcl_data['temp_set']
            model.add_component(f"initial_temperature_condition{i}", pyo.Constraint(rule=initial_temperature_condition_rule))



        i_discrete = 0
        # 1. 确保每个EV的功率和能量满足边界条件
        for i in model.EVs:
            ev_data = self.EV_list[i]  # 默认使用第一个EV的数据
            # 计算离散化的到达和离开时间索引
            ta_idx = int(np.floor(ev_data['ta'] / self.deltaT))
            td_idx = int(np.ceil(ev_data['td'] / self.deltaT))
            p_min[ta_idx:td_idx]-=ev_data['P_chg']
            if ev_data['is_discrete']:
                def power_limit_rule(m, t):
                    if ta_idx <= t < td_idx:
                        return m.p_EV[i, t] == m.u_EV[i_discrete, t] * ev_data['P_chg']  # 功率为0或ev_data['P_chg']
                    else:
                        return m.p_EV[i, t] == 0  # 离开时功率为 0
                model.add_component(f"ev_power_limit_discrete{i}", pyo.Constraint(model.T, rule=power_limit_rule))
                i_discrete+=1
            else:
                # 约束1: 功率上下限约束
                def power_limit_rule(m, t):
                    if ta_idx <= t < td_idx:
                        return (0, m.p_EV[i,t], ev_data['P_chg'])
                    else:
                        return m.p_EV[i,t] == 0
                model.add_component(f"ev_power_limit{i}", pyo.Constraint(model.T, rule=power_limit_rule))

            # 约束2: 初始能量状态
            def initial_energy_rule(m):
                return m.e_EV[i,ta_idx] == ev_data['SOCa'] * ev_data['B']
            model.add_component(f"initial_energy{i}", pyo.Constraint(rule=initial_energy_rule))

            # 约束3: 能量动态更新
            def energy_dynamics_rule(m, t):
                if ta_idx <= t < td_idx:
                    return m.e_EV[i,t + 1] == m.e_EV[i,t] + m.p_EV[i,t] * ev_data['eta_chg'] * self.deltaT
                else:
                    return pyo.Constraint.Skip
            model.add_component(f"energy_dynamics{i}", pyo.Constraint(model.T, rule=energy_dynamics_rule))

            # 约束4: 能量状态上下限
            def energy_limit_rule(m, t):
                if ta_idx <= t <= td_idx:
                    return (ev_data['SOCa'] * ev_data['B'], m.e_EV[i,t], ev_data['SOCmax'] * ev_data['B'])
                else:
                    return pyo.Constraint.Skip
            model.add_component(f"energy_limit{i}", pyo.Constraint(model.T, rule=energy_limit_rule))

            # 约束5: 离开时最小能量要求
            def departure_energy_rule(m):
                return m.e_EV[i, td_idx] >= ev_data['SOCd'] * ev_data['B']
            model.add_component(f"departure_energy{i}", pyo.Constraint(rule=departure_energy_rule))

        # PV约束 - 光伏
        for i in model.PVs:
            pv_data = self.PV_list[i]
            p_max += pv_data['output_profile']  # PV增加发电能力上限
            baseline += pv_data['baseline']
            # PV出力限制，根据时间变化
            def pv_power_limit_rule(m, t):
                return pyo.inequality(0, m.p_PV[i, t], pv_data['output_profile'][t])

            model.add_component(f"pv_power_limit_{i}", pyo.Constraint(model.T, rule=pv_power_limit_rule))

        # CL约束 - 可中断负荷
        for i in model.CLs:
            cl_data = self.CL_list[i]
            p_min -= (cl_data['p_max_profile'])  # CL增加负荷能力（负值）
            p_max -= (cl_data['p_max_profile'])*(1-cl_data['curt_ratio'])  # CL增加负荷能力（负值）
            baseline -= cl_data['baseline']

            # 负荷削减限制
            def cl_curtailment_limit_rule(m, t):
                return pyo.inequality((1-cl_data['curt_ratio'])*cl_data['p_max_profile'][t], m.p_CL[i, t], cl_data['p_max_profile'][t])

            model.add_component(f"cl_curtailment_limit_{i}", pyo.Constraint(model.T, rule=cl_curtailment_limit_rule))
        self.pmax = p_max
        self.pmin = p_min
        def agg_power_rule(m, t):
            generation = (sum(m.p_EV[i, t] for i in m.EVs) +
                         sum(m.p_PV[i, t] for i in m.PVs) +
                         sum(m.p_dis_ESS[i, t] for i in m.ESSs))
            consumption = (sum(m.p_TCL[i, t] for i in m.TCLs) +
                          sum(m.p_CL[i, t] for i in m.CLs) +
                          sum(m.p_chg_ESS[i, t] for i in m.ESSs))
            return m.var_proj[t]*(p_max[t]-p_min[t])+p_min[t] == generation - consumption
        model.agg_power_constraint = pyo.Constraint(model.T, rule=agg_power_rule)
        self.model = model
        if build_only:
            return model
        lower_tri = np.tril(np.ones((self.T, self.T)))
        lower_tri_normalized = lower_tri / lower_tri.sum(axis=1, keepdims=True)
        if A is None:
            A = np.vstack([np.eye(self.T), -np.eye(self.T),
                           lower_tri_normalized, -lower_tri_normalized])
        calculator = ErrorCalculator(
            original_model={"model": model}, A_hat=A, b_hat=b,
            solver="gurobi", gradient_mode=gradient_mode,
        )
        self.errorcalculator = calculator
        template = (template or "peb").lower()
        output_dir = Path(output_dir) if output_dir is not None else (
            RESULTS_ROOT / "aggregation" / f"T{self.T}" / gradient_mode / template)
        return {
            "casename": f"aggregation_T{self.T}_{template}",
            "A_hat": calculator.A_hat,
            "b_hat": calculator.b_hat,
            "errorcalculator": calculator,
            "trainer_configure": {
                "call_interval": 1, "optimizer": "SGD", "lr": 1e-3,
                "batch_size": 1,
                "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.98},
                "n_cal": 1, "cal_feas": True, "cal_opt": True, "rate_opt_feas": 1.0,
            },
            "params": {"params_dict": {}, "dataloader": [None], "count": 0},
            "result_path": str(output_dir / "pretrainnet_weights.pth"),
            "metadata": {
                "T": self.T, "deltaT": self.deltaT,
                "EV_list": self.EV_list, "TCL_list": self.TCL_list,
                "ESS_list": self.ESS_list, "PV_list": self.PV_list,
                "CL_list": self.CL_list, "discrete_rate": self.discrete_rate,
            },
            "n_train": 300,
        }

    def gen_boundaries(self):
        """
        生成PV、CL和TCL的功率和能量边界
        参考MATLAB版本gen_boundaries.m的实现

        返回:
            dict: 包含PV、CL、TCL的功率和能量边界的字典
        """
        boundaries = {
            'PV': {'Pmin': [], 'Pmax': [], 'Emin': [], 'Emax': []},
            'CL': {'Pmin': [], 'Pmax': [], 'Emin': [], 'Emax': []},
            'TCL': {'Pmin': [], 'Pmax': [], 'Emin': [], 'Emax': []}
        }

        # PV边界计算 (参考MATLAB代码 lines 50-61)
        # PV是发电设备，功率为正值（发电为正）
        for i in range(len(self.PV_list)):
            pv_data = self.PV_list[i]
            # PV的output_profile即为Pmax，Pmin为0
            P_max_profile = pv_data['output_profile']
            P_min_profile = np.zeros(self.T)

            # 功率边界 (kW) - 发电为正
            LP = P_min_profile  # 下限 = 最小出力 (0)
            UP = P_max_profile  # 上限 = 最大出力 (正值)

            # 能量边界 (kWh) - 累积和
            LE = np.cumsum(P_min_profile) * self.deltaT  # 下限
            UE = np.cumsum(P_max_profile) * self.deltaT  # 上限
            # 确保Emin <= Emax
            UE = np.maximum(UE, LE)

            boundaries['PV']['Pmin'].append(LP)
            boundaries['PV']['Pmax'].append(UP)
            boundaries['PV']['Emin'].append(LE)
            boundaries['PV']['Emax'].append(UE)

        # CL边界计算 (CL是负荷消耗，功率为负值，负荷为负)
        for i in range(len(self.CL_list)):
            cl_data = self.CL_list[i]
            # CL的p_max_profile是最大负荷，可削减到(1-curt_ratio)
            P_max_profile = cl_data['p_max_profile']
            curt_ratio = cl_data['curt_ratio']
            P_min_profile = P_max_profile * (1 - curt_ratio)

            # 功率边界 (kW) - 负荷为负
            UP = -P_min_profile  # 上限 = 最小负荷（削减后，负值）
            LP = -P_max_profile  # 下限 = 最大负荷（负值）

            # 能量边界 (kWh) - 累积和
            UE = -np.cumsum(P_min_profile) * self.deltaT  # 上限
            LE = -np.cumsum(P_max_profile) * self.deltaT  # 下限
            # 确保Emin <= Emax
            LE = np.minimum(LE, UE)

            boundaries['CL']['Pmin'].append(LP)
            boundaries['CL']['Pmax'].append(UP)
            boundaries['CL']['Emin'].append(LE)
            boundaries['CL']['Emax'].append(UE)

        # TCL边界计算 (加热设备，与HVAC制冷方向相反)
        for i in range(len(self.TCL_list)):
            tcl_data = self.TCL_list[i]

            # 热动态参数计算
            H = tcl_data['H']  # 热容
            C = tcl_data['C']  # 热阻
            COP = tcl_data['COP']
            T_min = tcl_data['temp_min']
            T_max = tcl_data['temp_max']
            T_set = tcl_data['temp_set']
            P_r = tcl_data['Qmax'] / COP  # 额定电功率

            # 计算热动态参数 alpha 和 beta
            # alpha = exp(-deltaT * H / C) 对应温度衰减系数
            alpha = np.exp(-self.deltaT * H / C)
            # beta = (1 - alpha) * COP / H 对应功率对温度的影响系数
            beta = (1 - alpha) * COP / H

            # 计算最慢功率轨迹 (P_slowest) - 维持温度不低于下限所需的最小功率（加热）
            P_slowest = np.zeros(self.T)
            T_in = np.zeros(self.T + 1)
            T_in[0] = T_set

            for t in range(self.T):
                # 温度自然衰减 (不开启加热时)
                T_in[t + 1] = alpha * T_in[t] + (1-alpha) * (self.theta_amb[t])
                # 如果温度低于下限，需要加热
                if T_in[t + 1] < T_min:
                    P_slowest[t] = (T_min - T_in[t + 1]) / (beta)
                    T_in[t + 1] = T_min

            # 计算最快功率轨迹 (P_fastest) - 全功率加热但不让温度超过上限
            P_fastest = np.zeros(self.T)
            T_in[0] = T_set

            for t in range(self.T):
                # 全功率加热时的温度变化（加热使温度升高，所以是加号）
                T_in[t + 1] = alpha * T_in[t] + (1-alpha) * (self.theta_amb[t]) + beta * P_r
                # 如果温度超过上限，需要减少功率
                if T_in[t + 1] > T_max:
                    P_fastest[t] = P_r - (T_in[t + 1] - T_max) / (beta)
                    T_in[t + 1] = T_max
                else:
                    P_fastest[t] = P_r

            # 功率边界 (kW) - 负荷为负
            UP = np.zeros(self.T)  # 上限 = 0
            LP = -P_r * np.ones(self.T)  # 下限 = 负的额定功率

            # 能量边界 (kWh) - 累积和，负荷为负
            UE = -np.cumsum(P_slowest) * self.deltaT  # 上限
            LE = -np.cumsum(np.maximum(P_fastest, P_slowest)) * self.deltaT  # 下限
            # 确保Emin <= Emax
            LE = np.minimum(LE, UE)

            boundaries['TCL']['Pmin'].append(LP)
            boundaries['TCL']['Pmax'].append(UP)
            boundaries['TCL']['Emin'].append(LE)
            boundaries['TCL']['Emax'].append(UE)

        return boundaries

    def count_complexity(self):
        # 1. 查看连续变量个数
        continuous_vars = sum(1 for v in self.model.component_data_objects(pyo.Var, active=True)
                              if not v.is_binary() and not v.is_integer())
        print(f"连续变量个数: {continuous_vars}")

        # 2. 查看0-1变量（二进制变量）个数
        binary_vars = sum(1 for v in self.model.component_data_objects(pyo.Var, active=True)
                          if v.is_binary())
        print(f"0-1变量个数: {binary_vars}")

        # 3. 查看约束数量
        constraints = sum(1 for c in self.model.component_data_objects(pyo.Constraint, active=True))
        print(f"约束数量: {constraints}")
