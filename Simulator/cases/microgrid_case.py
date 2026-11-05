"""Microgrid devices, uncertain physical constraints and disaggregation."""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import torch
from torch.utils.data import DataLoader, Dataset

from Simulator.paths import DATA_ROOT, RESULTS_ROOT
from Simulator.Approximator import ErrorCalculator, pyomo_params_to_numpy


class MicroGrid:
    def __init__(self, seed=None, T=24, data=None):
        """
        初始化微网设备参数生成器

        参数:
            seed (int, optional): 随机种子，用于复现结果
            T (int): 时间步数，默认为24小时
            data (dict, optional): 预定义的设备数据
        """
        self.rng = np.random.default_rng(seed)
        self.case_name = 'microgrid'
        if not data:
            self.MT_list = []  # 可控分布式发电（燃气轮机）
            self.PV_list = []  # 光伏
            self.TL_list = []  # 可转移负荷
            self.CL_list = []  # 可中断负荷
            self.ESS_list = [] # 储能
            self.HP_list = []  # 热泵/中央空调
        else:
            self.MT_list = data['MT_list']
            self.PV_list = data['PV_list']
            self.TL_list = data['TL_list']
            self.CL_list = data['CL_list']
            self.ESS_list = data['ESS_list']
            self.HP_list = data['HP_list']

        self.T = T
        self.deltaT = 24/self.T

        self.data_path = DATA_ROOT
        prof_data = np.load(self.data_path / 'profiles_data' / 'profiles_data.npz')
        data_deltaT = 5/60
        self.temp_amb = prof_data['temp_data'][0:int(self.T*self.deltaT/data_deltaT):int(self.deltaT/data_deltaT)]
        self.pv_profiles = prof_data['pv_data'][0:int(self.T*self.deltaT/data_deltaT):int(self.deltaT/data_deltaT)]
        self.load_profiles = prof_data['load_data'][0:int(self.T*self.deltaT/data_deltaT):int(self.deltaT/data_deltaT)]

    def gen_MT(self, N):
        """
        生成可控分布式发电（燃气轮机）参数
        二次成本函数
        """
        for i in range(N):
            self.MT_list.append({
                'P_min': 20,  # 最小技术出力 100kW
                'P_max': 100,  # 最大出力 500kW
                'a': 0.015,  # 二次成本系数
                'b': 0.15,  # 一次成本系数
                'c': 5,  # 固定成本系数
                'min_on_time': 4,  # 最小运行时间 小时
                'initial_status': 0,  # 初始状态 (0:停机, 1:运行)
                'baseline': np.zeros(self.T)
            })
        print('MT (MicroTurbine) generated')

    def gen_PV(self, N):
        """
        生成光伏参数
        无成本
        """
        for i in range(N):
            # 使用从数据文件中读取的光伏出力曲线
            base_output = 1.0  # 基础出力系数设为1，使用标准曲线
            # 基于数据文件中的光伏曲线
            pv_output_profile = np.zeros(self.T)
            capacity = 200.0
            for t in range(self.T):
                # 直接使用数据文件中的光伏出力曲线，乘以额定容量
                pv_output_profile[t] = self.pv_profiles[t] * capacity  # 额定容量1000kW
            self.PV_list.append({
                'P_max': capacity,  # 最大出力 1000kW
                'output_profile': pv_output_profile,  # 出力时间序列
                'baseline': pv_output_profile
            })
        print('PV (Photovoltaic) generated')

    def gen_TL(self, N):
        """
        生成可转移负荷参数
        包括化工生产、电动汽车充电等
        常数成本
        """
        # 创建两种类型的可转移负荷：电动汽车充电和化工生产
        for i in range(N):
            if i % 2 == 0:  # 电动汽车充电

                data_dict = {
                    'P_rated': 70.0,  # 额定功率 400kW
                    'E_TL': 280.0,  # 总需消耗能量 2000kWh
                    'start_window': 8,  # 开始时间窗 14:00
                    'end_window': 18,  # 结束时间窗 次日08:00（32时段，如果每时段1小时）
                    'cost': 0.2*280,  # 常数成本（完成奖励）
                    'type': 'EV_charging'  # 负荷类型
                }

                baseline = np.zeros(self.T)
                baseline[data_dict['start_window']:data_dict['end_window']]=data_dict['E_TL']/(data_dict['end_window']-data_dict['start_window'])

                data_dict['baseline'] = baseline
                self.TL_list.append(data_dict)
            else:  # 化工生产
                data_dict = {
                    'P_rated': 120.0,  # 额定功率 300kW
                    'E_TL': 720.0,  # 总需消耗能量 1500kWh
                    'start_window': 0,  # 开始时间窗 00:00
                    'end_window': 23,  # 结束时间窗 24:00
                    'cost': 0.2*720.0,  # 常数成本（生产收益，按批次）
                    'type': 'chemical_production'  # 负荷类型
                }
                baseline = np.zeros(self.T)
                # baseline[data_dict['start_window']:data_dict['end_window']]=data_dict['E_TL']/(data_dict['end_window']-data_dict['start_window'])
                baseline[data_dict['start_window']:data_dict['start_window']+int(data_dict['E_TL']/data_dict['P_rated'])] = data_dict['P_rated']
                data_dict['baseline'] = baseline
                self.TL_list.append(data_dict)
        print('TL (Transferable Load) generated')

    def gen_CL(self, N):
        """
        生成可中断负荷参数
        需求响应项目
        时变成本
        """
        for i in range(N):
            # 生成时变最大可削减功率，基于负荷基准值
            p_max_profile = 300*self.load_profiles

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
                'p_max_profile': p_max_profile,  # 时变最大可削减功率
                'curt_ratio': 0.4,  # 削减比例曲线
                'cost_profile':cost_profile,
                'baseline':p_max_profile,
            })
        print('CL (Interruptible Load) generated')

    def gen_ESS(self, N):
        """
        生成储能参数
        充放电功率成本
        """
        for i in range(N):
            self.ESS_list.append({
                'Pchg': 100.0,  # 充电功率上限 500kW
                'Pdis': 100.0,  # 放电功率上限 500kW
                'init_SOC': 0.5,  # 初始SOC 50%
                'eta_chg': 0.95,  # 充电效率 95%
                'eta_dis': 0.98,  # 放电效率 95%
                'E_rated': 250.0,  # 额定容量 1000kWh
                'cost_chg': 0.05,  # 充电成本（循环损耗）
                'cost_dis': 0.05,  # 放电成本（循环损耗）
                'SOC_min': 0.1,  # 最小SOC 10%
                'SOC_max': 1.0,  # 最大SOC 100%
                'baseline': np.zeros(self.T)
            })
        print('ESS (Energy Storage System) generated')

    @staticmethod
    def calculate_HP_baseline(HP_data, temp_amb, deltaT=1.0):
        """
        计算HP的基线（始终维持温度为设定点的电功率曲线）
        基于MATLAB代码cal_HP_bounds_sum_daily.m中的baseline部分实现

        参数:
        - HP_data: HP设备数据字典
        - temp_amb: 环境温度数组
        - deltaT: 时间步长（小时）

        返回:
        - P_baseline: 基线电功率曲线
        """
        T = len(temp_amb)  # 时间步数
        H = HP_data['H']  # 热阻
        C = HP_data['C']  # 热容
        temp_set = HP_data['temp_set']  # 设定温度
        temp_init = HP_data['temp_init']  # 初始温度
        Qmax = HP_data['Qmax']  # 最大热功率
        COP = HP_data['COP']  # COP系数
        # 计算时间常数相关的系数
        alpha = np.exp(-deltaT * H / C)
        # 初始化温度和功率数组
        T_in = np.zeros(T + 1)  # 室内温度
        P_baseline = np.zeros(T)  # 基线电功率
        Q_baseline = np.zeros(T)  # 基线热功率
        # 设置初始温度
        T_in[0] = temp_init
        # 时间步循环
        for t in range(T):
            # 计算自然温度变化后的室内温度
            T_in[t + 1] = alpha * T_in[t] + (1 - alpha) * temp_amb[t]
            # 如果室内温度低于设定温度，需要加热
            if T_in[t + 1] < temp_set:
                # 计算需要的热量来维持设定温度
                Q_required = H * ((temp_set - alpha * T_in[t]) / (1 - alpha) - temp_amb[t])
                # 限制热功率不超过最大值
                Q_required = min(Q_required, Qmax)
                # 计算对应的电功率
                P_baseline[t] = Q_required / COP
                # 更新实际室内温度（考虑加热）
                T_in[t + 1] = alpha * T_in[t] + (1 - alpha) * (temp_amb[t] + Q_required / H)
                Q_baseline[t] = Q_required

        return P_baseline

    def gen_HP(self, N):
        """
        生成热泵/中央空调参数
        无成本
        """
        # 使用固定参数，避免依赖外部文件
        for i in range(N):
            data_dict = {
                'H': 4.61,  # 热阻（典型值）
                'C': 372,  # 热阻（典型值）
                'temp_min': 18.0,  # 最低温度
                'temp_max': 22.0,  # 最高温度
                'temp_set': 20.0,  # 设定温度
                'temp_init': 20.0,  # 设定温度
                'Qmax':185.0,  # 热功率 300kW
                'COP': 3.85,  # COP系数 3.0
            }
            baseline = self.calculate_HP_baseline(HP_data=data_dict, temp_amb=self.temp_amb, deltaT=self.deltaT)
            data_dict['baseline'] = baseline
            self.HP_list.append(data_dict)
        print('HP (Heat Pump/Air Conditioning) generated')

    def build_mg_constraints(self, model=None, conservativeness=0.0):

        """
        构建微网聚合模型
        """
        # 构建Pyomo模型
        if model == None:
            model = pyo.ConcreteModel()
        model.T = pyo.Set(initialize=range(self.T))
        model.var_proj = pyo.Var(model.T)  # 聚合功率变量
        p_max = np.zeros(self.T)
        p_min = np.zeros(self.T)
        baseline = np.zeros(self.T)

        # 定义各设备集合
        model.MTs = pyo.Set(initialize=range(len(self.MT_list)))

        model.PVs = pyo.Set(initialize=range(len(self.PV_list)))
        model.TLs = pyo.Set(initialize=range(len(self.TL_list)))

        model.CLs = pyo.Set(initialize=range(len(self.CL_list)))
        model.ESSs = pyo.Set(initialize=range(len(self.ESS_list)))
        model.HPs = pyo.Set(initialize=range(len(self.HP_list)))

        # 定义变量
        # MT变量
        model.p_MT = pyo.Var(model.MTs, model.T, within=pyo.NonNegativeReals)  # MT出力
        model.u_MT = pyo.Var(model.MTs, model.T, within=pyo.Binary)  # 运行状态 (u)
        model.y_MT = pyo.Var(model.MTs, model.T, within=pyo.Binary)  # 启动动作 (y)
        model.z_MT = pyo.Var(model.MTs, model.T, within=pyo.Binary)  # 关停动作 (z)

        # PV变量
        model.p_PV = pyo.Var(model.PVs, model.T, within=pyo.NonNegativeReals)  # PV出力
        init_dict = {(pv, t): self.PV_list[pv]['output_profile'][t]
                     for pv in model.PVs for t in model.T}
        if len(self.PV_list):
            model.p_PV_rated = pyo.Param(model.PVs, model.T, within=pyo.NonNegativeReals,
                                         initialize=init_dict, mutable=True)

        # TL变量
        model.I_TL = pyo.Var(model.TLs, model.T, within=pyo.Binary)            # TL运行状态
        model.p_TL = pyo.Var(model.TLs, model.T, within=pyo.NonNegativeReals)  # TL功率

        # CL变量
        model.p_CL = pyo.Var(model.CLs, model.T, within=pyo.NonNegativeReals)  # 负荷削减量
        init_dict = {(cl, t): self.CL_list[cl]['p_max_profile'][t]
                         for cl in model.CLs for t in model.T}
        if len(self.CL_list):
            model.p_CL_rated = pyo.Param(model.CLs, model.T, within=pyo.NonNegativeReals,
                                         initialize=init_dict, mutable=True)
        # ESS变量
        # Power index t=0..T-1 represents physical interval (t,t+1].
        # SOC has T+1 boundary states: initial E_0 through terminal E_T.
        model.ESS_state_times = pyo.RangeSet(0, self.T)
        model.e_ESS = pyo.Var(model.ESSs, model.ESS_state_times, within=pyo.Reals)
        model.p_chg_ESS = pyo.Var(model.ESSs, model.T, within=pyo.NonNegativeReals)  # ESS充电功率
        model.p_dis_ESS = pyo.Var(model.ESSs, model.T, within=pyo.NonNegativeReals)  # ESS放电功率
        model.u_ESS = pyo.Var(model.ESSs, model.T, within=pyo.Binary)          # ESS充放电状态

        # HP变量
        model.temp_HP = pyo.Var(model.HPs, model.T, within=pyo.Reals)          # HP室内温度
        init_dict = {t: self.temp_amb[t] for t in model.T}
        if len(self.HP_list):
            model.temp_amb = pyo.Param(model.T, within=pyo.Reals,
                                         initialize=init_dict, mutable=True)


        model.p_HP = pyo.Var(model.HPs, model.T, within=pyo.NonNegativeReals)  # HP功率

        shrinking_rate = conservativeness

        # MT约束 - 可控分布式发电（燃气轮机）
        for i in model.MTs:
            mt_data = self.MT_list[i]
            p_max += mt_data['P_max']
            baseline += mt_data['baseline']

            # 1. MT出力下限约束（与运行状态关联）
            def mt_power_lower_limit_rule(m, t):
                return mt_data['P_min'] * m.u_MT[i, t] <= m.p_MT[i, t]

            model.add_component(f"mt_power_lower_limit_{i}",
                                pyo.Constraint(model.T, rule=mt_power_lower_limit_rule))

            # 2. MT出力上限约束（与运行状态关联）
            def mt_power_upper_limit_rule(m, t):
                return m.p_MT[i, t] <= mt_data['P_max'] * m.u_MT[i, t]

            model.add_component(f"mt_power_upper_limit_{i}",
                                pyo.Constraint(model.T, rule=mt_power_upper_limit_rule))
            # 2. 状态-动作关系约束
            min_on_time = mt_data['min_on_time']
            initial_status = mt_data['initial_status']

            # 2.1 状态转移方程约束
            def state_transition_rule(m, t):
                if t == 0:  # 第一个时段
                    # u[t] - u_initial = y[t] - z[t]
                    return m.u_MT[i, t] - initial_status == m.y_MT[i, t] - m.z_MT[i, t]
                else:
                    # u[t] - u[t-1] = y[t] - z[t]
                    return m.u_MT[i, t] - m.u_MT[i, t - 1] == m.y_MT[i, t] - m.z_MT[i, t]

            model.add_component(f"mt_state_transition_{i}", pyo.Constraint(model.T, rule=state_transition_rule))

            # 2.2 启动和关停动作互斥约束
            def action_mutex_rule(m, t):
                # y[t] + z[t] <= 1 (不能同时启动和关停)
                return m.y_MT[i, t] + m.z_MT[i, t] <= 1

            model.add_component(f"mt_action_mutex_{i}", pyo.Constraint(model.T, rule=action_mutex_rule))

            # 3. 最小运行时间约束
            def min_uptime_rule(m, t):
                # Include starts at t=0 even before a full uptime window exists.
                # Only periods inside this horizon are constrained, as before.
                first_start = max(0, t - min_on_time + 1)
                sum_startup = sum(m.y_MT[i, tau] for tau in range(first_start, t + 1))
                return m.u_MT[i, t] >= sum_startup

            model.add_component(f"mt_min_uptime_{i}", pyo.Constraint(model.T, rule=min_uptime_rule))

            # 5. 初始状态约束
            # 5.1 初始运行时间约束
            # With no prior on-duration supplied, retain the conservative
            # convention that an initially-on unit stays on for the first U
            # periods. This includes t=0 and is clipped to the horizon.
            if initial_status == 1:
                for t in range(min(min_on_time, len(model.T))):
                    model.add_component(f"mt_initial_uptime_{i}_{t}",
                                        pyo.Constraint(expr=model.u_MT[i, t] == 1))

            # 6. 启动和关停动作的逻辑约束（可选，加强约束）
            def startup_logic_rule(m, t):
                if t == 0:
                    return pyo.Constraint.Skip
                # y[t] <= 1 - u[t-1] (只有停机状态才能启动)
                return m.y_MT[i, t] <= 1 - m.u_MT[i, t - 1]

            def shutdown_logic_rule(m, t):
                if t == 0:
                    return pyo.Constraint.Skip
                # z[t] <= u[t-1] (只有运行状态才能关停)
                return m.z_MT[i, t] <= m.u_MT[i, t - 1]

            model.add_component(f"mt_startup_logic_{i}", pyo.Constraint(model.T, rule=startup_logic_rule))
            model.add_component(f"mt_shutdown_logic_{i}", pyo.Constraint(model.T, rule=shutdown_logic_rule))

        # PV约束 - 光伏
        for i in model.PVs:
            pv_data = self.PV_list[i]
            p_max += pv_data['output_profile']  # PV增加发电能力上限
            baseline += pv_data['baseline']
            # PV出力限制，根据时间变化
            def pv_power_limit_rule(m, t):
                return pyo.inequality(0, m.p_PV[i, t], (1-shrinking_rate)*model.p_PV_rated[i,t])
            
            model.add_component(f"pv_power_limit_{i}", pyo.Constraint(model.T, rule=pv_power_limit_rule))

        # TL约束 - 可转移负荷
        for i in model.TLs:
            tl_data = self.TL_list[i]
            p_min[tl_data['start_window']:tl_data['end_window']] -= tl_data['P_rated']  # 转移负荷增加负荷能力（负值）
            baseline -= tl_data['baseline']
            # 操作时段约束
            def tl_operation_time_rule(m, t):
                if tl_data['start_window'] <= t < tl_data['end_window']:
                    return pyo.Constraint.Skip  # 在时间窗内可运行
                else:
                    return m.I_TL[i, t] == 0  # 在时间窗外必须停止
            
            model.add_component(f"tl_operation_time_{i}", pyo.Constraint(model.T, rule=tl_operation_time_rule))
            
            # 功率-状态关系
            def tl_power_state_rule(m, t):
                return m.p_TL[i, t] == tl_data['P_rated'] * m.I_TL[i, t]
            
            model.add_component(f"tl_power_state_{i}", pyo.Constraint(model.T, rule=tl_power_state_rule))

            # 总能量约束
            def tl_total_energy_rule(m):
                return sum(m.p_TL[i, t] * self.deltaT for t in model.T) >= tl_data['E_TL']
            
            model.add_component(f"tl_total_energy_{i}", pyo.Constraint(rule=tl_total_energy_rule))

        # CL约束 - 可中断负荷
        for i in model.CLs:
            cl_data = self.CL_list[i]
            p_min -= (cl_data['p_max_profile'])  # 转移负荷增加负荷能力（负值）
            p_max -= (cl_data['p_max_profile'])*(1-cl_data['curt_ratio'])  # 转移负荷增加负荷能力（负值）
            baseline -= cl_data['baseline']

            # 负荷削减限制
            def cl_curtailment_limit_rule(m, t):
                return pyo.inequality((1-cl_data['curt_ratio']+shrinking_rate)*model.p_CL_rated[i,t], m.p_CL[i, t], (1-shrinking_rate)*model.p_CL_rated[i,t])
            
            model.add_component(f"cl_curtailment_limit_{i}", pyo.Constraint(model.T, rule=cl_curtailment_limit_rule))

        # ESS约束 - 储能
        for i in model.ESSs:
            ess_data = self.ESS_list[i]
            p_max += ess_data['Pdis']  # 放电增加发电能力上限
            p_min -= ess_data['Pchg']  # 充电增加负荷能力上限（负值）
            baseline += ess_data['baseline']
            # SOC动态方程
            def ess_energy_dynamics_rule(m, t):
                return m.e_ESS[i, t + 1] == m.e_ESS[i, t] + (self.deltaT * (
                            m.p_chg_ESS[i, t] * ess_data['eta_chg'] - m.p_dis_ESS[i, t] / ess_data['eta_dis'])) / ess_data['E_rated']

            model.add_component(f"ess_energy_dynamics_{i}", pyo.Constraint(model.T, rule=ess_energy_dynamics_rule))

            # SOC范围限制
            def ess_soc_limit_rule(m, t):
                return pyo.inequality(ess_data['SOC_min'], m.e_ESS[i, t], ess_data['SOC_max'])

            model.add_component(f"ess_soc_limit_{i}", pyo.Constraint(model.ESS_state_times, rule=ess_soc_limit_rule))

            # 充电功率限制
            def ess_charging_power_limit_rule(m, t):
                return m.p_chg_ESS[i, t] <= m.u_ESS[i, t] * ess_data['Pchg']

            model.add_component(f"ess_charging_power_limit_{i}", pyo.Constraint(model.T, rule=ess_charging_power_limit_rule))

            # 放电功率限制
            def ess_discharging_power_limit_rule(m, t):
                return m.p_dis_ESS[i, t] <= (1 - m.u_ESS[i, t]) * ess_data['Pdis']

            model.add_component(f"ess_discharging_power_limit_{i}", pyo.Constraint(model.T, rule=ess_discharging_power_limit_rule))

            # 初始SOC条件
            def ess_initial_soc_rule(m):
                return m.e_ESS[i, 0] == ess_data['init_SOC']

            model.add_component(f"ess_initial_soc_{i}", pyo.Constraint(rule=ess_initial_soc_rule))

            # 最终SOC条件（保持能量平衡）
            def ess_final_soc_rule(m):
                return m.e_ESS[i, self.T] >= ess_data['init_SOC']

            model.add_component(f"ess_final_soc_{i}", pyo.Constraint(rule=ess_final_soc_rule))

        # HP约束 - 热泵/中央空调
        for i in model.HPs:
            hp_data = self.HP_list[i]
            p_min -= hp_data['Qmax'] / hp_data['COP']  # HP增加负荷能力上限（负值）
            baseline -= hp_data['baseline']
            
            # 功率限制
            def hp_power_limit_rule(m, t):
                return pyo.inequality(0, m.p_HP[i, t], hp_data['Qmax'] / hp_data['COP'])
            
            model.add_component(f"hp_power_limit_{i}", pyo.Constraint(model.T, rule=hp_power_limit_rule))

            # 室内温度动态
            alpha = np.exp(-self.deltaT * hp_data['H'] / hp_data['C'])
            def hp_temperature_dynamics_rule(m, t):
                if t > 0:
                    return m.temp_HP[i, t] == (alpha * m.temp_HP[i, t - 1] + (1 - alpha) * (
                            model.temp_amb[t] + hp_data['COP'] / hp_data['H'] * model.p_HP[i, t]))
                return pyo.Constraint.Skip

            model.add_component(f"hp_temperature_dynamics_{i}", pyo.Constraint(model.T, rule=hp_temperature_dynamics_rule))

            # 温度限制约束
            def hp_temperature_limit_rule(m, t):
                return pyo.inequality(hp_data['temp_min'], m.temp_HP[i, t], hp_data['temp_max'])

            model.add_component(f"hp_temperature_limit_{i}", pyo.Constraint(model.T, rule=hp_temperature_limit_rule))

            # 初始温度条件
            def hp_initial_temperature_rule(m):
                return m.temp_HP[i, 0] == hp_data['temp_set']
            
            model.add_component(f"hp_initial_temperature_{i}", pyo.Constraint(rule=hp_initial_temperature_rule))

        # 功率平衡约束 - 总功率聚合
        def mg_power_balance_rule(m, t):
            generation = (
                sum(m.p_MT[i, t] for i in m.MTs) + 
                sum(m.p_PV[i, t] for i in m.PVs) +
                sum(m.p_dis_ESS[i, t] for i in m.ESSs)  # 放电为正
            )
            
            consumption = (
                sum(m.p_TL[i, t] for i in m.TLs) + 
                sum(m.p_CL[i, t] for i in m.CLs) +
                sum(m.p_HP[i, t] for i in m.HPs) +
                sum(m.p_chg_ESS[i, t] for i in m.ESSs)  # 充电为负（负荷）
            )
            
            return m.var_proj[t] * (p_max[t] - p_min[t]) + p_min[t] == generation - consumption
        model.mg_power_balance_constraint = pyo.Constraint(model.T, rule=mg_power_balance_rule)
        self.pmax = p_max
        self.pmin = p_min
        self.baseline = baseline


        self.model = model
        metadata = {
            'T': self.T,
            'MT_list': self.MT_list,
            'PV_list': self.PV_list,
            'TL_list': self.TL_list,
            'CL_list': self.CL_list,
            'ESS_list': self.ESS_list,
            'HP_list': self.HP_list,
            'deltaT': self.deltaT,
            'p_max':p_max,
            'p_min':p_min,
            'baseline':baseline
        }

        self.metadata = metadata

    def case_mg(self, conservativeness=0.0, A=None, b=None, num_samples=100,
                noise_scale=0.05, batch_size=5, device="cpu", gradient_mode="approx",
                is_uncertain=True, output_dir=None, solver_options=None):
        """Build a microgrid case without saving weights, snapshots or metadata.

        Uncertainty changes the physical Pyomo parameters; all scenarios train
        one shared PreTrainNet polytope. The runner owns result serialization.
        """
        self.build_mg_constraints(conservativeness=conservativeness)
        model = self.model
        nCL = len(self.CL_list)
        nPV = len(self.PV_list)
        nHP = len(self.HP_list)
        T = self.T
        pv_base = np.array([[self.PV_list[pv]['output_profile'][t] for t in model.T] for pv in model.PVs])
        cl_base = np.array([[self.CL_list[cl]['p_max_profile'][t] for t in model.T] for cl in model.CLs])
        temp_amb_base = np.array(self.temp_amb)

        class CaseData(Dataset):
            def __init__(self, size=num_samples):
                self.size = size
                self.pv_base_tensor = torch.from_numpy(pv_base).float().to(device)  # 形状: [T]
                self.cl_base_tensor = torch.from_numpy(cl_base).float().to(device)  # 形状: [T]
                self.temp_amb_base_tensor = torch.from_numpy(temp_amb_base).float().to(device)  # 形状: [T]

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                sample_dict = {}
                if nPV:
                    sample_dict['p_PV_rated']=torch.randn([nPV, T], dtype=torch.float32, device=device) * noise_scale * self.pv_base_tensor
                if nCL:
                    sample_dict['p_CL_rated']=torch.randn([nCL, T], dtype=torch.float32, device=device) * noise_scale * self.cl_base_tensor
                if nHP:
                    sample_dict['temp_amb']=torch.randn(T, dtype=torch.float32, device=device) * noise_scale * self.temp_amb_base_tensor
                return sample_dict


        if A is None:
            lower_tri = np.tril(np.ones((self.T, self.T)))
            lower_tri_normalized = lower_tri / np.sqrt(lower_tri.sum(axis=1, keepdims=True))
            A = np.vstack([np.eye(self.T), -np.eye(self.T),
                           lower_tri_normalized, -lower_tri_normalized])
        calculator = ErrorCalculator(
            original_model={"model": model}, A_hat=A, b_hat=b,
            solver="gurobi", gradient_mode=gradient_mode,
            solver_options=solver_options,
        )
        self.errorcalculator = calculator
        params_dict, param_count = pyomo_params_to_numpy(model)
        params = {
            "params_dict": params_dict,
            "dataloader": DataLoader(CaseData(), batch_size=batch_size, shuffle=True)
                          if is_uncertain else [None],
            "count": param_count,
        }
        output_dir = Path(output_dir) if output_dir is not None else (
            RESULTS_ROOT / "microgrid" / "training")
        return {
            "casename": self.case_name,
            "A_hat": calculator.A_hat,
            "b_hat": calculator.b_hat,
            "errorcalculator": calculator,
            "trainer_configure": {
                "call_interval": 10, "optimizer": "SGD", "lr": 1e-3,
                "batch_size": 1,
                "scheduler": {"type": "StepLR", "step_size": 200, "gamma": 0.98},
                "n_cal": 1, "cal_feas": True, "cal_opt": True, "rate_opt_feas": 1.0,
            },
            "params": params,
            "metadata": self.metadata,
            "result_path": str(output_dir / "pretrainnet_weights.pth"),
            "n_train": 5,
        }

    def count_complexity(self):
        """
        计算模型复杂度
        """
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

    def generate_sample(self, c, in_approx = True):
        """
        1: 沿给定方向c最小化线性目标，在近似可行域A_hat var_proj<=b_hat的边界上求功率曲线P，
        输出成本CP和曲线P（注意P是var_proj反标幺化的结果）
        """
        p_min = self.pmin
        p_max = self.pmax
        # c由调用方提供；此处不改变方向或随机缩放边界点。

        var_proj = self.errorcalculator.optimize_direction(c, in_approx=in_approx)
        if var_proj is not None:
            P = var_proj * (p_max - p_min) + p_min
            CP = np.dot(c, P)
            return CP, P
        else:
            return None, None

    def disaggregate(self, power_curve):
        """
        3: 调用errorcalculator里面的projection方法，给定一条微网总功率曲线，分解到每个设备的原可行域内，
        输出分解误差，以及分解之后的总功率曲线。
        """
        p_min = self.pmin
        p_max = self.pmax
        # 标幺化输入功率曲线
        var_proj = (power_curve - p_min) / (p_max - p_min)
        # 调用errorcalculator的project方法
        projected_var_proj = self.errorcalculator.project(var_proj, to_approx=False)
        if projected_var_proj is not None:
            power_curve = var_proj * (p_max - p_min) + p_min
            projected_power_curve = projected_var_proj * (p_max - p_min) + p_min
            error= np.sum((power_curve - projected_power_curve) ** 2)
            error_rated = np.sum((var_proj - projected_var_proj) ** 2)
            return error, error_rated, projected_power_curve
        else:
            return None, None, None

    def test_disaggregation_error(self, n_test, num_samples=100, noise_scale=0.05, device='cpu',
                                 pv_shift=0.0, cl_shift=0.0, temp_shift=0.0, progress=None,
                                 optimality_reference='nominal'):
        """Evaluate directional distances with explicit physical reference.

        The default retains the historical nominal optimality reference. With
        ``perturbed``, both physical support and disaggregation use the same
        sampled resource realization. Every sample restores nominal parameters.
        Failed solver calls retain a row with NaN metrics and explicit status.
        """
        if optimality_reference not in ('nominal', 'perturbed'):
            raise ValueError("optimality_reference must be nominal or perturbed")
        # 从case_mg中获取CaseData类的参数
        nCL = len(self.CL_list)
        nPV = len(self.PV_list)
        T = self.T
        pv_base = np.array([[self.PV_list[pv]['output_profile'][t] for t in range(T)] for pv in range(len(self.PV_list))])
        cl_base = np.array([[self.CL_list[cl]['p_max_profile'][t] for t in range(T)] for cl in range(len(self.CL_list))])
        temp_amb_base = np.array(self.temp_amb)

        class CaseData:
            def __init__(self, size=1):
                self.size = size
                self.pv_base_tensor = torch.from_numpy(pv_base).float().to(device)  # 形状: [T]
                self.cl_base_tensor = torch.from_numpy(cl_base).float().to(device)  # 形状: [T]
                self.temp_amb_base_tensor = torch.from_numpy(temp_amb_base).float().to(device)  # 形状: [T]

            def get_item(self, idx):
                return {
                    'p_PV_rated': (torch.randn([nPV, T], dtype=torch.float32, device=device) * noise_scale + pv_shift) * self.pv_base_tensor,
                    'p_CL_rated': (torch.randn([nCL, T], dtype=torch.float32, device=device) * noise_scale + cl_shift) * self.cl_base_tensor,
                    'temp_amb': (torch.randn(T, dtype=torch.float32, device=device) * noise_scale + temp_shift) * self.temp_amb_base_tensor,
                }

        params_dict, _ = pyomo_params_to_numpy(self.model)
        nominal_parameters = {name: info['initial_value'] for name, info in params_dict.items()}
        results = []

        def evaluate_sample(result):
            self.errorcalculator.update_parameters(nominal_parameters)
            c = np.random.randn(self.T)
            result['direction'] = c.copy()
            # Draw the complete pair before any solve can fail, so subsequent
            # sample indices keep the same independent RNG pairing.
            case_data = CaseData().get_item(0)
            params_upd = {
                name: nominal_parameters[name] + data.detach().cpu().numpy()
                for name, data in case_data.items() if name in nominal_parameters
            }
            result['perturbed_parameters'] = {name: value.copy() for name, value in params_upd.items()}
            result['optimality_reference'] = optimality_reference
            cost, power_curve = self.generate_sample(c=c, in_approx=True)
            result['power_curve'] = power_curve
            if power_curve is None or cost is None:
                result.update(solver_status='approximate_support_failed',
                              failure='Approximate-region support solve failed')
                return
            if optimality_reference == 'perturbed':
                self.errorcalculator.update_parameters(params_upd)
            cost0, reference_power = self.generate_sample(c=c, in_approx=False)
            result['reference_power_curve'] = reference_power
            if optimality_reference == 'nominal':
                result['nominal_power_curve'] = reference_power
            if cost0 is None or reference_power is None:
                result.update(solver_status='original_support_failed',
                              failure=f'{optimality_reference} original-region support solve failed')
                return
            reference_x = (reference_power - self.pmin) / (self.pmax - self.pmin)
            approximate_x = self.errorcalculator.project(reference_x, to_approx=True)
            if approximate_x is None:
                result.update(solver_status='approximate_projection_failed',
                              failure=f'{optimality_reference} support projection to approximation failed')
                return
            result['optimality_error_rated'] = float(np.sum((reference_x - approximate_x) ** 2))
            result['approx_projected_reference_power_curve'] = (
                approximate_x * (self.pmax - self.pmin) + self.pmin)
            if optimality_reference == 'nominal':
                result['approx_projected_nominal_power_curve'] = result['approx_projected_reference_power_curve']
            self.errorcalculator.update_parameters(params_upd)
            error, error_rated, projected_power_curve = self.disaggregate(power_curve)
            result['projected_power_curve'] = projected_power_curve
            if error is None or error_rated is None or projected_power_curve is None:
                result.update(solver_status='projection_failed',
                              failure='Perturbed original-region projection solve failed')
                return
            objective_error = cost - cost0
            result.update(
                solver_status='completed',
                feasibility_error=error, feasibility_error_rated=error_rated,
                objective_error=objective_error,
                objective_error_rated=(objective_error / np.abs(cost0) if cost0 != 0 else np.nan),
            )

        try:
            for i in range(n_test):
                result = {
                    'sample_index': i, 'solver_status': 'pending', 'failure': '',
                    'feasibility_error': np.nan, 'feasibility_error_rated': np.nan,
                    'objective_error': np.nan, 'objective_error_rated': np.nan,
                    'optimality_error_rated': np.nan,
                    'power_curve': None, 'projected_power_curve': None,
                    'nominal_power_curve': None, 'approx_projected_nominal_power_curve': None,
                    'reference_power_curve': None, 'approx_projected_reference_power_curve': None,
                }
                results.append(result)
                started = time.perf_counter()
                try:
                    evaluate_sample(result)
                finally:
                    self.errorcalculator.update_parameters(nominal_parameters)
                    result['elapsed_seconds'] = time.perf_counter() - started
                    if progress is not None:
                        progress(result, i + 1, n_test)
        finally:
            self.errorcalculator.update_parameters(nominal_parameters)
        return results
