import pickle

import numpy as np
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from scipy.spatial import ConvexHull
from torch.utils.data import DataLoader, Dataset

from Simulator import PROJECT_ROOT
from Simulator.Approximator import ErrorCalculator, pyomo_params_to_numpy
import pyomo.environ as pyo
from Simulator.Plotter import ErrorVisualizer

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
        self.solver = pyo.SolverFactory('gurobi')
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

        self.data_path = f'{PROJECT_ROOT}\\data\\'
        prof_data = np.load(self.data_path + 'profiles_data\\profiles_data.npz')
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
    def build_mg_constraints(self, model = None, conservativeness=0.0, save_flag = False):

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
        model.e_ESS = pyo.Var(model.ESSs, model.T, within=pyo.Reals)           # ESS SOC
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
                if t < min_on_time:
                    return pyo.Constraint.Skip
                # 最小运行时间约束：如果启动，则后续min_on_time-1小时内不能关停
                sum_startup = sum(m.y_MT[i, tau] for tau in range(t - min_on_time + 1, t + 1))
                return m.u_MT[i, t] >= sum_startup

            model.add_component(f"mt_min_uptime_{i}", pyo.Constraint(model.T, rule=min_uptime_rule))

            # 5. 初始状态约束
            # 5.1 初始运行时间约束
            if initial_status == 1 and min_on_time > 1:
                for t in range(1, min(min_on_time, len(model.T) + 1)):
                    def initial_uptime_rule(m, t_fixed=t):
                        return m.u_MT[i, t_fixed] == 1

                    model.add_component(f"mt_initial_uptime_{i}_{t}",
                                        pyo.Constraint(rule=initial_uptime_rule))

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
                if t > 0:
                    return m.e_ESS[i, t] == m.e_ESS[i, t - 1] + (self.deltaT * (
                                m.p_chg_ESS[i, t] * ess_data['eta_chg'] - m.p_dis_ESS[i, t] / ess_data['eta_dis'])) / ess_data['E_rated']
                return pyo.Constraint.Skip

            model.add_component(f"ess_energy_dynamics_{i}", pyo.Constraint(model.T, rule=ess_energy_dynamics_rule))

            # SOC范围限制
            def ess_soc_limit_rule(m, t):
                return pyo.inequality(ess_data['SOC_min'], m.e_ESS[i, t], ess_data['SOC_max'])

            model.add_component(f"ess_soc_limit_{i}", pyo.Constraint(model.T, rule=ess_soc_limit_rule))

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
                return m.e_ESS[i, self.T - 1] >= ess_data['init_SOC']  # 最终SOC不低于初始SOC的80%

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


        # 定义成本目标函数
        def mg_cost_rule(m):
            # MT成本 (二次成本函数): a*P^2 + b*P + c
            mt_cost = sum(
                self.MT_list[i]['a'] * m.p_MT[i, t] ** 2 + 
                self.MT_list[i]['b'] * m.p_MT[i, t] +
                self.MT_list[i]['c'] * m.u_MT[i, t]
                for i in m.MTs for t in m.T
            )
            
            # ESS成本 (充放电成本)
            ess_cost = sum(
                self.ESS_list[i]['cost_chg'] * m.p_chg_ESS[i, t] + 
                self.ESS_list[i]['cost_dis'] * m.p_dis_ESS[i, t]
                for i in m.ESSs for t in m.T
            )
            
            # TL成本 (完成奖励，作为负成本)
            # tl_cost = sum(
            #     self.TL_list[i]['cost']
            #     for i in m.TLs  # 在最后时段考虑完成成本
            # )
            tl_cost = 0
            
            # CL成本 (时变削减成本)
            cl_cost = sum(
                self.CL_list[i]['cost_profile'][t] * (self.CL_list[i]['p_max_profile'][t]-m.p_CL[i, t])
                for i in m.CLs for t in m.T
            )
            
            # 总成本表达式
            return mt_cost + ess_cost + tl_cost + cl_cost

        # 使用pyomo的Expression来定义目标函数，以便在其他地方调用
        model.mg_cost_expression = pyo.Expression(rule=mg_cost_rule)

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

        if save_flag:
            # 直接保存metadata到文件
            import pickle
            filepath = f'{PROJECT_ROOT}\\results\\{self.case_name}\\metadata.pkl'
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(metadata, f)
            print(f'Metadata saved to {filepath}')
        self.metadata = metadata

    def cal_cost_single_interval(self, t=1, intervals=50):
        """
        计算成本函数
        按照要求的步骤实现：
        1. 构建pyomo模型，定义各个设备的功率以及总功率变量
        """
        # 构建Pyomo模型
        model = pyo.ConcreteModel()
        # 定义各设备集合
        model.MTs = pyo.Set(initialize=range(len(self.MT_list)))
        model.PVs = pyo.Set(initialize=range(len(self.PV_list)))
        model.TLs = pyo.Set(initialize=range(len(self.TL_list)))
        model.CLs = pyo.Set(initialize=range(len(self.CL_list)))
        model.ESSs = pyo.Set(initialize=range(len(self.ESS_list)))
        model.HPs = pyo.Set(initialize=range(len(self.HP_list)))

        # 定义变量
        model.p_MT = pyo.Var(model.MTs, within=pyo.NonNegativeReals)  # MT出力
        model.u_MT = pyo.Var(model.MTs, within=pyo.Binary)  # MT运行状态
        model.p_PV = pyo.Var(model.PVs, within=pyo.NonNegativeReals)  # PV出力
        model.I_TL = pyo.Var(model.TLs, within=pyo.Binary)  # TL运行状态
        model.p_TL = pyo.Var(model.TLs, within=pyo.NonNegativeReals)  # TL功率
        model.p_CL = pyo.Var(model.CLs, within=pyo.NonNegativeReals)  # 负荷削减量
        model.p_chg_ESS = pyo.Var(model.ESSs, within=pyo.NonNegativeReals)  # ESS充电功率
        model.p_dis_ESS = pyo.Var(model.ESSs, within=pyo.NonNegativeReals)  # ESS放电功率
        model.u_ESS = pyo.Var(model.ESSs, within=pyo.Binary)  # ESS充放电状态
        model.p_HP = pyo.Var(model.HPs, within=pyo.NonNegativeReals)  # HP功率

        # 定义总功率变量
        model.p_total = pyo.Var(within=pyo.Reals)  # 总功率（净功率）

        # MT约束 - 微型燃气轮机
        for i in model.MTs:
            mt_data = self.MT_list[i]

            # MT出力下限约束（与运行状态关联）
            def mt_power_lower_limit_rule(m):
                return mt_data['P_min'] * m.u_MT[i] <= m.p_MT[i]

            model.add_component(f"mt_power_lower_limit_{i}",
                                pyo.Constraint(rule=mt_power_lower_limit_rule))

            # MT出力上限约束（与运行状态关联）
            def mt_power_upper_limit_rule(m):
                return m.p_MT[i] <= mt_data['P_max'] * m.u_MT[i]

            model.add_component(f"mt_power_upper_limit_{i}",
                                pyo.Constraint(rule=mt_power_upper_limit_rule))

        # PV约束 - 光伏
        for i in model.PVs:
            pv_data = self.PV_list[i]

            # PV出力限制（0到时段t的额定出力）
            def pv_power_limit_rule(m):
                return pyo.inequality(0, m.p_PV[i], pv_data['output_profile'][t])

            model.add_component(f"pv_power_limit_{i}",
                                pyo.Constraint(rule=pv_power_limit_rule))

        # TL约束 - 可转移负荷
        for i in model.TLs:
            tl_data = self.TL_list[i]

            # 操作时段约束
            if not (tl_data['start_window'] <= t < tl_data['end_window']):
                # 在时间窗外必须为0
                def tl_outside_window_rule(m):
                    return m.I_TL[i] == 0

                model.add_component(f"tl_outside_window_{i}",
                                    pyo.Constraint(rule=tl_outside_window_rule))

            # 功率-状态关系（要么0，要么额定功率）
            def tl_power_state_rule(m):
                return m.p_TL[i] == tl_data['P_rated'] * m.I_TL[i]

            model.add_component(f"tl_power_state_{i}",
                                pyo.Constraint(rule=tl_power_state_rule))

        # CL约束 - 可中断负荷
        for i in model.CLs:
            cl_data = self.CL_list[i]

            # 负荷削减限制（在最大值*(1-削减率)到最大值之间）
            def cl_curtailment_limit_rule(m):
                p_max_t = cl_data['p_max_profile'][t]
                p_min_t = p_max_t * (1 - cl_data['curt_ratio'])
                return pyo.inequality(p_min_t, m.p_CL[i], p_max_t)

            model.add_component(f"cl_curtailment_limit_{i}",
                                pyo.Constraint(rule=cl_curtailment_limit_rule))

        # ESS约束 - 储能系统
        for i in model.ESSs:
            ess_data = self.ESS_list[i]

            # 充电功率限制
            def ess_charging_power_limit_rule(m):
                return m.p_chg_ESS[i] <= m.u_ESS[i] * ess_data['Pchg']

            model.add_component(f"ess_charging_power_limit_{i}",
                                pyo.Constraint(rule=ess_charging_power_limit_rule))

            # 放电功率限制
            def ess_discharging_power_limit_rule(m):
                return m.p_dis_ESS[i] <= (1 - m.u_ESS[i]) * ess_data['Pdis']

            model.add_component(f"ess_discharging_power_limit_{i}",
                                pyo.Constraint(rule=ess_discharging_power_limit_rule))

        # HP约束 - 热泵
        for i in model.HPs:
            hp_data = self.HP_list[i]

            # 功率限制（0到最大功率）
            def hp_power_limit_rule(m):
                return pyo.inequality(0, m.p_HP[i], hp_data['Qmax'] / hp_data['COP'])

            model.add_component(f"hp_power_limit_{i}",
                                pyo.Constraint(rule=hp_power_limit_rule))

        # 功率平衡约束 - 总功率聚合（参考build_mg_constraints）
        def power_balance_rule(m):
            generation = (
                    sum(m.p_MT[i] for i in m.MTs) +
                    sum(m.p_PV[i] for i in m.PVs) +
                    sum(m.p_dis_ESS[i] for i in m.ESSs)  # 放电为正（发电侧）
            )

            consumption = (
                    sum(m.p_TL[i] for i in m.TLs) +
                    sum(m.p_CL[i] for i in m.CLs) +
                    sum(m.p_HP[i] for i in m.HPs) +
                    sum(m.p_chg_ESS[i] for i in m.ESSs)  # 充电为正（负荷侧）
            )

            # 净功率 = 发电 - 用电
            return m.p_total == generation - consumption

        model.power_balance_constraint = pyo.Constraint(rule=power_balance_rule)

        model.p_setpoint = pyo.Param(initialize=1e6, mutable=True)  # 默认值很大
        model.power_limit_constraint = pyo.Constraint(rule=lambda m: m.p_total == m.p_setpoint)

        # 定义成本目标函数（时段t的成本）
        def cost_objective_rule(m):
            # MT成本 (二次成本函数): a*P^2 + b*P + c*u
            mt_cost = sum(
                self.MT_list[i]['a'] * m.p_MT[i] ** 2 +
                self.MT_list[i]['b'] * m.p_MT[i] +
                self.MT_list[i]['c'] * m.u_MT[i]
                for i in m.MTs
            )

            # ESS成本 (充放电成本)
            ess_cost = sum(
                self.ESS_list[i]['cost_chg'] * m.p_chg_ESS[i] +
                self.ESS_list[i]['cost_dis'] * m.p_dis_ESS[i]
                for i in m.ESSs
            )

            # CL成本 (削减成本，削减量 = 最大负荷 - 实际负荷)
            cl_cost = sum(
                self.CL_list[i]['cost_profile'][t] * (self.CL_list[i]['p_max_profile'][t] - m.p_CL[i])
                for i in m.CLs
            )

            # 总成本
            return mt_cost + ess_cost + cl_cost

        # 定义目标函数：最小化成本
        model.cost_objective = pyo.Objective(rule=cost_objective_rule, sense=pyo.minimize)

        solver = pyo.SolverFactory('gurobi')
        p_points = np.linspace(self.pmax[t], self.pmin[t], intervals)
        obj_points = np.zeros(intervals)
        last_valid_obj = None
        for i in range(intervals):
            model.p_setpoint.set_value(p_points[i])
            result = solver.solve(model, tee=False)  # tee=False不输出求解信息

            # 检查求解状态
            if (result.solver.status == pyo.SolverStatus.ok) and \
                    (result.solver.termination_condition == pyo.TerminationCondition.optimal):
                # 求解成功，获取目标函数值
                obj_points[i] = pyo.value(model.cost_objective)
                last_valid_obj = obj_points[i]  # 更新上一个有效值
            else:
                # 求解失败但没有抛出异常
                print(f"Warning: 求解点 {i} (p={p_points[i]:.2f}) 失败 - "
                      f"状态: {result.solver.termination_condition}")
                if last_valid_obj is not None:
                    obj_points[i] = last_valid_obj
                else:
                    obj_points[i] = np.inf  # 如果是第一个点就失败，设为无穷大
        hull_p, hull_obj = self.lower_convex_hull(p_points, obj_points)

        return hull_p, hull_obj
    @staticmethod
    def lower_convex_hull(x, y):
        """
        计算下凸包（适用于最小化成本函数）使用叉积判断
        """
        # 按x排序，如果x相同则按y排序
        points = np.column_stack((x, y))
        sorted_indices = np.lexsort((y, x))
        points_sorted = points[sorted_indices]

        # 初始化栈
        stack = []

        for i in range(len(points_sorted)):
            point = points_sorted[i]

            # 当栈中至少有两个点，且当前点与栈顶两个点构成非下凸时
            while len(stack) >= 2:
                p1 = stack[-2]  # 倒数第二个点
                p2 = stack[-1]  # 最后一个点
                p3 = point  # 当前点

                # 计算叉积 (p2-p1) × (p3-p2)
                cross_product = (p2[0] - p1[0]) * (p3[1] - p2[1]) - \
                                (p2[1] - p1[1]) * (p3[0] - p2[0])

                # 如果叉积 <= 0，说明p2不是下凸包上的点
                if cross_product <= 0:
                    stack.pop()  # 移除栈顶点
                else:
                    break

            stack.append(point)

        # 转换回数组
        stack = np.array(stack)
        return stack[:, 0], stack[:, 1]

    def cal_apx_cost(self, intervals = 100):
        # model.obj_pwl = pyo.Var(self.T, within=pyo.Reals)
        # model.pwl_constraints = pyo.ConstraintList()
        p_points_list = []
        obj_points_list = []
        for t in range(self.T):
            p_points, obj_points = self.cal_cost_single_interval(t = t, intervals=intervals)
            p_points_list.append(p_points)
            obj_points_list.append(obj_points)
            # n = len(p_points)
            # for i in range(1, n):
            #     # 计算当前段的斜率和截距
            #     slope = (obj_points[i] - obj_points[i - 1]) / (p_points[i] - p_points[i - 1])
            #     intercept = obj_points[i - 1] - slope * p_points[i - 1]
            #     model.pwl_constraints.add(
            #         model.obj_pwl[t] >= slope * model.p[t] + intercept
            #     )
        # model.sumed_obj = pyo.Expression(expr=sum(model.obj_pwl[t] for t in self.T))
        return p_points_list, obj_points_list

    def case_mg(self, conservativeness = 0.0, A=None, b=None, num_samples=100, noise_scale=0.05, batch_size=5, model_type='pretrainnet',
                    device='cpu'):
        self.build_mg_constraints(conservativeness=conservativeness, save_flag=True)
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
            # 近似器矩阵（功率+能量约束）
            lower_tri = np.tril(np.ones((self.T, self.T)))
            lower_tri_normalized = lower_tri / np.sqrt(lower_tri.sum(axis=1, keepdims=True))
            A_hat = np.vstack([
                np.eye(self.T),  # 功率上限
                -np.eye(self.T),  # 功率下限
                lower_tri_normalized,  # 累积能量下限
                -lower_tri_normalized  # 累积能量上限
            ])
            # 误差计算器
            errorcalculator = ErrorCalculator(
                original_model={'model': model},
                A_hat=A_hat,
                solver='gurobi',
            )
        else:
            errorcalculator = ErrorCalculator(
                original_model={'model': model},
                A_hat=A,
                b_hat=b,
                solver='gurobi',
            )
        self.errorcalculator = errorcalculator

        # 训练回调函数
        case_name = self.case_name
        figure_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\figures'
        os.makedirs(figure_folder, exist_ok=True)
        np.savetxt(f'{figure_folder}/b_{errorcalculator._iter}.csv', errorcalculator.b_hat, delimiter=',')
        np.savetxt(f'{figure_folder}/A_{errorcalculator._iter}.csv', errorcalculator.A_hat, delimiter=',')
        n_train = 5
        
        def training_callback(error_calculator, epoch=None):
            len_his = len(error_calculator.training_history['feas'])
            print(
                f"Iter {error_calculator._iter}: FeasErr={np.mean(error_calculator.training_history['feas'][-min(10, len_his):]):.2e}, "
                f"OptErr={np.mean(error_calculator.training_history['opt'][-min(10, len_his):]):.2e}")
            if error_calculator._iter % 50 == 1:
                np.savetxt(f'{figure_folder}/b_{error_calculator._iter}.csv', error_calculator.b_hat, delimiter=',')
                np.savetxt(f'{figure_folder}/A_{error_calculator._iter}.csv', error_calculator.A_hat, delimiter=',')


        # 训练参数配置
        trainer_configure = {
            "call_interval": 10,
            "training_callback": training_callback,
            "optimizer": 'SGD',
            "lr": 1e-3,
            "batch_size": 1,
            "scheduler": {"type": "StepLR", "step_size": 200, "gamma": 0.98},
            "n_cal": 1,
            "cal_feas": True,
            "cal_opt": True,
            "rate_opt_feas": 1.0
        }

        params_dict, param_count = pyomo_params_to_numpy(model)
        params = {  # 名字，初值，误差数据集
            'params_dict': params_dict,
            'dataloader': DataLoader(
                CaseData(),
                batch_size=batch_size,
                shuffle=True
            ),
            'count': param_count,
        }

        return {
            'casename': case_name,
            'A_hat': errorcalculator.A_hat,
            'b_hat': errorcalculator.b_hat,
            'errorcalculator': errorcalculator,
            'trainer_configure': trainer_configure,
            'params': params,
            'result_path': f'{PROJECT_ROOT}\\results\\{case_name}\\{model_type}_weights.pth',
            'n_train': n_train
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
        1: 随机在近似可行域A_hat var_proj<=b_hat内生成一个功率曲线P（随机生成全为正的c后调用errorcalculator里面的optimize_direction方法(in_approx = True)得到），
        输出成本CP和曲线P（注意P是var_proj反标幺化的结果）
        """
        p_min = self.pmin
        p_max = self.pmax
        # 随机生成全为负的c

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
    def disaggregate_with_cost(self, power_curve, baseline = None, flex_comp = 0.):
        """
        3: 调用errorcalculator里面的projection方法，给定一条微网总功率曲线，分解到每个设备的原可行域内，
        输出分解误差，以及分解之后的总功率曲线。
        """
        model = self.model
        p_min = self.pmin
        p_max = self.pmax
        # 标幺化输入功率曲线
        target = (power_curve - p_min) / (p_max - p_min)
        model.force_equal = pyo.Constraint(range(self.T), rule = lambda m,t: model.var_proj[t]==target[t])
        flex_revenue = np.sum(np.abs(power_curve-baseline)*flex_comp)

        model.dis_obj = pyo.Expression(expr = sum((model.var_proj[t]-target[t])**2 for t in range(self.T)))

        price = [0.5557] * 7 + [0.833] * 3 + [1.15] * 3 + [0.833] * 4 + [1.15] * 5 + [0.833] * 1 + [0.5557] * 1
        model.energy_cost = pyo.Expression(expr = sum(-price[t]*(model.var_proj[t]*(p_max[t] - p_min[t])+p_min[t]) for t in range(self.T)))
        alpha = 1e-12
        model.total_obj = pyo.Objective(expr=model.dis_obj+alpha*(model.mg_cost_expression+model.energy_cost-flex_revenue), sense=pyo.minimize)
        # model.total_obj = pyo.Objective(expr=(model.mg_cost_expression + model.energy_cost),
        #                                 sense=pyo.minimize)
        self.solver.solve(model)

        proj_curve_rated = np.array([model.var_proj[t]() for t in range(self.T)])
        proj_curve = proj_curve_rated* (p_max - p_min) + p_min
        dis_error_rated = model.dis_obj()
        dis_error = sum((proj_curve - power_curve)**2)

        mg_flex_cost = model.mg_cost_expression()
        mg_total_cost = model.mg_cost_expression()+model.energy_cost()-flex_revenue
        mg_baseline_cost = -sum(price[t]*baseline[t] for t in range(self.T))
        return mg_total_cost, mg_flex_cost, mg_baseline_cost, dis_error, proj_curve


    def test_disaggregation_error(self, n_test, num_samples=100, noise_scale=0.05, device='cpu'):
        """
        4: 输入n_test表示测试次数，每次迭代，随机生成一个近似可行域内的功率曲线，并且从CaseData中随机生成一组噪声数据，
        用errorcalculator里面的update_parameter方法更新原模型，然后计算这条功率曲线projection到原可行域内的可行性误差
        """
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
                    'p_PV_rated': (torch.randn([nPV, T], dtype=torch.float32, device=device) * noise_scale) * self.pv_base_tensor,
                    'p_CL_rated': (torch.randn([nCL, T], dtype=torch.float32, device=device) * noise_scale) * self.cl_base_tensor,
                    'temp_amb': torch.randn(T, dtype=torch.float32, device=device) * noise_scale * self.temp_amb_base_tensor,
                }

        results = []
        for i in range(n_test):
            # 使用generate_sample方法在近似可行域内生成功率曲线
            c = -np.random.rand(self.T)
            cost, power_curve = self.generate_sample( c = c, in_approx=True)
            if power_curve is None:
                continue
            cost0, _ = self.generate_sample(c = c, in_approx=False)
            print(cost0)
            if cost0 is None:
                continue
            objective_error = cost-cost0
            # 从CaseData中随机生成一组噪声数据
            case_data_obj = CaseData()
            case_data = case_data_obj.get_item(0)

            # 更新模型参数
            params_upd = {}
            params_dict, _ = pyomo_params_to_numpy(self.model)
            for name, data in case_data.items():
                if name in params_dict:
                    params_upd[name] = params_dict[name]['initial_value'] + data.detach().cpu().numpy()
            self.errorcalculator.update_parameters(params_upd)

            # 使用disaggregate方法计算这条功率曲线投影到原可行域内的可行性误差
            error, error_rated, projected_power_curve = self.disaggregate(power_curve)
            results.append({
                'feasibility_error': error,
                'feasibility_error_rated': error_rated,
                'power_curve': power_curve,
                'projected_power_curve': projected_power_curve,
                'objective_error': objective_error,
                'objective_error_rated': objective_error/np.abs(cost0),
            })


        return results



# 使用示例
if __name__ == "__main__":
    # 可设置随机种子以便复现结果
    mg = MicroGrid(seed=42)

    # 生成各种设备
    mg.gen_MT(1)  # 2个可控分布式发电
    mg.gen_PV(1)  # 3个光伏
    mg.gen_TL(1)  # 5个可转移负荷
    mg.gen_CL(1)  # 4个可中断负荷
    mg.gen_ESS(1)  # 2个储能
    mg.gen_HP(2)  # 6个热泵/中央空调

    # 构建微网聚合模型
    case = mg.case_mg()
