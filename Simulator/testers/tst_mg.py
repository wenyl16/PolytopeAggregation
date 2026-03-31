import os
import numpy as np
import pandas as pd

from Simulator.Plotter import ErrorVisualizer
from Simulator.Approximator import PreTrainNet
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from Simulator.cases.microgrid_case import MicroGrid
from Simulator import PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
model_type = 'pretrainnet'
mg = MicroGrid(seed=42)

mg.gen_MT(1)  # 2个可控分布式发电
mg.gen_PV(1)  # 3个光伏
mg.gen_TL(1)  # 5个可转移负荷
mg.gen_CL(1)  # 4个可中断负荷
mg.gen_ESS(1)  # 2个储能
mg.gen_HP(2)  # 6个热泵/中央空调
case = mg.case_mg(conservativeness = 0.0,device=device)

# 训练后计算可行性和目标误差
model_path = f'{PROJECT_ROOT}\\results\\microgrid\\pretrainnet_weights.pth'

# 加载预训练模型
A_init = mg.errorcalculator.A_hat
b_init = mg.errorcalculator.b_hat
model = PreTrainNet(A_init=A_init, b_init=b_init, device=device)
errorcalculator = mg.errorcalculator
#
# 加载训练好的权重
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"成功加载模型权重: {model_path}")
else:
    print(f"警告: 模型权重文件不存在: {model_path}")

# 获取训练后的A和b值
A_trained, b_trained = model()
A_trained = A_trained.detach().cpu().numpy()[0]  # 取出第一个批次的值
b_trained = b_trained.detach().cpu().numpy()[0]  # 取出第一个批次的值

# 更新errorcalculator的A和b值
mg.errorcalculator.update_polytope(A_hat=A_trained, b_hat=b_trained)
print("已更新errorcalculator的A和b值为训练后的值")

# 运行分解测试
n_test = 100 # 测试次数
results = mg.test_disaggregation_error(n_test=n_test)

# 统计结果
feasibility_errors = [r['feasibility_error'] for r in results if r['feasibility_error'] is not None]
objective_errors = [r['objective_error'] for r in results if r['objective_error'] is not None]
feasibility_errors_rated = [r['feasibility_error_rated'] for r in results if r['feasibility_error_rated'] is not None]
objective_errors_rated = [r['objective_error_rated'] for r in results if r['objective_error_rated'] is not None]
projected_curves = [r['projected_power_curve'] for r in results if r['projected_power_curve'] is not None]
original_curves = [r['power_curve'] for r in results if r['power_curve'] is not None]


# 创建结果DataFrame
df_results = pd.DataFrame({
    'feasibility_error': feasibility_errors if feasibility_errors else [None] * len(objective_errors),
    'feasibility_error_rated': feasibility_errors_rated if feasibility_errors_rated else [None] * len(objective_errors),
    'objective_error': objective_errors if objective_errors else [None] * len(feasibility_errors),
    'objective_error_rated': objective_errors_rated if objective_errors_rated else [None] * len(feasibility_errors_rated)
})

# 去除同时包含NaN的行
df_results = df_results.dropna(how='all')

# 创建结果保存目录
result_dir = f'{PROJECT_ROOT}\\results\\microgrid\\'
os.makedirs(result_dir, exist_ok=True)

# 保存为Excel文件
excel_path = f'{result_dir}decomposition_test_results.xlsx'
df_results.to_excel(excel_path, index=False)
print(f"结果已保存到: {excel_path}")


# visualizer = ErrorVisualizer()
# num_sample = 50
# folder_path = f'{PROJECT_ROOT}\\results\\microgrid\\figures'
# # folder_path = f'{PROJECT_ROOT}\\results\\aggregation\\figures'
# files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')],
#                key=lambda x: int(x.split('_')[1].split('.')[0]))
# iteration_list = []
# for i in range(0, len(files), 2):  # 假设每两个文件是一组(A和b)
#     if i + 1 < len(files):
#         print(i)
#         a_file = [f for f in files if f.startswith('A_')][i // 2]
#         b_file = [f for f in files if f.startswith('b_')][i // 2]
#         # 读取并打印
#         A = pd.read_csv(os.path.join(folder_path, a_file), header=None).values
#         b = pd.read_csv(os.path.join(folder_path, b_file), header=None).values.flatten()
#         i_iter = a_file.split('_')[1].split('.')[0]
#         iteration_list.append(i_iter)
#         errorcalculator.update_polytope(A_hat=A, b_hat=b)
#         visualizer.compute_errors(errorcalculator, num_sample=num_sample)
# result_path = f'{PROJECT_ROOT}\\results\\microgrid\\error_history.pkl'
# # result_path = f'{PROJECT_ROOT}\\results\\aggregation\\error_history.pkl'
# import pickle
# with open(result_path, "wb") as f:
#     pickle.dump(visualizer.error_history, f)


# folder_path = f'{PROJECT_ROOT}\\results\\aggregation\\data_cube\\'
# A = pd.read_csv(os.path.join(folder_path, 'A_cube.csv'), header=None).values
# b = pd.read_csv(os.path.join(folder_path, 'b_cube.csv'), header=None).values.flatten()
# case = agg.case_aggregator(A = A, b = b, model_type=model_type)
# errorcalculator = case['errorcalculator']
# visualizer.compute_errors(errorcalculator, num_sample=num_sample)
# # print(visualizer.error_history)
# result_path = f'{PROJECT_ROOT}\\results\\aggregation\\error_history_cube.pkl'
# import pickle
# with open(result_path, "wb") as f:
#     pickle.dump(visualizer.error_history, f)
#
# folder_path = f'{PROJECT_ROOT}\\results\\aggregation\\data_cube\\'
# A = pd.read_csv(os.path.join(folder_path, 'A_cube.csv'), header=None).values
# b = pd.read_csv(os.path.join(folder_path, 'b_cube.csv'), header=None).values.flatten()
# case = agg.case_aggregator(A = A, b = b, model_type=model_type)
# agg.count_complexity()
#
# folder_path = f'{PROJECT_ROOT}\\results\\aggregation\\discrete\\figures'
# # folder_path = f'{PROJECT_ROOT}\\results\\aggregation\\figures'
# files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')],
#                key=lambda x: int(x.split('_')[1].split('.')[0]))
# iteration_list = []
# i=len(files)-2
# a_file = [f for f in files if f.startswith('A_')][i // 2]
# b_file = [f for f in files if f.startswith('b_')][i // 2]
# # 读取并打印
# A = pd.read_csv(os.path.join(folder_path, a_file), header=None).values
# b = pd.read_csv(os.path.join(folder_path, b_file), header=None).values.flatten()
# errorcalculator.update_polytope(A_hat=A, b_hat=b)


# T = 24
# n_test = 10
# actions_rand = np.random.rand(n_test//2,T)
# actions_feas = np.array([
#     [0.        , 0.12227902, 0.13093164, 0.70359684, 0.9154062 , 0.93361783,
#      0.36337373, 0.06521021, 0.05479282, 0.63237062, 0.7912069 , 0.86521444,
#      0.05581981, 0.07096851, 0.87142061, 0.85252465, 0.88020974, 0.08594851,
#      0.08767512, 0.20478644, 0.09248264, 0.09145326, 0.97659704, 0.97859177],
#     [0.19100453, 0.12279355, 0.90858855, 0.109034  , 0.06544886, 0.86499908,
#      0.93772923, 0.0570253 , 0.86661712, 0.05768608, 0.88032965, 0.07912243,
#      0.0584331 , 0.86907194, 0.06687999, 0.20122116, 0.87751526, 0.82559947,
#      0.06194041, 0.90767717, 0.07999532, 0.08916374, 0.9709392 , 0.97510207],
#     [0.20748169, 0.92971945, 0.09937248, 0.11703826, 0.06858586, 0.94161029,
#      0.50972595, 0.91208238, 0.06187957, 0.29967297, 0.0834183 , 0.86928718,
#      0.43591326, 0.86098374, 0.06458626, 0.85196676, 0.8692467 , 0.08364255,
#      0.0695082 , 0.90358179, 0.0898512 , 0.97320738, 0.10193014, 0.7087338 ],
#     [0.        , 0.15014873, 0.11172917, 0.91623611, 0.27344514, 0.94273403,
#      0.03857266, 0.91022953, 0.87379364, 0.06065246, 0.86626798, 0.06748514,
#      0.07085175, 0.86053583, 0.47196061, 0.8525117 , 0.08531828, 0.07748523,
#      0.9028976 , 0.06624319, 0.97188003, 0.09374322, 0.36671871, 0.983121  ],
#     [0.        , 0.91357289, 0.07732961, 0.90694771, 0.05016652, 0.89841224,
#      0.04308278, 0.07009375, 0.88105151, 0.06470593, 0.32351372, 0.87187624,
#      0.86556066, 0.0795814 , 0.8879796 , 0.09712649, 0.0847747 , 0.8858776,
#      0.06859193, 0.89445903, 0.0874217 , 0.971029  , 0.10094444, 0.75249288]
# ])
# actions = np.vstack([actions_feas,actions_rand])
# import time
# import pyomo.environ as pyo
# errorcalculator.original_model.given_value_max = pyo.Constraint(range(T), rule=lambda model, t: model.var_proj[t] <= model.x_apx[t]+1e-5)
# errorcalculator.original_model.given_value_min = pyo.Constraint(range(T), rule=lambda model, t: model.var_proj[t] >= model.x_apx[t]-1e-5)
# errorcalculator.original_model.min_error.deactivate()
# errorcalculator.original_model.min_direction.deactivate()
# errorcalculator.original_model.zero = pyo.Objective(expr=0,sense=pyo.minimize)
# judge_start = time.perf_counter()
# for action in actions:
#     # print(np.all(A@action<=b+1e-5))#统计近似可行域的判断时间
#
#     # 统计原始可行域的判断时间
#     for t in range(T):
#         errorcalculator.original_model.x_apx[t].value = action[t]
#     print(errorcalculator.solver.solve(errorcalculator.original_model))
# judge_end = time.perf_counter()
# print('总耗时',(judge_end-judge_start)/n_test)

# result_path = f'{PROJECT_ROOT}\\results\\aggregation\\discrete\\error_history.pkl'
# result_path = f'{PROJECT_ROOT}\\results\\aggregation\\error_history.pkl'
# import pickle
# with open(result_path, "wb") as f:
#     pickle.dump(visualizer.error_history, f)
