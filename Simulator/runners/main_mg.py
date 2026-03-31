import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet,BiasNet,FullNet,compute_loss,Trainer
import torch
from Simulator.cases.microgrid_case import MicroGrid
from Simulator import  PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
model_type = 'pretrainnet'
mg = MicroGrid(seed=42)
# 生成各种设备
mg.gen_MT(1)  # 2个可控分布式发电
mg.gen_PV(1)  # 3个光伏
mg.gen_TL(1)  # 5个可转移负荷
mg.gen_CL(1)  # 4个可中断负荷
mg.gen_ESS(1)  # 2个储能
mg.gen_HP(2)  # 6个热泵/中央空调
case = mg.case_mg(conservativeness = 0.00, device=device)

model   = PreTrainNet(case['A_hat'],case['b_hat'],device = device).to(device)

trainer = Trainer(
    model=model,
    error_calculator=case['errorcalculator'],
    compute_loss=compute_loss,
    is_uncertain=True,
    parallel=False,
    params_data=case['params']
)

import time
training_start = time.time()
trainer.configure(**case['trainer_configure'])
trainer.configure(
    #feas_tol = 1e-1,
                  rate_opt_feas = 1.0,
                  lr = 5e-2,
)
trainer.initialize()
n_train = 29
trainer.train(n_train = n_train)

trainer.configure(**case['trainer_configure'])
trainer.configure(
    #feas_tol = 1e-1,
                  rate_opt_feas = 1e-2,
                  lr = 1e-1,
)
trainer.initialize()
trainer.train(n_train = 29)

trainer.configure(**case['trainer_configure'])
trainer.configure(
    #feas_tol = 1e-1,
                  rate_opt_feas = 1e-6,
                  lr = 2e-1,
)
trainer.initialize()
trainer.train(n_train = 39)

torch.save(model.state_dict(), case['result_path'])
training_end = time.time()
print('总耗时',training_end-training_start)

# 保存误差序列到pkl文件
# import pickle
# error_history = {
#     'feasibility_errors': trainer.error_calculator.training_history['feas'],
#     'optimality_errors': trainer.error_calculator.training_history['opt']
# }
# error_save_path = f'{PROJECT_ROOT}\\results\\{case["casename"]}\\error_history.pkl'
# os.makedirs(os.path.dirname(error_save_path), exist_ok=True)
# with open(error_save_path, 'wb') as f:
#     pickle.dump(error_history, f)
# print(f'误差序列已保存到: {error_save_path}')
