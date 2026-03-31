
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet,BiasNet,FullNet,compute_loss,Trainer
import torch
from Simulator.cases.aggregation_case import Aggregator
from Simulator import  PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
model_type = 'pretrainnet'
agg = Aggregator(seed=0, discrete_rate = 0.0)

# 生成设备
agg.gen_CL(5)  # 100个可中断负荷
agg.gen_TCL(40)  # 200个温度控制负荷
# agg.gen_ESS(100)  # 100个储能
agg.gen_PV(5)  # 200个光伏

import pandas as pd
folder_path = f'{PROJECT_ROOT}\\results\\aggregation\\data_cube\\'
A = pd.read_csv(os.path.join(folder_path, 'A_cube.csv'), header=None).values
b = pd.read_csv(os.path.join(folder_path, 'b_cube.csv'), header=None).values.flatten()
case = agg.case_aggregator(A = A, b = b, model_type=model_type, templete='PB')

if model_type=='pretrainnet':
    # n_train = case.get('n_train', 1000)
    model   = PreTrainNet(case['A_hat'],case['b_hat'],device = device).to(device)
else:
    # n_train = case.get('n_train',1000)
    model = PreTrainNet(case['A_hat'], case['b_hat'])
    model.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\{case['casename']}\\pretrainnet_weights.pth'))
    A_pretrained, b_pretrained = model()
    b_pretrained = b_pretrained[0].detach().cpu().numpy()
    if model_type == 'biasnet':
        A_pretrained = A_pretrained[0].detach().to(device)
        case['trainer_configure'].update(A_pretrained = A_pretrained)
        model = BiasNet(dim_theta=case['params']['count'], b_init=b_pretrained,device = device).to(device)
    elif model_type == 'fullnet':
        A_pretrained = A_pretrained[0].detach().cpu().numpy()
        model= FullNet(dim_theta = case['params']['count'], A_init=A_pretrained,b_init = b_pretrained,device = device).to(device)

trainer = Trainer(
    model=model,
    error_calculator=case['errorcalculator'],
    compute_loss=compute_loss,
    is_uncertain=False,
    parallel=False,
    params_data=case['params']
)

import time
training_start = time.time()
trainer.configure(**case['trainer_configure'])
trainer.configure(
    #feas_tol = 1e-1,
                  rate_opt_feas = 1.0,
                  lr = 2e-2,
)
trainer.initialize()
trainer.train(n_train = 300)

trainer.configure(**case['trainer_configure'])
trainer.configure(
    #feas_tol = 1e-1,
                  rate_opt_feas = 1e-2,
                    lr = 1e-1,
)
trainer.initialize()
trainer.train(n_train = 200)

trainer.configure(**case['trainer_configure'])
trainer.configure(
    #feas_tol = 1e-1,
                  rate_opt_feas = 1e-6,
                  lr = 2e-1,
)
trainer.initialize()
trainer.train(n_train = 500)

training_end = time.time()
print('总耗时',training_end-training_start)
torch.save(model.state_dict(), case['result_path'])
# 保存误差序列到pkl文件
import pickle
error_history = {
    'feasibility_errors': trainer.error_calculator.training_history['feas'],
    'optimality_errors': trainer.error_calculator.training_history['opt']
}
error_save_path = f'{PROJECT_ROOT}\\results\\{case["casename"]}\\error_history.pkl'
os.makedirs(os.path.dirname(error_save_path), exist_ok=True)
with open(error_save_path, 'wb') as f:
    pickle.dump(error_history, f)
print(f'误差序列已保存到: {error_save_path}')