"""Produce the requested table and independently audit every ESS energy ledger."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path('code').resolve()))
from Simulator.testers.evaluate_microgrid_nonconvexity import load_case, write_json

folder = Path('code/results/microgrid/scale3/nonconvexity')
meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
assert meta['status'] == 'completed' and meta['completed'] == 5000
assert meta['reporting_threshold_percent'] == 0.01
mg = load_case(Path('code/results/microgrid/scale3/training'), device_scale=3)
with np.load(folder / 'endpoints.npz') as z:
    names = z['variable_names'].tolist()
    groups = {'raw_endpoints': z['raw_primal']}
with np.load(folder / 'projections.npz') as z:
    groups['formal_projections'] = z['primal']
with np.load(folder / 'controls.npz') as z:
    groups.update({key: z[key] for key in ('endpoint_primal', 'feasible_witnesses', 'primal')})
checks = {}
for label, primal in groups.items():
    maximum = 0.0
    for j, ess in enumerate(mg.ESS_list):
        ch = primal[:, [names.index(f'p_chg_ESS[{j},{t}]') for t in range(24)]]
        dis = primal[:, [names.index(f'p_dis_ESS[{j},{t}]') for t in range(24)]]
        soc = primal[:, [names.index(f'e_ESS[{j},{t}]') for t in range(25)]]
        expected = np.c_[np.full(len(primal), ess['init_SOC']),
                         ess['init_SOC'] + np.cumsum(
                             mg.deltaT * (ess['eta_chg'] * ch - dis / ess['eta_dis'])
                             / ess['E_rated'], axis=1)]
        maximum = max(maximum, float(np.max(np.abs(soc - expected))))
        assert np.min(soc) >= ess['SOC_min'] - 1e-8
        assert np.max(soc) <= ess['SOC_max'] + 1e-8
        assert np.min(soc[:, -1]) >= ess['init_SOC'] - 1e-8
    assert maximum < 1e-8
    checks[label] = {'passed': True, 'ESS_trajectories_checked': len(primal) * 3,
                     'max_cumulative_SOC_residual': maximum}
write_json(folder / 'independent_ESS_energy_verification.json',
           {'passed': True, 'physical_intervals': 24, 'state_boundaries': 25,
            'checks': checks, 'all_first_and_last_interval_powers_included': True})

rows = []
for key, label in [('local', '局部连线'), ('global', '全局连线'),
                   ('multi', '多点组合'), ('overall', '总体')]:
    group = summary['groups'][key]
    c = group['classification_counts']
    row = {'探针类型': label, '样本数': group['count']}
    for field, name in [('mean', '均值'), ('median', '中位数'), ('p95', 'P95'), ('max', '最大gap')]:
        row[f'{name}下界(%)'] = group['lower_percent'][field]
        row[f'{name}上界(%)'] = group['upper_percent'][field]
    row['超过0.01%的点数'] = c.get('detected_gap', 0)
    row['超过0.01%的比例(%)'] = 100 * row['超过0.01%的点数'] / group['count']
    row['阈值判定未确定数'] = c.get('undetermined', 0)
    row['最优性精度未达到数'] = group['convergence_counts']['not_converged']
    rows.append(row)
pd.DataFrame(rows).to_csv(folder / 'gap_table.csv', index=False, encoding='utf-8-sig')

def value(row, name):
    lo, hi = row[f'{name}下界(%)'], row[f'{name}上界(%)']
    low_text, high_text = f'{lo:.5f}', f'{hi:.5f}'
    return high_text if low_text == high_text else f'[{low_text}, {high_text}]'

text = ['# 三倍设备场景非凸性诊断', '',
        '24时段；3 MT、3 PV、3 TL、3 CL、3 ESS、6 HP，共21台设备；360个二进制变量。',
        '单台参数与原始7设备算例完全一致；每台ESS为100 kW/250 kWh。固定名义场景，无随机扰动，conservativeness=0；不重新训练。',
        '使用修复MT最短运行时间、ESS全24时段能量递推后的现有设备模型。HP温度动态保持现有实现。',
        '1000个支撑方向(seed=0)生成端点，5000个凸组合探针(seed=1)，100个零间隙对照。',
        '距离单位为每时段固定功率跨度归一化的RMS百分比：100*sqrt(mean(((P-Q)/D)^2))。', '',
        '| 探针类型 | 样本数 | 均值(%) | 中位数(%) | P95(%) | 最大gap(%) | >0.01%点数 | 比例 | 未确定数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    text.append(f"| {r['探针类型']} | {r['样本数']} | {value(r, '均值')} | {value(r, '中位数')} | {value(r, 'P95')} | {value(r, '最大gap')} | {r['超过0.01%的点数']} | {r['超过0.01%的比例(%)']:.2f}% | {r['阈值判定未确定数']} |")
text += ['', '区间来自求解器目标上下界，非精确零值不修改为零。最大距离下界仅为全域非凸度的抽样下界，不是全域上界；检出比例仅描述本次抽样。',
         '完整原始整数变量在每次正式投影中均自由优化。独立原始约束、MT正向最短运行时间及ESS累计能量检查见同目录验证报告。',
         f"结果指纹：`{meta['fingerprint']}`。"]
(folder / 'gap_table.md').write_text('\n\n'.join(text[:8]) + '\n\n' + '\n'.join(text[8:]) + '\n', encoding='utf-8')
print(json.dumps({'table': rows, 'ESS_validation': checks}, ensure_ascii=False))
