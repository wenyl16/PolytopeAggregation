from pathlib import Path
import json
import numpy as np
import pandas as pd

base = Path('code/results/microgrid/scale3')
old, out = base / 'nonconvexity', base / 'nonconvexity_no_min_uptime'
read = lambda path: json.loads(path.read_text(encoding='utf-8'))
baseline, paired, fresh = [read(path) for path in
                           (old / 'summary.json', out / 'paired_summary.json', out / 'summary.json')]
assert read(out / 'metadata.json')['status'] == 'completed'
assert read(out / 'independent_storage_verification.json')['passed']
assert paired['physical_validation']['passed']
previous_samples, paired_samples = [pd.read_csv(path) for path in
                                     (old / 'samples.csv', out / 'paired_samples.csv')]
worst = int(previous_samples.lower_percent.idxmax())
rows = []
for strategy, sampling, data in [('保留4时段最短运行时间', '原5000条轨迹', baseline),
                                ('移除最短运行时间', '原5000条轨迹（配对）', paired),
                                ('移除最短运行时间', '重新支撑采样5000条轨迹', fresh)]:
    for group_name, label in [('overall', '总体'), ('local', '局部连线'),
                              ('global', '全局连线'), ('multi', '多点组合')]:
        group = data['groups'][group_name]
        counts = group['classification_counts']
        row = {'策略': strategy, '测试轨迹': sampling, '组别': label, '样本数': group['count']}
        for field, title in [('mean', '均值'), ('median', '中位数'), ('p95', 'P95'), ('max', '最大gap')]:
            for side, side_label in [('lower', '下界'), ('upper', '上界')]:
                row[f'{title}{side_label}(%)'] = group[f'{side}_percent'][field]
        row['超过0.01%点数'] = counts.get('detected_gap', 0)
        row['超过0.01%比例(%)'] = 100 * row['超过0.01%点数'] / group['count']
        row['未确定数'] = counts.get('undetermined', 0)
        row['最优性未达到数'] = group['convergence_counts']['not_converged']
        rows.append(row)
pd.DataFrame(rows).to_csv(out / 'min_uptime_comparison.csv', index=False, encoding='utf-8-sig')

def display(row, name):
    lo, hi = [row[f'{name}{side}(%)'] for side in ('下界', '上界')]
    l, u = f'{lo:.5f}', f'{hi:.5f}'
    return u if l == u else f'[{l}, {u}]'

text = ['# 三倍场景：MT最短运行时间消融', '',
        '场景：3 MT、3 PV、3 TL、3 CL、3 ESS、6 HP，24时段，360个二进制变量。',
        '仅移除3台MT共72条最短连续运行时间约束；最低出力20 kW、最高出力100 kW、启停状态转移和全部其他设备约束保留。每台ESS仍为100 kW/250 kWh。',
        '名义场景、无随机扰动、conservativeness=0，原始功率跨度归一化；阈值为0.01%。不重新训练，不松弛整数变量。', '',
        '主对比复用上一轮完全相同的5000个凸组合目标，原端点均在移除约束后的模型中重新验证可行。补充试验重新查询1000个支撑方向(seed=0)，生成5000个凸组合(seed=1)。', '',
        '| 策略 | 测试轨迹 | 均值(%) | P95(%) | 最大gap(%) | >0.01% | 未确定 |',
        '|---|---|---:|---:|---:|---:|---:|']
for row in rows:
    if row['组别'] != '总体':
        continue
    text.append(f"| {row['策略']} | {row['测试轨迹']} | {display(row,'均值')} | {display(row,'P95')} | {display(row,'最大gap')} | {row['超过0.01%点数']}/5000（{row['超过0.01%比例(%)']:.2f}%） | {row['未确定数']} |")
original_mean = baseline['groups']['overall']['upper_percent']['mean']
new_mean = paired['groups']['overall']['upper_percent']['mean']
original_max = baseline['groups']['overall']['upper_percent']['max']
new_max = paired['groups']['overall']['upper_percent']['max']
text += ['', f'配对均值减少{100*(1-new_mean/original_mean):.2f}%，样本最大值减少{100*(1-new_max/original_max):.2f}%。',
         f'上一轮最大gap的样本索引（从0开始）为{worst}，原gap为{previous_samples.loc[worst,"lower_percent"]:.8f}%，移除约束后距离上界为{paired_samples.loc[worst,"upper_percent"]:.8f}%。',
         '新采样和配对结果不可混称同一批测试点。完整分组及上下界保存在CSV中；求解与数值残差检查见验证报告。',
         f"配对结果未达到最优性精度的点数：{paired['groups']['overall']['convergence_counts']['not_converged']}；新采样未达到最优性精度的点数：{fresh['groups']['overall']['convergence_counts']['not_converged']}。阈值判定未确定数另列于表中；达到时间上限的点仍保留距离上下界，未人为改为零。",
         '结论：移除最短运行时间显著减小本次抽样非凸间隙，但仍有正间隙，不能断言整个聚合域凸。最大距离下界仅为全域非凸度的抽样下界，检出比例不是实际运行不可行概率。',
         '设备模型沿用前一轮实现；HP温度动态未在本次消融中修改。']
(out / 'min_uptime_comparison.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
print(json.dumps([r for r in rows if r['组别']=='总体'], ensure_ascii=False))
