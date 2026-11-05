# Federated PEB (modified)：用户授权的退化数值处理

当前对比采用用户明确授权的辅助选解版本，图中标为 **Federated PEB (modified)**。它保留论文的完整仿射包含问题、PEB 基集和外层体积目标，但增加了以下数值约定；不能称为与原论文逐步完全相同的实现。

## 固定场景与评估

使用 `Simulator.cases.aggregation_case_T24.Aggregator`，`counts=(5,40,5)`、case seed=0、连续 TCL、T=24、deltaT=1。CL/TCL/PV 均使用实际设备的物理净注入功率域；TCL 保留真实温度递推和所有有效温度上下界。归一化、原九方法的模型、原始误差及成本数组均不改变。

训练预算上限为 200 秒，包含首次设备 LP、辅助 LP、梯度、体积积分、回溯和失败候选；只把物理场景生成及纯模型组装单列为准备。评估使用完全相同的 5000 个 RandomState(1) 标准正态方向，以及 2025-10-19 至 2026-10-18 的全部 365 个 UTC 日 AEMO VIC1 电价。价格及成本单位分别为 AUD/MWh、AUD/24 h，成本系数为 `-price*deltaT/1000`。

## 数值处理规则

1. 合并首期重复能量/功率参数，采用 raw `H0=[I;-I;L[1:];-L[1:]]`，共 94 行。仍表示相同的 PEB 集合族；参数化及投影度量与原 96 行坐标不同。初始 h=ones(94)。
2. 每设备先求论文原始主问题：`max trace(Gamma)`，约束 `Lambda>=0, Lambda H0=Ai Gamma, Lambda h+Ai gamma<=bi`。Gamma 是完整 24×24 自由矩阵，没有限制成对角、标量或正定。
3. 设主问题最优值为 p。再解辅助 LP，最小化 Gamma 非对角元素的 L1 范数，同时固定 `trace(Gamma)=max(0,p-tol)`，其中 `tol=1e-8+1e-9*abs(p)`。这是显式数值容差内的近最优选解，目标值并非严格等于原 argmax；不增加更大的 trace 损失或其它正则化。
4. 原始设备包含约束保持不变，实际采用的辅助解仍须通过独立 Farkas 及残差修正的包含证书。主、次 LP 均要求 OPTIMAL，分别保存原始/对偶证据；不以主问题解冒充失败的辅助解。
5. 对所选辅助 LP 的 simplex 分支做稀疏伴随求导，包含 trace target 随主问题最优值变化的链式项。这是分支导数，**没有认证唯一经典导数、Clarke 广义梯度或跨最优面连续性**；不对原奇异 KKT 直接求逆，也不使用伪逆或 ridge。
6. PEB 体积及其边界导数使用一维递推、分段线性 CDF 和梯形求积，训练网格为 1025，最终另以 2049 网格复核。这是针对相同积分的数值求积约定，并非原 Algorithm 1 的逐行 Riemann 实现。
7. 外层仍优化 `J=volume(U0)*abs(det(sum Gamma))`，原 J 的 Armijo 条件通过稳定 log 比值计算。接受完整、通过包含核验的候选；奇异聚合映射、数值失败或超时候选均不替换当前最佳完整点。
8. 下一轮回溯从 `min(1,2*上轮接受步长)` 开始，缩减比例 0.5、Armijo 常数 1e-4、最多 40 次回溯。只减少重复试算，不宣称解决 Gamma 最优面切换。原文未指定这些常数。

主、次解及相应对偶分开存储在 `devices/device_*.npz`；`certificate.npz` 保存基集和聚合仿射映射；`provenance.json` 保存执行源码、场景、预算及受保护旧输入的 SHA。固定覆盖本目录，不生成多套当前测试结果。

## 本次运行结果

| 项目 | 结果 |
| --- | ---: |
| 预算上限 | 200 秒 |
| 实际训练时间 | 173.087343 秒 |
| 纯模型准备 | 1.320634 秒 |
| 接受的参数更新（不含初始化） | 6 |
| 完整 50 设备候选评估（含初始化） | 74 |
| 设备级求解调用（通常含主、次两个 LP） | 4485 |
| 停止原因 | line_search_stalled |
| log J 增量 | 9.44696409988e-09 |
| 平均可行性误差 | 8.4875735042e-13 |
| 最大可行性误差 | 3.49038538908e-12 |
| 平均最优性误差 | 1.10404843968 |
| 最大最优性误差 | 3.59592862309 |
| 365 天平均成本偏差 | 76.34960760 AUD |

本次数值处理允许训练继续并返回有包含证书的结果，但目标改善仍约 1e-8，相比初始化没有可分辨的实质改善，不能以接受更新次数增加替代性能结论。回溯停滞后提前结束，没有空耗剩余预算，也没有声称已收敛。最终粗细网格体积差约 2.17e-6，不能用更小的目标变化宣称有效提升。

新模型已独立评估并加入现有两张 summary PDF，位置在 PEB outer 后，保持其余方法顺序、Times New Roman、箱型图加抖动、配色、轴范围及高度；正式绘图仅输出 PDF。

## 严格原式的证据与复现入口

原始 96 行、单次 max-trace LP 和附录 B 完整 KKT 的执行路径仍可通过 `--degeneracy-treatment strict` 调用。原场景 50 个原始 LP 均最优且包含证书通过，但 50 个完整 KKT 均存在明确非零零空间方向；五个 CL 还存在连续的 Gamma 最优面。原式 (38) 的直接逆在该初始化处不存在。这个结论不等于局部 LP 无解，也不排除另行规定的选解方法。

严格原式的证据、原始 50 个证书及核验记录已存为单个历史审计归档 `tmp/federated_aggregation/strict_initial_evidence.zip`，不再占用当前结果位置。用户在确认该障碍后授权采用本页所述数值处理；当前 provenance 明确标记 `paper_identical_implementation=false`。

```powershell
python -m Simulator.runners.run_federated_peb --time-limit 200 --grid-points 1025 --degeneracy-treatment lexicographic
python -m Simulator.runners.run_federated_peb --evaluate-only
```

`independent_recheck.json` 检查 50 台设备的真实物理域、主次问题最优性与包含证书、完整误差/成本数据一致性，并独立复算两个误差极值及三个成本日。原九方法和价格的 43 个受保护输入均逐字节核验不变，原始误差不做裁剪、删点或人工替换。单机顺序模拟未部署真实联邦网络或 secure summation，不宣称部署级隐私/通信性能。
