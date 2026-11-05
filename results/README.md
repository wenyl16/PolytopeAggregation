# 论文结果与复现索引

整理依据：`OptimalPolytope_tracked_change.pdf`，以及作者指定的 `figs/results_original`、`figs/results_revision`。整理日期：2026-11-03。

所有指定图片统一在 [figures](figures/)，共 12 份 PDF，逐字节保留作者指定文件。绘图代码位于 `Simulator/drawers`；训练、测试、数值、权重、快照和求解日志保留在各实验目录。只在表格或文字中报告的实验也保留。

| 论文结果 | 保留位置（相对 results） | 主要模块（前缀 Simulator） |
|---|---|---|
| 图 4、5、表 I：PB/PEB 训练及几何变化 | `aggregation/T24/exact`，包括比较程序引用的子目录 | `runners.train_aggregation`；`drawers.plot_original_training`；`drawers.visualize_A_matrix_evolution` |
| 图 6：十个种子的精确／近似梯度比较 | `aggregation/T24/gradient_comparison` | `testers.case_gradients train`；`testers.case_gradients`；`drawers.plot_gradient_comparison` |
| 图 7：单／双／三阶段训练 | `aggregation/T24/stage_comparison` | `testers.case_training_stages`；`drawers.plot_training_stages` |
| 图 8：PEB 训练预算比较 | `aggregation/T24/training_budget_comparison/peb` 及共同输入与核验文件 | `testers.case_training_budgets`；`drawers.plot_training_budgets` |
| 图 9、表 II：方法误差与电价成本 | `aggregation/T24/comparisons`、`runners.baselines.polytope`、`exact`；同级 data 的 CAISO 输入 | `runners.runners.baselines.polytope.run`；`runners.runners.baselines.polytope.run_federated_peb`；`testers.case_methods evaluate`；`testers.case_methods market`；`drawers.plot_comparison` |
| 表 III：500 台资源，T24/T96 | `aggregation/T24/large_scale/approx/peb`、`aggregation/T96/approx/peb` | `runners.train_scaling`；`runners.train_scaling`；`testers.case_scaling evaluate` |
| 作者指定的旧大规模训练图 | `aggregation/T24/large_scale/exact/peb`，不同于表 III 的训练 | `drawers.plot_original_training` |
| 表 IV：三种非凸性场景 | `microgrid/nonconvexity`、`microgrid/scale3/nonconvexity`、`microgrid/scale3/nonconvexity_no_min_uptime` | `testers.case_microgrid_nonconvexity evaluate`；`testers.case_microgrid_nonconvexity ablation`；独立核验入口 |
| 图 10、表 V、7 台设备训练耗时 | `microgrid/training`；旧图来源限制见下文 | `runners.train_microgrid`；`drawers.plot_original_training` |
| 图 11：六种不确定性条件 | `microgrid/disaggregation` 及其训练模型 | `testers.case_microgrid_uncertainty`；`drawers.plot_microgrid_shift` |
| 正文：21 台设备、360 个二进制变量的耗时及误差 | `microgrid/scale3` 的训练、分解、核验和源码快照 | `runners.train_microgrid`；`testers.case_microgrid_scale evaluate`；`testers.case_microgrid_scale summarize` |
| 正文：消元生成 4,764,694 条约束后触发行数保护 | `microgrid/taheri2022/run_20261028_budget1000_retry2` | `runners.runners.baselines.polytope.run_taheri2022`、`taheri_data`、`taheri_geometry` |

## 表格复算

在包含 `Simulator`、`data`、`results` 的目录执行：

```powershell
python -B -m Simulator.testers.paper_tables
```

输出 [paper_tables.md](figures/paper_tables.md)、[JSON](figures/paper_tables.json)、[LaTeX](figures/paper_tables.tex)。表 II–IV 从保存数据复算并核对论文报告精度；表 I/V 导出实际保存的阶段及等效权重。表 IV 去除最短开机约束这一行使用 **paired_samples.csv**；普通 samples.csv 属于另一组辅助核验。

## 已有的历史来源缺口

以下问题在本次清理前已存在，没有通过修改原始数据来统一。

1. `mg_training_trajectory.pdf` 的历史记录对应保守参数 0.02；现存训练模型是后来的 0.05。旧图及来源哈希保留，但未找到哈希匹配的旧训练历史，不能保证以当前模型精确重画旧曲线。
2. 梯度耗时 197.215/195.594 秒来自作者提供的记录，现存逐种子日志均值不同。预算实验正文的 1000/2000 次、197.2/390.7 秒也不同于现存的 1003/2006 次、225.768/461.017 秒。现存梯度图管线包含 10000 个方向，正文写 5000。详见 [论文结果审计](provenance/paper_results_audit.md) 和 [作者耗时来源](provenance/manuscript_runtime_provenance.json)。
3. T96 历史训练原始初始化来自已缺失的 `code.zip`。最终模型、完整训练历史和表 III 评估保留；精确复跑仍需哈希匹配的原始 `A_0.csv`/`b_0.csv`，通过 `--raw-initialization` 或 `--initialization-archive` 提供。当前归一化快照不能充当原始输入。
4. 指定的旧五方法 `comparison.pdf` 无字节一致的现存导出；指定 PDF、相关数据及绘图代码均保留，旧图数值来源尚未完全确认。另三张版本不同的修回图也以作者指定 PDF 为准。

逐图来源和哈希见 [figure_sources.json](figures/figure_sources.json)，取舍与修改记录见 [cleanup_manifest.json](cleanup_manifest.json)。历史元数据与运行时源码快照保留原字节，其中旧绝对路径表示当时来源，不是当前执行入口。

## 清理和验证

删除未引用的 lambda 数值扫描、PB 专用训练预算实验、Taheri 前两次已替代尝试、非正文重复图片、IDE 配置和缓存。原有 13 个输入文件全部保留。旧图片汇总目录已移除，活跃绘图入口统一输出到 figures。

本次执行了保存数据的表格复算、绘图入口检查、源码语法检查和复制哈希核验，没有重新训练或求解。当前环境缺少 gurobipy 和 cvxpy；完整求解还需相应依赖及许可证。绘图入口能运行不代表历史来源缺口已解决。

## ??????????

?? testers ???? case ????????????????????????????????????????????????????????????????????????? polytope_evaluation.py?????????????? paper_tables.py?

?????????????????????? [????](../Simulator/README.md)??????????????????????????????????????????????????????????????? case ???????????????
