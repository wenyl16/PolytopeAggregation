# PolytopeAggregation 代码

命令在包含 `Simulator`、`data`、`results` 的目录执行。原工作区是 `code`，OA 副本是 `code-OA/PolytopeAggregation`。

## 按算例组织的 testers

`testers` 现有八个算例文件。同一算例的评估、核验及数据汇总都放在对应文件中；梯度对照还包含该实验的成对训练流程。

| 算例文件 | 对应结果目录（相对 results） | 入口 |
|---|---|---|
| `case_gradients.py` | `aggregation/T24/gradient_comparison`，图 6 | `train`、`evaluate`、`check`、`recheck`、`summarize` |
| `case_training_stages.py` | `aggregation/T24/stage_comparison`，图 7 | 直接运行，选项见 `--help` |
| `case_training_budgets.py` | `aggregation/T24/training_budget_comparison`，图 8 | 直接运行，默认 PEB |
| `case_methods.py` | `aggregation/T24/comparisons`，图 9、表 II | `evaluate`、`market`、`refresh` |
| `case_scaling.py` | `aggregation/T24/large_scale` 和 `aggregation/T96`，表 III | `evaluate`、`summarize` |
| `case_microgrid_uncertainty.py` | `microgrid/disaggregation`，图 11 的六组分布 | 直接运行，选项见 `--help` |
| `case_microgrid_scale.py` | `microgrid/scale3`，21 台设备的评估、误差百分比与核验 | `evaluate`、`verify`、`summarize` |
| `case_microgrid_nonconvexity.py` | `microgrid/nonconvexity`、`microgrid/scale3/nonconvexity`、`nonconvexity_no_min_uptime`，表 IV | `evaluate`、`ablation`、`verify`、`verify-storage`、`summarize`、`scale3-summary`、`ablation-summary` |

仅另保留两个跨算例文件：`polytope_evaluation.py` 是共用的 Pyomo/Gurobi 误差与成本计算；`paper_tables.py` 汇总全篇表 I–V。加上包初始化文件，testers 共 11 个 Python 文件。各算例中的子命令使用 `子命令 --help` 查看参数。

```powershell
python -B -m Simulator.testers.case_gradients train --help
python -B -m Simulator.testers.case_training_stages --help
python -B -m Simulator.testers.case_training_budgets --help
python -B -m Simulator.testers.case_methods evaluate --metric error --help
python -B -m Simulator.testers.case_methods market --help
python -B -m Simulator.testers.case_scaling evaluate --horizon 96 --help
python -B -m Simulator.testers.case_microgrid_uncertainty --help
python -B -m Simulator.testers.case_microgrid_scale evaluate --help
python -B -m Simulator.testers.case_microgrid_nonconvexity evaluate --help
```

微网规模算例默认读取 `microgrid/scale3/training`，评估输出放在选中训练目录同级的 `disaggregation`，可显式指定其他训练目录和输出位置。新训练或评估会写入对应结果，运行前请查看覆盖和续跑选项。

保存数据的汇总可单独执行，不需要重新训练：

```powershell
python -B -m Simulator.testers.paper_tables
python -B -m Simulator.testers.case_scaling summarize
python -B -m Simulator.testers.case_microgrid_scale summarize
python -B -m Simulator.testers.case_microgrid_nonconvexity scale3-summary
python -B -m Simulator.testers.case_microgrid_nonconvexity ablation-summary
```

## 其余目录

| 位置 | 用途 |
|---|---|
| `Approximator.py` | 本文的多面体拟合算法 |
| `paths.py` | 共享路径与历史来源路径映射 |
| `cases` | 设备模型、微网计数求解器及模型约束测试 |
| `runners/train_aggregation.py` | 连续资源 PB/PEB 基础训练 |
| `runners/train_scaling.py --horizon 24/96` | 500 台连续资源训练 |
| `runners/train_microgrid.py train/scale` | 微网训练及规模实验全流程 |
| `runners/baselines` | 其它对比方法及 MATLAB 基线 |
| `drawers` | 绘图，默认输出 `results/figures` |

基线入口是 `Simulator.runners.baselines.run`、`run_federated_peb`、`run_taheri2022`。`legacy_model_reference.py` 是只读旧模型参考，用于 MATLAB 基线约束一致性核验。

模型约束测试与设备定义放在一起：

```powershell
python -B -m unittest Simulator.cases.test_microgrid_constraints
python -B -m unittest Simulator.runners.baselines.test_taheri2022
```

## 绘图与复现边界

```powershell
python -B -m Simulator.drawers.refresh_original_figures
python -B -m Simulator.drawers.plot_gradient_comparison
python -B -m Simulator.drawers.plot_comparison results/aggregation/T24/comparisons --metric error
python -B -m Simulator.drawers.plot_training_stages results/aggregation/T24/stage_comparison
python -B -m Simulator.drawers.plot_training_budgets results/aggregation/T24/training_budget_comparison
python -B -m Simulator.drawers.plot_microgrid_shift results/microgrid/disaggregation
```

原始数值、模型、图片及历史元数据未随文件合并改写。严格来源核验仍会拒绝实际执行版本与当前合并代码的哈希差异；历史核验记录和源码快照继续保留。旧文件入口已删除，使用上面的算例入口。旧汇总脚本的原字节单独保留为历史来源记录，不作为当前运行入口。

现存旧图、部分正文耗时及 T96 原始初始化的来源限制见 [结果索引](results/README.md)。Python 依赖见 [requirements.txt](Simulator/requirements.txt)；训练和求解需要 Gurobi 及许可证，Taheri 分类器需要 CVXPY/Clarabel，MATLAB 基线需要 MATLAB/YALMIP/CPLEX。
