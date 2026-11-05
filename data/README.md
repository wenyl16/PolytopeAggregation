# 论文输入数据

原有 13 个输入文件全部保留，内容未修改。

| 路径 | 用途 |
|---|---|
| `aggregator_data/ZH_buildings.csv` | 场景构建所用建筑输入 |
| `profiles_data/profiles_data.npz` | 环境、负荷与光伏曲线 |
| `profiles_data/PV_samples.mat` | MATLAB 基线光伏输入 |
| `market_prices/caiso_poole_spot` | 表 II 的 2025 年 CAISO 电价、原始报价摘录、来源和哈希记录 |

电价来源及原始报价恢复方式见目录内 README 和审计 JSON。`Simulator/runners/baselines/matlab/data` 中另保留构造器直接调用的必要数据。计算入口见 [结果索引](../results/README.md)。
