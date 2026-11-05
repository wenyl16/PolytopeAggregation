"""Write the single retained Markdown summary for the two 500-device cases."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import zipfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from Simulator.paths import CODE_ROOT, RESULTS_ROOT


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def describe_training(training, horizon):
    stages = training["stages"]
    updates = "/".join(str(stage.get("epochs", stage["n_train"] + 1)) for stage in stages)
    rates = "/".join(f"{stage['lr']:g}" for stage in stages)
    return (f"T={horizon}采用{training['dtype'].removeprefix('torch.')}、"
            f"训练seed={training['training_seed']}，共{training['optimizer_updates']}次更新；"
            f"三阶段分别为{updates}次，初始学习率为{rates}。")


def write_summary():
    cases = []
    for horizon, relative in ((24, "aggregation/T24/large_scale/approx/peb"),
                              (96, "aggregation/T96/approx/peb")):
        directory = RESULTS_ROOT / relative
        summary = load(directory / "evaluation_summary.json")
        evaluation = load(directory / "evaluation_config.json")
        assert summary["status"] == evaluation["status"] == "completed"
        assert summary["samples"] == evaluation["samples"] == 5000
        assert summary["direction_seed"] == evaluation["direction_seed"] == 1
        assert summary["devices"] == 500 and summary["horizon"] == horizon
        assert summary["protected_training_files_unchanged"]
        model_path = directory / "final_polytope.npz"
        if not model_path.is_file():
            model_path = directory / "pretrainnet_weights.pth"
        assert hashlib.sha256(model_path.read_bytes()).hexdigest() == evaluation[
            "protected_training_file_sha256"][model_path.name]
        assert load(directory / "evaluation_extrema_recheck.json")["status"] == "passed"
        np.testing.assert_array_equal(np.load(directory / "evaluation_directions.npy"),
                                      np.random.RandomState(1).randn(5000, horizon))
        with np.load(directory / "evaluation_errors.npz", allow_pickle=False) as data:
            assert int(data["num_sample"]) == 5000
            assert int(data["native_successful_solves"]) == 20000
            for label, key in (("feasibility", "error_feas"), ("optimality", "error_opt")):
                values = data[key]
                assert values.shape == (5000,) and np.isfinite(values).all() and np.all(values >= 0)
                assert float(values.mean()) == summary[label]["mean"]
                assert float(values.max()) == summary[label]["max"]
        cases.append(summary)

    training = load(RESULTS_ROOT / "aggregation/T24/large_scale/approx/peb/training_config.json")
    assert training["status"] == "completed" and training["gradient_mode"] == "approx"
    assert training["optimizer_updates"] == 1000 and training["dtype"] == "torch.float64"
    assert training["counts"] == [100, 200, 200]
    seconds24 = float(training["elapsed_seconds"])
    training96_path = RESULTS_ROOT / "aggregation/T96/approx/peb/training_config.json"
    estimated96 = not training96_path.is_file()
    if not estimated96:
        training96 = load(training96_path)
        assert training96["status"] == "completed" and training96["gradient_mode"] == "approx"
        assert training96["horizon"] == 96 and training96["counts"] == [100, 200, 200]
        seconds96 = float(training96["elapsed_seconds"])
        assert np.isfinite(seconds96) and seconds96 > 0
        timing_note = ("**训练计时。** 两个场景均采用本次实际训练循环计时，包含回调与快照保存，"
                       "不含前置准备和后续误差评估。T=96已使用重新训练记录的耗时，"
                       "没有继续采用历史文件时间戳估算值。")
        training96_text = describe_training(training96, 96)
        source96 = ("- T=96：[训练配置](../aggregation/T96/approx/peb/training_config.json)、"
                    "[评估汇总](../aggregation/T96/approx/peb/evaluation_summary.json)、"
                    "[物理场景](../aggregation/T96/approx/peb/scenario.json)。")
    else:
        recovery = load(RESULTS_ROOT / "aggregation/T96/approx/peb/evaluation_historical_recovery.json")
        assert recovery["status"] == "passed" and recovery["recorded_updates"] == 1003
        weights = RESULTS_ROOT / "aggregation/T96/approx/peb/pretrainnet_weights.pth"
        assert hashlib.sha256(weights.read_bytes()).hexdigest() == recovery["final_network_recovery"]["weights_sha256"]
        prefix = "results/aggregation/T96/approx/peb/"
        with zipfile.ZipFile(CODE_ROOT.parent / "code.zip") as archive:
            initial = datetime(*archive.getinfo(prefix + "snapshots/A_0.csv").date_time)
            finish = datetime(*archive.getinfo(prefix + "error_history.pkl").date_time)
        seconds96 = (finish - initial).total_seconds()
        timing_note = (f"* **T=96训练时间为估算值。** 历史结果未保存精确计时；归档初始快照时间为"
                       f"{initial:%Y-%m-%d %H:%M:%S}，最终误差记录和权重时间为{finish:%Y-%m-%d %H:%M:%S}，"
                       f"相差{seconds96:.0f}秒。该文件时间跨度不能作为正式训练计时或独占测速。"
                       "T=24为训练循环的实际计时，包含回调与快照保存，不含前置准备和后续误差评估。")
        training96_text = ("T=96采用历史最终权重，共1003次更新；归档入口的三阶段为301/201/501次，"
                           "初始学习率为0.01/0.05/0.1。T=96的输入文件、设备参数、物理约束和初始支撑边界已经恢复核验。")
        source96 = ("- T=96：[历史记录](../aggregation/T96/approx/peb/migration_info.json)、"
                    "[评估汇总](../aggregation/T96/approx/peb/evaluation_summary.json)、"
                    "[恢复核验](../aggregation/T96/approx/peb/evaluation_historical_recovery.json)；"
                    "时间戳来源为项目根目录`code.zip`。")

    lines = [
        "# 500台连续设备场景：训练时间与误差汇总", "",
        "| 场景 | 训练时间（秒 / 分钟） | 平均可行性误差 | 最大可行性误差 | 平均最优性误差 | 最大最优性误差 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for horizon, seconds, result in zip((24, 96), (seconds24, seconds96), cases):
        timing = f"{seconds:.2f} / {seconds / 60:.2f}"
        if horizon == 96 and estimated96:
            timing = f"约 {seconds:.0f} / {seconds / 60:.2f}*"
        lines.append(f"| T={horizon}，500台连续设备 | {timing} | "
                     f"{result['feasibility']['mean']:.8g} | {result['feasibility']['max']:.8g} | "
                     f"{result['optimality']['mean']:.8g} | {result['optimality']['max']:.8g} |")
    lines += [
        "",
        timing_note,
        "",
        "**训练设置。** 两者均采用PEB初始化、近似梯度，设备组成为100 CL、200 TCL、200 PV，"
        "设备场景seed=0。" + describe_training(training, 24) + training96_text +
        "各阶段分别重新初始化SGD及StepLR调度器，阶段内每100次更新将学习率乘以0.98。",
        "",
        "**误差测试。** 每个场景使用`RandomState(1).randn(5000,T)`生成5000个标准正态方向。"
        "可行性误差是近似域支撑点到原始可行域的投影距离平方；最优性误差是原始域支撑点到近似域的投影距离平方。"
        "两项均在归一化功率坐标中计算平方欧氏距离，未开方、未除以T，不是百分比。"
        "T=24和T=96分别对24与96个分量求和。表中最大值为本次5000个方向的样本最大值。",
        "",
        "**数值核验。** 每方向实际求解两次LP、两次QP；Gurobi的Threads=1、Seed=0，"
        "FeasibilityTol=1e-9、OptimalityTol=1e-9、BarConvTol=1e-12。各场景的两项最大值对应方向均已通过原有Pyomo路径独立复核，"
        "误差未截断为零，评估期间训练文件哈希保持不变。",
        "",
        "**数据来源。**",
        "",
        "- T=24：[训练配置](../aggregation/T24/large_scale/approx/peb/training_config.json)、"
        "[评估汇总](../aggregation/T24/large_scale/approx/peb/evaluation_summary.json)。",
        source96,
        "- 两个模型目录的`evaluation_directions.npy`和`evaluation_errors.npz`分别保留全部方向与逐样本误差。"
        "可用[统一汇总入口](../../Simulator/testers/summarize_aggregation_scaling.py)重新生成本文件。",
    ]
    target = RESULTS_ROOT / "figures_summary/500_device_training_and_error_summary.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(write_summary())
