"""
精炼的训练误差可视化代码 - Aggregation Case
生成两张高质量SVG图表
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from holoviews.plotting.bokeh.styles import font_size

from Simulator import PROJECT_ROOT

# 设置出版级样式
font_size = 20
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = font_size  # 字体加大一倍（从14到28）
plt.rcParams['axes.labelsize'] = font_size  # 加大一倍（从16到32）
plt.rcParams['axes.titlesize'] = font_size  # 加大一倍（从18到36）
plt.rcParams['xtick.labelsize'] = font_size  # 加大一倍（从12到24）
plt.rcParams['ytick.labelsize'] = font_size  # 加大一倍（从12到24）
plt.rcParams['legend.fontsize'] = font_size  # 加大一倍（从12到24）
plt.rcParams['axes.linewidth'] = 1.5  # 边缘线加粗一倍（默认1到2）

# 颜色方案
COLOR_FEAS = '#2E86AB'  # 蓝色 - 可行性误差
COLOR_OPT = '#A23B72'   # 紫红色 - 最优性误差

def load_error_data(error_path):
    """加载误差数据"""
    with open(error_path, 'rb') as f:
        data = pickle.load(f)
    return np.array(data['feasibility_errors']), np.array(data['optimality_errors'])

def apply_threshold(errors, threshold=1e-6):
    """应用阈值：小于threshold的值设为threshold"""
    return np.where(errors < threshold, threshold, errors)

def moving_average_padded(data, window=20):
    """
    计算滑动平均，并在开始部分补齐
    对于前i个点（i<window），使用0到i的平均值
    """
    n = len(data)
    ma = np.zeros(n)

    for i in range(n):
        if i < window - 1:
            # 前(window-1)个点，使用从0到i的平均值
            ma[i] = np.mean(data[:i+1])
        else:
            # 后面的点使用正常的滑动平均
            ma[i] = np.mean(data[i-window+1:i+1])

    return ma

def plot_training_trajectory(feas_errors, opt_errors, save_path):
    """
    图1: 训练误差轨迹图
    - 对数坐标，10^-6以下显示为0
    - 原始误差曲线 + 滑动平均曲线
    - 阶段背景色
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    n_total = len(feas_errors)
    iterations = np.arange(n_total)

    # 定义阶段（基于1000次迭代: 300+200+500）
    # Phase 1: 1-300 (索引0-299)
    # Phase 2: 301-500 (索引300-499)
    # Phase 3: 501-1000 (索引500-999)
    phase1_end = 300
    phase2_end = 500

    # 添加阶段背景（不添加label，避免与legend冲突）
    ax.axvspan(0, phase1_end, alpha=0.15, color='green')
    ax.axvspan(phase1_end, phase2_end, alpha=0.15, color='orange')
    ax.axvspan(phase2_end, n_total, alpha=0.15, color='red')

    # 添加阶段分界线
    ax.axvline(phase1_end, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(phase2_end, color='black', linestyle='--', alpha=0.5, linewidth=1)

    # 应用阈值
    feas_errors_thresh = apply_threshold(feas_errors, 1e-6)
    opt_errors_thresh = apply_threshold(opt_errors, 1e-6)

    # 计算滑动平均（补齐开始部分）
    window = 20
    feas_ma = moving_average_padded(feas_errors_thresh, window)
    opt_ma = moving_average_padded(opt_errors_thresh, window)

    # 绘制原始误差曲线（细线，半透明）
    ax.semilogy(iterations, feas_errors_thresh,
                color=COLOR_FEAS, alpha=0.3, linewidth=0.8, label='Feasibility Error (raw)')
    ax.semilogy(iterations, opt_errors_thresh,
                color=COLOR_OPT, alpha=0.3, linewidth=0.8, label='Optimality Error (raw)')

    # 绘制滑动平均曲线（粗线，覆盖全部迭代）
    ax.semilogy(iterations, feas_ma,
                color=COLOR_FEAS, linewidth=2.5, label='Feasibility Error (moving average)')
    ax.semilogy(iterations, opt_ma,
                color=COLOR_OPT, linewidth=2.5, label='Optimality Error (moving average)')

    # 设置坐标轴
    ax.set_xlabel('Iteration steps')
    ax.set_ylabel('Error (p.u.)')
    # ax.set_title('Aggregation Training Error Convergence with Phase Analysis', fontweight='bold', pad=15)
    ax.set_xlim(0, min(n_total, 1000))  # 第一张图x轴限制为0-1000
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')  # 仅保留横向网格线

    y_loc = 0.9
    ax.text(phase1_end/2, y_loc, 'Phase 1: Balancing', ha='center', va='top', fontsize=font_size, fontweight='bold', color='darkgreen', alpha=0.8)
    ax.text((phase1_end + phase2_end)/2, y_loc, 'Phase 2: Refining', ha='center', va='top', fontsize=font_size, fontweight='bold', color='darkorange', alpha=0.8)
    ax.text((phase2_end + n_total)/2, y_loc, 'Phase 3: Converging', ha='center', va='top', fontsize=font_size, fontweight='bold', color='darkred', alpha=0.8)

    # 添加legend显示曲线意义，放置在右侧中部
    ax.legend(loc='center right', framealpha=0.5, fontsize=font_size-3)

    # 自定义y轴标签：将1e-6显示为0，固定范围为10^(-6)到10^(0)
    ax.set_ylim(1e-6, 1)  # 固定范围为10^(-6)到10^(0)
    ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_yticklabels(['0', '10⁻⁵', '10⁻⁴', '10⁻³', '10⁻²', '10⁻¹', '1'])  # 使用10的幂次方形式

    # 保存为SVG
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"图1已保存: {save_path}")
    plt.close()

def plot_phase_boxplot(feas_errors, opt_errors, save_path):
    """
    图2: 三阶段误差分布box图
    - 显示每个阶段最后部分误差数据的分布
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    n_total = len(feas_errors)
    phase1_end = 300
    phase2_end = 500

    # 提取各阶段的最后100个数据（如果阶段足够大）
    phase_data = []
    phase_labels = []

    # Phase 1: 取最后100个 (索引200-299)
    start_p1, end_p1 = max(0, phase1_end - 100), phase1_end
    feas_phase1 = apply_threshold(feas_errors[start_p1:end_p1], 1e-6)
    opt_phase1 = apply_threshold(opt_errors[start_p1:end_p1], 1e-6)
    phase_data.append(feas_phase1)
    phase_data.append(opt_phase1)
    phase_labels.extend(['Phase 1\n(1-300)\nFeas', 'Phase 1\n(1-300)\nOpt'])

    # Phase 2: 取最后100个 (索引400-499)
    start_p2, end_p2 = max(0, phase2_end - 100), phase2_end
    feas_phase2 = apply_threshold(feas_errors[start_p2:end_p2], 1e-6)
    opt_phase2 = apply_threshold(opt_errors[start_p2:end_p2], 1e-6)
    phase_data.append(feas_phase2)
    phase_data.append(opt_phase2)
    phase_labels.extend(['Phase 2\n(301-500)\nFeas', 'Phase 2\n(301-500)\nOpt'])

    # Phase 3: 取最后100个 (索引900-999)
    start_p3, end_p3 = max(0, n_total - 100), n_total
    feas_phase3 = apply_threshold(feas_errors[start_p3:end_p3], 1e-6)
    opt_phase3 = apply_threshold(opt_errors[start_p3:end_p3], 1e-6)
    phase_data.append(feas_phase3)
    phase_data.append(opt_phase3)
    phase_labels.extend(['Phase 3\n(501-1000)\nFeas', 'Phase 3\n(501-1000)\nOpt'])

    print(f"Box plot data: Phase 1 uses iterations {start_p1+1}-{end_p1}, "
          f"Phase 2 uses {start_p2+1}-{end_p2}, Phase 3 uses {start_p3+1}-{end_p3}")

    # 绘制box plot（显示平均值）
    bp = ax.boxplot(phase_data, tick_labels=phase_labels, patch_artist=True,
                    widths=0.6, showfliers=True,
                    showmeans=True,  # 显示平均值
                    meanprops=dict(marker='D', markerfacecolor='#FFD700',  # 金黄色
                                   markeredgecolor='black', markersize=6),
                    medianprops=dict(linewidth=0))  # 隐藏中位数线（设为不可见)

    # 着色
    for i, patch in enumerate(bp['boxes']):
        if i % 2 == 0:
            patch.set_facecolor(COLOR_FEAS)
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor(COLOR_OPT)
            patch.set_alpha(0.7)

    # 设置坐标轴
    ax.set_ylabel('Error Value (p.u.)', fontweight='bold')
    # ax.set_title('Aggregation Phase-wise Error Distribution', fontweight='bold', pad=15)
    ax.set_yscale('log')

    # 固定y轴范围为10^(-6)到10^(0)
    ax.set_ylim(1e-6, 1)  # 固定范围为10^(-6)到10^(0)
    ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_yticklabels(['0', '10⁻⁵', '10⁻⁴', '10⁻³', '10⁻²', '10⁻¹', '10⁰'])  # 使用10的幂次方形式
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    # 旋转x轴标签
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

    # 保存为SVG
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"图2已保存: {save_path}")
    plt.close()

def main():
    """主函数"""
    # 读取误差数据
    error_path = f'{PROJECT_ROOT}\\results\\aggregation\\error_history.pkl'
    feas_errors, opt_errors = load_error_data(error_path)

    print(f"加载数据: {len(feas_errors)} 次迭代")
    print(f"可行性误差范围: [{np.min(feas_errors):.2e}, {np.max(feas_errors):.2e}]")
    print(f"最优性误差范围: [{np.min(opt_errors):.2e}, {np.max(opt_errors):.2e}]")

    # 创建保存目录
    save_dir = f'{PROJECT_ROOT}\\results\\aggregation\\figures'
    os.makedirs(save_dir, exist_ok=True)

    # 绘制图1: 训练误差轨迹
    plot_training_trajectory(
        feas_errors, opt_errors,
        save_path=f'{save_dir}\\aggregation_training_trajectory.pdf'
    )

    # 绘制图2: 阶段box图
    plot_phase_boxplot(
        feas_errors, opt_errors,
        save_path=f'{save_dir}\\aggregation_phase_boxplot.pdf'
    )

    print(f"\n所有图表已保存到: {save_dir}")

if __name__ == '__main__':
    main()
