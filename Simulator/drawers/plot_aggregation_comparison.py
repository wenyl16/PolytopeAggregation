"""
绘制聚合方法误差比较图
双y轴展示可行性误差和最优性误差的分布
IEEE Transactions 风格
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path


def load_error_comparison(filepath: str) -> dict:
    """加载误差比较数据"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def plot_error_comparison(error_comparison: dict, save_path: str = None):
    """
    绘制误差比较图 (IEEE Trans 风格)

    使用双y轴：
    - 左轴：可行性误差 (error_feas)，1e-6以下视为0
    - 右轴：最优性误差 (error_opt)，线性刻度
    """
    # IEEE Trans 风格设置 - 统一字体大小为8pt
    font_size = 8
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': font_size,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 0.5,
        'lines.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
    })

    methods = ['cube', 'proposed_PB', 'outer', 'matlab_polytope', 'proposed']
    method_labels = ['Box\n(inner)', 'Proposed\n(2T inner)', 'PEB\n(outer)', 'PEB\n(inner)', 'Proposed\n(4T inner)']

    # 提取数据并展平，1e-6以下设为1e-6
    threshold = 1e-6
    feas_data = []
    for m in methods:
        data = np.array(error_comparison[m]['error_feas']).flatten()
        data[data < threshold] = 1e-6
        feas_data.append(data)

    opt_data = [np.array(error_comparison[m]['error_opt']).flatten() for m in methods]

    # 创建图形
    fig, ax1 = plt.subplots(figsize=(3.5, 2))  # IEEE单栏宽度约3.5英寸，高度2英寸

    # 设置位置
    x = np.arange(len(methods))
    width = 0.3

    # IEEE风格配色
    feas_color = '#0C5DA5'   # 深蓝色
    opt_color = '#FF2C00'    # 深红色

    # 左轴：可行性误差 (小提琴图)
    positions_feas = x - width/2
    parts_feas = ax1.violinplot(
        feas_data,
        positions=positions_feas,
        widths=width * 0.9,
        showmeans=True,
        showmedians=False,
        showextrema=True
    )
    # 设置可行性误差violin的颜色和线宽
    for pc in parts_feas['bodies']:
        pc.set_facecolor(feas_color)
        pc.set_edgecolor(feas_color)
        pc.set_alpha(0.5)
        pc.set_linewidth(0.5)
    parts_feas['cmeans'].set_color(feas_color)
    parts_feas['cmeans'].set_linewidth(0.5)
    parts_feas['cbars'].set_color(feas_color)
    parts_feas['cbars'].set_linewidth(0.5)
    parts_feas['cmins'].set_color(feas_color)
    parts_feas['cmins'].set_linewidth(0.5)
    parts_feas['cmaxes'].set_color(feas_color)
    parts_feas['cmaxes'].set_linewidth(0.5)

    ax1.set_yscale('log')

    # 取消左轴小刻度
    ax1.minorticks_off()

    # 设置可行性误差的范围
    ax1.set_ylim(5e-7, 1e1)
    ax1.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])
    ax1.set_yticklabels(['$0$', '$10^{-5}$', '$10^{-4}$', '$10^{-3}$',
                         '$10^{-2}$', '$10^{-1}$', '$10^{0}$', '$10^{1}$'])

    # 右轴：最优性误差 (小提琴图)
    ax2 = ax1.twinx()
    positions_opt = x + width/2
    parts_opt = ax2.violinplot(
        opt_data,
        positions=positions_opt,
        widths=width * 0.9,
        showmeans=True,
        showmedians=False,
        showextrema=True
    )
    # 设置最优性误差violin的颜色和线宽
    for pc in parts_opt['bodies']:
        pc.set_facecolor(opt_color)
        pc.set_edgecolor(opt_color)
        pc.set_alpha(0.5)
        pc.set_linewidth(0.5)
    parts_opt['cmeans'].set_color(opt_color)
    parts_opt['cmeans'].set_linewidth(0.5)
    parts_opt['cbars'].set_color(opt_color)
    parts_opt['cbars'].set_linewidth(0.5)
    parts_opt['cmins'].set_color(opt_color)
    parts_opt['cmins'].set_linewidth(0.5)
    parts_opt['cmaxes'].set_color(opt_color)
    parts_opt['cmaxes'].set_linewidth(0.5)

    ax2.tick_params(axis='y', labelcolor=opt_color, width=0.5)

    # 最优性误差：线性刻度，-log10(2)到7
    ax2.set_ylim(-np.log10(2), 7)
    ax2.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])

    # 设置x轴 - 取消文字
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_labels)
    ax1.tick_params(axis='x', width=0.5)

    # 取消图例
    # ax1.legend().set_visible(False)

    # 添加网格
    ax1.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    # 设置左轴刻度线宽度
    ax1.tick_params(axis='y', width=0.5)

    plt.tight_layout()

    if save_path:
        # 保存为SVG格式
        svg_path = str(save_path).replace('.pdf', '.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {svg_path}")

    plt.show()

    return fig


def print_statistics(error_comparison: dict):
    """打印各方法的统计信息"""
    methods = ['outer', 'cube', 'matlab_polytope','proposed_PB', 'proposed']

    print("\n" + "="*70)
    print("Error Statistics Summary")
    print("="*70)
    print(f"{'Method':<15} {'Feas Mean':>12} {'Feas Std':>12} {'Opt Mean':>12} {'Opt Std':>12}")
    print("-"*70)

    for m in methods:
        feas = np.array(error_comparison[m]['error_feas']).flatten()
        opt = np.array(error_comparison[m]['error_opt']).flatten()
        print(f"{m:<15} {np.mean(feas):>12.2e} {np.std(feas):>12.2e} "
              f"{np.mean(opt):>12.2e} {np.std(opt):>12.2e}")

    print("="*70 + "\n")


if __name__ == '__main__':
    # 设置路径
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'results' / 'aggregation' / 'error_comparison.pkl'
    save_dir = project_root / 'results' / 'aggregation'

    # 加载数据
    print(f"Loading data from: {data_path}")
    error_comparison = load_error_comparison(data_path)

    # 打印统计信息
    print_statistics(error_comparison)

    # 绘制双y轴图
    print("Plotting dual-axis violin plot...")
    plot_error_comparison(
        error_comparison,
        save_path=save_dir / 'error_comparison_dual_axis.svg'
    )
