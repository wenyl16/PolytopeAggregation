"""
可视化MicroGrid Case训练前后A矩阵的演化
精简版：只生成差异热力图
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from Simulator import PROJECT_ROOT

fz = 16
# 设置出版级样式
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = fz
plt.rcParams['axes.labelsize'] = fz
plt.rcParams['axes.titlesize'] = fz
plt.rcParams['xtick.labelsize'] = fz
plt.rcParams['ytick.labelsize'] = fz

def load_A_matrices(figure_folder):
    """加载初始和最终的A矩阵"""
    # 初始A矩阵
    A_initial = np.loadtxt(f'{figure_folder}/A_0.csv', delimiter=',')

    # 最终A矩阵（找到最大的迭代次数）
    A_files = [f for f in os.listdir(figure_folder) if f.startswith('A_') and f.endswith('.csv')]
    iterations = [int(f.split('_')[1].split('.')[0]) for f in A_files]
    final_iter = max(iterations)
    A_final = np.loadtxt(f'{figure_folder}/A_{final_iter}.csv', delimiter=',')

    return A_initial, A_final, final_iter

def plot_A_difference_heatmap(A_initial, A_final, save_path):
    """
    A矩阵差异热力图
    - 白色 = 无变化（差异为0）
    - 红色 = 增加，蓝色 = 减少
    """
    fig, ax = plt.subplots(figsize=(9, 2.6))
    # fig, ax = plt.subplots(figsize=(5.2, 2.6)) #PB
    # 计算差异
    A_diff = A_final - A_initial

    # 绘制差异热力图
    im = ax.imshow(A_diff.T, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)

    # 添加网格线作为边框
    ax.set_xticks(np.arange(A_diff.shape[0] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(A_diff.shape[1] + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # 设置标题和标签
    # ax.set_title('A Matrix Evolution (Trained - Initial)', fontweight='bold', pad=15)
    # ax.set_xlabel('Index of rows', fontweight='bold')
    # ax.set_ylabel('Index of columns', fontweight='bold')

    # 将x轴刻度移到上方
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    # 设置刻度
    n_rows, n_cols = A_diff.shape

    # 横坐标：1, 12, 24（对应索引0, 11, 23）
    x_ticks = [0, 23, 47, 71, 95]
    x_labels = [1, 24, 48, 72, 96]
    # x_ticks = [0, 23, 47] #PB
    # x_labels = [1, 24, 48] #PB
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    # 纵坐标：1, 24, 48, 72, 96
    y_ticks = [0, 11, 23]  # 索引位置
    y_labels = [1, 12, 24]  # 显示的刻度标签
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Difference', rotation=270, labelpad=20, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"差异热力图已保存: {save_path}")
    plt.close()

def main():
    """主函数"""
    # 加载数据 - 支持多个 case
    case_name = 'aggregation'  # 可选: 'microgrid', 'aggregation'
    cont = 'continuous'  # 可选: 'continuous', 'discrete'
    figure_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\{cont}\\figures'

    # 如果文件夹不存在,尝试其他路径
    import os
    if not os.path.exists(figure_folder):
        # 尝试直接使用 case_name/figures
        figure_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\figures'

    print(f"加载数据从: {figure_folder}")
    A_initial, A_final, final_iter = load_A_matrices(figure_folder)

    print(f"A矩阵形状: {A_initial.shape}")
    print(f"训练迭代次数: {final_iter}")
    print(f"初始A矩阵 - 范围: [{np.min(A_initial):.4f}, {np.max(A_initial):.4f}]")
    print(f"训练A矩阵 - 范围: [{np.min(A_final):.4f}, {np.max(A_final):.4f}]")

    A_diff = A_final - A_initial
    print(f"A矩阵差异 - 最大差异: {np.min((A_diff)):.4f}, {np.max((A_diff)):.4f}")

    print(f"           - 非零元素数: {np.count_nonzero(A_diff)} / {A_diff.size}")

    # 创建保存目录
    save_dir = f'{PROJECT_ROOT}\\results\\{case_name}\\{cont}\\A_analysis'
    if not os.path.exists(save_dir):
        save_dir = f'{PROJECT_ROOT}\\results\\{case_name}\\A_analysis'
    os.makedirs(save_dir, exist_ok=True)

    # 绘制差异热力图
    plot_A_difference_heatmap(
        A_initial, A_final,
        save_path=f'{save_dir}\\A_difference_heatmap.svg'
    )

    print(f"\n图表已保存到: {save_dir}")

if __name__ == '__main__':
    main()
