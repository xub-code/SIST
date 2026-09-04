import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# 1. 实验数据
# =========================
models = ["S1", "S2", "S3", "S4"]

accuracy = [83.02, 87.56, 83.19, 90.26]
f1_scores = [82.49, 87.13, 82.56, 89.99]

flops = [0.01, 0.40, 0.08, 0.47]   # G
params = [1.32, 7.45, 1.58, 7.72]  # M

plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# =========================
# 2. 保存路径
# =========================
save_dir = "plot"
os.makedirs(save_dir, exist_ok=True)

# =========================
# 3. 全局样式
# =========================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 6,
    "axes.labelsize": 6.5,
    "xtick.labelsize": 5.2,
    "ytick.labelsize": 5.2,
    "legend.fontsize": 4.6,  # 标签变长后微调字号，避免拥挤，可按需改回4.4
    "figure.dpi": 100,
    "axes.linewidth": 0.55,
    "grid.linewidth": 0.32,
})

# =========================
# 4. 通用绘图函数（新增字段名参数，标签带指标名称）
# =========================
def plot_tradeoff(
    x_values,
    y_values,
    x_label,
    y_label,
    metric_values,      # Acc 或 F1 的数值
    metric_name,        # 新增：指标名称（Acc/F1）
    metric_unit,        # Acc/F1 的单位 ("%")
    resource_values,    # Params 或 FLOPs 的数值
    resource_name,      # 新增：资源名称（Params/FLOPs）
    resource_unit,      # Params/FLOPs 的单位 ("M" 或 "G")
    file_prefix,
    x_lim,
    y_lim,
    x_ticks,
    y_ticks,
    legend_loc="upper left"
):
    fig, ax = plt.subplots(figsize=(1.5, 1.5))

    # 散点绘制
    for x, y, color in zip(x_values, y_values, plot_colors):
        ax.scatter(
            x=x,
            y=y,
            s=16,
            facecolor=color,
            edgecolor="black",
            linewidth=0.55,
            zorder=3
        )

    # 图例：明确标注指标名称+数值+单位，格式直观
    legend_handles = []
    for m, met, res, color in zip(models, metric_values, resource_values, plot_colors):
        # 标签格式示例：S1: Acc=83.02%, Params=1.32M
        label = f"{m}: {metric_name}={met:.2f}{metric_unit}, {resource_name}={res:.2f}{resource_unit}"
        handle = Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markerfacecolor=color,
            markeredgecolor='black',
            markeredgewidth=0.55,
            markersize=3.8,
            label=label
        )
        legend_handles.append(handle)

    # 图例样式设置
    legend = ax.legend(
        handles=legend_handles,
        loc=legend_loc,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.12,
        handletextpad=0.22,
        labelspacing=0.15,  # 微调行间距，避免标签重叠
        borderaxespad=0.15
    )
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.4)

    # 坐标轴设置
    ax.set_xlabel(x_label, labelpad=0.5)
    ax.set_ylabel(y_label, labelpad=0.5)

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    ax.grid(True, linestyle="--", alpha=0.40, zorder=0)
    ax.tick_params(axis="both", which="major", length=2.5, width=0.55, pad=1.0)

    # 坐标轴标签位置优化
    ax.xaxis.set_label_coords(0.5, -0.11)
    ax.yaxis.set_label_coords(-0.13, 0.5)

    # 画布留白压缩
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.18, top=0.985)

    # 图片保存
    png_path = os.path.join(save_dir, f"{file_prefix}.png")
    pdf_path = os.path.join(save_dir, f"{file_prefix}.pdf")

    plt.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.004)
    plt.savefig(pdf_path, dpi=600, bbox_inches="tight", pad_inches=0.004)
    plt.close()

    print(f"图片已保存至: {png_path}")
    print(f"PDF矢量图已保存至: {pdf_path}")


# =========================
# 5. 四张图分别绘制（精准传入对应指标名称，完全匹配图意）
# =========================

# (a) Accuracy (%) versus FLOPs (G) | 图例：Acc + Params
plot_tradeoff(
    x_values=flops,
    y_values=accuracy,
    x_label="FLOPs (G)",
    y_label="Accuracy (%)",
    metric_values=accuracy,
    metric_name="Acc",          # 明确标注Acc
    metric_unit="%",
    resource_values=params,
    resource_name="Params",     # 明确标注Params
    resource_unit="M",
    file_prefix="subfig_a_accuracy_flops",
    x_lim=(0.0, 0.50),
    y_lim=(82.0, 91.0),
    x_ticks=[0.0, 0.2, 0.4],
    y_ticks=[82, 86, 90],
    legend_loc="upper left"
)

# (b) F1 Score (%) versus FLOPs (G) | 图例：F1 + Params
plot_tradeoff(
    x_values=flops,
    y_values=f1_scores,
    x_label="FLOPs (G)",
    y_label="F1 Score (%)",
    metric_values=f1_scores,
    metric_name="F1",           # 明确标注F1
    metric_unit="%",
    resource_values=params,
    resource_name="Params",     # 明确标注Params
    resource_unit="M",
    file_prefix="subfig_b_f1_flops",
    x_lim=(0.0, 0.50),
    y_lim=(81.8, 90.8),
    x_ticks=[0.0, 0.2, 0.4],
    y_ticks=[82, 86, 90],
    legend_loc="upper left"
)

# (c) Accuracy (%) versus Params (M) | 图例：Acc + FLOPs
plot_tradeoff(
    x_values=params,
    y_values=accuracy,
    x_label="Params (M)",
    y_label="Accuracy (%)",
    metric_values=accuracy,
    metric_name="Acc",          # 明确标注Acc
    metric_unit="%",
    resource_values=flops,
    resource_name="FLOPs",      # 明确标注FLOPs
    resource_unit="G",
    file_prefix="subfig_c_accuracy_params",
    x_lim=(1.0, 8.0),
    y_lim=(82.0, 91.0),
    x_ticks=[2, 4, 6, 8],
    y_ticks=[82, 86, 90],
    legend_loc="upper left"
)

# (d) F1 Score (%) versus Params (M) | 图例：F1 + FLOPs
plot_tradeoff(
    x_values=params,
    y_values=f1_scores,
    x_label="Params (M)",
    y_label="F1 Score (%)",
    metric_values=f1_scores,
    metric_name="F1",           # 明确标注F1
    metric_unit="%",
    resource_values=flops,
    resource_name="FLOPs",      # 明确标注FLOPs
    resource_unit="G",
    file_prefix="subfig_d_f1_params",
    x_lim=(1.0, 8.0),
    y_lim=(81.8, 90.8),
    x_ticks=[2, 4, 6, 8],
    y_ticks=[82, 86, 90],
    legend_loc="upper left"
)