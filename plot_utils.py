import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# ============== 全局风格 ============== #
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 100
})

SAVE_DIR = "plot"
os.makedirs(SAVE_DIR, exist_ok=True)


# ============== 混淆矩阵 ============== #
def plot_confusion_matrix(cm, class_names, filename="confusion_matrix.png"):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(1.0, 1.0))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    # 如果觉得 colorbar 在 1.5x1.5 图里太挤，可以直接注释掉下面两行
    # cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    # cbar.ax.tick_params(labelsize=5)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=0, fontsize=6)
    ax.set_yticklabels(class_names, rotation=90, fontsize=6)

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(int(cm[i, j]), "d"),
                ha="center", va="center",
                fontsize=6,
                fontweight="bold",
                color="white" if cm[i, j] > thresh else "black"
            )

    ax.set_ylabel("True labels", fontsize=7, labelpad=1)
    ax.set_xlabel("Predicted labels", fontsize=7, labelpad=1)
    ax.set_aspect("equal")

    # 只修改线的粗细
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(axis="both", which="both", width=0.5)

    # 尽量减小留白
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.18, top=0.98)

    plt.savefig(
        os.path.join(SAVE_DIR, filename),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.005
    )
    plt.close()


# ============== 工具：one-hot ============== #
def _one_hot(y, n_classes):
    y = np.asarray(y).astype(int)
    return np.eye(n_classes, dtype=int)[y]  # (N, C)


# ============== ROC（每类一对其余，二分类也画两条） ============== #
def plot_roc_curves(all_labels, all_probs, class_names, filename="roc_curves.png"):
    y = np.asarray(all_labels)
    P = np.asarray(all_probs)
    n_classes = len(class_names)

    if np.unique(y).size < 2:
        print("[WARN] ROC: y_true 只包含一个类别，无法绘制，已跳过。")
        return
    if P.ndim != 2 or P.shape[1] != n_classes:
        raise ValueError(f"[ERROR] all_probs 应为 (N,{n_classes})，当前形状 {P.shape}")

    y_bin = _one_hot(y, n_classes)

    plt.figure(figsize=(2, 2))
    ax = plt.gca()
    ax.set_facecolor("#f9f9f9")
    colors = cycle(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])

    drew_any = False
    for i, color in zip(range(n_classes), colors):
        pos = y_bin[:, i].sum()
        neg = len(y) - pos
        if pos == 0 or neg == 0:
            print(f"[WARN] ROC: 类别 '{class_names[i]}' 在测试集中正/负样本不足（pos={pos}, neg={neg}），跳过该类。")
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], P[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=0.5, label=f"{class_names[i]} (AUC={roc_auc:.2f})")
        drew_any = True

    if not drew_any:
        print("[WARN] ROC: 所有类别都无法绘制（可能测试集极端不平衡）。")
        plt.close()
        return

    plt.plot([0, 1], [0, 1], "k--", lw=0.5)
    plt.xlabel("False Positive Rate", fontsize=8)
    plt.ylabel("True Positive Rate", fontsize=8)

    # 只修改线的粗细
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(axis="both", which="both", width=0.5)

    legend = plt.legend(loc="lower right", fontsize=5, frameon=True, facecolor="white", framealpha=0.5)
    legend.get_frame().set_linewidth(0.5)
    for line in legend.get_lines():
        line.set_linewidth(0.5)

    plt.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=600, bbox_inches="tight", pad_inches=0.01)
    plt.close()


# ============== PR（每类一对其余，二分类也画两条） ============== #
def plot_pr_curves(all_labels, all_probs, class_names, filename="pr_curves.png"):
    y = np.asarray(all_labels)
    P = np.asarray(all_probs)
    n_classes = len(class_names)

    if np.unique(y).size < 2:
        print("[WARN] PR: y_true 只包含一个类别，无法绘制，已跳过。")
        return
    if P.ndim != 2 or P.shape[1] != n_classes:
        raise ValueError(f"[ERROR] all_probs 应为 (N,{n_classes})，当前形状 {P.shape}")

    y_bin = _one_hot(y, n_classes)

    plt.figure(figsize=(2, 2))
    ax = plt.gca()
    ax.set_facecolor("#f9f9f9")
    colors = cycle(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])

    drew_any = False
    for i, color in zip(range(n_classes), colors):
        pos = y_bin[:, i].sum()
        neg = len(y) - pos
        if pos == 0 or neg == 0:
            print(f"[WARN] PR: 类别 '{class_names[i]}' 在测试集中正/负样本不足（pos={pos}, neg={neg}），跳过该类。")
            continue
        precision, recall, _ = precision_recall_curve(y_bin[:, i], P[:, i])
        ap = average_precision_score(y_bin[:, i], P[:, i])
        plt.plot(recall, precision, color=color, lw=0.5, label=f"{class_names[i]} (AP={ap:.2f})")
        drew_any = True

    if not drew_any:
        print("[WARN] PR: 所有类别都无法绘制（可能测试集极端不平衡）。")
        plt.close()
        return

    plt.xlabel("Recall", fontsize=8)
    plt.ylabel("Precision", fontsize=8)

    # 只修改线的粗细
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(axis="both", which="both", width=0.5)

    legend = plt.legend(loc="lower left", fontsize=5, frameon=True, facecolor="white", framealpha=0.5)
    legend.get_frame().set_linewidth(0.5)
    for line in legend.get_lines():
        line.set_linewidth(0.5)

    plt.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=600, bbox_inches="tight", pad_inches=0.01)
    plt.close()