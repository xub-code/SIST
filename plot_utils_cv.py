import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# ============== 全局风格 (保持一致) ============== #
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 100
})


# ============== 工具：one-hot ============== #
def _one_hot(y, n_classes):
    y = np.asarray(y).astype(int)
    return np.eye(n_classes, dtype=int)[y]  # (N, C)


# ============== 混淆矩阵 ============== #
def plot_confusion_matrix(cm, class_names, filename="cv_confusion_matrix.png", save_dir="plot_cv"):
    """
    绘制混淆矩阵并保存
    :param cm: 混淆矩阵 (numpy array)
    :param class_names: 类别名称列表
    :param filename: 保存的文件名
    :param save_dir: 保存目录 (解决了之前的路径报错问题)
    """
    # 自动创建目录
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    fig, ax = plt.subplots(figsize=(2, 2))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks);
    ax.set_xticklabels(class_names, ha="right")
    ax.set_yticks(ticks);
    ax.set_yticklabels(class_names)

    # 阈值设置，用于调整字体颜色
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                fontsize=7, fontweight="bold",
                color="white" if cm[i, j] > thresh else "black"
            )

    ax.set_ylabel("True Labels", fontsize=8)
    ax.set_xlabel("Predicted Labels", fontsize=8)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.tight_layout()

    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Confusion Matrix saved to {save_path}")


# ============== ROC 曲线 ============== #
def plot_roc_curves(all_labels, all_probs, class_names, filename="cv_roc_curves.png", save_dir="plot_cv"):
    """
    绘制 ROC 曲线 (支持多分类 One-vs-Rest)
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    y = np.asarray(all_labels)
    P = np.asarray(all_probs)
    n_classes = len(class_names)

    # 安全检查
    if np.unique(y).size < 2:
        print("[WARN] ROC: 只有一个类别，跳过绘制。")
        return
    if P.ndim != 2 or P.shape[1] != n_classes:
        print(f"[ERROR] Probabilities shape mismatch. Expected (N, {n_classes}), got {P.shape}")
        return

    y_bin = _one_hot(y, n_classes)

    plt.figure(figsize=(2, 2))
    ax = plt.gca()
    ax.set_facecolor("#f9f9f9")
    colors = cycle(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])

    drew_any = False
    for i, color in zip(range(n_classes), colors):
        # 如果该类在测试集中没有正样本，跳过，防止报错
        if y_bin[:, i].sum() == 0:
            continue

        fpr, tpr, _ = roc_curve(y_bin[:, i], P[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=0.8, label=f"{class_names[i]} (AUC={roc_auc:.2f})")
        drew_any = True

    if not drew_any:
        print("[WARN] No valid ROC curves to draw.")
        plt.close()
        return

    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False Positive Rate", fontsize=8)
    plt.ylabel("True Positive Rate", fontsize=8)
    plt.legend(loc="lower right", fontsize=5, frameon=True, facecolor="white", framealpha=0.5)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.tight_layout()

    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] ROC Curves saved to {save_path}")


# ============== PR 曲线 ============== #
def plot_pr_curves(all_labels, all_probs, class_names, filename="cv_pr_curves.png", save_dir="plot_cv"):
    """
    绘制 PR 曲线
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    y = np.asarray(all_labels)
    P = np.asarray(all_probs)
    n_classes = len(class_names)

    if np.unique(y).size < 2:
        return
    if P.ndim != 2 or P.shape[1] != n_classes:
        return

    y_bin = _one_hot(y, n_classes)

    plt.figure(figsize=(2, 2))
    ax = plt.gca()
    ax.set_facecolor("#f9f9f9")
    colors = cycle(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])

    drew_any = False
    for i, color in zip(range(n_classes), colors):
        if y_bin[:, i].sum() == 0:
            continue

        precision, recall, _ = precision_recall_curve(y_bin[:, i], P[:, i])
        ap = average_precision_score(y_bin[:, i], P[:, i])
        plt.plot(recall, precision, color=color, lw=1.0, label=f"{class_names[i]} (AP={ap:.2f})")
        drew_any = True

    if not drew_any:
        plt.close()
        return

    plt.xlabel("Recall", fontsize=8)
    plt.ylabel("Precision", fontsize=8)
    plt.legend(loc="lower left", fontsize=5, frameon=True, facecolor="white", framealpha=0.5)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.tight_layout()

    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] PR Curves saved to {save_path}")