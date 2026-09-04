import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
import argparse
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score, accuracy_score
)
from sklearn.preprocessing import label_binarize
import warnings
from collections import defaultdict

from dataset import MultimodalDataset, collate_fn
from model import MultiModalNet
from plot_utils_cv import plot_confusion_matrix, plot_roc_curves, plot_pr_curves

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

DATA_ROOT_CV = "ADReSS2020_5CV"
WEIGHTS_DIR = "weights_adress_cv"
PLOT_DIR = "plot_cv"
os.makedirs(PLOT_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def format_percent(val):
    return "-" if val is None else f"{val * 100:.2f}%"


def format_ms_percent(mean_val, std_val):
    return f"{mean_val * 100:.2f}±{std_val * 100:.2f}%"


def format_ms_number(mean_val, std_val):
    return f"{mean_val:.2f}±{std_val:.2f}"


def test_single_fold(model, loader, device, fold_idx):
    """测试单个 Fold，返回预测结果列表"""
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    print(f"[INFO] Running Inference for Fold {fold_idx}...")
    with torch.no_grad():
        for audio_x, audio_mask, text_x, text_mask, y in tqdm(loader, desc=f"Testing Fold {fold_idx}"):
            audio_x, text_x, y = audio_x.to(device), text_x.to(device), y.to(device)
            audio_mask, text_mask = audio_mask.to(device), text_mask.to(device)

            logits = model(audio_x, audio_mask, text_x, text_mask)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
            y_probs.extend(F.softmax(logits, dim=1).cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_probs)


def calculate_metrics(y_true, y_pred, y_probs, class_names):
    """计算详细指标，返回与 test.py 风格一致的字典结构"""
    n_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred)
    global_acc = accuracy_score(y_true, y_pred)

    y_onehot = label_binarize(y_true, classes=range(n_classes))
    if n_classes == 2 and y_onehot.shape[1] == 1:
        y_onehot = np.hstack([1 - y_onehot, y_onehot])

    metrics = {}
    prec_list, rec_list, f1_list = [], [], []
    supports = []

    for i, cls in enumerate(class_names):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        Support = TP + FN

        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0.0

        if np.sum(y_onehot[:, i]) == 0:
            AUC, AP = 0.0, 0.0
        else:
            AUC = roc_auc_score(y_onehot[:, i], y_probs[:, i])
            AP = average_precision_score(y_onehot[:, i], y_probs[:, i])

        metrics[cls] = {
            "prec": Precision,
            "rec": Recall,
            "f1": F1,
            "auc": AUC,
            "ap": AP,
            "support": Support,
        }
        prec_list.append(Precision)
        rec_list.append(Recall)
        f1_list.append(F1)
        supports.append(Support)

    macro_auc = roc_auc_score(y_onehot, y_probs, average="macro")
    macro_ap = average_precision_score(y_onehot, y_probs, average="macro")

    metrics["macro"] = {
        "prec": np.mean(prec_list),
        "rec": np.mean(rec_list),
        "f1": np.mean(f1_list),
        "auc": macro_auc,
        "ap": macro_ap,
        "support": np.sum(supports),
    }
    metrics["overall_acc"] = global_acc
    metrics["cm"] = cm
    return metrics


def print_single_report(metrics, class_names, title):
    """打印单次评估报告，风格对齐 test.py"""
    print("\n" + "=" * 95)
    print(title)
    print("=" * 95)
    print("{:<12} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<8}".format(
        "Class", "Acc", "Precision", "Recall", "F1-Score", "AUC", "AP", "Support"
    ))
    print("-" * 95)

    for cls in class_names:
        m = metrics[cls]
        print("{:<12} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<8}".format(
            cls,
            "-",
            format_percent(m["prec"]),
            format_percent(m["rec"]),
            format_percent(m["f1"]),
            format_percent(m["auc"]),
            format_percent(m["ap"]),
            m["support"],
        ))

    print("-" * 95)
    mm = metrics["macro"]
    print("{:<12} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<8}".format(
        "Macro avg",
        "-",
        format_percent(mm["prec"]),
        format_percent(mm["rec"]),
        format_percent(mm["f1"]),
        format_percent(mm["auc"]),
        format_percent(mm["ap"]),
        mm["support"],
    ))
    print("-" * 95)
    print(f"Overall Acc : {format_percent(metrics['overall_acc'])}")
    print("=" * 95 + "\n")


def print_cv_report(metrics_history, class_names):
    """打印 5-Fold CV 汇总报告，风格尽量对齐 test.py，但指标显示为 Mean ± Std"""
    agg = defaultdict(lambda: defaultdict(list))
    overall_accs = []

    for res in metrics_history:
        overall_accs.append(res["overall_acc"])
        for cls in class_names + ["macro"]:
            for metric in ["prec", "rec", "f1", "auc", "ap", "support"]:
                agg[cls][metric].append(res[cls][metric])

    print("\n" + "=" * 120)
    print("【FINAL 5-FOLD CV REPORT】")
    print("=" * 120)
    print("{:<12} {:<10} {:<18} {:<18} {:<18} {:<18} {:<18} {:<14}".format(
        "Class", "Acc", "Precision", "Recall", "F1-Score", "AUC", "AP", "Support"
    ))
    print("-" * 120)

    for cls in class_names:
        print("{:<12} {:<10} {:<18} {:<18} {:<18} {:<18} {:<18} {:<14}".format(
            cls,
            "-",
            format_ms_percent(np.mean(agg[cls]["prec"]), np.std(agg[cls]["prec"])),
            format_ms_percent(np.mean(agg[cls]["rec"]), np.std(agg[cls]["rec"])),
            format_ms_percent(np.mean(agg[cls]["f1"]), np.std(agg[cls]["f1"])),
            format_ms_percent(np.mean(agg[cls]["auc"]), np.std(agg[cls]["auc"])),
            format_ms_percent(np.mean(agg[cls]["ap"]), np.std(agg[cls]["ap"])),
            format_ms_number(np.mean(agg[cls]["support"]), np.std(agg[cls]["support"])),
        ))

    print("-" * 120)
    print("{:<12} {:<10} {:<18} {:<18} {:<18} {:<18} {:<18} {:<14}".format(
        "Macro avg",
        "-",
        format_ms_percent(np.mean(agg["macro"]["prec"]), np.std(agg["macro"]["prec"])),
        format_ms_percent(np.mean(agg["macro"]["rec"]), np.std(agg["macro"]["rec"])),
        format_ms_percent(np.mean(agg["macro"]["f1"]), np.std(agg["macro"]["f1"])),
        format_ms_percent(np.mean(agg["macro"]["auc"]), np.std(agg["macro"]["auc"])),
        format_ms_percent(np.mean(agg["macro"]["ap"]), np.std(agg["macro"]["ap"])),
        format_ms_number(np.mean(agg["macro"]["support"]), np.std(agg["macro"]["support"])),
    ))
    print("-" * 120)
    print(f"Overall Acc : {format_ms_percent(np.mean(overall_accs), np.std(overall_accs))}")
    print("=" * 120 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Cross-validation data root: {DATA_ROOT_CV}")

    class_map = {"cc": 0, "cd": 1}
    class_names = [k for k, v in sorted(class_map.items(), key=lambda item: item[1])]

    global_y_true = []
    global_y_probs = []
    global_y_pred = []
    metrics_history = []

    BATCH_SIZE = 32
    SHARED_DIM = 768
    DROPOUT = 0.3

    for fold_idx in range(1, 6):
        print(f"\n{'#' * 28} Fold {fold_idx} {'#' * 28}")

        test_path = os.path.join(DATA_ROOT_CV, f"fold{fold_idx}", "test")
        weights_file = os.path.join(WEIGHTS_DIR, f"best_fold{fold_idx}.pth")

        print(f"[INFO] Loading Data from: {test_path}")
        if not os.path.exists(weights_file):
            print(f"[ERROR] Weights file NOT found at: {weights_file}")
            print(f"[WARNING] Fold {fold_idx} skipped.")
            continue

        test_dataset = MultimodalDataset(test_path, class_map)
        loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

        model = MultiModalNet(
            audio_dim=test_dataset[0][0].shape[1],
            text_dim=test_dataset[0][1].shape[1],
            num_classes=len(class_names),
            fusion_type="gated_bi_cross_attention",
            dropout=DROPOUT,
            shared_dim=SHARED_DIM,
        ).to(device)

        model.load_state_dict(torch.load(weights_file, map_location=device))
        print(f"[INFO] Weights loaded from: {weights_file}")

        f_true, f_pred, f_probs = test_single_fold(model, loader, device, fold_idx)

        global_y_true.extend(f_true)
        global_y_pred.extend(f_pred)
        global_y_probs.extend(f_probs)

        fold_metrics = calculate_metrics(f_true, f_pred, f_probs, class_names)
        metrics_history.append(fold_metrics)
        print_single_report(fold_metrics, class_names, f"【FOLD {fold_idx} EVALUATION REPORT】")

        print(f"[INFO] Saving plots for Fold {fold_idx}...")
        plot_confusion_matrix(
            fold_metrics["cm"],
            class_names,
            filename=f"fold{fold_idx}_confusion_matrix.png",
            save_dir=PLOT_DIR,
        )
        plot_roc_curves(
            f_true,
            f_probs,
            class_names,
            filename=f"fold{fold_idx}_roc_curves.png",
            save_dir=PLOT_DIR,
        )
        plot_pr_curves(
            f_true,
            f_probs,
            class_names,
            filename=f"fold{fold_idx}_pr_curves.png",
            save_dir=PLOT_DIR,
        )

    if not metrics_history:
        print("[ERROR] No valid testing results were produced.")
        return

    print_cv_report(metrics_history, class_names)

    global_y_true = np.array(global_y_true)
    global_y_pred = np.array(global_y_pred)
    global_y_probs = np.array(global_y_probs)

    global_metrics = calculate_metrics(global_y_true, global_y_pred, global_y_probs, class_names)
    print_single_report(global_metrics, class_names, "【GLOBAL POOLED EVALUATION REPORT】")

    print(f"[INFO] Generating global summary plots...")
    plot_confusion_matrix(
        global_metrics["cm"],
        class_names,
        filename="cv_global_confusion_matrix.png",
        save_dir=PLOT_DIR,
    )
    plot_roc_curves(
        global_y_true,
        global_y_probs,
        class_names,
        filename="cv_global_roc_curves.png",
        save_dir=PLOT_DIR,
    )
    plot_pr_curves(
        global_y_true,
        global_y_probs,
        class_names,
        filename="cv_global_pr_curves.png",
        save_dir=PLOT_DIR,
    )
    print(f"[SUCCESS] All plots have been saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
