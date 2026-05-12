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

# 导入工具函数
from dataset import MultimodalDataset, collate_fn
from model import MultiModalNet
# 确保你已经有了 plot_utils_cv.py
from plot_utils_cv import plot_confusion_matrix, plot_roc_curves, plot_pr_curves

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

DATA_ROOT_CV = "NCMMSC2021_5CV"
WEIGHTS_DIR = "weights_ncmmsc_cv"
PLOT_DIR = "plot_cv"
os.makedirs(PLOT_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def format_ms(mean_val, std_val):
    """格式化为 Mean ± Std %"""
    return f"{mean_val * 100:.2f}±{std_val * 100:.2f}%"


def test_single_fold(model, loader, device):
    """测试单个 Fold，返回预测结果列表"""
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for audio_x, audio_mask, text_x, text_mask, y in loader:
            audio_x, text_x, y = audio_x.to(device), text_x.to(device), y.to(device)
            audio_mask, text_mask = audio_mask.to(device), text_mask.to(device)

            logits = model(audio_x, audio_mask, text_x, text_mask)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
            y_probs.extend(F.softmax(logits, dim=1).cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_probs)


def calculate_fold_metrics(y_true, y_pred, y_probs, class_names):
    """计算单个 Fold 的详细指标"""
    n_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    y_onehot = label_binarize(y_true, classes=range(n_classes))
    if n_classes == 2 and y_onehot.shape[1] == 1:
        y_onehot = np.hstack([1 - y_onehot, y_onehot])

    fold_res = {}

    prec_list, rec_list, f1_list = [], [], []

    for i, cls in enumerate(class_names):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP

        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0.0

        if np.sum(y_onehot[:, i]) == 0:
            AUC, AP = 0.0, 0.0
        else:
            AUC = roc_auc_score(y_onehot[:, i], y_probs[:, i])
            AP = average_precision_score(y_onehot[:, i], y_probs[:, i])

        fold_res[cls] = {
            "Acc": acc,
            "Precision": Precision,
            "Recall": Recall,
            "F1-Score": F1,
            "AUC": AUC,
            "AP": AP
        }

        prec_list.append(Precision)
        rec_list.append(Recall)
        f1_list.append(F1)

    # Macro Average
    macro_auc = roc_auc_score(y_onehot, y_probs, average="macro")
    macro_ap = average_precision_score(y_onehot, y_probs, average="macro")

    fold_res['Macro avg'] = {
        "Acc": acc,
        "Precision": np.mean(prec_list),
        "Recall": np.mean(rec_list),
        "F1-Score": np.mean(f1_list),
        "AUC": macro_auc,
        "AP": macro_ap
    }

    fold_res['Overall Acc'] = acc
    return fold_res


def print_cv_report(metrics_history, class_names):
    """打印 5-Fold CV 统计报告 (Mean ± Std)"""
    n_folds = len(metrics_history)
    agg = defaultdict(lambda: defaultdict(list))
    overall_accs = []

    for res in metrics_history:
        overall_accs.append(res['Overall Acc'])
        for cls in class_names + ['Macro avg']:
            for metric, val in res[cls].items():
                agg[cls][metric].append(val)

    print("\n" + "=" * 115)
    print(f"【5-FOLD CV STATISTICS REPORT (Mean ± Std)】")
    print("=" * 115)

    headers = ["Class", "Precision", "Recall", "F1-Score", "AUC", "AP"]
    print(f"{headers[0]:<12} {headers[1]:<18} {headers[2]:<18} {headers[3]:<18} {headers[4]:<18} {headers[5]:<18}")
    print("-" * 115)

    def get_ms(cls, m):
        vals = agg[cls][m]
        return format_ms(np.mean(vals), np.std(vals))

    for cls in class_names:
        print(
            f"{cls:<12} {get_ms(cls, 'Precision'):<18} {get_ms(cls, 'Recall'):<18} {get_ms(cls, 'F1-Score'):<18} {get_ms(cls, 'AUC'):<18} {get_ms(cls, 'AP'):<18}")

    print("-" * 115)
    cls = 'Macro avg'
    print(
        f"{cls:<12} {get_ms(cls, 'Precision'):<18} {get_ms(cls, 'Recall'):<18} {get_ms(cls, 'F1-Score'):<18} {get_ms(cls, 'AUC'):<18} {get_ms(cls, 'AP'):<18}")

    print("-" * 115)
    acc_mean = np.mean(overall_accs)
    acc_std = np.std(overall_accs)
    print(f"Overall Acc : {format_ms(acc_mean, acc_std)}")
    print("=" * 115 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=9999)
    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    class_map = {"AD": 0, "HC": 1, "MCI": 2}
    class_names = [k for k, v in sorted(class_map.items(), key=lambda item: item[1])]

    global_y_true = []
    global_y_probs = []
    global_y_pred = []
    metrics_history = []

    # --- Config ---
    BATCH_SIZE = 4
    SHARED_DIM = 512
    DROPOUT = 0.3

    # ==========================================
    # Loop over 5 folds
    # ==========================================
    for fold_idx in range(1, 6):
        print(f"\n--- Testing Fold {fold_idx} ---")

        test_path = os.path.join(DATA_ROOT_CV, f"fold{fold_idx}", "test")
        weights_file = os.path.join(WEIGHTS_DIR, f"best_fold{fold_idx}.pth")

        if not os.path.exists(weights_file):
            print(f"[ERROR] 权重文件不存在: {weights_file}。跳过此折。")
            continue

        test_dataset = MultimodalDataset(test_path, class_map)
        loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

        audio_dim = test_dataset[0][0].shape[1]
        text_dim = test_dataset[0][1].shape[1]

        model = MultiModalNet(
            audio_dim=audio_dim, text_dim=text_dim, num_classes=len(class_names),
            fusion_type="gated_bi_cross_attention", dropout=DROPOUT, shared_dim=SHARED_DIM
        ).to(device)

        model.load_state_dict(torch.load(weights_file, map_location=device))

        # 推理
        f_true, f_pred, f_probs = test_single_fold(model, loader, device)

        # 1. 收集全局数据 (Pooling)
        global_y_true.extend(f_true)
        global_y_pred.extend(f_pred)
        global_y_probs.extend(f_probs)

        # 2. 计算并保存指标
        fold_metrics = calculate_fold_metrics(f_true, f_pred, f_probs, class_names)
        metrics_history.append(fold_metrics)

        # 3. 【新增】为当前 Fold 绘制并保存图片
        print(f"  > Saving plots for Fold {fold_idx}...")
        cm_fold = confusion_matrix(f_true, f_pred)
        plot_confusion_matrix(
            cm_fold, class_names,
            filename=f"fold{fold_idx}_confusion_matrix.png",  # 区分文件名
            save_dir=PLOT_DIR
        )
        plot_roc_curves(
            f_true, f_probs, class_names,
            filename=f"fold{fold_idx}_roc_curves.png",
            save_dir=PLOT_DIR
        )
        plot_pr_curves(
            f_true, f_probs, class_names,
            filename=f"fold{fold_idx}_pr_curves.png",
            save_dir=PLOT_DIR
        )

    # ==========================================
    # 全局评估 & 绘图
    # ==========================================
    if not metrics_history:
        print("[ERROR] 没有有效的测试结果。")
        return

    # 1. 打印带方差的报表
    print_cv_report(metrics_history, class_names)

    # 2. 绘制全局汇总图表 (Global Pooling)
    print(f"[INFO] 正在生成全局汇总图表 (Global) ...")

    cm_global = confusion_matrix(global_y_true, global_y_pred)
    plot_confusion_matrix(
        cm_global, class_names,
        filename="cv_global_confusion_matrix.png",  # 加上 global 前缀
        save_dir=PLOT_DIR
    )

    plot_roc_curves(
        global_y_true, global_y_probs, class_names,
        filename="cv_global_roc_curves.png",
        save_dir=PLOT_DIR
    )
    plot_pr_curves(
        global_y_true, global_y_probs, class_names,
        filename="cv_global_pr_curves.png",
        save_dir=PLOT_DIR
    )
    print(f"[SUCCESS] 所有图片已保存至 {PLOT_DIR} 文件夹。")


if __name__ == "__main__":
    main()