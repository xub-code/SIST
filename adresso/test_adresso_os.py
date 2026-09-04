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
from dataset import MultimodalDataset, collate_fn
from model import MultiModalNet
from plot_utils import plot_confusion_matrix, plot_roc_curves, plot_pr_curves
import warnings

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def format_percent(val):
    return "-" if val is None else f"{val * 100:.2f}%"


def calculate_metrics_and_print(y_true, y_pred, y_probs, class_names, seed):
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
            "prec": Precision, "rec": Recall, "f1": F1,
            "auc": AUC, "ap": AP, "support": Support
        }
        prec_list.append(Precision)
        rec_list.append(Recall)
        f1_list.append(F1)
        supports.append(Support)

    macro_auc = roc_auc_score(y_onehot, y_probs, average="macro")
    macro_ap = average_precision_score(y_onehot, y_probs, average="macro")

    metrics['macro'] = {
        "prec": np.mean(prec_list), "rec": np.mean(rec_list), "f1": np.mean(f1_list),
        "auc": macro_auc, "ap": macro_ap, "support": np.sum(supports)
    }

    print("\n" + "=" * 95)
    print(f"【FINAL EVALUATION REPORT】 Seed: {seed}")
    print("=" * 95)
    print("{:<12} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<8}".format(
        "Class", "Acc", "Precision", "Recall", "F1-Score", "AUC", "AP", "Support"))
    print("-" * 95)

    for cls in class_names:
        m = metrics[cls]
        print("{:<12} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<8}".format(
            cls, "-", format_percent(m['prec']), format_percent(m['rec']),
            format_percent(m['f1']), format_percent(m['auc']), format_percent(m['ap']), m['support']
        ))
    print("-" * 95)
    mm = metrics['macro']
    print("{:<12} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<8}".format(
        "Macro avg", "-", format_percent(mm['prec']), format_percent(mm['rec']),
        format_percent(mm['f1']), format_percent(mm['auc']), format_percent(mm['ap']), mm['support']
    ))
    print("-" * 95)
    print(f"Overall Acc : {format_percent(global_acc)}")
    print("=" * 95 + "\n")

    return cm


def test_model(model, loader, device, class_names, seed):
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    print("[INFO] Running Inference...")
    with torch.no_grad():
        for audio_x, audio_mask, text_x, text_mask, y in tqdm(loader, desc="Testing"):
            audio_x, text_x, y = audio_x.to(device), text_x.to(device), y.to(device)
            audio_mask, text_mask = audio_mask.to(device), text_mask.to(device)

            # [修改点] 适配新模型，只接收一个返回值
            logits = model(audio_x, audio_mask, text_x, text_mask)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
            y_probs.extend(F.softmax(logits, dim=1).cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    cm = calculate_metrics_and_print(y_true, y_pred, y_probs, class_names, seed)
    return y_true, y_probs, cm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_ROOT = r"ADReSSo2021"
    BATCH_SIZE = 4
    SHARED_DIM = 128
    DROPOUT = 0.1

    WEIGHTS_FILE = os.path.join("weights_adresso_os", "best_4.pth")

    class_map = {"ad": 0, "cn": 1}
    class_names = [k for k, v in sorted(class_map.items(), key=lambda item: item[1])]

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading Data from: {DATA_ROOT}")

    test_dataset = MultimodalDataset(os.path.join(DATA_ROOT, "test"), class_map)
    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = MultiModalNet(
        audio_dim=test_dataset[0][0].shape[1],
        text_dim=test_dataset[0][1].shape[1],
        num_classes=len(class_names),
        fusion_type="gated_bi_cross_attention",
        dropout=DROPOUT,
        shared_dim=SHARED_DIM
    ).to(device)

    if os.path.exists(WEIGHTS_FILE):
        model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
        print(f"[INFO] Weights loaded from: {WEIGHTS_FILE}")
    else:
        print(f"[ERROR] Weights file NOT found at: {WEIGHTS_FILE}")
        print("请先运行 train.py 进行训练。")
        exit()

    y_true, y_probs, cm = test_model(model, loader, device, class_names, args.seed)

    if not os.path.exists("plot"): os.makedirs("plot")
    plot_confusion_matrix(cm, class_names)
    plot_roc_curves(y_true, y_probs, class_names)
    plot_pr_curves(y_true, y_probs, class_names)
    print("[INFO] Plots saved to 'plot' directory.")