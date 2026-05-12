import os
import sys
import random
import torch
import numpy as np
import argparse
import warnings
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn

# 导入基础模块
from dataset import MultimodalDataset, collate_fn
from model import MultiModalNet

# 忽略警告并设置环境
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# 交叉验证数据根目录
DATA_ROOT_CV = "ADReSSo2021_5CV"
SAVE_DIR = "weights_cv"
os.makedirs(SAVE_DIR, exist_ok=True)


def set_seed(seed: int):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, epochs, fold_idx):
    model.train()
    total_loss, correct = 0.0, 0
    # 进度条显示 Fold 信息
    loader_tqdm = tqdm(loader, desc=f"[Fold {fold_idx}] Epoch {epoch + 1}/{epochs}", file=sys.stdout)

    for audio_x, audio_mask, text_x, text_mask, y in loader_tqdm:
        audio_x, text_x, y = audio_x.to(device), text_x.to(device), y.to(device)
        audio_mask, text_mask = audio_mask.to(device), text_mask.to(device)

        optimizer.zero_grad()

        # Forward (只接收 logits)
        logits = model(audio_x, audio_mask, text_x, text_mask)

        # Loss (纯净 CE)
        loss = criterion(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * audio_x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()

        loader_tqdm.set_postfix({
            "loss": f"{total_loss / len(loader.dataset):.4f}",
            "acc": f"{correct / len(loader.dataset):.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
        })
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def eval_epoch(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for audio_x, audio_mask, text_x, text_mask, y in loader:
            audio_x, text_x, y = audio_x.to(device), text_x.to(device), y.to(device)
            audio_mask, text_mask = audio_mask.to(device), text_mask.to(device)

            logits = model(audio_x, audio_mask, text_x, text_mask)
            correct += (logits.argmax(dim=1) == y).sum().item()

    acc = correct / len(loader.dataset)
    return acc


def main():
    parser = argparse.ArgumentParser()
    # 这里的 seed 主要控制每一折的模型初始化随机性
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}, Global Seed: {args.seed}")

    # 检查数据目录
    if not os.path.exists(DATA_ROOT_CV):
        raise RuntimeError(f"未找到交叉验证数据目录: {DATA_ROOT_CV}。请先运行数据集切分脚本。")

    # --- 超参数配置 (使用之前 Grid Search 找到的最佳参数) ---
    CONFIG = {
        "epochs": 60,
        "batch_size": 16,  # 假设搜索出的最佳值
        "lr": 0.0002,  # 假设搜索出的最佳值
        "weight_decay": 0.0001,
        "dropout": 0.3,  # 假设搜索出的最佳值
        "shared_dim": 768,  # 假设搜索出的最佳值
        "fusion_type": "gated_bi_cross_attention"
    }

    print(f"[INFO] Training Config: {CONFIG}")
    class_map = {"ad": 0, "cn": 1}

    # 记录每一折的最佳准确率
    fold_best_accs = []

    # ==========================================
    # 5-Fold Loop
    # ==========================================
    for fold_idx in range(1, 6):
        print(f"\n{'=' * 20} Start Training Fold {fold_idx} {'=' * 20}")

        # 1. 每一折都要重置种子，保证可复现性
        set_seed(args.seed)
        g = torch.Generator()
        g.manual_seed(args.seed)

        # 2. 路径构建
        fold_path = os.path.join(DATA_ROOT_CV, f"fold{fold_idx}")
        train_path = os.path.join(fold_path, "train")
        test_path = os.path.join(fold_path, "test")

        # 3. 数据加载
        train_dataset = MultimodalDataset(train_path, class_map)
        val_dataset = MultimodalDataset(test_path, class_map)

        train_loader = DataLoader(
            train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
            num_workers=0, collate_fn=collate_fn, drop_last=True, generator=g
        )
        val_loader = DataLoader(
            val_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
            num_workers=0, collate_fn=collate_fn, generator=g
        )

        # 自动获取维度
        audio_dim = train_dataset[0][0].shape[-1]
        text_dim = train_dataset[0][1].shape[-1]

        # 4. 模型初始化 (每个 Fold 必须是新模型)
        model = MultiModalNet(
            audio_dim=audio_dim,
            text_dim=text_dim,
            num_classes=len(class_map),
            fusion_type=CONFIG["fusion_type"],
            dropout=CONFIG["dropout"],
            shared_dim=CONFIG["shared_dim"]
        ).to(device)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

        # Scheduler
        total_steps = CONFIG["epochs"] * len(train_loader)
        warmup_steps = int(0.1 * total_steps)
        scheduler = LambdaLR(optimizer, lambda s: 0.5 * (1.0 + np.cos(
            np.pi * float(s - warmup_steps) / float(max(1, total_steps - warmup_steps))))
        if s >= warmup_steps else float(s + 1) / float(max(1, warmup_steps)))

        # 5. 训练当前 Fold
        best_acc_this_fold = 0.0

        for epoch in range(CONFIG["epochs"]):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, scheduler,
                device, epoch, CONFIG["epochs"], fold_idx
            )
            val_acc = eval_epoch(model, val_loader, device)

            # 保存当前 Fold 的最佳权重
            if val_acc > best_acc_this_fold:
                best_acc_this_fold = val_acc
                save_path = os.path.join(SAVE_DIR, f"best_fold{fold_idx}.pth")
                torch.save(model.state_dict(), save_path)
                print(f"  [Fold {fold_idx}] New Best Acc: {val_acc:.4f} -> Saved to {save_path}")

        print(f"✅ Fold {fold_idx} Finished. Best Acc: {best_acc_this_fold:.4f}")
        fold_best_accs.append(best_acc_this_fold)

    # ==========================================
    # Final Report
    # ==========================================
    mean_acc = np.mean(fold_best_accs)
    std_acc = np.std(fold_best_accs)

    print(f"\n{'=' * 50}")
    print(f" 5-Fold CV Completed ")
    print(f"{'=' * 50}")
    print(f"Accuracies: {fold_best_accs}")
    print(f"Mean Acc: {mean_acc:.4f}")
    print(f"Std Dev : {std_acc:.4f}")

    with open("cv_results.txt", "w") as f:
        f.write(f"Fold Accs: {fold_best_accs}\n")
        f.write(f"Mean: {mean_acc:.4f}\n")
        f.write(f"Std : {std_acc:.4f}\n")


if __name__ == "__main__":
    main()