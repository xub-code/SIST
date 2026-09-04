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

from dataset import MultimodalDataset, collate_fn
from model import MultiModalNet

# 忽略警告并设置环境
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def set_seed(seed: int):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, epochs):
    model.train()
    total_loss, correct = 0.0, 0
    loader_tqdm = tqdm(loader, desc=f"Epoch [{epoch + 1}/{epochs}] Train", file=sys.stdout)

    for audio_x, audio_mask, text_x, text_mask, y in loader_tqdm:
        audio_x, text_x, y = audio_x.to(device), text_x.to(device), y.to(device)
        audio_mask, text_mask = audio_mask.to(device), text_mask.to(device)

        optimizer.zero_grad()

        # [修改点] 模型现在只返回 logits
        logits = model(audio_x, audio_mask, text_x, text_mask)

        # [修改点] 使用标准交叉熵计算 Loss
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


def eval_epoch(model, loader, criterion, device, stage="Val", epoch=None):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for audio_x, audio_mask, text_x, text_mask, y in loader:
            audio_x, text_x, y = audio_x.to(device), text_x.to(device), y.to(device)
            audio_mask, text_mask = audio_mask.to(device), text_mask.to(device)

            # [修改点] 只接收 logits
            logits = model(audio_x, audio_mask, text_x, text_mask)
            loss = criterion(logits, y)

            total_loss += loss.item() * audio_x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()

    acc = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader.dataset)
    if epoch is not None:
        print(f"[{stage}] Epoch {epoch + 1}: Loss={avg_loss:.4f}, Acc={acc:.4f}")
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()

    # --- 1. 配置区域 (Configuration) ---
    SEED = args.seed
    set_seed(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 路径配置
    DATA_ROOT = r"NCMMSC2021"
    SAVE_DIR = "weights"
    LOG_DIR = "runs"
    os.makedirs(SAVE_DIR, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=LOG_DIR)

    # [修改点] 纯净的超参数配置，删除了 loss 相关参数
    CONFIG = {
        "epochs": 60,
        "batch_size": 4,
        "lr": 0.0001,
        "weight_decay": 0.0001,
        "dropout": 0.5,
        "shared_dim": 512,
        "fusion_type": "gated_bi_cross_attention"
    }

    class_map = {"AD": 0, "HC": 1,"MCI":2}

    print(f"[INFO] Device: {device}, Seed: {SEED}")
    print(f"[INFO] Config: {CONFIG}")

    # --- 2. 数据加载 ---
    train_dataset = MultimodalDataset(os.path.join(DATA_ROOT, "train"), class_map)
    val_dataset = MultimodalDataset(os.path.join(DATA_ROOT, "test"), class_map)

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=0, collate_fn=collate_fn, drop_last=True, generator=g
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=0, collate_fn=collate_fn, generator=g
    )

    # 自动获取维度
    if len(train_dataset) > 0:
        audio_dim = train_dataset[0][0].shape[-1]
        text_dim = train_dataset[0][1].shape[-1]
    else:
        raise ValueError("Dataset is empty.")

    # --- 3. 模型与优化器 ---
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
    # 预热余弦退火调度器
    total_steps = CONFIG["epochs"] * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # --- 4. 训练循环 ---
    print(f"[INFO] Start Training...")
    best_val_acc = 0.0

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch, CONFIG["epochs"]
        )
        val_loss, val_acc = eval_epoch(
            model, val_loader, criterion, device,
            stage="Val", epoch=epoch
        )

        # 记录日志
        tb_writer.add_scalar("Loss/Train", train_loss, epoch)
        tb_writer.add_scalar("Acc/Val", val_acc, epoch)

        # 保存策略
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best.pth"))
            print(f"  >>> Best Model Saved! (Val Acc: {best_val_acc:.4f})")

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "last.pth"))

    print(f"[INFO] Finished. Best Val Acc: {best_val_acc:.4f}")

    # 记录结果到文件
    with open("experiment_results.txt", "a") as f:
        f.write(f"Seed={SEED}, Acc={best_val_acc:.4f}\n")

    tb_writer.close()


if __name__ == "__main__":
    main()