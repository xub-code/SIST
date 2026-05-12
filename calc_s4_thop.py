import os
import argparse
import warnings

import torch
from torch.utils.data import DataLoader
from thop import profile

from dataset import MultimodalDataset, collate_fn
from model import MultiModalNet

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def human_readable_count(num):
    """将数字格式化为更易读的 K / M / G / T 单位字符串。"""
    num = float(num)
    if num >= 1e12:
        return f"{num / 1e12:.3f}T"
    elif num >= 1e9:
        return f"{num / 1e9:.3f}G"
    elif num >= 1e6:
        return f"{num / 1e6:.3f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.3f}K"
    else:
        return f"{num:.3f}"


def count_trainable_params(model):
    """统计可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(test_dataset, class_names, device, shared_dim, dropout):
    """构建 S4 模型。"""
    model = MultiModalNet(
        audio_dim=test_dataset[0][0].shape[1],
        text_dim=test_dataset[0][1].shape[1],
        num_classes=len(class_names),
        fusion_type="gated_bi_cross_attention",
        dropout=dropout,
        shared_dim=shared_dim,
    ).to(device)
    return model


def get_one_batch(data_root, split, class_map, batch_size):
    """加载一个 batch，用于 THOP 统计。"""
    dataset = MultimodalDataset(os.path.join(data_root, split), class_map)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    batch = next(iter(loader))
    return dataset, batch


def main():
    parser = argparse.ArgumentParser(description="使用 THOP 统计 S4 模型参数量和 FLOPs")
    parser.add_argument("--data_root", type=str, default="NCMMSC2021", help="数据根目录")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="统计使用的数据划分")
    parser.add_argument("--batch_size", type=int, default=4, help="用于 profile 的 batch size")
    parser.add_argument("--shared_dim", type=int, default=512, help="模型 shared_dim")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout")
    parser.add_argument("--weights_file", type=str, default=os.path.join("weights_2024", "best.pth"), help="权重路径")
    parser.add_argument("--use_cpu", action="store_true", help="强制使用 CPU 统计")
    parser.add_argument("--save_txt", type=str, default="s4_complexity_thop.txt", help="结果保存 txt 文件名")
    args = parser.parse_args()

    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")

    class_map = {"AD": 0, "HC": 1, "MCI": 2}
    class_names = [k for k, v in sorted(class_map.items(), key=lambda item: item[1])]

    print(f"[INFO] 使用设备: {device}")
    print(f"[INFO] 正在加载数据集: {os.path.join(args.data_root, args.split)}")

    dataset, batch = get_one_batch(args.data_root, args.split, class_map, args.batch_size)
    audio_x, audio_mask, text_x, text_mask, _ = batch

    audio_x = audio_x.to(device)
    audio_mask = audio_mask.to(device)
    text_x = text_x.to(device)
    text_mask = text_mask.to(device)

    model = build_model(
        test_dataset=dataset,
        class_names=class_names,
        device=device,
        shared_dim=args.shared_dim,
        dropout=args.dropout,
    )
    model.eval()

    if os.path.exists(args.weights_file):
        state = torch.load(args.weights_file, map_location=device)
        model.load_state_dict(state)
        print(f"[INFO] 已加载权重: {args.weights_file}")
    else:
        print(f"[WARN] 未找到权重文件: {args.weights_file}")
        print("[WARN] 将基于随机初始化模型统计 Params / FLOPs（参数量不受影响，FLOPs 也可正常统计）")

    with torch.no_grad():
        macs, params = profile(
            model,
            inputs=(audio_x, audio_mask, text_x, text_mask),
            verbose=False,
        )

    # 常见写法：FLOPs ≈ 2 * MACs
    flops = macs * 2
    trainable_params = count_trainable_params(model)

    params_m = params / 1e6
    trainable_params_m = trainable_params / 1e6
    macs_g = macs / 1e9
    flops_g = flops / 1e9

    result_lines = [
        "=" * 72,
        "S4 模型复杂度统计（THOP）",
        "=" * 72,
        f"Data Root              : {args.data_root}",
        f"Split                  : {args.split}",
        f"Batch Size             : {args.batch_size}",
        f"Device                 : {device}",
        f"Shared Dim             : {args.shared_dim}",
        f"Dropout                : {args.dropout}",
        f"Fusion Type            : gated_bi_cross_attention",
        f"Weights File           : {args.weights_file}",
        "-" * 72,
        f"Total Params           : {int(params):,} ({params_m:.2f}M)",
        f"Trainable Params       : {int(trainable_params):,} ({trainable_params_m:.2f}M)",
        f"MACs                   : {int(macs):,} ({macs_g:.2f}G)",
        f"FLOPs (= 2 x MACs)     : {int(flops):,} ({flops_g:.2f}G)",
        "=" * 72,
    ]
    print("\n" + "\n".join(result_lines) + "\n")

    with open(args.save_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))

    print(f"[INFO] 结果已保存到: {args.save_txt}")


if __name__ == "__main__":
    main()