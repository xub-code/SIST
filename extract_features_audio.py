# extract_features_audio.py
import os
import argparse
import numpy as np
import librosa
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import warnings
warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000
SEGMENT_SECONDS = 6
OVERLAP_SECONDS = 2
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS


# -----------------------------
# 音频切片（带重叠 + mask）
# -----------------------------
def slice_audio(file_path, sr=SAMPLE_RATE, segment_len=SEGMENT_SAMPLES,
                overlap_seconds=OVERLAP_SECONDS):
    """
    将音频切成多个 segment_len 的片段；支持 overlap_seconds 重叠。
    返回 segments(list[np.ndarray]) 与 masks(list[np.ndarray])，mask 记录每个采样点是否为真实语音。
    """
    audio, _ = librosa.load(file_path, sr=sr)
    overlap_samples = int(sr * overlap_seconds)
    hop_size = max(1, segment_len - overlap_samples)

    segments, masks = [], []
    for start in range(0, len(audio), hop_size):
        end = start + segment_len
        seg = audio[start:end]

        if len(seg) < segment_len:
            real_len = len(seg)
            seg = np.pad(seg, (0, segment_len - len(seg)), mode="constant")
        else:
            real_len = segment_len

        mask = np.zeros(segment_len, dtype=np.float32)
        mask[:real_len] = 1.0

        segments.append(seg.astype(np.float32))
        masks.append(mask.astype(np.float32))

        if end >= len(audio):
            break

    return segments, masks


# -----------------------------
# 层聚合策略
# -----------------------------
def combine_hidden_layers(hidden_states, strategy="last4avg"):
    """
    根据策略把多层 hidden_states 合成一个 [1, T, D] 的张量。
    hidden_states: tuple，通常为 [conv_feat, layer1, ..., layerL]
    仅对 encoder 层聚合（跳过 conv_feat）。

    可选策略：
      - "last":           仅最后一层
      - "last4avg":       最后4层平均（推荐默认）
      - "last4weighted":  最后4层加权(0.4,0.3,0.2,0.1)
      - "upperhalf":      上半层平均（更偏语义）
      - "allweighted":    全层按线性权重（越高层权重越大）
      - "concat_last4":   最后4层拼接（输出维度=4*D；注意下游维度会变）
    返回: [1, T, D] 或 [1, T, 4D]（concat 时）
    """
    # 通常 hidden_states[0] 是特征提取器输出，后面是 encoder 层
    encoder_layers = hidden_states[1:]
    L = len(encoder_layers)

    if L == 0:  # 极少数模型实现差异防御
        return hidden_states[-1]

    if strategy == "last":
        return encoder_layers[-1]  # [1,T,D]

    elif strategy == "last4avg":
        if L >= 4:
            last4 = encoder_layers[-4:]
        else:
            last4 = encoder_layers
        stacked = torch.stack(last4, dim=0)  # [K,1,T,D]
        return stacked.mean(dim=0)           # [1,T,D]

    elif strategy == "last4weighted":
        if L >= 4:
            last4 = encoder_layers[-4:]
        else:
            last4 = encoder_layers
        K = len(last4)
        # 简单线性权重: 最近层权重最大
        weights = torch.linspace(1.0, 2.0, steps=K, device=last4[0].device)
        weights = weights / weights.sum()
        stacked = torch.stack(last4, dim=0)  # [K,1,T,D]
        w = weights.view(K, 1, 1, 1)
        return (stacked * w).sum(dim=0)      # [1,T,D]

    elif strategy == "upperhalf":
        half_start = L // 2
        upper = encoder_layers[half_start:]
        stacked = torch.stack(upper, dim=0)
        return stacked.mean(dim=0)

    elif strategy == "allweighted":
        # 层索引从 1..L，越高层权重越大
        idx = torch.arange(1, L + 1, device=encoder_layers[0].device, dtype=torch.float32)
        weights = idx / idx.sum()
        stacked = torch.stack(encoder_layers, dim=0)
        w = weights.view(L, 1, 1, 1)
        return (stacked * w).sum(dim=0)

    elif strategy == "concat_last4":
        if L >= 4:
            last4 = encoder_layers[-4:]
        else:
            # 若层数不足4，就把所有层 concat
            last4 = encoder_layers
        return torch.cat(last4, dim=-1)  # [1,T,4D 或 <4D]

    else:
        raise ValueError(f"Unknown layer_strategy={strategy}")


# -----------------------------
# 提取单个文件的 embedding（改进版）
# -----------------------------
def extract_audio_embedding(processor, model, device, file_path,
                            layer_strategy="last4avg", pool="seg_stat"):
    """对单个音频文件提取分段 embedding（使用 mask 精确截断到真实语音长度）。

    Args:
        processor: Wav2Vec2Processor
        model:     Wav2Vec2Model
        device:    torch.device
        file_path: 音频路径
        layer_strategy: 层聚合策略
        pool: 段内池化方式
    """
    segments, masks = slice_audio(file_path)
    seg_embs = []

    for seg, mask in zip(segments, masks):
        # seg: [segment_len] 已经在 slice_audio 中补零到固定长度
        # mask: [segment_len] 0/1，1 表示真实语音位置
        inputs = processor(seg, sampling_rate=SAMPLE_RATE,
                           return_tensors="pt", padding=True,
                           return_attention_mask=True)
        input_values = inputs.input_values.to(device)

        # === 关键改动：用 slice_audio 提供的 mask 来构造更精确的 attention_mask ===
        real_len = int(mask.sum())  # 真实语音采样点数
        if real_len <= 0:
            # 理论上不会发生；保险起见，退回到整段长度
            real_len = seg.shape[0]

        # attention_mask 形状与 processor 生成的一致：[1, seq_len]
        attention_mask = torch.zeros_like(inputs.attention_mask, device=device)
        max_len = attention_mask.size(1)
        real_len = min(real_len, max_len)
        attention_mask[:, :real_len] = 1
        input_len = attention_mask.sum(dim=1)

        with torch.no_grad():
            outputs = model(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hs = outputs.hidden_states                    # [conv, l1..lL]
            mixed = combine_hidden_layers(hs, strategy=layer_strategy)  # [1,T,D or 4D]
            hidden = mixed.squeeze(0)                     # [T, D or 4D]

        # 用真实语音长度推导特征帧数，避免把补零帧也当成有效语音
        if hasattr(model, "_get_feat_extract_output_lengths"):
            frame_len = int(model._get_feat_extract_output_lengths(input_len).item())
        else:
            frame_len = hidden.shape[0]
        frame_len = min(frame_len, hidden.shape[0])
        valid = hidden[:frame_len]                        # [Tf, D]

        if pool == "seg_mean":
            vec = valid.mean(dim=0)

        elif pool == "seg_stat":
            mu  = valid.mean(dim=0)
            var = valid.var(dim=0, unbiased=False)
            std = torch.sqrt(var + 1e-6)
            vec = torch.cat([mu, std], dim=0)             # [2D]

        elif pool == "seg_quantile":
            p25 = torch.quantile(valid, 0.25, dim=0)
            med = torch.quantile(valid, 0.50, dim=0)
            p75 = torch.quantile(valid, 0.75, dim=0)
            vec = torch.cat([p25, med, p75], dim=0)       # [3D]

        else:
            raise ValueError(f"Unknown pool={pool}")

        seg_embs.append(vec.detach().cpu().numpy().astype(np.float32))

    seg_embs = np.vstack(seg_embs).astype(np.float32)     # [T, D’]
    return seg_embs


# -----------------------------
# 主流程
# -----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 根据命令行参数更新全局切片设置
    global SEGMENT_SECONDS, OVERLAP_SECONDS, SEGMENT_SAMPLES
    SEGMENT_SECONDS = args.segment_seconds
    OVERLAP_SECONDS = args.overlap_seconds
    SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS

    print(f"[INFO] 加载 Wav2Vec2 模型: {args.model_path}")
    processor = Wav2Vec2Processor.from_pretrained(args.model_path, local_files_only=args.local_only)
    model = Wav2Vec2Model.from_pretrained(args.model_path, local_files_only=args.local_only)
    model.to(device).eval()

    # 遍历 splits
    splits = ["train", "val", "test"]
    for split_name in splits:
        split_dir = os.path.join(args.data_dir, split_name)
        if not os.path.exists(split_dir):
            print(f"[WARN] {split_dir} 不存在，跳过")
            continue

        for class_name in sorted(os.listdir(split_dir)):
            class_folder = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_folder):
                continue

            out_class_dir = os.path.join(args.cache_dir, split_name, class_name)
            os.makedirs(out_class_dir, exist_ok=True)

            files = [f for f in os.listdir(class_folder) if f.lower().endswith(".wav")]
            for f in tqdm(files, desc=f"{split_name}/{class_name} [{args.layer_strategy}]"):
                file_path = os.path.join(class_folder, f)
                # out_path = os.path.join(out_class_dir, Path(f).stem + ".npy")
                out_filename = f"audio_{Path(f).stem}.npy"  # 生成 "audio_adrso046.npy"
                out_path = os.path.join(out_class_dir, out_filename)

                if os.path.exists(out_path) and not args.force:
                    continue
                try:
                    emb = extract_audio_embedding(
                        processor, model, device, file_path,
                        layer_strategy=args.layer_strategy,
                        pool=args.pool
                    )
                    np.save(out_path, emb)
                except Exception as e:
                    print(f"[ERROR] {file_path}: {e}")

    print("[INFO] Done.")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # "D:\Datasets\Audio\ADReSS2020"
    # "D:\Datasets\Audio\ADReSSo2021"
    # "D:\Datasets\Audio\NCMMSC2021"
    parser.add_argument("--data_dir", type=str, default=r"D:\Datasets\Audio\ADReSSo2021",
                        help="数据根目录 (train/val/test 子文件夹)")
    parser.add_argument("--model_path", type=str, default="D:\pretrain\wav2vec2-base",
                        help="本地或在线 wav2vec2 模型路径（如 facebook/wav2vec2-base）")
    parser.add_argument("--cache_dir", type=str, default=r"ADReSSo2021",
                        help="embedding 保存目录")
    parser.add_argument("--force", action="store_true",
                        help="是否覆盖已有 .npy 文件")
    parser.add_argument("--local_only", action="store_true",
                        help="仅本地加载（若本地无缓存会报错）")
    parser.add_argument("--layer_strategy", type=str, default="last4avg",
                        choices=["last", "last4avg", "last4weighted", "upperhalf", "allweighted", "concat_last4"],
                        help="Wav2Vec2 层聚合策略")
    parser.add_argument("--pool", type=str, default="seg_mean",
                        choices=["seg_mean", "seg_stat", "seg_quantile"],
                        help="段内池化方式：均值/均值+方差/分位数(25,50,75)")
    parser.add_argument("--segment_seconds", type=int, default=6)
    parser.add_argument("--overlap_seconds", type=int, default=3)

    args = parser.parse_args()
    main(args)
