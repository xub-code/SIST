import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from dataset import MultimodalDataset
from model import MultiModalNet


# ================= 配置 =================
CLASS_MAP = {"AD": 0, "HC": 1, "MCI": 2}

DEFAULT_SHARED_DIM = 512
DEFAULT_DROPOUT = 0.5
DEFAULT_NUM_CLASSES = 3

# 当前 S4 中固定使用的结构超参数。
AUDIO_LA_HEADS = 4
AUDIO_POOL_HEADS = 4
TEXT_ATTN_HEADS = 4
TEXT_LOW_RANK_DIM = 64
TEXT_ASP_HIDDEN = 256
CROSS_ATTN_HEADS = 4
JCR_REDUCTION = 8


# ================= 参数统计 =================
def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """使用 PyTorch 原生参数树统计总参数量和可训练参数量。"""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def parameter_breakdown(model: MultiModalNet) -> Dict[str, int]:
    """按 S4 主要模块统计参数量，便于复核。"""
    return {
        "audio_subnet": sum(
            parameter.numel()
            for parameter in model.audio_subnet.parameters()
        ),
        "text_subnet": sum(
            parameter.numel()
            for parameter in model.text_subnet.parameters()
        ),
        "audio_projection": sum(
            parameter.numel()
            for parameter in model.audio_proj_pre.parameters()
        ),
        "text_projection": sum(
            parameter.numel()
            for parameter in model.text_proj_pre.parameters()
        ),
        "audio_from_text_attention": sum(
            parameter.numel()
            for parameter in model.a_from_t.parameters()
        ),
        "text_from_audio_attention": sum(
            parameter.numel()
            for parameter in model.t_from_a.parameters()
        ),
        "jcr": sum(
            parameter.numel()
            for parameter in model.jcr_module.parameters()
        ),
        "classifier": sum(
            parameter.numel()
            for parameter in model.classifier.parameters()
        ),
    }


# ================= FLOPs 统计 =================
def audio_subnet_macs(
    sequence_length: int,
    dim: int,
    num_heads: int,
) -> int:
    """
    统计 AudioSubNet 的主要矩阵乘法 MACs。

    包括：
    - LinearAttention Q/K/V/out projections；
    - K^T V、Q(K^T V)、Qz；
    - MultiHeadQueryPooling Q/K/V/out projections；
    - pooling attention score 和 attention-value。

    不统计 LayerNorm、ELU、mask、除法、Dropout 等逐元素操作。
    """
    t = int(sequence_length)
    d = int(dim)
    h = int(num_heads)

    if t <= 0 or d <= 0:
        raise ValueError(
            "Audio sequence length and dimension must be positive."
        )
    if d % h != 0:
        raise ValueError(
            "Audio feature dimension must be divisible by num_heads."
        )

    # LinearAttention Q/K/V/out projections.
    projection_macs = 4 * t * d * d

    # K^T V and Q(K^T V): each T * D^2 / H.
    linear_attention_macs = 2 * t * d * d // h

    # Q z.
    normalization_product_macs = t * d

    # Query pooling MHA, query length=1, key/value length=T.
    # Q/K/V/out projections.
    pooling_projection_macs = (2 * t + 2) * d * d

    # QK^T and attention @ V.
    pooling_attention_macs = 2 * t * d

    return (
        projection_macs
        + linear_attention_macs
        + normalization_product_macs
        + pooling_projection_macs
        + pooling_attention_macs
    )


def text_subnet_macs(
    sequence_length: int,
    dim: int,
    num_heads: int,
    low_rank_dim: int,
    asp_hidden: int,
) -> int:
    """
    统计 TextSubNet 的主要矩阵乘法 MACs。

    包括：
    - Q/K/V/out projections；
    - q_low/k_low；
    - low-rank attention score；
    - original attention score；
    - attention @ V；
    - alpha gate；
    - AttentiveStatsPool scorer、bmm 和 projection。

    不统计 LayerNorm、normalize、softmax、mask、GELU、sqrt 等逐元素操作。
    """
    t = int(sequence_length)
    d = int(dim)
    h = int(num_heads)
    r = int(low_rank_dim)
    a = int(asp_hidden)

    if t <= 0 or d <= 0:
        raise ValueError(
            "Text sequence length and dimension must be positive."
        )
    if d % h != 0:
        raise ValueError(
            "Text feature dimension must be divisible by num_heads."
        )

    projection_macs = 4 * t * d * d
    low_rank_projection_macs = 2 * t * d * r
    low_rank_score_macs = h * t * t * r
    full_attention_macs = 2 * t * t * d
    gate_macs = d * h
    scorer_macs = t * d * a + t * a
    stats_pool_macs = 2 * t * d
    stats_projection_macs = 2 * d * d

    return (
        projection_macs
        + low_rank_projection_macs
        + low_rank_score_macs
        + full_attention_macs
        + gate_macs
        + scorer_macs
        + stats_pool_macs
        + stats_projection_macs
    )


def cross_attention_macs(
    query_length: int,
    key_value_length: int,
    dim: int,
) -> int:
    """
    统计一个 CrossAttnSeqToSeq 的主要矩阵乘法 MACs。

    包括 Q/K/V/out projections、QK^T 和 attention @ V。
    """
    lq = int(query_length)
    lkv = int(key_value_length)
    d = int(dim)

    if lq <= 0 or lkv <= 0 or d <= 0:
        raise ValueError(
            "Query length, key/value length, and dimension must be positive."
        )

    query_side_projection = 2 * lq * d * d
    key_value_projection = 2 * lkv * d * d
    attention_matrix_macs = 2 * lq * lkv * d

    return (
        query_side_projection
        + key_value_projection
        + attention_matrix_macs
    )


def s4_macs_per_sample(
    audio_length: int,
    text_length: int,
    audio_dim: int,
    text_dim: int,
    shared_dim: int,
    num_classes: int,
) -> int:
    """
    统计一条真实样本对应的 S4 主要矩阵乘法 MACs。

    S4:
    AudioSubNet/TextSubNet
    -> pooled feature projections
    -> sequence projections
    -> bidirectional cross-attention
    -> JCR
    -> classifier

    统计口径与最终 S1/S2/S3 完全一致：
    只统计 Linear / matmul / bmm 的乘加运算。
    """
    la = int(audio_length)
    lt = int(text_length)
    da = int(audio_dim)
    dt = int(text_dim)
    ds = int(shared_dim)

    if la <= 0 or lt <= 0:
        raise ValueError(
            "Audio and text sequence lengths must be positive."
        )
    if ds % CROSS_ATTN_HEADS != 0:
        raise ValueError(
            "shared_dim must be divisible by cross-attention heads."
        )
    if ds % JCR_REDUCTION != 0:
        raise ValueError(
            "shared_dim must be divisible by JCR reduction."
        )

    macs = 0

    # 1) Modality-specific representation learning.
    macs += audio_subnet_macs(
        sequence_length=la,
        dim=da,
        num_heads=AUDIO_LA_HEADS,
    )
    macs += text_subnet_macs(
        sequence_length=lt,
        dim=dt,
        num_heads=TEXT_ATTN_HEADS,
        low_rank_dim=TEXT_LOW_RANK_DIM,
        asp_hidden=TEXT_ASP_HIDDEN,
    )

    # 2) Global-vector projections.
    macs += da * ds
    macs += dt * ds

    # 3) Sequence projections using the same shared projection modules.
    # These reuse parameters but incur additional computation.
    macs += la * da * ds
    macs += lt * dt * ds

    # 4) Bidirectional cross-attention.
    macs += cross_attention_macs(
        query_length=la,
        key_value_length=lt,
        dim=ds,
    )
    macs += cross_attention_macs(
        query_length=lt,
        key_value_length=la,
        dim=ds,
    )

    # 5) JCR.
    bottleneck_dim = ds // JCR_REDUCTION
    macs += (4 * ds) * bottleneck_dim
    macs += bottleneck_dim * (4 * ds)

    # 6) Classifier: [2D] -> D -> C.
    macs += (2 * ds) * ds
    macs += ds * num_classes

    return int(macs)


# ================= 数据 =================
def get_feature_shape(path: str) -> Tuple[int, int]:
    """只读取 .npy 元信息，返回 [sequence_length, feature_dim]。"""
    array = np.load(path, mmap_mode="r")
    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2-D feature array, got {array.shape}: {path}"
        )
    return int(array.shape[0]), int(array.shape[1])


def collect_sample_shapes(
    dataset: MultimodalDataset,
) -> List[Tuple[int, int, int, int]]:
    """读取所有成对样本的真实音频/文本序列长度和维度。"""
    shapes: List[Tuple[int, int, int, int]] = []

    for audio_path, text_path, _ in dataset.samples:
        audio_length, audio_dim = get_feature_shape(audio_path)
        text_length, text_dim = get_feature_shape(text_path)

        shapes.append(
            (
                audio_length,
                text_length,
                audio_dim,
                text_dim,
            )
        )

    if not shapes:
        raise RuntimeError(
            "No paired feature samples were found."
        )

    return shapes


# ================= 主程序 =================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Accurate parameter and FLOPs analysis for S4."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="NCMMSC2021",
        help="Feature dataset root containing train/test.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split used for sequence-length-dependent FLOPs.",
    )
    parser.add_argument(
        "--shared_dim",
        type=int,
        default=DEFAULT_SHARED_DIM,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
    )
    parser.add_argument(
        "--save_txt",
        type=str,
        default="s4_complexity.txt",
    )
    args = parser.parse_args()

    split_root = Path(args.data_root) / args.split

    dataset = MultimodalDataset(
        root_dir=split_root,
        class_map=CLASS_MAP,
    )

    sample_shapes = collect_sample_shapes(dataset)

    first_audio_dim = sample_shapes[0][2]
    first_text_dim = sample_shapes[0][3]

    for _, _, audio_dim, text_dim in sample_shapes:
        if audio_dim != first_audio_dim:
            raise RuntimeError(
                "Audio feature dimension is inconsistent."
            )
        if text_dim != first_text_dim:
            raise RuntimeError(
                "Text feature dimension is inconsistent."
            )

    model = MultiModalNet(
        audio_dim=first_audio_dim,
        text_dim=first_text_dim,
        num_classes=DEFAULT_NUM_CLASSES,
        fusion_type="gated_bi_cross_attention",
        dropout=args.dropout,
        shared_dim=args.shared_dim,
    )
    model.eval()

    total_params, trainable_params = count_parameters(model)
    breakdown = parameter_breakdown(model)

    if total_params != trainable_params:
        raise RuntimeError(
            "S4 total parameters and trainable parameters should be identical."
        )

    sample_macs: List[int] = []

    for (
        audio_length,
        text_length,
        audio_dim,
        text_dim,
    ) in sample_shapes:
        sample_macs.append(
            s4_macs_per_sample(
                audio_length=audio_length,
                text_length=text_length,
                audio_dim=audio_dim,
                text_dim=text_dim,
                shared_dim=args.shared_dim,
                num_classes=DEFAULT_NUM_CLASSES,
            )
        )

    mean_macs = float(np.mean(sample_macs))
    mean_flops = 2.0 * mean_macs

    result_lines = [
        "=" * 76,
        "S4 Complexity Analysis",
        "=" * 76,
        f"Data Root                  : {args.data_root}",
        f"Split                      : {args.split}",
        f"Samples                    : {len(sample_shapes)}",
        f"Audio Feature Dim          : {first_audio_dim}",
        f"Text Feature Dim           : {first_text_dim}",
        f"Shared Dim                 : {args.shared_dim}",
        f"Cross-Attention Heads      : {CROSS_ATTN_HEADS}",
        f"JCR Reduction              : {JCR_REDUCTION}",
        f"Fusion                     : Two-Branch + Bi-Cross Attention + JCR",
        "-" * 76,
        f"Total Params               : {total_params:,} ({total_params / 1e6:.6f} M)",
        f"Trainable Params           : {trainable_params:,} ({trainable_params / 1e6:.6f} M)",
        f"AudioSubNet Params         : {breakdown['audio_subnet']:,}",
        f"TextSubNet Params          : {breakdown['text_subnet']:,}",
        f"Audio Projection Params    : {breakdown['audio_projection']:,}",
        f"Text Projection Params     : {breakdown['text_projection']:,}",
        f"Audio<-Text Attn Params    : {breakdown['audio_from_text_attention']:,}",
        f"Text<-Audio Attn Params    : {breakdown['text_from_audio_attention']:,}",
        f"JCR Params                 : {breakdown['jcr']:,}",
        f"Classifier Params          : {breakdown['classifier']:,}",
        "-" * 76,
        f"Mean MACs / Sample         : {mean_macs:,.0f} ({mean_macs / 1e9:.6f} G)",
        f"Mean FLOPs / Sample        : {mean_flops:,.0f} ({mean_flops / 1e9:.6f} G)",
        f"Min FLOPs / Sample         : {2.0 * min(sample_macs) / 1e9:.6f} G",
        f"Max FLOPs / Sample         : {2.0 * max(sample_macs) / 1e9:.6f} G",
        "-" * 76,
        "FLOPs convention          : 2 x MACs",
        "Counted operations        : Linear / matmul / bmm",
        "Excluded operations       : masked mean, normalization, activation, mask, softmax, dropout",
        "=" * 76,
    ]

    text = "\n".join(result_lines)
    print("\n" + text + "\n")

    output_path = Path(args.save_txt)
    output_path.write_text(
        text,
        encoding="utf-8",
    )
    print(
        f"[INFO] Saved to: {output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
