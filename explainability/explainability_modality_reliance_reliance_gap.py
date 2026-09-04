"""Aggregate leave-one-unit-out modality sensitivity over five model runs.

For each sample and run, the target is the class predicted from the complete
input. Removing one speech window or one text clause produces a change in the
predicted-class decision score

    D = z_c - logsumexp(z_q, q != c) = log(P_c / (1 - P_c)).

The speech/text effect is the mean absolute ``delta_D`` over all units of that
modality. The reported relative reliance is the modality effect divided by the
sum of the speech and text effects. Probability changes are retained only as
diagnostics and must not be reported as ``Mean |delta_D|``.

The five runs are first aggregated within each independent test sample. Group
means and sample standard deviations are then computed across test samples.
The script also records whether the predicted class is stable across all five
runs, because the run-specific predicted class defines the LOU target.
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from dataset import MultimodalDataset
from extract_features_text import (
    TOKENIZER_MAX_LEN,
    chunk_text,
    combine_hidden_layers,
    normalize_text,
)
from model import MultiModalNet


# =============================================================================
# Configuration
# 直接修改本区域后运行本文件，不使用命令行参数。
# =============================================================================
DATA_ROOT = Path("NCMMSC2021")
TEXT_SOURCE_ROOT = Path(r"D:\Dataset\NCMMSC2021_text_xunfei")
TEXT_MODEL_PATH = Path(r"D:\pretrain\bert-base-chinese")
WEIGHT_TEMPLATE = "weights_{seed}/best.pth"
OUTPUT_DIR = Path("explainability_modality_reliance")

SEEDS = [2024, 42, 0, 1, 123]
CLASS_MAP = {"AD": 0, "HC": 1, "MCI": 2}
CLASS_NAMES = [
    name for name, index in sorted(CLASS_MAP.items(), key=lambda item: item[1])
]

FUSION_TYPE = "gated_bi_cross_attention"
SHARED_DIM = 512
DROPOUT = 0.5

TEXT_LAYER_STRATEGY = "last4avg"
TEXT_POOL = "seg_mean"
TEXT_MAX_LEN = 256
TEXT_STRIDE = 64
LOCAL_FILES_ONLY = True

FEATURE_ATOL = 1e-5
FEATURE_RTOL = 1e-4
OCCLUSION_BATCH_SIZE = 16
RELIANCE_EPS = 1e-12

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """固定随机性，保证不同运行的解释结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def derive_sample_id(feature_path: str, prefix: str) -> str:
    """从 audio_xxx.npy 或 text_xxx.npy 中恢复样本 ID。"""
    stem = Path(feature_path).stem
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected feature filename: {feature_path}")
    return stem[len(prefix):]


def resolve_raw_text_path(
    text_root: Path,
    class_name: str,
    sample_id: str,
) -> Path:
    """
    按确定性顺序寻找测试集原始转录文本。

    不进行全目录模糊搜索，避免同名文件跨 split 错配。
    若找到多个候选文件则直接报错，不做隐式选择。
    """
    filename = f"{sample_id}.txt"
    candidates = [
        text_root / "test" / class_name / filename,
        text_root / class_name / filename,
        text_root / "test" / filename,
        text_root / filename,
    ]

    existing: List[Path] = []
    seen = set()
    for path in candidates:
        if not path.is_file():
            continue
        key = str(path.resolve())
        if key not in seen:
            existing.append(path)
            seen.add(key)

    if len(existing) == 1:
        return existing[0]

    if len(existing) > 1:
        raise RuntimeError(
            f"Multiple raw transcripts found for {class_name}/{sample_id}: "
            + "; ".join(str(path) for path in existing)
        )

    raise FileNotFoundError(
        f"Raw transcript not found for {class_name}/{sample_id}. "
        f"Checked under: {text_root}"
    )


def read_text(path: Path) -> str:
    """读取 UTF-8 文本；不可解码字符按原特征提取代码的策略忽略。"""
    return path.read_text(encoding="utf-8", errors="ignore")


def split_text_clauses(text: str) -> List[str]:
    """
    按标点划分文本子句，并保持原始字符顺序与标点不变。

    解释单元：
    - 中文/英文逗号、句号、问号、感叹号、分号、冒号和省略号；
    - “、”不作为默认边界，避免将并列枚举过度切碎。

    不进行固定字符数合并，不重新拼接标点。
    """
    normalized = normalize_text(text)
    if not normalized:
        return []

    pattern = re.compile(r".+?(?:……|…+|[，,。.!！？?；;：:]+|$)")
    clauses = [
        match.group(0)
        for match in pattern.finditer(normalized)
        if match.group(0)
    ]

    if "".join(clauses) != normalized:
        raise RuntimeError(
            "Clause segmentation changed transcript content unexpectedly."
        )

    return clauses


def remove_one_clause(
    clauses: Sequence[str],
    remove_index: int,
) -> str:
    """仅删除指定子句，其余文本保持原顺序和原标点。"""
    if remove_index < 0 or remove_index >= len(clauses):
        raise IndexError(f"remove_index out of range: {remove_index}")
    return "".join(
        clause
        for index, clause in enumerate(clauses)
        if index != remove_index
    )


def validate_paths() -> Dict[int, Path]:
    """检查正式实验依赖的目录和五个模型权重。"""
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Data root not found: {DATA_ROOT}")
    if not TEXT_SOURCE_ROOT.is_dir():
        raise FileNotFoundError(
            f"Text source root not found: {TEXT_SOURCE_ROOT}"
        )
    if not TEXT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Text model path not found: {TEXT_MODEL_PATH}"
        )

    weight_paths: Dict[int, Path] = {}
    for seed in SEEDS:
        weight_path = Path(WEIGHT_TEMPLATE.format(seed=seed))
        if not weight_path.is_file():
            raise FileNotFoundError(
                f"Weight file not found for seed={seed}: {weight_path}"
            )
        weight_paths[seed] = weight_path

    return weight_paths


def validate_dataset_samples(dataset: MultimodalDataset) -> None:
    """检查测试集样本配对、标签、数组形状和特征维度。"""
    expected_audio_dim = None
    expected_text_dim = None

    for index, (audio_path, text_path, label) in enumerate(dataset.samples):
        audio = np.load(audio_path, allow_pickle=False)
        text = np.load(text_path, allow_pickle=False)

        if audio.ndim != 2 or text.ndim != 2:
            raise RuntimeError(
                f"Sample {index} must contain 2D feature arrays, "
                f"got audio={audio.shape}, text={text.shape}."
            )
        if audio.shape[0] < 2:
            raise RuntimeError(
                f"Sample {index} has only {audio.shape[0]} speech unit(s). "
                "Single-unit LOU would remove all valid speech input."
            )
        if text.shape[0] < 1:
            raise RuntimeError(
                f"Sample {index} has no valid text feature row."
            )

        if expected_audio_dim is None:
            expected_audio_dim = int(audio.shape[1])
            expected_text_dim = int(text.shape[1])
        if int(audio.shape[1]) != expected_audio_dim:
            raise RuntimeError(
                f"Inconsistent speech feature dimension at sample {index}: "
                f"{audio.shape[1]} != {expected_audio_dim}"
            )
        if int(text.shape[1]) != expected_text_dim:
            raise RuntimeError(
                f"Inconsistent text feature dimension at sample {index}: "
                f"{text.shape[1]} != {expected_text_dim}"
            )

        class_name = Path(audio_path).parent.name
        if class_name not in CLASS_MAP:
            raise RuntimeError(f"Unexpected class folder: {class_name}")
        if int(label) != CLASS_MAP[class_name]:
            raise RuntimeError(
                f"Label mismatch at sample {index}: "
                f"folder={class_name}, label={label}"
            )

        audio_id = derive_sample_id(audio_path, "audio_")
        text_id = derive_sample_id(text_path, "text_")
        if audio_id != text_id:
            raise RuntimeError(
                f"Audio/Text sample ID mismatch at sample {index}: "
                f"{audio_id} vs {text_id}"
            )


def build_model(
    audio_dim: int,
    text_dim: int,
    weight_path: Path,
) -> MultiModalNet:
    """
    按正式训练/测试配置构建模型并严格加载 state_dict。

    train.py 保存的是 model.state_dict()，因此不保留其他旧 checkpoint 接口。
    """
    model = MultiModalNet(
        audio_dim=audio_dim,
        text_dim=text_dim,
        num_classes=len(CLASS_MAP),
        fusion_type=FUSION_TYPE,
        dropout=DROPOUT,
        shared_dim=SHARED_DIM,
    ).to(DEVICE)

    state_dict = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def extract_text_embedding_from_text(
    text: str,
    tokenizer,
    text_encoder,
) -> np.ndarray:
    """
    在内存中复现 extract_features_text.py 的文本特征提取逻辑。

    目的：
    - 不创建临时 txt 文件；
    - 原始文本和遮挡文本使用同一套 BERT、layer、pool、max_len、stride；
    - 不修改训练时的文本特征提取定义。
    """
    normalized = normalize_text(text)

    hidden_size = int(text_encoder.config.hidden_size)
    output_dim = (
        hidden_size * 4
        if TEXT_LAYER_STRATEGY == "concat_last4"
        else hidden_size
    )

    if not normalized:
        if TEXT_POOL == "seg_mean":
            return np.zeros((1, output_dim), dtype=np.float32)
        if TEXT_POOL == "seg_stat":
            return np.zeros((1, output_dim * 2), dtype=np.float32)
        if TEXT_POOL == "seg_quantile":
            return np.zeros((1, output_dim * 3), dtype=np.float32)
        raise ValueError(f"Unknown text pool: {TEXT_POOL}")

    encoding = tokenizer(
        normalized,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
        max_length=TOKENIZER_MAX_LEN,
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    chunks = chunk_text(
        input_ids,
        attention_mask,
        max_len=TEXT_MAX_LEN,
        stride=TEXT_STRIDE,
    )

    segment_embeddings: List[np.ndarray] = []

    with torch.inference_mode():
        for ids_chunk, mask_chunk in chunks:
            outputs = text_encoder(
                input_ids=ids_chunk,
                attention_mask=mask_chunk,
                output_hidden_states=True,
            )
            mixed_layer = combine_hidden_layers(
                outputs.hidden_states,
                strategy=TEXT_LAYER_STRATEGY,
            )
            hidden = mixed_layer.squeeze(0)
            valid_length = int(mask_chunk.sum().item())
            valid_hidden = hidden[:valid_length, :]

            if TEXT_POOL == "seg_mean":
                vector = valid_hidden.mean(dim=0)
            elif TEXT_POOL == "seg_stat":
                mean = valid_hidden.mean(dim=0)
                std = torch.sqrt(
                    valid_hidden.var(dim=0, unbiased=False) + 1e-6
                )
                vector = torch.cat([mean, std], dim=0)
            elif TEXT_POOL == "seg_quantile":
                p25 = torch.quantile(valid_hidden, 0.25, dim=0)
                p50 = torch.quantile(valid_hidden, 0.50, dim=0)
                p75 = torch.quantile(valid_hidden, 0.75, dim=0)
                vector = torch.cat([p25, p50, p75], dim=0)
            else:
                raise ValueError(f"Unknown text pool: {TEXT_POOL}")

            segment_embeddings.append(
                vector.detach().cpu().numpy().astype(np.float32)
            )

    return np.vstack(segment_embeddings).astype(np.float32)


def verify_original_text_feature(
    raw_text_path: Path,
    stored_text_feature_path: str,
    tokenizer,
    text_encoder,
) -> Tuple[Tuple[int, ...], float]:
    """
    重新编码原始转录，并与训练/测试实际使用的 text_*.npy 严格核对。

    若不一致则终止正式 LOU，避免 baseline 与 perturbation 来自不同文本管线。
    """
    stored = np.load(
        stored_text_feature_path,
        allow_pickle=False,
    ).astype(np.float32)

    reencoded = extract_text_embedding_from_text(
        read_text(raw_text_path),
        tokenizer,
        text_encoder,
    )

    if stored.shape != reencoded.shape:
        raise RuntimeError(
            f"Text feature shape mismatch for {raw_text_path.name}: "
            f"stored={stored.shape}, reencoded={reencoded.shape}"
        )

    difference = stored.astype(np.float64) - reencoded.astype(np.float64)
    max_abs_error = (
        float(np.max(np.abs(difference)))
        if difference.size
        else 0.0
    )

    if not np.allclose(
        stored,
        reencoded,
        atol=FEATURE_ATOL,
        rtol=FEATURE_RTOL,
    ):
        mean_abs_error = float(np.mean(np.abs(difference)))
        rmse = float(np.sqrt(np.mean(difference * difference)))

        stored_flat = stored.reshape(-1).astype(np.float64)
        reencoded_flat = reencoded.reshape(-1).astype(np.float64)
        denominator = (
            np.linalg.norm(stored_flat)
            * np.linalg.norm(reencoded_flat)
        )
        cosine = (
            float(np.dot(stored_flat, reencoded_flat) / denominator)
            if denominator > 0.0
            else np.nan
        )

        raise RuntimeError(
            f"Text feature mismatch for {raw_text_path.name}. "
            f"stored_shape={stored.shape}, "
            f"reencoded_shape={reencoded.shape}, "
            f"max_abs_error={max_abs_error:.8g}, "
            f"mean_abs_error={mean_abs_error:.8g}, "
            f"rmse={rmse:.8g}, cosine={cosine:.8g}. "
            "Check transcript source, BERT checkpoint, layer strategy, "
            "pooling, max_len, and stride."
        )

    return tuple(stored.shape), max_abs_error


def verify_and_index_transcripts(
    dataset: MultimodalDataset,
    tokenizer,
    text_encoder,
) -> Tuple[List[Dict[str, object]], pd.DataFrame]:
    """
    为全部测试样本建立文本索引，并在正式 LOU 前完成特征一致性校验。
    """
    records: List[Dict[str, object]] = []
    verification_rows: List[Dict[str, object]] = []

    iterator = tqdm(
        range(len(dataset)),
        desc="Verifying text features",
        ncols=110,
    )

    for index in iterator:
        audio_path, text_path, label = dataset.samples[index]
        sample_id = derive_sample_id(audio_path, "audio_")
        class_name = Path(audio_path).parent.name

        raw_text_path = resolve_raw_text_path(
            TEXT_SOURCE_ROOT,
            class_name,
            sample_id,
        )

        feature_shape, max_abs_error = verify_original_text_feature(
            raw_text_path,
            text_path,
            tokenizer,
            text_encoder,
        )

        records.append(
            {
                "sample_index": index,
                "sample_id": sample_id,
                "class_name": class_name,
                "label": int(label),
                "audio_feature_path": audio_path,
                "text_feature_path": text_path,
                "raw_text_path": raw_text_path,
            }
        )

        verification_rows.append(
            {
                "sample_index": index,
                "sample_id": sample_id,
                "class": class_name,
                "text_feature_rows": int(feature_shape[0]),
                "text_feature_dim": int(feature_shape[1]),
                "max_abs_error": max_abs_error,
            }
        )

    return records, pd.DataFrame(verification_rows)


def prepare_lou_inputs(
    dataset: MultimodalDataset,
    records: Sequence[Dict[str, object]],
    tokenizer,
    text_encoder,
) -> List[Dict[str, object]]:
    """
    预计算所有文本子句遮挡后的 BERT 特征。

    BERT 在该任务中为冻结特征提取器，因此同一扰动文本的特征可供五个
    SIST seed 模型共同使用，不需要重复编码五次。
    """
    prepared_samples: List[Dict[str, object]] = []

    iterator = tqdm(
        records,
        desc="Preparing text LOU features",
        ncols=110,
    )

    for record in iterator:
        index = int(record["sample_index"])
        audio_x, text_x, label_tensor = dataset[index]

        raw_text = read_text(Path(record["raw_text_path"]))
        clauses = split_text_clauses(raw_text)

        if not clauses:
            raise RuntimeError(
                f"No text clause found for sample {record['sample_id']}."
            )

        occluded_text_features: List[torch.Tensor] = []
        for clause_index in range(len(clauses)):
            perturbed_text = remove_one_clause(
                clauses,
                clause_index,
            )
            feature = extract_text_embedding_from_text(
                perturbed_text,
                tokenizer,
                text_encoder,
            )

            if feature.ndim != 2:
                raise RuntimeError(
                    f"Perturbed text feature must be 2D for "
                    f"{record['sample_id']} clause {clause_index}, "
                    f"got {feature.shape}."
                )
            if int(feature.shape[1]) != int(text_x.shape[1]):
                raise RuntimeError(
                    f"Perturbed text feature dimension mismatch for "
                    f"{record['sample_id']} clause {clause_index}: "
                    f"{feature.shape[1]} != {text_x.shape[1]}"
                )

            occluded_text_features.append(
                torch.from_numpy(
                    feature.astype(np.float32, copy=False)
                )
            )

        prepared_samples.append(
            {
                **record,
                "audio_x": audio_x.float(),
                "text_x": text_x.float(),
                "label": int(label_tensor.item()),
                "clauses": clauses,
                "occluded_text_features": occluded_text_features,
            }
        )

    return prepared_samples


def pad_feature_batch(
    features: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """补齐变长文本特征并生成有效位置 mask。"""
    if not features:
        raise ValueError("features must not be empty.")

    feature_dim = int(features[0].shape[1])
    max_length = max(int(feature.shape[0]) for feature in features)

    batch_x = torch.zeros(
        len(features),
        max_length,
        feature_dim,
        dtype=torch.float32,
    )
    batch_mask = torch.zeros(
        len(features),
        max_length,
        dtype=torch.bool,
    )

    for row, feature in enumerate(features):
        if feature.ndim != 2:
            raise RuntimeError(
                f"Perturbed text feature must be 2D, got {feature.shape}."
            )
        if int(feature.shape[1]) != feature_dim:
            raise RuntimeError(
                "Inconsistent perturbed text feature dimensions."
            )

        length = int(feature.shape[0])
        batch_x[row, :length] = feature
        batch_mask[row, :length] = True

    return batch_x, batch_mask


def predicted_class_decision_score(
    logits: torch.Tensor,
    target_class: int,
) -> torch.Tensor:
    """
    计算原预测类别相对于其余类别的稳定决策分数：

        D = z_c - logsumexp(z_q, q != c)
          = log(P_c / (1 - P_c))

    直接从 logits 计算，避免 softmax 概率接近 0/1 时的数值饱和。
    """
    if logits.ndim != 2:
        raise ValueError(
            f"logits must be [B, C], got {tuple(logits.shape)}"
        )
    if target_class < 0 or target_class >= int(logits.shape[1]):
        raise IndexError(
            f"target_class={target_class} is outside valid class range."
        )

    target_logits = logits[:, target_class]
    other_mask = torch.ones(
        logits.shape[1],
        dtype=torch.bool,
        device=logits.device,
    )
    other_mask[target_class] = False

    return target_logits - torch.logsumexp(
        logits[:, other_mask],
        dim=1,
    )


def baseline_prediction(
    model: MultiModalNet,
    audio_x: torch.Tensor,
    text_x: torch.Tensor,
) -> Tuple[int, float, float, np.ndarray]:
    """计算完整输入下的原预测类别、概率、决策分数和 logits。"""
    audio_batch = audio_x.unsqueeze(0).to(DEVICE)
    text_batch = text_x.unsqueeze(0).to(DEVICE)

    audio_mask = torch.ones(
        (1, audio_x.shape[0]),
        dtype=torch.bool,
        device=DEVICE,
    )
    text_mask = torch.ones(
        (1, text_x.shape[0]),
        dtype=torch.bool,
        device=DEVICE,
    )

    with torch.inference_mode():
        logits = model(
            audio_batch,
            audio_mask,
            text_batch,
            text_mask,
        )
        probabilities = F.softmax(logits, dim=1)

    pred = int(torch.argmax(logits, dim=1).item())
    pred_prob = float(probabilities[0, pred].item())
    decision_score = float(
        predicted_class_decision_score(logits, pred)[0].item()
    )
    logits_np = (
        logits[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=True)
    )

    return pred, pred_prob, decision_score, logits_np


def speech_lou_effects(
    model: MultiModalNet,
    audio_x: torch.Tensor,
    text_x: torch.Tensor,
    target_class: int,
    baseline_prob: float,
    baseline_score: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对每个 speech feature unit 做一次 LOU。

    与原 explainability.py 一致：
    - 对应特征行置零；
    - 同时将该位置 mask 设为 False。
    """
    num_units = int(audio_x.shape[0])
    probability_drops = np.empty(num_units, dtype=np.float64)
    score_drops = np.empty(num_units, dtype=np.float64)

    for start in range(0, num_units, OCCLUSION_BATCH_SIZE):
        indices = list(
            range(
                start,
                min(start + OCCLUSION_BATCH_SIZE, num_units),
            )
        )
        batch_size = len(indices)

        audio_batch = audio_x.unsqueeze(0).repeat(
            batch_size,
            1,
            1,
        )
        audio_mask = torch.ones(
            batch_size,
            num_units,
            dtype=torch.bool,
        )

        row_indices = torch.arange(batch_size)
        unit_indices = torch.tensor(
            indices,
            dtype=torch.long,
        )
        audio_batch[row_indices, unit_indices, :] = 0.0
        audio_mask[row_indices, unit_indices] = False

        text_batch = text_x.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        text_mask = torch.ones(
            batch_size,
            text_x.shape[0],
            dtype=torch.bool,
        )

        with torch.inference_mode():
            logits = model(
                audio_batch.to(DEVICE),
                audio_mask.to(DEVICE),
                text_batch.to(DEVICE),
                text_mask.to(DEVICE),
            )
            probabilities = F.softmax(
                logits,
                dim=1,
            )[:, target_class]
            scores = predicted_class_decision_score(
                logits,
                target_class,
            )

        probability_drops[start:start + batch_size] = (
            baseline_prob
            - probabilities.detach().cpu().numpy()
        )
        score_drops[start:start + batch_size] = (
            baseline_score
            - scores.detach().cpu().numpy()
        )

    return probability_drops, score_drops


def text_lou_effects(
    model: MultiModalNet,
    audio_x: torch.Tensor,
    occluded_text_features: Sequence[torch.Tensor],
    target_class: int,
    baseline_prob: float,
    baseline_score: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """逐个移除文本子句，并使用原 BERT 配置重新编码后进行 LOU 前向。"""
    num_units = len(occluded_text_features)
    probability_drops = np.empty(num_units, dtype=np.float64)
    score_drops = np.empty(num_units, dtype=np.float64)

    for start in range(0, num_units, OCCLUSION_BATCH_SIZE):
        feature_batch = occluded_text_features[
            start:min(
                start + OCCLUSION_BATCH_SIZE,
                num_units,
            )
        ]
        batch_size = len(feature_batch)

        text_batch, text_mask = pad_feature_batch(
            feature_batch
        )
        audio_batch = audio_x.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        audio_mask = torch.ones(
            batch_size,
            audio_x.shape[0],
            dtype=torch.bool,
        )

        with torch.inference_mode():
            logits = model(
                audio_batch.to(DEVICE),
                audio_mask.to(DEVICE),
                text_batch.to(DEVICE),
                text_mask.to(DEVICE),
            )
            probabilities = F.softmax(
                logits,
                dim=1,
            )[:, target_class]
            scores = predicted_class_decision_score(
                logits,
                target_class,
            )

        probability_drops[start:start + batch_size] = (
            baseline_prob
            - probabilities.detach().cpu().numpy()
        )
        score_drops[start:start + batch_size] = (
            baseline_score
            - scores.detach().cpu().numpy()
        )

    return probability_drops, score_drops


def validate_lou_effects(
    probability_drops: np.ndarray,
    score_drops: np.ndarray,
    sample_id: str,
    modality: str,
) -> None:
    """检查单模态 LOU 输出的完整性和数值合法性。"""
    if probability_drops.size == 0 or score_drops.size == 0:
        raise RuntimeError(
            f"No {modality} LOU result for sample {sample_id}."
        )
    if probability_drops.shape != score_drops.shape:
        raise RuntimeError(
            f"{modality} LOU output shape mismatch for {sample_id}."
        )
    if not np.all(np.isfinite(probability_drops)):
        raise RuntimeError(
            f"Non-finite {modality} probability drop for {sample_id}."
        )
    if not np.all(np.isfinite(score_drops)):
        raise RuntimeError(
            f"Non-finite {modality} decision-score drop for {sample_id}."
        )
    if np.any(np.abs(probability_drops) > 1.0 + 1e-6):
        raise RuntimeError(
            f"Invalid {modality} probability drop for {sample_id}."
        )


def analyze_sample_for_seed(
    model: MultiModalNet,
    sample: Dict[str, object],
    seed: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    计算一个样本在一个随机种子模型下的完整 LOU 结果。

    正式模态依赖使用 |ΔD|；
    ΔP 同时保存，用于诊断和后续局部可视化。
    """
    audio_x = sample["audio_x"]
    text_x = sample["text_x"]
    sample_id = str(sample["sample_id"])

    (
        pred,
        baseline_prob,
        baseline_score,
        baseline_logits,
    ) = baseline_prediction(
        model,
        audio_x,
        text_x,
    )

    (
        speech_probability_drops,
        speech_score_drops,
    ) = speech_lou_effects(
        model,
        audio_x,
        text_x,
        pred,
        baseline_prob,
        baseline_score,
    )

    (
        text_probability_drops,
        text_score_drops,
    ) = text_lou_effects(
        model,
        audio_x,
        sample["occluded_text_features"],
        pred,
        baseline_prob,
        baseline_score,
    )

    validate_lou_effects(
        speech_probability_drops,
        speech_score_drops,
        sample_id,
        "speech",
    )
    validate_lou_effects(
        text_probability_drops,
        text_score_drops,
        sample_id,
        "text",
    )

    speech_probability_effect = float(
        np.mean(np.abs(speech_probability_drops))
    )
    text_probability_effect = float(
        np.mean(np.abs(text_probability_drops))
    )

    speech_effect = float(
        np.mean(np.abs(speech_score_drops))
    )
    text_effect = float(
        np.mean(np.abs(text_score_drops))
    )
    total_effect = speech_effect + text_effect

    reliance_defined = bool(
        np.isfinite(total_effect)
        and total_effect > RELIANCE_EPS
    )

    if reliance_defined:
        speech_reliance = speech_effect / total_effect
        text_reliance = text_effect / total_effect
        if not np.isclose(
            speech_reliance + text_reliance,
            1.0,
            atol=1e-10,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"Reliance normalization failed for {sample_id}."
            )
    else:
        speech_reliance = np.nan
        text_reliance = np.nan

    probability_saturated = bool(
        speech_probability_effect <= 1e-15
        and text_probability_effect <= 1e-15
        and total_effect > RELIANCE_EPS
    )

    true_label = int(sample["label"])
    true_class = CLASS_NAMES[true_label]
    pred_class = CLASS_NAMES[pred]

    sample_row = {
        "sample_index": int(sample["sample_index"]),
        "sample_id": sample_id,
        "seed": int(seed),
        "true_label": true_label,
        "true_class": true_class,
        "pred_label": pred,
        "pred_class": pred_class,
        "correct": bool(pred == true_label),
        "baseline_pred_prob": baseline_prob,
        "baseline_decision_score": baseline_score,
        "baseline_logits": "|".join(
            f"{value:.10g}"
            for value in baseline_logits
        ),
        "speech_unit_count": int(
            len(speech_score_drops)
        ),
        "text_unit_count": int(
            len(text_score_drops)
        ),
        "speech_mean_abs_delta_p": speech_probability_effect,
        "text_mean_abs_delta_p": text_probability_effect,
        "speech_mean_abs_delta_p_pp": (
            speech_probability_effect * 100.0
        ),
        "text_mean_abs_delta_p_pp": (
            text_probability_effect * 100.0
        ),
        "speech_mean_abs_delta_score": speech_effect,
        "text_mean_abs_delta_score": text_effect,
        "speech_reliance": speech_reliance,
        "text_reliance": text_reliance,
        "speech_reliance_pct": (
            speech_reliance * 100.0
            if reliance_defined
            else np.nan
        ),
        "text_reliance_pct": (
            text_reliance * 100.0
            if reliance_defined
            else np.nan
        ),
        "reliance_defined": reliance_defined,
        "probability_saturated": probability_saturated,
    }

    unit_rows: List[Dict[str, object]] = []

    for unit_index, (delta_p, delta_score) in enumerate(
        zip(
            speech_probability_drops,
            speech_score_drops,
        )
    ):
        unit_rows.append(
            {
                "sample_index": int(sample["sample_index"]),
                "sample_id": sample_id,
                "seed": int(seed),
                "true_class": true_class,
                "pred_class": pred_class,
                "modality": "Speech",
                "unit_index": int(unit_index),
                "unit_text": "",
                "baseline_pred_prob": baseline_prob,
                "baseline_decision_score": baseline_score,
                "delta_p": float(delta_p),
                "abs_delta_p": float(abs(delta_p)),
                "delta_p_pp": float(delta_p * 100.0),
                "abs_delta_p_pp": float(abs(delta_p) * 100.0),
                "delta_score": float(delta_score),
                "abs_delta_score": float(abs(delta_score)),
            }
        )

    clauses = sample["clauses"]
    for unit_index, (delta_p, delta_score) in enumerate(
        zip(
            text_probability_drops,
            text_score_drops,
        )
    ):
        unit_rows.append(
            {
                "sample_index": int(sample["sample_index"]),
                "sample_id": sample_id,
                "seed": int(seed),
                "true_class": true_class,
                "pred_class": pred_class,
                "modality": "Text",
                "unit_index": int(unit_index),
                "unit_text": str(clauses[unit_index]),
                "baseline_pred_prob": baseline_prob,
                "baseline_decision_score": baseline_score,
                "delta_p": float(delta_p),
                "abs_delta_p": float(abs(delta_p)),
                "delta_p_pp": float(delta_p * 100.0),
                "abs_delta_p_pp": float(abs(delta_p) * 100.0),
                "delta_score": float(delta_score),
                "abs_delta_score": float(abs(delta_score)),
            }
        )

    return sample_row, unit_rows


def build_sample_mean_metrics(
    sample_seed_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    对同一样本的五个固定运行先求平均。

    最终测试集/类别统计以独立测试样本为统计单位，
    不把 119×5 个 sample-seed 记录误作独立受试者。
    """
    grouped = sample_seed_df.groupby(
        [
            "sample_index",
            "sample_id",
            "true_label",
            "true_class",
        ],
        sort=False,
        as_index=False,
    )

    sample_mean_df = grouped.agg(
        speech_mean_abs_delta_score=(
            "speech_mean_abs_delta_score",
            "mean",
        ),
        text_mean_abs_delta_score=(
            "text_mean_abs_delta_score",
            "mean",
        ),
        speech_mean_abs_delta_p_pp=(
            "speech_mean_abs_delta_p_pp",
            "mean",
        ),
        text_mean_abs_delta_p_pp=(
            "text_mean_abs_delta_p_pp",
            "mean",
        ),
        speech_reliance_pct=(
            "speech_reliance_pct",
            "mean",
        ),
        text_reliance_pct=(
            "text_reliance_pct",
            "mean",
        ),
        seed_count=(
            "seed",
            "nunique",
        ),
        pred_label_nunique=(
            "pred_label",
            "nunique",
        ),
        predicted_classes=(
            "pred_class",
            lambda values: "|".join(
                sorted({str(value) for value in values})
            ),
        ),
        correct_runs=(
            "correct",
            "sum",
        ),
        reliance_defined_runs=(
            "reliance_defined",
            "sum",
        ),
        saturated_runs=(
            "probability_saturated",
            "sum",
        ),
    )

    sample_mean_df["prediction_stable"] = (
        sample_mean_df["pred_label_nunique"] == 1
    )
    sample_mean_df["all_runs_correct"] = (
        sample_mean_df["correct_runs"] == len(SEEDS)
    )

    # Sample-level reliance gap:
    # positive values indicate relatively higher speech reliance,
    # negative values indicate relatively higher text reliance.
    sample_mean_df["reliance_gap_pct"] = (
        sample_mean_df["speech_reliance_pct"]
        -
        sample_mean_df["text_reliance_pct"]
    )

    required_columns = [
        "speech_mean_abs_delta_score",
        "text_mean_abs_delta_score",
        "speech_mean_abs_delta_p_pp",
        "text_mean_abs_delta_p_pp",
        "speech_reliance_pct",
        "text_reliance_pct",
        "reliance_gap_pct",
    ]

    if not np.all(
        np.isfinite(
            sample_mean_df[required_columns].to_numpy()
        )
    ):
        raise RuntimeError(
            "Non-finite values found in sample-level metrics."
        )

    reliance_sum = (
        sample_mean_df["speech_reliance_pct"]
        + sample_mean_df["text_reliance_pct"]
    )
    if not np.allclose(
        reliance_sum.to_numpy(),
        100.0,
        atol=1e-8,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Sample-level Speech/Text reliance does not sum to 100%."
        )

    return sample_mean_df


def validate_sample_seed_results(
    sample_seed_df: pd.DataFrame,
) -> None:
    """检查五次运行记录的一一对应关系、标签一致性与数值合法性。"""
    required_columns = {
        "sample_index",
        "sample_id",
        "seed",
        "true_label",
        "true_class",
        "pred_label",
        "pred_class",
        "baseline_pred_prob",
        "baseline_decision_score",
        "speech_mean_abs_delta_score",
        "text_mean_abs_delta_score",
        "speech_reliance_pct",
        "text_reliance_pct",
        "reliance_defined",
    }
    missing_columns = sorted(
        required_columns.difference(sample_seed_df.columns)
    )
    if missing_columns:
        raise RuntimeError(
            "Sample-seed results are missing columns: "
            + ", ".join(missing_columns)
        )
    if sample_seed_df.empty:
        raise RuntimeError("Sample-seed results are empty.")

    duplicate_mask = sample_seed_df.duplicated(
        ["sample_id", "seed"],
        keep=False,
    )
    if duplicate_mask.any():
        duplicate_pairs = sample_seed_df.loc[
            duplicate_mask,
            ["sample_id", "seed"],
        ].drop_duplicates()
        raise RuntimeError(
            "Duplicate sample-seed records found:\n"
            + duplicate_pairs.to_string(index=False)
        )

    observed_seeds = set(
        pd.to_numeric(sample_seed_df["seed"], errors="raise")
        .astype(int)
        .unique()
        .tolist()
    )
    expected_seeds = set(SEEDS)
    if observed_seeds != expected_seeds:
        raise RuntimeError(
            f"Observed seeds {sorted(observed_seeds)} do not match "
            f"the required seeds {SEEDS}."
        )

    seeds_by_sample = sample_seed_df.groupby("sample_id")["seed"].agg(
        lambda values: set(int(value) for value in values)
    )
    invalid_samples = seeds_by_sample[
        seeds_by_sample.map(lambda values: values != expected_seeds)
    ]
    if not invalid_samples.empty:
        details = "; ".join(
            f"{sample_id}:{sorted(values)}"
            for sample_id, values in invalid_samples.items()
        )
        raise RuntimeError(
            "Some samples do not contain exactly the five required runs: "
            + details
        )

    consistency_columns = [
        "sample_index",
        "true_label",
        "true_class",
    ]
    consistency = sample_seed_df.groupby("sample_id")[
        consistency_columns
    ].nunique(dropna=False)
    if (consistency > 1).any(axis=None):
        invalid_ids = consistency.index[
            (consistency > 1).any(axis=1)
        ].astype(str).tolist()
        raise RuntimeError(
            "Sample identity or ground-truth labels change across runs: "
            + ", ".join(invalid_ids)
        )

    finite_columns = [
        "baseline_pred_prob",
        "baseline_decision_score",
        "speech_mean_abs_delta_score",
        "text_mean_abs_delta_score",
    ]
    finite_values = sample_seed_df[finite_columns].apply(
        pd.to_numeric,
        errors="raise",
    ).to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(finite_values)):
        raise RuntimeError(
            "Non-finite baseline or decision-score effects were found."
        )

    defined_mask = sample_seed_df["reliance_defined"].astype(bool)
    defined_reliance = sample_seed_df.loc[
        defined_mask,
        ["speech_reliance_pct", "text_reliance_pct"],
    ].apply(pd.to_numeric, errors="raise")
    if not np.all(np.isfinite(defined_reliance.to_numpy(dtype=np.float64))):
        raise RuntimeError("Non-finite defined reliance values were found.")
    if not np.allclose(
        defined_reliance.sum(axis=1).to_numpy(dtype=np.float64),
        100.0,
        atol=1e-8,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Speech and Text reliance do not sum to 100% for all defined rows."
        )


def mean_std(values: pd.Series) -> Tuple[float, float]:
    """返回均值和样本标准差（ddof=1）。"""
    array = pd.to_numeric(
        values,
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if array.size == 0:
        raise RuntimeError(
            "Cannot summarize an empty group."
        )

    mean_value = float(np.mean(array))
    std_value = (
        float(np.std(array, ddof=1))
        if array.size > 1
        else 0.0
    )
    return mean_value, std_value


def format_mean_std(
    mean_value: float,
    std_value: float,
) -> str:
    """论文显示值统一保留两位小数。"""
    return f"{mean_value:.2f}±{std_value:.2f}"


def build_modality_reliance_summary(
    sample_mean_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """生成论文显示版和高精度数值版模态依赖汇总。"""
    display_rows: List[Dict[str, object]] = []
    numeric_rows: List[Dict[str, object]] = []

    groups: List[Tuple[str, str]] = [
        ("Overall", ""),
        ("AD", "AD"),
        ("HC", "HC"),
        ("MCI", "MCI"),
    ]

    for group_name, class_name in groups:
        group_df = (
            sample_mean_df
            if not class_name
            else sample_mean_df[
                sample_mean_df["true_class"] == class_name
            ]
        )

        speech_effect = mean_std(
            group_df["speech_mean_abs_delta_score"]
        )
        text_effect = mean_std(
            group_df["text_mean_abs_delta_score"]
        )
        speech_reliance = mean_std(
            group_df["speech_reliance_pct"]
        )
        text_reliance = mean_std(
            group_df["text_reliance_pct"]
        )
        reliance_gap = mean_std(
            group_df["reliance_gap_pct"]
        )

        display_rows.append(
            {
                "Group": group_name,
                "N": int(len(group_df)),
                "Speech Mean |ΔD|": format_mean_std(
                    *speech_effect
                ),
                "Text Mean |ΔD|": format_mean_std(
                    *text_effect
                ),
                "Speech Reliance (%)": format_mean_std(
                    *speech_reliance
                ),
                "Text Reliance (%)": format_mean_std(
                    *text_reliance
                ),
                "Reliance Gap (%)": format_mean_std(
                    *reliance_gap
                ),
            }
        )

        numeric_rows.append(
            {
                "Group": group_name,
                "N": int(len(group_df)),
                "speech_mean_abs_delta_d_mean": speech_effect[0],
                "speech_mean_abs_delta_d_std": speech_effect[1],
                "text_mean_abs_delta_d_mean": text_effect[0],
                "text_mean_abs_delta_d_std": text_effect[1],
                "speech_reliance_pct_mean": speech_reliance[0],
                "speech_reliance_pct_std": speech_reliance[1],
                "text_reliance_pct_mean": text_reliance[0],
                "text_reliance_pct_std": text_reliance[1],
                "reliance_gap_pct_mean": reliance_gap[0],
                "reliance_gap_pct_std": reliance_gap[1],
            }
        )

    return (
        pd.DataFrame(display_rows),
        pd.DataFrame(numeric_rows),
    )


def save_metadata(
    num_samples: int,
    stable_prediction_samples: int,
) -> None:
    """保存正式实验配置和统计定义，便于后续复核。"""
    metadata = {
        "title_zh": "测试集上的模态依赖统计",
        "dataset": "NCMMSC2021 test set",
        "num_samples": int(num_samples),
        "stable_prediction_samples": int(stable_prediction_samples),
        "seeds": SEEDS,
        "class_map": CLASS_MAP,
        "model": {
            "fusion_type": FUSION_TYPE,
            "shared_dim": SHARED_DIM,
            "dropout": DROPOUT,
            "weight_template": WEIGHT_TEMPLATE,
        },
        "text_feature_extraction": {
            "source_root": str(TEXT_SOURCE_ROOT),
            "model_path": str(TEXT_MODEL_PATH),
            "layer_strategy": TEXT_LAYER_STRATEGY,
            "pool": TEXT_POOL,
            "max_len": TEXT_MAX_LEN,
            "stride": TEXT_STRIDE,
        },
        "lou": {
            "target": (
                "original predicted class of each run"
            ),
            "speech_unit": (
                "one stored speech feature row; "
                "the row is zeroed and its mask is set to False"
            ),
            "text_unit": (
                "one punctuation-delimited transcript clause; "
                "the clause is removed and the remaining text is "
                "re-encoded by the unchanged BERT feature extractor"
            ),
            "decision_score": (
                "D = z_c - logsumexp(z_q, q != c) "
                "= log(P_c/(1-P_c))"
            ),
            "sample_modality_effect": (
                "mean absolute delta_D over all units "
                "of the corresponding modality"
            ),
            "sample_reliance": (
                "modality mean absolute delta_D divided by "
                "the sum of Speech and Text mean absolute delta_D"
            ),
            "reliance_gap": (
                "sample-level Speech Reliance (%) minus Text Reliance (%) "
                "computed before group aggregation"
            ),
            "cross_run_aggregation": (
                "compute per seed first, then average the five "
                "fixed runs within each sample"
            ),
            "group_aggregation": (
                "mean and sample SD across unique test samples; "
                "class rows use ground-truth diagnostic labels"
            ),
            "prediction_stability_diagnostic": (
                "a sample is stable when all five runs predict the same class; "
                "this is recorded because each run's predicted class is its "
                "own LOU target"
            ),
        },
    }

    metadata_path = OUTPUT_DIR / "modality_reliance_metadata.json"
    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def save_raw_results(
    sample_seed_df: pd.DataFrame,
    unit_df: pd.DataFrame,
) -> Tuple[Path, Path, Path, Path]:
    """
    保存完整 LOU 原始结果和诊断结果。

    即使后续发现无法定义 reliance 的样本，
    已完成的 sample-seed 和 unit-level 结果仍会保留。
    """
    sample_seed_path = (
        OUTPUT_DIR / "sample_seed_lou_metrics.csv"
    )
    unit_path = OUTPUT_DIR / "lou_unit_scores.csv"
    saturation_path = (
        OUTPUT_DIR / "probability_saturation_cases.csv"
    )
    undefined_path = (
        OUTPUT_DIR / "undefined_reliance_cases.csv"
    )

    sample_seed_df.to_csv(
        sample_seed_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    unit_df.to_csv(
        unit_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )

    saturation_df = sample_seed_df[
        sample_seed_df["probability_saturated"].astype(bool)
    ].copy()
    saturation_df.to_csv(
        saturation_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )

    undefined_df = sample_seed_df[
        ~sample_seed_df["reliance_defined"].astype(bool)
    ].copy()
    undefined_df.to_csv(
        undefined_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )

    if not saturation_df.empty:
        print(
            f"[WARN] Probability saturation: "
            f"{len(saturation_df)} sample-seed pairs. "
            f"Saved to: {saturation_path}"
        )

    if not undefined_df.empty:
        columns = [
            "sample_id",
            "seed",
            "true_class",
            "pred_class",
            "baseline_pred_prob",
            "baseline_decision_score",
            "speech_mean_abs_delta_score",
            "text_mean_abs_delta_score",
        ]
        details = undefined_df[columns].to_string(
            index=False
        )
        raise RuntimeError(
            "Modality reliance is undefined for one or more "
            "sample-seed pairs even when using the stable decision "
            "score D. No arbitrary 50/50 value is assigned. "
            f"Inspect: {undefined_path}\n{details}"
        )

    return (
        sample_seed_path,
        unit_path,
        saturation_path,
        undefined_path,
    )


def save_summary_results(
    sample_mean_df: pd.DataFrame,
    summary_display: pd.DataFrame,
    summary_numeric: pd.DataFrame,
) -> Tuple[Path, Path, Path, Path, Path]:
    """
    保存样本级均值、高精度数值表，以及两个面向论文使用的汇总 CSV。

    modality_reliance_paper_table.csv 的列顺序和列名固定为：
    Group | N | Speech Mean |ΔD| | Text Mean |ΔD| |
    Speech Reliance (%) | Text Reliance (%) | Reliance Gap (%)

    Reliance Gap is calculated as sample-level Speech Reliance minus
    Text Reliance before group-level mean and standard deviation aggregation.

    其中每个数值单元格采用 mean±SD，保留两位小数，可直接用于论文制表。
    """
    sample_mean_path = (
        OUTPUT_DIR / "sample_mean_lou_metrics.csv"
    )
    summary_path = (
        OUTPUT_DIR / "modality_reliance_summary.csv"
    )
    numeric_path = (
        OUTPUT_DIR / "modality_reliance_summary_numeric.csv"
    )
    paper_table_path = (
        OUTPUT_DIR / "modality_reliance_paper_table.csv"
    )
    prediction_stability_path = (
        OUTPUT_DIR / "sample_prediction_stability.csv"
    )

    sample_mean_df.to_csv(
        sample_mean_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    summary_display.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary_numeric.to_csv(
        numeric_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )

    paper_table = summary_display[
        [
            "Group",
            "N",
            "Speech Mean |ΔD|",
            "Text Mean |ΔD|",
            "Speech Reliance (%)",
            "Text Reliance (%)",
            "Reliance Gap (%)",
        ]
    ].copy()

    paper_table.to_csv(
        paper_table_path,
        index=False,
        encoding="utf-8-sig",
    )

    prediction_columns = [
        "sample_index",
        "sample_id",
        "true_class",
        "seed_count",
        "pred_label_nunique",
        "predicted_classes",
        "prediction_stable",
        "correct_runs",
        "all_runs_correct",
        "reliance_defined_runs",
        "saturated_runs",
    ]
    sample_mean_df[prediction_columns].to_csv(
        prediction_stability_path,
        index=False,
        encoding="utf-8-sig",
    )

    return (
        sample_mean_path,
        summary_path,
        numeric_path,
        paper_table_path,
        prediction_stability_path,
    )


def main() -> None:
    """完整执行文本核验、五次 LOU、样本级汇总和 CSV 导出。"""
    if SEEDS != [2024, 42, 0, 1, 123]:
        raise ValueError(
            "Formal explainability experiment must use "
            "SEEDS = [2024, 42, 0, 1, 123]."
        )
    if OCCLUSION_BATCH_SIZE <= 0:
        raise ValueError(
            "OCCLUSION_BATCH_SIZE must be positive."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    weight_paths = validate_paths()

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Seeds: {SEEDS}")
    print(f"[INFO] Data root: {DATA_ROOT}")
    print(
        f"[INFO] Text source root: "
        f"{TEXT_SOURCE_ROOT}"
    )

    dataset = MultimodalDataset(
        str(DATA_ROOT / "test"),
        CLASS_MAP,
    )
    validate_dataset_samples(dataset)

    first_audio, first_text, _ = dataset[0]
    audio_dim = int(first_audio.shape[1])
    text_dim = int(first_text.shape[1])

    print(f"[INFO] Test samples: {len(dataset)}")
    print(
        f"[INFO] Feature dims: "
        f"speech={audio_dim}, text={text_dim}"
    )

    set_seed(SEEDS[0])

    tokenizer = AutoTokenizer.from_pretrained(
        str(TEXT_MODEL_PATH),
        local_files_only=LOCAL_FILES_ONLY,
    )
    text_encoder = AutoModel.from_pretrained(
        str(TEXT_MODEL_PATH),
        local_files_only=LOCAL_FILES_ONLY,
    ).to(DEVICE)
    text_encoder.eval()

    records, verification_df = verify_and_index_transcripts(
        dataset,
        tokenizer,
        text_encoder,
    )

    verification_path = (
        OUTPUT_DIR / "text_feature_verification.csv"
    )
    verification_df.to_csv(
        verification_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    print(
        f"[OK] Text-feature verification passed for "
        f"all {len(records)} samples."
    )

    prepared_samples = prepare_lou_inputs(
        dataset,
        records,
        tokenizer,
        text_encoder,
    )

    del text_encoder
    del tokenizer
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    sample_seed_rows: List[Dict[str, object]] = []
    unit_rows: List[Dict[str, object]] = []

    for seed in SEEDS:
        print(
            f"\n[INFO] Running LOU analysis "
            f"for seed={seed}"
        )
        set_seed(seed)

        model = build_model(
            audio_dim,
            text_dim,
            weight_paths[seed],
        )

        iterator = tqdm(
            prepared_samples,
            desc=f"LOU seed={seed}",
            ncols=110,
        )
        for sample in iterator:
            sample_row, current_unit_rows = (
                analyze_sample_for_seed(
                    model,
                    sample,
                    seed,
                )
            )
            sample_seed_rows.append(sample_row)
            unit_rows.extend(current_unit_rows)

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    sample_seed_df = pd.DataFrame(
        sample_seed_rows
    )
    unit_df = pd.DataFrame(unit_rows)

    expected_rows = len(dataset) * len(SEEDS)
    if len(sample_seed_df) != expected_rows:
        raise RuntimeError(
            f"Unexpected sample-seed row count: "
            f"{len(sample_seed_df)} != {expected_rows}"
        )

    validate_sample_seed_results(sample_seed_df)

    (
        sample_seed_path,
        unit_path,
        saturation_path,
        undefined_path,
    ) = save_raw_results(
        sample_seed_df,
        unit_df,
    )

    sample_mean_df = build_sample_mean_metrics(
        sample_seed_df
    )
    if len(sample_mean_df) != len(dataset):
        raise RuntimeError(
            f"Unexpected sample-level row count: "
            f"{len(sample_mean_df)} != {len(dataset)}"
        )

    (
        summary_display,
        summary_numeric,
    ) = build_modality_reliance_summary(
        sample_mean_df
    )

    (
        sample_mean_path,
        summary_path,
        numeric_path,
        paper_table_path,
        prediction_stability_path,
    ) = save_summary_results(
        sample_mean_df,
        summary_display,
        summary_numeric,
    )

    stable_prediction_samples = int(
        sample_mean_df["prediction_stable"].sum()
    )
    save_metadata(
        len(dataset),
        stable_prediction_samples,
    )

    print("\n" + "=" * 104)
    print(
        "Test-set modality reliance statistics "
        "based on LOU decision-score changes"
    )
    print("=" * 104)
    print(
        summary_display.to_string(index=False)
    )
    print("=" * 104)

    print(f"[OK] Summary CSV: {summary_path}")
    print(f"[OK] Numeric CSV: {numeric_path}")
    print(f"[OK] Paper-ready CSV: {paper_table_path}")
    print(
        f"[OK] Prediction-stability diagnostics: "
        f"{prediction_stability_path} "
        f"({stable_prediction_samples}/{len(dataset)} stable samples)"
    )
    print(
        f"[OK] Sample-seed metrics: "
        f"{sample_seed_path}"
    )
    print(
        f"[OK] Unit-level LOU scores: "
        f"{unit_path}"
    )
    print(
        f"[OK] Sample-mean metrics: "
        f"{sample_mean_path}"
    )
    print(
        f"[OK] Text verification: "
        f"{verification_path}"
    )
    print(
        f"[OK] Saturation diagnostics: "
        f"{saturation_path}"
    )
    print(
        f"[OK] Undefined-reliance diagnostics: "
        f"{undefined_path}"
    )


if __name__ == "__main__":
    main()
