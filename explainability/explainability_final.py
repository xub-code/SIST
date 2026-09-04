"""Sample-level leave-one-unit-out (LOU) explainability analysis.

This script generates the quantitative values used by Table 12 and the
sample-level evidence panels used by Figure 11. Its perturbation units and
text-feature pipeline are intentionally identical to those in
``explainability_modality_reliance.py``:

- one stored speech-feature row is one speech unit;
- one punctuation-delimited transcript clause is one text unit;
- removing a text clause is followed by re-encoding the remaining transcript
  with the unchanged BERT feature extractor;
- the re-encoded original transcript must match the stored ``text_*.npy``
  feature before any LOU analysis is allowed to run.

The two explainability scripts use different scores for different purposes.
This file ranks local evidence with the signed predicted-class logit drop
``delta_z`` and reports the probability drop from the same unit. The modality
reliance script aggregates mean absolute changes in the stable decision score
``D``. The perturbations are shared; the reported scores are not interchangeable.

For Table 12 and Figure 11, the script automatically selects six correctly
classified test samples that satisfy the display-unit requirement and have the
largest combined top-unit score,
``max(S-delta_z) + max(T-delta_z)``. The selection is therefore reproducible
and must be described as high-evidence illustrative cases rather than random
or population-representative examples. Every selected case must contain at
least three speech units and three text clauses so that Figure 11 consistently
displays independently ranked ``S1``--``S3`` and ``T1``--``T3`` evidence.
"""

import json
import random
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
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


try:
    import soundfile as sf
except Exception as exc:
    sf = None
    warnings.warn(f"soundfile is unavailable: {exc}", RuntimeWarning)

try:
    import librosa
except Exception as exc:
    librosa = None
    warnings.warn(f"librosa is unavailable: {exc}", RuntimeWarning)

try:
    from scipy.io import wavfile
except Exception as exc:
    wavfile = None
    warnings.warn(f"scipy.io.wavfile is unavailable: {exc}", RuntimeWarning)


# =============================================================================
# Configuration
# 直接修改本区域后运行本文件，不使用命令行参数。
# =============================================================================
DATA_ROOT = Path("NCMMSC2021")
AUDIO_SOURCE_ROOT = Path(r"D:\Dataset\Audio\NCMMSC2021")
TEXT_SOURCE_ROOT = Path(r"D:\Dataset\NCMMSC2021_text_xunfei")
TEXT_MODEL_PATH = Path(r"D:\pretrain\bert-base-chinese")
WEIGHT_PATH = Path("weights_2024") / "best.pth"
OUTPUT_DIR = Path("explain_readable_all")

SEED = 2024
CLASS_MAP = {"AD": 0, "HC": 1, "MCI": 2}
CLASS_NAMES = [
    name for name, _ in sorted(CLASS_MAP.items(), key=lambda item: item[1])
]

FUSION_TYPE = "gated_bi_cross_attention"
SHARED_DIM = 512
DROPOUT = 0.5

# 必须与生成训练/测试 text_*.npy 时的配置完全一致。
TEXT_LAYER_STRATEGY = "last4avg"
TEXT_POOL = "seg_mean"
TEXT_MAX_LEN = 256
TEXT_STRIDE = 64
LOCAL_FILES_ONLY = True

FEATURE_ATOL = 1e-5
FEATURE_RTOL = 1e-4
OCCLUSION_BATCH_SIZE = 16

SPEECH_SEGMENT_DURATION = 6.0
SPEECH_SEGMENT_HOP = 3.0
TOP_K_SPEECH = 3
TOP_K_TEXT = 3

FIGURE_SIZE = (1.5, 1.5)
FIGURE_DPI = 600
WAVEFORM_MAX_POINTS = 1600
# False 时仅输出自动选择的六个论文案例；True 时额外输出全测试集图片。
SAVE_ALL_SAMPLE_FIGURES = False

# Table 12/Figure 11 案例选择：
# 只在正确识别且至少具有三个 speech unit 与三个 text clause 的测试样本中，
# 按最强 speech unit 与最强 text clause 的预测类别 logit 下降量之和降序，
# 自动选择前20个高证据案例。
PAPER_CASE_COUNT = 20
PAPER_CASES_REQUIRE_CORRECT = True

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".wma")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Figure style
# =============================================================================
FP_EN = FontProperties(family="Times New Roman")
FP_CN = FontProperties(family="SimSun")
rcParams["axes.unicode_minus"] = False

COL_BG = "#FFFFFF"
COL_BORDER = "#8A94A6"
COL_WAVE = "#3A4452"
COL_WAVE_FILL = "#D0DCEB"
COL_SPEECH_HIGHLIGHT = ("#002D66", "#1A579E", "#4A7FBD")
COL_TEXT_STRIP = ("#002D66", "#1A579E", "#4A7FBD")
COL_TEXT_BOX = "#FFFFFF"
COL_TEXT = "#000000"


def set_seed(seed: int) -> None:
    """固定随机性，使解释结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_configuration() -> None:
    """检查会改变实验定义或导致索引错误的配置。"""
    class_indices = sorted(CLASS_MAP.values())
    if class_indices != list(range(len(CLASS_MAP))):
        raise ValueError("CLASS_MAP values must be contiguous and start at zero.")
    if OCCLUSION_BATCH_SIZE <= 0:
        raise ValueError("OCCLUSION_BATCH_SIZE must be positive.")
    if TOP_K_SPEECH <= 0 or TOP_K_SPEECH > len(COL_SPEECH_HIGHLIGHT):
        raise ValueError(
            "TOP_K_SPEECH must be between 1 and the number of speech colors."
        )
    if TOP_K_TEXT <= 0 or TOP_K_TEXT > len(COL_TEXT_STRIP):
        raise ValueError(
            "TOP_K_TEXT must be between 1 and the number of text colors."
        )
    if SPEECH_SEGMENT_DURATION <= 0.0 or SPEECH_SEGMENT_HOP <= 0.0:
        raise ValueError("Speech segment duration and hop must be positive.")
    if WAVEFORM_MAX_POINTS < 2:
        raise ValueError("WAVEFORM_MAX_POINTS must be at least 2.")
    if PAPER_CASE_COUNT <= 0:
        raise ValueError("PAPER_CASE_COUNT must be positive.")


def validate_paths() -> None:
    """检查 LOU 与 Figure 11 所需的正式数据、模型和原始音频路径。"""
    required_paths = {
        "test data": DATA_ROOT / "test",
        "raw audio root": AUDIO_SOURCE_ROOT,
        "raw transcript root": TEXT_SOURCE_ROOT,
        "text model": TEXT_MODEL_PATH,
        "model weight": WEIGHT_PATH,
    }
    for name, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")


def derive_sample_id(feature_path: str, prefix: str) -> str:
    """从 audio_xxx.npy 或 text_xxx.npy 中恢复样本 ID。"""
    stem = Path(feature_path).stem
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected feature filename: {feature_path}")
    sample_id = stem[len(prefix):]
    if not sample_id:
        raise ValueError(f"Empty sample ID in feature filename: {feature_path}")
    return sample_id


def resolve_raw_text_path(
    text_root: Path,
    class_name: str,
    sample_id: str,
) -> Path:
    """按确定性顺序定位测试集转写；缺失或多重匹配均直接报错。"""
    filename = f"{sample_id}.txt"
    candidates = (
        text_root / "test" / class_name / filename,
        text_root / class_name / filename,
        text_root / "test" / filename,
        text_root / filename,
    )

    existing: List[Path] = []
    seen = set()
    for path in candidates:
        if not path.is_file():
            continue
        resolved = str(path.resolve())
        if resolved not in seen:
            existing.append(path)
            seen.add(resolved)

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


def resolve_raw_audio_path(
    audio_root: Path,
    class_name: str,
    sample_id: str,
) -> Path:
    """按确定性顺序定位原始音频，保证 Figure 11 使用真实波形。"""
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Raw audio root not found: {audio_root}")

    candidate_dirs = (
        audio_root / "test" / class_name,
        audio_root / class_name,
        audio_root / "test",
        audio_root,
    )
    existing: List[Path] = []
    seen = set()

    for directory in candidate_dirs:
        for extension in AUDIO_EXTENSIONS:
            path = directory / f"{sample_id}{extension}"
            if not path.is_file():
                continue
            resolved = str(path.resolve())
            if resolved not in seen:
                existing.append(path)
                seen.add(resolved)

    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise RuntimeError(
            f"Multiple raw audio files found for {class_name}/{sample_id}: "
            + "; ".join(str(path) for path in existing)
        )
    raise FileNotFoundError(
        f"Raw audio not found for {class_name}/{sample_id}. "
        f"Checked under: {audio_root}"
    )


def read_text(path: Path) -> str:
    """读取 UTF-8 文本；不可解码字符沿用原特征提取策略予以忽略。"""
    return path.read_text(encoding="utf-8", errors="ignore")


def split_text_clauses(text: str) -> List[str]:
    """按标点划分子句，同时保持规范化文本的字符顺序和标点不变。"""
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
        raise RuntimeError("Clause segmentation changed transcript content.")
    return clauses


def remove_one_clause(clauses: Sequence[str], remove_index: int) -> str:
    """仅删除指定子句，其余文本保持原顺序和原标点。"""
    if remove_index < 0 or remove_index >= len(clauses):
        raise IndexError(f"remove_index out of range: {remove_index}")
    return "".join(
        clause for index, clause in enumerate(clauses) if index != remove_index
    )


def validate_dataset_samples(dataset: MultimodalDataset) -> Tuple[int, int]:
    """检查样本配对、标签、数组形状、有限值和跨样本特征维度。"""
    if len(dataset) == 0:
        raise RuntimeError("The test dataset is empty.")

    expected_audio_dim: Optional[int] = None
    expected_text_dim: Optional[int] = None
    sample_ids = set()

    for index, (audio_path, text_path, label) in enumerate(dataset.samples):
        audio = np.load(audio_path, allow_pickle=False)
        text = np.load(text_path, allow_pickle=False)

        if audio.ndim != 2 or text.ndim != 2:
            raise RuntimeError(
                f"Sample {index} must contain 2D feature arrays, "
                f"got speech={audio.shape}, text={text.shape}."
            )
        if audio.shape[0] < 2:
            raise RuntimeError(
                f"Sample {index} has only {audio.shape[0]} speech unit(s). "
                "LOU must not remove the only valid speech input."
            )
        if text.shape[0] < 1:
            raise RuntimeError(f"Sample {index} has no valid text feature row.")
        if not np.all(np.isfinite(audio)) or not np.all(np.isfinite(text)):
            raise RuntimeError(f"Non-finite feature value found at sample {index}.")

        audio_dim = int(audio.shape[1])
        text_dim = int(text.shape[1])
        if expected_audio_dim is None:
            expected_audio_dim = audio_dim
            expected_text_dim = text_dim
        if audio_dim != expected_audio_dim:
            raise RuntimeError(
                f"Inconsistent speech feature dimension at sample {index}: "
                f"{audio_dim} != {expected_audio_dim}"
            )
        if text_dim != expected_text_dim:
            raise RuntimeError(
                f"Inconsistent text feature dimension at sample {index}: "
                f"{text_dim} != {expected_text_dim}"
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
                f"Speech/Text sample ID mismatch at sample {index}: "
                f"{audio_id} vs {text_id}"
            )
        if audio_id in sample_ids:
            raise RuntimeError(f"Duplicate sample ID in test set: {audio_id}")
        sample_ids.add(audio_id)

    if expected_audio_dim is None or expected_text_dim is None:
        raise RuntimeError("Unable to determine feature dimensions.")
    return expected_audio_dim, expected_text_dim


def build_model(audio_dim: int, text_dim: int) -> MultiModalNet:
    """按 train.py 的正式配置构建模型并严格加载原始 state_dict。"""
    model = MultiModalNet(
        audio_dim=audio_dim,
        text_dim=text_dim,
        num_classes=len(CLASS_MAP),
        fusion_type=FUSION_TYPE,
        dropout=DROPOUT,
        shared_dim=SHARED_DIM,
    ).to(DEVICE)

    state_dict = torch.load(WEIGHT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def extract_text_embedding_from_text(
    text: str,
    tokenizer,
    text_encoder,
) -> np.ndarray:
    """在内存中严格复现 extract_features_text.py 的文本特征提取流程。"""
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
    if not chunks:
        raise RuntimeError("Text chunking produced no valid chunk.")

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
            if valid_length <= 0:
                raise RuntimeError("A text chunk contains no valid token.")
            valid_hidden = hidden[:valid_length, :]

            if TEXT_POOL == "seg_mean":
                vector = valid_hidden.mean(dim=0)
            elif TEXT_POOL == "seg_stat":
                mean = valid_hidden.mean(dim=0)
                std = torch.sqrt(valid_hidden.var(dim=0, unbiased=False) + 1e-6)
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
    """核验原始转写重编码结果与训练/测试实际使用的特征完全兼容。"""
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
    max_abs_error = float(np.max(np.abs(difference))) if difference.size else 0.0

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
        denominator = np.linalg.norm(stored_flat) * np.linalg.norm(reencoded_flat)
        cosine = (
            float(np.dot(stored_flat, reencoded_flat) / denominator)
            if denominator > 0.0
            else np.nan
        )
        raise RuntimeError(
            f"Text feature mismatch for {raw_text_path.name}. "
            f"stored_shape={stored.shape}, reencoded_shape={reencoded.shape}, "
            f"max_abs_error={max_abs_error:.8g}, "
            f"mean_abs_error={mean_abs_error:.8g}, rmse={rmse:.8g}, "
            f"cosine={cosine:.8g}. Check transcript source, BERT checkpoint, "
            "layer strategy, pooling, max_len, and stride."
        )

    return tuple(stored.shape), max_abs_error


def prepare_lou_inputs(
    dataset: MultimodalDataset,
    tokenizer,
    text_encoder,
) -> Tuple[List[Dict[str, object]], pd.DataFrame]:
    """核验原文本特征，并预计算每个文本子句删除后的 BERT 特征。"""
    prepared_samples: List[Dict[str, object]] = []
    verification_rows: List[Dict[str, object]] = []

    iterator = tqdm(
        range(len(dataset)),
        desc="Preparing text LOU features",
        ncols=110,
    )
    for index in iterator:
        audio_path, text_path, _ = dataset.samples[index]
        sample_id = derive_sample_id(audio_path, "audio_")
        class_name = Path(audio_path).parent.name
        raw_text_path = resolve_raw_text_path(
            TEXT_SOURCE_ROOT,
            class_name,
            sample_id,
        )
        raw_audio_path = resolve_raw_audio_path(
            AUDIO_SOURCE_ROOT,
            class_name,
            sample_id,
        )

        feature_shape, max_abs_error = verify_original_text_feature(
            raw_text_path,
            text_path,
            tokenizer,
            text_encoder,
        )

        audio_x, text_x, label_tensor = dataset[index]
        raw_text = read_text(raw_text_path)
        clauses = split_text_clauses(raw_text)
        if not clauses:
            raise RuntimeError(f"No text clause found for sample {sample_id}.")

        occluded_text_features: List[torch.Tensor] = []
        for clause_index in range(len(clauses)):
            perturbed_text = remove_one_clause(clauses, clause_index)
            feature = extract_text_embedding_from_text(
                perturbed_text,
                tokenizer,
                text_encoder,
            )
            if feature.ndim != 2:
                raise RuntimeError(
                    f"Perturbed text feature must be 2D for {sample_id} "
                    f"clause {clause_index}, got {feature.shape}."
                )
            if int(feature.shape[1]) != int(text_x.shape[1]):
                raise RuntimeError(
                    f"Perturbed text feature dimension mismatch for {sample_id} "
                    f"clause {clause_index}: {feature.shape[1]} != {text_x.shape[1]}"
                )
            if not np.all(np.isfinite(feature)):
                raise RuntimeError(
                    f"Non-finite perturbed text feature for {sample_id} "
                    f"clause {clause_index}."
                )
            occluded_text_features.append(
                torch.from_numpy(feature.astype(np.float32, copy=False))
            )

        prepared_samples.append(
            {
                "sample_index": index,
                "sample_id": sample_id,
                "label": int(label_tensor.item()),
                "raw_audio_path": raw_audio_path,
                "audio_x": audio_x.float(),
                "text_x": text_x.float(),
                "clauses": clauses,
                "occluded_text_features": occluded_text_features,
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

    return prepared_samples, pd.DataFrame(verification_rows)


def select_paper_cases(sample_df: pd.DataFrame) -> pd.DataFrame:
    """选择可完整展示 S1--S3/T1--T3 的联合局部证据最强案例。"""
    required_columns = {
        "sample_id",
        "correct",
        "baseline_pred_prob",
        "speech_unit_count",
        "text_unit_count",
        "speech_top_delta_z",
        "text_top_delta_z",
        "combined_top_delta_z",
    }
    missing_columns = sorted(required_columns.difference(sample_df.columns))
    if missing_columns:
        raise RuntimeError(
            "Cannot select paper cases; missing columns: "
            + ", ".join(missing_columns)
        )

    candidate_df = sample_df.copy()
    if PAPER_CASES_REQUIRE_CORRECT:
        candidate_df = candidate_df[candidate_df["correct"]].copy()
    candidate_df = candidate_df[
        (candidate_df["speech_unit_count"] >= TOP_K_SPEECH)
        & (candidate_df["text_unit_count"] >= TOP_K_TEXT)
    ].copy()

    if len(candidate_df) < PAPER_CASE_COUNT:
        candidate_name = (
            "correctly classified" if PAPER_CASES_REQUIRE_CORRECT else "all"
        )
        raise RuntimeError(
            f"Only {len(candidate_df)} {candidate_name} test samples contain "
            f"at least {TOP_K_SPEECH} speech units and {TOP_K_TEXT} text "
            f"clauses, fewer than PAPER_CASE_COUNT={PAPER_CASE_COUNT}."
        )

    selected_df = candidate_df.sort_values(
        [
            "combined_top_delta_z",
            "speech_top_delta_z",
            "text_top_delta_z",
            "baseline_pred_prob",
            "sample_id",
        ],
        ascending=[False, False, False, False, True],
        kind="stable",
    ).head(PAPER_CASE_COUNT).copy()
    selected_df.insert(0, "selection_rank", range(1, len(selected_df) + 1))
    return selected_df


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
            raise RuntimeError("Inconsistent perturbed text feature dimensions.")
        length = int(feature.shape[0])
        if length <= 0:
            raise RuntimeError("Perturbed text feature has no valid row.")
        batch_x[row, :length] = feature
        batch_mask[row, :length] = True

    return batch_x, batch_mask


def baseline_prediction(
    model: MultiModalNet,
    audio_x: torch.Tensor,
    text_x: torch.Tensor,
) -> Tuple[int, float, float, np.ndarray]:
    """计算完整输入下的原预测类别、概率、类别 logit 和全部 logits。"""
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
        logits = model(audio_batch, audio_mask, text_batch, text_mask)
        probabilities = F.softmax(logits, dim=1)

    pred = int(torch.argmax(logits, dim=1).item())
    pred_prob = float(probabilities[0, pred].item())
    pred_logit = float(logits[0, pred].item())
    logits_np = logits[0].detach().cpu().numpy().astype(np.float64, copy=True)
    return pred, pred_prob, pred_logit, logits_np


def speech_lou_effects(
    model: MultiModalNet,
    audio_x: torch.Tensor,
    text_x: torch.Tensor,
    target_class: int,
    baseline_prob: float,
    baseline_logit: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """逐行移除 speech feature unit，返回 delta_z、delta_p 及遮挡后值。"""
    num_units = int(audio_x.shape[0])
    delta_z = np.empty(num_units, dtype=np.float64)
    delta_p = np.empty(num_units, dtype=np.float64)
    occluded_logits = np.empty(num_units, dtype=np.float64)
    occluded_probs = np.empty(num_units, dtype=np.float64)

    for start in range(0, num_units, OCCLUSION_BATCH_SIZE):
        indices = list(range(start, min(start + OCCLUSION_BATCH_SIZE, num_units)))
        batch_size = len(indices)
        audio_batch = audio_x.unsqueeze(0).repeat(batch_size, 1, 1)
        audio_mask = torch.ones(batch_size, num_units, dtype=torch.bool)
        row_indices = torch.arange(batch_size)
        unit_indices = torch.tensor(indices, dtype=torch.long)
        audio_batch[row_indices, unit_indices, :] = 0.0
        audio_mask[row_indices, unit_indices] = False

        text_batch = text_x.unsqueeze(0).expand(batch_size, -1, -1)
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
            current_logits = logits[:, target_class]
            current_probs = F.softmax(logits, dim=1)[:, target_class]

        logits_np = current_logits.detach().cpu().numpy().astype(np.float64)
        probs_np = current_probs.detach().cpu().numpy().astype(np.float64)
        end = start + batch_size
        occluded_logits[start:end] = logits_np
        occluded_probs[start:end] = probs_np
        delta_z[start:end] = baseline_logit - logits_np
        delta_p[start:end] = baseline_prob - probs_np

    return delta_z, delta_p, occluded_logits, occluded_probs


def text_lou_effects(
    model: MultiModalNet,
    audio_x: torch.Tensor,
    occluded_text_features: Sequence[torch.Tensor],
    target_class: int,
    baseline_prob: float,
    baseline_logit: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """逐个移除文本子句、重新编码，并返回 delta_z、delta_p 及遮挡后值。"""
    num_units = len(occluded_text_features)
    delta_z = np.empty(num_units, dtype=np.float64)
    delta_p = np.empty(num_units, dtype=np.float64)
    occluded_logits = np.empty(num_units, dtype=np.float64)
    occluded_probs = np.empty(num_units, dtype=np.float64)

    for start in range(0, num_units, OCCLUSION_BATCH_SIZE):
        feature_batch = occluded_text_features[
            start:min(start + OCCLUSION_BATCH_SIZE, num_units)
        ]
        batch_size = len(feature_batch)
        text_batch, text_mask = pad_feature_batch(feature_batch)
        audio_batch = audio_x.unsqueeze(0).expand(batch_size, -1, -1)
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
            current_logits = logits[:, target_class]
            current_probs = F.softmax(logits, dim=1)[:, target_class]

        logits_np = current_logits.detach().cpu().numpy().astype(np.float64)
        probs_np = current_probs.detach().cpu().numpy().astype(np.float64)
        end = start + batch_size
        occluded_logits[start:end] = logits_np
        occluded_probs[start:end] = probs_np
        delta_z[start:end] = baseline_logit - logits_np
        delta_p[start:end] = baseline_prob - probs_np

    return delta_z, delta_p, occluded_logits, occluded_probs


def validate_lou_effects(
    delta_z: np.ndarray,
    delta_p: np.ndarray,
    occluded_logits: np.ndarray,
    occluded_probs: np.ndarray,
    sample_id: str,
    modality: str,
) -> None:
    """检查 LOU 输出形状、有限值和概率范围。"""
    arrays = (delta_z, delta_p, occluded_logits, occluded_probs)
    if any(array.size == 0 for array in arrays):
        raise RuntimeError(f"No {modality} LOU result for sample {sample_id}.")
    if len({array.shape for array in arrays}) != 1:
        raise RuntimeError(f"{modality} LOU shape mismatch for {sample_id}.")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise RuntimeError(f"Non-finite {modality} LOU value for {sample_id}.")
    if np.any(np.abs(delta_p) > 1.0 + 1e-6):
        raise RuntimeError(f"Invalid {modality} probability drop for {sample_id}.")
    if np.any(occluded_probs < -1e-7) or np.any(occluded_probs > 1.0 + 1e-7):
        raise RuntimeError(f"Invalid {modality} probability for {sample_id}.")


def select_top_indices(values: np.ndarray, top_k: int) -> List[int]:
    """按有符号下降量从大到小稳定排序；并列时保留较小单元索引。"""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError("values contains a non-finite entry.")
    order = np.argsort(-array, kind="stable")
    return [int(index) for index in order[:min(top_k, array.size)]]


def analyze_sample(
    model: MultiModalNet,
    sample: Dict[str, object],
) -> Tuple[Dict[str, object], Dict[str, object], List[Dict[str, object]]]:
    """计算一个样本的完整 speech/text LOU，并提取两侧最高响应单元。"""
    audio_x = sample["audio_x"]
    text_x = sample["text_x"]
    sample_id = str(sample["sample_id"])

    pred, baseline_prob, baseline_logit, baseline_logits = baseline_prediction(
        model,
        audio_x,
        text_x,
    )

    speech_delta_z, speech_delta_p, speech_occ_z, speech_occ_p = (
        speech_lou_effects(
            model,
            audio_x,
            text_x,
            pred,
            baseline_prob,
            baseline_logit,
        )
    )
    text_delta_z, text_delta_p, text_occ_z, text_occ_p = text_lou_effects(
        model,
        audio_x,
        sample["occluded_text_features"],
        pred,
        baseline_prob,
        baseline_logit,
    )

    validate_lou_effects(
        speech_delta_z,
        speech_delta_p,
        speech_occ_z,
        speech_occ_p,
        sample_id,
        "speech",
    )
    validate_lou_effects(
        text_delta_z,
        text_delta_p,
        text_occ_z,
        text_occ_p,
        sample_id,
        "text",
    )

    speech_top_indices = select_top_indices(speech_delta_z, TOP_K_SPEECH)
    text_top_indices = select_top_indices(text_delta_z, TOP_K_TEXT)
    speech_top_index = speech_top_indices[0]
    text_top_index = text_top_indices[0]

    true_label = int(sample["label"])
    true_class = CLASS_NAMES[true_label]
    pred_class = CLASS_NAMES[pred]

    sample_row: Dict[str, object] = {
        "sample_index": int(sample["sample_index"]),
        "sample_id": sample_id,
        "true_label": true_label,
        "true_class": true_class,
        "pred_label": pred,
        "pred_class": pred_class,
        "correct": bool(pred == true_label),
        "baseline_pred_prob": baseline_prob,
        "baseline_pred_logit": baseline_logit,
        "baseline_logits": "|".join(f"{value:.10g}" for value in baseline_logits),
        "speech_unit_count": int(speech_delta_z.size),
        "text_unit_count": int(text_delta_z.size),
        "figure_display_eligible": bool(
            speech_delta_z.size >= TOP_K_SPEECH
            and text_delta_z.size >= TOP_K_TEXT
        ),
        "speech_top_unit_index": speech_top_index,
        "speech_top_delta_z": float(speech_delta_z[speech_top_index]),
        "speech_top_delta_p": float(speech_delta_p[speech_top_index]),
        "speech_top_delta_p_pp": float(speech_delta_p[speech_top_index] * 100.0),
        "speech_top_occluded_logit": float(speech_occ_z[speech_top_index]),
        "speech_top_occluded_prob": float(speech_occ_p[speech_top_index]),
        "text_top_unit_index": text_top_index,
        "text_top_clause": str(sample["clauses"][text_top_index]),
        "text_top_delta_z": float(text_delta_z[text_top_index]),
        "text_top_delta_p": float(text_delta_p[text_top_index]),
        "text_top_delta_p_pp": float(text_delta_p[text_top_index] * 100.0),
        "text_top_occluded_logit": float(text_occ_z[text_top_index]),
        "text_top_occluded_prob": float(text_occ_p[text_top_index]),
        "combined_top_delta_z": float(
            speech_delta_z[speech_top_index] + text_delta_z[text_top_index]
        ),
    }

    plot_data: Dict[str, object] = {
        "sample_id": sample_id,
        "raw_audio_path": sample["raw_audio_path"],
        "clauses": sample["clauses"],
        "speech_top_indices": speech_top_indices,
        "text_top_indices": text_top_indices,
    }

    unit_rows: List[Dict[str, object]] = []
    for unit_index in range(speech_delta_z.size):
        unit_rows.append(
            {
                "sample_index": int(sample["sample_index"]),
                "sample_id": sample_id,
                "true_class": true_class,
                "pred_class": pred_class,
                "modality": "Speech",
                "unit_index": unit_index,
                "unit_text": "",
                "baseline_pred_logit": baseline_logit,
                "baseline_pred_prob": baseline_prob,
                "occluded_pred_logit": float(speech_occ_z[unit_index]),
                "occluded_pred_prob": float(speech_occ_p[unit_index]),
                "delta_z": float(speech_delta_z[unit_index]),
                "delta_p": float(speech_delta_p[unit_index]),
                "delta_p_pp": float(speech_delta_p[unit_index] * 100.0),
            }
        )

    for unit_index in range(text_delta_z.size):
        unit_rows.append(
            {
                "sample_index": int(sample["sample_index"]),
                "sample_id": sample_id,
                "true_class": true_class,
                "pred_class": pred_class,
                "modality": "Text",
                "unit_index": unit_index,
                "unit_text": str(sample["clauses"][unit_index]),
                "baseline_pred_logit": baseline_logit,
                "baseline_pred_prob": baseline_prob,
                "occluded_pred_logit": float(text_occ_z[unit_index]),
                "occluded_pred_prob": float(text_occ_p[unit_index]),
                "delta_z": float(text_delta_z[unit_index]),
                "delta_p": float(text_delta_p[unit_index]),
                "delta_p_pp": float(text_delta_p[unit_index] * 100.0),
            }
        )

    return plot_data, sample_row, unit_rows


def coerce_mono_waveform(
    audio_data: np.ndarray,
    sample_rate: int,
) -> Tuple[np.ndarray, int]:
    """将解码后的音频转换为有限的一维单声道波形。"""
    waveform = np.asarray(audio_data)
    if waveform.ndim == 2:
        waveform = waveform.astype(np.float64).mean(axis=1)
    if waveform.ndim != 1:
        raise ValueError(
            f"Expected one- or two-dimensional audio, got {waveform.shape}."
        )
    waveform = waveform.astype(np.float32, copy=False)
    if waveform.size == 0 or not np.all(np.isfinite(waveform)):
        raise ValueError("waveform is empty or contains non-finite values")
    if int(sample_rate) <= 0:
        raise ValueError(f"invalid sample rate: {sample_rate}")
    return waveform, int(sample_rate)


def load_audio_waveform(audio_path: Path) -> Tuple[np.ndarray, int]:
    """依次尝试可用后端读取单声道波形。"""
    errors: List[str] = []

    if sf is not None:
        try:
            waveform, sample_rate = sf.read(str(audio_path))
            return coerce_mono_waveform(waveform, sample_rate)
        except Exception as exc:
            errors.append(f"soundfile: {exc}")

    if librosa is not None:
        try:
            waveform, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
            return coerce_mono_waveform(waveform, sample_rate)
        except Exception as exc:
            errors.append(f"librosa: {exc}")

    if wavfile is not None and audio_path.suffix.lower() == ".wav":
        try:
            sample_rate, waveform = wavfile.read(str(audio_path))
            return coerce_mono_waveform(waveform, sample_rate)
        except Exception as exc:
            errors.append(f"scipy.io.wavfile: {exc}")

    details = "; ".join(errors) if errors else "no compatible audio backend"
    raise RuntimeError(f"Unable to read audio file {audio_path}: {details}")


def prepare_waveform_plot_data(
    waveform: np.ndarray,
    sample_rate: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """构建时间不变的 min--max 波形包络，避免高采样率直接抽样的混叠。"""
    num_bins = min(WAVEFORM_MAX_POINTS, int(waveform.size))
    bin_edges = np.linspace(0, waveform.size, num_bins + 1, dtype=np.int64)
    lower = np.empty(num_bins, dtype=np.float64)
    upper = np.empty(num_bins, dtype=np.float64)
    for bin_index in range(num_bins):
        segment = waveform[bin_edges[bin_index]:bin_edges[bin_index + 1]]
        lower[bin_index] = float(np.min(segment))
        upper[bin_index] = float(np.max(segment))

    peak = float(max(np.max(np.abs(lower)), np.max(np.abs(upper))))
    if peak > 0.0:
        lower = lower / peak
        upper = upper / peak
    time_plot = (
        (bin_edges[:-1] + bin_edges[1:] - 1).astype(np.float64)
        / (2.0 * float(sample_rate))
    )
    duration = float(waveform.size) / float(sample_rate)
    return time_plot, lower, upper, duration


def draw_speech_waveform(
    axis: Axes,
    waveform: np.ndarray,
    sample_rate: int,
    top_speech_indices: Sequence[int],
    sample_id: str,
) -> None:
    """绘制原始语音波形，并标注按 delta_z 排名的 S1--S3 时间窗。"""
    time_plot, lower, upper, duration = prepare_waveform_plot_data(
        waveform,
        sample_rate,
    )

    for rank, unit_index in enumerate(top_speech_indices):
        start = float(unit_index) * SPEECH_SEGMENT_HOP
        end = min(start + SPEECH_SEGMENT_DURATION, duration)
        if start >= duration or end <= start:
            raise RuntimeError(
                f"Speech unit {unit_index} of sample {sample_id} cannot be "
                "mapped onto its raw waveform. Check the audio segmentation "
                "configuration and source file."
            )

        color = COL_SPEECH_HIGHLIGHT[rank]
        axis.axvspan(
            start,
            end,
            color=color,
            alpha=0.34 - 0.07 * rank,
            linewidth=0.0,
            zorder=1,
        )

    axis.fill_between(
        time_plot,
        lower,
        upper,
        color=COL_WAVE_FILL,
        alpha=0.55,
        linewidth=0.0,
        zorder=2,
    )
    axis.plot(
        time_plot,
        lower,
        color=COL_WAVE,
        linewidth=0.30,
        zorder=3,
    )
    axis.plot(
        time_plot,
        upper,
        color=COL_WAVE,
        linewidth=0.30,
        zorder=3,
    )
    axis.axhline(0.0, color=COL_BORDER, linewidth=0.25, zorder=3)

    for rank, unit_index in enumerate(top_speech_indices):
        start = float(unit_index) * SPEECH_SEGMENT_HOP
        end = min(start + SPEECH_SEGMENT_DURATION, duration)
        color = COL_SPEECH_HIGHLIGHT[rank]
        axis.text(
            (start + end) / 2.0,
            0.88,
            f"S{rank + 1}",
            ha="center",
            va="center",
            fontproperties=FP_EN,
            fontsize=4.9,
            fontweight="bold",
            color="#FFFFFF",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": color,
                "edgecolor": "none",
                "alpha": 0.96,
            },
            zorder=4,
        )

    axis.set_xlim(0.0, duration)
    axis.set_ylim(-1.05, 1.15)
    axis.axis("off")


def wrap_text(text: str, width: int = 17, max_lines: int = 2) -> str:
    """将中文子句按字符宽度换行，仅用于小尺寸论文图显示。"""
    characters = list(text)
    lines = [
        "".join(characters[start:start + width])
        for start in range(0, min(len(characters), width * max_lines), width)
    ]
    wrapped = "\n".join(lines)
    if len(characters) > width * max_lines:
        wrapped += "…"
    return wrapped


def build_text_box_layout(num_boxes: int) -> List[Tuple[float, float]]:
    """返回从上到下均匀排列的文本证据框的顶部坐标和高度。"""
    if num_boxes <= 0:
        raise ValueError("num_boxes must be positive.")

    margin = 0.025
    gap = 0.030
    box_height = (1.0 - 2.0 * margin - (num_boxes - 1) * gap) / num_boxes
    if box_height <= 0.0:
        raise ValueError("Text evidence boxes do not fit in the figure panel.")
    return [
        (1.0 - margin - rank * (box_height + gap), box_height)
        for rank in range(num_boxes)
    ]


def draw_text_evidence(
    axis: Axes,
    clauses: Sequence[str],
    top_text_indices: Sequence[int],
) -> None:
    """绘制按 delta_z 独立排名的 T1--T3 文本子句。"""
    box_layout = build_text_box_layout(len(top_text_indices))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    for rank, unit_index in enumerate(top_text_indices):
        top_y, box_height = box_layout[rank]
        bottom_y = top_y - box_height
        center_y = (top_y + bottom_y) / 2.0

        axis.add_patch(
            FancyBboxPatch(
                (0.01, bottom_y),
                0.98,
                box_height,
                boxstyle="round,pad=0.008,rounding_size=0.028",
                facecolor=COL_TEXT_BOX,
                edgecolor=COL_BORDER,
                linewidth=0.55,
                zorder=1,
            )
        )
        axis.add_patch(
            FancyBboxPatch(
                (0.023, bottom_y + 0.018),
                0.020,
                box_height - 0.036,
                boxstyle="round,pad=0.002,rounding_size=0.014",
                facecolor=COL_TEXT_STRIP[rank],
                edgecolor="none",
                zorder=2,
            )
        )
        axis.text(
            0.065,
            center_y,
            f"T{rank + 1}",
            fontproperties=FP_EN,
            fontsize=4.9,
            fontweight="bold",
            color=COL_TEXT,
            va="center",
            ha="left",
            zorder=3,
        )
        axis.text(
            0.145,
            center_y,
            wrap_text(str(clauses[unit_index])),
            fontproperties=FP_CN,
            fontsize=4.6,
            color=COL_TEXT,
            va="center",
            ha="left",
            linespacing=1.25,
            zorder=3,
        )


def plot_sample(save_path: Path, plot_data: Dict[str, object]) -> None:
    """以原始波形绘制局部证据；入选的 Figure 11 案例均展示三项证据。"""
    sample_id = str(plot_data["sample_id"])
    raw_audio_path = Path(plot_data["raw_audio_path"])
    waveform, sample_rate = load_audio_waveform(raw_audio_path)

    fig = plt.figure(
        figsize=FIGURE_SIZE,
        dpi=FIGURE_DPI,
        facecolor=COL_BG,
    )
    fig.text(
        0.07,
        0.955,
        "Speech",
        fontproperties=FP_EN,
        fontsize=7.6,
        fontweight="bold",
        color=COL_TEXT,
    )
    wave_axis = fig.add_axes([0.07, 0.635, 0.86, 0.285], facecolor=COL_BG)
    fig.text(
        0.07,
        0.585,
        "Text",
        fontproperties=FP_EN,
        fontsize=7.6,
        fontweight="bold",
        color=COL_TEXT,
    )
    text_axis = fig.add_axes([0.07, 0.070, 0.86, 0.475], facecolor=COL_BG)

    draw_speech_waveform(
        wave_axis,
        waveform,
        sample_rate,
        plot_data["speech_top_indices"],
        sample_id,
    )
    draw_text_evidence(
        text_axis,
        plot_data["clauses"],
        plot_data["text_top_indices"],
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        save_path,
        dpi=FIGURE_DPI,
        facecolor=COL_BG,
        bbox_inches="tight",
        pad_inches=0.0,
        format="png",
    )
    plt.close(fig)


def build_paper_table(selected_df: pd.DataFrame) -> pd.DataFrame:
    """按自动选择顺序生成 Table 12；S/T 数值对应排名第一的 S1/T1。"""
    if len(selected_df) != PAPER_CASE_COUNT:
        raise RuntimeError(
            f"Expected {PAPER_CASE_COUNT} paper cases, got {len(selected_df)}."
        )

    rows: List[Dict[str, object]] = []
    for _, row in selected_df.iterrows():
        rows.append(
            {
                "No.": int(row["selection_rank"]),
                "Label": str(row["sample_id"]),
                "Base p (%)": float(row["baseline_pred_prob"] * 100.0),
                "S-Δz": float(row["speech_top_delta_z"]),
                "S-Δp (pp)": float(row["speech_top_delta_p_pp"]),
                "T-Δz": float(row["text_top_delta_z"]),
                "T-Δp (pp)": float(row["text_top_delta_p_pp"]),
            }
        )
    return pd.DataFrame(rows)


def save_metadata(
    num_samples: int,
    figure_count: int,
    selected_df: pd.DataFrame,
) -> None:
    """保存方法定义与正式配置，防止局部解释和模态依赖指标混用。"""
    metadata = {
        "title": "Sample-level LOU evidence analysis for Table 12 and Figure 11",
        "dataset": "NCMMSC2021 test set",
        "num_test_samples": int(num_samples),
        "selected_paper_sample_ids": selected_df["sample_id"].tolist(),
        "saved_figure_count": int(figure_count),
        "seed": SEED,
        "class_map": CLASS_MAP,
        "model": {
            "fusion_type": FUSION_TYPE,
            "shared_dim": SHARED_DIM,
            "dropout": DROPOUT,
            "weight_path": str(WEIGHT_PATH),
        },
        "text_feature_extraction": {
            "source_root": str(TEXT_SOURCE_ROOT),
            "model_path": str(TEXT_MODEL_PATH),
            "layer_strategy": TEXT_LAYER_STRATEGY,
            "pool": TEXT_POOL,
            "max_len": TEXT_MAX_LEN,
            "stride": TEXT_STRIDE,
            "original_feature_verification_atol": FEATURE_ATOL,
            "original_feature_verification_rtol": FEATURE_RTOL,
        },
        "lou": {
            "target": "original predicted class of the seed-2024 model",
            "speech_unit": (
                "one stored speech feature row; the row is zeroed and its mask "
                "is set to False"
            ),
            "speech_window_seconds": SPEECH_SEGMENT_DURATION,
            "speech_hop_seconds": SPEECH_SEGMENT_HOP,
            "text_unit": (
                "one punctuation-delimited transcript clause; the clause is "
                "removed while all other normalized characters and punctuation "
                "retain their original order"
            ),
            "text_perturbation": (
                "the remaining transcript is re-encoded by the unchanged BERT "
                "feature extractor"
            ),
            "ranking_metric": (
                "signed predicted-class logit drop: delta_z = z_before - z_after"
            ),
            "reported_probability_metric": (
                "signed probability drop from the same top-ranked unit: "
                "delta_p = p_before - p_after"
            ),
            "cross_modal_selection": (
                "speech and text units are ranked independently for the same "
                "predicted class and are not assumed to be one-to-one aligned"
            ),
            "relation_to_modality_reliance": (
                "perturbation units and text pipeline are identical; the separate "
                "modality-reliance analysis aggregates mean absolute delta_D and "
                "must not be substituted for the local delta_z values"
            ),
        },
        "figure_11": {
            "speech_background": "mandatory raw-audio waveform",
            "speech_top_k": TOP_K_SPEECH,
            "speech_labels": [f"S{rank}" for rank in range(1, TOP_K_SPEECH + 1)],
            "text_top_k": TOP_K_TEXT,
            "text_labels": [f"T{rank}" for rank in range(1, TOP_K_TEXT + 1)],
            "ranking": (
                "speech and text units are independently ranked by signed "
                "predicted-class delta_z"
            ),
        },
        "case_selection": {
            "count": PAPER_CASE_COUNT,
            "candidate_set": (
                "correctly classified test samples with sufficient displayed units"
                if PAPER_CASES_REQUIRE_CORRECT
                else "test samples with sufficient displayed units"
            ),
            "minimum_speech_units": TOP_K_SPEECH,
            "minimum_text_clauses": TOP_K_TEXT,
            "ranking_score": (
                "combined_top_delta_z = max(S-delta_z) + max(T-delta_z)"
            ),
            "tie_breakers": [
                "higher max(S-delta_z)",
                "higher max(T-delta_z)",
                "higher base predicted-class probability",
                "lexicographically smaller sample ID",
            ],
        },
    }
    (OUTPUT_DIR / "method_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_paper_table(paper_table: pd.DataFrame) -> None:
    """在终端打印与 Table 12 相同的两位小数显示值。"""
    numeric_columns = [
        "Base p (%)",
        "S-Δz",
        "S-Δp (pp)",
        "T-Δz",
        "T-Δp (pp)",
    ]
    formatters = {
        column: (lambda value: f"{value:.2f}")
        for column in numeric_columns
    }
    print("\n" + "=" * 92)
    print(
        "Table 12. Quantitative results of top-ranked speech and text "
        "evidence in six samples."
    )
    print("=" * 92)
    print(paper_table.to_string(index=False, formatters=formatters))
    print("=" * 92)


def main() -> None:
    """执行特征核验、单模型 LOU、论文表格生成、原始结果导出和绘图。"""
    validate_configuration()
    validate_paths()
    set_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_dir = OUTPUT_DIR / "samples"
    figure_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Data root: {DATA_ROOT}")
    print(f"[INFO] Weight: {WEIGHT_PATH}")

    dataset = MultimodalDataset(str(DATA_ROOT / "test"), CLASS_MAP)
    audio_dim, text_dim = validate_dataset_samples(dataset)
    print(
        f"[INFO] Test samples: {len(dataset)}, "
        f"speech_dim={audio_dim}, text_dim={text_dim}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(TEXT_MODEL_PATH),
        local_files_only=LOCAL_FILES_ONLY,
    )
    text_encoder = AutoModel.from_pretrained(
        str(TEXT_MODEL_PATH),
        local_files_only=LOCAL_FILES_ONLY,
    ).to(DEVICE)
    text_encoder.eval()

    prepared_samples, verification_df = prepare_lou_inputs(
        dataset,
        tokenizer,
        text_encoder,
    )
    verification_path = OUTPUT_DIR / "text_feature_verification.csv"
    verification_df.to_csv(
        verification_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    print(f"[OK] Text-feature verification passed for all {len(dataset)} samples.")

    del text_encoder
    del tokenizer
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    model = build_model(audio_dim, text_dim)
    sample_rows: List[Dict[str, object]] = []
    unit_rows: List[Dict[str, object]] = []
    plot_data_by_id: Dict[str, Dict[str, object]] = {}

    iterator = tqdm(prepared_samples, desc="Running sample-level LOU", ncols=110)
    for sample in iterator:
        plot_data, sample_row, current_unit_rows = analyze_sample(model, sample)
        sample_rows.append(sample_row)
        unit_rows.extend(current_unit_rows)
        sample_id = str(sample["sample_id"])
        plot_data_by_id[sample_id] = plot_data

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    sample_df = pd.DataFrame(sample_rows).sort_values(
        "sample_index",
        kind="stable",
    )
    unit_df = pd.DataFrame(unit_rows).sort_values(
        ["sample_index", "modality", "unit_index"],
        kind="stable",
    )
    if len(sample_df) != len(dataset):
        raise RuntimeError(
            f"Unexpected sample result count: {len(sample_df)} != {len(dataset)}"
        )

    selected_df = select_paper_cases(sample_df)
    selected_sample_ids = [str(sample_id) for sample_id in selected_df["sample_id"]]
    figure_sample_ids = (
        [str(sample_id) for sample_id in sample_df["sample_id"]]
        if SAVE_ALL_SAMPLE_FIGURES
        else selected_sample_ids
    )
    for sample_id in figure_sample_ids:
        plot_sample(figure_dir / f"{sample_id}.png", plot_data_by_id[sample_id])

    paper_table = build_paper_table(selected_df)
    paper_table_path = OUTPUT_DIR / "table_12.csv"
    sample_path = OUTPUT_DIR / "all_sample_metrics_raw.csv"
    unit_path = OUTPUT_DIR / "lou_unit_scores.csv"

    paper_table.to_csv(
        paper_table_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.2f",
    )
    sample_df.to_csv(
        sample_path,
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
    save_metadata(len(dataset), len(figure_sample_ids), selected_df)
    print_paper_table(paper_table)

    print("[INFO] Selected Table 12 cases: " + ", ".join(selected_sample_ids))
    print(f"[OK] Table 12: {paper_table_path}")
    print(f"[OK] All-sample metrics: {sample_path}")
    print(f"[OK] Unit-level LOU scores: {unit_path}")
    print(f"[OK] Text verification: {verification_path}")
    print(f"[OK] Sample figures: {figure_dir} ({len(figure_sample_ids)} files)")


if __name__ == "__main__":
    main()
