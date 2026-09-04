# import os
#
# # Force Transformers to use local files only during formal profiling.
# os.environ.setdefault("HF_HUB_OFFLINE", "1")
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
#
# import argparse
# import csv
# import re
# import time
# import wave
# from pathlib import Path
# from typing import Dict, List, Tuple
#
# import numpy as np
# import opencc
# import torch
# from tqdm import tqdm
# from transformers import (
#     AutoModel,
#     AutoTokenizer,
#     Wav2Vec2Model,
#     Wav2Vec2Processor,
#     WhisperForConditionalGeneration,
#     WhisperProcessor,
#     pipeline,
# )
# from transformers.utils import logging as transformers_logging
#
# from model import MultiModalNet
#
#
# # ============================================================
# # 1. Paths and experiment configuration
# # ============================================================
# # Expected layout:
# # Computational_cost/
# # ├── NCMMSC2021/
# # ├── NCMMSC2021_text/
# # ├── bert-base-chinese/
# # ├── wav2vec2-base/
# # ├── whisper-large-v3/
# # ├── SIST/
# # │   ├── weights_0/best.pth
# # │   ├── weights_1/best.pth
# # │   ├── weights_42/best.pth
# # │   ├── weights_123/best.pth
# # │   └── weights_2024/best.pth
# # └── Complete_pipeline/
# #     ├── profile_complete_pipeline.py
# #     ├── model.py
# #     ├── model_audio.py
# #     └── model_text.py
# SCRIPT_DIR = Path(__file__).resolve().parent
# ROOT_DIR = SCRIPT_DIR.parent
#
# AUDIO_ROOT = ROOT_DIR / "NCMMSC2021"
# TEXT_ROOT = ROOT_DIR / "NCMMSC2021_text"
# WHISPER_DIR = ROOT_DIR / "whisper-large-v3"
# WAV2VEC2_DIR = ROOT_DIR / "wav2vec2-base"
# BERT_DIR = ROOT_DIR / "bert-base-chinese"
# SIST_ROOT = ROOT_DIR / "SIST"
#
# SUMMARY_CSV = SCRIPT_DIR / "complete_pipeline_profile_summary.csv"
# RECORD_CSV = SCRIPT_DIR / "complete_pipeline_profile_records.csv"
#
# TEST_SPLIT = "test"
# CLASS_MAP = {"AD": 0, "HC": 1, "MCI": 2}
# SUPPORTED_SEEDS = (0, 1, 42, 123, 2024)
# DEFAULT_SEED = 2024
# WARMUP_RUNS = 5
#
# # Complete-pipeline timing protocol:
# # - raw WAV disk read / PCM decode: excluded;
# # - Whisper transcription and transcript post-processing: included;
# # - BERT text feature extraction: included;
# # - Wav2Vec 2.0 speech feature extraction: included;
# # - CPU/GPU transfers between stages: included;
# # - SIST forward + softmax + argmax: included.
# # All four models remain resident on the same CUDA device during profiling.
#
#
# # ============================================================
# # 2. Whisper configuration and transcript post-processing
# # ============================================================
# LANGUAGE = "zh"
# EXPECTED_SAMPLE_RATE = 16000
#
# WHISPER_GENERATE_KWARGS = {
#     "language": LANGUAGE,
#     "task": "transcribe",
#     "num_beams": 1,
#     "do_sample": False,
#     "condition_on_prev_tokens": False,
#     "temperature": 0.0,
#     "compression_ratio_threshold": 2.4,
#     "logprob_threshold": -1.0,
#     "no_speech_threshold": 0.6,
#
#     # Whisper 的 decoder 最大总长度为 448 token。
#     # 当前中文 transcribe + timestamps 模式会先占用 3 个 decoder
#     # special/prompt token，因此 max_new_tokens 不能再设为 448。
#     # 448 - 3 = 445，显式设为 445 可与当前 Transformers 版本兼容，
#     # 同时保持与 Whisper 原始 448-token decoder 上限一致。
#     "max_new_tokens": 445,
# }
#
# PAUSE_T_SHORT = 0.5
# PAUSE_T_MEDIUM = 1.0
# PAUSE_T_LONG = 1.5
# INSERT_FOR_NONE = True
# SEP_LIMIT = {"zh": 18, "en": 30}
# TERM_LIMIT = {"zh": 40, "en": 80}
#
# PAUSE_PUNCT = {
#     "zh": {"NONE": "", "SHORT": "，", "MEDIUM": "。", "LONG": "……"},
#     "en": {"NONE": "", "SHORT": ",", "MEDIUM": ".", "LONG": "..."},
#     "default": {"NONE": "", "SHORT": ",", "MEDIUM": ".", "LONG": "..."},
# }
#
# def create_chinese_converter():
#     """
#     Create an OpenCC Traditional-to-Simplified converter.
#
#     opencc-python-reimplemented expects the configuration name without
#     the .json suffix ("t2s"). Some other OpenCC Python bindings accept
#     "t2s.json" instead, so both forms are tried for portability.
#     """
#     errors = []
#     for config_name in ("t2s", "t2s.json"):
#         try:
#             return opencc.OpenCC(config_name)
#         except (FileNotFoundError, OSError, ValueError) as error:
#             errors.append(f"{config_name}: {error}")
#
#     raise RuntimeError(
#         "Unable to initialize OpenCC Traditional-to-Simplified conversion. "
#         "Install opencc-python-reimplemented and verify its config files.\n"
#         + "\n".join(errors)
#     )
#
#
# _CN_CONVERTER = create_chinese_converter()
# _CN_ASCII_TO_FULL = str.maketrans({
#     ",": "，",
#     ".": "。",
#     "?": "？",
#     "!": "！",
#     ":": "：",
#     ";": "；",
# })
# _RE_TAIL_PUNCT_ZH = re.compile(r"[，。！？；：、…]+$")
# _RE_TAIL_PUNCT_EN = re.compile(r"[,\.!?;:]+$")
#
#
# # ============================================================
# # 3. Wav2Vec 2.0 / BERT feature configuration
# # ============================================================
# # Wav2Vec 2.0: same final configuration as the standalone profiler.
# W2V_SAMPLE_RATE = 16000
# SEGMENT_SECONDS = 6
# OVERLAP_SECONDS = 3
# SEGMENT_SAMPLES = W2V_SAMPLE_RATE * SEGMENT_SECONDS
#
# # BERT: same final configuration as the standalone profiler.
# BERT_MAX_LEN = 256
# BERT_STRIDE = 64
# BERT_TOKENIZER_MAX_LEN = 10000
#
# # SIST: same S4 configuration as the current test code.
# SIST_FUSION_TYPE = "gated_bi_cross_attention"
# SIST_DROPOUT = 0.5
# SIST_SHARED_DIM = 512
# SIST_NUM_CLASSES = 3
#
#
# # ============================================================
# # 4. Dataset and path validation
# # ============================================================
# def enumerate_audio_files(dataset_root: Path) -> List[Path]:
#     split_dir = dataset_root / TEST_SPLIT
#     if not split_dir.is_dir():
#         raise FileNotFoundError(f"Dataset split not found: {split_dir}")
#
#     return sorted(
#         path
#         for path in split_dir.rglob("*.wav")
#         if path.is_file()
#     )
#
#
# def get_paired_text_path(audio_path: Path) -> Path:
#     relative_path = audio_path.relative_to(AUDIO_ROOT)
#     return (TEXT_ROOT / relative_path).with_suffix(".txt")
#
#
# def get_class_and_sample(audio_path: Path) -> Tuple[str, str]:
#     relative_path = audio_path.relative_to(AUDIO_ROOT / TEST_SPLIT)
#     if len(relative_path.parts) < 2:
#         raise ValueError(f"Unexpected test path: {audio_path}")
#
#     class_name = relative_path.parts[0]
#     if class_name not in CLASS_MAP:
#         raise ValueError(f"Unknown class directory: {class_name}")
#
#     return class_name, audio_path.stem
#
#
# def get_sist_checkpoint(seed: int) -> Path:
#     return SIST_ROOT / f"weights_{seed}" / "best.pth"
#
#
# def validate_audio_text_pairs(audio_files: List[Path]) -> None:
#     """
#     NCMMSC2021_text is used only to verify one-to-one sample identity.
#
#     Stored text is intentionally NOT used as the BERT input in the timed
#     complete pipeline; using it would bypass Whisper and would no longer be
#     an automatic raw-speech-to-prediction pipeline.
#     """
#     missing_text: List[str] = []
#
#     audio_relative_txt = {
#         str(audio.relative_to(AUDIO_ROOT).with_suffix(".txt"))
#         for audio in audio_files
#     }
#
#     for audio_path in audio_files:
#         text_path = get_paired_text_path(audio_path)
#         if not text_path.is_file():
#             missing_text.append(str(text_path))
#
#     if missing_text:
#         preview = "\n".join(missing_text[:10])
#         raise RuntimeError(
#             "Some test WAV files do not have paired text files:\n" + preview
#         )
#
#     text_split = TEXT_ROOT / TEST_SPLIT
#     text_relative = {
#         str(path.relative_to(TEXT_ROOT))
#         for path in text_split.rglob("*.txt")
#         if path.is_file()
#     }
#
#     extra_text = sorted(text_relative - audio_relative_txt)
#     if extra_text:
#         preview = "\n".join(extra_text[:10])
#         raise RuntimeError(
#             "NCMMSC2021_text contains test samples without paired WAV files:\n"
#             + preview
#         )
#
#
# def validate_model_directory(model_dir: Path, name: str) -> None:
#     if not model_dir.is_dir():
#         raise FileNotFoundError(f"{name} directory not found: {model_dir}")
#     if not (model_dir / "config.json").is_file():
#         raise FileNotFoundError(f"Missing config.json in: {model_dir}")
#
#
# def validate_paths(seed: int) -> List[Path]:
#     if not AUDIO_ROOT.is_dir():
#         raise FileNotFoundError(f"Audio dataset not found: {AUDIO_ROOT}")
#     if not TEXT_ROOT.is_dir():
#         raise FileNotFoundError(f"Text dataset not found: {TEXT_ROOT}")
#     if not SIST_ROOT.is_dir():
#         raise FileNotFoundError(f"SIST directory not found: {SIST_ROOT}")
#
#     validate_model_directory(WHISPER_DIR, "Whisper")
#     validate_model_directory(WAV2VEC2_DIR, "Wav2Vec 2.0")
#     validate_model_directory(BERT_DIR, "BERT")
#
#     checkpoint = get_sist_checkpoint(seed)
#     if not checkpoint.is_file():
#         raise FileNotFoundError(f"SIST checkpoint not found: {checkpoint}")
#
#     audio_files = enumerate_audio_files(AUDIO_ROOT)
#     if not audio_files:
#         raise RuntimeError(f"No WAV files found in: {AUDIO_ROOT / TEST_SPLIT}")
#
#     validate_audio_text_pairs(audio_files)
#     validate_audio_headers(audio_files)
#     return audio_files
#
#
# def get_cuda_device() -> torch.device:
#     if not torch.cuda.is_available():
#         raise RuntimeError(
#             "CUDA is unavailable. Complete-pipeline profiling requires a CUDA GPU."
#         )
#     return torch.device("cuda:0")
#
#
# # ============================================================
# # 5. WAV loading
# # ============================================================
# def decode_pcm_samples(raw_bytes: bytes, sample_width: int) -> np.ndarray:
#     if sample_width == 1:
#         samples = np.frombuffer(raw_bytes, dtype=np.uint8)
#         return (samples.astype(np.float32) - 128.0) / 128.0
#
#     if sample_width == 2:
#         samples = np.frombuffer(raw_bytes, dtype="<i2")
#         return samples.astype(np.float32) / 32768.0
#
#     if sample_width == 3:
#         byte_array = np.frombuffer(raw_bytes, dtype=np.uint8)
#         if byte_array.size % 3 != 0:
#             raise ValueError("Invalid 24-bit PCM byte length.")
#         byte_array = byte_array.reshape(-1, 3)
#         samples = (
#             byte_array[:, 0].astype(np.int32)
#             | (byte_array[:, 1].astype(np.int32) << 8)
#             | (byte_array[:, 2].astype(np.int32) << 16)
#         )
#         samples = np.where(samples & 0x800000, samples - 0x1000000, samples)
#         return samples.astype(np.float32) / 8388608.0
#
#     if sample_width == 4:
#         samples = np.frombuffer(raw_bytes, dtype="<i4")
#         return samples.astype(np.float32) / 2147483648.0
#
#     raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")
#
#
# def load_wav_audio(audio_path: Path) -> Tuple[np.ndarray, int, float]:
#     """Read one PCM WAV outside the timed region."""
#     with wave.open(str(audio_path), "rb") as wav_file:
#         channels = wav_file.getnchannels()
#         sample_width = wav_file.getsampwidth()
#         sample_rate = wav_file.getframerate()
#         frame_count = wav_file.getnframes()
#         compression = wav_file.getcomptype()
#
#         if compression != "NONE":
#             raise ValueError(f"Compressed WAV is not supported: {audio_path}")
#         if channels <= 0 or sample_rate <= 0 or frame_count <= 0:
#             raise ValueError(f"Invalid WAV header: {audio_path}")
#
#         raw_bytes = wav_file.readframes(frame_count)
#
#     audio = decode_pcm_samples(raw_bytes, sample_width)
#
#     if channels > 1:
#         audio = audio.reshape(-1, channels).mean(axis=1)
#
#     audio = np.ascontiguousarray(audio, dtype=np.float32)
#     duration = frame_count / sample_rate
#     return audio, sample_rate, duration
#
#
# def validate_audio_headers(audio_files: List[Path]) -> None:
#     invalid: List[str] = []
#     for audio_path in audio_files:
#         try:
#             with wave.open(str(audio_path), "rb") as wav_file:
#                 sample_rate = wav_file.getframerate()
#                 compression = wav_file.getcomptype()
#             if sample_rate != EXPECTED_SAMPLE_RATE or compression != "NONE":
#                 invalid.append(
#                     f"{audio_path}: sample_rate={sample_rate}, compression={compression}"
#                 )
#         except (wave.Error, OSError) as error:
#             invalid.append(f"{audio_path}: {error}")
#
#     if invalid:
#         raise RuntimeError(
#             "Some test WAV files do not satisfy the PCM/16-kHz protocol:\n"
#             + "\n".join(invalid[:10])
#         )
#
#
# # ============================================================
# # 6. Whisper transcription and post-processing
# # ============================================================
# def load_whisper_resources(device: torch.device):
#     dtype = torch.float16
#
#     processor = WhisperProcessor.from_pretrained(
#         str(WHISPER_DIR),
#         local_files_only=True,
#     )
#     model = WhisperForConditionalGeneration.from_pretrained(
#         str(WHISPER_DIR),
#         local_files_only=True,
#         use_safetensors=True,
#         dtype=dtype,
#     )
#     model.to(device)
#     model.eval()
#     model.requires_grad_(False)
#
#     max_target_positions = int(model.config.max_target_positions)
#     requested_max_new_tokens = int(
#         WHISPER_GENERATE_KWARGS["max_new_tokens"]
#     )
#     decoder_prefix_tokens = 3
#
#     if requested_max_new_tokens + decoder_prefix_tokens > max_target_positions:
#         raise RuntimeError(
#             "Invalid Whisper decoding length configuration: "
#             f"decoder prefix={decoder_prefix_tokens}, "
#             f"max_new_tokens={requested_max_new_tokens}, "
#             f"max_target_positions={max_target_positions}."
#         )
#
#     asr = pipeline(
#         task="automatic-speech-recognition",
#         model=model,
#         tokenizer=processor.tokenizer,
#         feature_extractor=processor.feature_extractor,
#         dtype=dtype,
#         device=device,
#         batch_size=1,
#     )
#
#     return model, asr
#
#
# def transcribe_one(audio: np.ndarray, sample_rate: int, asr) -> Dict:
#     result = asr(
#         {"raw": audio, "sampling_rate": sample_rate},
#         return_timestamps=True,
#         generate_kwargs=WHISPER_GENERATE_KWARGS,
#     )
#
#     segments: List[Dict[str, object]] = []
#     for chunk in result.get("chunks", []):
#         timestamp = chunk.get("timestamp", (None, None))
#         start = timestamp[0] if timestamp else None
#         end = timestamp[1] if timestamp else None
#         segments.append({
#             "text": chunk.get("text", ""),
#             "start": start,
#             "end": end,
#         })
#
#     if not segments and result.get("text"):
#         segments.append({
#             "text": result["text"],
#             "start": None,
#             "end": None,
#         })
#
#     return {
#         "text": result.get("text", ""),
#         "segments": segments,
#         "language": LANGUAGE,
#     }
#
#
# def normalize_ellipsis_zh(text: str) -> str:
#     text = re.sub(r"(\.{3,})", "……", text)
#     text = re.sub(r"…{2,}", "……", text)
#     return text
#
#
# def normalize_zh_defaults(text: str) -> str:
#     text = normalize_ellipsis_zh(text.translate(_CN_ASCII_TO_FULL))
#     text = re.sub(r"\s*([，。！？；：、（）《》“”‘’……])\s*", r"\1", text)
#     text = re.sub(r"([，。！？；：、（《”’)])\1+", r"\1", text)
#     text = re.sub(r"(……)+", "……", text)
#     text = re.sub(r"^[，。！？；：、……]+", "", text)
#     return text.strip()
#
#
# def convert_to_simplified_chinese(text: str) -> str:
#     return _CN_CONVERTER.convert(text)
#
#
# def split_tail_punct(text: str, language: str) -> Tuple[str, str]:
#     stripped = text.rstrip()
#     if not stripped:
#         return "", ""
#     pattern = _RE_TAIL_PUNCT_ZH if language == "zh" else _RE_TAIL_PUNCT_EN
#     match = pattern.search(stripped)
#     if match:
#         return stripped[:match.start()], stripped[match.start():]
#     return stripped, ""
#
#
# def pause_bucket(duration: float) -> str:
#     if duration >= PAUSE_T_LONG:
#         return "LONG"
#     if duration >= PAUSE_T_MEDIUM:
#         return "MEDIUM"
#     if duration >= PAUSE_T_SHORT:
#         return "SHORT"
#     return "NONE"
#
#
# def pick_pause_punct(language: str, bucket: str) -> str:
#     key = language if language in PAUSE_PUNCT else "default"
#     return PAUSE_PUNCT[key].get(bucket, "")
#
#
# def is_terminal(language: str, punct: str) -> bool:
#     if not punct:
#         return False
#     if language == "zh":
#         return "……" in punct or any(mark in punct for mark in "。！？")
#     return "..." in punct or any(mark in punct for mark in ".?!")
#
#
# def is_separator(language: str, punct: str) -> bool:
#     if not punct:
#         return False
#     marks = "，；：、" if language == "zh" else ",;:"
#     return any(mark in punct for mark in marks)
#
#
# def stitch_with_pause_override(result: Dict, language: str = "zh") -> str:
#     segments = result.get("segments", [])
#     if not segments:
#         return ""
#
#     normalized_segments: List[str] = []
#     for segment in segments:
#         text = str(segment.get("text", "")).strip()
#         if language == "zh":
#             text = convert_to_simplified_chinese(normalize_zh_defaults(text))
#         normalized_segments.append(text)
#
#     output_parts: List[str] = []
#     since_terminal = 0
#     since_separator = 0
#
#     for index in range(len(segments) - 1):
#         current_text = normalized_segments[index]
#         current_end = segments[index].get("end")
#         next_start = segments[index + 1].get("start")
#
#         if current_end is None or next_start is None:
#             pause_duration = 0.0
#         else:
#             pause_duration = max(0.0, float(next_start) - float(current_end))
#
#         base, tail = split_tail_punct(current_text, language)
#         bucket = pause_bucket(pause_duration)
#
#         if bucket != "NONE":
#             selected_tail = pick_pause_punct(language, bucket)
#         else:
#             selected_tail = tail
#             if INSERT_FOR_NONE and not selected_tail:
#                 if since_terminal >= TERM_LIMIT[language]:
#                     selected_tail = pick_pause_punct(language, "MEDIUM")
#                 elif since_separator >= SEP_LIMIT[language]:
#                     selected_tail = pick_pause_punct(language, "SHORT")
#
#         output_parts.append(base + selected_tail)
#         effective_tail = selected_tail if selected_tail else tail
#         added_length = len(base)
#
#         if is_terminal(language, effective_tail):
#             since_terminal = 0
#             since_separator = 0
#         elif is_separator(language, effective_tail):
#             since_separator = 0
#             since_terminal += added_length
#         else:
#             since_separator += added_length
#             since_terminal += added_length
#
#     last_text = normalized_segments[-1]
#     last_base, last_tail = split_tail_punct(last_text, language)
#     if len(last_base.strip()) > 10 and not is_terminal(language, last_tail):
#         last_tail = pick_pause_punct(language, "MEDIUM")
#
#     output_parts.append(last_base + last_tail)
#     joined = "".join(output_parts)
#     joined = normalize_zh_defaults(joined)
#     joined = re.sub(r"([，。！？；：、（《”’)])\1+", r"\1", joined)
#     joined = re.sub(r"(……)+", "……", joined)
#     return joined.strip()
#
#
# # ============================================================
# # 7. Wav2Vec 2.0 feature extraction
# # ============================================================
# def load_wav2vec2_resources(device: torch.device):
#     processor = Wav2Vec2Processor.from_pretrained(
#         str(WAV2VEC2_DIR),
#         local_files_only=True,
#     )
#     model = Wav2Vec2Model.from_pretrained(
#         str(WAV2VEC2_DIR),
#         local_files_only=True,
#     )
#     model.to(device=device, dtype=torch.float32)
#     model.eval()
#     model.requires_grad_(False)
#     return processor, model
#
#
# def slice_audio_array(audio: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
#     overlap_samples = W2V_SAMPLE_RATE * OVERLAP_SECONDS
#     hop_size = max(1, SEGMENT_SAMPLES - overlap_samples)
#
#     segments: List[np.ndarray] = []
#     masks: List[np.ndarray] = []
#
#     for start in range(0, len(audio), hop_size):
#         end = start + SEGMENT_SAMPLES
#         segment = audio[start:end]
#
#         if len(segment) < SEGMENT_SAMPLES:
#             real_length = len(segment)
#             segment = np.pad(
#                 segment,
#                 (0, SEGMENT_SAMPLES - len(segment)),
#                 mode="constant",
#             )
#         else:
#             real_length = SEGMENT_SAMPLES
#
#         mask = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
#         mask[:real_length] = 1.0
#
#         segments.append(segment.astype(np.float32, copy=False))
#         masks.append(mask)
#
#         if end >= len(audio):
#             break
#
#     if not segments:
#         raise ValueError("No Wav2Vec 2.0 segments were generated.")
#
#     return segments, masks
#
#
# def combine_last4_hidden(hidden_states) -> torch.Tensor:
#     encoder_layers = hidden_states[1:]
#     if not encoder_layers:
#         return hidden_states[-1]
#     selected = encoder_layers[-4:] if len(encoder_layers) >= 4 else encoder_layers
#     return torch.stack(selected, dim=0).mean(dim=0)
#
#
# def extract_wav2vec2_features(
#     audio: np.ndarray,
#     processor,
#     model,
#     device: torch.device,
# ) -> Tuple[np.ndarray, int]:
#     segments, masks = slice_audio_array(audio)
#     embeddings: List[np.ndarray] = []
#
#     for segment, mask in zip(segments, masks):
#         inputs = processor(
#             segment,
#             sampling_rate=W2V_SAMPLE_RATE,
#             return_tensors="pt",
#             padding=True,
#             return_attention_mask=True,
#         )
#
#         input_values = inputs.input_values.to(device=device, dtype=torch.float32)
#         if not hasattr(inputs, "attention_mask"):
#             raise RuntimeError("Wav2Vec2Processor did not return attention_mask.")
#
#         attention_mask = torch.zeros_like(inputs.attention_mask, device=device)
#         real_length = min(int(mask.sum()), attention_mask.size(1))
#         attention_mask[:, :real_length] = 1
#         input_length = attention_mask.sum(dim=1)
#
#         with torch.inference_mode():
#             outputs = model(
#                 input_values,
#                 attention_mask=attention_mask,
#                 output_hidden_states=True,
#             )
#             mixed_hidden = combine_last4_hidden(outputs.hidden_states)
#             hidden = mixed_hidden.squeeze(0)
#
#             if hasattr(model, "_get_feat_extract_output_lengths"):
#                 frame_length = int(
#                     model._get_feat_extract_output_lengths(input_length).item()
#                 )
#             else:
#                 frame_length = hidden.shape[0]
#
#             frame_length = min(frame_length, hidden.shape[0])
#             if frame_length <= 0:
#                 raise ValueError("Invalid Wav2Vec 2.0 feature-frame length.")
#
#             embedding = hidden[:frame_length].mean(dim=0)
#
#         embeddings.append(
#             embedding.detach().cpu().numpy().astype(np.float32, copy=False)
#         )
#
#         del outputs, mixed_hidden, hidden, embedding
#         del input_values, attention_mask
#
#     return np.vstack(embeddings).astype(np.float32, copy=False), len(segments)
#
#
# # ============================================================
# # 8. BERT feature extraction
# # ============================================================
# def load_bert_resources(device: torch.device):
#     tokenizer = AutoTokenizer.from_pretrained(
#         str(BERT_DIR),
#         local_files_only=True,
#     )
#     model = AutoModel.from_pretrained(
#         str(BERT_DIR),
#         local_files_only=True,
#     )
#     model.to(device=device, dtype=torch.float32)
#     model.eval()
#     model.requires_grad_(False)
#     return tokenizer, model
#
#
# def normalize_bert_text(text: str) -> str:
#     text = text.replace("\u3000", " ").replace("\xa0", " ")
#     return re.sub(r"\s+", " ", text).strip()
#
#
# def chunk_text(
#     input_ids: torch.Tensor,
#     attention_mask: torch.Tensor,
# ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
#     token_length = input_ids.shape[1]
#     if token_length <= BERT_MAX_LEN:
#         return [(input_ids, attention_mask)]
#
#     chunks: List[Tuple[torch.Tensor, torch.Tensor]] = []
#     start = 0
#     while start < token_length:
#         end = min(start + BERT_MAX_LEN, token_length)
#         chunks.append((input_ids[:, start:end], attention_mask[:, start:end]))
#         if end == token_length:
#             break
#         start = max(0, end - BERT_STRIDE)
#
#     return chunks
#
#
# def extract_bert_features(
#     raw_text: str,
#     tokenizer,
#     model,
#     device: torch.device,
# ) -> Tuple[np.ndarray, int, int]:
#     text = normalize_bert_text(raw_text)
#
#     if not text:
#         hidden_size = int(model.config.hidden_size)
#         return np.zeros((1, hidden_size), dtype=np.float32), 0, 1
#
#     encoded = tokenizer(
#         text,
#         return_tensors="pt",
#         add_special_tokens=True,
#         truncation=False,
#         max_length=BERT_TOKENIZER_MAX_LEN,
#     )
#
#     input_ids = encoded["input_ids"].to(device)
#     attention_mask = encoded["attention_mask"].to(device)
#     token_count = int(attention_mask.sum().item())
#     chunks = chunk_text(input_ids, attention_mask)
#
#     embeddings: List[np.ndarray] = []
#     for ids_chunk, mask_chunk in chunks:
#         with torch.inference_mode():
#             outputs = model(
#                 input_ids=ids_chunk,
#                 attention_mask=mask_chunk,
#                 output_hidden_states=True,
#             )
#             mixed_hidden = combine_last4_hidden(outputs.hidden_states)
#             hidden = mixed_hidden.squeeze(0)
#             valid_length = min(int(mask_chunk.sum().item()), hidden.shape[0])
#             if valid_length <= 0:
#                 raise ValueError("Invalid BERT valid-token length.")
#             embedding = hidden[:valid_length].mean(dim=0)
#
#         embeddings.append(
#             embedding.detach().cpu().numpy().astype(np.float32, copy=False)
#         )
#
#         del outputs, mixed_hidden, hidden, embedding
#
#     return np.vstack(embeddings).astype(np.float32, copy=False), token_count, len(chunks)
#
#
# # ============================================================
# # 9. SIST model
# # ============================================================
# def load_sist_model(
#     seed: int,
#     audio_dim: int,
#     text_dim: int,
#     device: torch.device,
# ) -> MultiModalNet:
#     model = MultiModalNet(
#         audio_dim=audio_dim,
#         text_dim=text_dim,
#         num_classes=SIST_NUM_CLASSES,
#         fusion_type=SIST_FUSION_TYPE,
#         dropout=SIST_DROPOUT,
#         shared_dim=SIST_SHARED_DIM,
#     ).to(device)
#
#     state_dict = torch.load(get_sist_checkpoint(seed), map_location=device)
#     model.load_state_dict(state_dict, strict=True)
#     model.eval()
#
#     # Do not freeze SIST parameters. Trainable Parameters in Table 12 reflect
#     # the training-time model property even though inference_mode is used here.
#     return model
#
#
# def prepare_sist_inputs(
#     audio_features: np.ndarray,
#     text_features: np.ndarray,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     audio_x = torch.from_numpy(audio_features).unsqueeze(0)
#     text_x = torch.from_numpy(text_features).unsqueeze(0)
#     audio_mask = torch.ones((1, audio_x.shape[1]), dtype=torch.bool)
#     text_mask = torch.ones((1, text_x.shape[1]), dtype=torch.bool)
#     return audio_x, audio_mask, text_x, text_mask
#
#
# def run_sist(
#     model: MultiModalNet,
#     audio_features: np.ndarray,
#     text_features: np.ndarray,
#     device: torch.device,
# ) -> Tuple[int, np.ndarray]:
#     audio_x, audio_mask, text_x, text_mask = prepare_sist_inputs(
#         audio_features,
#         text_features,
#     )
#
#     audio_x = audio_x.to(device)
#     audio_mask = audio_mask.to(device)
#     text_x = text_x.to(device)
#     text_mask = text_mask.to(device)
#
#     with torch.inference_mode():
#         logits = model(audio_x, audio_mask, text_x, text_mask)
#         probabilities = torch.softmax(logits, dim=1)
#         prediction = torch.argmax(probabilities, dim=1)
#
#     prediction_value = int(prediction.cpu().item())
#     probability_values = probabilities.cpu().numpy()[0]
#     return prediction_value, probability_values
#
#
# # ============================================================
# # 10. Parameters
# # ============================================================
# def count_model_parameters(model: torch.nn.Module) -> Tuple[int, int]:
#     total = sum(parameter.numel() for parameter in model.parameters())
#     trainable = sum(
#         parameter.numel()
#         for parameter in model.parameters()
#         if parameter.requires_grad
#     )
#     return total, trainable
#
#
# def build_parameter_summary(
#     whisper_model,
#     wav2vec2_model,
#     bert_model,
#     sist_model,
# ) -> Tuple[float, float, Dict[str, float]]:
#     stage_models = {
#         "Whisper-large-v3": whisper_model,
#         "Wav2Vec 2.0": wav2vec2_model,
#         "BERT": bert_model,
#         "SIST": sist_model,
#     }
#
#     stage_params: Dict[str, float] = {}
#     total_params = 0
#     trainable_params = 0
#
#     for name, model in stage_models.items():
#         total, trainable = count_model_parameters(model)
#         stage_params[name] = total / 1e6
#         total_params += total
#         trainable_params += trainable
#
#     return total_params / 1e6, trainable_params / 1e6, stage_params
#
#
# # ============================================================
# # 11. Complete pipeline execution
# # ============================================================
# def run_complete_pipeline(
#     audio: np.ndarray,
#     sample_rate: int,
#     whisper_asr,
#     wav2vec2_processor,
#     wav2vec2_model,
#     bert_tokenizer,
#     bert_model,
#     sist_model,
#     device: torch.device,
# ) -> Dict[str, object]:
#     """
#     Execute raw waveform -> automatic transcript -> speech/text features -> SIST.
#
#     The two feature branches are executed sequentially on one GPU to provide a
#     deterministic single-device latency measurement.
#     """
#     whisper_result = transcribe_one(
#         audio=audio,
#         sample_rate=sample_rate,
#         asr=whisper_asr,
#     )
#     transcript = stitch_with_pause_override(whisper_result, LANGUAGE)
#     del whisper_result
#
#     text_features, token_count, bert_chunk_count = extract_bert_features(
#         raw_text=transcript,
#         tokenizer=bert_tokenizer,
#         model=bert_model,
#         device=device,
#     )
#
#     audio_features, wav2vec2_segment_count = extract_wav2vec2_features(
#         audio=audio,
#         processor=wav2vec2_processor,
#         model=wav2vec2_model,
#         device=device,
#     )
#
#     prediction, probabilities = run_sist(
#         model=sist_model,
#         audio_features=audio_features,
#         text_features=text_features,
#         device=device,
#     )
#
#     result = {
#         "Transcript Length": len(transcript),
#         "BERT Token Count": token_count,
#         "BERT Chunk Count": bert_chunk_count,
#         "Wav2Vec2 Segment Count": wav2vec2_segment_count,
#         "Audio Sequence Length": int(audio_features.shape[0]),
#         "Text Sequence Length": int(text_features.shape[0]),
#         "Prediction": prediction,
#         "Probabilities": probabilities,
#     }
#
#     del transcript, text_features, audio_features
#     return result
#
#
# def warm_up(
#     audio_path: Path,
#     whisper_asr,
#     wav2vec2_processor,
#     wav2vec2_model,
#     bert_tokenizer,
#     bert_model,
#     sist_model,
#     device: torch.device,
# ) -> None:
#     audio, sample_rate, _ = load_wav_audio(audio_path)
#
#     for _ in range(WARMUP_RUNS):
#         result = run_complete_pipeline(
#             audio=audio,
#             sample_rate=sample_rate,
#             whisper_asr=whisper_asr,
#             wav2vec2_processor=wav2vec2_processor,
#             wav2vec2_model=wav2vec2_model,
#             bert_tokenizer=bert_tokenizer,
#             bert_model=bert_model,
#             sist_model=sist_model,
#             device=device,
#         )
#         del result
#
#     torch.cuda.synchronize(device)
#     del audio
#
#
# def profile_recording(
#     audio_path: Path,
#     whisper_asr,
#     wav2vec2_processor,
#     wav2vec2_model,
#     bert_tokenizer,
#     bert_model,
#     sist_model,
#     device: torch.device,
# ) -> Dict[str, object]:
#     # Raw WAV I/O and PCM decoding are intentionally excluded from timing.
#     audio, sample_rate, audio_duration = load_wav_audio(audio_path)
#     class_name, sample_name = get_class_and_sample(audio_path)
#     target = CLASS_MAP[class_name]
#
#     torch.cuda.synchronize(device)
#     torch.cuda.reset_peak_memory_stats(device)
#
#     start_time = time.perf_counter()
#
#     pipeline_result = run_complete_pipeline(
#         audio=audio,
#         sample_rate=sample_rate,
#         whisper_asr=whisper_asr,
#         wav2vec2_processor=wav2vec2_processor,
#         wav2vec2_model=wav2vec2_model,
#         bert_tokenizer=bert_tokenizer,
#         bert_model=bert_model,
#         sist_model=sist_model,
#         device=device,
#     )
#
#     torch.cuda.synchronize(device)
#     inference_time = time.perf_counter() - start_time
#
#     peak_gpu_memory_gb = torch.cuda.max_memory_allocated(device) / 1e9
#
#     record = {
#         "Class": class_name,
#         "Sample": sample_name,
#         "Audio Duration (s)": audio_duration,
#         "Transcript Length": pipeline_result["Transcript Length"],
#         "BERT Token Count": pipeline_result["BERT Token Count"],
#         "BERT Chunk Count": pipeline_result["BERT Chunk Count"],
#         "Wav2Vec2 Segment Count": pipeline_result["Wav2Vec2 Segment Count"],
#         "Audio Sequence Length": pipeline_result["Audio Sequence Length"],
#         "Text Sequence Length": pipeline_result["Text Sequence Length"],
#         "Target": target,
#         "Prediction": pipeline_result["Prediction"],
#         "Inference Time (s)": inference_time,
#         "Peak GPU Memory (GB)": peak_gpu_memory_gb,
#         "RTF": inference_time / audio_duration,
#     }
#
#     del pipeline_result, audio
#     return record
#
#
# # ============================================================
# # 12. Summary and CSV output
# # ============================================================
# def build_summary(
#     total_params_m: float,
#     trainable_params_m: float,
#     records: List[Dict[str, object]],
# ) -> Dict[str, object]:
#     total_time = sum(float(record["Inference Time (s)"]) for record in records)
#     total_duration = sum(float(record["Audio Duration (s)"]) for record in records)
#
#     if not records or total_duration <= 0:
#         raise ValueError("Invalid complete-pipeline profiling records.")
#
#     return {
#         "Stage": "Complete pipeline",
#         "Function": "Automatic recognition",
#         "Parameters (M)": total_params_m,
#         "Trainable Parameters (M)": trainable_params_m,
#         "Inference Time (s/recording)": total_time / len(records),
#         "Peak GPU Memory (GB)": max(
#             float(record["Peak GPU Memory (GB)"])
#             for record in records
#         ),
#         "RTF": total_time / total_duration,
#     }
#
#
# def save_records(records: List[Dict[str, object]]) -> None:
#     fieldnames = [
#         "Class",
#         "Sample",
#         "Audio Duration (s)",
#         "Transcript Length",
#         "BERT Token Count",
#         "BERT Chunk Count",
#         "Wav2Vec2 Segment Count",
#         "Audio Sequence Length",
#         "Text Sequence Length",
#         "Target",
#         "Prediction",
#         "Inference Time (s)",
#         "Peak GPU Memory (GB)",
#         "RTF",
#     ]
#
#     with RECORD_CSV.open("w", newline="", encoding="utf-8-sig") as file:
#         writer = csv.DictWriter(file, fieldnames=fieldnames)
#         writer.writeheader()
#         for record in records:
#             row = dict(record)
#             for key in (
#                 "Audio Duration (s)",
#                 "Inference Time (s)",
#                 "Peak GPU Memory (GB)",
#                 "RTF",
#             ):
#                 row[key] = f"{float(row[key]):.6f}"
#             writer.writerow(row)
#
#
# def save_summary(summary: Dict[str, object]) -> None:
#     fieldnames = [
#         "Stage",
#         "Function",
#         "Parameters (M)",
#         "Trainable Parameters (M)",
#         "Inference Time (s/recording)",
#         "Peak GPU Memory (GB)",
#         "RTF",
#     ]
#
#     with SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as file:
#         writer = csv.DictWriter(file, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerow({
#             "Stage": summary["Stage"],
#             "Function": summary["Function"],
#             "Parameters (M)": f'{float(summary["Parameters (M)"]):.6f}',
#             "Trainable Parameters (M)": (
#                 f'{float(summary["Trainable Parameters (M)"]):.6f}'
#             ),
#             "Inference Time (s/recording)": (
#                 f'{float(summary["Inference Time (s/recording)"]):.6f}'
#             ),
#             "Peak GPU Memory (GB)": (
#                 f'{float(summary["Peak GPU Memory (GB)"]):.6f}'
#             ),
#             "RTF": f'{float(summary["RTF"]):.6f}',
#         })
#
#
# def print_summary(
#     summary: Dict[str, object],
#     stage_params: Dict[str, float],
#     recording_count: int,
#     seed: int,
#     gpu_name: str,
# ) -> None:
#     print("\n" + "=" * 76)
#     print("Complete pipeline computational profile")
#     print("=" * 76)
#     print(f"GPU: {gpu_name}")
#     print(f"SIST checkpoint seed: {seed}")
#     print(f"Recordings: {recording_count}")
#     print("Stage Parameters (M):")
#     for name, value in stage_params.items():
#         print(f"  {name:<18}: {value:.6f}")
#     print(f'Parameters (M): {float(summary["Parameters (M)"]):.6f}')
#     print(
#         "Trainable Parameters (M): "
#         f'{float(summary["Trainable Parameters (M)"]):.6f}'
#     )
#     print(
#         "Inference Time (s/recording): "
#         f'{float(summary["Inference Time (s/recording)"]):.6f}'
#     )
#     print(
#         "Peak GPU Memory (GB): "
#         f'{float(summary["Peak GPU Memory (GB)"]):.6f}'
#     )
#     print(f'RTF: {float(summary["RTF"]):.6f}')
#     print("=" * 76)
#
#
# # ============================================================
# # 13. Main
# # ============================================================
# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description="Direct end-to-end profiling of the complete automatic pipeline."
#     )
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=DEFAULT_SEED,
#         choices=SUPPORTED_SEEDS,
#         help=(
#             "SIST checkpoint seed. Computational structure is identical across "
#             "seeds; default=2024."
#         ),
#     )
#     args = parser.parse_args()
#
#     audio_files = validate_paths(args.seed)
#     device = get_cuda_device()
#     gpu_name = torch.cuda.get_device_name(device)
#
#     print(f"[INFO] GPU: {gpu_name}")
#     print(f"[INFO] Audio dataset: {AUDIO_ROOT / TEST_SPLIT}")
#     print(f"[INFO] Text dataset: {TEXT_ROOT / TEST_SPLIT}")
#     print(f"[INFO] Test recordings: {len(audio_files)}")
#     print(f"[INFO] Whisper: {WHISPER_DIR}")
#     print(f"[INFO] Wav2Vec 2.0: {WAV2VEC2_DIR}")
#     print(f"[INFO] BERT: {BERT_DIR}")
#     print(f"[INFO] SIST checkpoint: {get_sist_checkpoint(args.seed)}")
#     print("[INFO] Stored text is used only for sample-pair validation.")
#     print("[INFO] Timed input is raw in-memory waveform.")
#     print("[INFO] Raw WAV file I/O/PCM decode are excluded from latency.")
#
#     transformers_logging.set_verbosity_error()
#
#     print("[INFO] Loading Whisper-large-v3...")
#     whisper_model, whisper_asr = load_whisper_resources(device)
#
#     print("[INFO] Loading Wav2Vec 2.0...")
#     wav2vec2_processor, wav2vec2_model = load_wav2vec2_resources(device)
#
#     print("[INFO] Loading BERT...")
#     bert_tokenizer, bert_model = load_bert_resources(device)
#
#     audio_dim = int(wav2vec2_model.config.hidden_size)
#     text_dim = int(bert_model.config.hidden_size)
#
#     print("[INFO] Loading SIST...")
#     sist_model = load_sist_model(
#         seed=args.seed,
#         audio_dim=audio_dim,
#         text_dim=text_dim,
#         device=device,
#     )
#
#     total_params_m, trainable_params_m, stage_params = build_parameter_summary(
#         whisper_model=whisper_model,
#         wav2vec2_model=wav2vec2_model,
#         bert_model=bert_model,
#         sist_model=sist_model,
#     )
#
#     # Frozen front ends must have zero trainable parameters.
#     for name, model in (
#         ("Whisper-large-v3", whisper_model),
#         ("Wav2Vec 2.0", wav2vec2_model),
#         ("BERT", bert_model),
#     ):
#         _, trainable = count_model_parameters(model)
#         if trainable != 0:
#             raise RuntimeError(f"{name} unexpectedly has trainable parameters.")
#
#     _, sist_trainable = count_model_parameters(sist_model)
#     if trainable_params_m != sist_trainable / 1e6:
#         raise RuntimeError(
#             "Complete-pipeline trainable parameters must equal SIST trainable parameters."
#         )
#
#     print(f"[INFO] Warm-up runs: {WARMUP_RUNS}")
#     warm_up(
#         audio_path=audio_files[0],
#         whisper_asr=whisper_asr,
#         wav2vec2_processor=wav2vec2_processor,
#         wav2vec2_model=wav2vec2_model,
#         bert_tokenizer=bert_tokenizer,
#         bert_model=bert_model,
#         sist_model=sist_model,
#         device=device,
#     )
#
#     records: List[Dict[str, object]] = []
#
#     for audio_path in tqdm(audio_files, desc="Profiling complete pipeline"):
#         record = profile_recording(
#             audio_path=audio_path,
#             whisper_asr=whisper_asr,
#             wav2vec2_processor=wav2vec2_processor,
#             wav2vec2_model=wav2vec2_model,
#             bert_tokenizer=bert_tokenizer,
#             bert_model=bert_model,
#             sist_model=sist_model,
#             device=device,
#         )
#         records.append(record)
#
#     summary = build_summary(
#         total_params_m=total_params_m,
#         trainable_params_m=trainable_params_m,
#         records=records,
#     )
#
#     save_records(records)
#     save_summary(summary)
#     print_summary(
#         summary=summary,
#         stage_params=stage_params,
#         recording_count=len(records),
#         seed=args.seed,
#         gpu_name=gpu_name,
#     )
#
#     print(f"[INFO] Summary CSV: {SUMMARY_CSV.resolve()}")
#     print(f"[INFO] Record CSV: {RECORD_CSV.resolve()}")
#
#
# if __name__ == "__main__":
#     main()
import os
# Force Transformers to use local files only during formal profiling.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse
import csv
import re
import sys
import time
import types
import wave
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import opencc
import torch
from tqdm import tqdm

# ========== PyTorch 1.10 兼容补丁：伪造 FSDP 模块 ==========
# 高版本 Transformers 内部会导入 torch.distributed.fsdp 做分布式检查
# PyTorch 1.10 无此模块，单卡推理场景也完全不会用到 FSDP 功能
# 伪造空的虚拟模块绕过导入报错，不影响任何实际推理逻辑
if not hasattr(torch.distributed, "fsdp"):
    _fake_fsdp_module = types.ModuleType("torch.distributed.fsdp")
    _fake_fsdp_module.FullyShardedDataParallel = type("FullyShardedDataParallel", (), {})
    sys.modules["torch.distributed.fsdp"] = _fake_fsdp_module
    torch.distributed.fsdp = _fake_fsdp_module
# ==========================================================

from transformers import (
    AutoModel,
    AutoTokenizer,
    Wav2Vec2Model,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)
from transformers.utils import logging as transformers_logging
from model import MultiModalNet

# ============================================================
# 1. Paths and experiment configuration
# ============================================================
# Expected layout:
# Computational_cost/
# ├── NCMMSC2021/
# ├── NCMMSC2021_text/
# ├── bert-base-chinese/
# ├── wav2vec2-base/
# ├── whisper-large-v3/
# ├── SIST/
# │   ├── weights_0/best.pth
# │   ├── weights_1/best.pth
# │   ├── weights_42/best.pth
# │   ├── weights_123/best.pth
# │   └── weights_2024/best.pth
# └── Complete_pipeline/
#     ├── profile_complete_pipeline.py
#     ├── model.py
#     ├── model_audio.py
#     └── model_text.py
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
AUDIO_ROOT = ROOT_DIR / "NCMMSC2021"
TEXT_ROOT = ROOT_DIR / "NCMMSC2021_text"
WHISPER_DIR = ROOT_DIR / "whisper-large-v3"
WAV2VEC2_DIR = ROOT_DIR / "wav2vec2-base"
BERT_DIR = ROOT_DIR / "bert-base-chinese"
SIST_ROOT = ROOT_DIR / "SIST"
SUMMARY_CSV = SCRIPT_DIR / "complete_pipeline_profile_summary.csv"
RECORD_CSV = SCRIPT_DIR / "complete_pipeline_profile_records.csv"
TEST_SPLIT = "test"
CLASS_MAP = {"AD": 0, "HC": 1, "MCI": 2}
SUPPORTED_SEEDS = (0, 1, 42, 123, 2024)
DEFAULT_SEED = 2024
WARMUP_RUNS = 5
# Complete-pipeline timing protocol:
# - raw WAV disk read / PCM decode: excluded;
# - Whisper transcription and transcript post-processing: included;
# - BERT text feature extraction: included;
# - Wav2Vec 2.0 speech feature extraction: included;
# - CPU/GPU transfers between stages: included;
# - SIST forward + softmax + argmax: included.
# All four models remain resident on the same CUDA device during profiling.

# ============================================================
# 2. Whisper configuration and transcript post-processing
# ============================================================
LANGUAGE = "zh"
EXPECTED_SAMPLE_RATE = 16000
WHISPER_GENERATE_KWARGS = {
    "language": LANGUAGE,
    "task": "transcribe",
    "num_beams": 1,
    "do_sample": False,
    "condition_on_prev_tokens": False,
    "temperature": 0.0,
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    # Whisper 的 decoder 最大总长度为 448 token。
    # 当前中文 transcribe + timestamps 模式会先占用 3 个 decoder
    # special/prompt token，因此 max_new_tokens 不能再设为 448。
    # 448 - 3 = 445，显式设为 445 可与当前 Transformers 版本兼容，
    # 同时保持与 Whisper 原始 448-token decoder 上限一致。
    "max_new_tokens": 445,
}
PAUSE_T_SHORT = 0.5
PAUSE_T_MEDIUM = 1.0
PAUSE_T_LONG = 1.5
INSERT_FOR_NONE = True
SEP_LIMIT = {"zh": 18, "en": 30}
TERM_LIMIT = {"zh": 40, "en": 80}
PAUSE_PUNCT = {
    "zh": {"NONE": "", "SHORT": "，", "MEDIUM": "。", "LONG": "……"},
    "en": {"NONE": "", "SHORT": ",", "MEDIUM": ".", "LONG": "..."},
    "default": {"NONE": "", "SHORT": ",", "MEDIUM": ".", "LONG": "..."},
}
def create_chinese_converter():
    """
    Create an OpenCC Traditional-to-Simplified converter.
    opencc-python-reimplemented expects the configuration name without
    the .json suffix ("t2s"). Some other OpenCC Python bindings accept
    "t2s.json" instead, so both forms are tried for portability.
    """
    errors = []
    for config_name in ("t2s", "t2s.json"):
        try:
            return opencc.OpenCC(config_name)
        except (FileNotFoundError, OSError, ValueError) as error:
            errors.append(f"{config_name}: {error}")
    raise RuntimeError(
        "Unable to initialize OpenCC Traditional-to-Simplified conversion. "
        "Install opencc-python-reimplemented and verify its config files.\n"
        + "\n".join(errors)
    )
_CN_CONVERTER = create_chinese_converter()
_CN_ASCII_TO_FULL = str.maketrans({
    ",": "，",
    ".": "。",
    "?": "？",
    "!": "！",
    ":": "：",
    ";": "；",
})
_RE_TAIL_PUNCT_ZH = re.compile(r"[，。！？；：、…]+$")
_RE_TAIL_PUNCT_EN = re.compile(r"[,\.!?;:]+$")

# ============================================================
# 3. Wav2Vec 2.0 / BERT feature configuration
# ============================================================
# Wav2Vec 2.0: same final configuration as the standalone profiler.
W2V_SAMPLE_RATE = 16000
SEGMENT_SECONDS = 6
OVERLAP_SECONDS = 3
SEGMENT_SAMPLES = W2V_SAMPLE_RATE * SEGMENT_SECONDS
# BERT: same final configuration as the standalone profiler.
BERT_MAX_LEN = 256
BERT_STRIDE = 64
BERT_TOKENIZER_MAX_LEN = 10000
# SIST: same S4 configuration as the current test code.
SIST_FUSION_TYPE = "gated_bi_cross_attention"
SIST_DROPOUT = 0.5
SIST_SHARED_DIM = 512
SIST_NUM_CLASSES = 3

# ============================================================
# 4. Dataset and path validation
# ============================================================
def enumerate_audio_files(dataset_root: Path) -> List[Path]:
    split_dir = dataset_root / TEST_SPLIT
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")
    return sorted(
        path
        for path in split_dir.rglob("*.wav")
        if path.is_file()
    )
def get_paired_text_path(audio_path: Path) -> Path:
    relative_path = audio_path.relative_to(AUDIO_ROOT)
    return (TEXT_ROOT / relative_path).with_suffix(".txt")
def get_class_and_sample(audio_path: Path) -> Tuple[str, str]:
    relative_path = audio_path.relative_to(AUDIO_ROOT / TEST_SPLIT)
    if len(relative_path.parts) < 2:
        raise ValueError(f"Unexpected test path: {audio_path}")
    class_name = relative_path.parts[0]
    if class_name not in CLASS_MAP:
        raise ValueError(f"Unknown class directory: {class_name}")
    return class_name, audio_path.stem
def get_sist_checkpoint(seed: int) -> Path:
    return SIST_ROOT / f"weights_{seed}" / "best.pth"
def validate_audio_text_pairs(audio_files: List[Path]) -> None:
    """
    NCMMSC2021_text is used only to verify one-to-one sample identity.
    Stored text is intentionally NOT used as the BERT input in the timed
    complete pipeline; using it would bypass Whisper and would no longer be
    an automatic raw-speech-to-prediction pipeline.
    """
    missing_text: List[str] = []
    audio_relative_txt = {
        str(audio.relative_to(AUDIO_ROOT).with_suffix(".txt"))
        for audio in audio_files
    }
    for audio_path in audio_files:
        text_path = get_paired_text_path(audio_path)
        if not text_path.is_file():
            missing_text.append(str(text_path))
    if missing_text:
        preview = "\n".join(missing_text[:10])
        raise RuntimeError(
            "Some test WAV files do not have paired text files:\n" + preview
        )
    text_split = TEXT_ROOT / TEST_SPLIT
    text_relative = {
        str(path.relative_to(TEXT_ROOT))
        for path in text_split.rglob("*.txt")
        if path.is_file()
    }
    extra_text = sorted(text_relative - audio_relative_txt)
    if extra_text:
        preview = "\n".join(extra_text[:10])
        raise RuntimeError(
            "NCMMSC2021_text contains test samples without paired WAV files:\n"
            + preview
        )
def validate_model_directory(model_dir: Path, name: str) -> None:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"{name} directory not found: {model_dir}")
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json in: {model_dir}")
def validate_paths(seed: int) -> List[Path]:
    if not AUDIO_ROOT.is_dir():
        raise FileNotFoundError(f"Audio dataset not found: {AUDIO_ROOT}")
    if not TEXT_ROOT.is_dir():
        raise FileNotFoundError(f"Text dataset not found: {TEXT_ROOT}")
    if not SIST_ROOT.is_dir():
        raise FileNotFoundError(f"SIST directory not found: {SIST_ROOT}")
    validate_model_directory(WHISPER_DIR, "Whisper")
    validate_model_directory(WAV2VEC2_DIR, "Wav2Vec 2.0")
    validate_model_directory(BERT_DIR, "BERT")
    checkpoint = get_sist_checkpoint(seed)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SIST checkpoint not found: {checkpoint}")
    audio_files = enumerate_audio_files(AUDIO_ROOT)
    if not audio_files:
        raise RuntimeError(f"No WAV files found in: {AUDIO_ROOT / TEST_SPLIT}")
    validate_audio_text_pairs(audio_files)
    validate_audio_headers(audio_files)
    return audio_files
def get_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Complete-pipeline profiling requires a CUDA GPU."
        )
    return torch.device("cuda:0")

# ============================================================
# 5. WAV loading
# ============================================================
def decode_pcm_samples(raw_bytes: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        samples = np.frombuffer(raw_bytes, dtype=np.uint8)
        return (samples.astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        samples = np.frombuffer(raw_bytes, dtype="<i2")
        return samples.astype(np.float32) / 32768.0
    if sample_width == 3:
        byte_array = np.frombuffer(raw_bytes, dtype=np.uint8)
        if byte_array.size % 3 != 0:
            raise ValueError("Invalid 24-bit PCM byte length.")
        byte_array = byte_array.reshape(-1, 3)
        samples = (
            byte_array[:, 0].astype(np.int32)
            | (byte_array[:, 1].astype(np.int32) << 8)
            | (byte_array[:, 2].astype(np.int32) << 16)
        )
        samples = np.where(samples & 0x800000, samples - 0x1000000, samples)
        return samples.astype(np.float32) / 8388608.0
    if sample_width == 4:
        samples = np.frombuffer(raw_bytes, dtype="<i4")
        return samples.astype(np.float32) / 2147483648.0
    raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")
def load_wav_audio(audio_path: Path) -> Tuple[np.ndarray, int, float]:
    """Read one PCM WAV outside the timed region."""
    with wave.open(str(audio_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        compression = wav_file.getcomptype()
        if compression != "NONE":
            raise ValueError(f"Compressed WAV is not supported: {audio_path}")
        if channels <= 0 or sample_rate <= 0 or frame_count <= 0:
            raise ValueError(f"Invalid WAV header: {audio_path}")
        raw_bytes = wav_file.readframes(frame_count)
    audio = decode_pcm_samples(raw_bytes, sample_width)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    duration = frame_count / sample_rate
    return audio, sample_rate, duration
def validate_audio_headers(audio_files: List[Path]) -> None:
    invalid: List[str] = []
    for audio_path in audio_files:
        try:
            with wave.open(str(audio_path), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                compression = wav_file.getcomptype()
            if sample_rate != EXPECTED_SAMPLE_RATE or compression != "NONE":
                invalid.append(
                    f"{audio_path}: sample_rate={sample_rate}, compression={compression}"
                )
        except (wave.Error, OSError) as error:
            invalid.append(f"{audio_path}: {error}")
    if invalid:
        raise RuntimeError(
            "Some test WAV files do not satisfy the PCM/16-kHz protocol:\n"
            + "\n".join(invalid[:10])
        )

# ============================================================
# 6. Whisper transcription and post-processing
# ============================================================
def load_whisper_resources(device: torch.device):
    dtype = torch.float16
    processor = WhisperProcessor.from_pretrained(
        str(WHISPER_DIR),
        local_files_only=True,
    )
    # 【修正】移除 from_pretrained 中不兼容的 dtype、use_safetensors 参数
    # 低/高版本 Transformers 通用写法：加载后用 .to() 统一设置设备与精度
    model = WhisperForConditionalGeneration.from_pretrained(
        str(WHISPER_DIR),
        local_files_only=True,
    )
    model.to(device=device, dtype=dtype)
    model.eval()
    model.requires_grad_(False)

    max_target_positions = int(model.config.max_target_positions)
    requested_max_new_tokens = int(
        WHISPER_GENERATE_KWARGS["max_new_tokens"]
    )
    decoder_prefix_tokens = 3
    if requested_max_new_tokens + decoder_prefix_tokens > max_target_positions:
        raise RuntimeError(
            "Invalid Whisper decoding length configuration: "
            f"decoder prefix={decoder_prefix_tokens}, "
            f"max_new_tokens={requested_max_new_tokens}, "
            f"max_target_positions={max_target_positions}."
        )

    # 【修正】移除 pipeline 中不兼容的 dtype 参数
    # 模型已指定精度，pipeline 会自动继承，避免低版本 pipeline 报错
    asr = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        batch_size=1,
    )
    return model, asr

def transcribe_one(audio: np.ndarray, sample_rate: int, asr) -> Dict:
    result = asr(
        {"raw": audio, "sampling_rate": sample_rate},
        return_timestamps=True,
        generate_kwargs=WHISPER_GENERATE_KWARGS,
    )
    segments: List[Dict[str, object]] = []
    for chunk in result.get("chunks", []):
        timestamp = chunk.get("timestamp", (None, None))
        start = timestamp[0] if timestamp else None
        end = timestamp[1] if timestamp else None
        segments.append({
            "text": chunk.get("text", ""),
            "start": start,
            "end": end,
        })
    if not segments and result.get("text"):
        segments.append({
            "text": result["text"],
            "start": None,
            "end": None,
        })
    return {
        "text": result.get("text", ""),
        "segments": segments,
        "language": LANGUAGE,
    }
def normalize_ellipsis_zh(text: str) -> str:
    text = re.sub(r"(\.{3,})", "……", text)
    text = re.sub(r"…{2,}", "……", text)
    return text
def normalize_zh_defaults(text: str) -> str:
    text = normalize_ellipsis_zh(text.translate(_CN_ASCII_TO_FULL))
    text = re.sub(r"\s*([，。！？；：、（）《》“”‘’……])\s*", r"\1", text)
    text = re.sub(r"([，。！？；：、（《”’)])\1+", r"\1", text)
    text = re.sub(r"(……)+", "……", text)
    text = re.sub(r"^[，。！？；：、……]+", "", text)
    return text.strip()
def convert_to_simplified_chinese(text: str) -> str:
    return _CN_CONVERTER.convert(text)
def split_tail_punct(text: str, language: str) -> Tuple[str, str]:
    stripped = text.rstrip()
    if not stripped:
        return "", ""
    pattern = _RE_TAIL_PUNCT_ZH if language == "zh" else _RE_TAIL_PUNCT_EN
    match = pattern.search(stripped)
    if match:
        return stripped[:match.start()], stripped[match.start():]
    return stripped, ""
def pause_bucket(duration: float) -> str:
    if duration >= PAUSE_T_LONG:
        return "LONG"
    if duration >= PAUSE_T_MEDIUM:
        return "MEDIUM"
    if duration >= PAUSE_T_SHORT:
        return "SHORT"
    return "NONE"
def pick_pause_punct(language: str, bucket: str) -> str:
    key = language if language in PAUSE_PUNCT else "default"
    return PAUSE_PUNCT[key].get(bucket, "")
def is_terminal(language: str, punct: str) -> bool:
    if not punct:
        return False
    if language == "zh":
        return "……" in punct or any(mark in punct for mark in "。！？")
    return "..." in punct or any(mark in punct for mark in ".?!")
def is_separator(language: str, punct: str) -> bool:
    if not punct:
        return False
    marks = "，；：、" if language == "zh" else ",;:"
    return any(mark in punct for mark in marks)
def stitch_with_pause_override(result: Dict, language: str = "zh") -> str:
    segments = result.get("segments", [])
    if not segments:
        return ""
    normalized_segments: List[str] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if language == "zh":
            text = convert_to_simplified_chinese(normalize_zh_defaults(text))
        normalized_segments.append(text)
    output_parts: List[str] = []
    since_terminal = 0
    since_separator = 0
    for index in range(len(segments) - 1):
        current_text = normalized_segments[index]
        current_end = segments[index].get("end")
        next_start = segments[index + 1].get("start")
        if current_end is None or next_start is None:
            pause_duration = 0.0
        else:
            pause_duration = max(0.0, float(next_start) - float(current_end))
        base, tail = split_tail_punct(current_text, language)
        bucket = pause_bucket(pause_duration)
        if bucket != "NONE":
            selected_tail = pick_pause_punct(language, bucket)
        else:
            selected_tail = tail
            if INSERT_FOR_NONE and not selected_tail:
                if since_terminal >= TERM_LIMIT[language]:
                    selected_tail = pick_pause_punct(language, "MEDIUM")
                elif since_separator >= SEP_LIMIT[language]:
                    selected_tail = pick_pause_punct(language, "SHORT")
        output_parts.append(base + selected_tail)
        effective_tail = selected_tail if selected_tail else tail
        added_length = len(base)
        if is_terminal(language, effective_tail):
            since_terminal = 0
            since_separator = 0
        elif is_separator(language, effective_tail):
            since_separator = 0
            since_terminal += added_length
        else:
            since_separator += added_length
            since_terminal += added_length
    last_text = normalized_segments[-1]
    last_base, last_tail = split_tail_punct(last_text, language)
    if len(last_base.strip()) > 10 and not is_terminal(language, last_tail):
        last_tail = pick_pause_punct(language, "MEDIUM")
    output_parts.append(last_base + last_tail)
    joined = "".join(output_parts)
    joined = normalize_zh_defaults(joined)
    joined = re.sub(r"([，。！？；：、（《”’)])\1+", r"\1", joined)
    joined = re.sub(r"(……)+", "……", joined)
    return joined.strip()

# ============================================================
# 7. Wav2Vec 2.0 feature extraction
# ============================================================
def load_wav2vec2_resources(device: torch.device):
    processor = Wav2Vec2Processor.from_pretrained(
        str(WAV2VEC2_DIR),
        local_files_only=True,
    )
    model = Wav2Vec2Model.from_pretrained(
        str(WAV2VEC2_DIR),
        local_files_only=True,
    )
    model.to(device=device, dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    return processor, model
def slice_audio_array(audio: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    overlap_samples = W2V_SAMPLE_RATE * OVERLAP_SECONDS
    hop_size = max(1, SEGMENT_SAMPLES - overlap_samples)
    segments: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    for start in range(0, len(audio), hop_size):
        end = start + SEGMENT_SAMPLES
        segment = audio[start:end]
        if len(segment) < SEGMENT_SAMPLES:
            real_length = len(segment)
            segment = np.pad(
                segment,
                (0, SEGMENT_SAMPLES - len(segment)),
                mode="constant",
            )
        else:
            real_length = SEGMENT_SAMPLES
        mask = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
        mask[:real_length] = 1.0
        segments.append(segment.astype(np.float32, copy=False))
        masks.append(mask)
        if end >= len(audio):
            break
    if not segments:
        raise ValueError("No Wav2Vec 2.0 segments were generated.")
    return segments, masks
def combine_last4_hidden(hidden_states) -> torch.Tensor:
    encoder_layers = hidden_states[1:]
    if not encoder_layers:
        return hidden_states[-1]
    selected = encoder_layers[-4:] if len(encoder_layers) >= 4 else encoder_layers
    return torch.stack(selected, dim=0).mean(dim=0)
def extract_wav2vec2_features(
    audio: np.ndarray,
    processor,
    model,
    device: torch.device,
) -> Tuple[np.ndarray, int]:
    segments, masks = slice_audio_array(audio)
    embeddings: List[np.ndarray] = []
    for segment, mask in zip(segments, masks):
        inputs = processor(
            segment,
            sampling_rate=W2V_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_values = inputs.input_values.to(device=device, dtype=torch.float32)
        if not hasattr(inputs, "attention_mask"):
            raise RuntimeError("Wav2Vec2Processor did not return attention_mask.")
        attention_mask = torch.zeros_like(inputs.attention_mask, device=device)
        real_length = min(int(mask.sum()), attention_mask.size(1))
        attention_mask[:, :real_length] = 1
        input_length = attention_mask.sum(dim=1)
        with torch.inference_mode():
            outputs = model(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            mixed_hidden = combine_last4_hidden(outputs.hidden_states)
            hidden = mixed_hidden.squeeze(0)
            if hasattr(model, "_get_feat_extract_output_lengths"):
                frame_length = int(
                    model._get_feat_extract_output_lengths(input_length).item()
                )
            else:
                frame_length = hidden.shape[0]
            frame_length = min(frame_length, hidden.shape[0])
            if frame_length <= 0:
                raise ValueError("Invalid Wav2Vec 2.0 feature-frame length.")
            embedding = hidden[:frame_length].mean(dim=0)
        embeddings.append(
            embedding.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        del outputs, mixed_hidden, hidden, embedding
        del input_values, attention_mask
    return np.vstack(embeddings).astype(np.float32, copy=False), len(segments)

# ============================================================
# 8. BERT feature extraction
# ============================================================
def load_bert_resources(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        str(BERT_DIR),
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        str(BERT_DIR),
        local_files_only=True,
    )
    model.to(device=device, dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    return tokenizer, model
def normalize_bert_text(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()
def chunk_text(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    token_length = input_ids.shape[1]
    if token_length <= BERT_MAX_LEN:
        return [(input_ids, attention_mask)]
    chunks: List[Tuple[torch.Tensor, torch.Tensor]] = []
    start = 0
    while start < token_length:
        end = min(start + BERT_MAX_LEN, token_length)
        chunks.append((input_ids[:, start:end], attention_mask[:, start:end]))
        if end == token_length:
            break
        start = max(0, end - BERT_STRIDE)
    return chunks
def extract_bert_features(
    raw_text: str,
    tokenizer,
    model,
    device: torch.device,
) -> Tuple[np.ndarray, int, int]:
    text = normalize_bert_text(raw_text)
    if not text:
        hidden_size = int(model.config.hidden_size)
        return np.zeros((1, hidden_size), dtype=np.float32), 0, 1
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
        max_length=BERT_TOKENIZER_MAX_LEN,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    token_count = int(attention_mask.sum().item())
    chunks = chunk_text(input_ids, attention_mask)
    embeddings: List[np.ndarray] = []
    for ids_chunk, mask_chunk in chunks:
        with torch.inference_mode():
            outputs = model(
                input_ids=ids_chunk,
                attention_mask=mask_chunk,
                output_hidden_states=True,
            )
            mixed_hidden = combine_last4_hidden(outputs.hidden_states)
            hidden = mixed_hidden.squeeze(0)
            valid_length = min(int(mask_chunk.sum().item()), hidden.shape[0])
            if valid_length <= 0:
                raise ValueError("Invalid BERT valid-token length.")
            embedding = hidden[:valid_length].mean(dim=0)
        embeddings.append(
            embedding.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        del outputs, mixed_hidden, hidden, embedding
    return np.vstack(embeddings).astype(np.float32, copy=False), token_count, len(chunks)

# ============================================================
# 9. SIST model
# ============================================================
def load_sist_model(
    seed: int,
    audio_dim: int,
    text_dim: int,
    device: torch.device,
) -> MultiModalNet:
    model = MultiModalNet(
        audio_dim=audio_dim,
        text_dim=text_dim,
        num_classes=SIST_NUM_CLASSES,
        fusion_type=SIST_FUSION_TYPE,
        dropout=SIST_DROPOUT,
        shared_dim=SIST_SHARED_DIM,
    ).to(device)
    state_dict = torch.load(get_sist_checkpoint(seed), map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    # Do not freeze SIST parameters. Trainable Parameters in Table 12 reflect
    # the training-time model property even though inference_mode is used here.
    return model
def prepare_sist_inputs(
    audio_features: np.ndarray,
    text_features: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    audio_x = torch.from_numpy(audio_features).unsqueeze(0)
    text_x = torch.from_numpy(text_features).unsqueeze(0)
    audio_mask = torch.ones((1, audio_x.shape[1]), dtype=torch.bool)
    text_mask = torch.ones((1, text_x.shape[1]), dtype=torch.bool)
    return audio_x, audio_mask, text_x, text_mask
def run_sist(
    model: MultiModalNet,
    audio_features: np.ndarray,
    text_features: np.ndarray,
    device: torch.device,
) -> Tuple[int, np.ndarray]:
    audio_x, audio_mask, text_x, text_mask = prepare_sist_inputs(
        audio_features,
        text_features,
    )
    audio_x = audio_x.to(device)
    audio_mask = audio_mask.to(device)
    text_x = text_x.to(device)
    text_mask = text_mask.to(device)
    with torch.inference_mode():
        logits = model(audio_x, audio_mask, text_x, text_mask)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
    prediction_value = int(prediction.cpu().item())
    probability_values = probabilities.cpu().numpy()[0]
    return prediction_value, probability_values

# ============================================================
# 10. Parameters
# ============================================================
def count_model_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable
def build_parameter_summary(
    whisper_model,
    wav2vec2_model,
    bert_model,
    sist_model,
) -> Tuple[float, float, Dict[str, float]]:
    stage_models = {
        "Whisper-large-v3": whisper_model,
        "Wav2Vec 2.0": wav2vec2_model,
        "BERT": bert_model,
        "SIST": sist_model,
    }
    stage_params: Dict[str, float] = {}
    total_params = 0
    trainable_params = 0
    for name, model in stage_models.items():
        total, trainable = count_model_parameters(model)
        stage_params[name] = total / 1e6
        total_params += total
        trainable_params += trainable
    return total_params / 1e6, trainable_params / 1e6, stage_params

# ============================================================
# 11. Complete pipeline execution
# ============================================================
def run_complete_pipeline(
    audio: np.ndarray,
    sample_rate: int,
    whisper_asr,
    wav2vec2_processor,
    wav2vec2_model,
    bert_tokenizer,
    bert_model,
    sist_model,
    device: torch.device,
) -> Dict[str, object]:
    """
    Execute raw waveform -> automatic transcript -> speech/text features -> SIST.
    The two feature branches are executed sequentially on one GPU to provide a
    deterministic single-device latency measurement.
    """
    whisper_result = transcribe_one(
        audio=audio,
        sample_rate=sample_rate,
        asr=whisper_asr,
    )
    transcript = stitch_with_pause_override(whisper_result, LANGUAGE)
    del whisper_result
    text_features, token_count, bert_chunk_count = extract_bert_features(
        raw_text=transcript,
        tokenizer=bert_tokenizer,
        model=bert_model,
        device=device,
    )
    audio_features, wav2vec2_segment_count = extract_wav2vec2_features(
        audio=audio,
        processor=wav2vec2_processor,
        model=wav2vec2_model,
        device=device,
    )
    prediction, probabilities = run_sist(
        model=sist_model,
        audio_features=audio_features,
        text_features=text_features,
        device=device,
    )
    result = {
        "Transcript Length": len(transcript),
        "BERT Token Count": token_count,
        "BERT Chunk Count": bert_chunk_count,
        "Wav2Vec2 Segment Count": wav2vec2_segment_count,
        "Audio Sequence Length": int(audio_features.shape[0]),
        "Text Sequence Length": int(text_features.shape[0]),
        "Prediction": prediction,
        "Probabilities": probabilities,
    }
    del transcript, text_features, audio_features
    return result
def warm_up(
    audio_path: Path,
    whisper_asr,
    wav2vec2_processor,
    wav2vec2_model,
    bert_tokenizer,
    bert_model,
    sist_model,
    device: torch.device,
) -> None:
    audio, sample_rate, _ = load_wav_audio(audio_path)
    for _ in range(WARMUP_RUNS):
        result = run_complete_pipeline(
            audio=audio,
            sample_rate=sample_rate,
            whisper_asr=whisper_asr,
            wav2vec2_processor=wav2vec2_processor,
            wav2vec2_model=wav2vec2_model,
            bert_tokenizer=bert_tokenizer,
            bert_model=bert_model,
            sist_model=sist_model,
            device=device,
        )
        del result
    torch.cuda.synchronize(device)
    del audio
def profile_recording(
    audio_path: Path,
    whisper_asr,
    wav2vec2_processor,
    wav2vec2_model,
    bert_tokenizer,
    bert_model,
    sist_model,
    device: torch.device,
) -> Dict[str, object]:
    # Raw WAV I/O and PCM decoding are intentionally excluded from timing.
    audio, sample_rate, audio_duration = load_wav_audio(audio_path)
    class_name, sample_name = get_class_and_sample(audio_path)
    target = CLASS_MAP[class_name]
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    pipeline_result = run_complete_pipeline(
        audio=audio,
        sample_rate=sample_rate,
        whisper_asr=whisper_asr,
        wav2vec2_processor=wav2vec2_processor,
        wav2vec2_model=wav2vec2_model,
        bert_tokenizer=bert_tokenizer,
        bert_model=bert_model,
        sist_model=sist_model,
        device=device,
    )
    torch.cuda.synchronize(device)
    inference_time = time.perf_counter() - start_time
    peak_gpu_memory_gb = torch.cuda.max_memory_allocated(device) / 1e9
    record = {
        "Class": class_name,
        "Sample": sample_name,
        "Audio Duration (s)": audio_duration,
        "Transcript Length": pipeline_result["Transcript Length"],
        "BERT Token Count": pipeline_result["BERT Token Count"],
        "BERT Chunk Count": pipeline_result["BERT Chunk Count"],
        "Wav2Vec2 Segment Count": pipeline_result["Wav2Vec2 Segment Count"],
        "Audio Sequence Length": pipeline_result["Audio Sequence Length"],
        "Text Sequence Length": pipeline_result["Text Sequence Length"],
        "Target": target,
        "Prediction": pipeline_result["Prediction"],
        "Inference Time (s)": inference_time,
        "Peak GPU Memory (GB)": peak_gpu_memory_gb,
        "RTF": inference_time / audio_duration,
    }
    del pipeline_result, audio
    return record

# ============================================================
# 12. Summary and CSV output
# ============================================================
def build_summary(
    total_params_m: float,
    trainable_params_m: float,
    records: List[Dict[str, object]],
) -> Dict[str, object]:
    total_time = sum(float(record["Inference Time (s)"]) for record in records)
    total_duration = sum(float(record["Audio Duration (s)"]) for record in records)
    if not records or total_duration <= 0:
        raise ValueError("Invalid complete-pipeline profiling records.")
    return {
        "Stage": "Complete pipeline",
        "Function": "Automatic recognition",
        "Parameters (M)": total_params_m,
        "Trainable Parameters (M)": trainable_params_m,
        "Inference Time (s/recording)": total_time / len(records),
        "Peak GPU Memory (GB)": max(
            float(record["Peak GPU Memory (GB)"])
            for record in records
        ),
        "RTF": total_time / total_duration,
    }
def save_records(records: List[Dict[str, object]]) -> None:
    fieldnames = [
        "Class",
        "Sample",
        "Audio Duration (s)",
        "Transcript Length",
        "BERT Token Count",
        "BERT Chunk Count",
        "Wav2Vec2 Segment Count",
        "Audio Sequence Length",
        "Text Sequence Length",
        "Target",
        "Prediction",
        "Inference Time (s)",
        "Peak GPU Memory (GB)",
        "RTF",
    ]
    with RECORD_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            for key in (
                "Audio Duration (s)",
                "Inference Time (s)",
                "Peak GPU Memory (GB)",
                "RTF",
            ):
                row[key] = f"{float(row[key]):.6f}"
            writer.writerow(row)
def save_summary(summary: Dict[str, object]) -> None:
    fieldnames = [
        "Stage",
        "Function",
        "Parameters (M)",
        "Trainable Parameters (M)",
        "Inference Time (s/recording)",
        "Peak GPU Memory (GB)",
        "RTF",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "Stage": summary["Stage"],
            "Function": summary["Function"],
            "Parameters (M)": f'{float(summary["Parameters (M)"]):.6f}',
            "Trainable Parameters (M)": (
                f'{float(summary["Trainable Parameters (M)"]):.6f}'
            ),
            "Inference Time (s/recording)": (
                f'{float(summary["Inference Time (s/recording)"]):.6f}'
            ),
            "Peak GPU Memory (GB)": (
                f'{float(summary["Peak GPU Memory (GB)"]):.6f}'
            ),
            "RTF": f'{float(summary["RTF"]):.6f}',
        })
def print_summary(
    summary: Dict[str, object],
    stage_params: Dict[str, float],
    recording_count: int,
    seed: int,
    gpu_name: str,
) -> None:
    print("\n" + "=" * 76)
    print("Complete pipeline computational profile")
    print("=" * 76)
    print(f"GPU: {gpu_name}")
    print(f"SIST checkpoint seed: {seed}")
    print(f"Recordings: {recording_count}")
    print("Stage Parameters (M):")
    for name, value in stage_params.items():
        print(f"  {name:<18}: {value:.6f}")
    print(f'Parameters (M): {float(summary["Parameters (M)"]):.6f}')
    print(
        "Trainable Parameters (M): "
        f'{float(summary["Trainable Parameters (M)"]):.6f}'
    )
    print(
        "Inference Time (s/recording): "
        f'{float(summary["Inference Time (s/recording)"]):.6f}'
    )
    print(
        "Peak GPU Memory (GB): "
        f'{float(summary["Peak GPU Memory (GB)"]):.6f}'
    )
    print(f'RTF: {float(summary["RTF"]):.6f}')
    print("=" * 76)

# ============================================================
# 13. Main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct end-to-end profiling of the complete automatic pipeline."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        choices=SUPPORTED_SEEDS,
        help=(
            "SIST checkpoint seed. Computational structure is identical across "
            "seeds; default=2024."
        ),
    )
    args = parser.parse_args()
    audio_files = validate_paths(args.seed)
    device = get_cuda_device()
    gpu_name = torch.cuda.get_device_name(device)
    print(f"[INFO] GPU: {gpu_name}")
    print(f"[INFO] Audio dataset: {AUDIO_ROOT / TEST_SPLIT}")
    print(f"[INFO] Text dataset: {TEXT_ROOT / TEST_SPLIT}")
    print(f"[INFO] Test recordings: {len(audio_files)}")
    print(f"[INFO] Whisper: {WHISPER_DIR}")
    print(f"[INFO] Wav2Vec 2.0: {WAV2VEC2_DIR}")
    print(f"[INFO] BERT: {BERT_DIR}")
    print(f"[INFO] SIST checkpoint: {get_sist_checkpoint(args.seed)}")
    print("[INFO] Stored text is used only for sample-pair validation.")
    print("[INFO] Timed input is raw in-memory waveform.")
    print("[INFO] Raw WAV file I/O/PCM decode are excluded from latency.")
    transformers_logging.set_verbosity_error()
    print("[INFO] Loading Whisper-large-v3...")
    whisper_model, whisper_asr = load_whisper_resources(device)
    print("[INFO] Loading Wav2Vec 2.0...")
    wav2vec2_processor, wav2vec2_model = load_wav2vec2_resources(device)
    print("[INFO] Loading BERT...")
    bert_tokenizer, bert_model = load_bert_resources(device)
    audio_dim = int(wav2vec2_model.config.hidden_size)
    text_dim = int(bert_model.config.hidden_size)
    print("[INFO] Loading SIST...")
    sist_model = load_sist_model(
        seed=args.seed,
        audio_dim=audio_dim,
        text_dim=text_dim,
        device=device,
    )
    total_params_m, trainable_params_m, stage_params = build_parameter_summary(
        whisper_model=whisper_model,
        wav2vec2_model=wav2vec2_model,
        bert_model=bert_model,
        sist_model=sist_model,
    )
    # Frozen front ends must have zero trainable parameters.
    for name, model in (
        ("Whisper-large-v3", whisper_model),
        ("Wav2Vec 2.0", wav2vec2_model),
        ("BERT", bert_model),
    ):
        _, trainable = count_model_parameters(model)
        if trainable != 0:
            raise RuntimeError(f"{name} unexpectedly has trainable parameters.")
    _, sist_trainable = count_model_parameters(sist_model)
    if trainable_params_m != sist_trainable / 1e6:
        raise RuntimeError(
            "Complete-pipeline trainable parameters must equal SIST trainable parameters."
        )
    print(f"[INFO] Warm-up runs: {WARMUP_RUNS}")
    warm_up(
        audio_path=audio_files[0],
        whisper_asr=whisper_asr,
        wav2vec2_processor=wav2vec2_processor,
        wav2vec2_model=wav2vec2_model,
        bert_tokenizer=bert_tokenizer,
        bert_model=bert_model,
        sist_model=sist_model,
        device=device,
    )
    records: List[Dict[str, object]] = []
    for audio_path in tqdm(audio_files, desc="Profiling complete pipeline"):
        record = profile_recording(
            audio_path=audio_path,
            whisper_asr=whisper_asr,
            wav2vec2_processor=wav2vec2_processor,
            wav2vec2_model=wav2vec2_model,
            bert_tokenizer=bert_tokenizer,
            bert_model=bert_model,
            sist_model=sist_model,
            device=device,
        )
        records.append(record)
    summary = build_summary(
        total_params_m=total_params_m,
        trainable_params_m=trainable_params_m,
        records=records,
    )
    save_records(records)
    save_summary(summary)
    print_summary(
        summary=summary,
        stage_params=stage_params,
        recording_count=len(records),
        seed=args.seed,
        gpu_name=gpu_name,
    )
    print(f"[INFO] Summary CSV: {SUMMARY_CSV.resolve()}")
    print(f"[INFO] Record CSV: {RECORD_CSV.resolve()}")

if __name__ == "__main__":
    main()
