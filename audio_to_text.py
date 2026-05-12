# audio_to_text.py

import os
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Tuple
import re
import opencc
import whisper
import torch
from tqdm import tqdm

# ================= 配置 =================
MODEL_CACHE_DIR = r"D:\pretrain\whisper_models\whisper-largev3"
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".wma")

# 停顿阈值（秒）
PAUSE_T_SHORT = 0.5
PAUSE_T_MEDIUM = 1.0
PAUSE_T_LONG = 1.5

# —— 关键开关：无明显停顿时是否也“软性插入”标点（避免整段无标点）——
INSERT_FOR_NONE = True

# —— 软性插入的跨度阈值（连续多少字符/英文更大），超过就强插标点 ——
SEP_LIMIT = {"zh": 18, "en": 30}  # 超过插入分隔标点（， / ,）
TERM_LIMIT = {"zh": 40, "en": 80}  # 超过插入句末标点（。 / .）

# 停顿映射（按语言覆盖标点，新增default兜底）
PAUSE_PUNCT = {
    "zh": {"NONE": "", "SHORT": "，", "MEDIUM": "。", "LONG": "……"},
    "en": {"NONE": "", "SHORT": ",", "MEDIUM": ".", "LONG": "..."},
    "default": {"NONE": "", "SHORT": ",", "MEDIUM": ".", "LONG": "..."},
}

# 全局初始化繁转简实例（避免重复创建）
_CN_CONVERTER = opencc.OpenCC('t2s.json')


# ================ 基本IO ================
def load_whisper_model(model_name: str = "large-v3",
                       cache_dir: str = MODEL_CACHE_DIR,
                       device: Optional[str] = None):
    # 确保模型缓存目录存在
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading whisper model: {model_name} on {device}")
    model = whisper.load_model(model_name, device=device, download_root=str(cache_path))
    return model


def is_audio_file(p: Path) -> bool:
    return p.suffix.lower() in AUDIO_EXTS


def enumerate_audio_files(dataset_root: Path,
                          splits: Optional[Iterable[str]] = None) -> List[Path]:
    files: List[Path] = []
    if splits:
        for sp in splits:
            sub = dataset_root / sp
            if sub.is_dir():
                files.extend([p for p in sub.rglob("*") if p.is_file() and is_audio_file(p)])
    else:
        files.extend([p for p in dataset_root.rglob("*") if p.is_file() and is_audio_file(p)])
    return files


def derive_out_path(audio_path: Path, dataset_root: Path, output_root: Path) -> Path:
    rel = audio_path.relative_to(dataset_root)
    return (output_root / rel).with_suffix(".txt")


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def transcribe_one(audio_path: Path, model, language: Optional[str] = None, **kwargs) -> Dict:
    result = model.transcribe(
        str(audio_path),
        task="transcribe",
        language=language,
        **kwargs
    )
    return result


# ============== 文本规范化工具 ==============
_CN_ASCII_TO_FULL = str.maketrans({
    ',': '，', '.': '。', '?': '？', '!': '！', ':': '：', ';': '；'
})


def normalize_ellipsis_zh(s: str) -> str:
    s = re.sub(r'(\.{3,})', '……', s)  # ... -> ……
    s = re.sub(r'…{2,}', '……', s)  # 连续 … -> ……
    return s


def normalize_zh_defaults(s: str) -> str:
    s = normalize_ellipsis_zh(s.translate(_CN_ASCII_TO_FULL))
    s = re.sub(r'\s*([，。！？；：、（）《》“”‘’……])\s*', r'\1', s)
    s = re.sub(r'([，。！？；：、（《”’)])\1+', r'\1', s)
    s = re.sub(r'(……)+', '……', s)
    s = re.sub(r'^[，。！？；：、……]+', '', s)  # 移除句首多余标点
    return s.strip()


def normalize_en_defaults(s: str) -> str:
    s = re.sub(r'(\.{3,}|…{2,})', '...', s)
    s = re.sub(r'\s*([,\.!?;:])\s*', r'\1 ', s)  # 标点后留一空格
    s = re.sub(r'\s{2,}', ' ', s).strip()
    s = re.sub(r'([,\.!?;:])\1+', r'\1', s)
    s = re.sub(r'(\.\.\.){2,}', '...', s)
    s = re.sub(r'^[,\.!?;:]+', '', s).strip()  # 移除句首多余标点
    return s


def convert_to_simplified_chinese(text: str) -> str:
    return _CN_CONVERTER.convert(text)


# ============== 停顿驱动的“覆盖标点” ==============
_RE_TAIL_PUNCT_ZH = re.compile(r'[，。！？；：、…]+$')
_RE_TAIL_PUNCT_EN = re.compile(r'[,\.\?!;:]+$')


def split_tail_punct(text: str, language: str) -> Tuple[str, str]:
    t = text.rstrip()
    if not t:
        return "", ""
    m = (_RE_TAIL_PUNCT_ZH if language == "zh" else _RE_TAIL_PUNCT_EN).search(t)
    return (t[:m.start()], t[m.start():]) if m else (t, "")


def pause_bucket(dt: float) -> str:
    if dt >= PAUSE_T_LONG:   return "LONG"
    if dt >= PAUSE_T_MEDIUM: return "MEDIUM"
    if dt >= PAUSE_T_SHORT:  return "SHORT"
    return "NONE"


def pick_pause_punct(language: str, bucket: str) -> str:
    # 优先匹配语言，无则用default兜底
    lang_key = language if language in PAUSE_PUNCT else "default"
    return PAUSE_PUNCT[lang_key].get(bucket, "")


def is_terminal(language: str, punct: str) -> bool:
    if not punct: return False
    return ('……' in punct or any(p in punct for p in '。！？')) if language == "zh" \
        else ('...' in punct or any(p in punct for p in '.?!'))


def is_separator(language: str, punct: str) -> bool:
    if not punct: return False
    return any(p in punct for p in ('，；：、' if language == "zh" else ',;:'))


def needs_space_en(prev_text: str, next_text: str) -> bool:
    if not prev_text or not next_text: return False
    prev_last = prev_text[-1] if prev_text else ''
    nxt_first = next((c for c in next_text if not c.isspace()), '')
    return prev_last.isalnum() and nxt_first.isalnum()


# ============== 主逻辑 ==============
def stitch_with_pause_override(result: Dict, language: str = "en") -> str:
    """
    1) 片段内部做语言规范（温和、不主动造句）。
    2) 段与段边界：按停顿档位覆盖末尾标点（替换已有标点或插入新标点）。
    3) 若连续太长都没有分隔/句末标点，NONE 边界也软性插入（避免整段无标点）。
    4) 末段做句末兜底（长文本无标点时补，短文本不强行补）。
    5) 全局再清理一次。
    """
    segments = result.get("segments", [])
    if not segments:
        return ""

    # 片段内部语言规范
    normed: List[str] = []
    for seg in segments:
        t = seg.get("text", "").strip()
        if language == "zh":
            t = convert_to_simplified_chinese(normalize_zh_defaults(t))
        else:
            t = normalize_en_defaults(t)
        normed.append(t)

    out_parts: List[str] = []
    # 软性插入所需的“自上次句末/分隔”的字符计数
    since_term = 0
    since_sep = 0

    for i in range(len(segments) - 1):
        cur_text = normed[i]
        nxt_text = normed[i + 1]
        cur_end = segments[i].get("end", None)
        nxt_start = segments[i + 1].get("start", None)
        dt = 0.0 if (cur_end is None or nxt_start is None) else max(0.0, nxt_start - cur_end)

        base, tail = split_tail_punct(cur_text, language)
        bucket = pause_bucket(dt)

        # 优先：有停顿 -> 覆盖末尾标点为停顿标点
        use_tail = ""
        if bucket != "NONE":
            use_tail = pick_pause_punct(language, bucket)
        else:
            # 无停顿：默认保留原有末尾标点
            use_tail = tail
            # 软性兜底：若没有任何标点且跨度过长，则插入分隔/句末标点
            if INSERT_FOR_NONE and not use_tail:
                # 严格按语言取阈值，避免跨语言阈值混用
                lang_sep_limit = SEP_LIMIT.get(language, SEP_LIMIT["en"])
                lang_term_limit = TERM_LIMIT.get(language, TERM_LIMIT["en"])
                if since_term >= lang_term_limit:
                    use_tail = pick_pause_punct(language, "MEDIUM")  # 句号/.
                elif since_sep >= lang_sep_limit:
                    use_tail = pick_pause_punct(language, "SHORT")  # 逗号/,

        new_cur = base + use_tail

        # 英文跨段无标点且两端都是字母数字时，补空格防止词黏连
        if language == "en" and not use_tail and needs_space_en(new_cur, nxt_text):
            new_cur += " "

        out_parts.append(new_cur)

        # 更新计数器（按最终使用的标点）
        eff_tail = use_tail if use_tail else tail
        add_len = len(base)
        if is_terminal(language, eff_tail):
            since_term = 0
            since_sep = 0
        elif is_separator(language, eff_tail):
            since_sep = 0
            since_term += add_len
        else:
            since_sep += add_len
            since_term += add_len

    # 末段：仅长文本无句末标点时兜底（中文>10字符，英文>20字符）
    last_text = normed[-1]
    base_last, tail_last = split_tail_punct(last_text, language)
    need_terminal = len(base_last.strip()) > (10 if language == "zh" else 20)
    if need_terminal and not is_terminal(language, tail_last):
        tail_last = pick_pause_punct(language, "MEDIUM")
    out_parts.append(base_last + tail_last)

    joined = "".join(out_parts)

    # 全局收尾清理（二次去重，确保标点纯净）
    if language == "zh":
        joined = normalize_zh_defaults(joined)
        joined = re.sub(r'([，。！？；：、（《”’)])\1+', r'\1', joined)
        joined = re.sub(r'(……)+', '……', joined)
    else:
        joined = normalize_en_defaults(joined)
        joined = re.sub(r'([,\.!?;:])\1+', r'\1', joined)
        joined = re.sub(r'(\.\.\.){2,}', '...', joined)

    return joined.strip()


# ============== 保存 & 批处理 ==============
def save_text(text: str, out_txt: Path):
    ensure_parent_dir(out_txt)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)


def batch_transcribe_tree(dataset_root: str,
                          output_root: str,
                          model,
                          splits: Optional[Iterable[str]] = ("train", "val", "test"),
                          language: Optional[str] = None,
                          overwrite: bool = False,
                          whisper_kwargs: Optional[dict] = None):
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    whisper_kwargs = whisper_kwargs or {}

    audio_files = enumerate_audio_files(dataset_root, splits)
    if not audio_files:
        print(f"[WARNING] Audio not found: {dataset_root}")
        return

    print(f"[INFO] 将处理 {len(audio_files)} 个音频文件")
    skipped, ok, fail = 0, 0, 0

    for ap in tqdm(audio_files, desc="Transcribing"):
        out_txt = derive_out_path(ap, dataset_root, output_root)
        if out_txt.exists() and not overwrite:
            skipped += 1
            continue

        try:
            res = transcribe_one(ap, model, language=language, **whisper_kwargs)
            # 关键：语言自动匹配（用户指定优先，否则用Whisper检测结果）
            used_language = language or res.get("language", "en")
            # 中文相关语言统一为"zh"（适配标点规则）
            used_language = "zh" if used_language.startswith("zh") else used_language
            final_text = stitch_with_pause_override(res, used_language)
            save_text(final_text, out_txt)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[ERROR] 失败: {ap}\n{e}")

    print(f"[DONE] 成功: {ok} | 跳过(已存在): {skipped} | 失败: {fail}")
    print(f"[INFO] 文本输出根目录: {output_root.resolve()}")


# ================== 入口 ==================
def main():
    # 示例配置（可根据需求修改）
    # 英文数据集示例
    dataset_root = r"D:\Datasets\ADReSSo2021"
    output_root = r"D:\Datasets\ADReSSo2021_text"
    # 中文数据集示例（取消注释切换）
    # dataset_root = r"D:\Datasets\NCMMSC2021"
    # output_root = r"D:\Datasets\NCMMSC2021_text"

    model_name = "large-v3"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    language = "en"  # 中文数据集改为 "zh"，不指定则自动检测

    # Whisper 核心参数（确保转录质量和速度）
    whisper_kwargs = dict(
        fp16=True if torch.cuda.is_available() else False,  # GPU加速（CPU自动禁用）
        condition_on_previous_text=None,  # 禁用上下文依赖，避免累积错误
        temperature=0.0,  # 确定性输出，保证重复转录结果一致
        best_of=1,  # 减少候选生成，提升速度（large-v3精度足够）
    )

    # 加载模型
    model = load_whisper_model(model_name=model_name, cache_dir=MODEL_CACHE_DIR, device=device)

    # 批量转录
    batch_transcribe_tree(
        dataset_root=dataset_root,
        output_root=output_root,
        model=model,
        splits=("train", "val", "test"),  # 要处理的子目录（无则设为None）
        language=language,  # 明确指定语言可提升转录精度（推荐）
        overwrite=False,  # 是否覆盖已存在的文本文件
        whisper_kwargs=whisper_kwargs
    )


if __name__ == "__main__":
    main()