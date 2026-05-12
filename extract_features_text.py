# extract_features_text.py
import os
import argparse
import re
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import warnings



warnings.filterwarnings("ignore")


MAX_LEN = 256  # 每块最大token数
STRIDE = 64  # 相邻块重叠token数
TOKENIZER_MAX_LEN = 10000  # 避免超长警告的上限


def normalize_text(s: str) -> str:
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# 文本分块
def chunk_text(input_ids, attention_mask, max_len, stride):
    """
    文本滑窗分块input_ids/attention_mask: [1, T]
    """
    T = input_ids.shape[1]
    if T <= max_len:
        return [(input_ids, attention_mask)]

    chunks = []
    start = 0
    while start < T:
        end = min(start + max_len, T)
        chunks.append((
            input_ids[:, start:end],
            attention_mask[:, start:end]
        ))
        if end == T:
            break
        # 重叠计算
        start = max(0, end - stride)
    return chunks

# import re
# def clean_text(text):
#     text=re.sub(r'[^\w\s]','',text) # 去除标点符号
#     text=text.strip() # 去除前后空格
#     return text

# 层聚合策略
def combine_hidden_layers(hidden_states, strategy="last4avg"):
    """
    聚合多层Transformer隐藏层为单一层特征
    """
    encoder_layers = hidden_states[1:]
    L = len(encoder_layers)
    if L == 0:
        return hidden_states[-1]

    if strategy == "last":
        return encoder_layers[-1]
    elif strategy == "last4avg":
        idxs = [max(L - 1 - i, 0) for i in range(4)]
        stack = torch.stack([encoder_layers[i] for i in idxs], dim=0)
        return stack.mean(dim=0)
    elif strategy == "last4weighted":
        idxs = [max(L - 1 - i, 0) for i in range(4)]
        ws = torch.tensor([0.4, 0.3, 0.2, 0.1],
                          device=encoder_layers[-1].device,
                          dtype=encoder_layers[-1].dtype).view(4, 1, 1, 1)
        stack = torch.stack([encoder_layers[i] for i in idxs], dim=0)
        return (stack * ws).sum(dim=0)
    elif strategy == "upperhalf":
        start = L // 2
        stack = torch.stack(encoder_layers[start:], dim=0)
        return stack.mean(dim=0)
    elif strategy == "allweighted":
        w = torch.linspace(1.0, 2.0, steps=L,
                           device=encoder_layers[-1].device,
                           dtype=encoder_layers[-1].dtype)
        w = (w / w.sum()).view(L, 1, 1, 1)
        stack = torch.stack(encoder_layers, dim=0)
        return (stack * w).sum(dim=0)
    elif strategy == "concat_last4":
        idxs = [max(L - 1 - i, 0) for i in range(4)]
        return torch.cat([encoder_layers[i] for i in idxs], dim=-1)
    else:
        raise ValueError(f"Unknown layer combine strategy: {strategy}")


# 文本特征提取主函数
def extract_text_embedding(tokenizer, model, device, txt_path,
                           layer_strategy="last4avg", pool="seg_stat",
                           max_len=MAX_LEN, stride=STRIDE):
    """
    提取文本分块特征
    """
    txt_path = Path(txt_path)
    # 加载文本
    try:
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text = txt_path.read_text(errors="ignore")

    # text = clean_text(text)  # 在这里调用 clean_text 进行清理
    text = normalize_text(text)

    # 空文本处理
    if not text:
        hid = model.config.hidden_size
        d = hid if layer_strategy != "concat_last4" else hid * 4
        if pool == "seg_mean":
            return np.zeros((1, d), dtype=np.float32)
        elif pool == "seg_stat":
            return np.zeros((1, d * 2), dtype=np.float32)
        elif pool == "seg_quantile":
            return np.zeros((1, d * 3), dtype=np.float32)

    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
        max_length=TOKENIZER_MAX_LEN
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    chunks = chunk_text(input_ids, attention_mask, max_len=max_len, stride=stride)
    seg_embs = []

    for (ids_chunk, mask_chunk) in chunks:
        with torch.no_grad():
            outputs = model(
                input_ids=ids_chunk,
                attention_mask=mask_chunk,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states

        mixed_layer = combine_hidden_layers(hidden_states, strategy=layer_strategy)
        hidden = mixed_layer.squeeze(0)
        input_len = int(mask_chunk.sum().item())
        valid_hidden = hidden[:input_len, :]

        if pool == "seg_mean":
            vec = valid_hidden.mean(dim=0)
        elif pool == "seg_stat":
            mu = valid_hidden.mean(dim=0)
            std = torch.sqrt(valid_hidden.var(dim=0, unbiased=False) + 1e-6)
            vec = torch.cat([mu, std], dim=0)
        elif pool == "seg_quantile":
            p25 = torch.quantile(valid_hidden, 0.25, dim=0)
            med = torch.quantile(valid_hidden, 0.50, dim=0)
            p75 = torch.quantile(valid_hidden, 0.75, dim=0)
            vec = torch.cat([p25, med, p75], dim=0)
        else:
            raise ValueError(f"Unknown pool strategy: {pool}")

        seg_embs.append(vec.cpu().numpy().astype(np.float32))

    return np.vstack(seg_embs).astype(np.float32)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    global MAX_LEN, STRIDE
    MAX_LEN = args.max_len
    STRIDE = args.stride

    print(f"[INFO] 加载文本模型: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              local_files_only=args.local_only)
    model = AutoModel.from_pretrained(args.model_path,
                                      local_files_only=args.local_only)
    model.to(device).eval()

    splits = ["train", "val", "test"]
    for split_name in splits:
        split_dir = os.path.join(args.data_dir, split_name)
        if not os.path.exists(split_dir):
            print(f"[WARN] 跳过不存在的目录: {split_dir}")
            continue

        for class_name in sorted(os.listdir(split_dir)):
            class_folder = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_folder):
                continue

            out_class_dir = os.path.join(args.cache_dir, split_name, class_name)
            os.makedirs(out_class_dir, exist_ok=True)

            txt_files = [f for f in os.listdir(class_folder) if f.lower().endswith(".txt")]
            for f in tqdm(txt_files, desc=f"{split_name}/{class_name} "
                                          f"[{args.layer_strategy}/{args.pool}]"):
                txt_path = os.path.join(class_folder, f)
                out_filename = f"text_{Path(f).stem}.npy"
                out_path = os.path.join(out_class_dir, out_filename)

                if os.path.exists(out_path) and not args.force:
                    continue

                try:
                    emb = extract_text_embedding(
                        tokenizer, model, device, txt_path,
                        layer_strategy=args.layer_strategy,
                        pool=args.pool,
                        max_len=args.max_len,
                        stride=args.stride
                    )
                    np.save(out_path, emb)
                except Exception as e:
                    print(f"[ERROR] 处理文件失败: {txt_path}，错误: {e}")

    print("[INFO] 文本特征提取完成")

# CLI参数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 路径参数（与音频完全一致）
    # D:\Datasets\Text\NCMMSC2021_text_xunfei
    # "D:\Datasets\Text\ADReSS2020_text"
    # "D:\Datasets\Text\ADReSSo2021_text"
    parser.add_argument("--data_dir", type=str, default=r"D:\Datasets\Text\ADReSSo2021_text",
                        help="数据根目录（含train/val/test，每个类别下为.txt文件）")
    # "D:\pretrain\bert-base-chinese"
    # "D:\pretrain\bert-base-uncased"
    parser.add_argument("--model_path", type=str, default=r"D:\pretrain\bert-base-uncased",# 注意中英文不一样 bert-base-uncased
                        help="文本模型路径")
    parser.add_argument("--cache_dir", type=str, default=r"ADReSSo2021",
                        help="特征保存目录")
    parser.add_argument("--force", action="store_true",
                        help="覆盖已有特征文件")
    parser.add_argument("--local_only", action="store_true",
                        help="仅加载本地模型")

    # 特征提取策略
    parser.add_argument("--layer_strategy", type=str, default="last4avg",
                        choices=["last", "last4avg", "last4weighted", "upperhalf",
                                 "allweighted", "concat_last4"],
                        help="层聚合策略")
    parser.add_argument("--pool", type=str, default="seg_mean",
                        choices=["seg_mean", "seg_stat", "seg_quantile"],
                        help="块内池化方式")

    # 文本分块参数
    parser.add_argument("--max_len", type=int, default=256,
                        help="每块最大token数")
    parser.add_argument("--stride", type=int, default=64,
                        help="块重叠token数")

    args = parser.parse_args()
    main(args)



