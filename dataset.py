import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset


class MultimodalDataset(Dataset):
    """多模态数据集（修复版：强制排序以保证复现性）"""

    def __init__(self, root_dir, class_map):
        self.samples = []
        self.class_map = class_map
        root_dir = Path(root_dir)
        print(f"[INFO] 正在加载数据集: {root_dir}")

        # 【关键修正1】强制对类别键进行排序，保证遍历顺序一致
        for cls_name in sorted(class_map.keys()):
            cls_idx = class_map[cls_name]
            cls_dir = root_dir / cls_name
            if not cls_dir.exists():
                print(f"[WARN] 类别目录不存在: {cls_dir}")
                continue

            # 【关键修正2】使用 sorted() 确保 glob 返回的文件列表有序
            audio_files = {}
            for f in sorted(cls_dir.glob("audio_*.npy")):
                if f.stem.startswith("audio_"):
                    raw_name = f.stem[len("audio_"):]
                    audio_files[raw_name] = f

            text_files = {}
            for f in sorted(cls_dir.glob("text_*.npy")):
                if f.stem.startswith("text_"):
                    raw_name = f.stem[len("text_"):]
                    text_files[raw_name] = f

            # 寻找交集
            common_raw_names = set(audio_files.keys()) & set(text_files.keys())

            if not common_raw_names:
                print(f"[WARN] 类别 {cls_name} 没有找到成对样本")
                continue
            for raw_name in sorted(list(common_raw_names)):
                self.samples.append((
                    str(audio_files[raw_name]),
                    str(text_files[raw_name]),
                    cls_idx
                ))
            print(f"[INFO] 类别 {cls_name} 匹配到 {len(common_raw_names)} 对样本")

        if not self.samples:
            raise RuntimeError(f"在 {root_dir} 中未找到任何样本！")

        print(f"[INFO] 总匹配到 {len(self.samples)} 对多模态样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, text_path, label = self.samples[idx]
        try:
            audio_emb = np.load(audio_path).astype(np.float32)
            text_emb = np.load(text_path).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"加载文件失败: {audio_path}\n错误: {e}")

        return (
            torch.from_numpy(audio_emb),
            torch.from_numpy(text_emb),
            torch.tensor(label, dtype=torch.long)
        )


def collate_fn(batch):
    audio_list, text_list, labels = zip(*batch)
    # 处理音频
    audio_lengths = [x.shape[0] for x in audio_list]
    audio_max_len = max(audio_lengths) if audio_lengths else 0
    audio_dim = audio_list[0].shape[1] if audio_list else 0
    audio_x = torch.zeros(len(batch), audio_max_len, audio_dim, dtype=torch.float32)
    audio_mask = torch.zeros(len(batch), audio_max_len, dtype=torch.bool)
    for i, x in enumerate(audio_list):
        l = x.shape[0]
        audio_x[i, :l] = x
        audio_mask[i, :l] = 1
    # 处理文本
    text_lengths = [x.shape[0] for x in text_list]
    text_max_len = max(text_lengths) if text_lengths else 0
    text_dim = text_list[0].shape[1] if text_list else 0
    text_x = torch.zeros(len(batch), text_max_len, text_dim, dtype=torch.float32)
    text_mask = torch.zeros(len(batch), text_max_len, dtype=torch.bool)
    for i, x in enumerate(text_list):
        l = x.shape[0]
        text_x[i, :l] = x
        text_mask[i, :l] = 1
    return audio_x, audio_mask, text_x, text_mask, torch.stack(labels)