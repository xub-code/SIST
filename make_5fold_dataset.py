import sys
import shutil
import random
from pathlib import Path
from typing import Dict, List

INPUT_DIR = Path(r"D:\数据集\Cross-validation\NCMMSC2021")
OUTPUT_DIR = Path(r"D:\数据集\Cross-validation\5CV-NCMMSC2021")
SEED = 42

# CLASSES = ["cc", "cd"]
CLASSES = ["AD", "HC", "MCI"]
PARTS = ["A", "B", "C", "D", "E"]
FOLD_PLANS = [
    (["A","B","C","D"], "E"),  # fold1
    (["A","B","C","E"], "D"),  # fold2
    (["A","B","D","E"], "C"),  # fold3
    (["A","C","D","E"], "B"),  # fold4
    (["B","C","D","E"], "A"),  # fold5
]
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_files_by_class(root: Path) -> Dict[str, List[Path]]:
    if not root.exists():
        raise RuntimeError(f"[ERROR] 输入目录不存在：{root.resolve()}")
    files_by_class: Dict[str, List[Path]] = {}
    for cls in CLASSES:
        cdir = root / cls
        if not cdir.exists():
            raise RuntimeError(f"[ERROR] 缺少类别目录：{cdir}")
        cur = [f for f in cdir.rglob("*") if f.is_file() and f.suffix.lower() in AUDIO_EXTS]
        if not cur:
            print(f"[WARN] 类别 {cls} 下未找到音频文件")
        files_by_class[cls] = sorted(cur)
    return files_by_class

def globally_balanced_5parts(files_by_class: Dict[str, List[Path]], seed: int
                             ) -> Dict[str, Dict[str, List[Path]]]:
    """
    返回 per_class_parts[cls][part] = 该类别分到该 part 的文件列表。
    思路：先对每类随机，再用“全局均衡分配余数”的贪心，把各类的 (n%5) 个‘+1’优先给
    当前总量最小的 part，保证 A..E 的总量差≤1。
    """
    rng = random.Random(seed)

    # 预处理：每类随机 & 计算 base / extra
    per_class_base: Dict[str, int] = {}
    per_class_extra: Dict[str, int] = {}
    per_class_lists: Dict[str, List[Path]] = {}
    for cls, flist in files_by_class.items():
        arr = flist[:]
        rng.shuffle(arr)
        per_class_lists[cls] = arr
        n = len(arr)
        per_class_base[cls] = n // 5
        per_class_extra[cls] = n % 5

    # 初始化每类每 part 的目标计数：先放 base
    per_class_target: Dict[str, Dict[str, int]] = {cls: {p: per_class_base[cls] for p in PARTS}
                                                   for cls in CLASSES}
    # 当前各 part 的总计数（只算目标，不是文件）
    total_per_part: Dict[str, int] = {p: sum(per_class_base.values()) for p in PARTS}

    # 把各类的 extra（+1）按全局最少的 part 分配，尽量平衡 A..E 总量
    # 类别顺序固定即可；也可打乱 CLASSES 获得不同但等价的分配
    for cls in CLASSES:
        for _ in range(per_class_extra[cls]):
            # 选择当前总计数最小的 part；并用 PARTS 顺序打破平局
            tgt_part = min(PARTS, key=lambda pp: (total_per_part[pp], PARTS.index(pp)))
            per_class_target[cls][tgt_part] += 1
            total_per_part[tgt_part] += 1

    # 根据 per_class_target 从随机后的列表中切片
    per_class_parts: Dict[str, Dict[str, List[Path]]] = {cls: {p: [] for p in PARTS} for cls in CLASSES}
    for cls in CLASSES:
        arr = per_class_lists[cls]
        idx = 0
        for p in PARTS:
            sz = per_class_target[cls][p]
            per_class_parts[cls][p] = arr[idx: idx+sz]
            idx += sz
        assert idx == len(arr), f"{cls} 切片数量不匹配"

    # 打印分配统计
    print("[INFO] 每类分配（A..E）：")
    for cls in CLASSES:
        counts = [len(per_class_parts[cls][p]) for p in PARTS]
        msg = ", ".join(f"{PARTS[i]}:{counts[i]}" for i in range(5))
        print(f"  {cls}: {msg}")
    tot = {p: sum(len(per_class_parts[c][p]) for c in CLASSES) for p in PARTS}
    print("[INFO] A..E 总量：", tot, "（应当只相差≤1）")
    return per_class_parts

def copy_with_dedup(src: Path, dst: Path):
    safe_mkdir(dst.parent)
    if not dst.exists():
        shutil.copy2(src, dst)
        return
    # 若同名文件已存在（不同子目录来的），加后缀避免覆盖
    stem, suf = dst.stem, dst.suffix
    k = 1
    while True:
        alt = dst.with_name(f"{stem}__dup{k}{suf}")
        if not alt.exists():
            shutil.copy2(src, alt)
            return
        k += 1

def build_folds(per_class_parts: Dict[str, Dict[str, List[Path]]]):
    # 清空输出目录
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    safe_mkdir(OUTPUT_DIR)

    # 5 个折
    for fold_idx, (train_parts, val_part) in enumerate(FOLD_PLANS, start=1):
        fold_dir = OUTPUT_DIR / f"fold{fold_idx}"
        train_root = fold_dir / "train"
        val_root   = fold_dir / "val"
        print(f"\n[INFO] 生成 fold{fold_idx}: train={'+'.join(train_parts)}  val={val_part}")

        train_count: Dict[str, int] = {}
        val_count: Dict[str, int] = {}

        for cls in CLASSES:
            train_files: List[Path] = []
            for p in train_parts:
                train_files.extend(per_class_parts[cls][p])
            val_files = per_class_parts[cls][val_part]

            for f in train_files:
                copy_with_dedup(f, train_root / cls / f.name)
            for f in val_files:
                copy_with_dedup(f, val_root / cls / f.name)

            train_count[cls] = len(train_files)
            val_count[cls] = len(val_files)

        print(f"[INFO] fold{fold_idx}/train 按类计数：{train_count} | 总计={sum(train_count.values())}")
        print(f"[INFO] fold{fold_idx}/val   按类计数：{val_count} | 总计={sum(val_count.values())}")

        # 泄漏自检（同一 fold 内，train 与 val 不得同类同名）
        train_names = set()
        val_names = set()
        for cls in CLASSES:
            for p in (train_root/cls).glob("*"):
                train_names.add((cls, p.name))
            for p in (val_root/cls).glob("*"):
                val_names.add((cls, p.name))
        leak = train_names & val_names
        if leak:
            raise RuntimeError(f"[ERROR] fold{fold_idx} 检测到泄漏：{list(leak)[:5]}")

def main():
    try:
        print(f"[INFO] 读取：{INPUT_DIR.resolve()}")
        files_by_class = list_files_by_class(INPUT_DIR)
        per_class_parts = globally_balanced_5parts(files_by_class, seed=SEED)
        build_folds(per_class_parts)
        # 双击运行时暂停
        if sys.stdin is None or not sys.stdin.isatty():
            input("\n完成，按回车退出...")
    except Exception as e:
        print(str(e))
        if sys.stdin is None or not sys.stdin.isatty():
            input("\n发生错误，按回车退出...")
        sys.exit(1)

if __name__ == "__main__":
    main()
