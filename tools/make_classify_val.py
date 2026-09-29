"""从 train 里分层切出一个真正的 val/，供分类训练使用。

背景：classification/Covid19-dataset 只有 train/ 与 test/ 两级目录，而 ultralytics
的 ClassificationTrainer 在找不到 val/ 时会回退拿 test/ 当验证集，训练日志里会打印
"WARNING Dataset 'split=val' not found, using 'split=test' instead."。这样一来
best.pt 是按测试集精度挑出来的——最后报出去的 test 准确率偏乐观，做消融对比时
站不住脚。

本脚本按类别分层，从 train/ 里**移动**（而不是复制）ratio 比例（默认 20%）到 val/，
使 train / val / test 三者互不相交：val 只用于训练期的模型选择，test 保持纯净、
只用于最终报告。搬动记录写入 <data>/val_split_manifest.json，因此可以精确撤销。

用法：
  python tools/make_classify_val.py                 # 默认 20%、seed=0
  python tools/make_classify_val.py --dry-run       # 只列出会动哪些文件
  python tools/make_classify_val.py --undo          # 按清单把文件搬回 train/

切好之后：
  python train.py classify --epochs 100 --imgsz 224 --batch 16
  python evaluate.py classify --split val           # 训练期口径
  python evaluate.py classify --split test          # 对外报的数
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MANIFEST_NAME = "val_split_manifest.json"
# ultralytics 会为每个 split 生成 <split>.cache，图片列表变了必须让它重建
CACHE_NAMES = ("train.cache", "val.cache", "test.cache")


def list_images(directory: Path) -> list:
    """按文件名排序返回目录下的图片，排序是为了让抽样结果只取决于 seed。"""
    return sorted(path for path in directory.iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def class_dirs(directory: Path) -> list:
    """train/ 下按名字排序的类别子目录（ultralytics 也按目录名决定类别顺序）。"""
    return sorted(path.name for path in directory.iterdir() if path.is_dir())


def clear_caches(root: Path) -> list:
    """删掉过期的 ultralytics 缓存，否则它会拿着旧的图片列表训练。"""
    removed = []
    for name in CACHE_NAMES:
        cache = root / name
        if cache.is_file():
            cache.unlink()
            removed.append(name)
    return removed


def split_train(data_root: Path, ratio: float, seed: int, force: bool,
                dry_run: bool) -> int:
    """按类别分层，把 train/<类别>/ 里的一部分图片移到 val/<类别>/。"""
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.is_dir():
        print(f"[错误] 找不到 {train_dir}")
        return 1

    existing = [path for path in val_dir.glob("*/*")] if val_dir.is_dir() else []
    if existing and not force:
        print(f"[错误] {val_dir} 里已经有 {len(existing)} 个文件，不重复切。")
        print("       想重切请先执行：python tools/make_classify_val.py --undo")
        return 1

    names = class_dirs(train_dir)
    expected = list(config.CLASSIFY_CLASS_NAMES)
    if names != expected:
        print(f"[警告] train/ 下的类别目录 {names} 与 config.CLASSIFY_CLASS_NAMES "
              f"{expected} 不一致，")
        print("       重训后 best.pt 的类别顺序会和界面上的名字对不上，请先确认。")

    manifest = {"seed": seed, "ratio": ratio, "data_root": str(data_root), "moved": {}}
    print(f"数据集：{data_root}")
    print(f"参数  ：ratio={ratio} seed={seed}" + ("  [dry-run]" if dry_run else ""))
    print(f"\n{'类别':<18}{'train 原':>10}{'移出':>8}{'train 后':>10}{'val':>7}")
    print("-" * 53)

    for name in names:
        source_dir = train_dir / name
        target_dir = val_dir / name
        images = list_images(source_dir)
        count = len(images)
        if count < 2:
            print(f"{name:<18}{count:>10}{0:>8}{count:>10}{0:>7}   [跳过：样本太少]")
            continue
        # 每个类别单独播种：以后新增了别的类别，也不会改变已经切好的类别
        rng = random.Random(f"{seed}:{name}")
        take = min(max(1, round(count * ratio)), count - 1)  # train 至少留 1 张
        chosen = sorted(rng.sample(images, take))

        moves = []
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
        for path in chosen:
            destination = target_dir / path.name
            if not dry_run:
                shutil.move(str(path), str(destination))
            moves.append([
                path.relative_to(data_root).as_posix(),
                destination.relative_to(data_root).as_posix(),
            ])
        manifest["moved"][name] = moves
        print(f"{name:<18}{count:>10}{take:>8}{count - take:>10}{take:>7}")

    if dry_run:
        print("\n[dry-run] 没有真的移动文件。")
        return 0

    manifest_path = data_root / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    removed = clear_caches(data_root)
    print(f"\n[完成] 清单已写入 {manifest_path}")
    if removed:
        print(f"[完成] 已清理过期缓存：{', '.join(removed)}")
    print("[完成] 现在 train / val / test 互不相交，test 只用于最终报告。")
    return 0


def undo(data_root: Path) -> int:
    """按清单把 val/ 里的文件搬回 train/，并删除清单。"""
    manifest_path = data_root / MANIFEST_NAME
    if not manifest_path.is_file():
        print(f"[错误] 找不到清单 {manifest_path}，无法撤销。")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    moved_back = 0
    for pairs in manifest.get("moved", {}).values():
        for old_path, new_path in pairs:
            source = data_root / new_path
            destination = data_root / old_path
            if source.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                moved_back += 1

    # 清掉空掉的 val/<类别> 与 val/（先深后浅）
    val_dir = data_root / "val"
    if val_dir.is_dir():
        for child in sorted(val_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if child.is_dir() and not any(child.iterdir()):
                child.rmdir()
        if not any(val_dir.iterdir()):
            val_dir.rmdir()

    removed = clear_caches(data_root)
    manifest_path.unlink()
    print(f"[完成] 已把 {moved_back} 个文件搬回 train/，val/ 已清理，清单已删除。")
    if removed:
        print(f"[完成] 已清理过期缓存：{', '.join(removed)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_classify_val.py",
        description="按类别分层，从分类数据集的 train/ 里切出 val/（移动，非复制）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", default=None,
                        help=f"数据集根目录，默认 {config.CLASSIFY_DATA_DIR}")
    parser.add_argument("--ratio", type=float, default=0.2, help="验证集比例，默认 0.2")
    parser.add_argument("--seed", type=int, default=0, help="随机种子，默认 0")
    parser.add_argument("--force", action="store_true", help="val/ 已存在时仍然继续切")
    parser.add_argument("--dry-run", action="store_true", help="只预览，不移动文件")
    parser.add_argument("--undo", action="store_true", help="按清单把 val/ 搬回 train/")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    data_root = Path(args.data) if args.data else config.CLASSIFY_DATA_DIR
    if not data_root.is_dir():
        print(f"[错误] 找不到数据集目录 {data_root}")
        return 1
    if args.undo:
        return undo(data_root)
    if not 0.0 < args.ratio < 1.0:
        print(f"[错误] --ratio 必须落在 (0, 1) 区间，收到 {args.ratio}")
        return 1
    return split_train(data_root, args.ratio, args.seed, args.force, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
