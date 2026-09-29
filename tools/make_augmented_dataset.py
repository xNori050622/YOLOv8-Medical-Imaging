"""产出一份「只在 train 上做了增强」的分割数据集，供消融实验使用。

背景：为什么不能直接增强 segmentation/data
-----------------------------------------
`augment.augment_split_dir` 只增补 train/，val/ 与 test/ 逐字节原样复制，
所以基线臂和增强臂**共用完全相同的验证集与测试集**，数字才可比。本脚本是它的
命令行外壳：一次产出目录 + data.yaml + 记录超参的 manifest，并自己再验一遍
val/test 未被改动（防止 augment.py 将来改坏了却没人发现）。

为什么 manifest 要单独存
------------------------
消融表里「+CLAHE」这一行必须能追溯到具体的 clip_limit / tile_grid_size / seed，
否则半年后没人说得清那一行到底跑的是什么。manifest 就是这一行的出处。

用法：
  # 基线臂：直接用 segmentation/data，不需要本脚本
  python train.py segment --name baseline

  # +CLAHE 臂（默认 clip_limit=2.0、tile_grid_size=8）
  python tools/make_augmented_dataset.py --out segmentation/data_aug/clahe --clahe

  # +ELASTIC 臂（默认 alpha=8.0、sigma=6.0）
  python tools/make_augmented_dataset.py --out segmentation/data_aug/elastic --elastic

  # 两个都要
  python tools/make_augmented_dataset.py --out segmentation/data_aug/both --clahe --elastic

  python tools/make_augmented_dataset.py --out D:\\temp\\probe --clahe --dry-run   # 先看看

产出之后（注意必须显式传 --data，别指望它自己找）：
  python train.py segment --data segmentation\\data_aug\\clahe\\data.yaml --name clahe
  python evaluate.py segment --name clahe --split val --save runs\\ablation\\clahe_val.json

关于 val/test：本脚本会用 SHA256 逐文件比对源目录与产出目录的 val/test，
只要有一个字节不同就直接报错退出 —— 静默错配比崩溃危险得多。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

MANIFEST_NAME = "augment_manifest.json"

# ultralytics 会为每个 split 生成 <split>.cache / labels.cache，内容里存的是
# **绝对路径**。它们是派生产物、不是数据，所以「源目录里有、产出目录里没有」才是
# 正确的：带着旧路径的 cache 会让 ultralytics 去读源目录的图（正是
# make_classify_val.py 会主动删掉它的原因）。比对时排除，否则必然误报。
IGNORED_SUFFIXES = (".cache",)


def _sha256(path: Path) -> str:
    """算文件摘要，用于证明 val/test 是逐字节复制的。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(directory: Path) -> dict:
    """返回 {相对路径: sha256}，用于整体的不可变性比对（不含 .cache 等派生物）。"""
    if not directory.is_dir():
        return {}
    return {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix not in IGNORED_SUFFIXES
    }


def verify_untouched(source_dir: Path, output_dir: Path, augmented_splits) -> list:
    """确认未被增强的 split 与源目录逐字节一致，返回被检查的 split 列表。

    这是消融结果的正确性前提：val/test 只要被增强过一次，基线与增强臂就失去了
    可比性。这里独立于 augment.py 重新校验一遍，避免「实现改了、测试没跟上」。

    只比对数据文件（图片与标签），ultralytics 的 *.cache 是绝对路径派生物，
    见 IGNORED_SUFFIXES 的说明。
    """
    checked = []
    for split_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        if split_dir.name in augmented_splits:
            continue
        target = output_dir / split_dir.name
        before = _tree_digest(split_dir)
        after = _tree_digest(target)
        if before != after:
            missing = sorted(set(before) - set(after))
            extra = sorted(set(after) - set(before))
            changed = sorted(
                name for name in set(before) & set(after) if before[name] != after[name]
            )
            raise RuntimeError(
                f"split '{split_dir.name}' 在产出目录里与源目录不一致，"
                "增强实现有 bug：\n"
                f"  缺失 {len(missing)} 个{('：' + ', '.join(missing[:5])) if missing else ''}\n"
                f"  多余 {len(extra)} 个{('：' + ', '.join(extra[:5])) if extra else ''}\n"
                f"  内容不同 {len(changed)} 个{('：' + ', '.join(changed[:5])) if changed else ''}\n"
                f"  产出目录 {output_dir} 不可用于实验，请删除后重试。"
            )
        print(f"[核对] {split_dir.name}：{len(before)} 个数据文件与源目录逐字节一致")
        checked.append(split_dir.name)
    return checked


def resolve_output(source_dir: Path, output_dir: Path, force: bool,
                   dry_run: bool = False) -> None:
    """校验输出目录可用：不能与源目录重叠，已存在则需要 --force。

    dry_run=True 时只做校验、绝不删目录 —— 预览不该改盘。
    """
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()

    if source_dir == output_dir:
        raise ValueError("输出目录不能与源目录相同，否则原图会被覆盖。")
    if source_dir in output_dir.parents:
        raise ValueError(
            f"输出目录不能放在源目录内部：{output_dir}\n"
            f"（源目录 {source_dir} 的每个子目录都会被当成一个 split）"
        )
    if output_dir in source_dir.parents:
        raise ValueError(
            f"源目录不能放在输出目录内部：{source_dir}\n"
            "否则复制源目录时会把输出目录一并卷进去。"
        )

    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise ValueError(f"输出路径已被同名文件占用：{output_dir}")

    existing = any(output_dir.iterdir())
    if existing and not force:
        raise FileExistsError(
            f"输出目录已存在且非空：{output_dir}\n"
            "  想重做请加 --force（会先删掉整个目录，避免残留上一次的 _aug*.png 混进来）。"
        )
    if existing and not dry_run:
        shutil.rmtree(output_dir)


def _parse_splits(raw: str, source_dir: Path) -> tuple:
    """把 --splits 解析成元组，并确认每个 split 在源目录里真的存在。"""
    splits = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not splits:
        raise ValueError("--splits 不能为空，默认是 train")
    available = {path.name for path in source_dir.iterdir() if path.is_dir()}
    unknown = [name for name in splits if name not in available]
    if unknown:
        raise ValueError(
            f"源目录里没有这些 split：{', '.join(unknown)}"
            f"（现有：{', '.join(sorted(available))}）"
        )
    return splits


def run(args) -> int:
    """执行一次数据集产出，返回进程退出码。"""
    source_dir = Path(args.source) if args.source else Path(config.SEGMENT_SPLIT_DIR)
    output_dir = Path(args.out)

    if not (source_dir / "train" / "images").is_dir():
        print(f"[错误] 源目录里找不到 train/images：{source_dir}")
        print("       请先准备数据：python train.py segment --prepare")
        return 1

    if not args.clahe and not args.elastic:
        print("[错误] 既没开 --clahe 也没开 --elastic，那只是复制一份数据集。")
        print("       基线臂直接用 segmentation/data 即可，不必产出新目录。")
        return 1

    if args.copies < 1:
        print(f"[错误] --copies 至少为 1，收到 {args.copies}；只复制不增强请直接用源目录")
        return 1

    splits = _parse_splits(args.splits, source_dir)
    clahe_options = None
    if args.clahe:
        clahe_options = {"clip_limit": args.clip_limit, "tile_grid_size": args.tile_grid}
    elastic_options = None
    if args.elastic:
        elastic_options = {"alpha": args.alpha, "sigma": args.sigma}

    resolve_output(source_dir, output_dir, args.force, args.dry_run)

    print(f"源数据集：{source_dir}")
    print(f"输出目录：{output_dir}")
    print(f"超参    ：copies={args.copies} seed={args.seed}")
    print(f"          CLAHE  ={clahe_options if clahe_options else '关闭'}")
    print(f"          elastic={elastic_options if elastic_options else '关闭'}")
    print(f"          增强 {', '.join(splits)}；其余 split 逐字节复制")

    if args.dry_run:
        print("\n[dry-run] 参数与路径检查通过，没有写任何文件。")
        return 0

    import augment  # noqa: E402  真正产出时才加载 cv2 / numpy

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = augment.augment_split_dir(
        source_dir,
        output_dir,
        copies=args.copies,
        clahe=clahe_options,
        elastic=elastic_options,
        seed=args.seed,
        augment_splits=splits,
    )

    # 独立复核 val/test 未被改动 —— 这是基线与增强臂可比的前提
    untouched = verify_untouched(source_dir, output_dir, splits)

    print(f"\n{'split':<10}{'原图':>8}{'增强':>8}{'合计':>8}{'标签':>8}")
    print("-" * 42)
    for split in sorted(stats):
        record = stats[split]
        # augment_split_dir 的语义：images/labels 只统计原图原标签，augmented 单列
        total_images = record["images"] + record["augmented"]
        total_labels = record["labels"] + record["augmented"]
        print(f"{split:<10}{record['images']:>8}{record['augmented']:>8}"
              f"{total_images:>8}{total_labels:>8}")

    for split in sorted(stats):
        failed = stats[split]["skipped"]
        if failed:
            print(f"\n[警告] {split} 有 {len(failed)} 张图解码失败被跳过："
                  f"{', '.join(failed[:5])}")

    # data.yaml 复用 dataset.py 的写法，保证 path/names 与训练侧完全一致
    import dataset  # noqa: E402

    yaml_path = dataset.build_segment_data_yaml(
        root=output_dir, yaml_path=output_dir / "data.yaml", overwrite=True)

    manifest = {
        "source": str(source_dir.resolve()),
        "output": str(output_dir.resolve()),
        "copies": args.copies,
        "seed": args.seed,
        "splits_augmented": list(splits),
        "splits_untouched": list(untouched),
        "clahe": clahe_options,
        "elastic": elastic_options,
        "stats": stats,
    }
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")

    print(f"\n[完成] data.yaml ：{yaml_path}")
    print(f"[完成] 超参记录  ：{manifest_path}")
    print(f"[完成] 已逐文件核对 val/test，与源目录完全一致（{', '.join(untouched) or '无'}）")
    print("\n下一步（必须显式传 --data，否则会静默用回默认数据集）：")
    print(f"  python train.py segment --data \"{yaml_path}\" --name <臂名>")
    print("  python evaluate.py segment --name <臂名> --split val")
    return 0


def build_parser() -> argparse.ArgumentParser:
    import augment  # noqa: E402  取默认超参，避免两处各写一份常量

    parser = argparse.ArgumentParser(
        prog="make_augmented_dataset.py",
        description="产出「只在 train 上增强」的分割数据集（val/test 逐字节保留）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", required=True,
                        help="输出目录，例如 segmentation/data_aug/clahe")
    parser.add_argument("--source", default=None,
                        help=f"源数据集根目录，默认 {config.SEGMENT_SPLIT_DIR}")
    parser.add_argument("--copies", type=int, default=1,
                        help="每张图生成的增强副本数，默认 1（train 随之翻倍）")
    parser.add_argument("--clahe", action="store_true",
                        help="开启 CLAHE 对比度增强（光度变换，标签不变）")
    parser.add_argument("--clip-limit", type=float,
                        default=augment.CLAHE_DEFAULTS["clip_limit"],
                        help="CLAHE 对比度上限，默认 %(default)s")
    parser.add_argument("--tile-grid", type=int,
                        default=int(augment.CLAHE_DEFAULTS["tile_grid_size"]),
                        help="CLAHE 网格边长，默认 %(default)s")
    parser.add_argument("--elastic", action="store_true",
                        help="开启弹性形变（几何变换，多边形顶点同步变换）")
    parser.add_argument("--alpha", type=float, default=augment.ELASTIC_DEFAULTS["alpha"],
                        help="弹性形变位移强度（像素），默认 %(default)s")
    parser.add_argument("--sigma", type=float, default=augment.ELASTIC_DEFAULTS["sigma"],
                        help="弹性形变平滑尺度，默认 %(default)s")
    parser.add_argument("--splits", default="train",
                        help="需要增强的 split，逗号分隔，默认 train")
    parser.add_argument("--seed", type=int, default=0, help="随机种子，默认 0")
    parser.add_argument("--force", action="store_true",
                        help="输出目录已存在时整个删掉重做，避免残留上次的 _aug* 文件")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不写文件")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as error:
        print(f"[错误] {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
