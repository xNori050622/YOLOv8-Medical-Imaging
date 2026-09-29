"""数据集准备与自检。

本模块只负责「数据集」这一件事：目录约定、data.yaml 生成、训练前的完整性检查。
三种任务沿用原仓库的目录布局，这样按原 README 下载的数据集不用改名：

  detection       detection/data/{train,valid,test}/{images,labels} + data.yaml
  classification  classification/Covid19-dataset/{train,val,test}/{Covid,Normal,Viral Pneumonia}
  segmentation    segmentation/Dataset_BUSI_with_GT/{benign,malignant,normal}/*.png
                  --(masks_to_polygons + splitfolders)-->
                  segmentation/data/{train,val,test}/{images,labels} + segmentation/data.yaml

原仓库从来没有生成过 segmentation/data.yaml，segment.train() 却直接引用它——
这是原项目无法训练的原因之一，这里把它补齐。
"""
from __future__ import annotations

from pathlib import Path

import yaml

import config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


# ------------------------------------------------------------------ 小工具
def _list_files(directory: Path, suffixes=None):
    """列出目录下的文件（按名字排序），目录不存在时返回空列表。"""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        item for item in directory.iterdir()
        if item.is_file() and (suffixes is None or item.suffix.lower() in suffixes)
    )


def count_images(directory) -> int:
    """统计目录下的图片数量。"""
    return len(_list_files(directory, IMAGE_SUFFIXES))


def _write_yaml(path: Path, payload: dict) -> Path:
    """写 UTF-8 的 YAML。

    Windows 上 Path.write_text 默认走 GBK，而 data.yaml 里可能有中文路径，
    必须显式指定 utf-8，否则会写出无法回读的文件。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, default_flow_style=False)
    path.write_text(text, encoding="utf-8")
    return path


# ------------------------------------------------------------------ 检测
def detect_splits(root=None) -> dict:
    """找出 detection 数据集里实际存在的 split，返回 {split: 'train/images' 这样的相对路径}。

    Roboflow 导出的验证集目录叫 valid/，别的来源可能叫 val/，两种都认。
    """
    root = Path(root or config.DETECT_DATA_YAML.parent)
    found = {}
    for split, candidates in (("train", ("train",)), ("val", ("valid", "val")), ("test", ("test",))):
        for name in candidates:
            if (root / name / "images").is_dir():
                found[split] = f"{name}/images"
                break
    return found


def build_detect_data_yaml(root=None, class_names=None, overwrite=False) -> Path:
    """生成 detection/data/data.yaml。

    Roboflow 导出通常自带 data.yaml，所以默认不覆盖已存在的文件；
    本函数用于「只有 images/labels 目录」的情况。
    """
    root = Path(root or config.DETECT_DATA_YAML.parent)
    target = root / "data.yaml"
    if target.is_file() and not overwrite:
        return target

    splits = detect_splits(root)
    if "train" not in splits:
        raise FileNotFoundError(f"在 {root} 下找不到 train/images，无法生成 data.yaml。")

    names = list(class_names or config.DETECT_CLASS_NAMES)
    payload = {
        "path": str(root),
        "train": splits["train"],
        "nc": len(names),
        "names": names,
    }
    for split in ("val", "test"):
        if split in splits:
            payload[split] = splits[split]
    if "val" not in payload and "test" not in payload:
        raise FileNotFoundError(f"{root} 下既没有 valid/ 也没有 test/，无法做验证。")
    return _write_yaml(target, payload)


def detect_label_stats(root=None) -> dict:
    """统计各 split 的图片数、标签数、实例数、每类实例数，以及可疑项。"""
    root = Path(root or config.DETECT_DATA_YAML.parent)
    stats = {}
    for split, relative in detect_splits(root).items():
        images_dir = root / relative
        labels_dir = root / relative[: -len("images")] / "labels"
        images = _list_files(images_dir, IMAGE_SUFFIXES)

        per_class = {name: 0 for name in config.DETECT_CLASS_NAMES}
        boxes = 0
        unknown_ids = 0
        orphan_images = 0
        for image in images:
            label_path = labels_dir / f"{image.stem}.txt"
            if not label_path.is_file():
                orphan_images += 1
                continue
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) < 5:  # 空标签文件属于正常情况（纯背景图）
                    continue
                boxes += 1
                class_id = int(float(parts[0]))
                if 0 <= class_id < len(config.DETECT_CLASS_NAMES):
                    per_class[config.DETECT_CLASS_NAMES[class_id]] += 1
                else:
                    unknown_ids += 1

        stats[split] = {
            "images": len(images),
            "labels": len(_list_files(labels_dir, {".txt"})),
            "boxes": boxes,
            "per_class": per_class,
            "orphan_images": orphan_images,
            "unknown_class_ids": unknown_ids,
        }
    return stats


# ------------------------------------------------------------------ 分类
def classify_splits(root=None) -> dict:
    """返回分类数据集里实际存在的 split 目录 {split: Path}。"""
    root = Path(root or config.CLASSIFY_DATA_DIR)
    return {split: root / split for split in ("train", "val", "test") if (root / split).is_dir()}


def classify_class_order(root=None) -> list:
    """按 ultralytics 的规则算出类别顺序。

    check_cls_dataset() 内部是 `sorted(x.name for x in (data_dir/'train').iterdir())`，
    也就是 train/ 下排序后的文件夹名，所以这里用同样的规则，
    以便校验文件夹命名是否与已训练权重的 names（Covid/Normal/Viral Pneumonia）一致。
    """
    train_dir = Path(root or config.CLASSIFY_DATA_DIR) / "train"
    if not train_dir.is_dir():
        return []
    return sorted(item.name for item in train_dir.iterdir() if item.is_dir())


def classify_class_counts(root=None) -> dict:
    """统计分类数据集每个 split、每个类别的图片数。"""
    counts = {}
    for split, directory in classify_splits(root).items():
        counts[split] = {
            item.name: count_images(item)
            for item in sorted(directory.iterdir())
            if item.is_dir()
        }
    return counts


# ------------------------------------------------------------------ 分割
def segment_split_dirs(root=None) -> dict:
    """返回分割数据集（split 之后）存在的 split {split: Path}。"""
    root = Path(root or config.SEGMENT_SPLIT_DIR)
    return {
        split: root / split
        for split in ("train", "val", "test")
        if (root / split / "images").is_dir()
    }


def segment_raw_counts(root=None) -> dict:
    """统计 BUSI 原始数据集（benign/malignant/normal）的图片与掩码数量。

    normal 目录只有图片、没有 *_mask.png，这是 BUSI 数据集本身的特点。
    """
    root = Path(root or config.SEGMENT_RAW_DIR)
    counts = {}
    for class_dir in (sorted(root.iterdir()) if root.is_dir() else []):
        if not class_dir.is_dir():
            continue
        images = [p for p in _list_files(class_dir, IMAGE_SUFFIXES) if "mask" not in p.stem.lower()]
        masks = [p for p in _list_files(class_dir, IMAGE_SUFFIXES) if "mask" in p.stem.lower()]
        counts[class_dir.name] = {"images": len(images), "masks": len(masks)}
    return counts


def build_segment_data_yaml(root=None, yaml_path=None, class_names=None,
                            overwrite=True) -> Path:
    """生成 data.yaml（原仓库缺失的文件）。

    指向 splitfolders 产出的 <root>/{train,val,test}/{images,labels}。
    默认位置是 config.SEGMENT_SPLIT_DIR 与 config.SEGMENT_DATA_YAML，
    也可以显式传 root/yaml_path（冒烟测试就是这么用的）。
    """
    root = Path(root or config.SEGMENT_SPLIT_DIR)
    target = Path(yaml_path or config.SEGMENT_DATA_YAML)
    if target.is_file() and not overwrite:
        return target

    splits = segment_split_dirs(root)
    if "train" not in splits:
        raise FileNotFoundError(
            f"在 {root} 下找不到 train/images。请先运行数据准备：\n"
            "  python train.py segment --prepare"
        )

    names = list(class_names or config.SEGMENT_CLASS_NAMES)
    payload = {"path": str(root), "train": "train/images", "nc": len(names), "names": names}
    for split in ("val", "test"):
        if split in splits:
            payload[split] = f"{split}/images"
    if "val" not in payload:
        payload["val"] = payload["train"]  # 没有验证集时退回训练集，避免训练直接报错
    return _write_yaml(target, payload)


def read_data_yaml(path) -> dict:
    """读取 data.yaml，失败时返回空 dict（供自检展示用）。"""
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


# ------------------------------------------------------------------ 汇总自检
def report() -> dict:
    """汇总三个任务的「基础权重 + 数据集」自检结果（结构化字典）。"""
    assets = config.check_training_assets()
    detect_root = config.DETECT_DATA_YAML.parent
    detect_yaml = read_data_yaml(config.DETECT_DATA_YAML)
    classify_order = classify_class_order()

    return {
        "detect": {
            "weights": assets["detect"]["weights"],
            "data_yaml": str(config.DETECT_DATA_YAML),
            "data_yaml_exists": config.DETECT_DATA_YAML.is_file(),
            "splits": detect_splits(detect_root),
            "stats": detect_label_stats(detect_root) if config.DETECT_DATA_YAML.is_file() else {},
            "classes_in_yaml": detect_yaml.get("names"),
            "classes_expected": list(config.DETECT_CLASS_NAMES),
        },
        "classify": {
            "weights": assets["classify"]["weights"],
            "data_dir": str(config.CLASSIFY_DATA_DIR),
            "data_dir_exists": config.CLASSIFY_DATA_DIR.is_dir(),
            "splits": {k: str(v) for k, v in classify_splits().items()},
            "class_order": classify_order,
            "classes_expected": list(config.CLASSIFY_CLASS_NAMES),
            "counts": classify_class_counts(),
        },
        "segment": {
            "weights": assets["segment"]["weights"],
            "raw_dir": str(config.SEGMENT_RAW_DIR),
            "raw_dir_exists": config.SEGMENT_RAW_DIR.is_dir(),
            "raw_counts": segment_raw_counts(),
            "data_yaml": str(config.SEGMENT_DATA_YAML),
            "data_yaml_exists": config.SEGMENT_DATA_YAML.is_file(),
            "splits": {k: str(v) for k, v in segment_split_dirs().items()},
        },
    }


def print_report(payload=None) -> None:
    """把 report() 的结果打印成人能读的清单。"""
    payload = payload or report()

    print("=" * 72)
    print("训练资源自检")
    print("=" * 72)

    for task in ("detect", "classify", "segment"):
        info = payload[task]
        print(f"\n[{task}]  基础权重: {'已就位' if info['weights'] else '缺失'}")

        if task == "detect":
            print(f"  data.yaml : {info['data_yaml']}"
                  f"  ({'存在' if info['data_yaml_exists'] else '不存在'})")
            if info["splits"]:
                for split, stats in info["stats"].items():
                    print(f"  {split:5s} images={stats['images']:5d} labels={stats['labels']:5d}"
                          f" boxes={stats['boxes']:6d} orphan={stats['orphan_images']:4d}"
                          f" per_class={stats['per_class']}")
            else:
                print("  (未找到 train/valid/test 目录)")
            print(f"  期望类别  : {info['classes_expected']}")
            if info["classes_in_yaml"]:
                print(f"  yaml 类别 : {info['classes_in_yaml']}")

        elif task == "classify":
            print(f"  数据目录  : {info['data_dir']}"
                  f"  ({'存在' if info['data_dir_exists'] else '不存在'})")
            print(f"  splits    : {list(info['splits']) or '(无)'}")
            print(f"  类别顺序  : {info['class_order'] or '(无)'}")
            print(f"  期望类别  : {info['classes_expected']}")
            if info["class_order"] and info["class_order"] != info["classes_expected"]:
                print("  !! 类别顺序与既有权重不一致，重训后界面上的类别名会错位")
            for split, counts in info["counts"].items():
                print(f"  {split:5s} {counts}")

        else:
            print(f"  BUSI 原始 : {info['raw_dir']}"
                  f"  ({'存在' if info['raw_dir_exists'] else '不存在'})")
            for name, counts in info["raw_counts"].items():
                print(f"    {name:9s} images={counts['images']:4d} masks={counts['masks']:4d}")
            print(f"  data.yaml : {info['data_yaml']}"
                  f"  ({'存在' if info['data_yaml_exists'] else '不存在'})")
            print(f"  已 split  : {list(info['splits']) or '(无)'}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    print_report()


