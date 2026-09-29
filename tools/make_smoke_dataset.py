"""生成合成数据，用于在没有真实数据集时端到端验证训练流程。

真实的三个数据集都需要 Kaggle / Roboflow 账号才能下载（见 README），
所以数据到手之前，可以用本脚本造一份极小的合成数据集，确认 train.py 的整条
链路（数据准备 -> 训练 -> 产出 best.pt）是通的。

生成的内容（在 --out 目录下）：
  detection/{train,valid}/{images,labels} + data.yaml
  classification/{train,test}/<类别名>/          （故意不给 val/，顺便验证回退逻辑）
  segmentation/raw/{benign,malignant,normal}     BUSI 风格：normal 目录没有掩码
  segmentation/data/{train,val,test}/{images,labels} + data.yaml

用法：
  python tools/make_smoke_dataset.py --out D:\\temp\\p3_smoke

随后（注意 --name smoke，避免覆盖仓库里可用的 best.pt）：
  python train.py detect   --data ...\\detection\\data.yaml --epochs 1 --imgsz 64 --batch 4 --no-amp --name smoke
  python train.py classify --data ...\\classification       --epochs 1 --imgsz 64 --batch 4 --no-amp --name smoke
  python train.py segment  --data ...\\segmentation\\data.yaml --epochs 1 --imgsz 64 --batch 4 --no-amp --name smoke
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

RNG = np.random.default_rng(0)

# 每个任务生成的图片尺寸都取小值，保证 1 个 epoch 在 CPU 上也能跑完
DETECT_SIZE = 96
CLASSIFY_SIZE = 64
SEGMENT_SIZE = 96


def _reset(path: Path) -> Path:
    if Path(path).exists():
        shutil.rmtree(path)
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def _save(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _blob(image: np.ndarray, center, radius: int, color, irregular: float = 0.0) -> None:
    """在图上画一个近似圆形的色块，irregular>0 时用多边形做出不规则边缘。"""
    cx, cy = center
    if irregular <= 0:
        cv2.circle(image, (cx, cy), radius, color, -1)
        return
    angles = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    radii = radius * (1 + irregular * RNG.normal(0, 0.18, size=angles.size))
    points = np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=1)
    cv2.fillPoly(image, [points.astype(np.int32)], color)


# ------------------------------------------------------------------ 检测
def make_detection(root: Path) -> Path:
    """造 RBC/WBC/Platelets 三种「细胞」，输出 YOLO 检测格式。"""
    root = _reset(root)
    # 类别 -> (半径范围, 颜色(BGR), 每图出现概率)
    spec = {
        "Platelets": ((3, 5), (80, 80, 220), 0.8),
        "RBC": ((9, 14), (200, 90, 90), 1.0),
        "WBC": ((6, 9), (120, 180, 240), 0.6),
    }
    names = list(config.DETECT_CLASS_NAMES)

    for split, count in (("train", 14), ("valid", 5)):
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for index in range(count):
            size = DETECT_SIZE
            image = RNG.integers(20, 45, size=(size, size, 3), dtype=np.uint8)
            lines = []

            for class_id, name in enumerate(names):
                (low, high), color, probability = spec[name]
                for _ in range(RNG.integers(1, 4)):
                    if RNG.random() > probability:
                        continue
                    radius = int(RNG.integers(low, high + 1))
                    cx = int(RNG.integers(radius + 1, size - radius - 1))
                    cy = int(RNG.integers(radius + 1, size - radius - 1))
                    _blob(image, (cx, cy), radius, tuple(int(c) for c in color))
                    lines.append(
                        f"{class_id} {cx / size:.6f} {cy / size:.6f} "
                        f"{2 * radius / size:.6f} {2 * radius / size:.6f}"
                    )

            # 故意留一张纯背景图（空标签文件），验证训练不会因空标签报错
            if split == "train" and index == 0:
                lines = []

            _save(image, images_dir / f"cell_{index:03d}.png")
            (labels_dir / f"cell_{index:03d}.txt").write_text("\n".join(lines), encoding="utf-8")

    # 只有 images/labels、没有 data.yaml，顺便验证 train.py 会自动补生成
    return root


# ------------------------------------------------------------------ 分类
def _classify_pattern(class_id: int) -> np.ndarray:
    """按类别生成视觉上可区分的图案，保证 1 个 epoch 后 loss 会下降。"""
    size = CLASSIFY_SIZE
    if class_id == 0:  # Covid：暗底 + 大量小亮点（磨玻璃样）
        image = RNG.integers(25, 45, size=(size, size, 3), dtype=np.uint8)
        for _ in range(45):
            center = (int(RNG.integers(0, size)), int(RNG.integers(0, size)))
            cv2.circle(image, center, int(RNG.integers(1, 3)), (215, 215, 215), -1)
        return image
    if class_id == 1:  # Normal：均匀灰底
        image = np.full((size, size, 3), int(RNG.integers(95, 125)), dtype=np.uint8)
        return cv2.add(image, RNG.integers(0, 12, size=(size, size, 3), dtype=np.uint8))
    # Viral Pneumonia：亮底 + 大块白斑
    image = RNG.integers(120, 150, size=(size, size, 3), dtype=np.uint8)
    for _ in range(5):
        _blob(image, (int(RNG.integers(12, size - 12)), int(RNG.integers(12, size - 12))),
              int(RNG.integers(6, 12)), (245, 245, 245))
    return image


def make_classification(root: Path) -> Path:
    """造分类数据集。只给 train/ 与 test/，故意不给 val/（验证回退逻辑）。"""
    root = _reset(root)
    for split, per_class in (("train", 10), ("test", 4)):
        for class_id, name in enumerate(config.CLASSIFY_CLASS_NAMES):
            class_dir = root / split / name
            class_dir.mkdir(parents=True, exist_ok=True)
            slug = name.lower().replace(" ", "_")
            for index in range(per_class):
                _save(_classify_pattern(class_id), class_dir / f"{slug}_{index:03d}.png")
    return root


# ------------------------------------------------------------------ 分割
# 类别 -> (病例数, 轮廓形状)；BUSI 的 normal 目录只有图片、没有掩码
SEGMENT_PLAN = {
    "benign": (5, "regular"),
    "malignant": (5, "irregular"),
    "normal": (3, None),
}


def make_segmentation(root: Path) -> Path:
    """造 BUSI 风格的原始数据，再走项目自己的「掩码转多边形 + 划分」流程。"""
    root = _reset(root)
    raw = root / "raw"
    size = SEGMENT_SIZE

    for class_name, (cases, shape) in SEGMENT_PLAN.items():
        class_dir = raw / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for index in range(cases):
            image = RNG.integers(30, 60, size=(size, size, 3), dtype=np.uint8)
            stem = f"{class_name} ({index})"
            mask = None

            if shape is not None:
                center = (size // 2, size // 2)
                radius = int(RNG.integers(18, 28))
                irregular = 0.0 if shape == "regular" else 0.45
                _blob(image, center, radius, (215, 215, 215), irregular=irregular)
                mask = np.zeros((size, size), dtype=np.uint8)
                _blob(mask, center, radius, 255, irregular=irregular)

            _save(image, class_dir / f"{stem}.png")
            if mask is not None:
                _save(mask, class_dir / f"{stem}_mask.png")
                # 最后一个 benign 病例再造第二张掩码，验证「多掩码追加进同一个 txt」
                if class_name == "benign" and index == cases - 1:
                    second = np.zeros((size, size), dtype=np.uint8)
                    _blob(second, (int(size * 0.28), int(size * 0.28)), 6, 255)
                    _save(second, class_dir / f"{stem}_mask_1.png")

    # 用项目自己的实现做转换与划分，这样验证的就是真实链路
    import dataset
    from segmentation.masks_to_polygons import masks_to_polygons, split_train_test_val

    staging = root / "staging"
    mask_stats = masks_to_polygons(
        input_dir=raw, image_dir=staging / "images", label_dir=staging / "labels"
    )
    split_stats = split_train_test_val(
        input_dir=staging, output_dir=root / "data", keep_staging=True
    )
    yaml_path = dataset.build_segment_data_yaml(
        root=root / "data", yaml_path=root / "data.yaml"
    )

    print(f"[segment] 掩码转多边形: {mask_stats}")
    print(f"[segment] train/val/test 划分: {split_stats}")
    return yaml_path


# ------------------------------------------------------------------ 入口
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="生成合成数据集，用于训练流程自检")
    parser.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "p3_smoke"),
                        help="输出目录，默认系统临时目录下的 p3_smoke")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"生成合成数据到：{out}\n")

    make_detection(out / "detection")
    make_classification(out / "classification")
    make_segmentation(out / "segmentation")

    detect_yaml = out / "detection" / "data.yaml"
    classify_dir = out / "classification"
    segment_yaml = out / "segmentation" / "data.yaml"

    print("\n完成。可用下面三条命令验证训练链路")
    print("（--name smoke 让产物写到 runs/<task>/smoke/，不会覆盖已有的 best.pt）：\n")
    print(f'  python train.py detect   --data "{detect_yaml}" '
          "--epochs 1 --imgsz 64 --batch 4 --no-amp --name smoke")
    print(f'  python train.py classify --data "{classify_dir}" '
          "--epochs 1 --imgsz 64 --batch 4 --no-amp --name smoke")
    print(f'  python train.py segment  --data "{segment_yaml}" '
          "--epochs 1 --imgsz 64 --batch 4 --no-amp --name smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
