"""BUSI 掩码 -> YOLO 多边形标签，以及 train/val/test 划分。

原实现全部使用相对路径 'segmentation/...'，隐含要求「必须在项目根目录下启动」，
换个工作目录就会在错误位置建目录甚至报错。这里改为统一走 config。

与原实现的差异（都在函数 docstring 里说明）：
  * 掩码文件名不再用固定切片 [-8:-4] / [-10:-6]，改用 split("_mask", 1)[0]，
    这样 benign (1)_mask.png 与 benign (1)_mask_1.png 都能正确归到
    「benign (1)」这个病例名下。
  * 同一病例的多张掩码仍以追加方式写进同一个 txt；因为输出目录每次先清空，
    重跑不会残留上一次的重复标签。
  * normal 类在 Kaggle 镜像里也有掩码文件，但实测 133 个掩码全部是纯黑（无病灶），
    转换后得到的是空标签文件，YOLO 会把它们当背景图。这与仓库里 nc=3（含 normal）
    的既有权重一致。（上游 README 说「normal 只有图片没有掩码」，与本镜像不符，
    两种情形本模块都能处理。）
  * split_train_test_val 不再用 splitfolders 分别划分 images/ 与 labels/，
    改为「按病例名归组后再划分」。原做法会把图片和它的掩码分到不同 split，
    训练时图片配错掩码却不报错（见该函数 docstring 的实测例子）。
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2

import config

# 小于该面积的轮廓会被丢弃（沿用原实现的阈值 200）
MIN_CONTOUR_AREA = 200

# 类别名 -> class_id，顺序取自 config.SEGMENT_CLASS_NAMES（benign/malignant/normal）
CLASS_IDS = {name: index for index, name in enumerate(config.SEGMENT_CLASS_NAMES)}

# 中间产物目录：data_/{images,labels}，split 完成后会被删除
STAGING_DIR = config.PROJECT_ROOT / "segmentation" / "data_"


def _reset_dir(path: Path) -> Path:
    """清空并重建目录。"""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _label_stem(filename: str) -> str:
    """从掩码文件名取出所属病例名。

    'benign (1)_mask.png'   -> 'benign (1)'
    'benign (1)_mask_1.png' -> 'benign (1)'
    """
    return filename.split("_mask", 1)[0]


def masks_to_polygons(input_dir=None, image_dir=None, label_dir=None) -> dict:
    """把 BUSI 掩码转成 YOLO 多边形标签，同时把原图整平到同一个 images/ 目录。"""
    input_dir = Path(input_dir or config.SEGMENT_RAW_DIR)
    image_dir = Path(image_dir or STAGING_DIR / "images")
    label_dir = Path(label_dir or STAGING_DIR / "labels")

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"找不到 BUSI 原始数据集：{input_dir}\n"
            "请按 README 下载 breast-ultrasound-images-dataset，"
            "解压后应包含 benign/ malignant/ normal/ 三个子目录。"
        )

    _reset_dir(image_dir)
    _reset_dir(label_dir)

    stats = {"images": 0, "masks": 0, "labels": 0, "skipped_classes": []}

    for class_dir in sorted(input_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_id = CLASS_IDS.get(class_dir.name)
        if class_id is None:
            stats["skipped_classes"].append(class_dir.name)
            continue

        for item in sorted(class_dir.iterdir()):
            if not item.is_file():
                continue

            # 不是掩码的图片：直接复制（这类图不会有标签文件，见模块 docstring）
            if "_mask" not in item.stem:
                shutil.copy2(item, image_dir / item.name)
                stats["images"] += 1
                continue

            stats["masks"] += 1
            mask = cv2.imread(str(item), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            height, width = mask.shape
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 轮廓点 -> 归一化的 [x1,y1,x2,y2,...] 平铺多边形
            polygons = []
            for contour in contours:
                if cv2.contourArea(contour) <= MIN_CONTOUR_AREA:
                    continue
                flat = []
                for point in contour:
                    x, y = point[0]
                    flat.append(x / width)
                    flat.append(y / height)
                polygons.append(flat)

            # "a" 追加模式：同一病例的多张掩码写进同一个 txt；
            # polygons 为空时也会建出空文件，等价于「背景图」。
            output_path = label_dir / f"{_label_stem(item.name)}.txt"
            with open(output_path, "a", encoding="utf-8") as handle:
                for polygon in polygons:
                    for index, value in enumerate(polygon):
                        if index == 0:
                            handle.write(f"{class_id} {value} ")
                        elif index == len(polygon) - 1:
                            handle.write(f"{value}\n")
                        else:
                            handle.write(f"{value} ")

    stats["labels"] = len(list(label_dir.glob("*.txt")))
    return stats


def _index_by_stem(directory: Path) -> dict:
    """{病例名: [文件...]}。用于按病例（而不是按文件）做划分。"""
    index: dict = {}
    directory = Path(directory)
    if not directory.is_dir():
        return index
    for path in sorted(directory.iterdir()):
        if path.is_file():
            index.setdefault(path.stem, []).append(path)
    return index


def split_train_test_val(input_dir=None, output_dir=None, ratio=(0.8, 0.1, 0.1),
                         seed=1337, keep_staging=False) -> dict:
    """按「病例名」把 data_/{images,labels} 划分成 data/{train,val,test}/{images,labels}。

    为什么不再用 splitfolders
    ------------------------
    原作者用 splitfolders.ratio 直接划分 data_/，它会**分别**洗牌并切分 images/
    与 labels/ 两个子目录，能否对上完全依赖「文件数相等且排序顺序一致」。

    在真实 Kaggle 镜像上实测（图片/标签都是 780 个、排序也一致），原作写法恰好
    对齐；但只要数量不等就必然错配。用「normal 无掩码」的合成数据实测到的错配：

        train/images 有 10 张、train/labels 有 8 个
        labels 里有 malignant (0).txt，但它的图片 malignant (0).png 在 val
        val/images/malignant (0).png 配的却是 val/labels/malignant (3).txt

    而「normal 无掩码」恰是上游 README 对 BUSI 的描述（换镜像/数据集版本就可能
    成立），掩码转多边形失败、拷贝中断等也会让两边数量不等。后果是训练时图片配到
    别人的掩码、评估时真值和预测对不上，而且不报任何错。所以这里不依赖这种巧合。

    这里改为先把图片与标签按病例名(stem)归组，对病例名洗牌后再按比例切分，
    同一病例的图片与标签必然落在同一个 split。随机性仍由 seed=1337 固定可复现。

    keep_staging=False 时删除中间目录 data_（沿用原实现的清理行为）。
    """
    input_dir = Path(input_dir or STAGING_DIR)
    output_dir = Path(output_dir or config.SEGMENT_SPLIT_DIR)

    image_index = _index_by_stem(input_dir / "images")
    if not image_index:
        raise FileNotFoundError(
            f"找不到 {input_dir / 'images'}，请先运行 masks_to_polygons()。"
        )
    label_index = _index_by_stem(input_dir / "labels")

    # 病例集合以图片为准，同时纳入「只有标签没有图片」的病例，
    # 让多出来的标签跟着病例走，而不是留在原地造成错位。
    stems = sorted(set(image_index) | set(label_index))
    random.Random(seed).shuffle(stems)

    total = len(stems)
    n_train = int(total * ratio[0])
    n_val = int(total * ratio[1])
    buckets = {
        "train": stems[:n_train],
        "val": stems[n_train:n_train + n_val],
        "test": stems[n_train + n_val:],
    }

    _reset_dir(output_dir)
    stats = {}
    for split, names in buckets.items():
        images_dir = output_dir / split / "images"
        labels_dir = output_dir / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        image_count = 0
        label_count = 0
        for stem in names:
            for source in image_index.get(stem, []):
                shutil.copy2(source, images_dir / source.name)
                image_count += 1
            for source in label_index.get(stem, []):
                shutil.copy2(source, labels_dir / source.name)
                label_count += 1
        stats[split] = {"images": image_count, "labels": label_count}

    if not keep_staging:
        shutil.rmtree(input_dir, ignore_errors=True)

    return stats
