"""模型评估入口：在留有真值标签的 split 上跑一遍，输出量化指标。

为什么需要这个脚本
------------------
原仓库没有任何评估代码，三个任务都只把标注图画出来给人看，无法回答
「模型到底多准」。本脚本补齐这一环：拿 runs/<task>/<name>/weights/best.pt
在带标签的 split 上做推理，用 metrics.py 里的指标算出数字。

用法
----
    python evaluate.py segment  --split val                 # 分割：Dice/IoU/HD95/实例 P-R
    python evaluate.py detect   --split valid               # 检测：逐类 P/R/F1/AP/mAP@0.5
    python evaluate.py classify --split test                # 分类：混淆矩阵/accuracy/macro-F1

    python evaluate.py segment --name baseline --save experiments/baseline.json
    python evaluate.py segment --data D:\\tmp\\smoke\\segmentation\\data.yaml --limit 5

约定
----
* --data 与 train.py 一致：detect/segment 传 data.yaml，classify 传数据集目录。
* --split 对检测同时接受 val 与 valid（Roboflow 导出用 valid）。
* 阈值 --conf 只影响预测（默认 0.25，比界面默认的 0.35 低，便于看召回）。
* 分割的 Dice/IoU 是「语义式」逐类统计；实例 P/R/F1 是逐类贪心匹配（IoU>=0.5）。
* BUSI 的 normal 类真值全空，会把 Dice 抬到 1.0，所以同时报告剔除空真值后的值。
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config
import dataset
import metrics

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ------------------------------------------------------------------ 工具
def _images(directory: Path, limit=None) -> list:
    """列出目录下的图片（按名字排序）。"""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    files = sorted(item for item in directory.iterdir()
                   if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    return files[:limit] if limit else files


def _read_image(path: Path):
    """读图，返回 BGR 数组；失败返回 None。"""
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _weights_path(task: str, name: str, explicit=None) -> Path:
    """权重路径：显式给就用显式的，否则用 runs/<task>/<name>/weights/best.pt。"""
    return Path(explicit) if explicit else config.RUNS_DIR / task / name / "weights" / "best.pt"


def _json_default(value):
    """把 numpy 标量/数组转成可 JSON 序列化的类型。"""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(payload, path: Path) -> Path:
    """把评估结果写成 JSON（供后续做实验对比表）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               default=_json_default), encoding="utf-8")
    return path


# ------------------------------------------------------- 数据集位置解析
def _split_entry(payload: dict, split: str):
    """从 data.yaml 里找出某个 split 的相对路径（兼容 val / valid 两种写法）。"""
    candidates = {
        "train": ("train",),
        "val": ("val", "valid"),
        "valid": ("valid", "val"),
        "test": ("test",),
    }.get(split, (split,))
    for key in candidates:
        if payload.get(key):
            return payload[key]
    return None


def resolve_split(task: str, data, split: str):
    """返回 (images_dir, labels_dir)；分类任务没有 labels_dir（返回 None）。"""
    if task == "classify":
        root = Path(data) if data else config.CLASSIFY_DATA_DIR
        images_dir = root / split
        if not images_dir.is_dir():
            raise FileNotFoundError(f"找不到分类 {split} 目录：{images_dir}")
        return images_dir, None

    yaml_path = Path(data) if data else (
        config.DETECT_DATA_YAML if task == "detect" else config.SEGMENT_DATA_YAML)
    if task == "detect" and not yaml_path.is_file():
        # 与 train.py 的 resolve_data 保持一致：只有 images/labels 时自动补生成
        try:
            yaml_path = dataset.build_detect_data_yaml(yaml_path.parent)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"找不到 {task} 的 data.yaml：{yaml_path}，且无法自动生成：{error}"
            ) from error
    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"找不到 {task} 的 data.yaml：{yaml_path}\n"
            "请先准备数据（分割需要 python train.py segment --prepare）。"
        )

    payload = dataset.read_data_yaml(yaml_path)
    entry = _split_entry(payload, split)
    if not entry:
        raise FileNotFoundError(f"{yaml_path} 里没有 {split} 这一项，可用的键：{list(payload)}")

    root = Path(payload.get("path") or yaml_path.parent)
    images_dir = root / entry
    labels_dir = Path(entry).parent / "labels"
    labels_dir = root / labels_dir
    return images_dir, labels_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="YOLOv8-Medical-Imaging 模型评估入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("task", choices=["detect", "classify", "segment"], help="要评估的任务")
    parser.add_argument("--split", default=None, help="评估用的 split，默认 val/valid")
    parser.add_argument("--name", default="train", help="runs/<task>/<name>/weights/best.pt")
    parser.add_argument("--weights", default=None, help="直接指定权重路径，优先于 --name")
    parser.add_argument("--data", default=None,
                        help="覆盖数据集位置：detect/segment 传 data.yaml，classify 传目录")
    parser.add_argument("--imgsz", type=int, default=None, help="推理尺寸，默认用 config 里的值")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值，默认 0.25")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 张图（冒烟用）")
    parser.add_argument("--device", default="auto", help="auto / 0 / cpu")
    parser.add_argument("--save", default=None, help="把结果写成 JSON 的路径")
    parser.add_argument("--cross-check", action="store_true",
                        help="同时跑 ultralytics 官方 val()，用来对照我们自己算的 mAP")
    return parser


# =============================================================== 评估实现
def load_model(weights: Path):
    """加载 YOLO 模型（延迟导入 ultralytics，避免 import 本模块就拉起 torch）。"""
    if not Path(weights).is_file():
        raise FileNotFoundError(
            f"找不到权重：{weights}\n"
            "请先训练（python train.py <task>），或用 --weights/--name 指定。"
        )
    from ultralytics import YOLO

    return YOLO(str(weights))


def _class_mean(records, key: str):
    """逐类明细的均值；全部为 inf/None 时返回 None（而不是 0，避免误读为「完美」）。"""
    values = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        value = float(value)
        if math.isinf(value) or math.isnan(value):
            continue
        values.append(value)
    return float(np.mean(values)) if values else None


def _mean_or_none(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def evaluate_segment(model, images_dir, labels_dir, conf, imgsz, device, limit=None) -> dict:
    """分割评估。

    * 语义式：同类实例并成一个前景，逐类算 Dice / IoU / HD95；
    * 实例式：同类内贪心匹配（IoU>=0.5），算 P / R / F1 与命中实例的平均 Dice。
    """
    files = _images(images_dir, limit)
    if not files:
        raise FileNotFoundError(f"没有找到图片：{images_dir}")

    names = model.names
    class_names = [names[index] for index in sorted(names)] if isinstance(names, dict) \
        else list(names)

    per_class_records = defaultdict(list)
    instance_totals = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "dices": []})
    processed = 0
    skipped = 0

    for path in files:
        image = _read_image(path)
        if image is None:
            skipped += 1
            continue
        processed += 1
        height, width = image.shape[:2]

        # ---- 真值：YOLO 多边形标签 -> 实例掩码 ----
        label_path = Path(labels_dir) / f"{path.stem}.txt" if labels_dir else None
        if label_path and label_path.is_file():
            parsed = metrics.parse_segment_label(
                label_path.read_text(encoding="utf-8", errors="ignore"), height, width)
        else:
            parsed = {"class_ids": [], "masks": []}  # 没有标签 = 纯背景（BUSI 的 normal 类）

        gt_instances = defaultdict(list)
        for class_id, mask in zip(parsed["class_ids"], parsed["masks"]):
            gt_instances[class_id].append(mask)

        # ---- 预测 ----
        result = model.predict(image, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        pred_instances = defaultdict(list)
        if result.masks is not None:
            boxes = result.boxes
            for index, one in enumerate(result.masks):
                raw = one.data[0].cpu().numpy()
                # 用最近邻缩放，避免插值把二值掩码糊成灰边
                mask = cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST) > 0.5
                class_id = int(boxes.cls[index]) if boxes is not None else -1
                pred_instances[class_id].append(mask)

        # ---- 语义式逐类指标 ----
        pred_by_class = {class_id: np.logical_or.reduce(masks)
                         for class_id, masks in pred_instances.items()}
        gt_by_class = {class_id: np.logical_or.reduce(masks)
                       for class_id, masks in gt_instances.items()}
        for class_id, record in metrics.per_class_mask_scores(
                pred_by_class, gt_by_class, class_names).items():
            per_class_records[class_id].append(record)

        # ---- 实例式逐类匹配 ----
        for class_id in sorted(set(pred_instances) | set(gt_instances)):
            matched = metrics.match_instances(pred_instances.get(class_id, []),
                                              gt_instances.get(class_id, []))
            totals = instance_totals[class_id]
            totals["tp"] += matched["tp"]
            totals["fp"] += matched["fp"]
            totals["fn"] += matched["fn"]
            if matched["tp"]:  # 只有存在命中实例时，平均 Dice 才有意义
                totals["dices"].append(matched["mean_dice"])

    class_summary = {}
    for class_id, records in sorted(per_class_records.items()):
        totals = instance_totals.get(class_id, {"tp": 0, "fp": 0, "fn": 0, "dices": []})
        precision = totals["tp"] / (totals["tp"] + totals["fp"]) if (totals["tp"] + totals["fp"]) else 0.0
        recall = totals["tp"] / (totals["tp"] + totals["fn"]) if (totals["tp"] + totals["fn"]) else 0.0
        class_summary[class_id] = {
            "name": metrics.class_name(class_id, class_names),
            "images_with_record": len(records),
            "dice": metrics.mean_of(records, "dice"),
            "dice_nonempty": metrics.mean_of(records, "dice", skip_empty_gt=True),
            "iou": metrics.mean_of(records, "iou"),
            "hd95": _class_mean(records, "hd95"),
            "instances_tp": totals["tp"],
            "instances_fp": totals["fp"],
            "instances_fn": totals["fn"],
            "instance_precision": precision,
            "instance_recall": recall,
            "instance_f1": metrics.fbeta(precision, recall),
            "mean_matched_dice": float(np.mean(totals["dices"])) if totals["dices"] else 0.0,
        }

    records_list = list(class_summary.values())
    return {
        "task": "segment",
        "num_images": processed,
        "skipped_images": skipped,
        "per_class": class_summary,
        "macro": {
            "dice": metrics.mean_of(records_list, "dice"),
            "dice_nonempty": metrics.mean_of(records_list, "dice_nonempty"),
            "iou": metrics.mean_of(records_list, "iou"),
            "hd95": _mean_or_none([record["hd95"] for record in records_list]),
            "instance_f1": metrics.mean_of(records_list, "instance_f1"),
        },
    }


def evaluate_detect(model, images_dir, labels_dir, conf, imgsz, device, limit=None) -> dict:
    """检测评估：逐类 P/R/F1/AP，以及整体 mAP@0.5（自实现，可与官方 val() 对照）。"""
    files = _images(images_dir, limit)
    if not files:
        raise FileNotFoundError(f"没有找到图片：{images_dir}")

    names = model.names
    class_names = [names[index] for index in sorted(names)] if isinstance(names, dict) \
        else list(names)

    pred_per_image, gt_per_image = [], []
    processed = 0
    skipped = 0

    for path in files:
        image = _read_image(path)
        if image is None:
            skipped += 1
            continue
        processed += 1
        height, width = image.shape[:2]

        label_path = Path(labels_dir) / f"{path.stem}.txt" if labels_dir else None
        if label_path and label_path.is_file():
            parsed = metrics.parse_detect_label(
                label_path.read_text(encoding="utf-8", errors="ignore"), height, width)
        else:
            parsed = {"class_ids": [], "boxes": []}  # 没有标签 = 该图无目标

        result = model.predict(image, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        boxes, scores, classes = [], [], []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                boxes.append([x1, y1, x2, y2])
                scores.append(float(box.conf[0]))
                classes.append(int(box.cls[0]))

        pred_per_image.append({"boxes": boxes, "scores": scores, "classes": classes})
        gt_per_image.append({"boxes": parsed["boxes"], "classes": parsed["class_ids"]})

    report = metrics.detection_report(pred_per_image, gt_per_image, class_names)
    report["task"] = "detect"
    report["num_images"] = processed
    report["skipped_images"] = skipped
    return report


def evaluate_classify(model, split_dir, imgsz, device, limit=None) -> dict:
    """分类评估：每个子目录名就是真值，逐张推理取 top1。

    真值不是按目录排序决定的，而是拿目录名去匹配模型自带的类别名，
    这样即使数据集目录顺序与权重里的顺序不一致也不会算错。
    """
    names = model.names
    class_names = [names[index] for index in sorted(names)] if isinstance(names, dict) \
        else list(names)
    folder_to_index = {name: index for index, name in enumerate(class_names)}

    y_true, y_pred = [], []
    per_folder = {}
    unknown_folders = []
    processed = 0
    skipped = 0

    split_dir = Path(split_dir)
    for folder in sorted(item for item in split_dir.iterdir() if item.is_dir()):
        if folder.name not in folder_to_index:
            unknown_folders.append(folder.name)
            continue
        truth = folder_to_index[folder.name]
        files = _images(folder, limit)
        correct = 0
        for path in files:
            image = _read_image(path)
            if image is None:
                skipped += 1
                continue
            result = model.predict(image, imgsz=imgsz, device=device, verbose=False)[0]
            if result.probs is None:
                skipped += 1
                continue
            predicted = int(np.argmax(result.probs.data.cpu().numpy()))
            processed += 1
            y_true.append(truth)
            y_pred.append(predicted)
            correct += int(predicted == truth)
        per_folder[folder.name] = {"images": len(files), "correct": correct}

    if not y_true:
        raise FileNotFoundError(
            f"在 {split_dir} 下没有评估到任何图片。"
            f"已知类别目录：{list(folder_to_index)}；"
            f"实际子目录：{sorted(item.name for item in split_dir.iterdir() if item.is_dir())}"
        )

    report = metrics.classification_report(y_true, y_pred, class_names)
    report["task"] = "classify"
    report["num_images"] = processed
    report["skipped_images"] = skipped
    report["per_folder"] = per_folder
    report["unknown_folders"] = unknown_folders
    return report


# =============================================================== 结果打印
def _title(text: str) -> None:
    print("=" * 72)
    print(text)
    print("=" * 72)


def _number(value, digits=4) -> str:
    """None 打成人能看懂的 n/a（HD95 全为空时会出现 None）。"""
    return "n/a" if value is None else f"{value:.{digits}f}"


def print_segment(result: dict) -> None:
    _title(f"分割评估  |  图片 {result['num_images']} 张"
           f"  |  跳过 {result['skipped_images']}")
    print(f"{'类别':<16}{'Dice':>9}{'Dice*':>9}{'IoU':>9}{'HD95':>9}"
          f"{'实例P':>9}{'实例R':>9}{'实例F1':>9}{'TP/FP/FN':>14}")
    print("-" * 72)
    for record in result["per_class"].values():
        print(f"{record['name'][:15]:<16}{record['dice']:>9.4f}"
              f"{record['dice_nonempty']:>9.4f}{record['iou']:>9.4f}"
              f"{_number(record['hd95'], 2):>9}"
              f"{record['instance_precision']:>9.4f}{record['instance_recall']:>9.4f}"
              f"{record['instance_f1']:>9.4f}"
              f"{record['instances_tp']:>5d}/{record['instances_fp']:d}/{record['instances_fn']:d}")
    macro = result["macro"]
    print("-" * 72)
    print(f"{'宏平均':<16}{macro['dice']:>9.4f}{macro['dice_nonempty']:>9.4f}"
          f"{macro['iou']:>9.4f}{_number(macro['hd95'], 2):>9}"
          f"{'':>9}{'':>9}{macro['instance_f1']:>9.4f}")
    print("\nDice* 说明：剔除「真值全空」样本后的 Dice。BUSI 的 normal 类没有病灶，"
          "空真值按约定记 Dice=1，\n会把均值抬高，所以两个数要一起看。HD95 单位像素。")


def print_detect(result: dict) -> None:
    _title(f"检测评估  |  图片 {result['num_images']} 张"
           f"  |  IoU 阈值 {result['iou_threshold']}")
    print(f"{'类别':<16}{'TP':>7}{'FP':>7}{'FN':>7}{'GT':>7}"
          f"{'精确率':>10}{'召回率':>10}{'F1':>10}{'AP':>10}")
    print("-" * 72)
    for record in result["per_class"].values():
        print(f"{record['name'][:15]:<16}{record['tp']:>7d}{record['fp']:>7d}"
              f"{record['fn']:>7d}{record['num_gt']:>7d}"
              f"{record['precision']:>10.4f}{record['recall']:>10.4f}"
              f"{record['f1']:>10.4f}{record['ap']:>10.4f}")
    macro = result["macro"]
    print("-" * 72)
    print(f"{'宏平均':<16}{'':>7}{'':>7}{'':>7}{'':>7}"
          f"{macro['precision']:>10.4f}{macro['recall']:>10.4f}{macro['f1']:>10.4f}"
          f"{result['mAP']:>10.4f}")
    print(f"\nmAP@0.5 = {result['mAP']:.4f}（本脚本自实现；"
          "加 --cross-check 可与 ultralytics 官方 val() 对照）")


def print_classify(result: dict) -> None:
    _title(f"分类评估  |  图片 {result['num_images']} 张"
           f"  |  跳过 {result['skipped_images']}")
    print(f"{'类别':<22}{'精确率':>10}{'召回率':>10}{'F1':>10}{'样本数':>10}")
    print("-" * 72)
    for record in result["per_class"].values():
        print(f"{record['name'][:21]:<22}{record['precision']:>10.4f}"
              f"{record['recall']:>10.4f}{record['f1']:>10.4f}{record['support']:>10d}")

    print("-" * 72)
    print(f"{'宏平均':<22}{result['macro_precision']:>10.4f}"
          f"{result['macro_recall']:>10.4f}{result['macro_f1']:>10.4f}"
          f"{result['num_samples']:>10d}")
    print(f"\naccuracy = {result['accuracy']:.4f}   macro-F1 = {result['macro_f1']:.4f}")

    names = [record["name"] for record in result["per_class"].values()]
    matrix = result["confusion_matrix"]
    print("\n混淆矩阵（行 = 真实，列 = 预测）：")
    print(" " * 20 + "".join(f"{name[:10]:>12}" for name in names))
    for index, name in enumerate(names):
        cells = "".join(f"{int(value):>12d}" for value in matrix[index])
        print(f"{name[:18]:<20}{cells}")

    if result.get("unknown_folders"):
        print(f"\n!! 以下目录不在模型类别里，已跳过：{result['unknown_folders']}")


def print_result(task: str, result: dict) -> None:
    {"segment": print_segment, "detect": print_detect, "classify": print_classify}[task](result)


# ============================================================= 官方 val 对照
def cross_check(model, task: str, data, split: str, imgsz: int, device) -> dict:
    """跑 ultralytics 官方 val()，与自实现指标对照。

    两者口径不同（官方 mAP 用 COCO 的 101 点插值，我们用的是 VOC 全点插值），
    数值不会完全相等，但量级应当一致——这就是「自实现 mAP 没写错」的证据。

    split 的坑：官方 check_det_dataset() 会把 split 当作 data.yaml 的键去查
    （ultralytics/data/utils.py 里的 `if split and not data.get(split)`），
    而 yaml 里写的键是 val。冒烟数据集的目录叫 valid/，如果直接把目录名 "valid"
    传进去，官方会报 “'valid:' images not found” 而跳过——所以这里统一映射成 val。
    分类则不同，官方是按 train/val/test 目录名去找的，原样传。

    还有一处口径差异：官方算 AP 时用的置信度阈值很低（让 PR 曲线尽量完整），
    而自实现默认按 --conf 过滤。要和官方严格对齐，请把 --conf 也降到 0.001，
    否则会出现「官方 mAP 非零、我们为 0」的假差异。

    另一个必踩的坑：官方 val() 在 data 参数为空时，会退回 checkpoint 里存的训练数据
    路径（权重同级 args.yaml 的 data 项）。仓库里现成的权重是上游在 Colab 上训的，
    存的是 /content/gdrive/... ，本机不存在——所以 detect/segment 必须显式传 data，
    不能指望官方自己找到数据集。
    """
    arguments = {"imgsz": imgsz, "device": device, "verbose": False,
                 # 官方 val() 默认写 runs/<task>/val，跑多次会堆出 val-2/val-3…
                 # 这里统一落到 runs/cross_check_<task>/ 并覆盖，免得污染训练产物目录。
                 "project": str(config.PROJECT_ROOT / "runs"),
                 "name": f"cross_check_{task}", "exist_ok": True}
    if task == "classify":
        arguments["split"] = split
    else:
        arguments["split"] = "val" if split in ("val", "valid") else split
    if data:
        arguments["data"] = str(data)
    elif task == "classify":
        arguments["data"] = str(config.CLASSIFY_DATA_DIR)
    else:
        # 官方 val() 在 data 为空时会退回 checkpoint 里存的训练数据路径
        # （model.overrides["data"]，即权重同级目录 args.yaml 里的 data 项）。
        # 仓库里现成的权重是上游在 Colab 上训的，存的是
        # /content/gdrive/MyDrive/YOLO8/data/google_colab_config.yaml，本机不存在，
        # 于是官方 val 直接 FileNotFoundError。所以这里必须显式补上 config 的默认
        # yaml，与 resolve_split 的兜底逻辑保持一致。
        arguments["data"] = str(
            config.DETECT_DATA_YAML if task == "detect" else config.SEGMENT_DATA_YAML)

    try:
        validator = model.val(**arguments)
    except Exception as error:  # 官方 val 失败不应影响自实现指标
        return {"error": f"{type(error).__name__}: {error}"}

    payload = {}
    if getattr(validator, "box", None) is not None:
        payload["box_mAP50"] = float(validator.box.map50)
        payload["box_mAP50_95"] = float(validator.box.map)
    if getattr(validator, "seg", None) is not None:
        payload["seg_mAP50"] = float(validator.seg.map50)
        payload["seg_mAP50_95"] = float(validator.seg.map)
    if getattr(validator, "top1", None) is not None:
        payload["top1"] = float(validator.top1)
        payload["top5"] = float(validator.top5)
    return payload


# ==================================================================== 入口
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    task = args.task
    split = args.split or ("test" if task == "classify" else "val")
    weights = _weights_path(task, args.name, args.weights)
    imgsz = args.imgsz or config.DEFAULT_IMGSZ[task]
    device = config.pick_device(args.device)

    images_dir, labels_dir = resolve_split(task, args.data, split)
    model = load_model(weights)

    print(f"[evaluate] task={task} split={split}")
    print(f"[evaluate] weights : {weights}")
    print(f"[evaluate] data    : {images_dir}")
    print(f"[evaluate] imgsz={imgsz} conf={args.conf} device={device}\n")

    if task == "segment":
        result = evaluate_segment(model, images_dir, labels_dir, args.conf, imgsz, device, args.limit)
    elif task == "detect":
        result = evaluate_detect(model, images_dir, labels_dir, args.conf, imgsz, device, args.limit)
    else:
        result = evaluate_classify(model, images_dir, imgsz, device, args.limit)

    result["weights"] = str(weights)
    result["split"] = split
    result["conf"] = args.conf
    result["imgsz"] = imgsz
    print_result(task, result)

    if args.cross_check:
        check = cross_check(model, task, args.data, split, imgsz, device)
        result["ultralytics_val"] = check
        print("\n[交叉验证] ultralytics 官方 val()：")
        if "error" in check:
            print(f"  跳过（不影响上面的自实现指标）：{check['error']}")
        else:
            for key, value in check.items():
                print(f"  {key} = {value:.4f}")

    if args.save:
        print(f"\n结果已写入：{write_json(result, args.save)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

