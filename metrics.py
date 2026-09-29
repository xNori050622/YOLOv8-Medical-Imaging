"""评估指标：分割 / 检测 / 分类，全部用 numpy + OpenCV 实现。

为什么不用现成库
----------------
* 原仓库没有任何评估代码——三个任务都只输出可视化图，没有任何量化数字；
* 医学分割的通用指标是 Dice 与 HD95，而 ultralytics 的 val() 只给 mAP，
  不会算 Dice / HD95 / 混淆矩阵；
* 本模块只依赖 numpy 与 OpenCV，不引入 torch / ultralytics / pandas，
  因此纯 CPU 环境与 GPU 环境都能跑，单元测试也不需要模型和数据集。

统计口径（会写进 README 的实验设置）
------------------------------------
* 二值掩码：非 0 即前景。
* Dice / IoU 在「预测与真值都为空」时记为 empty_value（默认 1.0）。
  这是多数医学分割评测的约定（BUSI 的 normal 类没有病灶，真值全空很常见），
  但它会抬高整体均值，所以 evaluate.py 同时报告「剔除空真值」后的 Dice。
* Precision / Recall 在分母为 0 时按 scikit-learn 惯例记 0.0。
* HD95 用像素距离（可用 spacing 换算成毫米），任一方为空时记无穷大。
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ==================================================================== 基础
def as_binary(mask) -> np.ndarray:
    """把任意输入转成 bool 前景掩码（非 0 即前景）。"""
    if mask is None:
        raise ValueError("mask 不能为 None")
    return np.asarray(mask).astype(bool)


def _check_same_shape(pred: np.ndarray, gt: np.ndarray) -> None:
    if pred.shape != gt.shape:
        raise ValueError(f"掩码尺寸不一致：pred{pred.shape} vs gt{gt.shape}")


def confusion_counts(pred, gt) -> Tuple[int, int, int]:
    """返回二值掩码的 (tp, fp, fn)：命中 / 误报 / 漏报的像素数。"""
    pred_b = as_binary(pred)
    gt_b = as_binary(gt)
    _check_same_shape(pred_b, gt_b)
    tp = int(np.count_nonzero(pred_b & gt_b))
    fp = int(np.count_nonzero(pred_b & ~gt_b))
    fn = int(np.count_nonzero(~pred_b & gt_b))
    return tp, fp, fn


def dice(pred, gt, empty_value: float = 1.0) -> float:
    """Dice 系数 = 2|P∩G| / (|P| + |G|)，取值 0~1，越大越好。

    pred 与 gt 都为空时返回 empty_value（见模块 docstring 的统计口径）。
    """
    tp, fp, fn = confusion_counts(pred, gt)
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return float(empty_value)
    return 2.0 * tp / denominator


def iou(pred, gt, empty_value: float = 1.0) -> float:
    """IoU（Jaccard）= |P∩G| / |P∪G|，取值 0~1，越大越好。"""
    tp, fp, fn = confusion_counts(pred, gt)
    denominator = tp + fp + fn
    if denominator == 0:
        return float(empty_value)
    return tp / denominator


def precision_recall(pred, gt, zero_division: float = 0.0) -> Tuple[float, float]:
    """像素级精确率与召回率；分母为 0 时返回 zero_division（与 sklearn 一致）。"""
    tp, fp, fn = confusion_counts(pred, gt)
    precision = tp / (tp + fp) if (tp + fp) else float(zero_division)
    recall = tp / (tp + fn) if (tp + fn) else float(zero_division)
    return precision, recall


def fbeta(precision: float, recall: float, beta: float = 1.0) -> float:
    """F-beta。precision 与 recall 同时为 0 时返回 0。"""
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    if denominator == 0:
        return 0.0
    return (1 + beta_squared) * precision * recall / denominator


def f1(precision: float, recall: float) -> float:
    """F1 = F-beta(beta=1)，精确率与召回率的调和平均。"""
    return fbeta(precision, recall, 1.0)


# ============================================================== HD95 距离
_SURFACE_KERNEL = np.ones((3, 3), np.uint8)


def surface(mask) -> np.ndarray:
    """取掩码的边界像素：3x3 腐蚀前后的差集。空掩码返回全 False。"""
    binary = as_binary(mask)
    if not binary.any():
        return np.zeros(binary.shape, bool)
    eroded = cv2.erode(binary.astype(np.uint8), _SURFACE_KERNEL).astype(bool)
    return binary & ~eroded


def _distance_to_surface(target: np.ndarray) -> Optional[np.ndarray]:
    """每个像素到 target 边界的最短欧氏距离；target 为空时返回 None。

    做法：把边界像素置 0、其余置 1 后做 distanceTransform，得到的就是
    「到最近边界」的距离场。这样只用 OpenCV，不依赖 scipy。

    必须用 DIST_MASK_PRECISE：maskSize=3 走的是近似 chamfer 权重
    （正交方向权重约 0.955，距离 4 会算出 3.82），会让 HD95 系统性偏小。
    """
    border = surface(target)
    if not border.any():
        return None
    source = np.ones(border.shape, np.uint8)
    source[border] = 0
    return cv2.distanceTransform(source, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)



def hd95(pred, gt, spacing: float = 1.0) -> float:
    """95% 豪斯多夫距离（HD95），单位像素；spacing 可换算成毫米。

    HD95 = 两侧边界互相最近距离集合的 95 分位数。
    预测与真值都为空时返回 0；只有一方为空时返回 inf（无法定义距离）。
    """
    pred_b = as_binary(pred)
    gt_b = as_binary(gt)
    _check_same_shape(pred_b, gt_b)

    if not pred_b.any() and not gt_b.any():
        return 0.0

    pred_to_gt = _distance_to_surface(gt_b)
    gt_to_pred = _distance_to_surface(pred_b)
    if pred_to_gt is None or gt_to_pred is None:
        return math.inf

    distances = np.concatenate([pred_to_gt[surface(pred_b)], gt_to_pred[surface(gt_b)]])
    if distances.size == 0:  # 理论上不会发生，保险起见
        return 0.0
    return float(np.percentile(distances, 95) * spacing)


# ================================================== YOLO 标签解析与栅格化
def _as_points(polygon, height: int, width: int) -> np.ndarray:
    """归一化多边形 -> 像素坐标点集 (N,2) int32。

    接受平铺的 [x1,y1,x2,y2,...]；不足 3 个点时返回空数组（无法围成面）。
    """
    flat = np.asarray(polygon, dtype=float).reshape(-1)
    if flat.size < 6 or flat.size % 2:
        return np.empty((0, 2), np.int32)
    xs = np.clip(np.round(flat[0::2] * width), 0, width - 1)
    ys = np.clip(np.round(flat[1::2] * height), 0, height - 1)
    return np.stack([xs, ys], axis=1).astype(np.int32)


def polygon_to_mask(polygons, height: int, width: int, value: int = 255) -> np.ndarray:
    """把（归一化的）多边形栅格化成 uint8 掩码，用于和模型输出的掩码比指标。"""
    canvas = np.zeros((height, width), np.uint8)
    for polygon in polygons:
        points = _as_points(polygon, height, width)
        if len(points) >= 3:
            cv2.fillPoly(canvas, [points], value)
    return canvas


def parse_segment_label(text: str, height: int, width: int) -> Dict[str, list]:
    """解析 YOLO 分割标签文本。

    每行形如：``class_id x1 y1 x2 y2 ... xn yn``（坐标已归一化到 0~1）。
    返回 {"class_ids": [...], "polygons": [...], "masks": [...]}，
    其中 masks 是已经栅格化好的 uint8 实例掩码。
    """
    class_ids: List[int] = []
    polygons: List[List[float]] = []
    masks: List[np.ndarray] = []

    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 7:  # 类别 + 至少 3 个点（6 个坐标）
            continue
        coordinates = [float(value) for value in parts[1:]]
        if len(coordinates) % 2:  # 坐标个数必须是偶数
            coordinates = coordinates[:-1]
        class_ids.append(int(float(parts[0])))
        polygons.append(coordinates)
        masks.append(polygon_to_mask([coordinates], height, width))

    return {"class_ids": class_ids, "polygons": polygons, "masks": masks}


def parse_detect_label(text: str, height: int, width: int) -> Dict[str, list]:
    """解析 YOLO 检测标签文本。

    每行形如：``class_id cx cy w h``（均为归一化值），转成像素坐标 xyxy。
    返回 {"class_ids": [...], "boxes": [[x1,y1,x2,y2], ...]}。
    """
    class_ids: List[int] = []
    boxes: List[List[float]] = []

    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_ids.append(int(float(parts[0])))
        center_x, center_y, box_w, box_h = (float(value) for value in parts[1:5])
        boxes.append([
            (center_x - box_w / 2) * width,
            (center_y - box_h / 2) * height,
            (center_x + box_w / 2) * width,
            (center_y + box_h / 2) * height,
        ])

    return {"class_ids": class_ids, "boxes": boxes}


def merge_by_class(instances, shape) -> Dict[int, np.ndarray]:
    """把 [(class_id, mask), ...] 按类别合并成 {class_id: 前景掩码}。

    语义式评估的预处理：同类别的多个实例并成一个前景（类别间互相独立）。
    """
    merged: Dict[int, np.ndarray] = {}
    for class_id, mask in instances:
        canvas = merged.get(class_id)
        if canvas is None:
            canvas = merged[class_id] = np.zeros(shape, bool)
        canvas |= as_binary(mask)
    return merged


# ============================================================ 实例匹配
def mask_iou_matrix(pred_masks: Sequence, gt_masks: Sequence) -> np.ndarray:
    """预测掩码 x 真值掩码 的 IoU 矩阵（P x G）。"""
    matrix = np.zeros((len(pred_masks), len(gt_masks)), float)
    if matrix.size == 0:
        return matrix
    gt_binary = [as_binary(gt) for gt in gt_masks]
    for row, pred in enumerate(pred_masks):
        pred_binary = as_binary(pred)
        for column, gt_binary_item in enumerate(gt_binary):
            intersection = int(np.count_nonzero(pred_binary & gt_binary_item))
            if intersection == 0:
                continue
            union = int(np.count_nonzero(pred_binary | gt_binary_item))
            matrix[row, column] = intersection / union if union else 0.0
    return matrix


def match_instances(pred_masks: Sequence, gt_masks: Sequence,
                    iou_threshold: float = 0.5) -> Dict[str, object]:
    """把预测实例与真值实例做贪心匹配（每轮取全局最大 IoU 的一对）。

    返回 tp / fp / fn，以及匹配对的平均 Dice 与平均 IoU。
    """
    matrix = mask_iou_matrix(pred_masks, gt_masks)
    pairs: List[Tuple[int, int, float]] = []

    while matrix.size:
        index = int(np.argmax(matrix))
        row, column = divmod(index, matrix.shape[1])
        best = matrix[row, column]
        if best < iou_threshold:
            break
        pairs.append((row, column, float(best)))
        matrix[row, :] = -1.0   # 该预测已配对
        matrix[:, column] = -1.0  # 该真值已配对

    dices = [dice(pred_masks[row], gt_masks[column]) for row, column, _ in pairs]
    tp = len(pairs)
    fp = len(pred_masks) - tp
    fn = len(gt_masks) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": fbeta(precision, recall),
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "mean_iou": float(np.mean([value for _, _, value in pairs])) if pairs else 0.0,
        "pairs": pairs,
    }


def per_class_mask_scores(pred_by_class: Dict[int, np.ndarray],
                          gt_by_class: Dict[int, np.ndarray],
                          class_names: Optional[Sequence[str]] = None,
                          empty_value: float = 1.0) -> Dict[int, dict]:
    """逐类算语义式 Dice / IoU / HD95（把该类所有实例并成一个前景）。

    gt_empty 标记该类别在这张图上真值是否为空——BUSI 的 normal 类就是这种情况，
    汇总时可以据此把「空真值」的样本单独统计，避免 Dice 被抬高。
    """
    records: Dict[int, dict] = {}
    for class_id in sorted(set(pred_by_class) | set(gt_by_class)):
        if class_id in gt_by_class:
            shape = np.asarray(gt_by_class[class_id]).shape
        else:
            shape = np.asarray(pred_by_class[class_id]).shape
        blank = np.zeros(shape, bool)
        pred = pred_by_class.get(class_id, blank)
        gt = gt_by_class.get(class_id, blank)

        records[class_id] = {
            "name": class_name(class_id, class_names),
            "dice": dice(pred, gt, empty_value),
            "iou": iou(pred, gt, empty_value),
            "hd95": hd95(pred, gt),
            "pixels_pred": int(as_binary(pred).sum()),
            "pixels_gt": int(as_binary(gt).sum()),
            "gt_empty": not as_binary(gt).any(),
        }
    return records


def class_name(class_id: int, class_names: Optional[Sequence[str]]) -> str:
    """类别索引 -> 名称；没有名称或越界时退化成字符串索引。"""
    if class_names and 0 <= class_id < len(class_names):
        return str(class_names[class_id])
    return str(class_id)


def mean_of(records, key: str, skip_inf: bool = True,
            skip_empty_gt: bool = False) -> float:
    """对一组记录求某字段的均值。

    skip_inf=True 时跳过 inf（例如 HD95 在一方为空时返回 inf）；
    skip_empty_gt=True 时跳过「空真值」样本，用于剔除 Dice 被抬高的影响。
    """
    values = []
    for record in records:
        if skip_empty_gt and record.get("gt_empty"):
            continue
        value = record.get(key)
        if value is None:
            continue
        value = float(value)
        if math.isnan(value):
            continue
        if skip_inf and math.isinf(value):
            continue
        values.append(value)
    return float(np.mean(values)) if values else 0.0


# ================================================================ 检测
def box_iou_matrix(boxes_a, boxes_b) -> np.ndarray:
    """两组 xyxy 框的 IoU 矩阵（A x B）。"""
    box_a = np.asarray(boxes_a, float).reshape(-1, 4)
    box_b = np.asarray(boxes_b, float).reshape(-1, 4)
    if box_a.size == 0 or box_b.size == 0:
        return np.zeros((len(box_a), len(box_b)), float)

    left_top = np.maximum(box_a[:, None, :2], box_b[None, :, :2])
    right_bottom = np.minimum(box_a[:, None, 2:], box_b[None, :, 2:])
    width_height = np.clip(right_bottom - left_top, 0.0, None)
    intersection = width_height[..., 0] * width_height[..., 1]

    area_a = np.clip(box_a[:, 2] - box_a[:, 0], 0.0, None) \
        * np.clip(box_a[:, 3] - box_a[:, 1], 0.0, None)
    area_b = np.clip(box_b[:, 2] - box_b[:, 0], 0.0, None) \
        * np.clip(box_b[:, 3] - box_b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - intersection

    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match_detections(pred_boxes, pred_scores, pred_classes,
                     gt_boxes, gt_classes,
                     iou_threshold: float = 0.5) -> Dict[str, object]:
    """按类别做贪心匹配（与 COCO 一致：每个预测找「尚未被占用」的最佳真值）。

    返回 tp / fp / fn 与布尔标记 tp_flags、gt_used，后者用于累计逐类指标。
    """
    pred_boxes = np.asarray(pred_boxes, float).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, float).reshape(-1)
    pred_classes = np.asarray(pred_classes, int).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, float).reshape(-1, 4)
    gt_classes = np.asarray(gt_classes, int).reshape(-1)

    tp_flags = np.zeros(len(pred_boxes), bool)
    gt_used = np.zeros(len(gt_boxes), bool)

    for class_id in sorted(set(pred_classes.tolist()) | set(gt_classes.tolist())):
        pred_index = np.where(pred_classes == class_id)[0]
        gt_index = np.where(gt_classes == class_id)[0]
        if pred_index.size == 0 or gt_index.size == 0:
            continue

        # 置信度从高到低处理
        order = pred_index[np.argsort(-pred_scores[pred_index], kind="stable")]
        ious = box_iou_matrix(pred_boxes[order], gt_boxes[gt_index])

        for row, pred_i in enumerate(order):
            for column in np.argsort(-ious[row], kind="stable"):
                if ious[row, column] < iou_threshold:
                    break  # 后面的只会更小
                if not gt_used[gt_index[column]]:
                    tp_flags[pred_i] = True
                    gt_used[gt_index[column]] = True
                    break

    tp = int(tp_flags.sum())
    return {
        "tp": tp,
        "fp": int(len(pred_boxes) - tp),
        "fn": int(len(gt_boxes) - tp),
        "tp_flags": tp_flags,
        "gt_used": gt_used,
    }


def average_precision(tp_flags, scores, num_gt: int) -> float:
    """单类别 AP（VOC 全点插值）。

    自己实现 AP 的目的不只是「有指标」：evaluate.py 会把它与 ultralytics
    的 box.map 对照，用来交叉验证两边口径是否一致。
    """
    scores = np.asarray(scores, float).reshape(-1)
    if num_gt <= 0 or scores.size == 0:
        return 0.0

    order = np.argsort(-scores, kind="stable")
    flags = np.asarray(tp_flags, bool).reshape(-1)[order]
    tp_cumulative = np.cumsum(flags).astype(float)
    fp_cumulative = np.cumsum(~flags).astype(float)

    recall = tp_cumulative / num_gt
    precision = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1e-12)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # 全点插值

    mrec = np.concatenate([[0.0], recall])
    mpre = np.concatenate([[0.0], precision])
    area = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            area += (mrec[index] - mrec[index - 1]) * mpre[index]
    return float(area)


def detection_report(pred_per_image: Sequence[dict], gt_per_image: Sequence[dict],
                     class_names: Optional[Sequence[str]] = None,
                     iou_threshold: float = 0.5) -> dict:
    """批量检测指标：逐类 P/R/F1/AP，加上宏平均与 mAP@iou_threshold。

    每张图传两个 dict：
      pred: {"boxes": [[x1,y1,x2,y2], ...], "scores": [...], "classes": [...]}
      gt:   {"boxes": [...], "classes": [...]}
    """
    flags_all: List[np.ndarray] = []
    scores_all: List[np.ndarray] = []
    classes_all: List[np.ndarray] = []
    num_gt: Dict[int, int] = defaultdict(int)
    tp_count: Dict[int, int] = defaultdict(int)
    fp_count: Dict[int, int] = defaultdict(int)
    fn_count: Dict[int, int] = defaultdict(int)

    for pred, gt in zip(pred_per_image, gt_per_image):
        classes = np.asarray(pred["classes"], int).reshape(-1)
        scores = np.asarray(pred["scores"], float).reshape(-1)
        gt_classes = np.asarray(gt["classes"], int).reshape(-1)

        matched = match_detections(pred["boxes"], scores, classes,
                                   gt["boxes"], gt_classes, iou_threshold)
        flags = matched["tp_flags"]

        flags_all.append(flags)
        scores_all.append(scores)
        classes_all.append(classes)

        for class_id in gt_classes.tolist():
            num_gt[class_id] += 1
        for class_id in sorted(set(classes.tolist())):
            selected = classes == class_id
            tp_count[class_id] += int(flags[selected].sum())
            fp_count[class_id] += int(selected.sum() - flags[selected].sum())
        for class_id in sorted(set(gt_classes.tolist())):
            selected = gt_classes == class_id
            fn_count[class_id] += int(selected.sum() - matched["gt_used"][selected].sum())

    flags = np.concatenate(flags_all) if flags_all else np.zeros(0, bool)
    scores = np.concatenate(scores_all) if scores_all else np.zeros(0, float)
    classes = np.concatenate(classes_all) if classes_all else np.zeros(0, int)

    per_class: Dict[int, dict] = {}
    for class_id in sorted(set(classes.tolist()) | set(num_gt)):
        selected = classes == class_id
        tp = tp_count[class_id]
        fp = fp_count[class_id]
        fn = fn_count[class_id]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        per_class[class_id] = {
            "name": class_name(class_id, class_names),
            "tp": tp, "fp": fp, "fn": fn,
            "num_gt": num_gt.get(class_id, 0),
            "precision": precision,
            "recall": recall,
            "f1": fbeta(precision, recall),
            "ap": average_precision(flags[selected], scores[selected], num_gt.get(class_id, 0)),
        }

    records = list(per_class.values())
    with_gt = [record for record in records if record["num_gt"] > 0]
    return {
        "per_class": per_class,
        "iou_threshold": iou_threshold,
        "macro": {
            "precision": mean_of(records, "precision"),
            "recall": mean_of(records, "recall"),
            "f1": mean_of(records, "f1"),
        },
        "mAP": mean_of(with_gt, "ap"),
    }


# ================================================================ 分类
def confusion_matrix(y_true, y_pred, num_classes: Optional[int] = None) -> np.ndarray:
    """混淆矩阵 cm[i, j] = 真实类别 i 被预测成 j 的样本数。"""
    y_true = np.asarray(y_true, int).reshape(-1)
    y_pred = np.asarray(y_pred, int).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true 与 y_pred 长度不一致")

    if num_classes is None:
        num_classes = int(max(y_true.max(initial=-1), y_pred.max(initial=-1))) + 1
    num_classes = max(int(num_classes), 0)

    matrix = np.zeros((num_classes, num_classes), int)
    for true, pred in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= true < num_classes and 0 <= pred < num_classes:
            matrix[true, pred] += 1
    return matrix


def classification_report(y_true, y_pred, class_names: Optional[Sequence[str]] = None,
                          num_classes: Optional[int] = None) -> dict:
    """分类指标：逐类 P/R/F1/support + accuracy + macro-F1 + 混淆矩阵。

    macro 平均只统计 support>0 的类别，避免未出现的类别把均值拉低。
    """
    y_true = np.asarray(y_true, int).reshape(-1)
    y_pred = np.asarray(y_pred, int).reshape(-1)
    if num_classes is None:
        num_classes = max(len(class_names) if class_names else 0,
                          int(max(y_true.max(initial=-1), y_pred.max(initial=-1))) + 1)

    matrix = confusion_matrix(y_true, y_pred, num_classes)
    total = int(matrix.sum())

    per_class: Dict[int, dict] = {}
    for class_id in range(num_classes):
        tp = int(matrix[class_id, class_id])
        fp = int(matrix[:, class_id].sum() - tp)
        fn = int(matrix[class_id, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        per_class[class_id] = {
            "name": class_name(class_id, class_names),
            "tp": tp, "fp": fp, "fn": fn,
            "support": tp + fn,
            "precision": precision,
            "recall": recall,
            "f1": fbeta(precision, recall),
        }

    present = [record for record in per_class.values() if record["support"] > 0]
    return {
        "per_class": per_class,
        "confusion_matrix": matrix,
        "num_samples": total,
        "accuracy": float(np.trace(matrix) / total) if total else 0.0,
        "macro_precision": mean_of(present, "precision"),
        "macro_recall": mean_of(present, "recall"),
        "macro_f1": mean_of(present, "f1"),
    }


# ================================================================= 自演示
def _demo() -> None:
    """python metrics.py —— 用合成掩码/框/类别跑一遍，验证指标口径。"""
    gt = np.zeros((8, 8), np.uint8)
    gt[2:6, 2:6] = 1          # 4x4 方块，面积 16
    pred = np.zeros((8, 8), np.uint8)
    pred[3:7, 3:7] = 1        # 右下平移 1 像素：交集 3x3=9

    print("== 掩码指标（合成数据，交集 9 像素，各面积 16）==")
    tp, fp, fn = confusion_counts(pred, gt)
    print(f"  tp={tp} fp={fp} fn={fn}")
    print(f"  dice = {dice(pred, gt):.4f}   (期望 18/32 = 0.5625)")
    print(f"  iou  = {iou(pred, gt):.4f}   (期望 9/23 = 0.3913)")
    print(f"  precision/recall = {precision_recall(pred, gt)}")

    single_a = np.zeros((8, 8), np.uint8); single_a[0, 0] = 1
    single_b = np.zeros((8, 8), np.uint8); single_b[0, 4] = 1
    print(f"  hd95(相距 4 像素的两个点) = {hd95(single_a, single_b):.1f}  (期望 4.0)")
    print(f"  hd95(完全相同)           = {hd95(gt, gt):.1f}  (期望 0.0)")

    print("\n== 检测指标（1 个真值，预测框完全重合 + 1 个误报）==")
    report = detection_report(
        [{"boxes": [[0, 0, 10, 10], [50, 50, 60, 60]], "scores": [0.9, 0.4], "classes": [0, 0]}],
        [{"boxes": [[0, 0, 10, 10]], "classes": [0]}],
        class_names=["RBC"],
    )
    print(f"  tp={report['per_class'][0]['tp']} fp={report['per_class'][0]['fp']} "
          f"fn={report['per_class'][0]['fn']}  P={report['per_class'][0]['precision']:.2f} "
          f"R={report['per_class'][0]['recall']:.2f}  mAP={report['mAP']:.2f}")

    print("\n== 分类指标（4 样本，1 个错分）==")
    result = classification_report([0, 1, 2, 2], [0, 2, 2, 2],
                                   class_names=["Covid", "Normal", "Viral Pneumonia"])
    print(f"  accuracy={result['accuracy']:.2f} macro-F1={result['macro_f1']:.2f}")
    print(f"  confusion matrix:\n{result['confusion_matrix']}")


if __name__ == "__main__":
    _demo()

