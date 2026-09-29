"""metrics.py 的单元测试。

用法（两种都行）：
    python tests/test_metrics.py      # 零依赖，直接跑
    pytest tests/test_metrics.py      # 装了 pytest 也可以

设计原则：所有期望值都是能拿纸笔算出来的解析解，而不是"跑一遍看看输出"。
例如「两个 4x4 方块错开 1 像素」交集恒为 9 像素、Dice 恒为 18/32=0.5625。
这样即便没有数据集、没有 GPU、没有 Torch，也能确认指标实现本身是对的。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # 支持 python tests/test_metrics.py 直接运行
    sys.path.insert(0, str(PROJECT_ROOT))

import metrics


# ------------------------------------------------------------------ 小工具
def _square(size=8, row=2, column=2, side=4, value=1) -> np.ndarray:
    """在 size x size 的画布上画一个方块掩码。"""
    canvas = np.zeros((size, size), np.uint8)
    canvas[row:row + side, column:column + side] = value
    return canvas


def _shapes_are_close(actual: float, expected: float, tolerance=1e-6) -> None:
    assert abs(actual - expected) < tolerance, f"期望 {expected}，实际 {actual}"


# ============================================================ 掩码基础指标
def test_confusion_counts_and_dice_analytic():
    """错开 1 像素的两个 4x4 方块：tp=9, fp=7, fn=7。"""
    gt = _square(row=2, column=2)          # 覆盖行 2-5、列 2-5
    pred = _square(row=3, column=3)        # 覆盖行 3-6、列 3-6
    tp, fp, fn = metrics.confusion_counts(pred, gt)
    assert (tp, fp, fn) == (9, 7, 7), f"实际 {(tp, fp, fn)}"


def test_dice_and_iou_analytic():
    gt = _square(row=2, column=2)
    pred = _square(row=3, column=3)
    _shapes_are_close(metrics.dice(pred, gt), 18 / 32)   # 2*9 / (16+16)
    _shapes_are_close(metrics.iou(pred, gt), 9 / 23)     # 9 / (16+16-9)


def test_dice_and_iou_identical_and_disjoint():
    gt = _square(row=2, column=2)
    assert metrics.dice(gt, gt) == 1.0
    assert metrics.iou(gt, gt) == 1.0

    disjoint = np.zeros((8, 8), np.uint8)
    disjoint[0, 7] = 1                            # 与 gt 完全不相交
    assert metrics.dice(disjoint, gt) == 0.0
    assert metrics.iou(disjoint, gt) == 0.0


def test_empty_mask_convention():
    """两边都空：Dice/IoU 记为 1.0（医学评测约定），P/R 按 sklearn 惯例记 0。"""
    empty = np.zeros((8, 8), np.uint8)
    assert metrics.dice(empty, empty) == 1.0
    assert metrics.iou(empty, empty) == 1.0
    assert metrics.precision_recall(empty, empty) == (0.0, 0.0)
    assert metrics.hd95(empty, empty) == 0.0

    # empty_value 可配置
    assert metrics.dice(empty, empty, empty_value=0.0) == 0.0


def test_one_side_empty():
    empty = np.zeros((8, 8), np.uint8)
    gt = _square(row=2, column=2)
    assert metrics.dice(empty, gt) == 0.0        # 全漏报
    assert metrics.iou(empty, gt) == 0.0
    assert metrics.precision_recall(empty, gt) == (0.0, 0.0)
    assert math.isinf(metrics.hd95(empty, gt))   # 只有一方为空 -> 距离无定义

    assert metrics.dice(gt, empty) == 0.0        # 全误报
    assert math.isinf(metrics.hd95(gt, empty))


def test_precision_recall_and_fbeta():
    gt = _square(row=2, column=2)
    pred = _square(row=3, column=3)
    precision, recall = metrics.precision_recall(pred, gt)
    _shapes_are_close(precision, 9 / 16)   # tp/(tp+fp) = 9/16
    _shapes_are_close(recall, 9 / 16)      # tp/(tp+fn) = 9/16

    assert metrics.f1(1.0, 1.0) == 1.0
    assert metrics.f1(0.0, 0.0) == 0.0            # 分母为 0 时不报错
    # F2 偏向召回：P=1, R=0.5 -> 5*0.5/(4*1+0.5) = 2.5/4.5
    _shapes_are_close(metrics.fbeta(1.0, 0.5, beta=2.0), 2.5 / 4.5)


def test_shape_mismatch_raises():
    raised = False
    try:
        metrics.confusion_counts(np.zeros((4, 4)), np.zeros((8, 8)))
    except ValueError:
        raised = True
    assert raised, "尺寸不一致时应当抛 ValueError"


# ================================================================= HD95
def test_hd95_exact_distance():
    """相距 4 像素的两个单点 -> HD95 必须正好是 4.0。

    这条测试是防回归用的：cv2.distanceTransform 的 maskSize=3 走近似 chamfer
    权重，会把 4 算成 3.82，从而让所有 HD95 系统性偏小。
    """
    point_a = np.zeros((8, 8), np.uint8)
    point_a[0, 0] = 1
    point_b = np.zeros((8, 8), np.uint8)
    point_b[0, 4] = 1
    _shapes_are_close(metrics.hd95(point_a, point_b), 4.0, tolerance=1e-4)


def test_hd95_zero_and_symmetry():
    gt = _square(row=2, column=2)
    assert metrics.hd95(gt, gt) == 0.0

    pred = _square(row=3, column=3)
    _shapes_are_close(metrics.hd95(pred, gt), metrics.hd95(gt, pred), tolerance=1e-6)


def test_hd95_spacing_scales_distance():
    """spacing 用于把像素距离换算成毫米（DICOM 的 PixelSpacing）。"""
    point_a = np.zeros((8, 8), np.uint8)
    point_a[0, 0] = 1
    point_b = np.zeros((8, 8), np.uint8)
    point_b[0, 4] = 1
    _shapes_are_close(metrics.hd95(point_a, point_b, spacing=0.5), 2.0, tolerance=1e-4)


def test_surface_is_perimeter():
    """4x4 方块的边界是 16-4=12 个像素。"""
    square = _square(row=2, column=2, side=4)
    assert int(metrics.surface(square).sum()) == 12

    empty = np.zeros((8, 8), np.uint8)
    assert int(metrics.surface(empty).sum()) == 0


# ========================================================= 标签解析与栅格化
def test_polygon_to_mask():
    """覆盖 20%~80% 的矩形 -> 61x61 像素，中心在前景、角落是背景。

    注意是 61 而不是 60：cv2.fillPoly 的顶点端点也算在内（20 与 80 都在内），
    所以边长是 80-20+1=61。
    """
    rectangle = [0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8]  # 平铺 x1 y1 x2 y2 ...
    mask = metrics.polygon_to_mask([rectangle], 100, 100)
    assert mask.shape == (100, 100)
    assert mask[50, 50] == 255
    assert mask[5, 5] == 0
    assert int((mask > 0).sum()) == 61 * 61


def test_polygon_to_mask_rejects_degenerate_polygon():
    """少于 3 个点围不成面，应当返回全零而不是报错。"""
    mask = metrics.polygon_to_mask([[0.1, 0.1, 0.2, 0.2]], 10, 10)
    assert int(mask.sum()) == 0


def test_parse_segment_label():
    text = "0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n"
    parsed = metrics.parse_segment_label(text, 100, 100)
    assert parsed["class_ids"] == [0]
    assert len(parsed["polygons"]) == 1
    assert len(parsed["masks"]) == 1
    assert parsed["masks"][0][50, 50] == 255


def test_parse_segment_label_skips_malformed_lines():
    text = (
        "0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n"   # 正常
        "0 0.1 0.1\n"                            # 点太少 -> 跳过
        "\n"                                     # 空行 -> 跳过
        "1 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"   # 正常
    )
    parsed = metrics.parse_segment_label(text, 100, 100)
    assert parsed["class_ids"] == [0, 1]


def test_parse_detect_label():
    """中心 (0.5, 0.5)、宽高各 0.5 的框 -> 像素坐标 (25,25,75,75)。"""
    parsed = metrics.parse_detect_label("0 0.5 0.5 0.5 0.5\n", 100, 100)
    assert parsed["class_ids"] == [0]
    assert parsed["boxes"] == [[25.0, 25.0, 75.0, 75.0]]

    skipped = metrics.parse_detect_label("0 0.5 0.5\n", 100, 100)  # 字段不足
    assert skipped["class_ids"] == [] and skipped["boxes"] == []


def test_merge_by_class():
    shape = (4, 4)
    first = np.zeros(shape, np.uint8); first[0, 0] = 1
    second = np.zeros(shape, np.uint8); second[1, 1] = 1
    merged = metrics.merge_by_class([(0, first), (0, second), (1, second)], shape)
    assert set(merged) == {0, 1}
    assert int(merged[0].sum()) == 2   # 同类合并
    assert int(merged[1].sum()) == 1


# ============================================================== 实例匹配
def _two_squares():
    """两个互不相交的 2x2 方块，用于构造「完美匹配」的实例集合。"""
    first = _square(row=0, column=0, side=2)
    second = _square(row=4, column=4, side=2)
    return first, second


def test_mask_iou_matrix():
    first, second = _two_squares()
    matrix = metrics.mask_iou_matrix([first, second], [first, second])
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 1.0 and matrix[1, 1] == 1.0
    assert matrix[0, 1] == 0.0 and matrix[1, 0] == 0.0

    empty = metrics.mask_iou_matrix([], [first])
    assert empty.shape == (0, 1)


def test_match_instances_perfect_plus_false_positive():
    first, second = _two_squares()
    extra = _square(row=6, column=0, side=2)
    matched = metrics.match_instances([first, second, extra], [first, second])
    assert (matched["tp"], matched["fp"], matched["fn"]) == (2, 1, 0)
    assert matched["mean_dice"] == 1.0
    _shapes_are_close(matched["precision"], 2 / 3)
    assert matched["recall"] == 1.0
    _shapes_are_close(matched["f1"], 0.8)


def test_match_instances_missed_detection():
    first, second = _two_squares()
    matched = metrics.match_instances([first], [first, second])
    assert (matched["tp"], matched["fp"], matched["fn"]) == (1, 0, 1)
    assert matched["precision"] == 1.0
    assert matched["recall"] == 0.5


def test_match_instances_respects_iou_threshold():
    """IoU = 2/6 = 0.333，低于 0.5，不应算命中。"""
    gt = _square(row=0, column=0, side=2)
    pred = _square(row=1, column=0, side=2)
    matched = metrics.match_instances([pred], [gt], iou_threshold=0.5)
    assert (matched["tp"], matched["fp"], matched["fn"]) == (0, 1, 1)


# ================================================================== 检测
def test_box_iou_matrix():
    matrix = metrics.box_iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 5, 15, 15]])
    assert matrix.shape == (1, 2)
    _shapes_are_close(matrix[0, 0], 1.0)
    _shapes_are_close(matrix[0, 1], 25 / 175)   # 交 25，并 100+100-25

    disjoint = metrics.box_iou_matrix([[0, 0, 10, 10]], [[20, 20, 30, 30]])
    assert disjoint[0, 0] == 0.0

    assert metrics.box_iou_matrix([], [[0, 0, 1, 1]]).shape == (0, 1)


def test_match_detections_is_class_aware():
    """框完全重合但类别不同，不能算命中。"""
    same_class = metrics.match_detections([[0, 0, 10, 10]], [0.9], [0], [[0, 0, 10, 10]], [0])
    assert same_class["tp"] == 1

    wrong_class = metrics.match_detections([[0, 0, 10, 10]], [0.9], [0], [[0, 0, 10, 10]], [1])
    assert (wrong_class["tp"], wrong_class["fp"], wrong_class["fn"]) == (0, 1, 1)


def test_average_precision_analytic():
    """TP,FP,TP 且共 2 个真值：AP = 0.5*1.0 + 0.5*(2/3) = 5/6。"""
    ap = metrics.average_precision([True, False, True], [0.9, 0.8, 0.7], num_gt=2)
    _shapes_are_close(ap, 5 / 6)

    assert metrics.average_precision([True], [0.9], num_gt=1) == 1.0
    assert metrics.average_precision([True], [0.9], num_gt=0) == 0.0   # 没有真值
    assert metrics.average_precision([], [], num_gt=3) == 0.0          # 没有预测


def test_detection_report():
    report = metrics.detection_report(
        [{"boxes": [[0, 0, 10, 10], [50, 50, 60, 60]], "scores": [0.9, 0.4], "classes": [0, 0]}],
        [{"boxes": [[0, 0, 10, 10]], "classes": [0]}],
        class_names=["RBC"],
    )
    record = report["per_class"][0]
    assert (record["tp"], record["fp"], record["fn"]) == (1, 1, 0)
    assert record["name"] == "RBC"
    _shapes_are_close(record["precision"], 0.5)
    _shapes_are_close(record["recall"], 1.0)
    _shapes_are_close(record["f1"], 2 / 3)
    assert report["mAP"] == 1.0                     # 召回在最高分处就到达 1
    _shapes_are_close(report["macro"]["precision"], 0.5)


# ================================================================== 分类
def test_confusion_matrix():
    matrix = metrics.confusion_matrix([0, 1, 2, 2], [0, 2, 2, 2])
    assert matrix.shape == (3, 3)
    assert matrix.tolist() == [[1, 0, 0], [0, 0, 1], [0, 0, 2]]


def test_classification_report():
    report = metrics.classification_report([0, 1, 2, 2], [0, 2, 2, 2],
                                           class_names=["Covid", "Normal", "Viral Pneumonia"])
    assert report["num_samples"] == 4
    _shapes_are_close(report["accuracy"], 0.75)
    # 逐类 F1：1.0 / 0.0 / 0.8 -> macro 0.6
    _shapes_are_close(report["macro_f1"], 0.6)
    _shapes_are_close(report["macro_precision"], (1.0 + 0.0 + 2 / 3) / 3)
    assert report["per_class"][2]["name"] == "Viral Pneumonia"
    assert report["per_class"][1]["support"] == 1


# ======================================================= 逐类指标与聚合
def test_per_class_mask_scores_and_empty_ground_truth():
    first, second = _two_squares()
    records = metrics.per_class_mask_scores({0: first}, {0: first, 1: second},
                                            class_names=["benign", "malignant"])
    assert records[0]["dice"] == 1.0
    assert records[0]["gt_empty"] is False
    assert records[1]["dice"] == 0.0          # 整个类别漏检
    assert records[1]["name"] == "malignant"
    assert math.isinf(records[1]["hd95"])

    values = list(records.values())
    _shapes_are_close(metrics.mean_of(values, "dice"), 0.5)
    assert metrics.mean_of(values, "hd95") == 0.0     # inf 被跳过


def test_mean_of_skip_empty_gt():
    """真值全空的样本会把 Dice 抬到 1.0，可用 skip_empty_gt 剔除。"""
    blank = np.zeros((8, 8), np.uint8)
    records = list(metrics.per_class_mask_scores({0: blank}, {0: blank}).values())
    assert records[0]["dice"] == 1.0
    assert records[0]["gt_empty"] is True
    assert metrics.mean_of(records, "dice") == 1.0
    assert metrics.mean_of(records, "dice", skip_empty_gt=True) == 0.0


# ================================================================ 运行器
def main() -> int:
    """收集本模块所有 test_* 函数并执行，返回退出码（0 = 全部通过）。"""
    tests = [(name, function) for name, function in sorted(globals().items())
             if name.startswith("test_") and callable(function)]

    failures = 0
    for name, function in tests:
        try:
            function()
        except Exception as error:  # 测试失败要报出原因而不是中断整轮
            failures += 1
            print(f"FAIL  {name}: {type(error).__name__}: {error}")
        else:
            print(f"PASS  {name}")

    print("-" * 68)
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
