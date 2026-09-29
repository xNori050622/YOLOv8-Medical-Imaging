"""增强模块的回归测试：CLAHE 不动标签，弹性形变的顶点必须跟着像素走。

用法（两种都行）：
    python tests/test_augment.py     # 零依赖，直接跑
    pytest tests/test_augment.py     # 装了 pytest 也可以

为什么要专门测这个
------------------
弹性形变是几何变换。只 warp 图像、不给多边形顶点施加同一个位移场，训练和
评估都不会报错，但模型学到的是「病灶位置系统性偏移」的数据集 —— 这类静默
错配本仓库已经踩过一次（segmentation/masks_to_polygons.py 里 splitfolders
把图片和掩码分到不同 split，见 tests/test_split_alignment.py）。

所以这里除了「顶点确实动了」，还额外验证两件事：
  1. 把解代回 remap 的定义式 x + dx(x) == u，残差必须在亚像素量级；
  2. 「把位移直接加到顶点上」这种常见错误做法的 IoU 明显更低（实测 0.836
     对 0.964）—— 让「必须用定点迭代求逆映射」这个断言真的能失败。

阈值都取自实测值（见 augment.py 模块 docstring 的表），留了足够余量，
既不会偶发抖动，也不会宽到失效。所有数据都是合成的，不读真实数据集。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # 支持 python tests/test_augment.py 直接运行
    sys.path.insert(0, str(PROJECT_ROOT))

import augment
import metrics

# 默认几何：160x128 的图，半径 22 的圆形病灶，用 24 边形近似
HEIGHT, WIDTH = 128, 160
CENTER_X, CENTER_Y, RADIUS = 86, 62, 22

# 实测：顶点同步 0.9635、朴素做法 0.8388、1x 光栅化噪声下限 0.9894
MIN_SYNC_IOU = 0.95
MAX_NAIVE_IOU = 0.92


# ------------------------------------------------------------------ 小工具
def _synthetic_image(height: int = HEIGHT, width: int = WIDTH) -> np.ndarray:
    """合成一张「像超声」的 BGR 图：横向亮度梯度 + 噪声 + 高回声病灶。"""
    rng = np.random.default_rng(0)
    base = np.linspace(30, 150, width, dtype=np.float32)[None, :].repeat(height, 0)
    gray = np.clip(base + rng.normal(0, 12, (height, width)), 0, 255).astype(np.uint8)
    cv2.circle(gray, (CENTER_X, CENTER_Y), RADIUS, 215, -1)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.5)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _circle_polygon(height: int = HEIGHT, width: int = WIDTH,
                    vertices: int = 24) -> np.ndarray:
    """圆形病灶的归一化多边形 (N,2)。"""
    angles = np.linspace(0, 2 * np.pi, vertices, endpoint=False)
    return np.stack([(CENTER_X + RADIUS * np.cos(angles)) / width,
                     (CENTER_Y + RADIUS * np.sin(angles)) / height], axis=1)


def _circle_instances(class_id: int = 1) -> list:
    return [(class_id, _circle_polygon())]


def _disc_polygon(radius: int, vertices: int, center_x: int = 80,
                  center_y: int = 64) -> np.ndarray:
    """任意半径的圆形病灶多边形，用于「小病灶」这类周长/面积比偏高的情形。"""
    angles = np.linspace(0, 2 * np.pi, vertices, endpoint=False)
    return np.stack([(center_x + radius * np.cos(angles)) / WIDTH,
                     (center_y + radius * np.sin(angles)) / HEIGHT], axis=1)


def _mask_from_polygon(polygon: np.ndarray) -> np.ndarray:
    """多边形 -> uint8 掩码，走 metrics 的光栅化（与评估口径一致）。"""
    return metrics.polygon_to_mask([np.asarray(polygon).reshape(-1)], HEIGHT, WIDTH)


def _mask_from_instances(instances) -> np.ndarray:
    return metrics.polygon_to_mask([polygon.reshape(-1) for _, polygon in instances],
                                   HEIGHT, WIDTH)


def _remapped_mask(polygon: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """把多边形先光栅化、再用同一个位移场 warp —— 同步性的参考结果。"""
    return augment.warp_image(_mask_from_polygon(polygon), dx, dy,
                              interpolation=cv2.INTER_NEAREST,
                              border=cv2.BORDER_CONSTANT)


def _naive_forward(points: np.ndarray, dx: np.ndarray,
                   dy: np.ndarray) -> np.ndarray:
    """错误做法：把 remap 的（反向）位移直接加到顶点上，当作正向映射。"""
    return points + np.stack([augment._sample_field(dx, points[:, 0], points[:, 1]),
                              augment._sample_field(dy, points[:, 0], points[:, 1])],
                             axis=1)


# ============================================================ 默认关闭（基线）
def test_identity_when_disabled():
    """两项都不传时，图像与标签必须原样返回 —— 基线不能被本模块污染。"""
    image = _synthetic_image()
    instances = _circle_instances()
    result_image, result_instances = augment.augment_sample(image, instances)

    assert result_image is image
    assert len(result_instances) == 1
    assert result_instances[0][0] == 1
    assert np.array_equal(result_instances[0][1], instances[0][1])


def test_disabled_with_falsey_values():
    """显式传 False 或空 dict 也应视为关闭（空 dict 是假值）。"""
    image = _synthetic_image()
    instances = _circle_instances()
    for kwargs in ({"clahe": False, "elastic": False}, {"clahe": {}, "elastic": {}}):
        result_image, result_instances = augment.augment_sample(image, instances, **kwargs)
        assert result_image is image
        assert np.array_equal(result_instances[0][1], instances[0][1])


# ================================================================== CLAHE
def test_clahe_changes_pixels_but_not_labels():
    """CLAHE 是光度变换：像素要变，标签必须一字不差。"""
    image = _synthetic_image()
    instances = _circle_instances()
    enhanced, result_instances = augment.augment_sample(image, instances, clahe=True)

    assert enhanced.shape == image.shape
    assert enhanced.dtype == np.uint8
    assert not np.array_equal(enhanced, image)
    assert np.array_equal(result_instances[0][1], instances[0][1])


def test_clahe_handles_grayscale_and_color():
    """灰度进去灰度出来，BGR 进去 BGR 出来。"""
    gray = cv2.cvtColor(_synthetic_image(), cv2.COLOR_BGR2GRAY)
    assert augment.apply_clahe(gray).shape == gray.shape
    color = _synthetic_image()
    assert augment.apply_clahe(color).shape == color.shape


def test_clahe_preserves_hue_on_color_image():
    """只在 LAB 的 L 通道上均衡，所以色调不该被拉偏。

    反例：对 BGR 三个通道分别做 CLAHE，各通道增益曲线不同，红蓝两半的色调会
    明显偏移。这里用「左半偏红、右半偏蓝」的合成图把这个差异体现出来。
    """
    height, width = 64, 64
    image = np.zeros((height, width, 3), np.uint8)
    image[:, :width // 2] = (40, 40, 200)   # 偏红
    image[:, width // 2:] = (200, 40, 40)   # 偏蓝

    enhanced = augment.apply_clahe(image)
    assert enhanced.shape == image.shape

    hue_before = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 0].astype(float).mean()
    hue_after = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)[..., 0].astype(float).mean()
    assert abs(hue_before - hue_after) < 5.0, f"色调被改动：{hue_before} -> {hue_after}"

    # 同时确认 CLAHE 真的起作用了（亮度直方图被拉宽）
    assert enhanced.reshape(-1, 3).mean() != 0
    assert not np.array_equal(enhanced, image)


# ============================================================== 位移场
def test_displacement_field_is_deterministic_and_bounded():
    """同 seed 完全一致、异 seed 有差异，且最大位移恰为 alpha。"""
    first = augment.displacement_field((HEIGHT, WIDTH), alpha=8.0, sigma=6.0, seed=7)
    again = augment.displacement_field((HEIGHT, WIDTH), alpha=8.0, sigma=6.0, seed=7)
    other = augment.displacement_field((HEIGHT, WIDTH), alpha=8.0, sigma=6.0, seed=8)

    assert np.array_equal(first[0], again[0]) and np.array_equal(first[1], again[1])
    assert not np.array_equal(first[0], other[0])

    amplitude = max(float(np.abs(first[0]).max()), float(np.abs(first[1]).max()))
    assert abs(amplitude - 8.0) < 1e-4, f"最大位移 {amplitude} 应等于 alpha=8"


def test_displacement_field_zero_alpha_is_identity():
    """alpha=0 得到全零场，warp 后图像不变。"""
    dx, dy = augment.displacement_field((HEIGHT, WIDTH), alpha=0.0, sigma=6.0, seed=0)
    assert not dx.any() and not dy.any()
    image = _synthetic_image()
    assert np.array_equal(augment.warp_image(image, dx, dy), image)


def test_displacement_field_rejects_bad_params():
    """sigma=0 或负数 alpha 必须报错（sigma=0 时定点迭代不收敛）。"""
    for kwargs in ({"alpha": 8.0, "sigma": 0.0}, {"alpha": -1.0, "sigma": 6.0}):
        try:
            augment.displacement_field((HEIGHT, WIDTH), **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"应当报错：{kwargs}")


# ============================================================ 正向映射
def test_forward_map_inverts_remap():
    """核心定律：变换后的顶点代回 remap 定义式 x + dx(x) == u 应回到原位。

    这是「顶点与像素同步」的充分判据，且不受光栅化噪声影响，所以最严格。
    """
    polygon = _circle_polygon()
    points = augment.polygon_to_pixels(polygon, HEIGHT, WIDTH)
    dx, dy = augment.displacement_field((HEIGHT, WIDTH), alpha=10.0, sigma=6.0, seed=0)
    moved = augment.forward_map(points, dx, dy)

    back_x = moved[:, 0] + augment._sample_field(dx, moved[:, 0], moved[:, 1])
    back_y = moved[:, 1] + augment._sample_field(dy, moved[:, 0], moved[:, 1])
    residual = max(float(np.abs(back_x - points[:, 0]).max()),
                   float(np.abs(back_y - points[:, 1]).max()))
    assert residual < augment._INVERSE_MAX_RESIDUAL, f"残差 {residual} 超过验收阈值"

    # 防止「原地不动也能过」的退化情况：顶点确实被移动了
    assert float(np.abs(moved - points).max()) > 1.0


def test_forward_map_raises_when_not_converged():
    """迭代轮数不足时必须报错，而不是悄悄返回错位的顶点。"""
    points = augment.polygon_to_pixels(_circle_polygon(), HEIGHT, WIDTH)
    dx, dy = augment.displacement_field((HEIGHT, WIDTH), alpha=10.0, sigma=6.0, seed=0)
    try:
        augment.forward_map(points, dx, dy, iterations=3)
    except ValueError as error:
        assert "未收敛" in str(error)
    else:
        raise AssertionError("3 轮迭代不足以收敛，应当报错")


def test_forward_map_handles_empty_points():
    """空点集（背景图）返回空数组，不抛异常。"""
    dx, dy = augment.displacement_field((HEIGHT, WIDTH), alpha=8.0, sigma=6.0, seed=0)
    moved = augment.forward_map(np.empty((0, 2)), dx, dy)
    assert moved.shape == (0, 2)


# ======================================================= 同步性（顶点 vs 像素）
def _sync_scores():
    """返回 (同步做法的 IoU, 朴素做法的 IoU, 参考掩码)。"""
    polygon = _circle_polygon()
    dx, dy = augment.displacement_field((HEIGHT, WIDTH), alpha=10.0, sigma=6.0, seed=0)
    _image, instances = augment.augment_sample(
        _synthetic_image(), _circle_instances(),
        elastic={"alpha": 10.0, "sigma": 6.0}, seed=0)
    reference = _remapped_mask(polygon, dx, dy)

    points = augment.polygon_to_pixels(polygon, HEIGHT, WIDTH)
    naive_polygon = augment.pixels_to_polygon(
        _naive_forward(points, dx, dy), HEIGHT, WIDTH)
    naive_mask = metrics.polygon_to_mask([naive_polygon.reshape(-1)], HEIGHT, WIDTH)

    return (metrics.iou(_mask_from_instances(instances), reference),
            metrics.iou(naive_mask, reference), reference)


def test_warped_polygon_matches_warped_mask():
    """顶点同步变换后栅格化的掩码，应与「掩码直接 warp」高度一致。"""
    synced, _naive, _reference = _sync_scores()
    assert synced >= MIN_SYNC_IOU, f"同步性 IoU={synced} 低于 {MIN_SYNC_IOU}"


def test_naive_vertex_shift_is_worse():
    """把（反向）位移直接加到顶点上是错的，IoU 明显更低 —— 断言必须能失败。"""
    synced, naive, _reference = _sync_scores()
    assert naive < MAX_NAIVE_IOU, f"朴素做法 IoU={naive} 意外地高"
    assert naive < synced - 0.05, f"朴素 {naive} vs 同步 {synced}，差距不显著"


def test_small_lesion_naive_shift_destroys_label():
    """小病灶（周长/面积比高）上朴素做法会彻底毁掉标签。

    实测半径 8px 的病灶（面积 217px²）：同步 0.954，朴素 0.497。小病灶恰恰是
    Dice/IoU 最敏感、临床上最要紧的情形，所以这条差距必须钉死。
    """
    polygon = _disc_polygon(radius=8, vertices=12)
    dx, dy = augment.displacement_field((HEIGHT, WIDTH), alpha=10.0, sigma=6.0, seed=0)
    source_mask = metrics.polygon_to_mask([polygon.reshape(-1)], HEIGHT, WIDTH)
    remap_mask = augment.warp_image(source_mask, dx, dy,
                                    interpolation=cv2.INTER_NEAREST,
                                    border=cv2.BORDER_CONSTANT)

    _image, moved = augment.augment_sample(
        np.zeros((HEIGHT, WIDTH, 3), np.uint8), [(1, polygon)],
        elastic={"alpha": 10.0, "sigma": 6.0}, seed=0)
    synced = metrics.iou(
        metrics.polygon_to_mask([moved[0][1].reshape(-1)], HEIGHT, WIDTH), remap_mask)

    points = augment.polygon_to_pixels(polygon, HEIGHT, WIDTH)
    naive_mask = metrics.polygon_to_mask(
        [augment.pixels_to_polygon(_naive_forward(points, dx, dy),
                                   HEIGHT, WIDTH).reshape(-1)], HEIGHT, WIDTH)
    naive = metrics.iou(naive_mask, remap_mask)

    assert synced >= 0.94, f"小病灶上的同步性只有 {synced}"
    assert naive <= 0.60, f"小病灶上的朴素做法居然有 {naive}，说明断言无效"


def test_warped_polygon_stays_in_bounds():
    """强形变下顶点被推出画布，必须夹回 [0,1]。"""
    polygon = np.array([[0.02, 0.02], [0.09, 0.02], [0.09, 0.09], [0.02, 0.09]])
    _image, instances = augment.augment_sample(
        _synthetic_image(), [(0, polygon)],
        elastic={"alpha": 20.0, "sigma": 8.0}, seed=3)
    assert len(instances) == 1
    values = instances[0][1]
    assert values.min() >= 0.0 and values.max() <= 1.0


# =============================================================== 参数校验
def test_clahe_rejects_bad_input():
    """非 uint8 与非法超参必须报错，而不是静默产出垃圾。"""
    for bad_image, kwargs in ((np.zeros((8, 8), np.float32), {}),
                              (np.zeros((8, 8), np.uint8), {"clip_limit": 0}),
                              (np.zeros((8, 8), np.uint8), {"tile_grid_size": 0})):
        try:
            augment.apply_clahe(bad_image, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"应当报错：shape={bad_image.shape} {kwargs}")


# ============================================================== 实例保持
def test_instances_preserved_by_warp():
    """形变不增删实例：类别 id 与顶点数都要原样保留。"""
    instances = [(0, _circle_polygon(vertices=12)),
                 (2, _circle_polygon(vertices=31))]
    _image, result = augment.augment_sample(
        _synthetic_image(), instances, elastic={"alpha": 8.0, "sigma": 6.0}, seed=1)

    assert [class_id for class_id, _ in result] == [0, 2]
    assert [len(polygon) for _, polygon in result] == [12, 31]


def test_background_sample_survives_elastic():
    """空标签必须能处理（BUSI 有 102 张无病灶图）：图像照常 warp，标签仍为空。"""
    image = _synthetic_image()
    warped_image, instances = augment.augment_sample(
        image, [], elastic={"alpha": 8.0, "sigma": 6.0}, seed=0)

    assert instances == []
    assert warped_image.shape == image.shape
    assert not np.array_equal(warped_image, image)


# ============================================================== 标签读写
def test_empty_label_round_trip():
    """空标签 dump 出来仍是空字符串 —— YOLO 用空文件表示背景图。"""
    assert augment.dump_label([]) == ""
    assert augment.load_label("") == []
    assert augment.load_label("\n\n") == []


def test_label_round_trip_is_numerically_exact():
    """dump 再 load 必须数值一致：不取整，否则恒等情形也会改标签。"""
    fields = [6, 0.5533807829181495, 0.28450106157112526, 0.1, 0.2, 0.3, 0.4]
    text = " ".join(str(value) for value in fields) + "\n"

    instances = augment.load_label(text)
    assert len(instances) == 1
    assert instances[0][0] == 6
    assert np.array_equal(instances[0][1], np.asarray(fields[1:]).reshape(-1, 2))

    again = augment.load_label(augment.dump_label(instances))
    assert np.array_equal(again[0][1], instances[0][1])


def test_load_label_skips_degenerate_lines():
    """点数不足、坐标个数为奇数、只有类别、空行都要按规则处理，不能崩。"""
    text = ("1 0.1 0.1 0.2 0.1\n"                  # 只有 2 个点 -> 整行跳过
            "\n"
            "3\n"                                    # 只有类别 -> 跳过
            "2 0.1 0.1 0.2 0.2 0.3 0.4 0.5\n"        # 7 个坐标(奇数) -> 丢掉末位成 3 点
            "4 0.1 0.1 0.5 0.1 0.5 0.5\n")           # 合法

    instances = augment.load_label(text)
    assert [class_id for class_id, _ in instances] == [2, 4]
    assert [len(polygon) for _, polygon in instances] == [3, 3]


def test_polygon_area_matches_rasterized_mask():
    """面积换算与光栅化口径一致（±5%），确认 x 乘 width、y 乘 height。"""
    polygon = _circle_polygon()
    area = augment.polygon_area(polygon, HEIGHT, WIDTH)
    pixels = int(np.count_nonzero(_mask_from_polygon(polygon)))
    assert abs(area - pixels) / pixels < 0.05, f"多边形面积 {area} vs 掩码像素 {pixels}"


def test_polygon_area_ignores_degenerate():
    """少于 3 个点无法围成面，返回 0 而不是报错。"""
    assert augment.polygon_area(np.array([[0.1, 0.1], [0.2, 0.2]]), HEIGHT, WIDTH) == 0.0


# ====================================================== 数据集层面（split）
_LABEL_TEXT = "0 0.30 0.30 0.70 0.30 0.70 0.70 0.30 0.70\n"


def _build_split_dataset(root: Path) -> Path:
    """造出 data/{train,val,test}/{images,labels}，每个 split 一张病灶图。

    train 里额外放一张无标签的背景图，用来验证空标签增强后仍然是空文件。
    """
    source = root / "data"
    for split in ("train", "val", "test"):
        (source / split / "images").mkdir(parents=True)
        (source / split / "labels").mkdir(parents=True)
        cv2.imwrite(str(source / split / "images" / f"{split}_lesion.png"),
                    _synthetic_image(48, 64))
        (source / split / "labels" / f"{split}_lesion.txt").write_text(
            _LABEL_TEXT, encoding="utf-8")

    cv2.imwrite(str(source / "train" / "images" / "background.png"),
                _synthetic_image(48, 64))
    (source / "train" / "labels" / "background.txt").write_text("", encoding="utf-8")
    return source


def _read_tree(directory: Path) -> dict:
    """{相对路径: 字节}，用于逐字节比较两次输出。"""
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in sorted(directory.rglob("*")) if path.is_file()}


def test_split_dir_keeps_val_and_test_byte_identical():
    """最重要的保证：val/test 逐字节不变，否则评估数字与基线不可比。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _build_split_dataset(root)
        output = root / "augmented"

        augment.augment_split_dir(source, output, copies=2, clahe=True,
                                  elastic={"alpha": 6.0, "sigma": 6.0}, seed=0)

        for split in ("val", "test"):
            assert _read_tree(source / split) == _read_tree(output / split), \
                f"{split} 被改动了 —— 评估将与基线不可比"

        before = _read_tree(source / "train")
        after = _read_tree(output / "train")
        assert all(name in after for name in before), "原图/原标签必须全部保留"

        # 2 张图 x 2 份副本 x (图 + 标签) = 8 个新文件
        assert len(after) == len(before) + 8, f"{len(before)} -> {len(after)}"


def test_split_dir_is_additive():
    """增强是增补而非替换，且只作用于 train。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _build_split_dataset(root)
        output = root / "augmented"

        stats = augment.augment_split_dir(source, output, copies=1, clahe=True, seed=0)

        assert stats["train"] == {"images": 2, "labels": 2, "augmented": 2, "skipped": []}
        assert stats["val"]["augmented"] == 0
        assert stats["test"]["augmented"] == 0
        assert (output / "train" / "images" / "train_lesion.png").is_file()
        assert (output / "train" / "images" / "train_lesion_aug1.png").is_file()
        assert not (output / "val" / "images" / "val_lesion_aug1.png").is_file()


def test_split_dir_augmented_labels_are_valid():
    """增强出的标签必须能解析回来、坐标在 [0,1]、背景图仍为空文件。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / "augmented"
        augment.augment_split_dir(_build_split_dataset(root), output, copies=2,
                                  clahe=True, elastic={"alpha": 6.0, "sigma": 6.0}, seed=1)

        for index in (1, 2):
            text = (output / "train" / "labels" / f"train_lesion_aug{index}.txt") \
                .read_text(encoding="utf-8")
            instances = augment.load_label(text)
            assert len(instances) == 1, f"第 {index} 份副本的实例数不对"
            assert instances[0][0] == 0
            values = instances[0][1]
            assert values.min() >= 0.0 and values.max() <= 1.0
            assert (output / "train" / "labels" / f"background_aug{index}.txt") \
                .read_text(encoding="utf-8") == ""


def test_split_dir_is_reproducible():
    """同参数跑两次逐字节一致，否则消融实验无法复现。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _build_split_dataset(root)
        first, second = root / "first", root / "second"

        for target in (first, second):
            augment.augment_split_dir(source, target, copies=2, clahe=True,
                                      elastic={"alpha": 6.0, "sigma": 6.0}, seed=5)

        assert _read_tree(first) == _read_tree(second)


def test_split_dir_rejects_bad_arguments():
    """输出目录等于源目录、copies 为负、源目录不存在都必须报错。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _build_split_dataset(root)
        cases = (
            (lambda: augment.augment_split_dir(source, source), ValueError),
            (lambda: augment.augment_split_dir(source, root / "out", copies=-1), ValueError),
            (lambda: augment.augment_split_dir(root / "missing", root / "out"),
             FileNotFoundError),
        )
        for call, exception in cases:
            try:
                call()
            except exception:
                pass
            else:
                raise AssertionError("应当报错却通过了")


def test_split_dir_preflight_blocks_bad_elastic():
    """位移场不收敛时必须在写文件之前失败，不能留下半个数据集。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / "augmented"
        try:
            augment.augment_split_dir(_build_split_dataset(root), output, copies=1,
                                      elastic={"alpha": 30.0, "sigma": 6.0})
        except ValueError as error:
            assert "未收敛" in str(error)
        else:
            raise AssertionError("alpha=30 / sigma=6 已知不收敛，应当报错")
        assert not output.exists(), "预检失败时不应该创建输出目录"


# ================================================================== 运行器
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
