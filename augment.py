"""医学图像增强：CLAHE 对比度增强 + 弹性形变，多边形标签同步变换。

为什么需要这个模块
------------------
BUSI 只有 624 张训练图（其中 102 张还是无病灶的背景图）。ultralytics 自带的
增强（mosaic / 随机仿射 / HSV 抖动）都是面向自然图像的：

* mosaic 把四张图拼成一张，会凭空造出超声图像里不存在的解剖结构；
* HSV 抖动改的是色相，而超声本质是灰度图，色相没有意义。

医学影像论文里真正常用的是「对比度受限自适应直方图均衡（CLAHE）」与
「弹性形变（elastic deformation）」，后者是模拟组织受压变形的经典做法
（Simard et al., 2003）。

核心难点：标签必须跟着像素一起动
--------------------------------
CLAHE 是**光度**变换，不移动像素，标签原样透传即可；弹性形变是**几何**
变换，如果只变换图像而不变换多边形顶点，模型就会学到一个「病灶位置系统性
偏移」的数据集 —— 训练和评估都不会报任何错。这类静默错配本仓库已经踩过：
见 segmentation/masks_to_polygons.py 里 splitfolders 把图片和掩码分到不同
split 的例子。

为什么不能把同一张位移场直接加到顶点上
--------------------------------------
cv2.remap 的语义是「输出 (x,y) 取自输入 (x+dx, y+dy)」，即 dx/dy 描述的是
**反向** 位移；而多边形顶点需要的是 **正向** 映射 —— 「输入的 (u,v) 会落到
输出的哪一点」。直接给顶点加 dx(u,v) 是常见但错误的近似（albumentations 的 ElasticTransform
干脆声明不支持 keypoints）。本模块用定点迭代求解 x = u - dx(x,y)，再把解代回
定义式验收残差：默认 32 轮内收敛到 4.5e-5 像素（float32 位移场的精度极限），
不收敛就报错，绝不悄悄产出一批系统性错位的标签。

实测（160x128 合成图、半径 22 的圆形病灶、alpha=10、sigma=6）：

    顶点同步变换 vs 掩码 warp 的 IoU     0.964
    位移直接加顶点上（错误做法）         0.836
    1x 最近邻 remap vs 4x 亚像素真值     0.989   <- 光栅化本身的噪声下限

第三行说明 0.964 离 1.0 的那段差距并不是映射错误：1x 二值光栅化（fillPoly 的
半像素边界偏差 + 最近邻重采样）本身就只到 0.989。真正有意义的是 0.964 与
0.836 之间那 0.13 的差距 —— 那才是「顶点有没有跟着像素一起动」。
tests/test_augment.py 把这三个数都钉住，让断言真的能失败。

真实数据（BUSI train 抽样，alpha=8、sigma=6）表现一致：密集标签（264 和 126 个
顶点）同步性 0.988 / 0.987；稀疏标签（48 个顶点、面积仅 984px²）只有 0.935。
后者更低**不是变换不准** —— 代数残差仍是 5.5e-05 像素，而且该多边形的原始顶点
全在整数格点上、1x 光栅化误差为 0，差距完全出在「变换后顶点变成小数坐标」时的
1x fillPoly 半像素边界偏差，量级约「周长 x 0.25px / 面积」：病灶越小越吃亏。
这是多边形标签表示自带的代价，与变换无关。真正要盯住的是朴素做法在同一个样本
上只有 0.764，在半径 8px 的病灶上更是掉到 0.497 —— 标签基本被毁掉。

默认关闭
--------
augment_sample() 的 clahe / elastic 默认都是 None，此时原样返回输入，保证
基线与消融对照不被本模块污染；要启用必须显式传参。数据集层面同理：
augment_split_dir() 只增补 train/，val/ 与 test/ 逐字节复制、绝不增强 ——
否则评估数字和基线就不可比了。
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

# 默认超参。clip_limit / tile_grid_size 取 OpenCV 教程对医学影像的常见取值；
# alpha 是最大位移像素数，sigma 控制位移场的光滑程度（越大越像整体形变）。
CLAHE_DEFAULTS: Dict[str, float] = {"clip_limit": 2.0, "tile_grid_size": 8}
ELASTIC_DEFAULTS: Dict[str, float] = {"alpha": 8.0, "sigma": 6.0}

# YOLO 多边形至少要 3 个点才能围成面
MIN_POLYGON_POINTS = 3

# 定点迭代的收敛阈值与最大轮数（见模块 docstring）
# 实测：24 轮后残差已到 float32 位移场的精度极限 4.5e-5 像素，32 轮留足余量；
# 正常参数下约 10 轮就触发容差提前退出。
_INVERSE_TOLERANCE = 1e-4
_INVERSE_ITERATIONS = 32

# 把解代回定义式后的验收阈值（像素）。4.5e-5 是精度极限，留 1000 倍余量：
# 正常参数不会触发，而真正发散时误差是「像素」级别，必然被抓住。
_INVERSE_MAX_RESIDUAL = 0.05


# ============================================================== 标签读写
Instance = Tuple[int, np.ndarray]
"""一个实例：(class_id, 归一化多边形)。多边形形如 (N,2) float64，坐标在 0~1。"""


def load_label(text: str) -> List[Instance]:
    """解析 YOLO 分割标签文本，返回 [(class_id, (N,2) 归一化多边形), ...]。

    与 metrics.parse_segment_label 的差别：这里保留浮点原值、只丢弃点数不足
    3 的行，不做取整。取整会让「无增强」这种恒等情形也改变标签，从而破坏
    「基线与消融只有增强这一个变量」的前提。
    """
    instances: List[Instance] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 1 + 2 * MIN_POLYGON_POINTS:
            continue
        coordinates = np.asarray([float(value) for value in parts[1:]], dtype=np.float64)
        if coordinates.size % 2:  # 坐标个数必须是偶数
            coordinates = coordinates[:-1]
        if len(coordinates) < 2 * MIN_POLYGON_POINTS:
            continue
        instances.append((int(float(parts[0])), coordinates.reshape(-1, 2)))
    return instances


def dump_label(instances: Sequence[Instance]) -> str:
    """把实例列表写回 YOLO 标签文本；空列表返回空字符串（背景图）。"""
    lines = []
    for class_id, polygon in instances:
        coordinates = " ".join(repr(float(value)) for value in np.asarray(polygon).reshape(-1))
        lines.append(f"{int(class_id)} {coordinates}")
    return "\n".join(lines) + "\n" if lines else ""


def polygon_to_pixels(polygon, height: int, width: int) -> np.ndarray:
    """归一化多边形 -> 像素坐标 (N,2) float64，不取整。

    坐标轴约定与 metrics._as_points 一致：x 乘 width，y 乘 height。
    """
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    return points * np.array([width, height], dtype=np.float64)


def pixels_to_polygon(points, height: int, width: int) -> np.ndarray:
    """像素坐标 -> 归一化多边形，并夹回 [0,1]（形变会把顶点推出画布）。"""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    normalized = points / np.array([width, height], dtype=np.float64)
    return np.clip(normalized, 0.0, 1.0)


def polygon_area(polygon, height: int, width: int) -> float:
    """多边形面积（像素²）。顶点过少时返回 0。"""
    points = polygon_to_pixels(polygon, height, width)
    if len(points) < MIN_POLYGON_POINTS:
        return 0.0
    return float(abs(cv2.contourArea(points.astype(np.float32))))


# ================================================================= CLAHE
def apply_clahe(image, clip_limit: float = None, tile_grid_size: int = None) -> np.ndarray:
    """对比度受限自适应直方图均衡（光度变换，不移动像素）。

    灰度图直接处理；BGR 三通道图转到 LAB 后只对 L（亮度）做 CLAHE 再转回，
    这样不会引入虚假色偏 —— 直接对 BGR 每个通道分别均衡会改变色相。

    输入须为 uint8（CLAHE 只支持 8 位单通道）。
    """
    clip_limit = CLAHE_DEFAULTS["clip_limit"] if clip_limit is None else clip_limit
    tile_grid_size = (CLAHE_DEFAULTS["tile_grid_size"] if tile_grid_size is None
                      else tile_grid_size)
    if clip_limit <= 0:
        raise ValueError(f"clip_limit 必须为正数，收到 {clip_limit}")
    if tile_grid_size <= 0:
        raise ValueError(f"tile_grid_size 必须为正数，收到 {tile_grid_size}")

    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError(f"CLAHE 只支持 uint8 图像，收到 {array.dtype}")
    if array.ndim not in (2, 3):
        raise ValueError(f"不支持的图像形状：{array.shape}")

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    if array.ndim == 2:
        return clahe.apply(array)
    if array.shape[2] == 3:
        lightness, channel_a, channel_b = cv2.split(cv2.cvtColor(array, cv2.COLOR_BGR2LAB))
        merged = cv2.merge([clahe.apply(lightness), channel_a, channel_b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    raise ValueError(f"只支持灰度图或 3 通道 BGR 图，收到 {array.shape}")


# ============================================================== 弹性形变
def image_grid(shape) -> Tuple[np.ndarray, np.ndarray]:
    """像素坐标网格 (grid_x, grid_y)，供 cv2.remap 使用。"""
    height, width = shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                 np.arange(height, dtype=np.float32))
    return grid_x, grid_y


def displacement_field(shape, alpha: float = None, sigma: float = None,
                       seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """生成光滑随机位移场 (dx, dy)，单位像素。

    做法：先取 [-1,1] 均匀噪声，高斯模糊得到光滑场，再等比缩放到最大位移恰为
    alpha。缩放放在模糊之后，这样 alpha 的物理含义就是「最大位移多少像素」，
    而不是「模糊前噪声的幅度」；否则同一个 alpha 在不同 sigma 下的实际位移会差
    一个数量级，超参就没法解释了。

    sigma 必须为正：位移场越光滑（sigma 越大）越像整体组织受压；sigma 太小时
    相邻像素位移方向随机，正向映射的定点迭代会不收敛。实测 alpha/sigma 不超过
    2 时收敛因子在 0.65 以内（32 轮足以收敛到精度极限），比例再大就会退化，
    forward_map 会直接报错而不是产出错位的标签。
    """
    alpha = ELASTIC_DEFAULTS["alpha"] if alpha is None else alpha
    sigma = ELASTIC_DEFAULTS["sigma"] if sigma is None else sigma
    if alpha < 0:
        raise ValueError(f"alpha 不能为负数，收到 {alpha}")
    if sigma <= 0:
        raise ValueError(f"sigma 必须为正数，收到 {sigma}（否则位移场不光滑）")

    height, width = shape[:2]
    rng = np.random.default_rng(seed)
    dx = rng.uniform(-1.0, 1.0, (height, width)).astype(np.float32)
    dy = rng.uniform(-1.0, 1.0, (height, width)).astype(np.float32)

    kernel = int(sigma * 4) | 1  # 高斯核必须是奇数
    dx = cv2.GaussianBlur(dx, (kernel, kernel), float(sigma))
    dy = cv2.GaussianBlur(dy, (kernel, kernel), float(sigma))

    if alpha > 0:
        # dx/dy 同比例缩放，只把幅度顶到 alpha，位移方向保持不动
        scale = alpha / max(float(np.abs(dx).max()), float(np.abs(dy).max()), 1e-8)
        dx = dx * scale
        dy = dy * scale
    else:
        dx = np.zeros_like(dx)
        dy = np.zeros_like(dy)
    return dx, dy


def _sample_field(field: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """在浮点坐标 (xs, ys) 处双线性采样标量场，越界坐标夹到边界。"""
    height, width = field.shape
    x0 = np.clip(np.floor(xs).astype(np.int64), 0, width - 1)
    y0 = np.clip(np.floor(ys).astype(np.int64), 0, height - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    fx = np.clip(xs - x0, 0.0, 1.0)
    fy = np.clip(ys - y0, 0.0, 1.0)

    top = field[y0, x0] * (1 - fx) + field[y0, x1] * fx
    bottom = field[y1, x0] * (1 - fx) + field[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def forward_map(points, dx: np.ndarray, dy: np.ndarray,
                iterations: int = _INVERSE_ITERATIONS) -> np.ndarray:
    """求「输入点会落到输出的哪里」，即 cv2.remap 的逆映射。

    remap 定义 output(x,y) = input(x + dx(x,y), y + dy(x,y))，所以输入点 u 的
    输出位置 x 满足 x + dx(x) = u，即不动点方程 x = u - dx(x)。光滑场里 dx 变化
    缓慢，迭代逐轮收缩（实测 alpha/sigma=1.67 时收敛因子 0.41、0.60 时 0.28）。

    收敛后会把解代回定义式验收 x + dx(x) == u；残差超限直接报错。位移场相对
    sigma 变化太快时（实测 alpha/sigma 到 3.33 收敛因子已退化到 0.92，到 5.0
    则放大到 1.17 而发散）宁可不产出，也不产出一批系统性错位的标签。
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if points.size == 0:
        return points.copy()

    target_x = points[:, 0].copy()
    target_y = points[:, 1].copy()
    xs = points[:, 0].copy()
    ys = points[:, 1].copy()

    for _ in range(max(1, int(iterations))):
        next_x = target_x - _sample_field(dx, xs, ys)
        next_y = target_y - _sample_field(dy, xs, ys)
        shift = max(float(np.abs(next_x - xs).max()), float(np.abs(next_y - ys).max()))
        xs, ys = next_x, next_y
        if shift < _INVERSE_TOLERANCE:
            break

    residual = max(float(np.abs(xs + _sample_field(dx, xs, ys) - target_x).max()),
                   float(np.abs(ys + _sample_field(dy, xs, ys) - target_y).max()))
    if residual > _INVERSE_MAX_RESIDUAL:
        raise ValueError(
            f"弹性形变的正向映射未收敛（最大残差 {residual:.3f} 像素，{iterations} 轮）。"
            "位移场相对 sigma 变化太快时会这样：请增大 sigma 或减小 alpha。"
        )
    return np.stack([xs, ys], axis=1)


def warp_image(image, dx: np.ndarray, dy: np.ndarray,
               interpolation: int = cv2.INTER_LINEAR,
               border: int = cv2.BORDER_REFLECT_101) -> np.ndarray:
    """按位移场重采样图像。

    原图用 INTER_LINEAR；要在测试里比对掩码同步性时用 INTER_NEAREST，
    因为掩码不能插值出 0.5 这种中间值。
    """
    grid_x, grid_y = image_grid(np.asarray(image).shape)
    map_x = (grid_x + dx).astype(np.float32)
    map_y = (grid_y + dy).astype(np.float32)
    return cv2.remap(np.asarray(image), map_x, map_y, interpolation, borderMode=border)


def warp_polygons(instances: Sequence[Instance], dx: np.ndarray, dy: np.ndarray,
                  height: int, width: int) -> List[Instance]:
    """把多边形顶点按同一个位移场搬到输出坐标系，并夹回画布内。

    顶点数不变（形变是逐点映射），夹取只影响被推出画布的顶点；若某条多边形
    夹取后不足 3 点则丢弃，因为 YOLO 标签无法表示一条线或一个点。
    """
    warped: List[Instance] = []
    for class_id, polygon in instances:
        points = polygon_to_pixels(polygon, height, width)
        moved = forward_map(points, dx, dy)
        normalized = pixels_to_polygon(moved, height, width)
        if len(normalized) >= MIN_POLYGON_POINTS:
            warped.append((int(class_id), normalized))
    return warped


# ============================================================ 单样本入口
def augment_sample(image, instances, clahe=None, elastic=None, seed: int = 0,
                   border: int = cv2.BORDER_REFLECT_101):
    """对一个样本做增强，返回 (新图像, 新实例列表)。

    clahe / elastic 传 None（默认）表示该项关闭；传 True 用默认超参；传 dict
    可只覆盖其中若干项。两项都关闭时返回原图与数值完全相同的标签 —— 这是
    「基线不被污染」的前提，也是 tests/test_augment.py 的第一条断言。

    顺序固定为「先 CLAHE（光度）再弹性形变（几何）」。两者其实可交换（CLAHE
    不移动像素），但固定顺序才能让 seed 的可复现性有明确含义。
    """
    result_image = np.asarray(image)
    result_instances: List[Instance] = [(int(class_id), np.array(polygon, dtype=np.float64))
                                        for class_id, polygon in instances]

    if clahe:
        clahe_options = CLAHE_DEFAULTS if clahe is True else dict(clahe)
        result_image = apply_clahe(result_image, **clahe_options)

    if elastic:
        elastic_options = dict(ELASTIC_DEFAULTS)
        if isinstance(elastic, dict):
            elastic_options.update(elastic)
        height, width = result_image.shape[:2]
        dx, dy = displacement_field((height, width), seed=seed, **elastic_options)
        result_instances = warp_polygons(result_instances, dx, dy, height, width)
        result_image = warp_image(result_image, dx, dy, border=border)

    return result_image, result_instances


# ========================================================== 数据集层面
def augment_split_dir(source_dir, output_dir, copies: int = 1, clahe=None,
                      elastic=None, seed: int = 0,
                      augment_splits: Sequence[str] = ("train",),
                      border: int = cv2.BORDER_REFLECT_101) -> dict:
    """把 segmentation/data 复制到新目录，只给指定 split 增补增强样本。

    为什么 val/ 与 test/ 必须逐字节复制
    ----------------------------------
    增强只应该扩大训练分布。若连 val/test 一起增强，评估数字就和基线不可比了
    （同一个模型在增强过的 test 上通常更好看），消融实验随即失去意义。所以对
    未列在 augment_splits 里的 split 用 shutil.copy2 原样复制，不解码不重编码。

    为什么保留原图
    ------------
    增强是**增补**而非替换：原图全部保留，另生成 copies 份增强副本。否则一旦
    增强引入分布偏移，就没有「有增强 / 无增强」之外的任何对照了。代价是训练
    时间按 (1+copies) 倍增长 —— 624 张图 copies=1 时约翻倍。

    背景图（空标签）同样参与增强：BUSI 有 102 张无病灶图，模型必须见过「没有
    病灶长什么样」，只是它们的标签增强后依然是空文件。

    返回 {split: {'images': n, 'labels': n, 'augmented': n, 'skipped': [...]}}。
    """
    if copies < 0:
        raise ValueError(f"copies 不能为负数，收到 {copies}")

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"找不到增强源目录：{source_dir}")
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("输出目录不能与源目录相同，否则原图会被覆盖")

    augment_splits = tuple(augment_splits)

    if copies > 0 and elastic:
        # 先用一张小图试算，把「位移场不收敛」这类超参错误挡在写文件之前，
        # 免得留下半个数据集再让人手动清理。
        probe = np.zeros((64, 64), np.uint8)
        augment_sample(probe, [(0, np.array([[0.3, 0.3], [0.7, 0.3], [0.7, 0.7]]))],
                       clahe=clahe, elastic=elastic, seed=seed, border=border)

    stats: dict = {}

    for split_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        split = split_dir.name
        source_images = split_dir / "images"
        source_labels = split_dir / "labels"
        target_images = output_dir / split / "images"
        target_labels = output_dir / split / "labels"
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        record = {"images": 0, "labels": 0, "augmented": 0, "skipped": []}
        images = sorted(path for path in source_images.glob("*") if path.is_file()) \
            if source_images.is_dir() else []

        for position, image_path in enumerate(images):
            label_path = source_labels / f"{image_path.stem}.txt"
            label_text = label_path.read_text(encoding="utf-8") if label_path.is_file() else ""

            # 原图与原标签先原样落地（增强是增补，不替换）
            shutil.copy2(image_path, target_images / image_path.name)
            record["images"] += 1
            if label_text:
                shutil.copy2(label_path, target_labels / label_path.name)
            else:
                (target_labels / f"{image_path.stem}.txt").write_text("", encoding="utf-8")
            record["labels"] += 1

            if split not in augment_splits or copies == 0:
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                record["skipped"].append(image_path.name)
                continue

            for index in range(1, copies + 1):
                # 用「排序后的位置」派生种子，与文件名无关，同一次运行必然可复现
                sample_seed = seed * 100003 + position * 97 + index
                augmented, instances = augment_sample(
                    image, load_label(label_text), clahe=clahe, elastic=elastic,
                    seed=sample_seed, border=border)
                stem = f"{image_path.stem}_aug{index}"
                cv2.imwrite(str(target_images / f"{stem}{image_path.suffix}"), augmented)
                (target_labels / f"{stem}.txt").write_text(
                    dump_label(instances), encoding="utf-8")
                record["augmented"] += 1

        stats[split] = record

    return stats


# ================================================================= 自演示
def _demo() -> None:
    """python augment.py —— 合成样本演示增强效果，并验证顶点确实同步。"""
    import metrics  # 只在演示里用到，避免模块级多一个耦合

    height, width = 128, 160
    rng = np.random.default_rng(0)
    base = np.linspace(30, 150, width, dtype=np.float32)[None, :].repeat(height, 0)
    image = np.clip(base + rng.normal(0, 12, (height, width)), 0, 255).astype(np.uint8)
    cv2.circle(image, (86, 62), 22, (215, 215, 215), -1)  # 高回声病灶
    image = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 1.5), cv2.COLOR_GRAY2BGR)

    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    polygon = np.stack([(86 + 22 * np.cos(angles)) / width,
                        (62 + 22 * np.sin(angles)) / height], axis=1)
    instances = [(1, polygon)]

    print("== CLAHE（光度变换，标签不应变化）==")
    enhanced, clahe_labels = augment_sample(image, instances, clahe=True)
    print(f"  对比度(标准差) {image.std():.2f} -> {enhanced.std():.2f}")
    print(f"  标签完全未变：{np.allclose(clahe_labels[0][1], polygon)}")

    print("\n== 弹性形变（几何变换，标签必须同步）==")
    _, warped_labels = augment_sample(image, instances, elastic={"alpha": 10.0})
    dx, dy = displacement_field((height, width), alpha=10.0, sigma=6.0, seed=0)
    source_mask = metrics.polygon_to_mask([polygon.reshape(-1)], height, width)
    remapped = warp_image(source_mask, dx, dy, interpolation=cv2.INTER_NEAREST,
                          border=cv2.BORDER_CONSTANT)
    moved_mask = metrics.polygon_to_mask([warped_labels[0][1].reshape(-1)], height, width)
    print(f"  顶点同步变换 vs 掩码 warp 的 IoU = {metrics.iou(moved_mask, remapped):.4f}")

    # 反例：把位移场直接加到顶点上（常见错误做法），同步性明显更差
    points = polygon_to_pixels(polygon, height, width)
    naive = points + np.stack([_sample_field(dx, points[:, 0], points[:, 1]),
                               _sample_field(dy, points[:, 0], points[:, 1])], axis=1)
    naive_polygon = pixels_to_polygon(naive, height, width)
    naive_mask = metrics.polygon_to_mask([naive_polygon.reshape(-1)], height, width)
    print(f"  位移直接加顶点上（错误做法）的 IoU = {metrics.iou(naive_mask, remapped):.4f}")

    print("\n== 两项都关闭（基线不被污染）==")
    same_image, same_labels = augment_sample(image, instances)
    print(f"  图像未变：{same_image is image}   标签未变：{np.allclose(same_labels[0][1], polygon)}")


if __name__ == "__main__":
    _demo()
