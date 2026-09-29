"""集中配置：项目路径 + YOLO 模型缓存。

背景：原代码用 os.path.join('.', 'runs', ...) 定位权重，隐含要求
「必须在项目根目录下启动」，换个目录就找不到 best.pt。这里统一改为
基于本文件位置(__file__)定位，使程序与当前工作目录无关。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

# ---------------------------------------------------------------- 路径
# 项目根目录（即 config.py 所在目录）
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# 演示图片目录
DEMO_IMAGES_DIR: Path = PROJECT_ROOT / "DEMO_IMAGES"

# 训练产物目录
RUNS_DIR: Path = PROJECT_ROOT / "runs"

# 三个任务的权重文件
DETECT_WEIGHTS: Path = RUNS_DIR / "detect" / "train" / "weights" / "best.pt"
CLASSIFY_WEIGHTS: Path = RUNS_DIR / "classify" / "train" / "weights" / "best.pt"
SEGMENT_WEIGHTS: Path = RUNS_DIR / "segment" / "train" / "weights" / "best.pt"

# 默认置信度阈值（与原界面滑块默认值保持一致）
DEFAULT_CONFIDENCE: float = 0.35


def demo_image(name: str) -> str:
    """返回演示图片的绝对路径。"""
    return str(DEMO_IMAGES_DIR / name)


def check_weights() -> Dict[str, Tuple[str, bool]]:
    """检查三个权重文件是否存在。

    返回 {任务名: (绝对路径, 是否存在)}，供界面做启动前自检。
    """
    return {
        "detection": (str(DETECT_WEIGHTS), DETECT_WEIGHTS.is_file()),
        "classification": (str(CLASSIFY_WEIGHTS), CLASSIFY_WEIGHTS.is_file()),
        "segmentation": (str(SEGMENT_WEIGHTS), SEGMENT_WEIGHTS.is_file()),
    }


# ------------------------------------------------------------ 模型缓存
# key = 权重文件绝对路径, value = YOLO 实例
_MODEL_CACHE: Dict[str, "object"] = {}


def load_model(weights_path) -> "object":
    """按需加载 YOLO 模型，并在进程内缓存（同一权重只加载一次）。

    这里用模块级字典而非 st.cache_resource，原因是 predict() 既要能被
    Streamlit 调用，也要能被普通 Python 脚本调用；Streamlit 每次 rerun
    不会重新 import 模块，所以缓存同样能跨 rerun 生效，避免每次交互都
    重新加载几十 MB 的权重。
    """
    key = str(weights_path)
    if key not in _MODEL_CACHE:
        if not os.path.isfile(key):
            raise FileNotFoundError(
                f"未找到模型权重文件：{key}\n"
                "请确认已完整克隆仓库（runs/*/train/weights/best.pt），"
                "或先运行 train() 训练模型。"
            )
        # 延迟导入：只用到路径常量时不必加载 torch
        from ultralytics import YOLO

        _MODEL_CACHE[key] = YOLO(key)
    return _MODEL_CACHE[key]


def clear_model_cache() -> None:
    """清空模型缓存（主要供测试使用）。"""
    _MODEL_CACHE.clear()
