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

# ------------------------------------------------------------ 训练相关配置
# 基础（COCO 预训练）权重目录。目录内的 *.pt 不入库，可用 tools/get_weights.py 下载。
WEIGHTS_DIR: Path = PROJECT_ROOT / "weights"

# 三个任务的模型骨架（同时用于拼 .pt 预训练权重名与 .yaml 结构文件名）
MODEL_STEMS: Dict[str, str] = {
    "detect": "yolov8n",
    "classify": "yolov8n-cls",
    "segment": "yolov8n-seg",
}

# 数据集位置。沿用原仓库的目录约定，这样按原 README 下载数据集后无需改名。
DETECT_DATA_YAML: Path = PROJECT_ROOT / "detection" / "data" / "data.yaml"
CLASSIFY_DATA_DIR: Path = PROJECT_ROOT / "classification" / "Covid19-dataset"
SEGMENT_RAW_DIR: Path = PROJECT_ROOT / "segmentation" / "Dataset_BUSI_with_GT"
SEGMENT_SPLIT_DIR: Path = PROJECT_ROOT / "segmentation" / "data"
SEGMENT_DATA_YAML: Path = PROJECT_ROOT / "segmentation" / "data.yaml"

# 类别定义。取自仓库内三个已训练权重的 names，顺序必须一致，
# 否则重训后的 best.pt 与界面上的类别名会对不上。
DETECT_CLASS_NAMES: Tuple[str, ...] = ("Platelets", "RBC", "WBC")
CLASSIFY_CLASS_NAMES: Tuple[str, ...] = ("Covid", "Normal", "Viral Pneumonia")
SEGMENT_CLASS_NAMES: Tuple[str, ...] = ("benign", "malignant", "normal")

# 训练超参。数值对齐 runs/*/train/args.yaml 里原作者在 Colab 上的配置。
DEFAULT_EPOCHS: int = 100
DEFAULT_BATCH: int = 16
DEFAULT_IMGSZ: Dict[str, int] = {"detect": 640, "classify": 224, "segment": 640}



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


# ------------------------------------------------------------ 训练辅助
def base_weights(task: str) -> Path:
    """返回某任务的基础（COCO 预训练）权重路径。"""
    return WEIGHTS_DIR / f"{MODEL_STEMS[task]}.pt"


def train_starting_point(task: str, pretrained: bool = True) -> str:
    """决定训练从哪份模型出发，返回可传给 YOLO() 的字符串。

    pretrained=True 且 weights/ 下已有对应 .pt —— 返回绝对路径，完全离线；
    pretrained=True 但本地没有 —— 返回裸文件名，交给 ultralytics 自行下载
      （官方 release 在国内可能很慢，建议先跑 tools/get_weights.py）；
    pretrained=False —— 返回 *.yaml，按结构从零初始化，不需要联网。
    """
    if not pretrained:
        return f"{MODEL_STEMS[task]}.yaml"
    local = base_weights(task)
    if local.is_file():
        return str(local)
    return f"{MODEL_STEMS[task]}.pt"


def pick_device(prefer="auto"):
    """选择训练设备。

    prefer='auto' 时优先用 CUDA；没有 GPU 或 torch 未安装则回退 'cpu'。
    也可以直接传 0 / 'cuda:0' / 'cpu' 等，原样返回。
    """
    if prefer != "auto":
        return prefer
    try:
        import torch
    except ImportError:
        return "cpu"
    return 0 if torch.cuda.is_available() else "cpu"


def check_training_assets() -> Dict[str, Dict[str, bool]]:
    """训练前的自检：基础权重与数据集是否就位。

    返回 {任务名: {'weights': bool, 'dataset': bool}}，供 train.py check 使用。
    """
    stems_ready = {task: base_weights(task).is_file() for task in MODEL_STEMS}
    return {
        "detect": {
            "weights": stems_ready["detect"],
            "dataset": DETECT_DATA_YAML.is_file(),
        },
        "classify": {
            "weights": stems_ready["classify"],
            "dataset": CLASSIFY_DATA_DIR.is_dir(),
        },
        "segment": {
            "weights": stems_ready["segment"],
            "dataset": SEGMENT_DATA_YAML.is_file(),
        },
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
