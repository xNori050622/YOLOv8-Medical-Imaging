"""统一的训练入口。

三个任务的训练流程完全一致，差别只在「数据集形态 / 默认 imgsz / 输出目录」，
所以集中在这里。detection/classification/segmentation 三个模块的 train()
只是薄封装，保持原有调用方式不变。

超参默认值对齐 runs/*/train/args.yaml —— 也就是原作者在 Colab 上跑出
仓库内 best.pt 时用的配置：epochs=100、batch=16、seed=0、deterministic=True。
"""
from __future__ import annotations

import shutil
from pathlib import Path

import config

# 每个任务的差异点
TASK_PROFILES = {
    "detect": {
        "data_kind": "detect_yaml",
        "data": config.DETECT_DATA_YAML,
        "imgsz": config.DEFAULT_IMGSZ["detect"],
        "runs": config.RUNS_DIR / "detect",
        "weights": config.DETECT_WEIGHTS,
    },
    "classify": {
        "data_kind": "dir",
        "data": config.CLASSIFY_DATA_DIR,
        "imgsz": config.DEFAULT_IMGSZ["classify"],
        "runs": config.RUNS_DIR / "classify",
        "weights": config.CLASSIFY_WEIGHTS,
    },
    "segment": {
        "data_kind": "segment_yaml",
        "data": config.SEGMENT_DATA_YAML,
        "imgsz": config.DEFAULT_IMGSZ["segment"],
        "runs": config.RUNS_DIR / "segment",
        "weights": config.SEGMENT_WEIGHTS,
    },
}


def _backup_weights(path: Path):
    """训练前把已有的 best.pt 备份成 .bak，避免重训把仓库里能用的权重覆盖掉。"""
    path = Path(path)
    if not path.is_file():
        return None
    backup = path.with_name(path.name + ".bak")
    shutil.copy2(path, backup)
    return backup


def resolve_data(task: str, data=None) -> str:
    """确定传给 model.train(data=...) 的路径。

    传了 data 就用传进来的；否则用 config 里的默认位置。
    检测/分割还需要 data.yaml，若只有 images/labels 会自动补生成一份。
    """
    profile = TASK_PROFILES[task]
    kind = profile["data_kind"]

    if kind == "dir":
        directory = Path(data or profile["data"])
        if not directory.is_dir():
            raise FileNotFoundError(
                f"找不到分类数据集目录：{directory}\n"
                "请先按 README 下载 Covid19-dataset，解压到该位置。"
            )
        return str(directory)

    import dataset

    if kind == "detect_yaml":
        yaml_path = Path(data or profile["data"])
        if not yaml_path.is_file():
            # 有 images/labels 但没有 data.yaml 时自动生成
            yaml_path = dataset.build_detect_data_yaml(yaml_path.parent)
        if not yaml_path.is_file():
            raise FileNotFoundError(
                f"找不到检测数据集配置：{yaml_path}\n"
                "请把 Roboflow 导出的数据集解压到 detection/data/ 下。"
            )
        return str(yaml_path)

    yaml_path = Path(data or profile["data"])
    if not yaml_path.is_file():
        if data:
            # 显式传了 --data 却找不到文件：绝不能静默回退到默认数据集。
            # 否则「增强臂」实际拿未增强的数据训练，而实验记录上写着增强 ——
            # 消融表整张都是错的，且全过程不会报任何错。
            raise FileNotFoundError(
                f"指定的分割数据集配置不存在：{yaml_path}\n"
                "  · 增强数据集请先用 tools/make_augmented_dataset.py 生成；\n"
                "  · 想用默认数据集就不要传 --data。"
            )
        yaml_path = dataset.build_segment_data_yaml()
    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"找不到分割数据集配置：{yaml_path}\n"
            "请先准备数据：python train.py segment --prepare"
        )
    return str(yaml_path)


def train_task(task: str, data=None, epochs=None, imgsz=None, batch=None,
               device="auto", pretrained=True, name="train", exist_ok=True,
               resume=False, verbose=True, **overrides):
    """训练某个任务，返回本次训练产出的 best.pt 路径。

    参数
      task        detect / classify / segment
      data        覆盖数据集位置；detect/segment 传 data.yaml，classify 传目录
      epochs      None 时用 config.DEFAULT_EPOCHS（100）
      imgsz       None 时用该任务在 args.yaml 里的值（detect/segment 640，classify 224）
      batch       None 时用 config.DEFAULT_BATCH（16）
      device      'auto' 自动挑 GPU，也可显式传 0 / 'cpu'
      pretrained  优先用 weights/ 下的 COCO 预训练权重；没有时交给 ultralytics 下载。
                  传 False 则用 *.yaml 从零初始化，完全离线。
      name        输出到 runs/<task>/<name>/；默认 'train'，即应用读取的那个目录
      exist_ok    允许写入已存在的 runs/<task>/<name>/（会先备份其中的 best.pt）
      overrides   其余 ultralytics 训练参数（如 amp=False、workers=0、cache=True）
    """
    if task not in TASK_PROFILES:
        raise ValueError(f"未知任务：{task}，可选 {list(TASK_PROFILES)}")

    from ultralytics import YOLO

    profile = TASK_PROFILES[task]
    epochs = config.DEFAULT_EPOCHS if epochs is None else epochs
    batch = config.DEFAULT_BATCH if batch is None else batch
    imgsz = profile["imgsz"] if imgsz is None else imgsz
    data_arg = resolve_data(task, data)

    project = profile["runs"]
    target_weights = project / name / "weights" / "best.pt"
    backup = _backup_weights(target_weights) if exist_ok else None

    starting_point = config.train_starting_point(task, pretrained)
    if verbose:
        print(f"[train] task={task}")
        print(f"[train] starting from : {starting_point}")
        print(f"[train] data          : {data_arg}")
        print(f"[train] epochs={epochs} batch={batch} imgsz={imgsz} "
              f"device={config.pick_device(device)}")
        print(f"[train] output        : {project / name}")
        if backup:
            print(f"[train] 已备份原权重  : {backup}")

    model = YOLO(starting_point)
    model.train(
        data=data_arg,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=config.pick_device(device),
        project=str(project),
        name=name,
        exist_ok=exist_ok,
        resume=resume,
        # 与 args.yaml 对齐：固定随机性，保证结果可复现
        seed=0,
        deterministic=True,
        **overrides,
    )

    if verbose:
        print(f"[train] 完成，权重：{target_weights}")
    return target_weights
