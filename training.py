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


_CHECKPOINT_SAVE_HARDENED = False


def harden_checkpoint_saving(attempts: int = 3, verbose: bool = True) -> bool:
    """加固 ultralytics 的 checkpoint 保存：要么完好落盘，要么报错，绝不写坏或写崩。

    为什么需要（ultralytics 8.4.135 实测，2026-10）：

    1. ``ultralytics/utils/__init__.py`` 在 import 时执行 ``torch.save = torch_save``，
       把 ``torch.save`` 全局换成了 ``utils/patches.py::torch_save`` —— 一个
       ``except RuntimeError`` 的重试循环（为「写文件」设计：每次重试都重新打开文件）。
    2. 但 ``Trainer.save_model()`` 先把整个 checkpoint 序列化进一个 ``io.BytesIO``，
       再 ``torch.save(ckpt, buffer)``。重试循环会**复用同一个 buffer**，于是：
         · 上一次尝试写了一半的 zip 被续写 → 重试「成功」，但 .pt 已损坏
           （实测 ``PytorchStreamReader failed reading zip archive: invalid header
           or archive is corrupted``）—— 静默数据损失，最坏的一种；
         · buffer 被关闭后再写 → ``ValueError: I/O operation on closed file`` 从
           ``write_record("data.pkl")`` 抛出，而 ``ValueError`` 不在
           ``except RuntimeError`` 的捕获范围内 → 整个训练直接崩掉
           （实测：100 epoch 跑到第 8 轮时崩，前 8 轮的计算全部白费）。
    3. 这里做两件事，把上面两个坑一起堵住：
         · 目标不是路径（即内存 buffer）时**不重试**，失败立刻抛出；
         · 在 ``Trainer.save_model`` 外面套一层重试，每次重试用**全新的 buffer**，
           于是「瞬时失败」能被真正救回来（这才是原作者想要的重试语义）。

    返回 True 表示本次确实安装；重复调用是幂等的（多次调用不会叠加包装）。
    """
    global _CHECKPOINT_SAVE_HARDENED
    if _CHECKPOINT_SAVE_HARDENED:
        return False

    import os
    import time

    import torch
    from torch.serialization import save as _plain_save
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.utils.patches import torch_save as _retrying_save

    def save(obj, f, *args, **kwargs):
        """写文件路径时保留 ultralytics 的重试；写内存 buffer 时只尝试一次。"""
        if isinstance(f, (str, os.PathLike)):
            return _retrying_save(obj, f, *args, **kwargs)
        return _plain_save(obj, f, *args, **kwargs)

    save._yolo_hardened = True
    torch.save = save

    # save_model 只定义在 BaseTrainer 上，detect/classify/segment 的 trainer 都继承它，
    # 所以补一处即可覆盖三个任务。
    original_save_model = BaseTrainer.save_model
    rounds = max(1, attempts)

    def save_model(self):
        """与 ultralytics 原版等价，区别只是整次失败后会换一个新 buffer 重来。"""
        for i in range(rounds):
            try:
                return original_save_model(self)
            except (OSError, RuntimeError, ValueError) as exc:
                if i == rounds - 1:
                    raise
                wait = 0.5 * 2 ** i
                print(f"[train] 保存 checkpoint 失败（第 {i + 1}/{rounds} 次，"
                      f"epoch {self.epoch + 1}）：{type(exc).__name__}: {exc}")
                print(f"[train] {wait:.1f}s 后用全新的内存 buffer 重试 ……")
                time.sleep(wait)

    save_model._yolo_hardened = True
    BaseTrainer.save_model = save_model

    _CHECKPOINT_SAVE_HARDENED = True
    if verbose:
        print("[train] 已加固 checkpoint 保存：内存 buffer 不重试 + save_model 整次重试")
    return True


def _commit_usage():
    """返回 (已提交 GB, 提交上限 GB, 可用 GB)；非 Windows 或调用失败时返回 None。

    Windows 的「提交内存 / commit charge」= 所有进程已申请的全部虚拟内存之和，
    上限 = 物理内存 + 页文件。C 进程 malloc 失败（哪怕只申请 1 MB）几乎都发生
    在提交上限被顶满时，而不是物理内存不够。
    """
    import ctypes
    import os

    if os.name != "nt":
        return None

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = _MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    try:
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return None
    except OSError:
        return None

    gib = 1024 ** 3
    total = stat.ullTotalPageFile / gib
    avail = stat.ullAvailPageFile / gib
    return total - avail, total, avail


def _worker_count():
    """当前进程派生的子进程数（DataLoader worker 用 spawn，各算一个进程）。"""
    try:
        import psutil
    except ImportError:
        return None
    try:
        return len(psutil.Process().children(recursive=False))
    except Exception:
        return None


class _MemoryWatchdog:
    """训练期间每隔一段时间打印一次提交内存余量，快撞墙时提前报警。

    为什么需要它：训练/验证两套 DataLoader 各起 args.workers 个常驻 worker
    （ultralytics 的 validator 只在 _setup_train 建一次、dataloader 复用，
    所以 worker 不会每轮累积，但会一直同时存活）。每个 worker 都要带着一份
    torch 常驻，worker 一多，提交内存就被顶满。实测（本机 16 GB 内存 +
    19.7 GB 页文件，提交上限约 35.5 GB，workers=8 → 16 个 worker）跑到第 12
    轮时，某个 worker 里 cv2 连 1.35 MB 都申请不到：

        cv2.error: ... (-4:Insufficient memory) Failed to allocate 1354752 bytes
        in function 'cv::OutOfMemoryError'

    整个训练当场终止，已跑的 12 轮全部作废。所以真正的瓶颈不是显存而是
    worker 数；把余量打出来，就能在崩之前看到它。

    打印格式：``[mem] 提交 27.3/35.5 GB（可用 8.2 GB），物理可用 5.1 GB，子进程 16``
    """

    INTERVAL = 60.0  # 秒
    WARN_GB = 4.0    # 可用提交内存低于此值就打警告

    def __init__(self):
        import threading

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._warned = False

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.wait(self.INTERVAL):
            self.sample()

    def sample(self):
        commit = _commit_usage()
        workers = _worker_count()
        if commit is None:
            return

        used, total, avail = commit
        pieces = [f"提交 {used:.1f}/{total:.1f} GB（可用 {avail:.1f} GB）"]

        try:
            import psutil

            vm = psutil.virtual_memory()
            pieces.append(f"物理可用 {vm.available / 1024 ** 3:.1f} GB")
        except ImportError:
            pass
        if workers is not None:
            pieces.append(f"子进程 {workers}")

        line = "[mem] " + "，".join(pieces)
        if avail < self.WARN_GB:
            if not self._warned:
                hint = "把 --workers 减半（或设 0），并关掉不用的大程序（浏览器/聊天软件）"
                if workers:
                    hint += f"；当前已有 {workers} 个 worker 进程，每个约占 1.2 GB"
                line += (f"\n[mem] !! 提交内存快用完了（可用 < {self.WARN_GB:.0f} GB），"
                         f"训练随时可能因 worker 里 malloc 失败而崩。"
                         f"\n[mem] !! 应急办法：Ctrl+C，然后{hint}，再用 --resume 接上。")
                self._warned = True
        else:
            self._warned = False
        print(line, flush=True)


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
      resume      从 runs/<task>/<name>/weights/last.pt 继续未跑完的训练。
                  True 时下面这些参数会被忽略，一律沿用 checkpoint 里记录的配置
                  （epochs / data / batch / imgsz / amp / seed ...），
                  否则「续训」会变成「换一套超参照着半个模型接着跑」，实验记录不可比。
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
    last_weights = project / name / "weights" / "last.pt"

    if resume:
        # 续训 = 同一次训练的后半段，不再备份：.bak 记的是「本次训练开始前」的权重，
        # 再备份一次会把上一次留下的、真正原始的权重挤掉。
        if not last_weights.is_file():
            raise FileNotFoundError(
                f"找不到可续训的 checkpoint：{last_weights}\n"
                "只有训练被中断/未跑完时才会有 last.pt；想从头训练请不要传 --resume。"
            )
        backup = None
        starting_point = str(last_weights)
    else:
        backup = _backup_weights(target_weights) if exist_ok else None
        starting_point = config.train_starting_point(task, pretrained)

    if verbose:
        print(f"[train] task={task}")
        print(f"[train] starting from : {starting_point}")
        print(f"[train] data          : {data_arg}")
        if resume:
            print("[train] 断点续训      : epochs/batch/imgsz/amp/seed 等沿用 checkpoint")
        print(f"[train] epochs={epochs} batch={batch} imgsz={imgsz} "
              f"device={config.pick_device(device)}")
        print(f"[train] output        : {project / name}")
        if backup:
            print(f"[train] 已备份原权重  : {backup}")

    harden_checkpoint_saving(verbose=verbose)

    model = YOLO(starting_point)
    watchdog = _MemoryWatchdog().start() if verbose else None
    if watchdog:
        watchdog.sample()  # 先打一次基线，便于对比后面涨了多少
    try:
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
    finally:
        if watchdog:
            watchdog.stop()

    if verbose:
        print(f"[train] 完成，权重：{target_weights}")
    return target_weights
