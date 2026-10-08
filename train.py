"""训练 / 数据准备命令行入口。

用法（在项目根目录下执行）：

    python train.py check                    # 自检：权重与数据集是否就位
    python train.py detect                   # 训练检测模型（100 epoch）
    python train.py detect --resume          # 断点续训：从 last.pt 接着跑
    python train.py classify --epochs 1      # 冒烟：只跑 1 个 epoch
    python train.py segment --prepare        # 先准备分割数据集（掩码转多边形 + 划分）
    python train.py segment                  # 再训练
    python train.py all                      # 三个任务依次训练

默认超参与 runs/*/train/args.yaml 一致：epochs=100、batch=16，
detect/segment 的 imgsz=640，classify 的 imgsz=224。
训练产物写在 runs/<task>/<name>/，name 默认 'train'，
也就是 app.py 读取权重的目录（覆盖前会自动备份成 best.pt.bak）。
"""
from __future__ import annotations

import argparse
import importlib

import config

# 任务 -> 实现模块
TASK_MODULES = {
    "detect": "detection.detect",
    "classify": "classification.classify",
    "segment": "segmentation.segment",
}

TASK_ORDER = ("detect", "classify", "segment")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="YOLOv8-Medical-Imaging 训练入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("task", choices=[*TASK_ORDER, "all", "check"],
                        help="要训练的任务，check 只做自检")
    parser.add_argument("--data", default=None,
                        help="覆盖数据集位置：detect/segment 传 data.yaml，classify 传目录")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"训练轮数，默认 {config.DEFAULT_EPOCHS}")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="输入尺寸，默认 detect/segment=640，classify=224")
    parser.add_argument("--batch", type=int, default=None,
                        help=f"batch size，默认 {config.DEFAULT_BATCH}")
    parser.add_argument("--device", default="auto",
                        help="auto / 0 / cpu，默认 auto（有 CUDA 就用 GPU）")
    parser.add_argument("--name", default="train",
                        help="输出到 runs/<task>/<name>/，默认 train（应用读取的目录）")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                        help="不用 COCO 预训练权重，从零开始（不需要联网）")
    parser.add_argument("--prepare", action="store_true",
                        help="训练前先准备数据集（目前仅 segment 需要）")
    parser.add_argument("--workers", type=int, default=None,
                        help="DataLoader 进程数，Windows 上卡住可设 0")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="关闭混合精度；也避免 ultralytics 为 AMP 自检下载 yolo26n.pt")
    parser.add_argument("--resume", action="store_true",
                        help="从 runs/<task>/<name>/weights/last.pt 继续未跑完的训练；"
                             "epochs/data/batch/amp/seed 等一律沿用 checkpoint 里的记录")
    return parser


def _extras(args) -> dict:
    """把只有显式指定才传递的参数整理成 ultralytics 的额外关键字。"""
    extra = {}
    if args.workers is not None:
        extra["workers"] = args.workers
    if not args.amp:
        extra["amp"] = False
    return extra


def train_one(task: str, args) -> None:
    """训练单个任务（先按需准备数据，再调用该模块的 train()）。"""
    if task == "segment" and args.prepare:
        from segmentation.segment import prepare_input

        prepare_input()

    module = importlib.import_module(TASK_MODULES[task])
    weights = module.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        pretrained=args.pretrained,
        name=args.name,
        resume=args.resume,
        **_extras(args),
    )
    print(f"[train.py] {task} 完成，best.pt: {weights}\n")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.task == "check":
        import dataset

        dataset.print_report()
        return 0

    tasks = TASK_ORDER if args.task == "all" else (args.task,)
    for task in tasks:
        train_one(task, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
