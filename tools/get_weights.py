"""下载训练需要的三个基础（COCO / ImageNet 预训练）权重到 weights/。

weights/*.pt 不入库（体积约 19 MB），所以刚克隆下来的仓库里是没有的。
ultralytics 在缺权重时也会自己去下，但官方 release 从部分网络访问很慢甚至卡死，
本脚本用「断点续传 + 短超时 + 多次重试」把它做得更稳。

用法：
    python tools/get_weights.py             # 缺什么下什么
    python tools/get_weights.py --force     # 全部重新下载
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

ASSET_BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
FILENAMES = ("yolov8n.pt", "yolov8n-cls.pt", "yolov8n-seg.pt")

# 单次尝试的读超时。卡住的连接会在这么多秒后抛错，然后带着断点重试。
READ_TIMEOUT = 30
MAX_ATTEMPTS = 40
CHUNK = 1 << 16
# .pt 是 zip 容器，前 4 字节固定为 PK\x03\x04；小于这个大小必然是坏的
MIN_SIZE = 1 << 20


def _size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def _progress(done: int, total) -> str:
    if not total:
        return f"{done / 1048576:.1f} MB"
    return f"{done / 1048576:.1f} / {total / 1048576:.1f} MB ({100 * done / total:.0f}%)"


def download(url: str, target: Path) -> bool:
    """下载 url 到 target，支持断点续传；返回是否完整。"""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        start = _size(target)
        headers = {"User-Agent": "Mozilla/5.0"}
        if start:
            headers["Range"] = f"bytes={start}-"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=READ_TIMEOUT) as response:
                # 服务器忽略了 Range 时会返回 200 而不是 206，那就得从头写
                if start and response.status != 206:
                    start = 0
                total = None
                if response.headers.get("Content-Length"):
                    total = start + int(response.headers["Content-Length"])

                with open(target, "ab" if start else "wb") as handle:
                    done = start
                    while True:
                        chunk = response.read(CHUNK)
                        if not chunk:
                            break
                        handle.write(chunk)
                        done += len(chunk)
                        print(f"\r  {_progress(done, total)}   ", end="", flush=True)

            print()
            if total is None or _size(target) >= total:
                return True
            print(f"  连接中断，从 {_progress(_size(target), total)} 处续传"
                  f"（第 {attempt} 次重试）")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print()
            have = _size(target)
            print(f"  第 {attempt} 次尝试失败：{type(exc).__name__}: {str(exc)[:80]}")
            if have:
                print(f"  已下载 {have / 1048576:.1f} MB，将从这个位置续传")
        except KeyboardInterrupt:
            print("\n  已取消")
            return False
    return False


def verify(path: Path) -> str:
    """粗略校验：大小够、且确实是 zip 容器。返回 'OK' 或问题描述。"""
    size = _size(path)
    if size < MIN_SIZE:
        return f"文件过小（{size} 字节）"
    with open(path, "rb") as handle:
        if handle.read(4) != b"PK\x03\x04":
            return "不是有效的 torch 权重（缺少 zip 头）"
    try:
        with zipfile.ZipFile(path) as archive:
            if not archive.namelist():
                return "zip 容器为空"
    except zipfile.BadZipFile as exc:
        return f"zip 结构损坏：{exc}"
    return "OK"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="下载训练用的基础权重")
    parser.add_argument("--force", action="store_true", help="已存在的也重新下载")
    parser.add_argument("--out", default=None, help="输出目录，默认 weights/")
    args = parser.parse_args(argv)

    out_dir = Path(args.out or config.WEIGHTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"权重目录：{out_dir}\n")

    failed = []
    for name in FILENAMES:
        target = out_dir / name
        if target.is_file() and not args.force:
            status = verify(target)
            if status == "OK":
                print(f"[跳过] {name} 已存在且校验通过（{_size(target)} 字节）")
                continue
            print(f"[重下] {name} 已存在但{status}")

        print(f"[下载] {name}")
        if not download(f"{ASSET_BASE}/{name}", target):
            failed.append(name)
            continue

        status = verify(target)
        if status == "OK":
            print(f"[完成] {name}  {_size(target)} 字节")
        else:
            failed.append(name)
            print(f"[损坏] {name}  {status}")

    print()
    if failed:
        print(f"以下权重未就绪：{failed}")
        print("可以重跑本脚本续传；或训练时加 --no-pretrained 从零开始（不需要联网）。")
        return 1
    print("三个基础权重均已就绪，现在可以直接运行 python train.py <task>。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
