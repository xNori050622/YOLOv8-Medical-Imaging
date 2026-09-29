"""结果导出：把 predict() 返回的结构化结果转成 CSV / JSON / PNG / ZIP。

三个任务的 predict() 统一返回 dict，其中：
  - 结构化字段（detections / probs / instances / counts ...）可序列化；
  - plot_rgb / plot_bgr / mask_gray 是图像数组，序列化时自动跳过；
  - filename 记录来源文件名（Web 端上传名或演示图名）。
"""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

# CSV 列顺序。三个任务共用同一套列，无关列留空，
# 这样批量导出后可以直接拼成一张表。
FIELDNAMES: List[str] = [
    "filename",
    "task",
    "rank",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "width",
    "height",
    "area",
    "polygon",
]

# predict() 结果中属于「图像对象」的键，序列化时跳过
_ARRAY_KEYS = {"plot_rgb", "plot_bgr", "mask_gray"}


# ---------------------------------------------------------------- 工具
def _r(value, ndigits: int = 2):
    """numpy 标量/浮点 -> 保留指定小数的原生 float；失败则原样返回。"""
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return value


def safe_name(filename: Optional[str], fallback: str = "image") -> str:
    """把文件名清洗成可安全用于压缩包内的名字（去掉扩展名与非法字符）。"""
    stem = str(filename or fallback).replace("\\", "/").split("/")[-1]
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", stem) or fallback
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", stem) or fallback


def _blank_row(filename: str, task: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {key: "" for key in FIELDNAMES}
    row["filename"] = filename
    row["task"] = task
    return row


# ------------------------------------------------------- 结构化 -> 表格
def results_to_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把单张图的结果拍平成「一行一个目标」的表格行。

    - detect  : 每个检测框一行
    - classify: 每个类别一行（rank 为置信度排名）
    - segment : 每个实例掩码一行（polygon 为归一化坐标点串）
    """
    task = results.get("task", "")
    filename = results.get("filename", "")
    rows: List[Dict[str, Any]] = []

    if task == "detect":
        for det in results.get("detections", []):
            x1, y1, x2, y2 = det["box"]
            row = _blank_row(filename, task)
            row.update(
                class_id=det["class_id"],
                class_name=det["class_name"],
                confidence=_r(det["confidence"], 4),
                x1=_r(x1), y1=_r(y1), x2=_r(x2), y2=_r(y2),
                width=_r(x2 - x1), height=_r(y2 - y1),
                area=_r((x2 - x1) * (y2 - y1)),
            )
            rows.append(row)

    elif task == "classify":
        for rank, prob in enumerate(results.get("probs", []), start=1):
            row = _blank_row(filename, task)
            row.update(
                rank=rank,
                class_id=prob["class_id"],
                class_name=prob["class_name"],
                confidence=_r(prob["confidence"], 4),
            )
            rows.append(row)

    elif task == "segment":
        for inst in results.get("instances", []):
            row = _blank_row(filename, task)
            row.update(
                class_id=inst["class_id"],
                class_name=inst["class_name"],
                confidence=_r(inst["confidence"], 4),
            )
            polygon = inst.get("polygon") or []
            if polygon:
                xs = [pt[0] for pt in polygon]
                ys = [pt[1] for pt in polygon]
                row.update(
                    x1=_r(min(xs)), y1=_r(min(ys)), x2=_r(max(xs)), y2=_r(max(ys)),
                    width=_r(max(xs) - min(xs)), height=_r(max(ys) - min(ys)),
                    area=_r((max(xs) - min(xs)) * (max(ys) - min(ys))),
                )
                row["polygon"] = json.dumps(
                    [[_r(x, 1), _r(y, 1)] for x, y in polygon],
                    separators=(",", ":"),
                )
            rows.append(row)

    return rows


def results_list_to_rows(results_list: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """多张图 -> 一张表的所有行。"""
    return [row for res in results_list for row in results_to_rows(res)]


# ------------------------------------------------------------- CSV / JSON
def rows_to_csv_bytes(rows: Sequence[Dict[str, Any]]) -> bytes:
    """表格 -> CSV 字节流。

    使用 utf-8-sig（带 BOM），这样 Excel 双击打开也不会乱码。
    """
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=FIELDNAMES, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def to_serializable(results: Dict[str, Any]) -> Dict[str, Any]:
    """去掉图像数组、把 numpy 值转成原生类型，得到可 JSON 序列化的 dict。"""
    out: Dict[str, Any] = {}
    for key, value in results.items():
        if key in _ARRAY_KEYS or isinstance(value, (np.ndarray, Image.Image)):
            continue
        if isinstance(value, dict):
            out[key] = {str(k): v for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            out[key] = [to_serializable(v) if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    return out


def results_to_json_bytes(results_list: Sequence[Dict[str, Any]]) -> bytes:
    """多张图结果 -> 单个 JSON 文件字节流。"""
    payload = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "num_images": len(results_list),
        "results": [to_serializable(res) for res in results_list],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


# ------------------------------------------------------------------ 图片
def _array_to_png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr)).save(buf, format="PNG")
    return buf.getvalue()


def annotated_png_bytes(results: Dict[str, Any]) -> Optional[bytes]:
    """带标注的结果图 -> PNG 字节流（优先 RGB，回退 BGR 转 RGB）。"""
    arr = results.get("plot_rgb")
    if arr is None:
        arr = results.get("plot_bgr")
        if arr is None:
            return None
        arr = arr[..., ::-1]
    return _array_to_png(np.ascontiguousarray(arr))


def mask_png_bytes(results: Dict[str, Any]) -> Optional[bytes]:
    """分割掩码图 -> PNG 字节流（非分割任务返回 None）。"""
    arr = results.get("mask_gray")
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return _array_to_png(arr)


# ------------------------------------------------------------------- ZIP
_ZIP_README = """YOLOv8-Medical-Imaging 导出说明
=================================
results.csv   : 每个目标一行（detect=检测框 / classify=各类别概率 / segment=各掩码实例）
results.json  : 每个目标的结构化结果，含类别名、置信度、坐标（segment 含多边形）
images/       : 带标注的结果图（PNG）
masks/        : 分割任务的掩码图（PNG，仅分割任务产生）

CSV 列说明：
  filename    来源图片名
  task        任务类型 detect / classify / segment
  rank        分类任务的置信度排名（1 = 最可能）
  class_id    类别索引
  class_name  类别名称
  confidence  置信度 0~1
  x1,y1,x2,y2 外接矩形（分类任务留空）
  width,height,area 外接矩形的宽 / 高 / 面积
  polygon     分割掩码多边形坐标（归一化 0~1，形式 "[[x,y],[x,y],...]"）
"""


def build_zip(
    results_list: Sequence[Dict[str, Any]],
    rows: Optional[Sequence[Dict[str, Any]]] = None,
    include_csv: bool = True,
    include_json: bool = True,
    include_images: bool = True,
) -> bytes:
    """把多张图的全部结果打包成一个 zip 字节流。"""
    if rows is None:
        rows = results_list_to_rows(results_list)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", _ZIP_README)
        if include_csv:
            zf.writestr("results.csv", rows_to_csv_bytes(rows))
        if include_json:
            zf.writestr("results.json", results_to_json_bytes(results_list))
        if include_images:
            for index, res in enumerate(results_list, start=1):
                stem = safe_name(res.get("filename"), fallback=f"image_{index:03d}")
                png = annotated_png_bytes(res)
                if png:
                    zf.writestr(f"images/{index:03d}_{stem}.png", png)
                mask = mask_png_bytes(res)
                if mask:
                    zf.writestr(f"masks/{index:03d}_{stem}_mask.png", mask)
    return buf.getvalue()
