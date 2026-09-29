from segmentation.masks_to_polygons import *
import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw

import config
import training


def prepare_input():
    """把 BUSI 原始数据集准备成 YOLO 分割数据集，返回 data.yaml 路径。

    步骤：Dataset_BUSI_with_GT 的掩码转多边形标签 -> 按 8:1:1 划分 train/val/test
    -> 生成 segmentation/data.yaml（原仓库缺了这一步，导致 train() 必然报错）。
    """
    masks_to_polygons()
    split_train_test_val()

    import dataset

    yaml_path = dataset.build_segment_data_yaml()
    print(f"[prepare] 已生成分割数据集配置：{yaml_path}")
    return yaml_path


def train(**kwargs):
    """训练分割模型。

    数据集来自 prepare_input() 产出的 segmentation/data/ 与 segmentation/data.yaml，
    输出到 runs/segment/train/，也就是应用实际读取权重的目录。

    注意 BUSI 的 normal/ 只有图片没有掩码，所以 normal 类没有正样本，
    只作为背景图参与训练——这与仓库里既有权重的 nc=3 保持一致。

    常用调用：
        python train.py segment --prepare   # 先准备数据
        python train.py segment             # 再训练
    """
    return training.train_task("segment", **kwargs)


def predict(img, confidence=config.DEFAULT_CONFIDENCE, filename=""):
    """实例分割。

    参数
      img        : BGR 图像数组（OpenCV 读入）
      confidence : 置信度阈值
      filename   : 来源文件名

    返回结构化 dict：
      instances  [{class_id, class_name, confidence, polygon:[[x,y],...]}, ...]
      counts     {类别名: 数量}
      mask_gray  所有掩码按位或合并后的灰度图（uint8，无掩码时为 None）
      plot_rgb   在原图上描出掩码轮廓后的 RGB 图像数组
    """
    model = config.load_model(config.SEGMENT_WEIGHTS)

    H, W = img.shape[:2]
    results = model.predict(img, conf=confidence, verbose=False)
    result = results[0]
    names = result.names

    instances = []
    mask_out = None
    if result.masks is not None:
        for index, one in enumerate(result.masks):
            # 单个掩码 -> 灰度图，并缩放到原图尺寸
            mask_gray = cv.resize(one.data[0].numpy() * 255, (W, H)).astype(np.uint8)
            mask_out = mask_gray if mask_out is None else cv.bitwise_or(mask_out, mask_gray)

            boxes = result.boxes
            class_id = int(boxes.cls[index]) if boxes is not None else -1
            score = float(boxes.conf[index]) if boxes is not None else 0.0
            instances.append({
                "class_id": class_id,
                "class_name": names.get(class_id, str(class_id)),
                "confidence": round(score, 4),
                "polygon": [[float(x), float(y)] for x, y in one.xy[0]],
            })

    print(f"\n[INFO] Number of masks detected: {len(instances)}")

    counts = {}
    for inst in instances:
        counts[inst["class_name"]] = counts.get(inst["class_name"], 0) + 1

    # 在原图上描出轮廓。原实现直接把 BGR 数组交给 PIL 再按 RGB 显示，
    # 颜色红蓝会互换；这里先转成 RGB 再画，显示与导出 PNG 的颜色才正确。
    overlay = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(overlay)
    for inst in instances:
        draw.polygon([tuple(point) for point in inst["polygon"]], outline=(0, 255, 0), width=5)

    return {
        "task": "segment",
        "filename": filename,
        "num_instances": len(instances),
        "instances": instances,
        "counts": counts,
        "class_names": dict(names),
        "mask_gray": mask_out,
        "plot_rgb": np.array(overlay),
    }
