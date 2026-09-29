from segmentation.masks_to_polygons import *
from ultralytics import YOLO
import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw

import config


def prepare_input():
    masks_to_polygons()
    split_train_test_val()


def train():

    model = YOLO("yolov8n-seg.yaml")
    model.train(data="D:\\computer-vision\\projects\\streamlit-dashboard\\segmentation\\data.yaml", epochs=100)


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
