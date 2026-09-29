import numpy as np

import config
import training


def train(**kwargs):
    """训练检测模型。

    原实现把数据集路径硬编码成 D:\\computer-vision\\projects\\...，换台机器
    必然找不到，且用 YOLO('yolov8n.yaml') 从零开始。现在统一走 config 与
    training.train_task：默认用 weights/yolov8n.pt（COCO 预训练）作为起点，
    输出到 runs/detect/train/，也就是应用实际读取权重的目录。

    常用调用：
        python train.py detect                     # 100 epoch
        python train.py detect --epochs 1 --device cpu   # 冒烟
    也可以命令行直接调 ultralytics：
        yolo detect train data=<data.yaml> model=yolov8n.pt epochs=100
    """
    return training.train_task("detect", **kwargs)


def predict(img, confidence=config.DEFAULT_CONFIDENCE, filename=""):
    """目标检测。

    参数
      img        : BGR 图像数组（OpenCV 读入）
      confidence : 置信度阈值
      filename   : 来源文件名，写进结果里方便导出时对应

    返回结构化 dict：
      task       任务类型
      detections [{class_id, class_name, confidence, box:[x1,y1,x2,y2]}, ...]
      counts     {类别名: 数量}
      plot_rgb   带检测框的 RGB 图像数组
    """
    model = config.load_model(config.DETECT_WEIGHTS)

    results = model.predict(img, conf=confidence, verbose=False)
    result = results[0]
    names = result.names

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            detections.append({
                "class_id": class_id,
                "class_name": names.get(class_id, str(class_id)),
                "confidence": round(float(box.conf[0]), 4),
                "box": [x1, y1, x2, y2],
            })

    counts = {}
    for det in detections:
        counts[det["class_name"]] = counts.get(det["class_name"], 0) + 1

    print(f"\n[INFO] Number of objects detected: {len(detections)}")

    # result.plot() 返回 BGR 数组，转成 RGB 以便显示与导出 PNG
    plot_rgb = np.ascontiguousarray(result.plot()[..., ::-1])

    return {
        "task": "detect",
        "filename": filename,
        "num_objects": len(detections),
        "detections": detections,
        "counts": counts,
        "class_names": dict(names),
        "plot_rgb": plot_rgb,
    }
