import cv2 as cv

import config
import training


def train(**kwargs):
    """训练分类模型。

    数据集目录结构（类别名必须是排序后与既有权重一致的 Covid/Normal/Viral Pneumonia）：
        classification/Covid19-dataset/train/<类别>/*.jpg
        classification/Covid19-dataset/test/<类别>/*.jpg

    ultralytics 在缺 val/ 时会自动回退用 test/ 做验证（见 check_cls_dataset）。

    常用调用：
        python train.py classify                   # 100 epoch, imgsz=224
    """
    return training.train_task("classify", **kwargs)


def predict(img, filename=""):
    """图像分类。

    参数
      img      : BGR 图像数组（OpenCV 读入）
      filename : 来源文件名

    返回结构化 dict：
      top1        {class_id, class_name, confidence} 置信度最高的类别
      probs       [{class_id, class_name, confidence}, ...] 按置信度降序
      plot_rgb    在原图上标注 top1 类别后的 RGB 图像数组
    """
    model = config.load_model(config.CLASSIFY_WEIGHTS)

    results = model.predict(img, verbose=False)
    result = results[0]
    names = result.names

    ranked = sorted(
        (
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "confidence": round(float(prob), 4),
            }
            for class_id, prob in enumerate(result.probs.data.tolist())
        ),
        key=lambda item: item["confidence"],
        reverse=True,
    )
    top1 = ranked[0]

    # 在原图上标注预测类别。沿用原实现的写法（x 偏移取 img.shape[0]-80，即图像高度），
    # 并改用 copy() 避免修改调用方传入的数组。
    annotated = img.copy()
    cv.putText(annotated, top1["class_name"].upper(), (annotated.shape[0] - 80, 60),
               cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv.LINE_AA)

    return {
        "task": "classify",
        "filename": filename,
        "top1": top1,
        "probs": ranked,
        "class_names": dict(names),
        "plot_rgb": cv.cvtColor(annotated, cv.COLOR_BGR2RGB),
    }
