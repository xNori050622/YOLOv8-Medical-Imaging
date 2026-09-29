"""YOLOv8 医学影像 Web 演示。

相对原始版本新增：
  1. 统一由 config.py 提供绝对路径，可脱离项目根目录启动；
  2. 模型常驻缓存，避免每次交互都重新加载权重；
  3. 支持批量上传（多图）推理，带进度条；
  4. 结果导出：单图 CSV / JSON / PNG（分割额外含掩码 PNG），
     批量额外提供汇总 CSV / JSON 与打包 ZIP。
"""
from __future__ import annotations

import os
from datetime import datetime

import cv2 as cv
import numpy as np
import streamlit as st

import classification.classify as classify
import config
import detection.detect as detect
import export
import segmentation.segment as segment

# 三个任务的运行配置
TASKS = {
    "Object Detection": {
        "key": "detect",
        "header": "Object Detection with YOLOv8",
        "demo": "BloodImage_00000_jpg.rf.5fb00ac1228969a39cee7cd6678ee704.jpg",
        "weights": config.DETECT_WEIGHTS,
        "with_confidence": True,
        "run": lambda img, conf, name: detect.predict(img, conf, filename=name),
    },
    "Object Classification": {
        "key": "classify",
        "header": "Classification with YOLOv8",
        "demo": "094.png",
        "weights": config.CLASSIFY_WEIGHTS,
        "with_confidence": False,
        "run": lambda img, conf, name: classify.predict(img, filename=name),
    },
    "Object Segmentation": {
        "key": "segment",
        "header": "Segmentation with YOLOv8",
        "demo": "benign (2).png",
        "weights": config.SEGMENT_WEIGHTS,
        "with_confidence": True,
        "run": lambda img, conf, name: segment.predict(img, conf, filename=name),
    },
}


def train_models():
    detect.train()
    print("[INFO] Training Detection model done!")
    classify.train()
    print("[INFO] Training Classification model done!")

    # RUN THE FOLLOWING FOR PREPERING INPUT DATA FOR TRAINIG SEGMENTATION MODEL
    segment.prepare_input()
    segment.train()
    print("[INFO] Training Segmentation model done!")


# ------------------------------------------------------------ 图像读取
def read_image(source):
    """读取图片，source 可以是上传得到的 bytes，也可以是本地文件路径。

    返回 (BGR 数组, RGB 数组)；读取失败返回 (None, None)。
    """
    if isinstance(source, (bytes, bytearray)):
        img_bgr = cv.imdecode(np.frombuffer(bytes(source), np.uint8), cv.IMREAD_COLOR)
    else:
        img_bgr = cv.imread(str(source), cv.IMREAD_COLOR)
    if img_bgr is None:
        return None, None
    return img_bgr, cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)


# ------------------------------------------------------------ 结果展示
def _display_rows(res):
    """界面上的明细表行（分割任务去掉很长的 polygon 列，导出时仍保留）。"""
    rows = export.results_to_rows(res)
    if res["task"] == "segment":
        for row in rows:
            row.pop("polygon", None)
    return rows


def _summary_line(res):
    """一行文字概述结果，用于批量模式下的折叠标题。"""
    task = res["task"]
    if task == "detect":
        return f"detect · {res['num_objects']} object(s)"
    if task == "classify":
        top1 = res["top1"]
        return f"classify · {top1['class_name']} ({top1['confidence']:.2%})"
    return f"segment · {res['num_instances']} instance(s)"


def render_result(res):
    """渲染单张图的结果：输出图像 + 关键指标 + 明细表。"""
    task = res["task"]
    st.subheader("Output Image")

    if task == "detect":
        st.image(res["plot_rgb"], width="stretch")
        cols = st.columns(2)
        cols[0].metric("Objects detected", res["num_objects"])
        cols[1].metric("Distinct classes", len(res["counts"]))
        if res["counts"]:
            st.bar_chart(
                [{"class": name, "count": count} for name, count in res["counts"].items()],
                x="class",
                y="count",
            )
        if res["detections"]:
            st.dataframe(_display_rows(res), width="stretch", hide_index=True)

    elif task == "classify":
        st.image(res["plot_rgb"], width="stretch")
        top1 = res["top1"]
        cols = st.columns(2)
        cols[0].metric("Predicted class", top1["class_name"])
        cols[1].metric("Confidence", f"{top1['confidence']:.2%}")
        st.bar_chart(
            [{"class": prob["class_name"], "probability": prob["confidence"]} for prob in res["probs"]],
            x="class",
            y="probability",
        )
        st.dataframe(_display_rows(res), width="stretch", hide_index=True)

    elif task == "segment":
        cols = st.columns(2)
        if res["mask_gray"] is not None:
            cols[0].image(res["mask_gray"], caption="Mask", clamp=True,
                          channels="GRAY", width="stretch")
        cols[1].image(res["plot_rgb"], caption="Overlay", width="stretch")
        cols = st.columns(2)
        cols[0].metric("Masks detected", res["num_instances"])
        cols[1].metric("Distinct classes", len(res["counts"]))
        if res["instances"]:
            st.dataframe(_display_rows(res), width="stretch", hide_index=True)


# ------------------------------------------------------------ 结果导出
def render_downloads(res, key_prefix):
    """单张图的结果下载按钮（按任务自动决定包含哪些文件）。"""
    stem = export.safe_name(res.get("filename"))
    items = []

    png = export.annotated_png_bytes(res)
    if png is not None:
        items.append(("Download annotated PNG", png, f"{stem}_annotated.png", "image/png", "png"))

    mask = export.mask_png_bytes(res)
    if mask is not None:
        items.append(("Download mask PNG", mask, f"{stem}_mask.png", "image/png", "mask"))

    rows = export.results_to_rows(res)
    items.append(("Download CSV", export.rows_to_csv_bytes(rows), f"{stem}.csv", "text/csv", "csv"))
    items.append(("Download JSON", export.results_to_json_bytes([res]), f"{stem}.json", "application/json", "json"))

    cols = st.columns(len(items))
    for col, (label, data, file_name, mime, tag) in zip(cols, items):
        col.download_button(
            label,
            data,
            file_name=file_name,
            mime=mime,
            key=f"{key_prefix}_{tag}",
            width="stretch",
        )


def render_batch_summary(results, mode_key):
    """批量结果汇总表 + 打包下载。"""
    st.markdown("---")
    st.subheader(f"Batch summary · {len(results)} image(s)")

    rows = export.results_list_to_rows(results)
    st.dataframe(rows, width="stretch", hide_index=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cols = st.columns(3)
    cols[0].download_button(
        "Download all results (CSV)",
        export.rows_to_csv_bytes(rows),
        file_name=f"batch_{mode_key}_{stamp}.csv",
        mime="text/csv",
        key=f"batch_csv_{mode_key}",
        width="stretch",
    )
    cols[1].download_button(
        "Download all results (JSON)",
        export.results_to_json_bytes(results),
        file_name=f"batch_{mode_key}_{stamp}.json",
        mime="application/json",
        key=f"batch_json_{mode_key}",
        width="stretch",
    )
    cols[2].download_button(
        "Download bundle (ZIP)",
        export.build_zip(results, rows),
        file_name=f"batch_{mode_key}_{stamp}.zip",
        mime="application/zip",
        key=f"batch_zip_{mode_key}",
        width="stretch",
    )


# ---------------------------------------------------------------- 主流程
def main():

    st.sidebar.title("Settings")
    st.sidebar.subheader("Parameters")
    st.markdown(
    """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width:300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width:300px;
            margin-left:-300px;
        }
        </style>
    """,
    unsafe_allow_html=True,
    )

    app_mode = st.sidebar.selectbox(
        'Choose the App Mode',
        ['About App', 'Object Detection', 'Object Classification', "Object Segmentation"],
    )

    if app_mode == 'About App':
        render_about()
        return

    task = TASKS[app_mode]
    st.header(task["header"])
    st.sidebar.markdown("----")

    # 启动前自检：权重不存在时给出明确提示，而不是等 ultralytics 抛异常
    if not task["weights"].is_file():
        st.error(
            f"未找到模型权重文件：{task['weights']}\n\n"
            "请确认仓库克隆完整（runs/*/train/weights/best.pt），或先运行 train_models() 训练模型。"
        )
        return

    confidence = config.DEFAULT_CONFIDENCE
    if task["with_confidence"]:
        confidence = st.sidebar.slider("Confidence", min_value=0.0, max_value=1.0,
                                       value=config.DEFAULT_CONFIDENCE)

    uploaded = st.sidebar.file_uploader(
        "Upload image(s)", type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True, key=task["key"],
    )

    if uploaded:
        sources = [(f.name, f.getvalue()) for f in uploaded]
    else:
        demo_path = config.demo_image(task["demo"])
        sources = [(os.path.basename(demo_path), demo_path)]

    # ---- 推理 ----
    results = []
    progress = st.progress(0.0, text="Running inference ...") if len(sources) > 1 else None
    previewed = False

    for index, (name, source) in enumerate(sources, start=1):
        img_bgr, img_rgb = read_image(source)
        if img_bgr is None:
            st.warning(f"无法读取图片（已跳过）：{name}")
            continue

        if not previewed:
            st.sidebar.text("Original Image" if len(sources) == 1 else "Original Image (first)")
            st.sidebar.image(img_rgb)
            previewed = True

        if progress is not None:
            progress.progress((index - 1) / len(sources),
                              text=f"Running inference ... {index}/{len(sources)} · {name}")

        results.append(task["run"](img_bgr, confidence, name))

    if progress is not None:
        progress.progress(1.0, text="Inference finished")
        progress.empty()

    if not results:
        st.warning("没有可用的图片。请在侧边栏上传图片，或确认 DEMO_IMAGES 下的演示图片存在。")
        return

    # ---- 展示与导出 ----
    if len(results) == 1:
        res = results[0]
        render_result(res)
        st.markdown("#### Download results")
        render_downloads(res, key_prefix=f"{task['key']}_0")
    else:
        st.markdown(f"Processed **{len(results)}** image(s).")
        for index, res in enumerate(results):
            with st.expander(f"{index + 1}. {res['filename']} — {_summary_line(res)}",
                             expanded=(index == 0)):
                render_result(res)
                render_downloads(res, key_prefix=f"{task['key']}_{index}")
        render_batch_summary(results, task["key"])


# ------------------------------------------------------------------ 关于页
def render_about():

    st.header("Introduction to YOLOv8")

    st.markdown("<style> p{margin: 10px auto; text-align: justify; font-size:20px;}</style>", unsafe_allow_html=True)
    st.markdown("<p>🚀Welcome to the introduction page of our project! In this project, we will be exploring the YOLO (You Only Look Once) algorithm. YOLO is known for its ability to detect objects in an image in a single pass, making it a highly efficient and accurate object detection algorithm.🎯</p>", unsafe_allow_html=True)
    st.markdown("<p>The latest version of YOLO, YOLOv8, released in January 2023 by Ultralytics, has introduced several modifications that have further improved its performance. 🌟</p>", unsafe_allow_html=True)
    st.markdown("""<p>🔍Some of these modifications are:<br>
                &#x2022; Introducing a new backbone network, Darknet-53,<br>
                &#x2022; Introducing a new anchor-free detection head. This means it predicts directly the center of an object instead of the offset from a known anchor box.<br>
                &#x2022; and a new loss function.<br></p>""", unsafe_allow_html=True)

    st.markdown("""<p>🎊One of the key advantages of YOLOv8 is its versatility. It not only supports object detection but also offers out-of-the-box support for classification and segmentation tasks. This makes it a powerful tool for various computer vision applications.<br><br>
                ✨In this project, we will focus on three major computer vision tasks that YOLOv8 can be used for: <b>classification</b>, <b>detection</b>, and <b>segmentation</b>. We will explore how YOLOv8 can be applied in the field of medical imaging to detect and classify various anomalies and diseases🧪💊.</p>""", unsafe_allow_html=True)

    st.markdown("""<p>We hope you find this project informative and inspiring.💡 Let's dive into the world of YOLOv8 and discover how easy it is to use it!🥁🎆</p>""", unsafe_allow_html=True)


if __name__ == "__main__":
    try:

        # RUN THE FOLLOWING ONLY IF YOU WANT TO TRAIN MODEL AGAIN
        # train_models()

        main()
    except SystemExit:
        pass




