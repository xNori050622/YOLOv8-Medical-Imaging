
# YOLOv8 Medical Imaging

YOLO is known for its ability to detect objects in an image in a single pass, making it a highly efficient and accurate object detection algorithm.🎯

The latest version of YOLO, YOLOv8, released in January 2023 by Ultralytics, has introduced several modifications that have further improved its performance.

In this project, I will focus on three major computer vision tasks that YOLOv8 can be used for: **classification**, **detection**, and **segmentation**. I will explore how YOLOv8 can be applied in the 
field of medical imaging to detect and classify various anomalies and diseases🧪💊.


## Introduction to YOLOv8
Some of the notable modifications in YOLOv8 include:

- **New Backbone Network**: YOLOv8 adopts the powerful Darknet-53 as its backbone network, enhancing feature extraction capabilities.

- **Anchor-Free Detection**: YOLOv8 employs an anchor-free detection head, which directly predicts the center of an object instead of relying on offset values from predefined anchor boxes.

- **New Loss Function**

## Tasks

In this project, I focus on three major computer vision tasks using YOLOv8, all accessible through the Streamlit web application:

1. **Classification:** Utilize the YOLOv8 model to classify medical images into three categories: COVID-19, Viral Pneumonia, and Normal, using the [COVID-19 Image 
Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset).

2. **Object Detection:** Employ YOLOv8 for detecting Red Blood Cells (RBC), White Blood Cells (WBC), and Platelets in blood cell images using the [RBC and WBC Blood Cells Detection 
Dataset](https://universe.roboflow.com/tfg-2nmge/yolo-yejbs).

3. **Segmentation:** Use YOLOv8 for segmenting breast ultrasound images with the [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset).

## Screenshots

I used Streamlit to create a user-friendly interface for easy interaction with the YOLOv8 model. Below are screenshots of each part:

### About page

![About](https://github.com/sevdaimany/YOLOv8-Medical-Imaging/blob/master/intro_screenshot.png)


### Object Detection

![Object Detection Screenshot](https://github.com/sevdaimany/YOLOv8-Medical-Imaging/blob/master/detection/detection_screenshot.png)

### Classification

![Classification Screenshot](https://github.com/sevdaimany/YOLOv8-Medical-Imaging/blob/master/classification/classification_screenshot.png)


### Segmentation

![Segmentation Screenshot](https://github.com/sevdaimany/YOLOv8-Medical-Imaging/blob/master/segmentation/segmentation_screenshot.png)

## Installation and Usage

### Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/sevdaimany/YOLOv8-Medical-Imaging.git
   ```
2. Navigate to the project directory:

   ```bash
   cd YOLOv8-Medical-Imaging
   ```
3. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   ```
4. Activate the virtual environment:

On Windows:

   ```bash
venv\Scripts\activate

   ```

On macOS and Linux:

   ```bash
source venv/bin/activate

   ```
5. Install the required dependencies from the provided requirements.txt file:


   ```bash
   pip install -r requirements.txt
   ```


## Usage
**Using the Provided Demo Images**

I've made it easy for you to get started with our project without the need to download a dataset. I've included a set of demo images in the DEMO_IMAGES directory. You can use these images to quickly see 
how our project works.

**Run the Streamlit App:**

Start the Streamlit app to see our project in action:
```bash
streamlit run app.py
```

`requirements.txt` pins every version (validated on Python 3.11) and is stored as
UTF-8 with a BOM, so `pip install -r requirements.txt` also works on Windows
consoles that default to a legacy code page.

### Windows launchers

Two convenience launchers are included at the repository root. They `cd` into the
script directory first, so they work regardless of the directory you start them
from:

```bat
run_web.bat
```

```powershell
.\run_web.ps1
```

Both prefer `D:\infynova\venv_main311\Scripts\python.exe` when it exists and fall
back to `python` on your `PATH` otherwise.

## Batch inference and result export

Every task page now accepts **multiple images at once**:

- Upload a single image (or upload nothing to run the bundled demo image) to see
  the annotated result, key metrics and a per-target table.
- Upload two or more images to run **batch inference**: a progress bar is shown
  while the images are processed, each image gets its own collapsible section,
  and a combined summary table is appended at the bottom.

Results can be downloaded directly from the UI:

| Scope | Files |
| --- | --- |
| Per image | annotated PNG, CSV (one row per target), JSON |
| Per image, segmentation only | the binary mask PNG as well |
| Batch, in addition | combined CSV, combined JSON, and a ZIP bundle |

The ZIP bundle contains `README.txt`, `results.csv`, `results.json`, `images/`
and `masks/` (segmentation only). CSV exports are UTF-8 **with BOM** so Excel
opens them without mojibake.

Internally, `predict()` in `detection/detect.py`, `classification/classify.py`
and `segmentation/segment.py` returns a plain results dictionary instead of
drawing into Streamlit, keeping inference and presentation separate.
`config.py` centralises all paths (resolved from `__file__`, so the app no longer
has to be launched from the repository root) and caches each YOLO model so the
weights are loaded only once per process. `export.py` turns those dictionaries
into CSV / JSON / PNG / ZIP.


## Training

`runs/*/train/weights/best.pt` were trained by the original author on Google
Colab. `train.py` reproduces that setup locally, and `train.py check` reports
what is currently present before you start.

### 1. Datasets

None of the three datasets is bundled — they are large and licensed separately.
Download them and unpack them **without renaming anything**, because `config.py`
expects the original layout:

| Task | Dataset | Expected location |
| --- | --- | --- |
| Detection | [RBC / WBC Blood Cells](https://universe.roboflow.com/tfg-2nmge/yolo-yejbs), Roboflow export in *YOLOv8* format | `detection/data/{train,valid,test}/{images,labels}` + `data.yaml` |
| Classification | [COVID-19 Image Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset) | `classification/Covid19-dataset/{train,test}/{Covid,Normal,Viral Pneumonia}` |
| Segmentation | [Breast Ultrasound Images (BUSI)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) | `segmentation/Dataset_BUSI_with_GT/{benign,malignant,normal}` |

Kaggle needs an API token (`kaggle.json`), Roboflow needs an account. The class
order is baked into the shipped checkpoints, so it must not change:

```text
detection       Platelets, RBC, WBC
classification  Covid, Normal, Viral Pneumonia     # sorted folder names
segmentation    benign, malignant, normal
```

Two helpful details:

- If `detection/data/` only has `images/` and `labels/`, `data.yaml` is generated
  automatically on the first run.
- Segmentation needs one extra step (`--prepare`), which turns the BUSI masks into
  polygon labels, splits them 8:1:1 and finally writes `segmentation/data.yaml` —
  a file the original repository never created, which is why `train()` could
  never run. BUSI has no masks for `normal`, so that class contributes background
  images only; `nc=3` is kept so the class list still matches the checkpoint.

### 2. Base weights

Training starts from the COCO / ImageNet checkpoints in `weights/`
(`yolov8n.pt`, `yolov8n-cls.pt`, `yolov8n-seg.pt`, ~19 MB in total). They are not
committed, so a fresh clone has none of them:

```bash
python tools/get_weights.py          # downloads whatever is missing
python tools/get_weights.py --force  # re-downloads all three
```

The script resumes interrupted transfers and validates each file, because the
official asset host is reachable but slow — and sometimes stalls — from some
networks. Ultralytics would also download these on demand, but without resume
support. Alternatively, `--no-pretrained` trains from scratch using the model
`*.yaml` bundled with ultralytics and needs no network at all.

### 3. Run

```bash
python train.py detect                      # 100 epochs, imgsz 640
python train.py classify                    # 100 epochs, imgsz 224
python train.py segment --prepare           # build the BUSI dataset first
python train.py segment
python train.py all                         # all three in order
python train.py detect --epochs 1 --device cpu   # quick smoke run
```

Output lands in `runs/<task>/train/` — exactly where the Streamlit app reads it.
The existing `best.pt` is copied to `best.pt.bak` first, so a retrain never
silently destroys the shipped weights. Use `--name smoke` to write somewhere else
instead. Defaults mirror `runs/*/train/args.yaml`: 100 epochs, batch 16, seed 0,
`deterministic=True`, imgsz 640 for detect/segment and 224 for classify.

The implementation is split so the same code serves the CLI and any script:
`training.py` holds the shared train loop, `dataset.py` the dataset layout,
validation and `data.yaml` generation, and each task module's `train()` is a thin
wrapper over `training.train_task()`.

### 4. Verify the pipeline without downloading a dataset

`tools/make_smoke_dataset.py` builds a tiny synthetic dataset (a few dozen
64–96 px images) in a temporary directory, so the whole chain — data preparation,
training, checkpoint output — can be checked in about a minute:

```powershell
python tools\make_smoke_dataset.py --out D:\temp\p3_smoke
```

It prints the three commands to run; each uses `--name smoke`, so the shipped
weights stay untouched.

### GPU training

`requirements.txt` installs CPU-only torch. For an NVIDIA GPU:

```powershell
D:\infynova\venv_gpu\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

`--device auto` (the default) picks CUDA when it is available and falls back to
CPU otherwise. On a GPU, ultralytics runs a one-off AMP self-check that tries to
download `yolo26n.pt`; `--no-amp` skips it.





