
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




