
# YOLOv8 Medical Imaging

YOLO is known for its ability to detect objects in an image in a single pass, making it a highly efficient and accurate object detection algorithm.🎯

The latest version of YOLO, YOLOv8, released in January 2023 by Ultralytics, has introduced several modifications that have further improved its performance.

In this project, I will focus on three major computer vision tasks that YOLOv8 can be used for: **classification**, **detection**, and **segmentation**. I will explore how YOLOv8 can be applied in the 
field of medical imaging to detect and classify various anomalies and diseases🧪💊.


> **Extended fork.** This repository started from
> [sevdaimany/YOLOv8-Medical-Imaging](https://github.com/sevdaimany/YOLOv8-Medical-Imaging),
> which ships a Streamlit demo for three tasks but can neither train a model nor
> measure one. The *Training*, *Evaluation*, *Data augmentation* and *Regression
> tests* sections below — and the summary in
> [What this fork adds](#what-this-fork-adds) — describe the work added on top of it.

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

![About](intro_screenshot.png)


### Object Detection

![Object Detection Screenshot](detection/detection_screenshot.png)

### Classification

![Classification Screenshot](classification/classification_screenshot.png)


### Segmentation

![Segmentation Screenshot](segmentation/segmentation_screenshot.png)

## Installation and Usage

### Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/xNori050622/YOLOv8-Medical-Imaging.git
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

Launchers are included at the repository root. They `cd` into the script directory
first, so they work regardless of the directory you start them from:

```bat
run_web.bat        :: CPU environment
run_web_gpu.bat    :: GPU environment
```

```powershell
.\run_web.ps1
.\run_web_gpu.ps1
```

Each launcher prefers its own virtual environment and falls back to `python` on your
`PATH` otherwise: `run_web.*` uses `D:\infynova\venv_main311\Scripts\python.exe`
(CPU build of torch), while `run_web_gpu.*` uses
`D:\infynova\venv_gpu\Scripts\python.exe` (`torch==2.13.0+cu126` plus `streamlit`)
so inference runs on the GPU. Measured on an RTX 4060 Laptop with the detection
`best.pt` at `imgsz=640`: 0.070 s/image on CPU vs 0.014 s/image on GPU (first call
includes ~2.8 s of model load). The CPU path was already fast enough for a demo —
the GPU launcher is a convenience, not a requirement.

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

### Export to ONNX

The exported graph is not committed: it is a pure function of `best.pt`, and one
command re-creates it (`.gitignore` ignores `runs/*/train/weights/*.onnx`).

```powershell
D:\infynova\venv_gpu\Scripts\python.exe -c "from ultralytics import YOLO; YOLO(r'runs\detect\train\weights\best.pt').export(format='onnx', opset=18, simplify=True, imgsz=640)"
```

It writes `best.onnx` next to the weights. For the detection checkpoint in this
repository that is 11.7 MB, opset 18, slimmed by `onnxslim`, taking input
`images (1, 3, 640, 640)` and producing output `output0 (1, 7, 8400)` — 7 being
4 box coordinates + 3 classes, 8400 being the anchor points. It needs no GPU and
finishes in about 11 s on CPU.

Check the graph before trusting it — this prints
`[('images', [1, 3, 640, 640])] [('output0', [1, 7, 8400])]`:

```powershell
D:\infynova\venv_gpu\Scripts\python.exe -c "import onnxruntime as ort; s = ort.InferenceSession(r'runs\detect\train\weights\best.onnx', providers=['CPUExecutionProvider']); print([(i.name, i.shape) for i in s.get_inputs()], [(o.name, o.shape) for o in s.get_outputs()])"
```

`onnxruntime` is deliberately not in `requirements.txt`: the project itself runs on
`.pt`, so only an environment that inspects or deploys the graph needs it, and the
CPU build is enough (`pip install onnxruntime`).

> **Note.** Running ONNX *on the GPU* is an install of its own, separate from
> torch's CUDA build. Creating a session with `CUDAExecutionProvider` on the
> CUDA 12.6 / cuDNN 8 machine this project was developed on fails with
> `Require cuDNN 9.* and CUDA 13.*, and the latest MSVC runtime`, so
> `get_providers()` returns `['CPUExecutionProvider']` and ONNX falls back to CPU.
> The demo therefore keeps using `.pt` (and does use the GPU there); the ONNX
> export is aimed at other runtimes.


## Evaluation

`evaluate.py` scores a checkpoint on a split that still has ground-truth labels.
The original project only ever drew annotated pictures, so there was no number to
report for any of the three tasks.

```bash
python evaluate.py segment  --split val            # Dice / IoU / HD95, instance P-R
python evaluate.py detect   --split valid          # per-class P / R / F1 / AP, mAP@0.5
python evaluate.py classify --split test           # confusion matrix, accuracy, macro-F1

python evaluate.py segment --name baseline --save runs\ablation\baseline_val.json
python evaluate.py segment --data D:\temp\smoke\segmentation\data.yaml --limit 5
python evaluate.py segment --cross-check           # also run ultralytics val(), then compare
```

`--name` resolves `runs/<task>/<name>/weights/best.pt`, and `--weights` overrides it
with an explicit path. `--split` defaults to `val` (`valid` is accepted for
detection), `--conf` defaults to 0.25, `--limit` scores only the first N images, and
`--save` writes the result as JSON.

All metrics live in `metrics.py` and use numpy + OpenCV only — no torch, no
ultralytics — so they run on CPU and the unit tests need neither a model nor a
dataset. The conventions below are part of the experiment setup, which is why they
are written down instead of left implicit:

| Situation | Convention |
| --- | --- |
| Mask polarity | any non-zero pixel counts as foreground |
| Dice / IoU, prediction and ground truth both empty | `1.0` |
| HD95, either side empty | infinity |
| Precision / recall with a zero denominator | `0.0` (scikit-learn convention) |
| HD95 distance unit | pixels |

Segment Dice is the customary medical-imaging figure, but BUSI's `normal` class has
empty ground truth, which scores a free `1.0` and inflates the mean. So in addition
to the standard segmentation and detection numbers, `evaluate.py` also reports the
mean with empty ground truths excluded, and `--cross-check` runs ultralytics' own
`val()` as a cross-check against the mAP computed here.

### Where the numbers live

Every figure quoted in this repository is checked in as JSON under `runs/eval_*.json`
(`.gitignore` re-admits exactly that pattern), so each claim can be traced to the run
that produced it:

| File | Task | Split | Result |
| --- | --- | --- | --- |
| `runs/eval_detect_baseline.json` | detect | valid, 175 images | mAP@0.5 **0.8974**, macro P/R/F1 0.7719 / 0.9511 / 0.8465 |
| `runs/eval_detect_retrained.json` | detect | valid, 175 images | mAP@0.5 **0.9290**, macro P/R/F1 0.8199 / 0.9679 / 0.8848 |
| `runs/eval_detect_retrained_cross.json` | detect | valid, 175 images | the same run, plus ultralytics `val()` box mAP@0.5 = 0.9582 |
| `runs/eval_classify_val.json` | classify | val, 50 images | accuracy **0.9200**, macro-F1 0.9199 |
| `runs/eval_classify_test.json` | classify | test, 66 images | accuracy **0.9848**, macro-F1 0.9833 |
| `runs/eval_segment_val.json` | segment | val, 78 images | Dice **0.5290** (0.6272 with empty ground truths dropped), IoU 0.4645, instance F1 0.6983 |

The two detection rows are directly comparable — same script, same split, same
`--conf 0.25` — so re-training is worth **+0.0316 mAP@0.5** (0.8974 → 0.9290). Do not
subtract ultralytics' number from `evaluate.py`'s: for the *same* re-trained checkpoint
the former reads 0.9582 and the latter 0.9290, because `val()` sweeps `--conf` down to
0.001 and integrates the PR curve differently.

Classification and segmentation were **not** re-trained here. Those two rows score the
upstream Colab checkpoints as they stand, and both are still contaminated by defect 2
in [What this fork adds](#what-this-fork-adds): upstream validated on `test/` while
training (which is why 0.9848 there beats 0.9200 on `val/` — the test set is the one
that was selected on), and `val/` was carved out of `train/` afterwards (so those 50
images were trained on). Neither figure is a clean generalisation estimate; getting
one needs the re-train, not another `evaluate.py` run.


## Data augmentation

`augment.py` implements the two augmentations that are actually standard for
ultrasound and similar medical images, and `tools/make_augmented_dataset.py` turns
them into a dataset you can train on:

```powershell
python tools\make_augmented_dataset.py --out segmentation\data_aug\clahe --clahe
python tools\make_augmented_dataset.py --out segmentation\data_aug\ela   --elastic
python tools\make_augmented_dataset.py --out segmentation\data_aug\both  --clahe --elastic --copies 2
```

Why not the augmentations ultralytics already provides? `mosaic` pastes four images
into one, inventing anatomy that does not exist in an ultrasound scan, and HSV
jitter changes hue while an ultrasound image is greyscale — hue carries no
information at all. CLAHE (contrast-limited adaptive histogram equalisation) and
elastic deformation (Simard et al., 2003) are what medical imaging papers use
instead.

The subtle part is that the two transforms relate to the labels in different ways:

- **CLAHE is photometric.** It does not move a single pixel, so the polygon labels
  are passed through unchanged.
- **Elastic deformation is geometric.** The polygon vertices have to be moved
  together with the image. Warping only the image produces a dataset in which every
  lesion sits at a *systematically offset* position — and neither training nor
  evaluation reports anything wrong. Vertices are mapped forward, because the
  displacement field `cv2.remap` consumes is defined backwards.

The tool augments the splits you ask for (`--splits`, default `train`), copies every
other split through **byte for byte**, and writes `data.yaml` plus
`augment_manifest.json` holding the hyper-parameters, seed, source directory and
per-split counts. It then re-checks, independently of `augment.py`, that the
untouched splits are still byte-identical to the source, and refuses to finish
otherwise: a validation split that got augmented even once makes the baseline and
the augmented arm incomparable. CLAHE defaults to `clip_limit=2.0`,
`tile_grid_size=8` (`--clip-limit`, `--tile-grid`); elastic deformation defaults come
from `augment.ELASTIC_DEFAULTS` (`--alpha`, `--sigma`). `--seed` (default 0) makes a
run reproducible, `--dry-run` prints the plan without writing anything, and
`--force` is required to overwrite an existing output directory.

The augmented dataset is self-describing, so training on it is a one-liner:

```powershell
python train.py segment --data segmentation\data_aug\clahe\data.yaml --name clahe
```


## A real validation split for classification

`classification/Covid19-dataset` ships only `train/` and `test/`. When ultralytics
cannot find `val/` it falls back to `test/` and prints nothing worse than
`WARNING Dataset 'split=val' not found, using 'split=test' instead.` — so `best.pt`
is selected on the test set, and the test accuracy reported at the end is
optimistically biased. That is a fatal flaw for an ablation study, which is exactly
the kind of comparison this fork needs to be able to make.

`tools/make_classify_val.py` stratifies by class and **moves** (not copies) 20% of
`train/` into a new `val/`, so train / val / test are disjoint: `val/` drives model
selection during training and `test/` stays clean for the final number.

```powershell
python tools\make_classify_val.py            # --ratio 0.2 --seed 0 by default
python tools\make_classify_val.py --dry-run  # list the files that would move
python tools\make_classify_val.py --undo     # move them back into train/
```

Every move is recorded in `<data>/val_split_manifest.json`, which is what makes
`--undo` exact instead of "a second random split in the other direction". The script
also deletes the `train.cache` / `val.cache` / `test.cache` files ultralytics leaves
behind, so the image list is rebuilt rather than read from a stale listing.


## Regression tests

`tests/` holds 104 plain-`assert` checks, split by module:

| File | Cases | Covers |
| --- | --- | --- |
| `tests/test_metrics.py` | 29 | Dice / IoU / HD95 / confusion matrix, including empty masks, mismatched shapes and the degenerate conventions above |
| `tests/test_augment.py` | 29 | CLAHE really is photometric, elastic deformation keeps label topology, identical seeds reproduce identical output |
| `tests/test_make_augmented_dataset.py` | 17 | split isolation, byte-identical untouched splits, manifest contents, `--force` / `--dry-run` |
| `tests/test_make_classify_val.py` | 13 | stratification ratio, idempotence, exact `--undo` |
| `tests/test_split_alignment.py` | 9 | an image and its own label always land in the same split |
| `tests/test_training.py` | 7 | the branches of `resolve_data()`, including the missing-file case |

They need no GPU, no dataset and no pytest — every file has its own `main()`, so
either runner works:

```bash
python tests/test_augment.py    # plain python
pytest tests/                   # if you happen to have pytest installed
```

`.github/workflows/tests.yml` runs exactly those six files on Windows with Python
3.11 for every push, installing only the three packages the suite actually touches
(`numpy`, `opencv-python`, `PyYAML`, pinned to the versions in `requirements.txt`).
That job is what keeps the sentence above honest: the suite has to pass on a machine
with neither the datasets nor torch, and running it on a clean checkout is the only
way to notice when one of the tests quietly stops meeting that bar.

The same job also runs two guards that only mean something on a clean machine.
`python tools/check_repo_consistency.py --strict` re-derives this README's numbers
from git and from the test files — the fork statistics below, the per-file case counts
above, and the dependency pins in the workflow itself — so a number that drifts fails
the build instead of quietly misleading a reader. And `python train.py --help` plus
`python evaluate.py --help` prove the two CLI entry points still import with neither
torch nor a dataset present, which is the one failure mode the six test files cannot
see (none of them imports an entry point). Both guards exist because both things have
already been wrong here once while looking green locally.


## What this fork adds

Upstream is five Python files: `app.py`, `classification/classify.py`,
`detection/detect.py`, `segmentation/masks_to_polygons.py` and
`segmentation/segment.py`. Measured against `5d13edf` — the tip of upstream's `master`,
the commit this fork grew from — and ignoring `runs/`, this fork adds 28 files and
modifies 7 (35 files, +8066 / −333 lines). CI re-derives every number in that sentence
with [`tools/check_repo_consistency.py`](tools/check_repo_consistency.py), so it cannot
rot quietly.

| Area | Files |
| --- | --- |
| Reproducible local training | `train.py`, `training.py`, `dataset.py`, `config.py`, `tools/get_weights.py`, `tools/make_smoke_dataset.py` |
| Quantitative evaluation | `metrics.py`, `evaluate.py` |
| Data augmentation | `augment.py`, `tools/make_augmented_dataset.py` |
| Data preparation | `tools/make_classify_val.py`, and the rewritten split in `segmentation/masks_to_polygons.py` |
| Interactive app | `export.py` plus refactored `predict()` in the three task modules and centralised paths in `config.py` |
| Windows launchers | `run_web.bat`, `run_web.ps1`, `run_web_gpu.bat`, `run_web_gpu.ps1` |
| Training helpers | `tools/train_detect_gpu.bat` (menu: check / eval / smoke / train / resume / web / status), `tools/train_status.ps1` + `tools/train_status.bat` |
| Regression tests | `tests/` — 104 cases |
| Doc consistency | `tools/check_repo_consistency.py` — asserts the numbers in this README (fork statistics, per-file case counts, dependency pins) against git and the workflow |

Three defects in the original pipeline were fixed along the way, all three of which
are silent — none of them raises an error or produces a wrong-looking output, they
just make the numbers mean something other than what they appear to mean.

1. **An image could be split away from its own label.** The original
   `split_train_test_val()` called `splitfolders.ratio`, which shuffles `images/` and
   `labels/` *independently*; alignment survived only as long as both sides held the
   same number of files in the same sorted order. That is not guaranteed — BUSI
   contributes images but no masks for its `normal` class — and once it breaks,
   `val/images/benign (12).png` is paired with `val/labels/benign (3).txt`. Nothing
   complains: training learns from mismatched masks, and evaluation compares
   predictions against someone else's ground truth. Splitting is now driven by case
   name, so a label always travels with its image, and `split-folders` is no longer a
   dependency.
2. **Classification validated on the test set.** See
   [A real validation split for classification](#a-real-validation-split-for-classification).
3. **`--data` fell back silently.** If the `data.yaml` passed to `train.py` did not
   exist, `resolve_data()` quietly fell back to the default dataset instead of
   failing. An augmentation experiment would then train on the unaugmented data while
   the run's own records claimed otherwise: every row of the ablation table wrong,
   and nothing raising an error. It now raises `FileNotFoundError`.


## Credits and licensing

Different parts of this repository carry different rights, and they do not point the
same way.

**The upstream work.** This repository is a fork of
[sevdaimany/YOLOv8-Medical-Imaging](https://github.com/sevdaimany/YOLOv8-Medical-Imaging)
and keeps its history. Upstream's five Python files all survive here, and this fork
modified all five: `app.py`, `classification/classify.py`, `detection/detect.py`,
`segmentation/masks_to_polygons.py`, `segmentation/segment.py` — plus `README.md` and
`requirements.txt`. The four screenshots further up are upstream's images: this fork
carries the same files unchanged, so they render from this repository rather than being
hot-linked. The three `runs/*/train/weights/best.pt` checkpoints were trained by
that author on Google Colab. Upstream carries **no `LICENSE` file**, and no licence
means no permission granted: by default all rights stay with its author. Those files
are therefore *not* covered by anything this fork grants, and anyone wanting to reuse
them — commercially in particular — has to ask the upstream author. The fork link in
the header is the attribution; nothing here claims authorship of them.

**What this fork adds.** Everything listed under
[What this fork adds](#what-this-fork-adds) — `train.py`, `training.py`, `dataset.py`,
`config.py`, `metrics.py`, `evaluate.py`, `augment.py`, `export.py`, `tools/`,
`tests/`, the Windows launchers and the `runs/eval_*.json` records — was written here.
No licence is granted on that either: treat it as source-available for study and
review, and ask before reusing it. A public fork is not a licence; it is only a public
fork.

**Ultralytics YOLOv8 is AGPL-3.0** — the one part with an explicit licence, and the one
part that constrains what you may do with the *output*. `pip show ultralytics` reports
`License: AGPL-3.0`; this project depends on `ultralytics==8.4.135` for every training
and inference path, and the fine-tuned checkpoints are derived from it, so they inherit
it. AGPL-3.0 is strong copyleft — a *network service* built on it must offer its users
the complete corresponding source under the same licence. Concretely:

- Reading this repository, running the app on your own machine, or quoting the numbers
  in this README is not a problem.
- Deploying the Streamlit app as a service, or shipping a model trained on it inside a
  closed product, is — Ultralytics sells a commercial licence for exactly that case.
- The AGPL governs Ultralytics' code. It says nothing about the datasets below, and
  nothing about the upstream files above.

The rest of the dependency list raises no such constraint: `torch` is
Apache-2.0/BSD/MIT, and `numpy`, `opencv-python`, `PyYAML` and `Pillow` are
permissively licensed.

**Datasets are not bundled and keep their own terms.** Every image under
`detection/data/`, `classification/Covid19-dataset/` and `segmentation/` comes from
[Roboflow Universe](https://universe.roboflow.com/tfg-2nmge/yolo-yejbs), the
[COVID-19 Image Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)
and [BUSI](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
respectively, each under whatever licence its own page states — check that page before
redistributing any image, and cite the dataset if you publish results. They are excluded
by `.gitignore` and exist only on the machine that produced the recorded numbers.





