# Real-Time Scene-Adaptive Tone Mapping for High-Dynamic Range Object Detection

This repository contains the implementation of our scene-adaptive tone mapping
method for high-dynamic-range (HDR) object detection. The detection pipeline is
built on [MMDetection](https://github.com/open-mmlab/mmdetection).

## 1. Authors

Gongzhe Li, Linwei Qiu, Peibei Cao, Fengying Xie, Xiangyang Ji, and Qilin Sun.

## 2. Downloads

- [Paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/8c83381162f247df48f101b3aaa7c440-Paper-Conference.pdf)
- [Dataset](https://openi.pcl.ac.cn/innovation_contest/innov202305091731448/datasets?lang=en-US)

## 3. Requirements

Install the dependencies listed in `requirements.txt`, then install this
repository in editable mode from its root directory:

```bash
pip install -r requirements.txt
pip install -v -e .
```

Training and inference require PyTorch installation. Use the
PyTorch, MMCV, MMEngine, and MMDetection versions compatible with the local
environment.

## 4. Dataset Preparation

### 4.1 Original dataset layout

Arrange the downloaded data as follows. Each RAW file and annotation file must
have the same basename.

```text
${DATA_PATH}/
|-- raws/
|   |-- 00Train/
|       |-- name1.raw
|       |-- name2.raw
|       |-- ...
|-- anno/
    |-- 00Train/
        |-- name1.json
        |-- name2.json
        |-- ...
```

The annotation labels used by the provided dataset configuration are:
`Pedestrian`, `Car`, `Cyclist`, `Tram`, `Tricycle`, and `Truck`.

### 4.2 RAW preprocessing

The offline RAW preprocessing pipeline performs 24-bit RAW decoding,
demosaicing, resizing to `1280 x 1280`, and Gray-World white balancing. The
implementation is in [preprocessing/parse_raw.py](preprocessing/parse_raw.py).
It expects the input directory to end in `00Train` and writes the processed
files to a sibling directory named `RAW`.

For example:

```bash
python preprocessing/parse_raw.py \
    --path ${DATA_PATH}/raws/00Train \
    --threads -1
```

The script uses CUDA for demosaicing. Make sure the processed file extension
matches the `file_name` values in the COCO annotation files.

### 4.3 Annotation format

The detector consumes COCO-style JSON files. Place the final annotation files
at the repository-relative paths below, or update `data_root` and `ann_file`
in the selected dataset configuration:

```text
./dataset/
|-- train.json
|-- val.json
|-- train/          # processed training images
|-- val/            # processed validation images
```

`train.json` and `val.json` must contain `images`, `annotations`, and
`categories`. Image entries must point to the corresponding processed image
basename, and image dimensions should be `1280 x 1280`. The conversion logic
for LabelMe-style annotations is available in
[preprocessing/parse_anno.py](preprocessing/parse_anno.py); adapt its input and
output paths to your dataset before generating the final JSON files.

## 5. Pretrained Weights

The repository provides three checkpoints under `pretrained_model/`. The two
detector checkpoints are stored with Git LFS because they exceed GitHub's
regular per-file size limit. After cloning the repository, install Git LFS and
download the checkpoint contents with:

```bash
git lfs install
git lfs pull --include="pretrained_model/**"
```

The checkpoint directory is organized as follows:

```text
pretrained_model/
|-- tmo_pre.pt
|-- faster-rcnn/
|   `-- epoch_12.pth
`-- yolo3/
    `-- epoch_12.pth
```

| Model | Checkpoint | Size | SHA256 |
| --- | --- | ---: | --- |
| Tone mapper | `pretrained_model/tmo_pre.pt` | 456 KiB | `09b8d543d98469e38239d77e0dc700291bc071438b950071f4623b310bc95c73` |
| Faster R-CNN | `pretrained_model/faster-rcnn/epoch_12.pth` | 321 MiB | `73a8dec41662674380aa1ae3b93a9adf619683f8d508067bd66c5f5aa7edb100` |
| YOLOv3 | `pretrained_model/yolo3/epoch_12.pth` | 475 MiB | `7f4f6799e89bc6aa414e5ff3db9a14c3dd1db9aa72a494934941b30fc31c5448` |

`tmo_pre.pt` is the tone-mapper pretraining checkpoint obtained with the NLPD
loss. The tone-mapper implementation is based on
[TMO_CAN](https://github.com/leshier/TMO_CAN). Set the checkpoint path used by
`logCANBaseResNet` to `pretrained_model/tmo_pre.pt` before training or
fine-tuning.

The Faster R-CNN checkpoint can be evaluated with:

```bash
python tools/test.py \
    configs/_baseline_raw/faster_rcnn_r50_fpn_1x.py \
    pretrained_model/faster-rcnn/epoch_12.pth
```

The YOLOv3 checkpoint can be evaluated with:

```bash
python tools/test.py \
    configs/_baseline_raw/yolov3_d53_coco.py \
    pretrained_model/yolo3/epoch_12.pth
```

The configuration files currently define their own `load_from` values. For
training or resuming from one of the detector checkpoints, update `load_from`
to the corresponding path above. Passing a checkpoint directly to
`tools/test.py`, as shown in the examples, overrides the need to edit
`load_from` for evaluation.

## 6. Training

Before running a job, update `data_root`, `data_prefix`, and `ann_file` in the
dataset base configuration used by the selected model. For the RAW detector,
the default HDR configuration is
[configs/_base_/datasets/raw_hdr_detection.py](configs/_base_/datasets/raw_hdr_detection.py).
For the RGB baseline, use
[configs/_base_/datasets/rgb_hdr_detection.py](configs/_base_/datasets/rgb_hdr_detection.py).

Example single-GPU training:

```bash
python tools/train.py \
    configs/_baseline_raw/faster_rcnn_r50_fpn_1x.py \
    --work-dir ./experiments/raw/faster_rcnn
```

Example distributed training on two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh \
    configs/_baseline_raw/faster_rcnn_r50_fpn_1x.py 2
```


## 8. Citation

If you find our work helpful, please cite the following paper.
```text
@InProceedings{li2025realtime,
title={Real-Time Scene-Adaptive Tone Mapping for High-Dynamic Range Object Detection},
author={Gongzhe Li, Linwei Qiu, Peibei Cao, Fengying Xie, Xiangyang Ji and Qilin Sun},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
}
```
