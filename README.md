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

This project is implemented as an extension of
[MMDetection](https://github.com/open-mmlab/mmdetection). Before using this
code, install MMDetection and its compatible dependencies, including PyTorch,
MMCV, and MMEngine, by following the official installation instructions.

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

The pretrained tone-mapper checkpoint is included in this repository:

```text
pretrained_model/
`-- tmo_pre.pt
```

| Model | Checkpoint | Size | SHA256 |
| --- | --- | ---: | --- |
| Tone mapper | `pretrained_model/tmo_pre.pt` | 456 KiB | `09b8d543d98469e38239d77e0dc700291bc071438b950071f4623b310bc95c73` |

`tmo_pre.pt` is the tone-mapper pretraining checkpoint obtained with the NLPD
loss. The tone-mapper implementation is based on
[TMO_CAN](https://github.com/leshier/TMO_CAN). Set the checkpoint path used by
`logCANBaseResNet` to `pretrained_model/tmo_pre.pt` before training or
fine-tuning.

Complete detector checkpoints for **Faster R-CNN** and **YOLOv3** are available
on [Google Drive](https://drive.google.com/drive/folders/1NVflxRPlnr1naFMG_EXgVjm1M85G9Jmm?usp=sharing). They are organized as follows:

```text
pretrained_model/
|-- faster r-cnn/
`-- yolov3/
```

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
