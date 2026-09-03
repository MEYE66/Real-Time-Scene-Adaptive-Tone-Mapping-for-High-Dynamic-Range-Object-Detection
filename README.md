# Real-Time Scene-Adaptive Tone Mapping for High-Dynamic Range Object Detection [NIPS205]

This repository contains the implementation of Real-Time Scene-Adaptive Tone Mapping for High-Dynamic Range Object Detection.
## 1. Summary
Gongzhe Li, Linwei Qiu, Peibei Cao, Fengying Xie, Xiangyang Ji, and Qilin Sun. [Paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/8c83381162f247df48f101b3aaa7c440-Paper-Conference.pdf)

Abstract:


High dynamic range (HDR) images, with their rich tone and detail reproduction, hold significant potential to enhance computer vision systems, particularly in autonomous driving. However, most neural networks for embedded vision are trained on low dynamic range (LDR) inputs and suffer substantial performance degradation when handling high-bit-depth HDR images due to the challenges posed by extreme dynamic ranges. In this paper, we propose a novel tone mapping method that not only bridges the gap between HDR RAW inputs and the LDR sRGB requirements of detection networks but also achieves end-to-end optimization with the downstream tasks. Instead of relying on traditional image signal processing (ISP) pipeline, we introduce neural photometric calibration to regularize dynamic ranges and a scaling-invariant local tone mapping module to preserve image details. In addition, our architecture also supports performance transfer finetuning, enabling efficient adaptation from the LDR model to the HDR RAW model with minimal cost. The proposed method outperforms traditional tone mapping algorithms and advanced AI-ISP methods in challenging automotive HDR scenes. Moreover, our pipeline achieves real-time processing of 4K high-bit-depth HDR inputs on the Nvidia Jetson platform.

Key Idea:

Tone mapping compresses dynamic range while preserving or enhancing local
details. For human viewing, it typically relies on perceptual priors to produce
visually pleasing results. For machine vision, however, we reduce such priors
to preserve a broader LDR solution space. This motivates our use of a
scale-invariant CNN as a local tone mapper with minimal handcrafted priors.

## 2. Requirements


The detection pipeline is
built on [MMDetection](https://github.com/open-mmlab/mmdetection).
Use the PyTorch, MMCV, MMEngine, and MMDetection versions compatible with the local
environment.

## 3. Dataset Preparation

### 3.1 Original dataset layout

Arrange the downloaded [Dataset](https://openi.pcl.ac.cn/innovation_contest/innov202305091731448/datasets?lang=en-US) as follows. Each RAW file and annotation file must
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

### 3.2 RAW preprocessing

The offline RAW preprocessing pipeline performs 24-bit RAW decoding,
demosaicing, resizing to `1280 x 1280`, and Gray-World white balancing. The
implementation is in [preprocessing/parse_raw.py](preprocessing/parse_raw.py).
It expects the input directory to end in `00Train/01Valid` and writes the processed
files to a sibling directory named `RAW`.

For example:

```bash
python preprocessing/parse_raw.py \
    --path ${DATA_PATH}/raws/00Train \
    --threads -1
```

The script uses CUDA for demosaicing. Make sure the processed file extension
matches the `file_name` values in the COCO annotation files.

### 3.3 Annotation format

The detector consumes COCO-style JSON files. Place the final annotation files
at the repository-relative paths below, or update `data_root` and `ann_file`
in the selected dataset configuration:

```text
./dataset/
|-- train.json
|-- val.json
```

`train.json` and `val.json` must contain `images`, `annotations`, and
`categories`. Image entries must point to the corresponding processed image
basename, and image dimensions should be `1280 x 1280`. The conversion logic
for LabelMe-style annotations is available in
[preprocessing/parse_anno.py](preprocessing/parse_anno.py); adapt its input and
output paths to your dataset before generating the final JSON files.

## 4. Pretrained Weights

The pretrained tone-mapper checkpoint is provided under `pretrained_model/`:
And the whold model (with detector) weigths is in https://drive.google.com/drive/folders/1NVflxRPlnr1naFMG_EXgVjm1M85G9Jmm?usp=sharing:

```text
pretrained_model/
`-- tmo_pre.pt
```

`tmo_pre.pt` is the tone-mapper pretraining checkpoint obtained with the NLPD
loss. The tone-mapper implementation is based on
[TMO_CAN](https://github.com/leshier/TMO_CAN). Set the checkpoint path used by
`logCANBaseResNet` to `pretrained_model/tmo_pre.pt` before training or
fine-tuning.

## 5. Training

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


## 6. Citation

If you find our work helpful, please cite the following paper. If you meet some problems when using our code, please feel free to contact me (gongzheli1@link.cuhk.edu.cn).
```text
@InProceedings{li2025realtime,
title={Real-Time Scene-Adaptive Tone Mapping for High-Dynamic Range Object Detection},
author={Gongzhe Li, Linwei Qiu, Peibei Cao, Fengying Xie, Xiangyang Ji and Qilin Sun},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
}
```
