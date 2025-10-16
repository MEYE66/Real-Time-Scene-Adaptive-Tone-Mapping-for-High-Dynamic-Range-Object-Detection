# [NIPS2025] Real-Time Scene-Adaptive Tone Mapping for High-Dynamic Range Object Detection

The implementation of our NIPS 2025 paper ["Real-Time Scene-Adaptive Tone Mapping for High-Dynamic Range Object Detection"](https://neurips.cc/virtual/2025/poster/116673#abstract_details).

## Installation
### Set up the python environment
```
conda create -n adaptiveisp python=3.10 \
conda activate adaptiveisp
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
git clone https://github.com/OpenImagingLab/AdaptiveISP.git
cd AdaptiveISP
pip install -r requirements.txt
```

## Prepare Dataset and Moodel
1. Download the RoD dataset from [Dataset](https://openi.pcl.ac.cn/innovation_contest/innov202305091731448/datasets?lang=en-US).
2. The pretrained tone mapper weights .

3.We apply some pre-processing on RAW images, including Demosaicing, Resizing, and Gray-World White Balance. Our scripts for RAW preprocessing can be seen here. We also provide the script for annotation preprocessing here.
```
${DATA_PATH}
|-- raws
    |-- 00Train
        |-- name1.raw
        |-- name2.raw
        |-- ...
|-- anno
    |-- 00Train
        |-- name1.json
        |-- name2.json
        |-- ...
```

To use our scripts, please arrange your dataset in the following format:



## Citation
```
@inproceedings{Hong2021Crafting,
	title={Crafting Object Detection in Very Low Light},
	author={Yang Hong, Kaixuan Wei, Linwei Chen, Ying Fu},
	booktitle={BMVC},
	year={2021}
}
```


