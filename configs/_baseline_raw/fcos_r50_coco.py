_base_ = [
    '../_base_/datasets/raw_hdr_detection.py',
    '../fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py',
    # '../_base_/schedules/schedule_1x.py',
    # '../_base_/default_runtime.py',
]
# dataset settings
img_size = (1200, 1280)
num_classes = 5

model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[1., 1., 1.],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        # type='ResNet',
        type='logCANResNet',
        frozen_preprocessor=False,
        in_channels=3,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe'))
    ,
    bbox_head=dict(num_classes=num_classes),
)

test_dataloader = None
test_cfg = None
test_evaluator = None  # 测试过程使用的评测器
work_dir = './experiments/raw/fcos-r50/' # 1. use coco 2. our checkpoints
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth'
default_hooks = dict(checkpoint=dict(max_keep_ckpts=5))