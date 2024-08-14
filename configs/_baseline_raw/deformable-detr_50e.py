_base_ = [
    '../_base_/models/deformable-detr_r50_16xb2-50e_coco.py', #models/deformable-detr_r50_16xb2-50e_coco.py
    '../_base_/datasets/raw_hdr_detection.py',
    # '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0, 0, 0],
        std=[1.,1.,1.], # maybe this is magic number
        bgr_to_rgb=False,
        pad_size_divisor=1),
  backbone=dict(
        type='logSCANResNet',
        # type='logCANResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
)


work_dir = './experiments/raw/dedeter_logscan_1e7' # 1. use coco 2. our checkpoints
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/deformable_detr/deformable-detr_r50_16xb2-50e_coco/deformable-detr_r50_16xb2-50e_coco_20221029_210934-6bc7d21b.pth'
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))