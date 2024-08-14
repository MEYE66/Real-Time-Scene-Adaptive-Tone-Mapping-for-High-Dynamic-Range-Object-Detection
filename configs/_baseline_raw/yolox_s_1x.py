_base_ = [
    '../_base_/models/yolox_s.py', # yolof_r50-c5_8xb8-1x_coco.py
    '../_base_/datasets/raw_hdr_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]
num_classes = 5
# model settings

model = dict(
    type='YOLOX',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0, 0, 0],
        std=[1.,1.,1.], # maybe this is magic number
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        # type='CSPDarknet',
         type='logCANCSPDarknet',
        ),
    bbox_head=dict(num_classes=num_classes)
)


# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4,
        nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))


work_dir = './experiments/raw/yolox' # 1. use coco 2. our checkpoints
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))