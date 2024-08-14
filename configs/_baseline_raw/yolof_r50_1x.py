_base_ = [
    '../_base_/models/yolof_r50-c5_8xb8-1x_coco.py', # yolof_r50-c5_8xb8-1x_coco.py
    '../_base_/datasets/raw_hdr_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]
num_classes = 5
# model settings

model = dict(
     data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0.,0.,0.],
        std=[1.,1.,1.], # maybe this is magic number
        bgr_to_rgb=False,
        pad_size_divisor=32),
    bbox_head=dict(num_classes=num_classes),
     backbone=dict(
        # type='ResNet',
        type='logCANResNet',
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    )
        


# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.12, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(
        norm_decay_mult=0., custom_keys={'backbone': dict(lr_mult=1. / 3)}))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.00066667,
        by_epoch=False,
        begin=0,
        end=1500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=14,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

work_dir = './experiments/raw/yolof' # 1. use coco 2. our checkpoints
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/yolof/yolof_r50_c5_8x8_1x_coco/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth'
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))