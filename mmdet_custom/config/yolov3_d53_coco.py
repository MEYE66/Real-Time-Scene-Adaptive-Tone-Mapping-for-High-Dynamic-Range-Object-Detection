_base_ = [
    '../_base_/models/yolo3_d53.py',
    '../_base_/datasets/rod_raw_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

num_classes = 6
model = dict(
    type='YOLOV3',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0.,0.,0.],
        std=[1.,1.,1.], # maybe this is magic number
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='logCANBaseResNet',
        depth=53,
        out_indices=(3, 4, 5),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://darknet53')),
    bbox_head=dict(num_classes=num_classes),
)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005),
    clip_grad=dict(max_norm=35, norm_type=2))

# # learning policy
# param_scheduler = [
#     dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=2000),
#     dict(type='MultiStepLR', by_epoch=True, milestones=[218, 246], gamma=0.1)
# ]
test_dataloader = None
test_cfg = None
test_evaluator = None  # 测试过程使用的评测器
work_dir = './experiments/' # 1. use coco 2. our checkpoints faster-rcnn_logbasecan-scale2
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth'
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))
