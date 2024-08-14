_base_ = [
    '../_base_/models/yolo3_d53.py',
    '../_base_/datasets/rgb_hdr_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

num_classes = 5
model = dict(
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
work_dir = './experiments/reinhard/yolo3-d53' # 1. use coco 2. our checkpoints
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth'
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))
