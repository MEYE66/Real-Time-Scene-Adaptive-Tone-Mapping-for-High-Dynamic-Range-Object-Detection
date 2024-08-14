_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/raw_hdr_detection.py',
    # '../_base_/schedules/schedule_1x.py', 
    '../_base_/schedules/schedule_1x_guided.py', 
    '../_base_/default_runtime.py',
]
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
        # type='CANResNet',
        type='logSCANResNet',
        frozen_preprocessor=False,
        in_channels=3,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    bbox_head=dict(num_classes=num_classes),
)

# optimizer
# optim_wrapper = dict(
#     optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

test_dataloader = None
test_cfg = None
test_evaluator = None  
# work_dir = './experiments/raw/retinanet-r50-dual_can' # 
work_dir = './experiments/arch_experiment/tmp_retinanet-r50_logscan_count' # 1. use coco 2. our checkpoints
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))