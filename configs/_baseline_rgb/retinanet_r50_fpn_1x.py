_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/rgb_hdr_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
]
img_size = (1200, 1280)
num_classes = 5

model = dict(
    bbox_head=dict(num_classes=num_classes),
)
# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

test_dataloader = None
test_cfg = None
test_evaluator = None  
work_dir = './experiments/baseline/retinanet-r50_0' # 1. use coco 2. our checkpoints
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))