_base_ = [
    '../fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py',
    '../_base_/datasets/rgb_hdr_detection.py',
    # '../_base_/schedules/schedule_1x.py',
    # '../_base_/default_runtime.py',
]



# dataset settings
img_size = (1200, 1280)
num_classes = 5

model = dict(
    bbox_head=dict(num_classes=num_classes),
)



test_dataloader = None
test_cfg = None
test_evaluator = None  # 测试过程使用的评测器
work_dir = './experiments/baseline/fcos-r50' # 1. use coco 2. our checkpoints
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth'
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))