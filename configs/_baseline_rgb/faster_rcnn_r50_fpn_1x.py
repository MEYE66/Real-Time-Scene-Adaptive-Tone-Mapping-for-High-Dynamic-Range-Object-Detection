_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/rgb_hdr_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]


num_classes = 5
# model settings
model = dict(
     data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0.,0.,0.],
        std=[255.,255.,255.], 
        bgr_to_rgb=True,
        pad_size_divisor=32),
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes)))

# test_dataloader = None
# test_cfg = None
# test_evaluator = None  # 测试过程使用的评测器
work_dir = './arch_experiment/faster-rcnn_day/' # 1. use coco 2. our checkpoints
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))