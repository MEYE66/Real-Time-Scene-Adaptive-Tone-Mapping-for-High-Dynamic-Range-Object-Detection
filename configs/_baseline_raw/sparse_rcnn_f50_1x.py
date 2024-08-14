_base_ = [
    '../_base_/models/sparse_rcnn_r50.py', # yolof_r50-c5_8xb8-1x_coco.py
    '../_base_/datasets/raw_hdr_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]
num_classes=5
model = dict(
    type='SparseRCNN',
     data_preprocessor=dict(
        mean=[0.,0.,0.],
        std=[1.,1.,1.], # maybe this is magic number
        bgr_to_rgb=False,
        pad_size_divisor=32),
     backbone=dict(
        # type='ResNet',
        type='logSCANResNet',
        norm_cfg=dict(type='BN', requires_grad=True)),
    #   roi_head=dict(
    #     bbox_head=dict(num_classes=num_classes)
    #     )
)


# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True, type='AdamW', lr=0.000025, weight_decay=0.0001),
    clip_grad=dict(max_norm=1, norm_type=2))


work_dir = './experiments/raw/sparse_rcnn_logscan_1e7' # 1. use coco 2. our checkpoints
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco/sparse_rcnn_r50_fpn_1x_coco_20201222_214453-dc79b137.pth'
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))