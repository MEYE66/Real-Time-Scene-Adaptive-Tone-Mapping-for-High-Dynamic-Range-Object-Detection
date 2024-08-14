# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=13, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=14,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=20,
#         by_epoch=True,
#         milestones=[9, 15, 17],
#         gamma=0.1)
# ]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))


# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
#     # paramwise_cfg = dict(
#     # for two stage
#     # custom_keys={
#     #     # 'backbone': dict(lr_mult=0.0, decay_mult=0.0),
#     #     'rpn_head': dict(lr_mult=0.0, decay_mult=0.0),
#     #     'roi_head': dict(lr_mult=0.0, decay_mult=0.0),
#     #     'neck': dict(lr_mult=0.0, decay_mult=0.0),
#     # }),
# #     # for single stage
# #     # custom_keys={
# #     #     'backbone': dict(lr_mult=0.0, decay_mult=0.0),
# #     #     'bbox_head': dict(lr_mult=0.0, decay_mult=0.0),
# #     #     'neck': dict(lr_mult=0.0, decay_mult=0.0),
# #     # }),
# )


# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
