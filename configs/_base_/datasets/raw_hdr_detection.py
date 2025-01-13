dataset_type = 'RoDDataset'
# data_root = '/home/wzh/data/RhoVision/'
# data_root = '/home/lgz/data/HDR_RAW/scene/' temp_npy
data_root = '/home/ligongzhe/data/'
metainfo = {
    'classes': ('Pedestrian', 'Car', 'Cyclist', 'Tram', 'Tricycle','Truck',),
}
num_classes = 6
backend_args = None
batch_size = 8
num_worker = 8
img_size = (1280, 1280)
GPU = 4


train_pipeline = [  # 训练数据处理流程
    dict(type='LoadImageFromNDRaw',color_type='unchanged',imdecode_backend='cv2'),  # 第 1 个流程，从文件路径里加载图像。
    dict(
        type='LoadAnnotations', with_bbox=True),  # 是否使用标注框(bounding box)， 目标检测需要设置为 True。
    dict(
        type='Resize',  # 变化图像和其标注大小的流程。
        scale=img_size,  # 图像的最大尺寸
        keep_ratio=True  # 是否保持图像的长宽比。
        ),
    dict(
        type='RandomFlip',  # 翻转图像和其标注的数据增广流程。
        prob=0.5),  # 翻转图像的概率。
    dict(type='PackDetInputs')  # 将数据转换为检测器输入格式的流程
]


# 测试数据处理流程
val_pipeline = [
    dict(type='LoadImageFromNDRaw',color_type='unchanged',imdecode_backend='cv2',),  # 第 1 个流程，从文件路径里加载图像。
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_size, keep_ratio=True),  # 变化图像大小的流程。
    dict(
            type='PackDetInputs',  # 将数据转换为检测器输入格式的流程
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(  # 训练 dataloader 配置
    batch_size=batch_size,  # 单个 GPU 的 batch size
    num_workers=num_worker,  # 单个 GPU 分配的数据加载线程数
    persistent_workers=True,  # 如果设置为 True，dataloader 在迭代完一轮之后不会关闭数据读取的子进程，可以加速训练
    sampler=dict(  # 训练数据的采样器
        type='DefaultSampler',  # 默认的采样器，同时支持分布式和非分布式训练。请参考 https://mmengine.readthedocs.io/zh_CN/latest/api/generated/mmengine.dataset.DefaultSampler.html#mmengine.dataset.DefaultSampler
        shuffle=True),  # 随机打乱每个轮次训练数据的顺序
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # 批数据采样器，用于确保每一批次内的数据拥有相似的长宽比，可用于节省显存
    dataset=dict(  # 训练数据集的配置
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        # data_prefix=dict(img='raws_val/night'),  # 图片路径前缀
        # data_prefix=dict(img='raws_val/day'),  # 图片路径前缀
        data_prefix=dict(img='RAWtiff2'),  # 图片路径前缀 # det_tmo
        ann_file='annotations/total/tiff/train.json',  # 标注文件路径
        filter_cfg=dict(filter_empty_gt=True, min_size=32),  # 图片和标注的过滤配置
        pipeline=train_pipeline))  # 这是由之前创建的 train_pipeline 定义的数据处理流程。


val_dataloader = dict(  # 验证 dataloader 配置
    batch_size=batch_size,  # 单个 GPU 的 Batch size。如果 batch-szie > 1，组成 batch 时的额外填充会影响模型推理精度
    num_workers=num_worker,  # 单个 GPU 分配的数据加载线程数
    persistent_workers=True,  # 如果设置为 True，dataloader 在迭代完一轮之后不会关闭数据读取的子进程，可以加速训练
    drop_last=False,  # 是否丢弃最后未能组成一个批次的数据
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),  # 验证和测试时不打乱数据顺序
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        # data_prefix=dict(img='raws_val/day'),  # 图片路径前缀
        data_prefix=dict(img='RAWtiff2'),  # 图片路径前缀
        ann_file='annotations/total/tiff/val.json',  # 标注文件路径
        test_mode=True,  # 开启测试模式，避免数据集过滤图片和标注
        pipeline=val_pipeline))


val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/total/tiff/val.json',
    metric=['bbox'],
    format_only=False,
    backend_args=backend_args)

test_dataloader = val_dataloader
test_evaluator = val_evaluator
