log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False),
                        dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoderCellPose',
    backbone=dict(
        type='mit_b2',
        style='pytorch',
        pretrained='pretrained/mit_b2.pth',
    ),
    decode_head=dict(
        type='CellPoseHead',
        # b1b2[64, 128, 320, 512]  b0[32, 64, 160, 256]
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg)),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    train_cfg=dict(
        work_dir='./work_dirs/tissuenet'
    ),
    # test_cfg=dict(mode='whole')
    test_cfg=dict(mode='slide2', crop_size=(256, 256),
                  stride=(256, 256), crop_output_size=(200, 200))
)

data_root = 'data/PanNuke/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsNpy'),
    # dict(type='Resize', ratio_range=(0.8, 1.2), multiscale_mode='range'),
    # dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', direction='horizontal', prob=0.5),
    dict(type='RandomFlip', direction='vertical', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
    dict(type='GetCellPoseMap'),
    dict(type='Normalize99'),
    # dict(type='Pad', size_divisor=32, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'gt_vec'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor',
                    'img_norm_cfg')
         )
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=None,
        img_ratios=[1.0, ],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize99'),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                                                          'img_shape', 'pad_shape', 'scale_factor',
                                                          'flip',
                                                          'flip_direction',
                                                          ))
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='NucleiCellPoseDataset',
        data_root=data_root,
        img_dir='images/Fold 1/',
        ann_dir='labels/Fold 1/',
        pipeline=train_pipeline),
    val=dict(
        type='NucleiCellPoseDataset',
        data_root=data_root,
        img_dir='images/Fold_2_1000/',
        ann_dir='labels/Fold_2_1000/',
        pipeline=test_pipeline),
    test=dict(
        type='NucleiCellPoseDataset',
        data_root="demo",  # 测试图片放在data_root/img_dir下，以img_suffix的后缀
        img_dir='images/',
        ann_dir='labels/Fold 3/',
        img_suffix='.png',
        pipeline=test_pipeline))

optimizer = dict(
    type='AdamW',
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            # head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))

optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
seed = 0

runner = dict(type='IterBasedRunner', max_iters=60000)
checkpoint_config = dict(by_epoch=False, interval=3000, save_optimizer=False)
evaluation = dict(interval=3000, metric=['mIoU', 'mDice'])

work_dir = '/comp_robot/dino-E/tb-log/1129/segformer/pannuk/1'
