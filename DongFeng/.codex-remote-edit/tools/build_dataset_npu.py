def build_dataset(cfg, default_args=None):
    # 1. 适配 mmdet 2.x 的导入路径
    from mmdet.datasets import ConcatDataset, ClassBalancedDataset, RepeatDataset
    # 2. 适配 mmdet3d 1.0.0rc4 的导入路径
    #from mmdet3d.datasets import CBGSDataset, CustomConcatDataset
    from mmdet3d.datasets.dataset_wrappers import CBGSDataset
    # from mmdet3d.datasets import CustomConcatDataset
    if isinstance(cfg, (list, tuple)):
        # 3. 使用 mmdet 2.x 的 ConcatDataset
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        # 4. 适配 ConcatDataset 参数
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True)
        )
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times']
        )
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr']
        )
    elif cfg['type'] == 'CBGSDataset':
        dataset = CBGSDataset(build_dataset(cfg['dataset'], default_args))
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        from projects.mmdet3d_plugin.datasets.builder import custom_concat_dataset
        dataset = custom_concat_dataset(cfg, default_args)
    else:
        # 6. 使用 mmdet 2.x 的 build_from_cfg
        from mmcv.utils import build_from_cfg
        from mmdet3d.datasets import DATASETS
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset
