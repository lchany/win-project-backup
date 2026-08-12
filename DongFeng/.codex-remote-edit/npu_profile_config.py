_base_ = [
    "./projects/configs/20260113st/"
    "mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_"
    "no-lidar-ca_new-bb_gop_occ_finetune.py"
]

custom_imports = dict(
    imports=[
        "projects.mmdet3d_plugin",
        "projects.fsdv2",
        "projects.mmdet3d_plugin.core.hook.gradient_fingerprint_optimizer_hook",
        "npu_profiler_hook",
    ],
    allow_failed_imports=False,
)

custom_hooks = [
    dict(
        type="NpuProfilerHook",
        wait=8,
        warmup=1,
        active=4,
        repeat=1,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        priority="LOWEST",
    )
]
