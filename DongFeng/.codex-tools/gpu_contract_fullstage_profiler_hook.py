import os

from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import get_dist_info
from torch_npu import profiler


@HOOKS.register_module()
class GPUContractFullStageProfilerHook(Hook):
    """Collect exactly one rank-0 trace after the training step is stable."""

    def __init__(self, output_dir, wait=22, warmup=1, active=4):
        self.output_dir = output_dir
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self._profiler = None

    def before_run(self, runner):
        rank, world_size = get_dist_info()
        runner.logger.info(
            "GPU-contract profiler gate rank=%s world_size=%s output=%s "
            "wait=%s warmup=%s active=%s expected_active_steps=23-26",
            rank,
            world_size,
            self.output_dir,
            self.wait,
            self.warmup,
            self.active,
        )
        if world_size != 8:
            raise RuntimeError(f"profiler requires world_size=8, got {world_size}")
        if rank != 0:
            return

        if os.path.exists(self.output_dir) and os.listdir(self.output_dir):
            raise RuntimeError(
                f"refusing to overwrite non-empty profile output: {self.output_dir}"
            )
        os.makedirs(self.output_dir, exist_ok=True)
        handler = profiler.tensorboard_trace_handler(
            self.output_dir,
            worker_name="rank0",
            analyse_flag=True,
            async_mode=False,
        )
        self._profiler = profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.NPU,
            ],
            schedule=profiler.schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=1,
            ),
            on_trace_ready=handler,
            record_shapes=True,
            profile_memory=False,
            with_stack=True,
            experimental_config=profiler._ExperimentalConfig(
                profiler_level=profiler.ProfilerLevel.Level0,
                aic_metrics=profiler.AiCMetrics.AiCoreNone,
                l2_cache=False,
                data_simplification=True,
                export_type="text",
            ),
        )
        self._profiler.start()

    def after_train_iter(self, runner):
        if self._profiler is not None:
            self._profiler.step()

    def after_run(self, runner):
        if self._profiler is not None:
            self._profiler.stop()
            self._profiler = None
