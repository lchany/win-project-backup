import os

import torch.distributed as dist
from mmcv.runner import HOOKS, Hook
from torch_npu.profiler import (
    AiCMetrics,
    ExportType,
    ProfilerActivity,
    ProfilerLevel,
    _ExperimentalConfig,
    profile,
    schedule,
    tensorboard_trace_handler,
)


@HOOKS.register_module()
class NpuProfilerHook(Hook):
    """Temporary per-rank TorchNPU profiler hook for the baseline run."""

    def __init__(
        self,
        wait=8,
        warmup=1,
        active=4,
        repeat=1,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ):
        self.wait = int(wait)
        self.warmup = int(warmup)
        self.active = int(active)
        self.repeat = int(repeat)
        self.record_shapes = bool(record_shapes)
        self.profile_memory = bool(profile_memory)
        self.with_stack = bool(with_stack)
        self.total_steps = (self.wait + self.warmup + self.active) * self.repeat
        self.steps = 0
        self.profiler = None
        self.closed = False

    def before_run(self, runner):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        output_root = os.environ.get("NPU_PROFILE_DIR")
        if not output_root:
            raise RuntimeError("NPU_PROFILE_DIR must be set for NpuProfilerHook")

        rank_dir = os.path.join(output_root, f"rank{rank}")
        os.makedirs(rank_dir, exist_ok=True)
        trace_handler = tensorboard_trace_handler(
            rank_dir,
            worker_name=f"rank{rank}",
            analyse_flag=True,
            async_mode=False,
        )
        experimental_config = _ExperimentalConfig(
            profiler_level=ProfilerLevel.Level1,
            aic_metrics=AiCMetrics.PipeUtilization,
            data_simplification=False,
            export_type=[ExportType.Text],
        )
        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            schedule=schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
            ),
            on_trace_ready=trace_handler,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            experimental_config=experimental_config,
        )
        self.profiler.__enter__()
        print(
            f"[NPU_PROFILER] rank={rank}/{world_size} output={rank_dir} "
            f"schedule={self.wait}/{self.warmup}/{self.active}/{self.repeat}",
            flush=True,
        )

    def after_train_iter(self, runner):
        if self.profiler is None or self.closed:
            return
        self.profiler.step()
        self.steps += 1
        if self.steps >= self.total_steps:
            self._close()

    def after_run(self, runner):
        self._close()

    def _close(self):
        if self.profiler is not None and not self.closed:
            self.profiler.__exit__(None, None, None)
            self.closed = True
            print(f"[NPU_PROFILER] completed after {self.steps} steps", flush=True)

