#!/usr/bin/env python3
"""STEP-247: hybrid QR — NPU FP32 for dim<=1024, CPU FP64 for larger."""
import re
from pathlib import Path

SOAP = Path("/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang/projects/mmdet3d_plugin/optimizers/soap.py")
data = SOAP.read_bytes()

old = (
    b"    def _parallel_qr_waves(self, power_iters):\n"
    b"        \"\"\"Run linalg.qr on up to world_size inputs concurrently per wave.\"\"\"\n"
    b"        if not power_iters:\n"
    b"            return []\n"
    b"        if not self._dist_qr_enabled():\n"
    b"            return [torch.linalg.qr(pi)[0] for pi in power_iters]\n"
    b"        import torch.distributed as dist\n"
    b"        rank = dist.get_rank()\n"
    b"        world = dist.get_world_size()\n"
    b"        results = [None] * len(power_iters)\n"
    b"        for wave_start in range(0, len(power_iters), world):\n"
    b"            wave_len = min(world, len(power_iters) - wave_start)\n"
    b"            local_Q = None\n"
    b"            if rank < wave_len:\n"
    b"                local_Q, _ = torch.linalg.qr(power_iters[wave_start + rank])\n"
    b"                local_Q = local_Q.contiguous()\n"
)

new = (
    b"    def _hybrid_qr(self, power_iter):\n"
    b"        \"\"\"NPU FP32 QR when dim<=SOAP_CPU_QR_DIM (default 1024); else CPU FP64.\"\"\"\n"
    b"        dim = int(power_iter.shape[-1])\n"
    b"        cpu_dim = int(os.environ.get(\"SOAP_CPU_QR_DIM\", \"1024\"))\n"
    b"        if dim > cpu_dim:\n"
    b"            orig_dtype = power_iter.dtype\n"
    b"            orig_device = power_iter.device\n"
    b"            Q, _ = torch.linalg.qr(\n"
    b"                power_iter.detach().to(device=\"cpu\", dtype=torch.float64))\n"
    b"            return Q.to(device=orig_device, dtype=orig_dtype).contiguous()\n"
    b"        Q, _ = torch.linalg.qr(power_iter)\n"
    b"        return Q.contiguous()\n"
    b"\n"
    b"    def _parallel_qr_waves(self, power_iters):\n"
    b"        \"\"\"Run hybrid QR on up to world_size inputs concurrently per wave.\"\"\"\n"
    b"        if not power_iters:\n"
    b"            return []\n"
    b"        if not self._dist_qr_enabled():\n"
    b"            return [self._hybrid_qr(pi) for pi in power_iters]\n"
    b"        import torch.distributed as dist\n"
    b"        rank = dist.get_rank()\n"
    b"        world = dist.get_world_size()\n"
    b"        results = [None] * len(power_iters)\n"
    b"        for wave_start in range(0, len(power_iters), world):\n"
    b"            wave_len = min(world, len(power_iters) - wave_start)\n"
    b"            local_Q = None\n"
    b"            if rank < wave_len:\n"
    b"                local_Q = self._hybrid_qr(power_iters[wave_start + rank])\n"
)

if old not in data:
    raise SystemExit("anchor _parallel_qr_waves block not found (LF). trying CRLF mix")

data = data.replace(old, new, 1)
SOAP.write_bytes(data)
print("patched_ok hybrid", data.count(b"_hybrid_qr"), "linalg.qr", data.count(b"torch.linalg.qr"))
