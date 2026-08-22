#!/usr/bin/env python3
"""One STEP-350 diagnostic QrV2 call per rank in the full shadow package."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import step343_qrv2_cold_case as common
from step350_decode_qrv2_context import decode


VISIBLE = "8,9,10,11,12,13,14,15"
EXPECTED_AIC = "QrV2_step350_context_capture_r1_mix_aic"
ORIGINAL_AIC = "QrV2_566c2e1c0e6c8c92152ad84416d77006_0_mix_aic"
ORIGINAL_AIV = "QrV2_566c2e1c0e6c8c92152ad84416d77006_0_mix_aiv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("candidate",), required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-kernel", required=True)
    parser.add_argument("--original-kernel", required=True)
    parser.add_argument("--installed-custom-opp", required=True)
    parser.add_argument("--overlay-custom-opp", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    rank = int(os.environ["RANK"]); local_rank = int(os.environ["LOCAL_RANK"]); world = int(os.environ["WORLD_SIZE"])
    try:
        shadow_pkg = Path(args.overlay_custom_opp).parents[2].resolve()
        sys.path.insert(0, str(shadow_pkg.parent))
        spec = importlib.util.find_spec("mx_driving_cloud")
        if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(shadow_pkg):
            raise RuntimeError("mx_driving_cloud import origin is not shadow")
        import torch
        import torch.distributed as dist
        import torch_npu
        import mx_driving_cloud
        import mx_driving_cloud._C as mx_c
        if not Path(mx_driving_cloud.__file__).resolve().is_relative_to(shadow_pkg):
            raise RuntimeError("mx_driving_cloud module escaped shadow")
        if not Path(mx_c.__file__).resolve().is_relative_to(shadow_pkg):
            raise RuntimeError("mx_driving_cloud._C escaped shadow")
        if os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":")[0] != args.overlay_custom_opp:
            raise RuntimeError("shadow OPP is not first after import")
        available = bool(torch.npu.is_available()); count = int(torch.npu.device_count())
        gate = rank == local_rank and world == 8 and os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == VISIBLE and available and count == 8
        if not gate: raise RuntimeError("rank/device gate failed")
        torch.npu.set_device(local_rank); dist.init_process_group("hccl")
        loaded = torch.load(Path(args.input_dir) / f"rank{rank}_step10_ind0_192x192_BAD.pt", map_location="cpu", weights_only=True)
        a_cpu = loaded["A"].float().contiguous(); a = a_cpu.to(f"npu:{local_rank}")
        common.atomic_json(output / "ready" / f"rank{rank}.json", {"mode":"candidate","rank":rank,"local_rank":local_rank,"world_size":world,"visible":VISIBLE,"npu_available":available,"device_count":count,"container_pid":os.getpid(),"gate_pass":gate,"shadow_gate":True})
        common.wait_release(output / "release_after_npu_smi"); dist.barrier()
        profile_root = output / "profile_rank0"
        if rank == 0:
            profile_root.mkdir(exist_ok=True)
            with common.profile_context(profile_root) as prof:
                q, raw_r = mx_c.qr(a); torch.npu.synchronize(); prof.step()
        else:
            q, raw_r = mx_c.qr(a); torch.npu.synchronize()
        dist.barrier()
        captured = {"A":a.cpu(),"Q":q.cpu(),"raw_R":raw_r.cpu(),"R":torch.triu(raw_r).cpu()}
        context = decode(captured["raw_R"])
        torch.save({**captured, **context}, output / "captures" / f"rank{rank}.pt")
        identity = None
        if rank == 0:
            common.CANDIDATE_AIC = EXPECTED_AIC; common.CANDIDATE_AIV = "QrV2_step350_context_capture_r1_mix_aiv"
            common.ORIGINAL_AIC = ORIGINAL_AIC; common.ORIGINAL_AIV = ORIGINAL_AIV
            identity = common.verify_profile_hit(profile_root, "candidate")
            mappings, references, _, _ = common.collect_runtime_identity(profile_root)
            referenced_qrv2 = {
                name
                for hash_value, name in mappings.items()
                if references[hash_value] > 0 and name.startswith("QrV2_")
            }
            allowed = {
                EXPECTED_AIC,
                "QrV2_step350_context_capture_r1_mix_aiv",
            }
            unknown = sorted(referenced_qrv2 - allowed)
            if unknown:
                raise RuntimeError(f"unexpected task-referenced QrV2 concrete identity: {unknown}")
            identity["all_referenced_qrv2_concrete"] = sorted(referenced_qrv2)
        common.atomic_json(output / "done" / f"rank{rank}.json", {"rank":rank,"payload_gate":True,"identity":identity})
        dist.destroy_process_group(); return 0
    except BaseException:
        (output / "failure" / f"rank{rank}.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
