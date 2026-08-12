from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


LARGE_AXIS_THRESHOLD = 1024


def make_spd(n: int, device: torch.device) -> torch.Tensor:
    x = torch.arange(n, device=device, dtype=torch.float32)
    diagonal = 1.0 + x / max(n, 1)
    u = ((x.remainder(31.0) - 15.0) / 31.0).unsqueeze(1)
    return torch.diag(diagonal).add_(u @ u.T, alpha=1.0 / max(n, 1))


def canonicalize_columns(q: torch.Tensor) -> torch.Tensor:
    pivots = q.abs().argmax(dim=0)
    columns = torch.arange(q.shape[1], device=q.device)
    signs = torch.where(q[pivots, columns] < 0, -1.0, 1.0)
    return q * signs.unsqueeze(0)


def one_qr_round(m: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    estimated = torch.diag(q.T @ m @ q)
    order = torch.argsort(estimated, descending=True, stable=True)
    q_sorted = q.index_select(1, order)
    q_new, _ = torch.linalg.qr(m @ q_sorted)
    return q_new, order


def active_axes(shape: tuple[int, ...]) -> tuple[bool, ...]:
    if len(shape) == 2 and max(shape) > LARGE_AXIS_THRESHOLD:
        smaller = min(range(2), key=lambda index: (shape[index], index))
        return tuple(index == smaller for index in range(2))
    return tuple(True for _ in shape)


def project(value: torch.Tensor, bases: list[torch.Tensor | None]) -> torch.Tensor:
    result = value
    for basis in bases:
        if basis is None:
            result = result.permute(tuple(range(1, result.ndim)) + (0,))
        else:
            result = torch.tensordot(result, basis, dims=([0], [0]))
    return result


def project_back(value: torch.Tensor, bases: list[torch.Tensor | None]) -> torch.Tensor:
    result = value
    for basis in bases:
        if basis is None:
            result = result.permute(tuple(range(1, result.ndim)) + (0,))
        else:
            result = torch.tensordot(result, basis, dims=([0], [1]))
    return result


def algebra_gate(device: torch.device) -> tuple[float, bool, tuple[int, ...]]:
    shape = (32, 64)
    value = (
        torch.arange(shape[0] * shape[1], device=device, dtype=torch.float32)
        .reshape(shape)
        .remainder_(97.0)
        .div_(97.0)
    )
    bases = []
    orders = []
    for dim in shape:
        q, order = one_qr_round(make_spd(dim, device), torch.eye(dim, device=device))
        bases.append(q)
        orders.append(order)
    restored = project_back(project(value, bases), bases)
    relative_error = (restored - value).norm() / value.norm()

    exp_avg_sq = value.square()
    original_sum = exp_avg_sq.sum()
    for axis, order in enumerate(orders):
        exp_avg_sq = exp_avg_sq.index_select(axis, order)
    permutation_ok = bool(
        exp_avg_sq.shape == shape
        and torch.allclose(exp_avg_sq.sum(), original_sum, rtol=1e-6, atol=1e-6)
    )
    return float(relative_error.item()), permutation_ok, tuple(exp_avg_sq.shape)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group(backend="hccl")

    if local_rank == 0:
        print("AXIS_SELECTION", (2560, 5120), active_axes((2560, 5120)), flush=True)
        print("AXIS_SELECTION", (256, 2560), active_axes((256, 2560)), flush=True)
        algebra_error, permutation_ok, permutation_shape = algebra_gate(device)
        print(
            "ALGEBRA_GATE",
            f"roundtrip_rel={algebra_error:.9e}",
            f"permutation_ok={permutation_ok}",
            f"shape={permutation_shape}",
            flush=True,
        )

    warm = make_spd(64, device)
    one_qr_round(warm, torch.eye(64, device=device))
    torch.npu.synchronize()
    dist.barrier()

    for n in (256, 1024, 2560):
        matrix = make_spd(n, device)
        initial_q = torch.eye(n, device=device, dtype=torch.float32)
        torch.npu.synchronize()
        start = time.perf_counter()
        q, _ = one_qr_round(matrix, initial_q)
        torch.npu.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        q = canonicalize_columns(q)
        identity = torch.eye(n, device=device, dtype=torch.float32)
        orthogonal_error = ((q.T @ q - identity).norm() / identity.norm()).float()
        transformed = q.T @ matrix @ q
        off_diagonal = transformed - torch.diag(torch.diag(transformed))
        diagonalization_error = (off_diagonal.norm() / transformed.norm()).float()

        probe_index = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
        probe_scale = torch.arange(1, 5, device=device, dtype=torch.float32).unsqueeze(0)
        probe = torch.cos((probe_index + 1.0) * probe_scale / max(n, 1))
        sketch = (q.T @ probe).reshape(-1)
        gathered_sketches = [torch.empty_like(sketch) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_sketches, sketch)

        metrics = torch.tensor(
            [elapsed_ms, orthogonal_error.item(), diagonalization_error.item()],
            device=device,
            dtype=torch.float64,
        )
        gathered_metrics = [torch.empty_like(metrics) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_metrics, metrics)

        if local_rank == 0:
            metric_matrix = torch.stack(gathered_metrics).cpu()
            reference = gathered_sketches[0]
            rank_diffs = torch.stack(
                [
                    (candidate - reference).norm() / reference.norm().clamp_min(1e-30)
                    for candidate in gathered_sketches[1:]
                ]
            )
            print(
                "QR_GATE",
                f"n={n}",
                f"time_ms_min={metric_matrix[:, 0].min().item():.3f}",
                f"time_ms_median={metric_matrix[:, 0].median().item():.3f}",
                f"time_ms_max={metric_matrix[:, 0].max().item():.3f}",
                f"orth_max={metric_matrix[:, 1].max().item():.9e}",
                f"diag_max={metric_matrix[:, 2].max().item():.9e}",
                f"rank_sketch_rel_max={rank_diffs.max().item():.9e}",
                flush=True,
            )
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
