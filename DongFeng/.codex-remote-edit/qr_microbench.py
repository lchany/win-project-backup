import math
import statistics
import time

import torch
import torch_npu


SHAPE_COUNTS = {
    1: 106,
    3: 30,
    4: 6,
    7: 37,
    8: 1,
    11: 1,
    22: 1,
    32: 4,
    40: 9,
    64: 28,
    96: 3,
    120: 1,
    128: 18,
    160: 1,
    192: 32,
    220: 4,
    256: 181,
    352: 1,
    440: 4,
    512: 43,
    768: 22,
    1024: 6,
    2560: 8,
    5120: 4,
}


def make_spd(n: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    diagonal = torch.linspace(1.0, 2.0, n, dtype=dtype, device=device)
    matrix = torch.diag(diagonal)
    if n > 1:
        off_diagonal = torch.full((n - 1,), 0.01, dtype=dtype, device=device)
        matrix = matrix + torch.diag(off_diagonal, 1) + torch.diag(off_diagonal, -1)
    return matrix


def soap_qr(matrix: torch.Tensor, orthogonal: torch.Tensor):
    product = matrix @ orthogonal
    eigenvalues = torch.diag(orthogonal.T @ product)
    order = torch.argsort(eigenvalues, descending=True)
    power_iteration = product[:, order]
    return (*torch.linalg.qr(power_iteration), power_iteration)


def percentile(values, p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def main() -> None:
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    results = []
    print("VERSIONS", torch.__version__, torch_npu.__version__, flush=True)

    for n, count in sorted(SHAPE_COUNTS.items()):
        cpu_matrix = make_spd(n, dtype=torch.float64, device=torch.device("cpu"))
        cpu_orthogonal = torch.eye(n, dtype=torch.float64)
        cpu_start = time.perf_counter()
        cpu_q, cpu_r, cpu_power = soap_qr(cpu_matrix, cpu_orthogonal)
        cpu_ms = (time.perf_counter() - cpu_start) * 1000

        npu_matrix = cpu_matrix.float().to(device)
        npu_orthogonal = cpu_orthogonal.float().to(device)
        soap_qr(npu_matrix, npu_orthogonal)
        torch.npu.synchronize()
        timings = []
        q = r = power = None
        for _ in range(2):
            start = time.perf_counter()
            q, r, power = soap_qr(npu_matrix, npu_orthogonal)
            torch.npu.synchronize()
            timings.append((time.perf_counter() - start) * 1000)

        identity = torch.eye(n, dtype=torch.float32, device=device)
        orthogonal_error = (
            torch.linalg.vector_norm(q.T @ q - identity) / max(n, 1) ** 0.5
        ).item()
        reconstruction_error = (
            torch.linalg.vector_norm(q @ r - power) / torch.linalg.vector_norm(power)
        ).item()
        q_cpu_from_npu = q.cpu().double()
        signs = torch.sign(torch.sum(q_cpu_from_npu * cpu_q, dim=0))
        signs[signs == 0] = 1
        aligned_error = (
            torch.linalg.vector_norm(q_cpu_from_npu * signs - cpu_q)
            / torch.linalg.vector_norm(cpu_q)
        ).item()
        npu_ms = statistics.median(timings)
        results.append((n, count, npu_ms, cpu_ms))
        print(
            "SHAPE",
            n,
            "count",
            count,
            "npu_ms",
            round(npu_ms, 3),
            "cpu_fp64_ms",
            round(cpu_ms, 3),
            "speedup",
            round(cpu_ms / npu_ms, 3),
            "orth",
            f"{orthogonal_error:.6e}",
            "recon",
            f"{reconstruction_error:.6e}",
            "aligned_q",
            f"{aligned_error:.6e}",
            flush=True,
        )
        del cpu_matrix, cpu_orthogonal, cpu_q, cpu_r, cpu_power
        del npu_matrix, npu_orthogonal, q, r, power, identity, q_cpu_from_npu, signs

    weighted_npu = sum(count * npu_ms for _, count, npu_ms, _ in results)
    weighted_cpu = sum(count * cpu_ms for _, count, _, cpu_ms in results)
    print(
        "WEIGHTED",
        "calls",
        sum(SHAPE_COUNTS.values()),
        "npu_ms",
        round(weighted_npu, 3),
        "cpu_fp64_ms",
        round(weighted_cpu, 3),
        "speedup",
        round(weighted_cpu / weighted_npu, 3),
        flush=True,
    )


if __name__ == "__main__":
    main()
