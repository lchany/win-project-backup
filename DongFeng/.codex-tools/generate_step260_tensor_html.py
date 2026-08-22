#!/usr/bin/env python3
"""Generate a self-contained HTML report for the STEP-260 QrV2 tensor dumps."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


EXPECTED_KEYS = {"A", "Q", "R", "meta", "note"}
EXPECTED_SHAPE = (192, 192)
EXPECTED_DTYPE = torch.float32
EXPECTED_META = {
    "opt_step": 10,
    "factor_ind": 0,
    "shape": [192, 192],
    "dtype": "float32",
    "A_finite": True,
    "Q_finite": False,
    "R_finite": False,
    "recon_max": None,
}
PACKAGE_SOURCE = (
    "cann-driving-cloud-ops-26.0.7-a3-cann8.3.rc1-torch2.7.1-py3.11-aarch64.zip"
    "!/mx_driving_cloud-26.0.7+CANN8.3.RC1.A3-cp311-cp311-linux_aarch64.whl"
    "!/mx_driving_cloud/packages/vendors/customize/op_impl/ai_core/tbe/"
    "customize_impl/dynamic/qr_v2.cpp"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def scalar_text(value: float) -> str:
    if value != value:
        return "nan"
    if value == float("inf"):
        return "+inf"
    if value == float("-inf"):
        return "-inf"
    return f"{value:.9e}"


def matrix_text(tensor: torch.Tensor) -> str:
    rows = tensor.detach().cpu().tolist()
    return "\n".join(
        f"row {row_index:03d}: " + " ".join(scalar_text(float(value)) for value in row)
        for row_index, row in enumerate(rows)
    )


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu()
    finite = torch.isfinite(value)
    nan = torch.isnan(value)
    positive_inf = torch.isposinf(value)
    negative_inf = torch.isneginf(value)
    bad_coordinates = (~finite).nonzero(as_tuple=False)
    bad_rows = bad_coordinates[:, 0].unique().tolist() if bad_coordinates.numel() else []
    bad_columns = bad_coordinates[:, 1].unique().tolist() if bad_coordinates.numel() else []
    full_bad_columns = [
        column
        for column in range(value.shape[1])
        if bool((~finite[:, column]).all().item())
    ]
    finite_values = value[finite]
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": value.numel(),
        "finite": int(finite.sum().item()),
        "nan": int(nan.sum().item()),
        "positive_inf": int(positive_inf.sum().item()),
        "negative_inf": int(negative_inf.sum().item()),
        "finite_min": None if finite_values.numel() == 0 else float(finite_values.min().item()),
        "finite_max": None if finite_values.numel() == 0 else float(finite_values.max().item()),
        "bad_row_range": None if not bad_rows else [min(bad_rows), max(bad_rows)],
        "bad_column_range": None if not bad_columns else [min(bad_columns), max(bad_columns)],
        "full_bad_columns": full_bad_columns,
        "content_sha256": tensor_sha256(value),
    }


def cpu_qr_metrics(input_a: torch.Tensor, dtype: torch.dtype) -> dict[str, Any]:
    value = input_a.to(dtype=dtype)
    q_value, r_value = torch.linalg.qr(value)
    reconstruction = q_value @ r_value
    identity = torch.eye(q_value.shape[1], dtype=dtype)
    difference = reconstruction - value
    denominator = max(float(torch.linalg.vector_norm(value).item()), torch.finfo(dtype).tiny)
    metrics = {
        "dtype": str(dtype),
        "q_finite": bool(torch.isfinite(q_value).all().item()),
        "r_finite": bool(torch.isfinite(r_value).all().item()),
        "recon_max_abs": float(difference.abs().max().item()),
        "recon_relative_l2": float(torch.linalg.vector_norm(difference).item()) / denominator,
        "orthogonality_max_abs": float(
            (q_value.transpose(-2, -1) @ q_value - identity).abs().max().item()
        ),
        "r_lower_max_abs": float(torch.tril(r_value, diagonal=-1).abs().max().item()),
    }
    if not metrics["q_finite"] or not metrics["r_finite"]:
        raise RuntimeError(f"CPU QR oracle produced a non-finite result for {dtype}")
    return metrics


def format_range(value: list[int] | None) -> str:
    return "—" if value is None else f"{value[0]}–{value[1]}"


def format_columns(columns: list[int]) -> str:
    if not columns:
        return "—"
    contiguous = columns == list(range(columns[0], columns[-1] + 1))
    return f"{columns[0]}–{columns[-1]} ({len(columns)} 列)" if contiguous else ", ".join(map(str, columns))


def stat_table(stats: dict[str, dict[str, Any]]) -> str:
    rows = []
    for name, item in stats.items():
        rows.append(
            "<tr>"
            f"<th>{html.escape(name)}</th>"
            f"<td>{item['finite']:,}/{item['numel']:,}</td>"
            f"<td>{item['nan']:,}</td>"
            f"<td>{item['positive_inf']:,}</td>"
            f"<td>{item['negative_inf']:,}</td>"
            f"<td>{format_range(item['bad_row_range'])}</td>"
            f"<td>{format_range(item['bad_column_range'])}</td>"
            f"<td>{html.escape(format_columns(item['full_bad_columns']))}</td>"
            "</tr>"
        )
    return "".join(rows)


def matrix_details(
    label: str,
    provenance: str,
    tensor: torch.Tensor,
    stats: dict[str, Any],
    *,
    open_by_default: bool = False,
) -> str:
    opened = " open" if open_by_default else ""
    summary = (
        f"{label} — finite {stats['finite']:,}/{stats['numel']:,}, "
        f"NaN {stats['nan']:,}, SHA256 {stats['content_sha256'][:16]}…"
    )
    return (
        f"<details class=\"tensor\"{opened}>"
        f"<summary>{html.escape(summary)}</summary>"
        f"<p class=\"provenance\">{html.escape(provenance)}</p>"
        f"<pre>{html.escape(matrix_text(tensor))}</pre>"
        "</details>"
    )


def load_rank(path: Path, expected_rank: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != EXPECTED_KEYS:
        raise RuntimeError(f"unexpected payload keys in {path}: {sorted(payload) if isinstance(payload, dict) else type(payload)}")
    for name in ("A", "Q", "R"):
        tensor = payload[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{path}:{name} is not a tensor")
        if tuple(tensor.shape) != EXPECTED_SHAPE or tensor.dtype != EXPECTED_DTYPE:
            raise RuntimeError(f"{path}:{name} contract mismatch: {tuple(tensor.shape)} {tensor.dtype}")
    meta = payload["meta"]
    if not isinstance(meta, dict) or int(meta.get("rank", -1)) != expected_rank:
        raise RuntimeError(f"rank metadata mismatch in {path}")
    for key, expected in EXPECTED_META.items():
        if meta.get(key) != expected:
            raise RuntimeError(f"metadata contract mismatch in {path}: {key}={meta.get(key)!r}")
    actual = payload["Q"] @ payload["R"]
    difference = actual - payload["A"]
    tensors = {
        "A (captured input / expected)": payload["A"],
        "Q (captured output)": payload["Q"],
        "R (captured output)": payload["R"],
        "Q@R (offline actual)": actual,
        "Q@R - A (offline diff)": difference,
    }
    stats = {name: tensor_stats(tensor) for name, tensor in tensors.items()}
    expected_q_bad = torch.zeros(EXPECTED_SHAPE, dtype=torch.bool)
    expected_q_bad[:, 128:192] = True
    expected_r_bad = torch.zeros(EXPECTED_SHAPE, dtype=torch.bool)
    expected_r_bad[:, 128:192] = torch.triu(
        torch.ones(EXPECTED_SHAPE, dtype=torch.bool)
    )[:, 128:192]
    if not bool(torch.isfinite(payload["A"]).all().item()):
        raise RuntimeError(f"captured A is not fully finite in {path}")
    if not torch.equal(torch.isnan(payload["Q"]), expected_q_bad):
        raise RuntimeError(f"captured Q NaN mask differs from the audited last-64-column pattern in {path}")
    if not torch.equal(torch.isnan(payload["R"]), expected_r_bad):
        raise RuntimeError(f"captured R NaN mask differs from the audited upper-triangular pattern in {path}")
    if bool(torch.isinf(payload["Q"]).any().item()) or bool(torch.isinf(payload["R"]).any().item()):
        raise RuntimeError(f"captured Q/R contains Inf rather than the audited NaN-only pattern in {path}")
    if not bool(torch.isnan(actual).all().item()) or not bool(torch.isnan(difference).all().item()):
        raise RuntimeError(f"offline reconstruction is not the audited all-NaN pattern in {path}")
    return {
        "path": path,
        "file_sha256": file_sha256(path),
        "file_size": path.stat().st_size,
        "meta": meta,
        "note": payload["note"],
        "tensors": tensors,
        "stats": stats,
        "cpu_oracles": {
            "CPU FP32": cpu_qr_metrics(payload["A"], torch.float32),
            "CPU FP64": cpu_qr_metrics(payload["A"], torch.float64),
        },
    }


def build_html(records: list[dict[str, Any]], generated_at: str) -> str:
    summary_rows = []
    rank_sections = []
    for record in records:
        rank = int(record["meta"]["rank"])
        stats = record["stats"]
        summary_rows.append(
            "<tr>"
            f"<td><a href=\"#rank-{rank}\">rank {rank}</a></td>"
            f"<td>{int(record['meta']['opt_step'])}</td>"
            f"<td>{int(record['meta']['factor_ind'])}</td>"
            f"<td>{stats['A (captured input / expected)']['finite']:,}/{stats['A (captured input / expected)']['numel']:,}</td>"
            f"<td>{stats['Q (captured output)']['nan']:,}</td>"
            f"<td>{format_range(stats['Q (captured output)']['bad_column_range'])}</td>"
            f"<td>{stats['R (captured output)']['nan']:,}</td>"
            f"<td>{format_range(stats['R (captured output)']['bad_column_range'])}</td>"
            f"<td>{stats['Q@R (offline actual)']['nan']:,}</td>"
            "</tr>"
        )
        meta_json = json.dumps(record["meta"], ensure_ascii=False, indent=2, sort_keys=True)
        cpu_rows = "".join(
            "<tr>"
            f"<th>{html.escape(label)}</th>"
            f"<td>{'PASS' if metrics['q_finite'] else 'FAIL'}</td>"
            f"<td>{'PASS' if metrics['r_finite'] else 'FAIL'}</td>"
            f"<td>{metrics['recon_max_abs']:.9e}</td>"
            f"<td>{metrics['recon_relative_l2']:.9e}</td>"
            f"<td>{metrics['orthogonality_max_abs']:.9e}</td>"
            f"<td>{metrics['r_lower_max_abs']:.9e}</td>"
            "</tr>"
            for label, metrics in record["cpu_oracles"].items()
        )
        matrices = "".join(
            matrix_details(
                name,
                (
                    "训练现场抓取。" if name in {"A (captured input / expected)", "Q (captured output)", "R (captured output)"}
                    else "使用本页现场抓取的张量在 CPU 上离线计算，不是额外的训练现场 dump。"
                ),
                record["tensors"][name],
                record["stats"][name],
            )
            for name in record["tensors"]
        )
        rank_sections.append(
            f"<section id=\"rank-{rank}\" class=\"rank-section\">"
            f"<h2>Rank {rank}</h2>"
            f"<p><code>{html.escape(record['path'].name)}</code> · {record['file_size']:,} bytes · "
            f"file SHA256 <code>{record['file_sha256']}</code></p>"
            "<table><thead><tr><th>张量</th><th>有限/总数</th><th>NaN</th><th>+Inf</th><th>-Inf</th>"
            "<th>异常行范围</th><th>异常列范围</th><th>整列异常</th></tr></thead>"
            f"<tbody>{stat_table(record['stats'])}</tbody></table>"
            "<h3>CPU 离线 QR 对照</h3>"
            "<p class=\"provenance\">对同一份现场 A 离线调用 <code>torch.linalg.qr</code>；不是训练现场额外 dump。</p>"
            "<table><thead><tr><th>路径</th><th>Q finite</th><th>R finite</th><th>recon max abs</th>"
            "<th>recon relative L2</th><th>orthogonality max abs</th><th>R lower max abs</th></tr></thead>"
            f"<tbody>{cpu_rows}</tbody></table>"
            "<details><summary>Meta 与原始 note</summary>"
            f"<pre>{html.escape(meta_json)}\n\nnote: {html.escape(str(record['note']))}</pre></details>"
            f"{matrices}</section>"
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>STEP-260 QrV2 训练现场 Tensor 报告</title>
<style>
:root {{ color-scheme: light dark; --accent:#1769aa; --bad:#b42318; --ok:#067647; --panel:#f5f7fa; }}
@media (prefers-color-scheme:dark) {{ :root {{ --panel:#171b22; --accent:#79b8ff; --bad:#ff7b72; --ok:#56d364; }} }}
body {{ font-family: system-ui,-apple-system,"Segoe UI",sans-serif; max-width:1500px; margin:0 auto; padding:24px; line-height:1.55; }}
h1,h2 {{ line-height:1.2; }}
.lead {{ font-size:1.08rem; }}
.notice {{ border-left:4px solid var(--accent); background:var(--panel); padding:12px 16px; margin:18px 0; }}
.bad {{ color:var(--bad); font-weight:700; }} .ok {{ color:var(--ok); font-weight:700; }}
.toolbar {{ position:sticky; top:0; z-index:2; background:Canvas; padding:10px 0; border-bottom:1px solid GrayText; }}
button {{ margin-right:8px; padding:7px 12px; cursor:pointer; }}
table {{ border-collapse:collapse; width:100%; margin:12px 0 20px; font-variant-numeric:tabular-nums; }}
th,td {{ border:1px solid GrayText; padding:6px 8px; text-align:right; }}
th:first-child,td:first-child {{ text-align:left; }}
details {{ margin:9px 0; border:1px solid GrayText; border-radius:6px; padding:8px 10px; }}
summary {{ cursor:pointer; font-weight:650; }}
pre {{ overflow:auto; max-height:68vh; background:var(--panel); padding:12px; border-radius:5px; font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace; white-space:pre; }}
code {{ overflow-wrap:anywhere; }}
.rank-section {{ margin-top:42px; padding-top:10px; border-top:3px solid var(--accent); }}
.provenance {{ color:GrayText; }}
</style>
</head>
<body>
<h1>STEP-260 QrV2 训练现场 Tensor 报告</h1>
<p class="lead">8 个 rank 的现场输入 A 全部有限；Q 的第 128–191 列全部为 NaN，R 的异常也集中在最后 64 列。</p>
<div class="notice"><strong>证据边界：</strong>A/Q/R 来自训练现场 `.pt` dump。<code>Q@R</code> 和 <code>Q@R-A</code> 是使用已抓取张量在 CPU 上离线计算的派生结果。报告未对非有限值做修复或过滤。</div>
<p><strong>对应算子源码的 ZIP 内路径：</strong><br><code>{html.escape(PACKAGE_SOURCE)}</code></p>
<p>生成时间：<code>{html.escape(generated_at)}</code>；加载方式：<code>torch.load(..., map_location='cpu', weights_only=True)</code>。</p>
<div class="toolbar"><button onclick="setAll(true)">展开全部</button><button onclick="setAll(false)">折叠全部</button></div>
<h2>8-rank 汇总</h2>
<table><thead><tr><th>Rank</th><th>opt step</th><th>factor</th><th>A 有限</th><th>Q NaN</th><th>Q 异常列</th><th>R NaN</th><th>R 异常列</th><th>Q@R NaN</th></tr></thead>
<tbody>{''.join(summary_rows)}</tbody></table>
{''.join(rank_sections)}
<script>function setAll(value){{document.querySelectorAll('details').forEach(x=>x.open=value);}}</script>
</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("step260_qr_bad_tensors"))
    parser.add_argument("--output", type=Path, default=Path("step260_qrv2_tensor_dump_report.html"))
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("rank*_step10_ind0_192x192_BAD.pt"))
    expected_names = [f"rank{rank}_step10_ind0_192x192_BAD.pt" for rank in range(8)]
    if [path.name for path in paths] != expected_names:
        raise RuntimeError(f"expected exact 8-rank input inventory, got {[path.name for path in paths]}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {args.output}")
    records = [load_rank(path, rank) for rank, path in enumerate(paths)]
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    output = build_html(records, generated_at)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite existing temporary report: {temporary}")
    temporary.write_text(output, encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "sha256": file_sha256(args.output),
        "rank_count": len(records),
        "embedded_matrices_per_rank": 5,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
