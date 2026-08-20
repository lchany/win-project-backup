#!/usr/bin/env python3
from __future__ import annotations

import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact


REL = "diagnostics/step315_qr_dump_monkeypatch_iter6_back8_20260819T222000"


def main() -> int:
    info = parse_machine_info()
    remote_dir = posixpath.join(str(info["shared"]).rstrip("/"), REL)

    jump = connect(
        str(info["jump_host"]),
        int(info["jump_port"]),
        str(info["jump_user"]),
        str(info["jump_password"]),
    )
    target = None
    try:
        channel = jump.get_transport().open_channel(
            "direct-tcpip",
            (str(info["target_host"]), int(info["target_port"])),
            ("127.0.0.1", 0),
        )
        target = connect(
            str(info["target_host"]),
            int(info["target_port"]),
            str(info["target_user"]),
            str(info["target_password"]),
            sock=channel,
        )

        # Keep the python snippet inside the container (so it has the right torch_npu + mx_driving_cloud environment).
        py_code = """
import os, re, json
from pathlib import Path
import torch
import torch_npu  # noqa: F401
import mx_driving_cloud

dump_dir = Path(r'__REMOTE_DIR__/qr_tensors')
summary_path = dump_dir / 'qr_single_test_summary.jsonl'
summary_path.parent.mkdir(parents=True, exist_ok=True)

files = sorted(dump_dir.glob('*.pt'))
print('total_pt_files', len(files))

rx = re.compile(r'^rank(?P<rank>\\d+)_.*?_qr\\d+_(?P<n>\\d+)x\\d+_(?P<note>.+)\\.pt$')

def parse_meta(p: Path):
    m = rx.match(p.name)
    if not m:
        return None
    return int(m.group('rank')), int(m.group('n')), m.group('note')

selected = []
nonfinite = []
for p in files:
    meta = parse_meta(p)
    if not meta:
        continue
    rank, n, note = meta
    if 'nonfinite' in note:
        nonfinite.append((p, rank, n, note))

max_files = int(os.environ.get('QR_SINGLE_TEST_MAX_FILES', '12'))
if len(nonfinite) >= max_files:
    nonfinite_sorted = sorted(nonfinite, key=lambda x: x[0].name)
    selected = nonfinite_sorted[:max_files]
else:
    selected = nonfinite
    if len(selected) < max_files:
        for p in files:
            meta = parse_meta(p)
            if not meta:
                continue
            rank, n, note = meta
            if (p, rank, n, note) in selected:
                continue
            selected.append((p, rank, n, note))
            if len(selected) >= max_files:
                break

print('selected_files', len(selected))

def to_tensor(obj):
    if isinstance(obj, dict) and 'A' in obj:
        return obj['A']
    return obj

def is_finite(x):
    return bool(torch.isfinite(x).all().item())

with open(summary_path, 'a', encoding='utf-8') as f:
    for idx, (pt, rank, n, note) in enumerate(selected):
        torch.npu.set_device(rank)
        A_cpu = torch.load(pt, map_location='cpu', weights_only=False)
        A_cpu = to_tensor(A_cpu)
        if not isinstance(A_cpu, torch.Tensor):
            raise RuntimeError('unexpected dump type: %s for %s' % (type(A_cpu), pt))
        if A_cpu.dtype != torch.float32:
            A_cpu = A_cpu.float()
        A_cpu = A_cpu.contiguous()
        A = A_cpu.to('npu:%d' % rank)
        torch.npu.synchronize()

        Q, R = mx_driving_cloud.linalg.qr(A)
        torch.npu.synchronize()

        qf = is_finite(Q)
        rf = is_finite(R)
        rec_max = None
        if qf and rf:
            try:
                rec_max = float((Q @ R - A).abs().max().to('cpu').item())
            except Exception:
                rec_max = None

        rec = {
            'idx': idx,
            'pt': pt.name,
            'rank': rank,
            'n': n,
            'note': note,
            'Q_finite': qf,
            'R_finite': rf,
            'recon_max': rec_max,
        }
        f.write(json.dumps(rec, ensure_ascii=False) + '\\n')
        print(rec)

print('summary_written', str(summary_path))
"""

        py_code = py_code.replace("__REMOTE_DIR__", remote_dir)

        command = (
            'docker exec mapqr-leicheng bash --noprofile --norc -lc '
            + "\"python3 - <<'PY'\n"
            + py_code
            + "\nPY\""
        )

        _, stdout, stderr = target.exec_command(command, timeout=600)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        text = redact(out + err, info)
        print(text, end="" if text.endswith("\n") else "\n")
        return stdout.channel.recv_exit_status()
    finally:
        if target is not None:
            target.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())

