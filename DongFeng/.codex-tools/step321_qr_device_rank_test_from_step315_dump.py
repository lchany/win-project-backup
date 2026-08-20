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

        py_code = """
import re, json
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import mx_driving_cloud

dump_dir = Path(r'__REMOTE_DIR__/qr_tensors')
files = sorted(dump_dir.glob('*.pt'))
print('total_pt_files', len(files))

rx = re.compile(r'^rank(?P<rank>\\d+)_.*?_qr\\d+_(?P<n>\\d+)x\\d+_(?P<note>.+)\\.pt$')

targets = []
ns = [192, 220, 256]
for rank in range(0, 8):
    for n in ns:
        targets.append((rank, n))

picked = {}  # (rank,n) -> Path
for p in files:
    m = rx.match(p.name)
    if not m:
        continue
    rank = int(m.group('rank'))
    n = int(m.group('n'))
    if (rank, n) in picked:
        continue
    if (rank, n) in picked:
        continue
    if n in ns:
        picked[(rank, n)] = p

missing = [k for k in targets if k not in picked]
print('picked_pairs', len(picked), 'missing_pairs', len(missing))
for k in missing:
    print('missing', k)

summary_path = dump_dir / 'qr_device_rank_test_summary.jsonl'
summary_path.parent.mkdir(parents=True, exist_ok=True)

def is_finite(x):
    return bool(torch.isfinite(x).all().item())

def to_tensor(obj):
    if isinstance(obj, dict) and 'A' in obj:
        return obj['A']
    return obj

with open(summary_path, 'a', encoding='utf-8') as f:
    for (rank, n) in targets:
        if (rank, n) not in picked:
            continue
        pt = picked[(rank, n)]
        torch.npu.set_device(rank)
        A_cpu = torch.load(pt, map_location='cpu', weights_only=False)
        A_cpu = to_tensor(A_cpu)
        if not isinstance(A_cpu, torch.Tensor):
            raise RuntimeError(f'unexpected dump type: {type(A_cpu)} for {pt}')
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
            'rank': rank,
            'n': n,
            'pt': pt.name,
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

