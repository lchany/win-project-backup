#!/usr/bin/env python3
"""STEP-326: compare remote NPU 30-step summary vs local GPU baseline log."""
from __future__ import annotations

import json
import posixpath
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_exec import connect, parse_machine_info, redact


GPU = Path(r"C:\project\win-project-backup\DongFeng\gpu去除随机性固定后loss.log")
REL = "diagnostics/step326_torch_qr_shortterm_30step_back8_20260819T233000"


def extract_gpu(path: Path, max_it: int = 30) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Iter [" not in line:
            continue
        m = re.search(r"Iter \[(\d+)/", line)
        if not m:
            continue
        it = int(m.group(1))
        if it > max_it:
            continue
        tm = re.search(r"time: ([0-9.]+)", line)
        lm = re.search(r"loss: ([0-9.eE+-]+|nan|inf)", line, re.I)
        if tm and lm and lm.group(1).lower() not in ("nan", "inf"):
            out[it] = (float(tm.group(1)), float(lm.group(1)))
    return out


def fetch_npu_rows() -> list[dict]:
    info = parse_machine_info()
    remote_dir = posixpath.join(str(info["shared"]).rstrip("/"), REL)
    jump = connect(str(info["jump_host"]), int(info["jump_port"]), str(info["jump_user"]), str(info["jump_password"]))
    target = None
    try:
        channel = jump.get_transport().open_channel(
            "direct-tcpip", (str(info["target_host"]), int(info["target_port"])), ("127.0.0.1", 0)
        )
        target = connect(str(info["target_host"]), int(info["target_port"]), str(info["target_user"]), str(info["target_password"]), sock=channel)
        cmd = f"python3 - <<'PY'\nimport json\nfrom pathlib import Path\np=Path('{remote_dir}/summary.json')\nif not p.is_file():\n    import re\n    text=Path('{remote_dir}/logs/launcher.log').read_text(encoding='utf-8', errors='replace')\n    rows=[]\n    for line in text.splitlines():\n        m=re.search(r'Iter \\\\[(\d+)/(\d+)\\\\].*?time: ([0-9.]+).*?loss: ([0-9.eE+-]+|nan|inf)', line, re.I)\n        if m:\n            rows.append({{'iter':int(m.group(1)),'time_s':float(m.group(3)),'loss':m.group(4).lower()}})\n    p.write_text(json.dumps(rows), encoding='utf-8')\nprint(p.read_text(encoding='utf-8'))\nPY"
        _, stdout, stderr = target.exec_command(cmd, timeout=45)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        if err.strip():
            print(redact(err, info), file=sys.stderr)
        return json.loads(out)
    finally:
        if target is not None:
            target.close()
        jump.close()


def main() -> int:
    gpu = extract_gpu(GPU)
    npu_rows = fetch_npu_rows()
    npu = {
        int(r["iter"]): (float(r["time_s"]), float(r["loss"]) if r["loss"] not in ("nan", "inf") else float("nan"))
        for r in npu_rows
        if r.get("loss") not in ("nan", "inf")
    }
    nan_iters = [int(r["iter"]) for r in npu_rows if r.get("loss") in ("nan", "inf")]

    fail1: list[tuple[int, float]] = []
    fail2: list[tuple[int, float]] = []
    print("iter | NPU loss | GPU loss | diff% | <=1% | <=2%")
    for it in range(1, 31):
        if it not in gpu or it not in npu:
            continue
        _, n_loss = npu[it]
        _, g_loss = gpu[it]
        d = (n_loss - g_loss) / g_loss * 100
        ok1 = abs(d) <= 1.0
        ok2 = abs(d) <= 2.0
        if not ok1:
            fail1.append((it, d))
        if not ok2:
            fail2.append((it, d))
        if it in (1, 6, 10, 11, 12, 20, 21, 22, 30) or not ok2:
            print(f"{it:4d} | {n_loss:8.4f} | {g_loss:8.4f} | {d:+6.2f}% | {'OK' if ok1 else 'FAIL'} | {'OK' if ok2 else 'FAIL'}")

    t_npu = sum(npu[i][0] for i in range(2, 31) if i in npu)
    t_gpu = sum(gpu[i][0] for i in range(2, 31) if i in gpu)
    print(f"\nIter2-30 sum time: NPU={t_npu:.1f}s GPU={t_gpu:.1f}s ratio={t_npu/t_gpu:.2f}x")
    print(f"<=1% pass: {30 - len(fail1)}/30  worst: {max(fail1, key=lambda x: abs(x[1])) if fail1 else 'none'}")
    print(f"<=2% pass: {30 - len(fail2)}/30  worst: {max(fail2, key=lambda x: abs(x[1])) if fail2 else 'none'}")
    soap = {k: npu[k][0] for k in (10, 20, 30) if k in npu}
    print("SOAP step times (NPU):", soap)
    if nan_iters:
        print("NAN iters:", nan_iters)
        return 1
    return 0 if len(fail2) >= 25 else 2


if __name__ == "__main__":
    raise SystemExit(main())
