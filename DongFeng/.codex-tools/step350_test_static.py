#!/usr/bin/env python3
"""Static and negative tests for the STEP-350 diagnostic source patch."""

from __future__ import annotations

from pathlib import Path

from step350_patch_qr_v2_context_capture import build_candidate
from step350_decode_qrv2_context import decode

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".codex-tools" / "qr_v2.cpp"


def expect_failure(source: bytes) -> None:
    try:
        build_candidate(source)
    except RuntimeError:
        return
    raise AssertionError("modified source unexpectedly passed the SHA gate")


def main() -> int:
    source = SOURCE.read_bytes()
    candidate = build_candidate(source)
    source_text = source.decode("utf-8")
    text = candidate.decode("utf-8")

    assert text.count("FreeTensor<DTYPE_A>") == source_text.count("FreeTensor<DTYPE_A>") == 13
    assert text.count("CaptureBeforeFree(k, tilingInfo);") == 1
    assert text.count("CaptureAfterFree(k, tilingInfo);") == 1
    assert text.count("FlushDiagnosticTail(k, tilingInfo);") == 1
    before = text.index("CaptureBeforeFree(k, tilingInfo);")
    assert before < text.index("tTQue.FreeTensor<DTYPE_A>(tLocal);", before)
    after = text.index("CaptureAfterFree(k, tilingInfo);")
    assert after < text.index("CalcQForLARFB(false);", after)
    assert "CalcQForLARFB(false) still consumes" not in text
    assert "Release them only after" not in text

    # Aligned stable lower-triangle capacity and final-block capacity.
    stable = sum((min(row, 128) // 8) * 8 for row in range(1, 192))
    final = sum(((row - 128) // 8) * 8 for row in range(129, 192))
    assert stable == 15_872
    assert final == 1_792
    assert 4 * 4096 - stable == 512
    assert 512 + 32 <= final

    raw_r = torch.zeros((192, 192), dtype=torch.float32)
    payload = torch.arange(16_384, dtype=torch.float32)
    stable_values = payload[:15_872]
    cursor = 0
    for row in range(1, 192):
        length = (min(row, 128) // 8) * 8
        raw_r[row, :length] = stable_values[cursor : cursor + length]
        cursor += length
    tail_and_header = torch.zeros(544, dtype=torch.float32)
    tail_and_header[:512] = payload[15_872:]
    tail_and_header[512:528] = torch.tensor(
        [350350, 1, 2, 0, 0, 3, 64, 4096, 4096, 4096, 4096, 1, 2, 3, 15872, 512],
        dtype=torch.float32,
    )
    cursor = 0
    for row in range(129, 192):
        length = ((row - 128) // 8) * 8
        take = min(length, 544 - cursor)
        if take:
            raw_r[row, 128 : 128 + take] = tail_and_header[cursor : cursor + take]
            cursor += take
        if cursor == 544:
            break
    decoded = decode(raw_r)
    assert torch.equal(decoded["t_before_free"].reshape(-1), payload[:4096])
    assert torch.equal(decoded["v_after_free"].reshape(-1), payload[12288:])
    bad = raw_r.clone()
    bad[164, 128] = 0
    try:
        decode(bad)
    except RuntimeError:
        pass
    else:
        raise AssertionError("bad diagnostic magic unexpectedly passed")

    expect_failure(source + b"\n")
    print("step350_static_tests=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
