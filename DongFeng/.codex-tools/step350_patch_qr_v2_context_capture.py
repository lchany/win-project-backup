#!/usr/bin/env python3
"""Build a fail-closed QrV2 source that exports k=2/core0 T/V snapshots.

The diagnostic payload is written into 32-byte-aligned slots of R's strict
lower triangle.  The mathematical Q and R diagonal/upper triangle are left
unchanged.  This patch deliberately preserves the original FreeTensor order.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_SOURCE_SHA256 = "2dbaf1e1b5383563c23cdac7a5151b14605f8585b6e48fd8c58065fb5c1206c9"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, got {count}")
    return text.replace(old, new, 1)


def build_candidate(source: bytes) -> bytes:
    digest = hashlib.sha256(source).hexdigest()
    if digest != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"source SHA-256 mismatch: {digest}")
    text = source.decode("utf-8")

    # Compatibility only: current CANN presents GET_TILING_DATA as const.
    text = replace_once(
        text,
        """    __aicore__ inline void Init(
        GM_ADDR a, GM_ADDR q, GM_ADDR r, GM_ADDR workspace, QrV2TilingData *tilingData, TPipe *pipe)
""",
        """    __aicore__ inline void Init(
        GM_ADDR a, GM_ADDR q, GM_ADDR r, GM_ADDR workspace, const QrV2TilingData *tilingData, TPipe *pipe)
""",
        "Init const tiling",
    )

    helper = r'''
    // STEP-350 diagnostic layout (float32 slots):
    //   0..4095      T before FreeTensor
    //   4096..8191   V before FreeTensor
    //   8192..12287  T read by CalcQ after FreeTensor
    //   12288..16383 V read by CalcQ after FreeTensor
    // The first 15872 slots use aligned lower-triangle rows with col < 128.
    // The final 512 V values plus a 32-float header are staged in ASubBuf and
    // flushed into the final 64x64 diagonal block after its normal R write.
    __aicore__ inline bool IsContextCaptureTarget(uint32_t k)
    {
        return this->m == 192 && this->n == 192 && this->blockSize == 64 &&
               this->blockp == 3 && k == 2 && this->coreId == 0;
    }

    __aicore__ inline void WaitDiagnosticMte3()
    {
        int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
    }

    __aicore__ inline void CopyToStableLowerR(
        const LocalTensor<DTYPE_A>& src, uint32_t payloadOffset, uint32_t count)
    {
        uint32_t payloadEnd = payloadOffset + count;
        uint32_t rowBase = 0;
        for (uint32_t row = 1; row < 192; ++row) {
            uint32_t rowLength = row < 128 ? row : 128;
            rowLength = (rowLength / 8) * 8;
            uint32_t rowEnd = rowBase + rowLength;
            uint32_t begin = payloadOffset > rowBase ? payloadOffset : rowBase;
            uint32_t end = payloadEnd < rowEnd ? payloadEnd : rowEnd;
            if (begin < end) {
                uint32_t srcOffset = begin - payloadOffset;
                uint32_t dstColumn = begin - rowBase;
                DataCopy(this->rGm[row * this->n + dstColumn], src[srcOffset], end - begin);
            }
            rowBase = rowEnd;
        }
    }

    __aicore__ inline void CaptureBeforeFree(uint32_t k, const TaskTilingInfo& tilingInfo)
    {
        if (!IsContextCaptureTarget(k) || tilingInfo.useCoreNum != 0) {
            return;
        }
        CopyToStableLowerR(this->tLocal, 0, 4096);
        CopyToStableLowerR(this->vLocal, 4096, 4096);
        WaitDiagnosticMte3();
    }

    __aicore__ inline void CaptureAfterFree(uint32_t k, const TaskTilingInfo& tilingInfo)
    {
        if (!IsContextCaptureTarget(k) || tilingInfo.useCoreNum != 0) {
            return;
        }
        CopyToStableLowerR(this->tLocal, 8192, 4096);
        CopyToStableLowerR(this->vLocal, 12288, 3584);
        WaitDiagnosticMte3();

        LocalTensor<DTYPE_A> staging = this->ASubBuf.Get<DTYPE_A>();
        DataCopy(staging, this->vLocal[3584], 512);
        int32_t mte2vEvent = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(mte2vEvent);
        WaitFlag<HardEvent::MTE2_V>(mte2vEvent);
        // Integer-valued float32 metadata is exactly representable.
        staging.SetValue(512, static_cast<DTYPE_A>(350350)); // magic
        staging.SetValue(513, static_cast<DTYPE_A>(1));      // schema
        staging.SetValue(514, static_cast<DTYPE_A>(2)); // target k
        staging.SetValue(515, static_cast<DTYPE_A>(0)); // target coreId
        staging.SetValue(516, static_cast<DTYPE_A>(0)); // target useCoreNum
        staging.SetValue(517, static_cast<DTYPE_A>(3)); // target blockp
        staging.SetValue(518, static_cast<DTYPE_A>(64)); // target blockSize
        staging.SetValue(519, static_cast<DTYPE_A>(4096));
        staging.SetValue(520, static_cast<DTYPE_A>(4096));
        staging.SetValue(521, static_cast<DTYPE_A>(4096));
        staging.SetValue(522, static_cast<DTYPE_A>(4096));
        staging.SetValue(523, static_cast<DTYPE_A>(1)); // before-free complete
        staging.SetValue(524, static_cast<DTYPE_A>(2)); // free completed in source order
        staging.SetValue(525, static_cast<DTYPE_A>(3)); // after-free capture complete
        staging.SetValue(526, static_cast<DTYPE_A>(15872));
        staging.SetValue(527, static_cast<DTYPE_A>(512));
        for (uint32_t index = 528; index < 544; ++index) {
            staging.SetValue(index, static_cast<DTYPE_A>(0));
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FlushDiagnosticTail(uint32_t k, const TaskTilingInfo& tilingInfo)
    {
        if (!IsContextCaptureTarget(k) || tilingInfo.useCoreNum != 0) {
            return;
        }
        LocalTensor<DTYPE_A> staging = this->ASubBuf.Get<DTYPE_A>();
        uint32_t srcOffset = 0;
        uint32_t remaining = 544;
        for (uint32_t row = 129; row < 192 && remaining > 0; ++row) {
            uint32_t rowLength = ((row - 128) / 8) * 8;
            uint32_t copyLength = remaining < rowLength ? remaining : rowLength;
            if (copyLength > 0) {
                DataCopy(this->rGm[row * this->n + 128], staging[srcOffset], copyLength);
                srcOffset += copyLength;
                remaining -= copyLength;
            }
        }
        WaitDiagnosticMte3();
    }

'''
    text = replace_once(
        text,
        """    __aicore__ inline void Process()
""",
        helper + """    __aicore__ inline void Process()
""",
        "diagnostic helpers",
    )

    text = replace_once(
        text,
        """            LARFB(k, tilingInfo);
            if (coreId == 0) {
                CalcQForLARFB(false);
""",
        """            LARFB(k, tilingInfo);
            CaptureAfterFree(k, tilingInfo);
            if (coreId == 0) {
                CalcQForLARFB(false);
""",
        "post-free capture point",
    )
    text = replace_once(
        text,
        """            akkTQue.FreeTensor<DTYPE_A>(akkLocal);
        }
    }
""",
        """            akkTQue.FreeTensor<DTYPE_A>(akkLocal);
            FlushDiagnosticTail(k, tilingInfo);
        }
    }
""",
        "tail flush point",
    )
    text = replace_once(
        text,
        """            aLocal = aTQue.DeQue<DTYPE_A>();
            tLocal = tTQue.DeQue<DTYPE_A>();
            vLocal = vTQue.DeQue<DTYPE_A>();
            tTQue.FreeTensor<DTYPE_A>(tLocal);
""",
        """            aLocal = aTQue.DeQue<DTYPE_A>();
            tLocal = tTQue.DeQue<DTYPE_A>();
            vLocal = vTQue.DeQue<DTYPE_A>();
            CaptureBeforeFree(k, tilingInfo);
            tTQue.FreeTensor<DTYPE_A>(tLocal);
""",
        "pre-free capture point",
    )

    text = replace_once(
        text,
        """    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    TPipe pipe;
""",
        """    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    auto vtvTilingData = tilingData.vtvTilingData;
    auto qaTilingData = tilingData.qaTilingData;
    auto blockQTilingData = tilingData.blockQTilingData;
    auto currentQTilingData = tilingData.currentQTilingData;
    TPipe pipe;
""",
        "entry tiling copies",
    )
    text = replace_once(
        text,
        """    op.vtvMatmulObj.Init(&tilingData.vtvTilingData);
    op.qaMatmulObj.Init(&(tilingData.qaTilingData));
    op.blockQMatmulObj.Init(&(tilingData.blockQTilingData));
    op.currentQMatmulObj.Init(&(tilingData.currentQTilingData));
""",
        """    op.vtvMatmulObj.Init(&vtvTilingData);
    op.qaMatmulObj.Init(&qaTilingData);
    op.blockQMatmulObj.Init(&blockQTilingData);
    op.currentQMatmulObj.Init(&currentQTilingData);
""",
        "Matmul tiling copies",
    )

    candidate = text.encode("utf-8")
    if candidate == source:
        raise RuntimeError("candidate is identical to source")
    # The diagnostic build must preserve all six original FreeTensor sites.
    if candidate.count(b"FreeTensor<DTYPE_A>") != source.count(b"FreeTensor<DTYPE_A>"):
        raise RuntimeError("FreeTensor call count changed")
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    candidate = build_candidate(args.source.read_bytes())
    print("candidate_sha256=" + hashlib.sha256(candidate).hexdigest())
    print("candidate_size=" + str(len(candidate)))
    if args.check:
        if args.output is not None:
            raise ValueError("output must be omitted with --check")
        return 0
    if args.output is None:
        raise ValueError("output is required unless --check is used")
    args.output.write_bytes(candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
