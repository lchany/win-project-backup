#!/usr/bin/env python3
"""Create and verify a fail-closed QrV2 release-candidate source copy.

The source is never opened for writing.  A candidate is emitted only when the
input is the audited vendor source and every approved edit plus the resulting
structure matches the release contract.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Iterable


EXPECTED_SOURCE_SHA256 = "2dbaf1e1b5383563c23cdac7a5151b14605f8585b6e48fd8c58065fb5c1206c9"
EXPECTED_V1_CANDIDATE_SHA256 = "5a4d140b8a473c3a0446d9e225431ff9f8be5e9b9f7355c5a166920e1814105b"
EXPECTED_V2_CANDIDATE_SHA256 = "c4eef5c1984c10953420a9f30b9361473e8f33e2ccf280eefd1d8398c0e199c1"
EXPECTED_V3_CANDIDATE_SHA256 = "fbfda044ef5a15f45a1c48a3818d3d3360aa9c54ff39a36a1cb00e43cc813b99"
EXPECTED_CANDIDATE_SHA256 = "2213dbae5027c179f09c8e3b43be51587ba22e3eb2a7546311048efc3844614b"

V3_MTE3_MTE2_SEQUENCE = """        int32_t eventIDMTE3_MTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3_MTE2);
"""

V3_CALC_Q_SCRATCH_BLOCK = (
    "        DataCopy(workspaceInGm[this->m * this->blockSize], vLocal, "
    "this->blockSize * this->blockSize);\n"
    + V3_MTE3_MTE2_SEQUENCE
    + "        this->vtvMatmulObj.SetTensorA("
    "workspaceInGm[this->m * this->blockSize]);\n"
)
V4_CALC_Q_SCRATCH_BLOCK = (
    "        uint64_t calcQScratchOffset = this->m * this->blockSize + "
    "this->coreId * this->blockElement;\n"
    "        DataCopy(workspaceInGm[calcQScratchOffset], vLocal, "
    "this->blockElement);\n"
    + V3_MTE3_MTE2_SEQUENCE
    + "        this->vtvMatmulObj.SetTensorA(workspaceInGm[calcQScratchOffset]);\n"
)
V4_UPDATE_A_MTE3_MTE2_SEQUENCE = """        int32_t eventIDUpdateAMTE3MTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDUpdateAMTE3MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDUpdateAMTE3MTE2);
"""
V4_PROCESS_SYNC = "            SyncAll();\n"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _find_matching_brace(text: str, opening: int) -> int:
    """Return the closing brace while ignoring braces in comments/strings."""
    if opening >= len(text) or text[opening] != "{":
        raise ValueError("opening does not point to a brace")
    depth = 0
    state = "code"
    index = opening
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if state == "code":
            if char == "/" and following == "/":
                state = "line_comment"
                index += 2
                continue
            if char == "/" and following == "*":
                state = "block_comment"
                index += 2
                continue
            if char == '"':
                state = "string"
            elif char == "'":
                state = "char"
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return index
        elif state == "line_comment":
            if char == "\n":
                state = "code"
        elif state == "block_comment":
            if char == "*" and following == "/":
                state = "code"
                index += 2
                continue
        elif state in {"string", "char"}:
            if char == "\\":
                index += 2
                continue
            if (state == "string" and char == '"') or (state == "char" and char == "'"):
                state = "code"
        index += 1
    raise RuntimeError("unbalanced C++ braces")


def extract_block(text: str, marker: str, *, start: int = 0) -> str:
    marker_count = text.count(marker, start)
    if marker_count != 1:
        raise RuntimeError(f"{marker}: expected exactly one block marker, got {marker_count}")
    marker_index = text.index(marker, start)
    opening = text.find("{", marker_index + len(marker))
    if opening < 0:
        raise RuntimeError(f"{marker}: opening brace not found")
    closing = _find_matching_brace(text, opening)
    return text[marker_index : closing + 1]


def verify_candidate_structure(source: bytes, candidate: bytes) -> dict[str, Any]:
    """Fail unless source and candidate match every approved structural invariant."""
    source_sha = sha256_bytes(source)
    candidate_sha = sha256_bytes(candidate)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"source SHA-256 mismatch: {source_sha}")
    if candidate_sha != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError(f"candidate SHA-256 mismatch: {candidate_sha}")

    original = source.decode("utf-8")
    patched = candidate.decode("utf-8")

    alpha_duplicate_pattern = re.compile(
        r"Duplicate\s*\(\s*alphaLocal\s*,[^;\n]*\);"
    )
    source_alpha_duplicates = alpha_duplicate_pattern.findall(original)
    candidate_alpha_duplicates = alpha_duplicate_pattern.findall(patched)
    expected_alpha_duplicate = (
        "Duplicate(alphaLocal, static_cast<DTYPE_A>(0), ALPHA_BUF_FP32_ELEMENTS);"
    )
    if len(source_alpha_duplicates) != 2:
        raise RuntimeError(
            "source alphaBuf Duplicate contract failed: "
            f"expected=2, actual={len(source_alpha_duplicates)}"
        )
    if candidate_alpha_duplicates != [expected_alpha_duplicate] * 2:
        raise RuntimeError(
            "candidate alphaBuf Duplicate element-count contract failed: "
            f"calls={candidate_alpha_duplicates}"
        )
    if any("2 * UB_ALIGN_SIZE" in call for call in candidate_alpha_duplicates):
        raise RuntimeError("candidate retains the old 64-element alphaBuf Duplicate count")

    alpha_constant_fragments = (
        "constexpr uint32_t ALPHA_BUF_BYTES = 2 * UB_ALIGN_SIZE;",
        "constexpr uint32_t ALPHA_BUF_FP32_ELEMENTS = ALPHA_BUF_BYTES / sizeof(float);",
        'static_assert(ALPHA_BUF_BYTES == 64, "alphaBuf must remain 64 bytes");',
        'static_assert(ALPHA_BUF_FP32_ELEMENTS == 16, "alphaBuf must contain 16 FP32 elements");',
    )
    for fragment in alpha_constant_fragments:
        if original.count(fragment) != 0 or patched.count(fragment) != 1:
            raise RuntimeError(f"alphaBuf constant contract failed: {fragment}")
    if original.count("pipe->InitBuffer(alphaBuf, 2 * UB_ALIGN_SIZE);") != 1:
        raise RuntimeError("source alphaBuf allocation contract failed")
    if patched.count("pipe->InitBuffer(alphaBuf, ALPHA_BUF_BYTES);") != 1:
        raise RuntimeError("candidate alphaBuf must retain one explicit 64-byte allocation")
    if "pipe->InitBuffer(alphaBuf, 2 * UB_ALIGN_SIZE);" in patched:
        raise RuntimeError("candidate retains the implicit alphaBuf allocation expression")

    if patched.count("SyncAll();") != original.count("SyncAll();") + 1:
        raise RuntimeError("v4 must add exactly one cross-core SyncAll call")
    source_base_tiling_refs = re.findall(r"baseTilingInfos\s*\[[^\]]+\]", original)
    candidate_base_tiling_refs = re.findall(r"baseTilingInfos\s*\[[^\]]+\]", patched)
    if source_base_tiling_refs != candidate_base_tiling_refs:
        raise RuntimeError("candidate must not add or change baseTilingInfos slot references")

    calc_q = extract_block(patched, "__aicore__ inline void CalcQForLARFB(")
    update_a = extract_block(patched, "__aicore__ inline void UpdateAForLARFB(")
    workspace_copy = "DataCopy(workspaceInGm[calcQScratchOffset], vLocal, this->blockElement);"
    workspace_matmul_a = "this->vtvMatmulObj.SetTensorA(workspaceInGm[calcQScratchOffset]);"
    scratch_offset = (
        "uint64_t calcQScratchOffset = this->m * this->blockSize + "
        "this->coreId * this->blockElement;"
    )
    if calc_q.count(scratch_offset) != 1:
        raise RuntimeError("CalcQForLARFB must use one per-core scratch offset")
    if "workspaceInGm[this->m * this->blockSize]" in calc_q:
        raise RuntimeError("CalcQForLARFB retains the shared scratch address")
    if calc_q.count(workspace_copy) != 1 or calc_q.count(workspace_matmul_a) != 1:
        raise RuntimeError("CalcQForLARFB DataCopy and SetTensorA must share the local offset")
    if patched.count(V3_MTE3_MTE2_SEQUENCE) != 1:
        raise RuntimeError("v4 must retain exactly one CalcQ MTE3_MTE2 sequence")
    if calc_q.count(V3_MTE3_MTE2_SEQUENCE) != 1:
        raise RuntimeError("v4 CalcQ MTE3_MTE2 sequence must be inside CalcQForLARFB")
    if update_a.count(V4_UPDATE_A_MTE3_MTE2_SEQUENCE) != 1:
        raise RuntimeError("v4 must contain one independent UpdateA MTE3_MTE2 sequence")
    copy_position = calc_q.index(workspace_copy)
    event_position = calc_q.index(V3_MTE3_MTE2_SEQUENCE)
    matmul_position = calc_q.index(workspace_matmul_a)
    if not (copy_position < event_position < matmul_position):
        raise RuntimeError(
            "v4 CalcQ MTE3_MTE2 sequence must order workspace DataCopy before Matmul SetTensorA"
        )

    init = extract_block(patched, "__aicore__ inline void Init(")
    init_task = extract_block(patched, "__aicore__ inline TaskTilingInfo InitTaskTiling(")
    active_core_bound_fragments = (
        "uint32_t colNum = this->blockp - k;",
        "if (colNum <= this->availableCoreNum * 2)",
        "useCoreNum = colNum;",
        "useCoreNum = this->availableCoreNum * 2;",
    )
    if any(fragment not in init_task for fragment in active_core_bound_fragments):
        raise RuntimeError("InitTaskTiling no longer proves useCoreNum <= colNum <= blockp")
    workspace_bound_fragments = (
        "this->blockElement = this->blockSize * this->blockSize;",
        "this->m = this->blockp * this->blockSize;",
        "workspaceInGm.SetGlobalBuffer((__gm__ DTYPE_A *)workspace, "
        "2 * this->m * this->blockSize);",
    )
    if any(fragment not in init for fragment in workspace_bound_fragments):
        raise RuntimeError("Init no longer proves per-core scratch is inside workspaceInGm")

    update_copy = "DataCopy(workspaceInGm[offsetW], aLocal, this->blockElement);"
    update_matmul_b = "qaMatmulObj.SetTensorB(workspaceInGm[offsetW]);"
    update_copy_position = update_a.index(update_copy)
    update_event_position = update_a.index(V4_UPDATE_A_MTE3_MTE2_SEQUENCE)
    update_matmul_position = update_a.index(update_matmul_b)
    if not (update_copy_position < update_event_position < update_matmul_position):
        raise RuntimeError(
            "v4 UpdateA MTE3_MTE2 sequence must order workspace DataCopy before Matmul SetTensorB"
        )

    process = extract_block(patched, "__aicore__ inline void Process(")
    larfb_call = process.index("LARFB(k, tilingInfo);")
    q_calc = process.index("CalcQForLARFB(false);", larfb_call)
    q_copy = process.index("DataCopy(colQGm, this->qLocal", q_calc)
    q_write_wait = process.index("WaitFlag<HardEvent::MTE3_V>(0);", q_copy)
    core0_block = extract_block(process, "if (coreId == 0)", start=larfb_call)
    if "SyncAll();" in core0_block:
        raise RuntimeError("v4 Process SyncAll must be outside the core0-only block")
    process_sync = process.index("SyncAll();", q_write_wait)
    release_positions = [
        process.index("tTQue.FreeTensor<DTYPE_A>(tLocal);", q_write_wait),
        process.index("vTQue.FreeTensor<DTYPE_A>(vLocal);", q_write_wait),
        process.index("aTQue.FreeTensor<DTYPE_A>(aLocal);", q_write_wait),
    ]
    tsqrt_loop = process.index("for (uint32_t i = k + 1; i < blockp; ++i)", q_write_wait)
    if not (
        q_write_wait < process_sync < min(release_positions)
        <= max(release_positions) < tsqrt_loop
    ):
        raise RuntimeError(
            "Process order must be Q writeback -> unconditional SyncAll -> release -> TSQRT"
        )

    v3_equivalent = patched.replace(V4_CALC_Q_SCRATCH_BLOCK, V3_CALC_Q_SCRATCH_BLOCK, 1)
    v3_equivalent = v3_equivalent.replace(V4_UPDATE_A_MTE3_MTE2_SEQUENCE, "", 1)
    v3_equivalent = v3_equivalent.replace(V4_PROCESS_SYNC, "", 1)
    v3_equivalent_sha = sha256_bytes(v3_equivalent.encode("utf-8"))
    if v3_equivalent_sha != EXPECTED_V3_CANDIDATE_SHA256:
        raise RuntimeError(
            "v4 delta exceeds the approved scratch/UpdateA-event/SyncAll scope: "
            f"reverted_sha256={v3_equivalent_sha}"
        )

    v2_equivalent = v3_equivalent.replace(V3_MTE3_MTE2_SEQUENCE, "", 1)
    v2_equivalent_sha = sha256_bytes(v2_equivalent.encode("utf-8"))
    if v2_equivalent_sha != EXPECTED_V2_CANDIDATE_SHA256:
        raise RuntimeError(
            "v3 delta exceeds the approved CalcQForLARFB MTE3_MTE2-only scope: "
            f"reverted_sha256={v2_equivalent_sha}"
        )
    v2_calc_q = extract_block(v2_equivalent, "__aicore__ inline void CalcQForLARFB(")
    event_fragments = (
        "FetchEventID(HardEvent::MTE3_MTE2)",
        "SetFlag<HardEvent::MTE3_MTE2>",
        "WaitFlag<HardEvent::MTE3_MTE2>",
    )
    for fragment in event_fragments:
        if patched.count(fragment) != v2_equivalent.count(fragment) + 2:
            raise RuntimeError(f"v4 must retain v3 and add one UpdateA {fragment} event")
        if calc_q.count(fragment) != v2_calc_q.count(fragment) + 1:
            raise RuntimeError(f"CalcQForLARFB must retain the v3 {fragment} event")
        if update_a.count(fragment) != 1:
            raise RuntimeError(f"UpdateAForLARFB must own one independent {fragment} event")

    alpha_constants_block = "\n".join(alpha_constant_fragments) + "\n"
    v1_equivalent = v2_equivalent.replace(alpha_constants_block, "", 1)
    v1_equivalent = v1_equivalent.replace(
        "pipe->InitBuffer(alphaBuf, ALPHA_BUF_BYTES);",
        "pipe->InitBuffer(alphaBuf, 2 * UB_ALIGN_SIZE);",
        1,
    )
    v1_equivalent = v1_equivalent.replace(
        expected_alpha_duplicate,
        "Duplicate(alphaLocal, static_cast<DTYPE_A>(0), 2 * UB_ALIGN_SIZE);",
    )
    v1_equivalent_sha = sha256_bytes(v1_equivalent.encode("utf-8"))
    if v1_equivalent_sha != EXPECTED_V1_CANDIDATE_SHA256:
        raise RuntimeError(
            "v2 delta exceeds the approved alphaBuf-only scope: "
            f"reverted_sha256={v1_equivalent_sha}"
        )

    original_free_count = len(re.findall(r"\bFreeTensor\s*<", original))
    candidate_free_count = len(re.findall(r"\bFreeTensor\s*<", patched))
    if (original_free_count, candidate_free_count) != (13, 10):
        raise RuntimeError(
            "FreeTensor count contract failed: "
            f"source={original_free_count}, candidate={candidate_free_count}"
        )

    original_larfb = extract_block(original, "__aicore__ inline void LARFB(")
    candidate_larfb = extract_block(patched, "__aicore__ inline void LARFB(")
    if len(re.findall(r"\bFreeTensor\s*<", candidate_larfb)) != 0:
        raise RuntimeError("LARFB must not free GEQRT-owned local tensors")
    for queue in ("aTQue", "tTQue", "vTQue"):
        if candidate_larfb.count(f"{queue}.DeQue<DTYPE_A>()") != 2:
            raise RuntimeError(f"LARFB {queue} DeQue active/inactive contract failed")
    inactive_larfb = extract_block(
        candidate_larfb,
        "if (this->coreId >= tilingInfo.useCoreNum)",
    )
    if "return;" not in inactive_larfb:
        raise RuntimeError("inactive LARFB cores must return before CalcQForLARFB")
    inactive_end = candidate_larfb.index(inactive_larfb) + len(inactive_larfb)
    if candidate_larfb.index("CalcQForLARFB(true);") < inactive_end:
        raise RuntimeError("CalcQForLARFB must execute only after the inactive-core return")
    if "CalcQForLARFB(false);" not in core0_block:
        raise RuntimeError("Process CalcQForLARFB(false) must remain core0-only")

    original_ssrfb = extract_block(original, "__aicore__ inline void SSRFB(")
    candidate_ssrfb = extract_block(patched, "__aicore__ inline void SSRFB(")
    if original_ssrfb != candidate_ssrfb:
        raise RuntimeError("SSRFB changed outside the approved patch scope")
    if len(re.findall(r"\bFreeTensor\s*<", candidate_ssrfb)) != 6:
        raise RuntimeError("SSRFB must retain exactly six FreeTensor calls")

    if "FreeTensor" in core0_block:
        raise RuntimeError("Process release must be outside the core0-only block")
    if process.count("akkTQue.FreeTensor<DTYPE_A>(akkLocal);") != 1:
        raise RuntimeError("Process must retain the single akkLocal release")

    required_fragments = (
        "workspace, const QrV2TilingData *tilingData, TPipe *pipe)",
        "uint32_t useCoreNum{0};",
        "uint32_t repeatNum{0};",
        "uint64_t offsetK{0};",
        "uint64_t offsetI{0};",
        "uint64_t offsetW{0};",
        "op.vtvMatmulObj.Init(&vtvTilingData);",
        "op.currentQMatmulObj.Init(&currentQTilingData);",
    )
    for fragment in required_fragments:
        if fragment not in patched:
            raise RuntimeError(f"required candidate fragment missing: {fragment}")
    for forbidden in ("const_cast", "isfinite", "isnan", "QrV2Dump", "printf("):
        if forbidden in patched:
            raise RuntimeError(f"forbidden release-candidate fragment present: {forbidden}")

    return {
        "source_sha256": source_sha,
        "candidate_sha256": candidate_sha,
        "source_free_tensor_calls": original_free_count,
        "candidate_free_tensor_calls": candidate_free_count,
        "larfb_free_tensor_calls": 0,
        "ssrfb_free_tensor_calls": 6,
        "process_release_after_q_writeback": True,
        "process_sync_after_q_writeback": True,
        "process_sync_before_release": True,
        "process_sync_outside_core0": True,
        "process_release_before_tsqrt": True,
        "process_release_outside_core0": True,
        "alpha_buffer_bytes": 64,
        "alpha_buffer_fp32_elements": 16,
        "alpha_duplicate_calls": 2,
        "v1_equivalent_sha256": v1_equivalent_sha,
        "v2_delta_alpha_only": True,
        "v2_candidate_sha256": v2_equivalent_sha,
        "v3_delta_mte3_mte2_only": True,
        "v3_mte3_mte2_sequence_count": 1,
        "v3_mte3_mte2_after_workspace_copy": True,
        "v3_mte3_mte2_before_matmul_a": True,
        "v3_candidate_sha256": v3_equivalent_sha,
        "v4_delta_sync_and_ownership_only": True,
        "v4_per_core_scratch_offset": True,
        "v4_active_core_upper_bound": "coreId < useCoreNum <= blockp",
        "v4_scratch_end_upper_bound": "2 * m * blockSize",
        "v4_active_core_bound_statically_proved": True,
        "v4_workspace_bound_statically_proved": True,
        "v4_update_a_mte3_mte2_sequence_count": 1,
        "v4_sync_all_calls_added": 1,
        "sync_all_calls": patched.count("SyncAll();"),
        "base_tiling_slot_references_unchanged": True,
    }


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one source match, got {count}")
    return text.replace(old, new, 1)


def replace_exact_count(text: str, old: str, new: str, count: int, label: str) -> str:
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(f"{label}: expected exactly {count} source matches, got {actual}")
    return text.replace(old, new)


def build_candidate(source: bytes) -> bytes:
    digest = sha256_bytes(source)
    if digest != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"source SHA-256 mismatch: {digest}")

    text = source.decode("utf-8")
    text = replace_once(
        text,
        "constexpr int32_t UB_ALIGN_SIZE = 32;\n",
        """constexpr int32_t UB_ALIGN_SIZE = 32;
constexpr uint32_t ALPHA_BUF_BYTES = 2 * UB_ALIGN_SIZE;
constexpr uint32_t ALPHA_BUF_FP32_ELEMENTS = ALPHA_BUF_BYTES / sizeof(float);
static_assert(ALPHA_BUF_BYTES == 64, "alphaBuf must remain 64 bytes");
static_assert(ALPHA_BUF_FP32_ELEMENTS == 16, "alphaBuf must contain 16 FP32 elements");
""",
        "alphaBuf byte and element constants",
    )
    text = replace_once(
        text,
        "        pipe->InitBuffer(alphaBuf, 2 * UB_ALIGN_SIZE);\n",
        "        pipe->InitBuffer(alphaBuf, ALPHA_BUF_BYTES);\n",
        "alphaBuf explicit byte allocation",
    )
    text = replace_exact_count(
        text,
        "        Duplicate(alphaLocal, static_cast<DTYPE_A>(0), 2 * UB_ALIGN_SIZE);\n",
        "        Duplicate(alphaLocal, static_cast<DTYPE_A>(0), ALPHA_BUF_FP32_ELEMENTS);\n",
        2,
        "alphaBuf Duplicate FP32 element count",
    )
    text = replace_once(
        text,
        """        DataCopy(workspaceInGm[this->m * this->blockSize], vLocal, this->blockSize * this->blockSize);
        this->vtvMatmulObj.SetTensorA(workspaceInGm[this->m * this->blockSize]);
""",
        V4_CALC_Q_SCRATCH_BLOCK,
        "CalcQForLARFB per-core workspace and MTE3_MTE2 dependency",
    )
    text = replace_once(
        text,
        """        DataCopy(workspaceInGm[offsetW], aLocal, this->blockElement);
        // 计算 A' = Q @ A
""",
        """        DataCopy(workspaceInGm[offsetW], aLocal, this->blockElement);
"""
        + V4_UPDATE_A_MTE3_MTE2_SEQUENCE
        + """        // 计算 A' = Q @ A
""",
        "UpdateAForLARFB workspace MTE3_MTE2 dependency",
    )
    text = replace_once(
        text,
        """    __aicore__ inline void Init(
        GM_ADDR a, GM_ADDR q, GM_ADDR r, GM_ADDR workspace, QrV2TilingData *tilingData, TPipe *pipe)
""",
        """    __aicore__ inline void Init(
        GM_ADDR a, GM_ADDR q, GM_ADDR r, GM_ADDR workspace, const QrV2TilingData *tilingData, TPipe *pipe)
""",
        "KernelLinalgQrV2 Init const tiling",
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
        "kernel entry nested TCubeTiling copies",
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
        "Matmul Init local tiling copies",
    )
    text = replace_once(
        text,
        """        uint32_t useCoreNum;
        uint32_t formerNum;
        uint32_t formerRepeatNum;
        uint32_t tailNum;
        uint32_t tailRepeatNum;
        // 不需要所有Vector核都参与计算
""",
        """        uint32_t useCoreNum{0};
        uint32_t formerNum{0};
        uint32_t formerRepeatNum{0};
        uint32_t tailNum{0};
        uint32_t tailRepeatNum{0};
        // 不需要所有Vector核都参与计算
""",
        "InitBaseTiling initialization",
    )
    text = replace_once(
        text,
        """        uint32_t useCoreNum;
        uint32_t formerNum;
        uint32_t formerRepeatNum;
        uint32_t tailNum;
        uint32_t tailRepeatNum;
        uint32_t repeatNum;
        uint64_t offsetK;
        uint64_t offsetI;
        uint64_t offsetW;
        // 一行有多少对[A_kj A_ij]
""",
        """        uint32_t useCoreNum{0};
        uint32_t formerNum{0};
        uint32_t formerRepeatNum{0};
        uint32_t tailNum{0};
        uint32_t tailRepeatNum{0};
        uint32_t repeatNum{0};
        uint64_t offsetK{0};
        uint64_t offsetI{0};
        uint64_t offsetW{0};
        // 一行有多少对[A_kj A_ij]
""",
        "InitTaskTiling initialization",
    )
    text = replace_once(
        text,
        """            if (coreId == 0) {
                CalcQForLARFB(false);
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                // copy first Q from ub to colQGm
                DataCopy(colQGm, this->qLocal, blockSize * blockSize);
                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
            }
            for (uint32_t i = k + 1; i < blockp; ++i) {
""",
        """            if (coreId == 0) {
                CalcQForLARFB(false);
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                // copy first Q from ub to colQGm
                DataCopy(colQGm, this->qLocal, blockSize * blockSize);
                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
            }
            SyncAll();
            // CalcQForLARFB(false) still consumes the GEQRT-owned local tensors.
            // Release them only after core0 has materialized the diagonal-block Q.
            tTQue.FreeTensor<DTYPE_A>(tLocal);
            vTQue.FreeTensor<DTYPE_A>(vLocal);
            aTQue.FreeTensor<DTYPE_A>(aLocal);
            for (uint32_t i = k + 1; i < blockp; ++i) {
""",
        "Process lifetime release",
    )
    text = replace_once(
        text,
        """        if (this->coreId >= tilingInfo.useCoreNum) {
            aLocal = aTQue.DeQue<DTYPE_A>();
            tLocal = tTQue.DeQue<DTYPE_A>();
            vLocal = vTQue.DeQue<DTYPE_A>();
            tTQue.FreeTensor<DTYPE_A>(tLocal);
            vTQue.FreeTensor<DTYPE_A>(vLocal);
            aTQue.FreeTensor<DTYPE_A>(aLocal);
            return;
        }
""",
        """        if (this->coreId >= tilingInfo.useCoreNum) {
            aLocal = aTQue.DeQue<DTYPE_A>();
            tLocal = tTQue.DeQue<DTYPE_A>();
            vLocal = vTQue.DeQue<DTYPE_A>();
            return;
        }
""",
        "LARFB inactive-core lifetime",
    )
    text = replace_once(
        text,
        """        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3_MTE2);
        qaMatmulObj.End();

        tTQue.FreeTensor<DTYPE_A>(tLocal);
        vTQue.FreeTensor<DTYPE_A>(vLocal);
        aTQue.FreeTensor<DTYPE_A>(aLocal);
    }

    __aicore__ inline void TSQRT(int32_t k, int32_t i)
""",
        """        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3_MTE2);
        qaMatmulObj.End();
    }

    __aicore__ inline void TSQRT(int32_t k, int32_t i)
""",
        "LARFB active-core lifetime",
    )

    candidate = text.encode("utf-8")
    if candidate == source:
        raise RuntimeError("candidate is identical to source")
    verify_candidate_structure(source, candidate)
    return candidate


def write_new_file(path: Path, payload: bytes) -> None:
    """Write a new regular file without following or replacing an existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    source_argument = args.source.absolute()
    if source_argument.is_symlink():
        raise ValueError("source must not be a symlink")
    source_path = source_argument.resolve(strict=True)
    if not source_path.is_file():
        raise ValueError("source must be a non-symlink regular file")
    source = source_path.read_bytes()
    candidate = build_candidate(source)
    report = verify_candidate_structure(source, candidate)
    print("source_sha256=" + report["source_sha256"])
    print("candidate_sha256=" + report["candidate_sha256"])
    print("candidate_size=" + str(len(candidate)))
    print("structure_gate=True")

    if args.check:
        if args.output is not None:
            raise ValueError("output must be omitted with --check")
        return 0
    if args.output is None:
        raise ValueError("output is required unless --check is used")
    output = args.output.absolute()
    if output == source_path:
        raise ValueError("output must differ from source")
    if output.exists() and output.is_symlink():
        raise ValueError("output must not be a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink():
        raise ValueError("output parent must not be a symlink")
    write_new_file(output, candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
