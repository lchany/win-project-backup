# -*- coding: utf-8 -*-
"""Generate Markdown adapt doc from HTML + patch materials."""
from __future__ import annotations

import html as html_lib
import re
from pathlib import Path

ROOT = Path(r"C:\project\win-project-backup\DongFeng")
HTML_PATH = ROOT / "东风MapQR_NPU性能优化适配文档.html"
OUT_PATH = ROOT / "东风MapQR_NPU性能优化适配文档.md"
MATERIALS = ROOT / "_adapt_doc_materials"

CFG = (
    "projects/configs/20260113st/"
    "mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py"
)
CFG_LABEL = "正式 config（mv2dfusion…finetune.py）"

FILE_ALIASES = {
    CFG: CFG_LABEL,
}

INLINE_DIFF_MAX = 80

EXTRA = {
    "3a1d763": {
        "why_label": "修复动机",
        "why": (
            "这是功能正确性修复，不是性能回退：torch QR 30 步全 finite，"
            "SOAP 周期步仍 ~5 s（STEP-326 已验证无 steady-state 耗时劣化）。"
        ),
    },
}


def unescape(text: str) -> str:
    text = html_lib.unescape(text)
    text = re.sub(r"<code>([^<]*)</code>", r"`\1`", text)
    return text.strip()


def parse_cards(doc: str) -> list[dict]:
    pattern = re.compile(
        r'<div class="card" id="item-(?P<sha>[0-9a-f]+)">'
        r'<h3><span class="sha-chip">(?P=sha)</span>(?P<title>[^<]+)</h3>'
        r"(?P<body>.*?)"
        r"</div></div>",
        re.S,
    )
    cards = []
    for m in pattern.finditer(doc):
        sha = m.group("sha")
        body = m.group("body")
        fields = {}
        for fm in re.finditer(
            r'<span class="label">([^<]+)</span>([^<]+(?:<[^/][^<]*)*)',
            body,
        ):
            label = fm.group(1).strip()
            val = unescape(re.sub(r"<[^>]+>", "", fm.group(2)))
            fields[label] = val
        cards.append(
            {
                "sha": sha,
                "title": unescape(m.group("title")),
                "fields": fields,
            }
        )
    return cards


def parse_table(doc: str) -> list[tuple[str, str, str]]:
    m = re.search(r'id="sec-list".*?<tbody>(.*?)</tbody>', doc, re.S)
    if not m:
        return []
    rows = []
    for tr in re.finditer(
        r"<tr><td><code>([0-9a-f]+)</code></td><td>([^<]+)</td><td>(.*?)</td></tr>",
        m.group(1),
        re.S,
    ):
        sha, title, files_raw = tr.group(1), tr.group(2), tr.group(3)
        files = unescape(re.sub(r"<br\s*/?>", "\n", files_raw))
        rows.append((sha, title.strip(), files))
    return rows


def patch_path(sha: str) -> Path | None:
    p = MATERIALS / f"diff_{sha}.patch"
    return p if p.is_file() else None


def split_paths(text: str) -> list[str]:
    parts = re.split(r"[；\n]|(?:\s*\+\s*)", text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = re.sub(r"（[^）]*）", "", p).strip()
        if p:
            out.append(p)
    return out


def format_path(path: str) -> str:
    label = FILE_ALIASES.get(path.strip(), path.strip())
    if label == path.strip():
        return f"`{label}`"
    return f"`{label}`"


def format_files_bullets(files_text: str) -> list[str]:
    raw = files_text.replace("；", "\n")
    items = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if "（" in line:
            path = line.split("（", 1)[0].strip()
            note = "（" + line.split("（", 1)[1]
            items.append(f"{format_path(path)}{note}")
        else:
            items.append(format_path(line))
    return items


def diff_section(sha: str) -> list[str]:
    p = patch_path(sha)
    rel = f"_adapt_doc_materials/diff_{sha}.patch"
    if p is None:
        return ["_（本地无 diff 补丁文件）_"]

    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("diff --git "):
            start = i
            break
    body_lines = lines[start:]
    total = len(body_lines)

    out = []
    if total <= INLINE_DIFF_MAX:
        out.append(f"> **完整 Diff**：[`{rel}`]({rel})（{total} 行）")
        out.append("")
        out.extend(["```diff", "\n".join(body_lines), "```"])
    else:
        out.append(
            f"> **完整 Diff**：[`{rel}`]({rel})（{total} 行）  \n"
            f"> 以下仅展示前 40 行预览。"
        )
        out.append("")
        out.extend(["```diff", "\n".join(body_lines[:40]), "```"])
    return out


def main() -> None:
    doc = HTML_PATH.read_text(encoding="utf-8")
    table = parse_table(doc)
    cards = parse_cards(doc)
    card_by_sha = {c["sha"]: c for c in cards}

    lines: list[str] = [
        "# 东风 MapQR 昇腾 NPU 性能优化适配文档",
        "",
        "| 项 | 值 |",
        "|----|-----|",
        "| Git 分支 | `ascend_npu_optimize` |",
        "| 当前 HEAD（QR 修复） | `3a1d763` |",
        "| HTML 对照版 | [`东风MapQR_NPU性能优化适配文档.html`](东风MapQR_NPU性能优化适配文档.html) |",
        "| 正式 config | `" + CFG + "` |",
        "",
        "## 目录",
        "",
        "- [1. 文档说明](#1-文档说明)",
        "- [2. 优化清单](#2-优化清单)",
        "- [3. 分项适配说明](#3-分项适配说明)",
        "- [4. 附录：终稿文件索引](#4-附录终稿文件索引)",
        "- [5. 启用与回退](#5-启用与回退)",
        "- [6. mx QR 试验链说明](#6-mx-qr-试验链说明669a138-之后)",
        "",
        "---",
        "",
        "## 1. 文档说明",
        "",
        "本文整理 MapQR / MV2DFusion（SOAP）在昇腾 NPU 上已落地的**性能适配**与**关键正确性修复**。",
        "每个提交包含：适配对象、原理、动机、落地动作；Diff 以补丁文件为准，分项内附预览。",
        "",
        "| 类型 | 说明 |",
        "|------|------|",
        "| **Diff** | `_adapt_doc_materials/diff_<sha>.patch` |",
        "| **Source 终稿** | HTML 版附录含整文件；本地镜像见 `projects/`、`mmcv/`、`tools/` |",
        "",
        "**生产启用**：`tools/ddp_train.sh` 默认 `SOAP_STALE_Q_K=4`、`expandable_segments`；配置 `pin_memory=True`。",
        "",
        "---",
        "",
        "## 2. 优化清单",
        "",
        "共 **15** 项（按时间顺序）。涉及文件详见 [§3 分项说明](#3-分项适配说明)。",
        "",
        "| # | 提交 | 主题 |",
        "|--:|------|------|",
    ]

    for i, (sha, title, _files) in enumerate(table, 1):
        lines.append(f"| {i} | [`{sha}`](#item-{sha}) | {title} |")

    lines.extend(["", "---", "", "## 3. 分项适配说明", ""])

    for i, (sha, title, files_from_table) in enumerate(table, 1):
        c = card_by_sha.get(sha, {"fields": {}})
        f = c["fields"]
        why_label = EXTRA.get(sha, {}).get("why_label", "为何可优化")
        why_text = (
            EXTRA.get(sha, {}).get("why")
            or f.get("为何可优化")
            or f.get("修复动机", "—")
        )
        files_field = f.get("涉及文件", files_from_table.replace("\n", "；"))
        file_items = format_files_bullets(files_field)

        lines.append(f'<a id="item-{sha}"></a>')
        lines.append("")
        lines.append(f"### {i}. `{sha}` — {title}")
        lines.append("")
        lines.append(f"- **适配对象**：{f.get('适配对象', '—')}")
        lines.append(f"- **原理**：{f.get('原理', '—')}")
        lines.append(f"- **{why_label}**：{why_text}")
        lines.append(f"- **做了什么**：{f.get('做了什么', '—')}")
        lines.append("- **涉及文件**：")
        for item in file_items:
            lines.append(f"  - {item}")
        lines.append("")
        lines.append("#### Diff")
        lines.append("")
        lines.extend(diff_section(sha))
        lines.append("")
        lines.append(f"[↑ 回到清单](#2-优化清单)")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.extend(
        [
            "## 4. 附录：终稿文件索引",
            "",
            "| 路径 | 说明 |",
            "|------|------|",
            f"| `{CFG}` | 正式训练 config |",
            "| `tools/ddp_train.sh` | 训练入口（`SOAP_STALE_Q_K=4`、`expandable_segments`） |",
            "| `tools/train_spetr.py` | 训练脚本 |",
            "| `projects/mmdet3d_plugin/optimizers/soap.py` | SOAP 优化器（HEAD `3a1d763`） |",
            "| `projects/mmdet3d_plugin/models/utils/multi_scale_deformable_attn_function.py` | MSDA DrivingSDK |",
            "| `projects/mmdet3d_plugin/models/losses/geo_loss.py` | GeometricLoss |",
            "| `projects/mmdet3d_plugin/models/detectors/spetr3d.py` | 检测器 |",
            "| `mmcv/parallel/data_container.py` | `pin_memory` |",
            "| `mmcv/runner/hooks/logger/text.py` | TextLogger |",
            "",
            "---",
            "",
            "## 5. 启用与回退",
            "",
            "| 场景 | 操作 |",
            "|------|------|",
            "| 完整收益栈 | `tools/ddp_train.sh` 默认 `SOAP_STALE_Q_K=4` + `expandable_segments`，配置 `pin_memory=True` |",
            "| stale-Q 回退 | `ddp_train.sh` 中 `SOAP_STALE_Q_K=4` → `0` |",
            "| allocator 回退 | 去掉 `ddp_train.sh` 中 `expandable_segments` 行 |",
            "| 配置/入口回退 | revert `669a138` |",
            "| SOAP QR（当前正式） | `torch.linalg.qr`（`3a1d763`，NaN 根因修复，非性能回退） |",
            "| 随机性固定 | 若需恢复诊断期行为，revert `27b1d6d` |",
            "",
            "---",
            "",
            "## 6. mx QR 试验链说明（669a138 之后）",
            "",
            "```text",
            "669a138 → 10f897d → 5899e94 → 9565044 → 27b1d6d → 3a1d763 (HEAD)",
            "```",
            "",
            "| 顺序 | 提交 | 性质 |",
            "|--:|------|------|",
            "| 1 | `10f897d` | mx QR 试验 + 暂撤 stale-Q |",
            "| 2 | `5899e94` | 撤回误入库 `mx_driving_cloud` 包 |",
            "| 3 | `9565044` | 恢复 stale-Q + mx QR（引入 NaN 风险） |",
            "| 4 | `27b1d6d` | 去除随机性固定（对齐 GPU 语义，非性能项） |",
            "| 5 | `3a1d763` | **NaN 根因修复**（torch QR，非性能回退） |",
            "",
            "**STEP-324~326 结论**：mx QR 与 SOAP 下游 Q/R 约定不兼容 → Iter6+ NaN；"
            "torch QR 30/30 finite，steady SOAP 步 ~5 s。",
            "",
        ]
    )

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT_PATH} lines={len(lines)} bytes={OUT_PATH.stat().st_size}")


if __name__ == "__main__":
    main()
