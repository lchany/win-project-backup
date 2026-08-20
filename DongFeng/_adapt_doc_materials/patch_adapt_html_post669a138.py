# -*- coding: utf-8 -*-
"""Append commits 10f897d..3a1d763 to 东风MapQR_NPU性能优化适配文档.html."""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

ROOT = Path(r"C:\project\win-project-backup\DongFeng")
HTML_PATH = ROOT / "东风MapQR_NPU性能优化适配文档.html"
MATERIALS = ROOT / "_adapt_doc_materials"
SOAP_PATH = ROOT / "projects" / "mmdet3d_plugin" / "optimizers" / "soap.py"

CFG_REL = (
    "projects/configs/20260113st/"
    "mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py"
)

ENTRIES = [
    {
        "sha": "10f897d",
        "title": "mx QR 192×192 绕开 QrV2（暂撤 stale-Q）",
        "files_html": (
            "mx_driving_cloud/ops/linalg.py（已撤回，见 5899e94）<br/>"
            "projects/mmdet3d_plugin/optimizers/soap.py"
        ),
        "adapt": "在 SOAP 中接入 mx_driving_cloud.linalg.qr，并临时入库本地 linalg 包装（192×192 走 torch QR 绕开 QrV2）；同期撤掉 stale-Q 异步路径以便单变量验证 mx QR。",
        "principle": "QrV2 在部分设备/shape 上存在末 tile 与设备绑定缺陷；192×192 固定走 AICPU torch QR 可规避 BAD dump，其余 shape 仍走 mx 算子以追求 NPU 性能。",
        "why": "Profiler 显示 SOAP 周期 QR 是最大瓶颈；mx QR 若稳定可用可显著缩短周期步。",
        "action": "新增 mx_driving_cloud/ops/linalg.py（48 行）；soap.py 改 mx QR 并移除 stale-Q 相关方法与周期分支。",
        "files_field": "mx_driving_cloud/ops/linalg.py；projects/mmdet3d_plugin/optimizers/soap.py",
        "diff": "diff_10f897d.patch",
        "sources": [],
    },
    {
        "sha": "5899e94",
        "title": "撤回误入库 mx_driving_cloud 包",
        "files_html": "mx_driving_cloud/__init__.py<br/>mx_driving_cloud/ops/__init__.py<br/>mx_driving_cloud/ops/linalg.py",
        "adapt": "删除误提交进业务仓库的 mx_driving_cloud 源码树，改回仅使用客户环境 driving-cloud-ops wheel。",
        "principle": "算子包应由客户 CANN/wheel 统一交付；仓库内嵌副本会与 site-packages 版本漂移并造成双路径维护。",
        "why": "10f897d 的本地 linalg 仅为临时 bypass 试验，不应作为正式交付物留在 Git 中。",
        "action": "删除 mx_driving_cloud/ 三个占位/包装文件（48 行 linalg 逻辑撤回）；SOAP 仍保留 mx QR 调用，运行时走 wheel。",
        "files_field": "mx_driving_cloud/__init__.py；mx_driving_cloud/ops/__init__.py；mx_driving_cloud/ops/linalg.py",
        "diff": "diff_5899e94.patch",
        "sources": [],
    },
    {
        "sha": "9565044",
        "title": "SOAP 使用 mx_driving_cloud.linalg.qr",
        "files_html": "projects/mmdet3d_plugin/optimizers/soap.py",
        "adapt": "在恢复 stale-Q（k=4）异步流水的前提下，将两处 QR 调用切换为客户 mx_driving_cloud.linalg.qr。",
        "principle": "stale-Q 把 AICPU QR 移出关键路径；mx QR 若在 NPU 上更快，可进一步压缩周期步墙时。",
        "why": "STEP-284 目标是把 torch.linalg.qr 替换为 NPU 原生 QR，与 fb979b2 + 2846401 性能栈叠加。",
        "action": "soap.py：import mx_driving_cloud；get_orthogonal_matrix_QR / _qr_finish 改用 mx_driving_cloud.linalg.qr；恢复 STEP-221 stale-Q 全套方法与周期分支。",
        "files_field": "projects/mmdet3d_plugin/optimizers/soap.py",
        "diff": "diff_9565044.patch",
        "sources": [("projects/mmdet3d_plugin/optimizers/soap.py", None)],
    },
    {
        "sha": "27b1d6d",
        "title": "去除随机性固定（对齐 GPU 训练语义）",
        "files_html": (
            "mmcv/runner/hooks/optimizer.py<br/>"
            f"{CFG_REL}<br/>"
            "projects/mmdet3d_plugin/core/hook/__init__.py<br/>"
            "projects/mmdet3d_plugin/datasets/internal_dataset_track_stream.py<br/>"
            "projects/mmdet3d_plugin/datasets/pipelines/vectorize_local_map.py<br/>"
            "projects/mmdet3d_plugin/models/detectors/spetr3d.py"
        ),
        "adapt": "移除为 loss 对齐临时加入的随机性固定、msprobe 与额外 hook，恢复与客户 GPU 训练语义一致的代码路径。",
        "principle": "随机性固定与额外 debugger 会改变数据顺序、算子选择与 Host 侧开销，不属于性能优化交付物。",
        "why": "669a138 之后仍需在正式 config/数据管线上撤掉诊断期随机性约束，避免与 GPU 基线合同偏离。",
        "action": "删除 optimizer hook 中的固定逻辑；config 去掉临时字段；dataset/spetr3d 移除 msprobe/随机性注释块与相关 import。",
        "files_field": "mmcv/runner/hooks/optimizer.py；正式 config；core/hook；internal_dataset_track_stream；vectorize_local_map；spetr3d.py",
        "diff": "diff_27b1d6d.patch",
        "sources": [],
    },
    {
        "sha": "3a1d763",
        "title": "SOAP QR 修复：torch.linalg.qr 消除 Iter6 NaN",
        "files_html": "projects/mmdet3d_plugin/optimizers/soap.py",
        "adapt": "STEP-324~326 验证 mx QR 与 SOAP 下游（sort_idx / exp_avg_sq / 预条件状态）Q/R 约定不兼容导致 Iter6+ NaN；正式修复为 torch.linalg.qr（SOAP 原设计语义）。",
        "principle": "mx 与 torch 的 Q/R 分解非等价（非单纯符号差）；SOAP 依赖 torch 约定做特征排序与状态更新，mx 路径会在 Iter6 起 seg loss NaN。",
        "why": "这是功能正确性修复，不是性能回退：torch QR 30 步全 finite，SOAP 周期步仍 ~5 s（STEP-326 已验证无 steady-state 耗时劣化）。",
        "action": "删除 import mx_driving_cloud；两处 QR 调用改回 torch.linalg.qr；不保留 SOAP_QR_BACKEND 运行时兼容层。",
        "files_field": "projects/mmdet3d_plugin/optimizers/soap.py",
        "diff": "diff_3a1d763.patch",
        "sources": [("projects/mmdet3d_plugin/optimizers/soap.py", SOAP_PATH)],
    },
]


def esc(text: str) -> str:
    return html.escape(text.replace("\r\n", "\n").replace("\r", "\n"), quote=False)


def fold(summary: str, body: str) -> str:
    return (
        '<details class="code-fold">\n'
        f"  <summary>{html.escape(summary, quote=False)}</summary>\n"
        f'  <pre class="code"><code>{esc(body)}</code></pre>\n'
        "</details>"
    )


def strip_patch_header(diff: str) -> str:
    lines = diff.replace("\r\n", "\n").splitlines()
    for i, line in enumerate(lines):
        if line.startswith("diff --git "):
            return "\n".join(lines[i:]) + "\n"
    return diff


def replace_source_by_path(doc: str, path: str, new_src: str, suffix: str = "整文件终稿") -> str:
    pattern = re.compile(
        r'(<details class="code-fold">\s*'
        r"<summary>Source — "
        + re.escape(path)
        + rf"（{re.escape(suffix)}）</summary>\s*"
        r'<pre class="code"><code>)'
        r"(.*?)"
        r"(</code></pre>\s*</details>)",
        re.S,
    )
    repl = r"\1" + esc(new_src) + r"\3"
    new_doc, n = pattern.subn(repl, doc)
    if n == 0:
        raise SystemExit(f"no Source fold matched for {path} ({suffix})")
    print(f"replaced {n} Source folds for {path}")
    return new_doc


def build_card(entry: dict) -> str:
    diff_text = strip_patch_header((MATERIALS / entry["diff"]).read_text(encoding="utf-8"))
    stacks = [fold(f"Diff — {entry['sha']} {entry['title']}", diff_text)]
    for path, src_path in entry.get("sources", []):
        if src_path is None:
            continue
        src = src_path.read_text(encoding="utf-8")
        stacks.append(fold(f"Source — {path}（整文件，HEAD 3a1d763）", src))
    file_stack = "\n".join(stacks)
    return (
        f'<div class="card" id="item-{entry["sha"]}">'
        f'<h3><span class="sha-chip">{entry["sha"]}</span>{html.escape(entry["title"], quote=False)}</h3>'
        f'<p class="field"><span class="label">适配对象</span>{html.escape(entry["adapt"], quote=False)}</p>'
        f'<p class="field"><span class="label">原理</span>{html.escape(entry["principle"], quote=False)}</p>'
        f'<p class="field"><span class="label">为何可优化</span>{html.escape(entry["why"], quote=False)}</p>'
        f'<p class="field"><span class="label">做了什么</span>{html.escape(entry["action"], quote=False)}</p>'
        f'<p class="field"><span class="label">涉及文件</span>{html.escape(entry["files_field"], quote=False)}</p>'
        f'<div class="file-stack">\n{file_stack}\n</div></div>'
    )


def main() -> None:
    if not SOAP_PATH.is_file():
        raise SystemExit(f"missing {SOAP_PATH}")
    doc = HTML_PATH.read_text(encoding="utf-8")

    anchor = (
        "<tr><td><code>669a138</code></td><td>配置最终版本修改（不涉及性能修改）</td>"
    )
    if anchor not in doc:
        raise SystemExit("669a138 row not found")

    rows = []
    toc_items = []
    for i, entry in enumerate(ENTRIES, start=11):
        rows.append(
            f"<tr><td><code>{entry['sha']}</code></td>"
            f"<td>{html.escape(entry['title'], quote=False)}</td>"
            f"<td>{entry['files_html']}</td></tr>"
        )
        toc_items.append(
            f"<li><a href='#item-{entry['sha']}'>{i}. {html.escape(entry['title'], quote=False)}（{entry['sha']}）</a></li>"
        )

    old_tail = (
        "<tr><td><code>669a138</code></td><td>配置最终版本修改（不涉及性能修改）</td>"
        f"<td>{html.escape(CFG_REL)}<br/>tools/ddp_train.sh<br/>tools/train_spetr.py</td></tr>"
        "</tbody></table>"
    )
    new_tail = (
        "<tr><td><code>669a138</code></td><td>配置最终版本修改（不涉及性能修改）</td>"
        f"<td>{html.escape(CFG_REL)}<br/>tools/ddp_train.sh<br/>tools/train_spetr.py</td></tr>"
        + "".join(rows)
        + "</tbody></table>"
    )
    if old_tail not in doc:
        raise SystemExit("table tail not found")
    doc = doc.replace(old_tail, new_tail)

    old_toc = (
        "<li><a href='#item-669a138'>10. 配置最终版本修改（669a138）</a></li>"
        "<li><a href='#soap-final'>附录：全部改动文件终稿索引</a></li></ol>"
    )
    new_toc = (
        "<li><a href='#item-669a138'>10. 配置最终版本修改（669a138）</a></li>"
        + "".join(toc_items)
        + "<li><a href='#soap-final'>附录：全部改动文件终稿索引</a></li></ol>"
    )
    if old_toc not in doc:
        raise SystemExit("toc block not found")
    doc = doc.replace(old_toc, new_toc)

    cards = "".join(build_card(e) for e in ENTRIES)
    marker = '</div></div></section><section class="block" id="soap-final">'
    if marker not in doc:
        raise SystemExit("sec-items close marker not found")
    doc = doc.replace(marker, "</div></div>" + cards + marker, 1)

    soap_src = SOAP_PATH.read_text(encoding="utf-8")
    doc = replace_source_by_path(doc, "projects/mmdet3d_plugin/optimizers/soap.py", soap_src)

    rb_old = (
        '    <li>配置/入口合同回退：回退提交 <code>669a138</code>（正式 config、'
        '<code>ddp_train.sh</code>、<code>train_spetr.py</code>）。</li>\n'
        "  </ul>"
    )
    rb_new = (
        '    <li>配置/入口合同回退：回退提交 <code>669a138</code>（正式 config、'
        '<code>ddp_train.sh</code>、<code>train_spetr.py</code>）。</li>\n'
        "    <li>SOAP QR 正确性修复：正式路径为 <code>torch.linalg.qr</code>（<code>3a1d763</code>，修复 mx QR 约定不兼容导致的 Iter6 NaN）；"
        "勿使用运行时 monkey-patch。长期 mx QR 性能路径待算子约定对齐后再单独提交。</li>\n"
        "    <li>随机性固定：若需恢复诊断期固定，回退 <code>27b1d6d</code>。</li>\n"
        "  </ul>"
    )
    if rb_old not in doc:
        raise SystemExit("rollback block not found")
    doc = doc.replace(rb_old, rb_new)

    HTML_PATH.write_text(doc, encoding="utf-8", newline="\n")
    print("wrote", HTML_PATH, "bytes", HTML_PATH.stat().st_size)
    for sha in [e["sha"] for e in ENTRIES]:
        print("present", sha, doc.count(f"item-{sha}"))


if __name__ == "__main__":
    main()
