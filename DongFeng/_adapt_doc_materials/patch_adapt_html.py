# -*- coding: utf-8 -*-
"""Patch 东风MapQR_NPU性能优化适配文档.html for commit 669a138."""
from __future__ import annotations

import html
import re
import shutil
from pathlib import Path

ROOT = Path(r"C:\project\win-project-backup\DongFeng")
HTML_PATH = ROOT / "东风MapQR_NPU性能优化适配文档.html"
BUNDLE = ROOT / "_adapt_doc_materials" / "step234_html_bundle"
MATERIALS = ROOT / "_adapt_doc_materials"

CFG_REL = (
    "projects/configs/20260113st/"
    "mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py"
)


def esc(text: str) -> str:
    return html.escape(text.replace("\r\n", "\n").replace("\r", "\n"), quote=False)


def fold(summary: str, body: str) -> str:
    return (
        '<details class="code-fold">\n'
        f"  <summary>{html.escape(summary, quote=False)}</summary>\n"
        f'  <pre class="code"><code>{esc(body)}</code></pre>\n'
        "</details>"
    )


def strip_commit_header(diff: str) -> str:
    lines = diff.replace("\r\n", "\n").splitlines()
    for i, line in enumerate(lines):
        if line.startswith("diff --git "):
            return "\n".join(lines[i:]) + "\n"
    return diff


def replace_source_by_path(doc: str, path: str, new_src: str) -> str:
    pattern = re.compile(
        r'(<details class="code-fold">\s*'
        r"<summary>Source — "
        + re.escape(path)
        + r"（[^<]*）</summary>\s*"
        r'<pre class="code"><code>)'
        r"(.*?)"
        r"(</code></pre>\s*</details>)",
        re.S,
    )
    repl = r"\1" + esc(new_src) + r"\3"
    new_doc, n = pattern.subn(repl, doc)
    if n == 0:
        raise SystemExit(f"no Source fold matched for {path}")
    print(f"replaced {n} Source folds for {path}")
    return new_doc


def main() -> None:
    ddp = (BUNDLE / "ddp_train.sh").read_text(encoding="utf-8")
    train = (BUNDLE / "train_spetr.py").read_text(encoding="utf-8")
    cfg = (BUNDLE / "official_config.py").read_text(encoding="utf-8")
    diff = strip_commit_header((BUNDLE / "669a138.diff").read_text(encoding="utf-8"))

    shutil.copyfile(BUNDLE / "ddp_train.sh", MATERIALS / "src_tools_ddp_train.sh")
    shutil.copyfile(BUNDLE / "train_spetr.py", MATERIALS / "src_tools_train_spetr.py")
    shutil.copyfile(BUNDLE / "official_config.py", MATERIALS / f"src_{CFG_REL.replace('/', '_')}")
    (MATERIALS / "diff_669a138.patch").write_text(diff, encoding="utf-8")

    doc = HTML_PATH.read_text(encoding="utf-8")

    old_enable = (
        "生产启用补充：<code>export SOAP_STALE_Q_K=4</code>（默认 0 为同步回退）；"
        "训练入口已默认 <code>expandable_segments</code>；配置 <code>pin_memory=True</code>。"
    )
    new_enable = (
        "生产启用：原名 <code>tools/ddp_train.sh</code> 已默认 "
        "<code>SOAP_STALE_Q_K=4</code> 与 <code>expandable_segments</code>；"
        "配置 <code>pin_memory=True</code>。算子侧 stale-Q 默认仍为 0，由训练入口注入 k=4。"
    )
    if old_enable not in doc:
        raise SystemExit("enable sentence not found")
    doc = doc.replace(old_enable, new_enable)

    old_row = (
        "<tr><td><code>fa95a2a</code></td><td>训练入口默认 expandable_segments</td>"
        "<td>tools/ddp_train.sh</td></tr></tbody></table>"
    )
    new_row = (
        "<tr><td><code>fa95a2a</code></td><td>训练入口默认 expandable_segments</td>"
        "<td>tools/ddp_train.sh</td></tr>"
        "<tr><td><code>669a138</code></td><td>配置最终版本修改（不涉及性能修改）</td>"
        f"<td>{html.escape(CFG_REL)}<br/>tools/ddp_train.sh<br/>tools/train_spetr.py</td></tr>"
        "</tbody></table>"
    )
    if old_row not in doc:
        raise SystemExit("fa95a2a table row not found")
    doc = doc.replace(old_row, new_row)

    old_toc_tail = (
        "<li><a href='#item-fa95a2a'>9. 10. 10. 训练入口默认 expandable_segments（fa95a2a）</a></li>"
        "<li><a href='#soap-final'>附录：全部改动文件终稿索引</a></li></ol>"
    )
    new_toc_tail = (
        "<li><a href='#item-fa95a2a'>9. 训练入口默认 expandable_segments（fa95a2a）</a></li>"
        "<li><a href='#item-669a138'>10. 配置最终版本修改（669a138）</a></li>"
        "<li><a href='#soap-final'>附录：全部改动文件终稿索引</a></li></ol>"
    )
    if old_toc_tail not in doc:
        raise SystemExit("toc tail not found")
    doc = doc.replace(old_toc_tail, new_toc_tail)

    doc = replace_source_by_path(doc, "tools/ddp_train.sh", ddp)
    doc = replace_source_by_path(doc, CFG_REL, cfg)

    card = f"""<div class="card" id="item-669a138"><h3><span class="sha-chip">669a138</span>配置最终版本修改（不涉及性能修改）</h3><p class="field"><span class="label">适配对象</span>把昨日 876 步 GPU 合同配置与原名训练入口固化为仓库唯一正式版本，去掉诊断 overlay / 双配置。</p><p class="field"><span class="label">原理</span>性能算子（SOAP stale-Q、pin_memory、expandable_segments）已在此前提交落地；本提交只对齐正式 config 与 <code>tools/ddp_train.sh</code> / <code>tools/train_spetr.py</code> 的启动合同，使原名命令与 876 验收逻辑一致。</p><p class="field"><span class="label">为何可优化</span>不改算子公式。目的是交付可直接 <code>bash tools/ddp_train.sh tools/train_spetr.py &lt;正式config&gt;</code> 的最终版本，避免再选诊断副本导致性能对不上。</p><p class="field"><span class="label">做了什么</span>正式 config：<code>use_grid_mask=True</code>、恢复 lidar dropout/mask、<code>dropout_sd_prob=0.2</code>、注释 <code>lidar_type</code>、<code>optimizer_config</code> 改为 <code>Fp16OptimizerHookProtectGradNan</code>。入口：硬编码 <code>SOAP_STALE_Q_K=4</code>、透传 <code>MAX_ITERS</code>；<code>train_spetr.py</code> 对齐 GPU 合同 seed=0 / deterministic=False。</p><p class="field"><span class="label">涉及文件</span>{html.escape(CFG_REL)}；tools/ddp_train.sh；tools/train_spetr.py</p><div class="file-stack">
{fold("Diff — 669a138 配置最终版本修改（不涉及性能修改）", diff)}
{fold(f"Source — {CFG_REL}（整文件）", cfg)}
{fold("Source — tools/ddp_train.sh（整文件）", ddp)}
{fold("Source — tools/train_spetr.py（整文件）", train)}
</div></div>"""

    marker = "</div></div></section><section class=\"block\" id=\"soap-final\">"
    if marker not in doc:
        raise SystemExit("sec-items close marker not found")
    doc = doc.replace(marker, "</div></div>" + card + "</section><section class=\"block\" id=\"soap-final\">", 1)

    appendix_ddp = (
        '<details class="code-fold">\n'
        "  <summary>Source — tools/ddp_train.sh（整文件终稿）</summary>"
    )
    insert_train = (
        fold("Source — tools/train_spetr.py（整文件终稿）", train)
        + "\n"
        + appendix_ddp
    )
    if appendix_ddp not in doc:
        raise SystemExit("appendix ddp fold not found")
    # Avoid duplicating if rerun
    if "Source — tools/train_spetr.py（整文件终稿）" not in doc:
        doc = doc.replace(appendix_ddp, insert_train, 1)
        print("inserted train_spetr.py appendix fold")

    old_rb = """    <li>完整收益栈：开启 <code>SOAP_STALE_Q_K=4</code> + 现有 pin_memory 配置 + 训练脚本中的 expandable_segments。</li>
    <li>stale-Q 回退：不设置或 <code>SOAP_STALE_Q_K=0</code>。</li>
    <li>allocator 回退：去掉 <code>ddp_train.sh</code> 中对应 export 行。</li>
    <li>pin 回退：恢复 DataContainer 无 pin_memory 实现或关闭配置 pin（不推荐单独回退 pin）。</li>"""
    new_rb = """    <li>完整收益栈：原名 <code>tools/ddp_train.sh</code> 已默认 <code>SOAP_STALE_Q_K=4</code> + <code>expandable_segments</code>，配合配置 <code>pin_memory=True</code>。</li>
    <li>stale-Q 回退：将 <code>ddp_train.sh</code> 中 <code>SOAP_STALE_Q_K=4</code> 改为 <code>0</code>（算子默认仍为同步）。</li>
    <li>allocator 回退：去掉 <code>ddp_train.sh</code> 中 <code>expandable_segments</code> 那一行。</li>
    <li>pin 回退：恢复 DataContainer 无 pin_memory 实现或关闭配置 pin（不推荐单独回退 pin）。</li>
    <li>配置/入口合同回退：回退提交 <code>669a138</code>（正式 config、<code>ddp_train.sh</code>、<code>train_spetr.py</code>）。</li>"""
    if old_rb not in doc:
        raise SystemExit("rollback block not found")
    doc = doc.replace(old_rb, new_rb)

    HTML_PATH.write_text(doc, encoding="utf-8", newline="\n")
    print("wrote", HTML_PATH, "bytes", HTML_PATH.stat().st_size)
    for needle in ("item-669a138", "SOAP_STALE_Q_K=4", "train_spetr.py"):
        print(needle, doc.count(needle))


if __name__ == "__main__":
    main()
