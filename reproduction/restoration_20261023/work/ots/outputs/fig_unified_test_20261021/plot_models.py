#!/usr/bin/env python3
"""Plot every saved Fig model using its unified TEST macro relative-L2.

Usage: python plot_models.py [--results results.json] [--output-dir DIR]
Metric values in results.json are fractions (0.06 means 6%), never percent.
This script only reads evaluation results; it does not load/train models.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fig-unified-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "fig-unified-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def scalar_text(value, *, integer=False):
    if value is None:
        return "未记录"
    if isinstance(value, (list, tuple)):
        return "/".join(scalar_text(item, integer=integer) for item in value)
    return f"{int(value):,}" if integer else (f"{value:g}" if isinstance(value, float) else str(value))


def architecture_lines(model):
    widths = model["hidden_sizes"]
    if widths and isinstance(widths[0], list):
        architectures = [tuple(item) for item in widths]
        if len(set(architectures)) == 1:
            return [f"{len(architectures)}成员 × {len(architectures[0])}隐藏层",
                    "宽 " + "×".join(map(str, architectures[0]))]
        lines = [f"组合：{len(architectures)}个两层模型" if all(len(arch) == 2 for arch in architectures)
                 else "组合：各成员隐藏层/宽度"]
        for arch in architectures:
            lines.append(f"{len(arch)}层 " + "×".join(map(str, arch)))
        return lines
    width_text = "宽 " + "×".join(map(str, widths))
    if len(widths) > 2:
        return [f"隐藏层数 {len(widths)}", "宽 " + "×".join(map(str, widths[:2])),
                "  ×" + "×".join(map(str, widths[2:]))]
    return [f"隐藏层数 {len(widths)}", width_text]


def annotation(model):
    lines = [f"参数 {scalar_text(model['parameter_count'], integer=True)}"]
    lines.extend(architecture_lines(model))
    lines.extend([
        "训练池 " + scalar_text(model["training_pool_samples"], integer=True),
        "优化 " + scalar_text(model["optimization_samples"], integer=True),
    ])
    # Keep the essential metadata legible in each bar's own column. Long
    # ensemble seed/epoch lists remain available in the source JSON.
    if not isinstance(model.get("seed"), list) and model.get("seed") is not None:
        lines.append("seed " + scalar_text(model["seed"]))
    if not isinstance(model.get("best_epoch"), list) and model.get("best_epoch") is not None:
        lines.append("最佳轮 " + scalar_text(model["best_epoch"]))
    lr = model.get("learning_rate")
    raw = model.get("raw_loss_weight")
    if lr is not None:
        lines.append("lr " + scalar_text(lr))
    if raw is not None:
        lines.append("raw loss 权重 " + scalar_text(raw))
    dropout = model.get("dropout")
    if dropout is not None and dropout != 0.02 and dropout != [0.02, 0.02]:
        lines.append("dropout " + scalar_text(model["dropout"]))
    return "\n".join(lines)


def model_label(model, rank):
    label = str(model.get("display_name") or model["model_id"])
    if model.get("kind") == "ensemble":
        label = "26k_ensemble_50-50"
    elif "smoke" in label:
        label = "smoke_16"
    # Identifiers stay verbatim except for line breaks, which allow all bars
    # to remain in one horizontal row at a readable font size.
    return f"#{rank:02d}\n" + "\n".join(textwrap.wrap(label, width=21, break_long_words=True,
                                                    break_on_hyphens=False))


def validate(payload):
    models = payload["models"]
    if not models:
        raise ValueError("results.json has no models")
    ids = [item["model_id"] for item in models]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate model_id in results.json")
    for model in models:
        error = float(model["macro_relative_l2"]["mean"])
        if not math.isfinite(error) or error < 0:
            raise ValueError(f"Invalid unified TEST macro relative-L2: {model['model_id']}")
        for key in ("parameter_count", "training_pool_samples", "optimization_samples", "hidden_sizes"):
            if model.get(key) is None:
                raise ValueError(f"Missing {key}: {model['model_id']}")
        if not isinstance(model["parameter_count"], int) or model["parameter_count"] < 1:
            raise ValueError(f"parameter_count must be an exact positive integer: {model['model_id']}")
    if int(payload["dataset"]["physical_samples"]) < 1:
        raise ValueError("dataset.physical_samples must be positive")
    return sorted(models, key=lambda item: (-float(item["macro_relative_l2"]["mean"]), item["model_id"]))


def depth_key(model):
    widths = model["hidden_sizes"]
    if widths and isinstance(widths[0], list):
        depths = sorted(set(len(item) for item in widths))
        return "/".join(map(str, depths))
    return str(len(widths))


DEPTH_COLORS = {"2": "#3b82b8", "3": "#9670b4", "4": "#249c91"}


def draw_chart(payload, models, output_dir, dpi, *, ranks, name, title, scale_max=None, scale_note="", pdf_pages=None, headroom_lines=None):
    errors = [100 * float(model["macro_relative_l2"]["mean"]) for model in models]
    depths = sorted(set(depth_key(model) for model in payload["models"]))
    depth_colors = {depth: DEPTH_COLORS.get(depth, "#c49353") for depth in depths}
    notes = [annotation(model) for model in models]
    max_lines = max(note.count("\n") + 1 for note in notes)
    if headroom_lines is not None:
        max_lines = max(max_lines, headroom_lines)
    fig_width = max(15, 1.62 * len(models) + 1.5)
    fig_height = max(11, 8.5 + 0.16 * max_lines)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.235, top=0.80)
    annotation_fraction = min(0.55, 0.030 * (max_lines + 2))
    maximum = max(errors) if scale_max is None else scale_max
    upper = (maximum or 1.0) / (1.0 - annotation_fraction - 0.05)
    ax.set_ylim(0, upper)
    bars = ax.bar(range(len(models)), errors, width=0.69,
                  color=[depth_colors[depth_key(m)] for m in models],
                  edgecolor="#334155", linewidth=0.6, zorder=3)
    for index, (model, bar, error, note) in enumerate(zip(models, bars, errors, notes)):
        if model.get("kind") == "ensemble":
            bar.set_hatch("///")
        ax.annotate(f"{error:.4f}%", (index, error), xytext=(0, 7),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color="#172033")
        ax.annotate(note, (index, error), xytext=(0, 28),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.7, linespacing=1.48, color="#334155")
    ax.set_xlim(-0.65, len(models) - 0.35)
    ax.set_xticks(range(len(models)), [model_label(model, ranks[model["model_id"]]) for model in models], fontsize=9)
    ax.tick_params(axis="x", length=0, pad=13)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("测试宏平均 VDF 相对 L2 误差 (%)", fontsize=15, labelpad=16)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=3))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.grid(axis="y", color="#dce3eb", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("#64748b")
    ax.spines["left"].set_color("#64748b")
    fig.text(0.055, 0.945, title, fontsize=24, weight="bold", va="top")
    detail_note = " · #01 smoke 见总览" if name.startswith("fig_models_detail") and ranks[models[0]["model_id"]] > 1 else ""
    fig.text(0.055, 0.889,
             f"同一现有测试集 · {payload['dataset']['physical_samples']:,} 个物理样本 · "
             f"全局 #{ranks[models[0]['model_id']]:02d}–#{ranks[models[-1]['model_id']]:02d} · "
             "按测试宏平均误差从高到低排列（越低越好）" + detail_note,
             fontsize=13, color="#475569", va="top")
    handles = [Patch(facecolor=depth_colors[depth], edgecolor="#334155", label=f"{depth} 隐藏层") for depth in depths]
    if any(model.get("kind") == "ensemble" for model in payload["models"]):
        handles.append(Patch(facecolor="white", edgecolor="#334155", hatch="///", label="集成模型"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.052, 0.856),
               ncol=min(len(handles), 8), frameon=False, fontsize=10)
    sha = str(payload["dataset"].get("sha256", "未记录"))
    footer = (
        "柱高 = 各测试样本的 e / H / C 相对 L2 误差等权平均后，再对全部测试样本取平均；单位为百分比。\n"
        "参数为精确可训练参数量；训练池包含优化与验证样本。集成参数为成员总量，样本数按结果记录展示。\n"
        "共同配置：LayerNorm、dropout 0.02；差异在柱上注明。纵轴从 0 开始，无截断。" + scale_note + "\n"
        f"测试集 SHA256: {sha}"
    )
    fig.text(0.055, 0.052, footer, fontsize=10, color="#475569", va="bottom", linespacing=1.7)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "pdf", "svg"):
        path = output_dir / f"{name}.{extension}"
        fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.3)
        paths.append(path)
    if pdf_pages is not None:
        pdf_pages.savefig(fig, facecolor="white", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return paths


def plot(payload, output_dir, dpi, page_size=10):
    models = validate(payload)
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Chinese font unavailable: {FONT_PATH}")
    font_manager.fontManager.addfont(str(FONT_PATH))
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=str(FONT_PATH)).get_name(),
        "axes.unicode_minus": False, "pdf.fonttype": 42,
        "svg.fonttype": "none", "font.size": 11,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "fig_models_all_pages.pdf"
    with PdfPages(combined_path) as combined:
        paths = render_pages(payload, models, output_dir, dpi, page_size, combined)
    return paths + [combined_path]


def render_pages(payload, models, output_dir, dpi, page_size, combined):
    ranks = {model["model_id"]: rank for rank, model in enumerate(models, 1)}
    paths = draw_chart(payload, models, output_dir, dpi, ranks=ranks,
                       name="fig_models_unified_test_macro_sorted",
                       title="已完成 Fig 模型 · 统一理论测试集",
                       scale_note="总览包含全部模型。", pdf_pages=combined)
    # The overview always includes every model at its actual full height.
    # If one model is far above all others, retain it in the complete overview;
    # remaining panels use one common zero-based scale and retain global ranks.
    isolated = len(models) > 1 and float(models[0]["macro_relative_l2"]["mean"]) > 4 * float(models[1]["macro_relative_l2"]["mean"])
    detail_models = models[1:] if isolated else models
    common_max = max(100 * float(model["macro_relative_l2"]["mean"]) for model in detail_models)
    common_lines = max(annotation(model).count("\n") + 1 for model in detail_models)
    # Balance page sizes, avoiding a final page with just one or two bars.
    page_count = math.ceil(len(detail_models) / page_size)
    per_page = math.ceil(len(detail_models) / page_count)
    for index in range(page_count):
        subset = detail_models[index * per_page:(index + 1) * per_page]
        paths += draw_chart(payload, subset, output_dir, dpi, ranks=ranks,
                            name=f"fig_models_detail_page{index + 1:02d}",
                            title=f"Fig 模型明细 · 第 {index + 1}/{page_count} 页",
                            scale_max=common_max,
                            scale_note="所有明细分页使用共同纵轴；" + ("#01 smoke 完整展示于总览。" if isolated else "完整保留全部模型。"),
                            pdf_pages=combined, headroom_lines=common_lines)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--page-size", type=int, default=10, help="Maximum bars per detail page (default: 10)")
    args = parser.parse_args()
    with args.results.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if args.page_size < 1:
        parser.error("--page-size must be positive")
    for path in plot(payload, args.output_dir or args.results.parent, args.dpi, args.page_size):
        print(path)


if __name__ == "__main__":
    main()
