#!/usr/bin/env python3
"""Read saved Fig.10c arrays and visualize a time-row discrepancy; no inference."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fig10c_diagnostic_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fig10c_diagnostic_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator
import numpy as np

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
REPO = BASE.parents[1]
PRED = BASE / "results/fig10c/predictions.npz"
RAW = REPO / "outputs/fig10_input_alignment_20261021/fig10c/original/digitized_spectra.npz"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def edges(x):
    mid = (x[:-1] + x[1:]) / 2
    return np.r_[2*x[0]-mid[0], mid, 2*x[-1]-mid[-1]]


def main():
    protected = json.loads((BASE / "inference_plan.json").read_text())["protected_sha256"]
    for path, sha in protected.items():
        assert digest(path) == sha, path
    prior = {str(p): digest(p) for p in BASE.rglob("*")
             if p.is_file() and OUT not in p.parents}
    completion = json.loads((BASE / "completion.json").read_text())
    for rel, sha in completion["hashes"].items():
        assert digest(BASE / rel) == sha, rel
    with np.load(PRED, allow_pickle=False) as src:
        data = {k: src[k] for k in src.files}
    with np.load(RAW, allow_pickle=False) as src:
        raw = {k: src[k] for k in src.files}
    t, f = data["time_paper_ns"], data["e_vdf"]
    v = data["e_velocity_m_s"] / 1000
    wave = data["wavelength_nm"]
    x = data["spectra_peak_normalized"]
    assert np.array_equal(t, raw["time_paper_ns"])
    assert np.array_equal(data["source_row_index"], raw["source_row_index"])
    representatives = [int(np.argmin(abs(t-target))) for target in (2, 3, 4)]
    assert representatives == [78, 239, 400]
    font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "font.size": 11,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    colors = {77: "#0072B2", 78: "#D55E00", 79: "#CC79A7", 80: "#009E73"}
    fig, axs = plt.subplots(2, 2, figsize=(15, 10.5), layout="constrained")
    fig.suptitle("Fig.10(c) 电子 VDF：2 ns 附近的两行异常\n"
                 "直接读取已保存的 D4 推理结果与原始抠图像素", fontsize=18, fontweight="bold")
    ax = axs[0, 0]
    limit = float(np.max(abs(f))) * 1e8
    mesh = ax.pcolormesh(edges(v), edges(t), f*1e8, shading="flat", cmap="RdBu_r",
                         vmin=-limit, vmax=limit, antialiased=False, rasterized=True)
    ax.set(xlim=(-50000, 50000), ylim=(1.96, 2.04), xlabel="速度 (km/s)", ylabel="时间 (ns)",
           title="A  时间局部放大：保留每一行原值")
    for i in (77, 78, 79, 80):
        ax.axhline(t[i], color=colors[i], linestyle="--", linewidth=.9)
    cb = fig.colorbar(mesh, ax=ax, pad=.02)
    cb.set_label(r"有符号 VDF ($10^{-8}$ s/m)")
    ax = axs[0, 1]
    for i in (77, 78, 79, 80):
        label = f"{t[i]:.6f} ns" + ("（原代表行）" if i == 78 else "")
        ax.plot(v, f[i]*1e8, color=colors[i], lw=2 if i == 78 else 1.5,
                ls="--" if i == 79 else "-", label=label)
    ax.set(xlim=(-50000, 50000), xlabel="速度 (km/s)", ylabel=r"有符号 VDF ($10^{-8}$ s/m)",
           title="B  同一数组逐行取曲线：邻行明显更窄、更高")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=.2)
    ax = axs[1, 0]
    selected = np.flatnonzero((t >= 1.96) & (t <= 2.04))
    te = edges(t)
    ax.imshow(raw["rgb"][selected, :12], origin="lower", interpolation="nearest", aspect="auto",
              extent=(-.5, 11.5, te[selected[0]], te[selected[-1]+1]))
    ax.set(xlabel="原始光谱左边缘列索引（0 起计）", ylabel="时间 (ns)",
           title="C  原 NPZ 保存的 RGB：异常行边缘出现暗色标记")
    ax.set_xticks([0, 3, 6, 9, 11])
    for i in (78, 79):
        ax.axhline(t[i], color="white", linestyle=":", linewidth=.8)
    ax = axs[1, 1]
    for i in (77, 78, 80):
        ax.plot(wave, x[i], color=colors[i], lw=1.5, label=f"{t[i]:.6f} ns")
    ax.set(xlabel="波长 (nm)", ylabel="模型实际输入（已保存的逐谱归一化强度）",
           title="D  原代表行输入的左边缘出现约 0.70 的异常峰",
           xlim=(wave[0]-.025, wave[-1]+.025))
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)
    for ax in axs.flat:
        ax.tick_params(labelsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(6, integer=(ax is axs[1, 0])))
    fig.supxlabel("每行间隔约 6.22 ps；两行异常带总宽约 12.44 ps，在全时间图中很薄。\n"
                  "本图仅放大坐标与展示邻行；未滤波、补值、修改输入或重新推理。", fontsize=10)
    figures = []
    for ext in ("png", "pdf"):
        path = OUT / f"fig10c_2ns_diagnostic.{ext}"
        fig.savefig(path, dpi=200)
        figures.append(path)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(16, 5.1), layout="constrained", sharex=True, sharey=True)
    fig.suptitle("Fig.10(c) 电子 VDF：原 2 / 3 / 4 ns 代表行与附近保存行对照", fontsize=17,
                 fontweight="bold")
    neighbor_pairs = [(77, 80), (238, 240), (399, 401)]
    for target, i, pair, ax in zip((2, 3, 4), representatives, neighbor_pairs, axs):
        for j, color in zip(pair, ("#0072B2", "#009E73")):
            ax.plot(v, f[j]*1e8, color=color, lw=1.6, label=f"附近行 {t[j]:.6f} ns")
        ax.plot(v, f[i]*1e8, color="#D55E00", lw=2, ls="--", label=f"原代表 {t[i]:.6f} ns")
        ax.set(xlim=(-50000, 50000), xlabel="速度 (km/s)", title=f"{target} ns 附近")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=.2)
        ax.xaxis.set_major_locator(MaxNLocator(5))
    axs[0].set_ylabel(r"有符号 VDF ($10^{-8}$ s/m)")
    fig.supxlabel("全部曲线均取自原预测 NPZ；附近行仅作诊断对照，不作为修复结果或物理真值。\n"
                  "原代表行的光谱边缘含抠图异常，不能据其近乎重叠断言电子温度不变。", fontsize=10)
    for ext in ("png", "pdf"):
        path = OUT / f"fig10c_representatives_vs_neighbors.{ext}"
        fig.savefig(path, dpi=200)
        figures.append(path)
    plt.close(fig)

    records = []
    for i in (77, 78, 79, 80, 238, 239, 240, 399, 400, 401):
        half = np.flatnonzero(f[i] >= .5*f[i].max())
        records.append({"row_index": i, "source_row_index": int(data["source_row_index"][i]),
                        "actual_time_ns": float(t[i]), "e_vdf_peak_s_m": float(f[i].max()),
                        "half_max_grid_span_km_s": float(v[half[-1]]-v[half[0]]),
                        "normalized_edge_max_first_last_5": float(max(x[i,:5].max(), x[i,-5:].max())),
                        "normalized_interior_max_excluding_5_edges": float(x[i,5:-5].max())})
    for path, sha in {**protected, **prior}.items():
        assert digest(path) == sha, path
    result = {"status": "complete_read_only_diagnostic", "no_new_inference": True,
              "no_input_or_prediction_changes": True, "protected_files_verified": len(protected),
              "existing_inference_files_verified": len(prior), "records": records,
              "raw_and_predictions_sha256": {str(p): digest(p) for p in (RAW, PRED)},
              "figures_sha256": {p.name: digest(p) for p in figures},
              "script_sha256": digest(Path(__file__)),
              "width_note": "Span of native grid samples above half maximum, not a temperature fit.",
              "causality_limit": "Edge anomalies and prediction changes correspond; no corrected-input counterfactual was run."}
    (OUT / "diagnostic_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({k: result[k] for k in ("status", "protected_files_verified", "existing_inference_files_verified")}))


if __name__ == "__main__":
    main()
