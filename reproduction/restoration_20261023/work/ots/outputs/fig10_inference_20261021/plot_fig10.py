#!/usr/bin/env python3
"""Plot saved signed Fig.10 predictions; never run inference or alter VDFs."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

SPECIES = (("e", r"Electrons $e^-$"), ("H", r"Hydrogen $H^+$"),
           ("C", r"Carbon $C^{6+}$"))
DATASETS = ("fig10a", "fig10b", "fig10c")
COLORS = ("#2166ac", "#d6604d", "#4d9221")


def load_predictions(root):
    """Read existing NPZs, preserving all rows and values on their saved grids."""
    datasets = []
    for name in DATASETS:
        with np.load(root / name / "predictions.npz", allow_pickle=False) as src:
            data = {key: src[key] for key in src.files}
        time = data["time_paper_ns"]
        if time.ndim != 1 or time.size < 2 or not np.all(np.diff(time) > 0):
            raise ValueError(f"{name}: expected a strictly increasing time axis")
        if data["source_row_index"].shape != time.shape:
            raise ValueError(f"{name}: source row indices do not match time")
        for key, _ in SPECIES:
            velocity, vdf = data[f"{key}_velocity_m_s"], data[f"{key}_vdf"]
            if velocity.ndim != 1 or not np.all(np.diff(velocity) > 0):
                raise ValueError(f"{name}/{key}: expected increasing velocity")
            if vdf.shape != (time.size, velocity.size):
                raise ValueError(f"{name}/{key}: incompatible VDF shape")
            if not all(np.isfinite(x).all() for x in (time, velocity, vdf)):
                raise ValueError(f"{name}/{key}: nonfinite plotting data")
        datasets.append(data)
    return datasets


def edges(centers):
    """Cell boundaries only: no resampling or interpolation of VDF values."""
    midpoint = (centers[:-1] + centers[1:]) / 2
    return np.r_[2 * centers[0] - midpoint[0], midpoint,
                 2 * centers[-1] - midpoint[-1]]


def save_figure(fig, root, stem):
    for suffix in ("png", "pdf"):
        path = root / f"{stem}.{suffix}"
        with path.open("xb") as handle:
            fig.savefig(handle, format=suffix, dpi=180, facecolor="white")
        print(f"Saved {path}")
    plt.close(fig)


def heatmaps(datasets, root):
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), layout="constrained")
    fig.suptitle("Fig. 10: model predictions / no VDF ground truth\n"
                 "Signed VDFs; every saved time and velocity sample retained", fontsize=15)
    for col, (key, title) in enumerate(SPECIES):
        limit = max(float(np.abs(data[f"{key}_vdf"]).max()) for data in datasets)
        norm = Normalize(vmin=-(limit or 1), vmax=(limit or 1), clip=False)
        for row, (name, data) in enumerate(zip(DATASETS, datasets)):
            ax = axes[row, col]
            velocity = data[f"{key}_velocity_m_s"] / 1000
            image = ax.pcolormesh(edges(velocity), edges(data["time_paper_ns"]),
                                  data[f"{key}_vdf"], shading="flat", cmap="RdBu_r",
                                  norm=norm, rasterized=True)
            ax.set_title(f"Fig. 10({name[-1]}) | {title}", fontsize=11)
            if col == 0:
                ax.set_ylabel("Paper time (ns)")
            if row == 2:
                ax.set_xlabel("Velocity (km/s)")
            ax.tick_params(labelsize=9)
        cb = fig.colorbar(image, ax=axes[:, col], location="bottom", shrink=0.91,
                          pad=0.02, aspect=32)
        cb.set_label("Signed VDF (s/m); shared scale across a/b/c")
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
    fig.supxlabel("Native saved grids; cell display only, no time interpolation", fontsize=10)
    save_figure(fig, root, "fig10_signed_vdf_heatmaps")


def representative_curves(datasets, root):
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), layout="constrained")
    fig.suptitle("Fig. 10: model predictions / no VDF ground truth\n"
                 "Nearest saved rows to 2, 3 and 4 ns; no time interpolation", fontsize=15)
    selected = [np.array([np.argmin(np.abs(data["time_paper_ns"] - target))
                          for target in (2, 3, 4)]) for data in datasets]
    for col, (key, title) in enumerate(SPECIES):
        low = min(float(data[f"{key}_vdf"][rows].min())
                  for data, rows in zip(datasets, selected))
        high = max(float(data[f"{key}_vdf"][rows].max())
                   for data, rows in zip(datasets, selected))
        margin = 0.06 * (high - low or 1)
        for row, (name, data, rows) in enumerate(zip(DATASETS, datasets, selected)):
            ax = axes[row, col]
            velocity = data[f"{key}_velocity_m_s"] / 1000
            for target, index, color in zip((2, 3, 4), rows, COLORS):
                actual = data["time_paper_ns"][index]
                source = data["source_row_index"][index]
                label = f"{target} ns target: {actual:.4f} ns (row {source})"
                ax.plot(velocity, data[f"{key}_vdf"][index], color=color,
                        linewidth=1.15, label=label)
            ax.axhline(0, color="0.45", linewidth=0.6, zorder=0)
            ax.set(xlim=(velocity[0], velocity[-1]), ylim=(low - margin, high + margin),
                   title=f"Fig. 10({name[-1]}) | {title}", ylabel="Signed VDF (s/m)")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85)
            ax.grid(alpha=0.18)
            if row == 2:
                ax.set_xlabel("Velocity (km/s)")
    fig.supxlabel("Original signed values: no clipping, smoothing or VDF normalization", fontsize=10)
    save_figure(fig, root, "fig10_signed_vdf_representative_curves")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    root = parser.parse_args().results_dir.resolve()
    stems = ("fig10_signed_vdf_heatmaps", "fig10_signed_vdf_representative_curves")
    for stem in stems:
        for suffix in ("png", "pdf"):
            if (root / f"{stem}.{suffix}").exists():
                raise FileExistsError(f"Refusing to overwrite {root / (stem + '.' + suffix)}")
    datasets = load_predictions(root)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "path.simplify": False, "pdf.fonttype": 42})
    heatmaps(datasets, root)
    representative_curves(datasets, root)


if __name__ == "__main__":
    main()
