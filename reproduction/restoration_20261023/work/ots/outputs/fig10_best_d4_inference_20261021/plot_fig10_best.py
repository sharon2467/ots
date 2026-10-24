#!/usr/bin/env python3
"""Render saved D4_s0 Fig.10 VDFs without inference or numerical postprocessing."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fig10_best_d4_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import numpy as np

SPECIES = (("e", r"Electrons  $e^-$"), ("H", r"Hydrogen  $H^+$"),
           ("C", r"Carbon  $C^{6+}$"))
GROUPS = ("fig10a", "fig10b", "fig10c")
SOURCES = ("Experimental spectrum", "Digitized simulation spectra", "Digitized simulation spectra")
COLORS = ("#0072B2", "#D55E00", "#009E73")
TARGETS = (2., 3., 4.)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edges(x):
    """Display-cell boundaries; never resample the underlying values."""
    mid = (x[:-1] + x[1:]) / 2
    return np.r_[2*x[0]-mid[0], mid, 2*x[-1]-mid[-1]]


def load(root):
    datasets, records = [], {}
    for name, source in zip(GROUPS, SOURCES):
        path = root / name / "predictions.npz"
        with np.load(path, allow_pickle=False) as src:
            keys = ["time_paper_ns", "source_row_index"]
            keys += [f"{s}_{field}" for s, _ in SPECIES
                     for field in ("velocity_m_s", "vdf")]
            data = {key: src[key] for key in keys}
        t = data["time_paper_ns"]
        if t.ndim != 1 or t.size < 2 or not np.all(np.diff(t) > 0):
            raise ValueError(f"{name}: time must increase strictly")
        if data["source_row_index"].shape != t.shape:
            raise ValueError(f"{name}: incompatible source indices")
        rows = np.array([np.argmin(np.abs(t-target)) for target in TARGETS])
        data["representative_indices"] = rows
        record = {"file": str(path), "sha256": sha256(path), "source_context": source,
                  "time_rows": int(t.size), "time_range_ns": t[[0, -1]].tolist(),
                  "representative_rows": [
                      {"target_time_ns": target, "saved_row_index": int(i),
                       "actual_time_ns": float(t[i]),
                       "source_row_index": int(data["source_row_index"][i])}
                      for target, i in zip(TARGETS, rows)], "species": {}}
        for s, _ in SPECIES:
            v, f = data[f"{s}_velocity_m_s"], data[f"{s}_vdf"]
            if (v.ndim != 1 or v.size < 2 or not np.all(np.diff(v) > 0)
                    or f.shape != (t.size, v.size)):
                raise ValueError(f"{name}/{s}: invalid velocity/VDF shapes")
            if not all(np.isfinite(x).all() for x in (t, v, f)):
                raise ValueError(f"{name}/{s}: nonfinite saved data")
            for x in (v, f):
                x.flags.writeable = False
            record["species"][s] = {"shape": list(f.shape), "min_s_m": float(f.min()),
                                    "max_s_m": float(f.max()),
                                    "velocity_range_km_s": (v[[0, -1]]/1000).tolist()}
        datasets.append(data)
        records[name] = record
    return datasets, records


def sci_formatter():
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    return formatter


def figure(datasets, norms, curve_limits, kind, group, focus):
    indices = list(range(3)) if group is None else [group]
    overview = group is None
    fig, axes = plt.subplots(len(indices), 3, squeeze=False,
                             figsize=(16, 11.2) if overview else (16, 5.3),
                             layout="constrained")
    mode = "Full velocity range" if focus is None else "Focused velocity view, full grids retained in NPZ"
    title = "Time-resolved signed VDFs" if kind == "heatmaps" else "Signed VDFs at representative times"
    fig.suptitle(f"Fig. 10  |  {title}\nD4_s0 direct model predictions  ·  {mode}",
                 fontsize=17, fontweight="semibold")
    for row, idx in enumerate(indices):
        data = datasets[idx]
        t = data["time_paper_ns"]
        for col, (s, species_title) in enumerate(SPECIES):
            ax = axes[row, col]
            v = data[f"{s}_velocity_m_s"]/1000
            f = data[f"{s}_vdf"]
            subtitle = f"({GROUPS[idx][-1]}) {SOURCES[idx]}"
            ax.set_title(f"{species_title}\n{subtitle}", fontsize=11.5, pad=10)
            ax.set_xlabel(r"Velocity (km s$^{-1}$)")
            if kind == "heatmaps":
                mesh = ax.pcolormesh(edges(v), edges(t), f, shading="flat",
                                     cmap="RdBu_r", norm=norms[s], rasterized=True,
                                     antialiased=False)
                ax.set_ylim(t[0], t[-1])
                ax.set_ylabel("Time (ns)" if col == 0 else "")
                if not overview:
                    cb = fig.colorbar(mesh, ax=ax, location="bottom", pad=.04,
                                      shrink=.96, aspect=35, format=sci_formatter())
                    cb.set_label(r"Signed VDF (s m$^{-1}$)", fontsize=10)
            else:
                ax.axhline(0, color="#77808c", linewidth=.75, zorder=0)
                for j, (index, color) in enumerate(zip(data["representative_indices"], COLORS)):
                    label = (f"{t[index]:.4f} ns  ·  source {data['source_row_index'][index]}"
                             if col == 0 else "_nolegend_")
                    ax.plot(v, f[index], color=color, linewidth=1.55, label=label)
                ax.set_ylim(*curve_limits[s])
                ax.set_ylabel(r"Signed VDF (s m$^{-1}$)")
                ax.yaxis.set_major_formatter(sci_formatter())
                ax.grid(color="#d9dfe5", linewidth=.6, alpha=.7)
                ax.set_axisbelow(True)
                if col == 0:
                    ax.legend(loc="upper right", fontsize=8.2, frameon=True,
                              facecolor="white", edgecolor="#dce1e6", framealpha=.94)
            ax.set_xlim(v[0], v[-1])
            if focus is not None:
                ax.set_xlim(-focus[col], focus[col])
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.tick_params(labelsize=9.5, color="#9ca6b2")
        if overview and kind == "heatmaps" and row == len(indices)-1:
            for col, (s, _) in enumerate(SPECIES):
                cb = fig.colorbar(plt.cm.ScalarMappable(norm=norms[s], cmap="RdBu_r"),
                                  ax=axes[:, col], location="bottom", pad=.025,
                                  shrink=.96, aspect=35, format=sci_formatter())
                cb.set_label(r"Signed VDF (s m$^{-1}$)  ·  shared a/b/c scale", fontsize=10)
    note = ("Every saved time row; native velocity cells; signed linear color scale spanning all extrema."
            if kind == "heatmaps" else
            "Nearest saved rows to 2, 3 and 4 ns; actual times and source row indices in legends.")
    if focus is not None:
        note += "  Focus changes axis limits only; see full-range companion."
    fig.supxlabel(note + "\nNo smoothing, clipping, VDF renormalization or temporal interpolation; no VDF ground truth.",
                  fontsize=9.2, color="#485563")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "plots")
    parser.add_argument("--focus-km-s", type=float, nargs=3, metavar=("E", "H", "C"),
                        help="Optional additional symmetric velocity views; modifies displayed limits only")
    parser.add_argument("--overview-only", action="store_true", help="Render only 3-by-3 overviews")
    parser.add_argument("--focus-only", action="store_true",
                        help="Render only focused companions to an already supplied full-range collection")
    args = parser.parse_args()
    if args.focus_km_s is not None and (not np.isfinite(args.focus_km_s).all()
                                       or min(args.focus_km_s) <= 0):
        parser.error("Focus half-widths must be finite and positive")
    if args.focus_only and args.focus_km_s is None:
        parser.error("--focus-only requires --focus-km-s")
    datasets, records = load(args.results_dir.resolve())
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "plotting_manifest.json").exists() or (out / "fig10_best_d4_all_plots.pdf").exists():
        raise FileExistsError(f"Refusing to overwrite an existing completed plot collection: {out}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#9ca6b2", "axes.titlecolor": "#182636",
                         "axes.labelcolor": "#28394a", "path.simplify": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white"})
    norms, curve_limits = {}, {}
    for s, _ in SPECIES:
        limit = max(float(np.abs(d[f"{s}_vdf"]).max()) for d in datasets)
        norms[s] = Normalize(-limit if limit else -1, limit if limit else 1, clip=False)
        low = min(float(d[f"{s}_vdf"][d["representative_indices"]].min()) for d in datasets)
        high = max(float(d[f"{s}_vdf"][d["representative_indices"]].max()) for d in datasets)
        margin = .075*(high-low or 1)
        curve_limits[s] = [min(low, 0)-margin, max(high, 0)+margin]
    manifest = {"model": "depth_fig_d4_s0", "inputs": records,
                "plotting_policy": {"all_time_rows_retained": True, "signed_values_retained": True,
                                    "smoothing": False, "VDF_clipping": False,
                                    "VDF_renormalization": False, "temporal_interpolation": False,
                                    "color_scale": "linear symmetric; full absolute extrema per species across a/b/c",
                                    "heatmap_rendering": "native pcolormesh flat cells; no data resampling",
                                    "representative_rule": "first argmin(abs(saved time - target)), targets [2,3,4] ns",
                                    "velocity_units": "km/s", "VDF_units": "s/m",
                                    "focus_half_widths_km_s": args.focus_km_s,
                                    "overview_only": args.overview_only,
                                    "focus_only_companion": args.focus_only,
                                    "VDF_ground_truth_available": False},
                "shared_species_color_limits_s_m": {s: [norms[s].vmin, norms[s].vmax] for s, _ in SPECIES},
                "shared_species_representative_y_limits_s_m": curve_limits, "outputs": {}}
    views = [] if args.focus_only else [("full_range", None)]
    if args.focus_km_s is not None:
        views.append(("focused", args.focus_km_s))
    combined = out / "fig10_best_d4_all_plots.pdf"
    with PdfPages(combined) as pdf:
        for view, focus in views:
            for kind in ("heatmaps", "representative_curves"):
                for group in ((None,) if args.overview_only else (None, 0, 1, 2)):
                    stem = f"{'fig10abc' if group is None else GROUPS[group]}_d4_{kind}_{view}"
                    fig = figure(datasets, norms, curve_limits, kind, group, focus)
                    for suffix in ("png", "pdf", "svg"):
                        path = out / f"{stem}.{suffix}"
                        with path.open("xb") as handle:
                            fig.savefig(handle, format=suffix, dpi=200)
                        manifest["outputs"][path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
                    pdf.savefig(fig, dpi=200)
                    plt.close(fig)
                    print(f"Saved {stem}", flush=True)
    manifest["outputs"][combined.name] = {"sha256": sha256(combined), "bytes": combined.stat().st_size}
    for name, record in records.items():
        if sha256(Path(record["file"])) != record["sha256"]:
            raise RuntimeError(f"Input NPZ changed while plotting: {name}")
    with (out / "plotting_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
