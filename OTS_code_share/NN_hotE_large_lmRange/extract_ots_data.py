"""Export complete time-wavelength arrays from OTS MATLAB v5 figures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io import loadmat


INSTRUMENT_FWHM_NM = {
    "20160909153": 0.065,
    "20160919163": 0.065,
    "20170811020": 0.057,
}


def inventory_sources(root: Path) -> list[dict]:
    """List and hash regular files without following links or reparse points."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    records = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in list(dirs):
            path = Path(directory) / name
            if path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & 0x400:
                dirs.remove(name)
        for name in files:
            path = Path(directory) / name
            if path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & 0x400:
                continue
            if not path.is_file():
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            records.append({
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "suffix": path.suffix.lower(),
                "sha256": digest.hexdigest(),
            })
    return sorted(records, key=lambda record: record["path"])


def find_spectral_images(node: object, path: str = "hgS_070000") -> list[dict]:
    """Find axis images whose numeric dimensions and physical labels agree."""
    children = list(np.asarray(getattr(node, "children", []), dtype=object).flat)
    candidates = []
    if getattr(node, "type", None) == "axes":
        strings = []
        for child in children:
            if getattr(child, "type", None) == "text":
                value = getattr(getattr(child, "properties", None), "String", [])
                strings.extend(str(item) for item in np.asarray(value, dtype=object).flat)
        x_labels = [s for s in strings if "nm" in s.lower() and
                    ("lambda" in s.lower() or "wavelength" in s.lower())]
        y_labels = [s for s in strings if "time" in s.lower() and "ns" in s.lower()]
        for index, child in enumerate(children):
            if getattr(child, "type", None) != "image":
                continue
            props = getattr(child, "properties", None)
            x = np.asarray(getattr(props, "XData", None))
            y = np.asarray(getattr(props, "YData", None))
            c = np.asarray(getattr(props, "CData", None))
            if (x.ndim == y.ndim == 1 and c.ndim == 2
                    and c.shape == (y.size, x.size) and x_labels and y_labels):
                candidates.append({
                    "wavelength_nm_raw": x.copy(),
                    "time_fig_ns_raw": y.copy(),
                    "intensity_raw": c.copy(),
                    "object_path": f"{path}.children[{index}]",
                    "x_label": x_labels[0], "y_label": y_labels[0],
                    "title": [s for s in strings if s not in x_labels + y_labels],
                    "display": {
                        key: np.asarray(getattr(node.properties, key)).tolist()
                        for key in ("XLim", "YLim", "CLim", "YDir")
                        if hasattr(node.properties, key)
                    },
                })
    for index, child in enumerate(children):
        if getattr(child, "type", None) not in ("scribe.colorbar", "scribe.scribeaxes"):
            candidates.extend(find_spectral_images(child, f"{path}.children[{index}]"))
    return candidates


def read_fig(path: Path) -> dict:
    """Read one unambiguous spectral image, preserving every value and dtype."""
    path = Path(path)
    with path.open("rb") as stream:
        if not stream.read(20).startswith(b"MATLAB 5.0 MAT-file"):
            raise ValueError(f"Expected MATLAB v5 figure: {path}")
    contents = loadmat(path, variable_names=["hgS_070000"],
                       struct_as_record=False, squeeze_me=True)
    candidates = find_spectral_images(contents.get("hgS_070000"))
    if len(candidates) != 1:
        raise ValueError(f"{path.name}: expected one spectral image, found "
                         f"{len(candidates)}: {[c['object_path'] for c in candidates]}")
    raw = candidates[0]
    title_ids = re.findall(r"shot\s*-\s*(\d+)", " ".join(raw["title"]), flags=re.I)
    if title_ids != [path.stem]:
        raise ValueError(f"{path.name}: shot title does not match filename: {raw['title']}")
    for key in ("wavelength_nm_raw", "time_fig_ns_raw", "intensity_raw"):
        array = raw[key]
        if array.dtype.kind not in "iuf" or not array.size or not np.isfinite(array).all():
            raise ValueError(f"{path.name}: {key} must contain finite real numeric data")
    for key in ("wavelength_nm_raw", "time_fig_ns_raw"):
        delta = np.diff(raw[key].astype(np.float64))
        if raw[key].size < 2 or not (np.all(delta > 0) or np.all(delta < 0)):
            raise ValueError(f"{path.name}: {key} must be strictly monotonic")
    raw["shot_id"] = path.stem
    return raw


def canonicalize_axes(raw: dict) -> dict:
    """Order both coordinates increasingly using only reversible permutations."""
    ordered = {}
    for raw_key, key, index_key in (
        ("wavelength_nm_raw", "wavelength_nm", "source_col_index"),
        ("time_fig_ns_raw", "time_fig_ns", "source_row_index"),
    ):
        axis = raw[raw_key]
        delta = np.diff(axis.astype(np.float64))
        if axis.ndim != 1 or axis.size < 2 or not np.isfinite(axis).all():
            raise ValueError(f"Invalid coordinate: {key}")
        index = np.arange(axis.size, dtype=np.int64)
        if np.all(delta < 0):
            index = index[::-1].copy()
        elif not np.all(delta > 0):
            raise ValueError(f"Non-monotonic coordinate: {key}")
        ordered[key] = axis[index].copy()
        ordered[index_key] = index
    ordered["intensity"] = raw["intensity_raw"][np.ix_(
        ordered["source_row_index"], ordered["source_col_index"])]
    return ordered


def write_shot_bundle(output_dir: Path, raw: dict, ordered: dict, source: dict) -> dict:
    """Export, plot and read back a complete shot; return hashed output records."""
    output_dir = Path(output_dir)
    row, col = ordered["source_row_index"], ordered["source_col_index"]
    expected = raw["intensity_raw"][np.ix_(row, col)]
    if (ordered["intensity"].dtype != expected.dtype
            or not np.array_equal(ordered["intensity"], expected)):
        raise ValueError("Intensity does not agree with source indices")
    for raw_key, key, index in (
        ("wavelength_nm_raw", "wavelength_nm", col),
        ("time_fig_ns_raw", "time_fig_ns", row),
    ):
        if (not np.array_equal(ordered[key], raw[raw_key][index])
                or not np.all(np.diff(ordered[key].astype(np.float64)) > 0)):
            raise ValueError(f"Coordinate does not agree with source indices: {key}")
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_arrays = {key: raw[key] for key in
                  ("wavelength_nm_raw", "time_fig_ns_raw", "intensity_raw")}
    for name, arrays in (("raw_arrays.npz", raw_arrays), ("ordered_arrays.npz", ordered)):
        np.savez_compressed(output_dir / name, **arrays)
        with np.load(output_dir / name, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays):
                raise ValueError(f"Readback fields mismatch: {name}")
            for key, array in arrays.items():
                if saved[key].dtype != array.dtype or not np.array_equal(saved[key], array):
                    raise ValueError(f"Readback mismatch: {name}/{key}")
    intensity = raw["intensity_raw"]
    metadata = {
        "shot_id": raw["shot_id"], "source": source,
        "object_path": raw["object_path"], "title": raw["title"],
        "axis_labels": {"wavelength": raw["x_label"], "time": raw["y_label"]},
        "units": {"wavelength": "nm", "time": "ns", "intensity": "fig_native"},
        "shape": list(intensity.shape), "intensity_dtype": str(intensity.dtype),
        "display": raw["display"], "intensity_processing_history": "unknown",
        "time_reference": "fig", "paper_time_offset_ns": None,
        "coordinate_reordering": {
            "rows_reversed": bool(row[0] != 0), "columns_reversed": bool(col[0] != 0),
            "index_base": 0,
        },
        "instrument_fwhm_nm": source.get("instrument_fwhm_nm"),
        "statistics": {
            "finite": bool(np.isfinite(intensity).all()),
            "minimum": float(intensity.min()), "maximum": float(intensity.max()),
            "negative_count": int(np.count_nonzero(intensity < 0)),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    with (output_dir / "metadata.json").open(encoding="utf-8") as stream:
        if json.load(stream) != metadata:
            raise ValueError("Metadata readback mismatch")
    with np.load(output_dir / "ordered_arrays.npz", allow_pickle=False) as saved:
        exported_x = saved["wavelength_nm"]
        exported_y = saved["time_fig_ns"]
        exported_intensity = saved["intensity"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True,
                             sharex=True, sharey=True)
    try:
        limits = raw["display"].get("CLim", [float(intensity.min()), float(intensity.max())])
        for ax, x, y, values, title in (
            (axes[0], raw["wavelength_nm_raw"], raw["time_fig_ns_raw"],
             intensity, "Original FIG view (reconstructed)"),
            (axes[1], exported_x, exported_y, exported_intensity,
             "Exported NPZ (reloaded)"),
        ):
            mesh = ax.pcolormesh(x, y, values, shading="nearest", cmap="jet",
                                 vmin=limits[0], vmax=limits[1], rasterized=True)
            ax.set(xlabel="Wavelength (nm)", title=title)
        axes[0].set_ylabel("FIG time (ns)")
        axes[0].set_xlim(raw["display"].get("XLim", [exported_x[0], exported_x[-1]]))
        time_limits = raw["display"].get("YLim", [exported_y[0], exported_y[-1]])
        axes[0].set_ylim(time_limits[::-1] if raw["display"].get("YDir") == "reverse"
                         else time_limits)
        fig.colorbar(mesh, ax=axes.tolist(), label="FIG native intensity", extend="both")
        fig.suptitle(f"Shot {raw['shot_id']} | original FIG axis and color limits\n"
                     "Full arrays retained in NPZ; display range only", fontsize=12)
        fig.savefig(output_dir / "quicklook.png", dpi=160)
    finally:
        plt.close(fig)
    return {"shot_id": raw["shot_id"], "shape": list(intensity.shape),
            "intensity_values": int(intensity.size), "readback_verified": True,
            "outputs": inventory_sources(output_dir)}


def main(argv: list[str] | None = None) -> int:
    """Extract all input FIG files into a new standalone output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    if output_dir.exists() or output_dir.is_relative_to(input_dir):
        parser.error("Output must be a new directory outside the input directory")
    sources = inventory_sources(input_dir)
    figures = [s for s in sources if s["suffix"] == ".fig"]
    if not figures or len({Path(s["path"]).stem for s in figures}) != len(figures):
        parser.error("Input must contain FIG files with unique shot filenames")
    notes = next((s for s in sources if s["path"] == "Notes.txt"), None)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1, "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir), "output_dir": str(output_dir),
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "source_files": sources, "shots": [], "errors": [],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    try:
        for source in figures:
            raw = read_fig(input_dir / source["path"])
            ordered = canonicalize_axes(raw)
            provenance = dict(source)
            provenance["instrument_fwhm_nm"] = (
                INSTRUMENT_FWHM_NM.get(raw["shot_id"]) if notes else None)
            provenance["instrument_fwhm_source"] = (
                {"path": notes["path"], "sha256": notes["sha256"],
                 "basis": "shot mapping verified in SPEC_OTS_DATA_EXTRACTION_PLAN.md (2026-09-14)"}
                if notes and raw["shot_id"] in INSTRUMENT_FWHM_NM else None)
            result = write_shot_bundle(output_dir / raw["shot_id"], raw, ordered, provenance)
            for item in result["outputs"]:
                item["path"] = f"{raw['shot_id']}/{item['path']}"
            manifest["shots"].append(result)
            print(f"{raw['shot_id']}: {tuple(result['shape'])}, readback verified", flush=True)
        if inventory_sources(input_dir) != sources:
            raise ValueError("Input file inventory or hashes changed during extraction")
        manifest["source_files_unchanged"] = True
        manifest["total_intensity_values"] = sum(s["intensity_values"] for s in manifest["shots"])
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["errors"].append(f"{type(exc).__name__}: {exc}")
        print(manifest["errors"][-1], file=sys.stderr)
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                             allow_nan=False) + "\n", encoding="utf-8")
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
