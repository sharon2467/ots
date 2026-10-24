"""Fit experimental OTS rows; pilot and full are separate explicit runs."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

import spectral_fitting as sf


def load_config(path: Path) -> dict:
    """Validate explicit physics, sampling and fitting settings."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for name in ("input_dir", "output_dir"):
        config[name] = str(Path(config[name]).resolve())
    root, output = Path(config["input_dir"]), Path(config["output_dir"])
    if not root.is_dir() or output.exists() or output.is_relative_to(root):
        raise ValueError("Input must exist; output must be new and outside the input tree")
    if config["mode"] not in ("pilot", "full"):
        raise ValueError("mode must be pilot or full")
    ids = [s["shot_id"] for s in config["shots"]]
    if sorted(ids) != ["20160909153", "20160919163", "20170811020"]:
        raise ValueError("Exactly the three specified shots are required")
    physics = config["physics"]
    for key in ("probe_wavelength_nm", "ion1_charge", "ion1_mass_amu", "ion2_charge", "ion2_mass_amu"):
        if not np.isfinite(physics[key]) or physics[key] <= 0:
            raise ValueError(f"Invalid physical parameter: {key}")
    if not 0 < physics["scattering_angle_deg"] < 180 or not 0 < physics["ion1_number_fraction"] < 1:
        raise ValueError("Invalid angle or ion number fraction")
    guesses = np.asarray(config["temperature_guesses_ev"], dtype=float)
    if (guesses.shape != (3, 3) or not np.isfinite(guesses).all()
            or np.any(guesses <= sf._LOWER_BOUNDS[:3]) or np.any(guesses >= sf._UPPER_BOUNDS[:3])):
        raise ValueError("Three interior temperature guesses are required")
    for key, value in config["thresholds"].items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid threshold: {key}")
    for key in ("bound_fraction", "temperature_relative_difference"):
        if config["thresholds"][key] >= 1:
            raise ValueError(f"Threshold must be below one: {key}")
    if config["thresholds"]["candidate_sse_ratio"] < 1:
        raise ValueError("Candidate SSE ratio must be at least one")
    for shot in config["shots"]:
        for key in ("time_range_fig_ns", "wavelength_range_nm"):
            span = np.asarray(shot[key], dtype=float)
            if span.shape != (2,) or not np.isfinite(span).all() or span[0] >= span[1]:
                raise ValueError(f"Invalid range: {key}")
        density = np.asarray(shot["electron_density_cm3"], dtype=float)
        if density.shape not in ((), (1024,)) or not np.isfinite(density).all() or np.any(density <= 0):
            raise ValueError("Density must be positive scalar or 1024 original-row values")
        if not isinstance(shot["density_basis"], str) or not shot["density_basis"].strip():
            raise ValueError("Density provenance is required")
        if not np.isfinite(shot["instrument_fwhm_nm"]) or shot["instrument_fwhm_nm"] <= 0:
            raise ValueError("Positive FWHM is required")
    return config


def load_shot(input_dir: Path, shot_id: str, settings: dict) -> dict:
    """Verify the stored shot and select documented contiguous wavelengths/rows."""
    root = Path(input_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    shot = next(s for s in manifest["shots"] if s["shot_id"] == shot_id)
    hashes = {}
    for name in ("ordered_arrays.npz", "metadata.json"):
        record = next(r for r in shot["outputs"] if r["path"] == f"{shot_id}/{name}")
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Input file escapes input root")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Input hash mismatch: {path}")
        hashes[str(path)] = digest
    with np.load(root / shot_id / "ordered_arrays.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    x, t, values = arrays["wavelength_nm"], arrays["time_fig_ns"], arrays["intensity"]
    if values.shape != (1024, 1344) or x.shape != (1344,) or t.shape != (1024,):
        raise ValueError("Unexpected experimental array shapes")
    for a in (x, t, values):
        if a.dtype.kind not in "iuf" or not np.isfinite(a).all():
            raise ValueError("Input arrays must be finite real numbers")
    if not np.all(np.diff(x) > 0) or not np.all(np.diff(t) > 0):
        raise ValueError("Coordinates must be strictly increasing")
    if not np.allclose(np.diff(x), np.diff(x)[0], rtol=1e-8, atol=1e-12):
        raise ValueError("Existing convolution requires a uniform wavelength grid")
    for key, count in (("source_row_index", 1024), ("source_col_index", 1344)):
        index = arrays[key]
        if index.shape != (count,) or index.dtype.kind not in "iu" or not np.array_equal(np.sort(index), np.arange(count)):
            raise ValueError(f"Invalid source index: {key}")
    selected = []
    for axis, key in ((t, "time_range_fig_ns"), (x, "wavelength_range_nm")):
        low, high = settings[key]
        if low < axis[0] or high > axis[-1]:
            raise ValueError(f"Requested range exceeds input: {key}")
        index = np.flatnonzero((axis >= low) & (axis <= high))
        if not index.size:
            raise ValueError(f"Empty selection: {key}")
        selected.append(index)
    rows, columns = selected
    full_row_count = len(rows)
    if settings["mode"] == "pilot":
        rows = rows[np.unique(np.rint(np.linspace(0, len(rows) - 1, min(5, len(rows)))).astype(int))]
    if len(columns) < 16:
        raise ValueError("At least 16 wavelengths are required")
    density = np.asarray(settings["electron_density_cm3"], dtype=float)
    densities = np.full(len(rows), density.item()) if density.ndim == 0 else density[arrays["source_row_index"][rows]]
    return {"shot_id": shot_id, "settings": settings, "input_hashes": hashes,
            "wavelength_nm": x[columns], "time_fig_ns": t[rows],
            "source_row_index": arrays["source_row_index"][rows],
            "source_col_index": arrays["source_col_index"][columns],
            "intensity": values[np.ix_(rows, columns)], "densities_cm3": densities,
            "full_row_count": full_row_count}


def prepare_row(data: dict, row_index: int) -> dict:
    """Normalize one spectrum and retain its scalar normalization and identity."""
    original = data["intensity"][row_index].astype(np.float64)
    row = {"shot_id": data["shot_id"], "source_row_index": int(data["source_row_index"][row_index]),
           "time_fig_ns": float(data["time_fig_ns"][row_index]),
           "electron_density_cm3": float(data["densities_cm3"][row_index]),
           "wavelength_nm": data["wavelength_nm"], "scale": None,
           "measured": np.full(original.shape, np.nan), "status": "invalid_input", "initial_guesses": []}
    if not np.isfinite(original).all():
        return row
    row["scale"] = float(np.max(np.abs(original)))
    if row["scale"] == 0:
        return row
    y = original / row["scale"]
    n_edge = max(1, int(np.ceil(len(y) * 0.1)))
    background = np.array([np.median(y[:n_edge]), np.median(y[-n_edge:])])
    amplitude = float(np.max(y) - np.mean(background))
    eps = 1e-4
    for temperatures in data["settings"]["temperature_guesses_ev"]:
        guess = np.array([*temperatures, 0.0, 0.0, amplitude, *background], dtype=float)
        guess[5:] = np.clip(guess[5:], sf._LOWER_BOUNDS[5:] + eps, sf._UPPER_BOUNDS[5:] - eps)
        row["initial_guesses"].append(guess.tolist())
    row.update(measured=y, status="ready")
    return row


def fit_row(row: dict, physics: dict, thresholds: dict) -> dict:
    """Try three fixed starts and retain failed/ambiguous results without repair."""
    result = {key: row[key] for key in ("shot_id", "source_row_index", "time_fig_ns", "electron_density_cm3", "scale")}
    result.update(status="invalid_input", parameters=None, candidates=[], flags=[], metrics=None,
                  fitted=np.full(row["measured"].shape, np.nan), residual=np.full(row["measured"].shape, np.nan))
    if row["status"] == "invalid_input":
        return result
    successful = []
    for index, guess in enumerate(row["initial_guesses"]):
        started = time.perf_counter()
        candidate = {"initial_guess": guess, "candidate_index": index}
        try:
            params = sf.fit_spectrum(row["wavelength_nm"], row["measured"],
                                     row["electron_density_cm3"], initial_guess=guess, **physics)
            vector = np.array([params[key] for key in sf._PARAMETER_NAMES])
            fitted = sf._fit_model(vector, row["wavelength_nm"], row["electron_density_cm3"], **physics)
            residual = fitted - row["measured"]
            sse = float(residual @ residual)
            if not np.isfinite(sse) or not np.isfinite(vector).all():
                raise ValueError("Non-finite fit result")
            candidate.update(status="success", parameters=params, sse=sse)
            successful.append((sse, index, params, vector, fitted, residual))
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            candidate.update(status="failed", parameters=None, error=f"{type(exc).__name__}: {exc}")
        candidate["elapsed_seconds"] = time.perf_counter() - started
        result["candidates"].append(candidate)
        print(f"  {row['shot_id']} row {row['source_row_index']} start {index+1}: "
              f"{candidate['status']} {candidate['elapsed_seconds']:.2f}s", flush=True)
    if not successful:
        result["status"] = "fit_failed"
        return result
    sse, index, params, vector, fitted, residual = min(successful, key=lambda item: item[0])
    amplitude = params["A"]
    r_value = float(np.sqrt(sse / len(residual)) / amplitude) if amplitude > thresholds["amplitude_floor"] else None
    distance = np.minimum(vector - sf._LOWER_BOUNDS, sf._UPPER_BOUNDS - vector) / (sf._UPPER_BOUNDS - sf._LOWER_BOUNDS)
    flags = []
    if r_value is None:
        flags.append("negligible_amplitude")
    if np.any(distance < thresholds["bound_fraction"]):
        flags.append("at_bound")
    if r_value is not None and r_value > thresholds["max_r"]:
        flags.append("large_residual")
    if amplitude < thresholds["weak_signal_sigma"] * np.std(residual):
        flags.append("weak_signal")
    for other in successful:
        if (other[1] != index and other[0] <= thresholds["candidate_sse_ratio"] * sse + thresholds["candidate_sse_atol"]
                and np.any(np.abs(other[3][:3] - vector[:3]) > thresholds["temperature_relative_difference"] * vector[:3])):
            flags.append("nonunique_temperature")
            break
    result.update(status="review" if flags else "ok", parameters=params, flags=flags,
                  selected_candidate=index, fitted=fitted, residual=residual,
                  metrics={"sse": sse, "R": r_value, "residual_mean": float(np.mean(residual)),
                           "residual_std": float(np.std(residual)), "bound_distance_fraction": distance.tolist()},
                  electron_relative_drift_cm_s=params["u_ek_cm_s"] - params["u_ik_cm_s"])
    return result


def plot_shot(data: dict, results: list[dict], output_dir: Path) -> None:
    """Show sampled rows, all pilot spectra and clearly marked parameter quality."""
    x, times = data["wavelength_nm"], data["time_fig_ns"]
    measured = np.stack([prepare_row(data, i)["measured"] for i in range(len(results))])
    fitted = np.stack([r["fitted"] for r in results])
    residual = np.stack([r["residual"] for r in results])
    finite = np.concatenate([a[np.isfinite(a)] for a in (measured, fitted)])
    low, high = (float(finite.min()), float(finite.max())) if finite.size else (0, 1)
    error_values = np.abs(residual[np.isfinite(residual)])
    error_max = max(float(error_values.max()), 1e-12) if error_values.size else 1.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, values, label in zip(axes, (measured, fitted, residual), ("Measured", "Fitted", "Residual: fit - measured")):
        is_residual = label.startswith("Residual")
        mesh = ax.imshow(values, aspect="auto", interpolation="nearest", origin="upper",
                         extent=[x[0], x[-1], len(times)-0.5, -0.5],
                         cmap="RdBu_r" if is_residual else "viridis",
                         vmin=-error_max if is_residual else low, vmax=error_max if is_residual else high)
        tick_ids = np.unique(np.rint(np.linspace(0, len(times)-1, min(8, len(times)))).astype(int))
        ax.set_yticks(tick_ids, [f"{times[i]:.4f}" for i in tick_ids])
        ax.set(title=label, xlabel="Wavelength (nm)", ylabel="Sampled FIG time (ns)")
        fig.colorbar(mesh, ax=ax, label="Normalized intensity")
    fig.suptitle(f"{data['shot_id']} | {len(times)} sampled rows; no time interpolation")
    fig.savefig(output_dir / "overview.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    groups = (("T_e_ev", "T_i1_ev", "T_i2_ev"), ("u_ik_cm_s", "u_ek_cm_s"), ("R",))
    colors = ("#1565c0", "#c62828", "#2e7d32")
    labels = {"T_e_ev": "Te", "T_i1_ev": "TH", "T_i2_ev": "TC", "u_ik_cm_s": "uik", "u_ek_cm_s": "uek", "R": "R"}
    for ax, keys, ylabel in zip(axes, groups, ("Temperature (eV)", "Velocity (cm/s)", "R")):
        for k, key in enumerate(keys):
            for status, marker in (("ok", "o"), ("review", "x")):
                chosen = [i for i, r in enumerate(results) if r["status"] == status and (key != "R" or r["metrics"]["R"] is not None)]
                values = [(results[i]["metrics"][key] if key == "R" else results[i]["parameters"][key]) for i in chosen]
                ax.scatter(times[chosen], values, marker=marker, color=colors[k], label=f"{labels[key]} / {status}")
        ax.set_ylabel(ylabel); ax.grid(alpha=0.2); ax.legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("FIG time (ns)")
    fig.suptitle(f"{data['shot_id']} | independent TH / TC | density: {data['settings']['density_basis']}")
    fig.savefig(output_dir / "parameters_vs_time.png", dpi=150); plt.close(fig)
    if data["settings"]["mode"] == "pilot":
        fig, axes = plt.subplots(len(results), 2, figsize=(13, 3*len(results)), squeeze=False, constrained_layout=True)
        for i, result in enumerate(results):
            axes[i, 0].plot(x, measured[i], color="0.35", lw=0.7, label="Measured")
            axes[i, 0].plot(x, fitted[i], color="#d84315", lw=1.3, label="Fit")
            axes[i, 1].plot(x, residual[i], color="#1565c0", lw=0.7)
            axes[i, 1].axhline(0, color="0.5", lw=0.5)
            axes[i, 0].set_title(f"row {result['source_row_index']}, t={times[i]:.4f} ns | {result['status']}")
            axes[i, 1].set_title(", ".join(result["flags"]) or result["status"], fontsize=9)
            axes[i, 0].set_ylabel("Normalized intensity"); axes[i, 1].set_ylabel("Residual")
            for ax in axes[i]: ax.set_xlabel("Wavelength (nm)"); ax.grid(alpha=0.15)
            axes[i, 0].legend(fontsize=8)
        fig.suptitle(f"{data['shot_id']} | all pilot rows; scale saved separately")
        fig.savefig(output_dir / "pilot_spectra.png", dpi=130); plt.close(fig)
    ok = [i for i, r in enumerate(results) if r["status"] == "ok"]
    if ok:
        selected = np.unique(np.rint(np.linspace(0, len(ok)-1, min(3, len(ok)))).astype(int))
        for position in selected:
            i = ok[position]
            physics = {**data["settings"]["physics"], "instrument_fwhm_nm": data["settings"]["instrument_fwhm_nm"]}
            title = (f"{data['shot_id']} | FIG t={times[i]:.4f} ns | ne={data['densities_cm3'][i]:.3g} cm^-3\n"
                     "Independent TH / TC; local sensitivity, not joint confidence")
            figure = sf.plot_fit_result(x, measured[i], results[i]["parameters"], data["densities_cm3"][i], title=title, **physics)
            figure.axes[0].lines[0].set_label("Measured spectrum")
            figure.axes[0].legend(frameon=False)
            destination = output_dir / "diagnostics" / f"row_{results[i]['source_row_index']}.png"
            destination.parent.mkdir(exist_ok=True)
            figure.savefig(destination, dpi=150, bbox_inches="tight"); plt.close(figure)


def write_results(output_dir: Path, config: dict, data: dict, results: list[dict]) -> dict:
    """Persist rows and spectra, then verify scales, residuals and source mapping."""
    output_dir.mkdir(parents=True, exist_ok=False)
    if [r["source_row_index"] for r in results] != data["source_row_index"].tolist():
        raise ValueError("Result row identities do not match requested rows")
    serial = [{k: v for k, v in r.items() if k not in ("fitted", "residual")} for r in results]
    payload = {"shot_id": data["shot_id"], "settings": data["settings"], "rows": serial}
    result_path = output_dir / "fit_results.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    arrays = {"wavelength_nm": data["wavelength_nm"], "time_fig_ns": data["time_fig_ns"],
              "source_row_index": data["source_row_index"], "source_col_index": data["source_col_index"],
              "scale": np.array([r["scale"] if r["scale"] is not None else np.nan for r in results]),
              "measured_normalized": np.stack([prepare_row(data, i)["measured"] for i in range(len(results))]),
              "fitted_normalized": np.stack([r["fitted"] for r in results]),
              "residual_normalized": np.stack([r["residual"] for r in results]),
              "fit_valid": np.array([r["status"] in ("ok", "review") for r in results])}
    np.savez_compressed(output_dir / "fitted_spectra.npz", **arrays)
    with np.load(output_dir / "fitted_spectra.npz", allow_pickle=False) as saved:
        for key, value in arrays.items():
            if saved[key].dtype != value.dtype or not np.array_equal(saved[key], value, equal_nan=True):
                raise ValueError(f"Readback mismatch: {key}")
        valid = saved["fit_valid"]
        if not np.isfinite(saved["fitted_normalized"][valid]).all():
            raise ValueError("Valid fitted rows contain non-finite values")
        np.testing.assert_allclose(saved["residual_normalized"][valid], saved["fitted_normalized"][valid]-saved["measured_normalized"][valid], atol=0, rtol=0)
        valid_input = np.isfinite(saved["scale"]) & (saved["scale"] > 0)
        np.testing.assert_allclose(saved["scale"][valid_input, None]*saved["measured_normalized"][valid_input], data["intensity"][valid_input].astype(float), rtol=1e-14, atol=1e-12)
    reloaded = json.loads(result_path.read_text(encoding="utf-8"))
    if reloaded != payload:
        raise ValueError("JSON readback mismatch")
    plot_shot(data, results, output_dir)
    files = [{"path": p.relative_to(output_dir.parent).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(output_dir.rglob("*")) if p.is_file()]
    counts = {s: sum(r["status"] == s for r in results) for s in ("ok", "review", "invalid_input", "fit_failed")}
    return {"shot_id": data["shot_id"], "row_count": len(results), "full_row_count": data["full_row_count"],
            "state_counts": counts, "fit_seconds": sum(c["elapsed_seconds"] for r in results for c in r["candidates"]),
            "readback_verified": True, "outputs": files}


def main(argv: list[str] | None = None) -> int:
    """Run only the configured mode; a pilot never starts a full run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    torch.set_num_threads(1)
    data_sets = []
    for shot in config["shots"]:
        settings = {**shot, "physics": config["physics"], "mode": config["mode"],
                    "temperature_guesses_ev": config["temperature_guesses_ev"]}
        data_sets.append(load_shot(Path(config["input_dir"]), shot["shot_id"], settings))
    source_dir = Path(__file__).resolve().parent
    watched = [Path(__file__).resolve(), source_dir/"spectral_fitting.py", source_dir/"phyEqns/ots_full_fourier.py",
               Path(config["input_dir"])/"manifest.json", source_dir.parents[1]/"OTS_data/exp_ref_para.pdf"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    for data in data_sets: hashes.update(data["input_hashes"])
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=False)
    (output/"config.json").write_text(json.dumps(config, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    manifest = {"status": "running", "mode": config["mode"], "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "input_and_code_hashes": hashes, "shots": [], "errors": [],
                "bounds": {"names": list(sf._PARAMETER_NAMES), "lower": sf._LOWER_BOUNDS.tolist(), "upper": sf._UPPER_BOUNDS.tolist()},
                "environment": {"python": sys.version, "executable": sys.executable, "numpy": np.__version__,
                                "scipy": scipy.__version__, "torch": torch.__version__, "matplotlib": matplotlib.__version__,
                                "torch_threads": torch.get_num_threads()}}
    manifest_path = output/"manifest.json"
    started = time.perf_counter()
    try:
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        for data in data_sets:
            results = []
            physics = {**config["physics"], "instrument_fwhm_nm": data["settings"]["instrument_fwhm_nm"]}
            for i in range(len(data["time_fig_ns"])):
                row = prepare_row(data, i)
                result = fit_row(row, physics, config["thresholds"])
                results.append(result)
                print(f"{data['shot_id']} {i+1}/{len(data['time_fig_ns'])}: {result['status']} {result['flags']}", flush=True)
            manifest["shots"].append(write_results(output/data["shot_id"], config, data, results))
            manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        for path, digest in hashes.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Input or code changed during run: {path}")
        manifest.update(status="completed", inputs_and_code_unchanged=True,
                        awaiting_pilot_review=config["mode"] == "pilot")
    except Exception as exc:
        manifest["status"] = "failed"; manifest["errors"].append(f"{type(exc).__name__}: {exc}")
        print(manifest["errors"][-1], file=sys.stderr)
    finally:
        manifest["elapsed_seconds"] = time.perf_counter()-started
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
