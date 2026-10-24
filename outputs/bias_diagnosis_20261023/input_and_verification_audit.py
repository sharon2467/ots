"""Read-only input audit and independent check of saved diagnostic arrays.

Writes only its own JSON report. No training/model inference or dataset creation.
"""
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
from pathlib import Path
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CODE = ROOT / "OTS_code_share/NN_hotE_large_lmRange"
sys.path.insert(0, str(CODE))
from data import _physical_config_fig1_fig11, ELEMENTARY_CHARGE, SPECIES_BLOCKS
from phyEqns.ots_full_fourier import OTS_spectrum_PV
from spectral_fitting import _gaussian_convolve

torch.set_num_threads(1)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    result = {"time_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Original uploaded b/c arrays, current sampler defaults, and independent arithmetic/PV verification. Latest processed inputs and D4 are unavailable."}
    original = {
        "b": "/workspace/attachments/39e99b16-ce5c-44f2-ad62-5e0643f5584b/digitized_spectra.npz",
        "c": "/workspace/attachments/221c92bf-8637-482a-b2db-e03441434eb5/digitized_spectra.npz",
    }
    result["digitization"] = {}
    for panel, path in original.items():
        with np.load(path) as d:
            y, index, x, t = (d[k] for k in ("intensity", "palette_index", "wavelength_nm", "time_paper_ns"))
            n = len(d["palette_rgb"])
            reconstructed = 1 - index / (n - 1)
            assert np.array_equal(y, reconstructed)
            inner = np.abs(x - 263.3) < .75
            rec = {"input_path": path, "sha256": sha(path), "palette_samples": n,
                   "exact_mapping": "intensity = 1 - palette_index/(480-1)",
                   "intensity_quantum": 1 / (n - 1),
                   "diagnostic_region_nm": "abs(wavelength - 263.3) < 0.75; read-only subset, no new processed dataset",
                   "rows": []}
            for wanted in (1.6, 2., 3., 4.):
                j = int(np.argmin(np.abs(t - wanted)))
                peak = float(y[j, inner].max())
                rec["rows"].append({"row": j, "requested_time_ns": wanted, "actual_time_ns": float(t[j]),
                    "central_peak": peak, "central_min": float(y[j, inner].min()),
                    "quantum_over_central_peak": 1/(n-1)/peak,
                    "constant_central_row": bool(np.ptp(y[j, inner]) == 0)})
            result["digitization"][panel] = rec
    config = _physical_config_fig1_fig11()
    anchor = Path("/workspace/ots-cloud-setup/data/paper_experimental_points.csv")
    with anchor.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["figure"] == "fig11" and r["shot_id"] == "20170811020"]
    support = {}
    for parameter, extra, factor in (("T_e_ev", [209., 1392.], "electron_temperature_factor"),
                                     ("T_i_ev", [78., 912.], "ion_temperature_factor")):
        pool = [float(r["value"]) for r in rows if r["parameter"] == parameter]
        all_values = pool + extra
        lo, hi = config["sampling"][factor]
        support[parameter] = {"fig11_count": len(pool), "fig11_min_ev": min(pool), "fig11_max_ev": max(pool),
            "all_anchor_min_ev": min(all_values), "all_anchor_max_ev": max(all_values),
            "default_global_target_range_ev": [min(all_values)*lo, max(all_values)*hi]}
    result["current_default_sampling_support"] = {"anchor_sha256": sha(anchor), "temperatures": support,
        "electron_flow_prior_mean_m_s": float(np.mean(config["sampling"]["electron_velocity_m_s"])),
        "limitation": "Default source configuration; absent D4 generation manifest prevents validating all run-specific overrides."}
    controlled_dir = HERE / "representation_audit"
    truth = json.loads((controlled_dir / "controlled_results.json").read_text())
    arrays = np.load(controlled_dir / "controlled_arrays.npz")
    verified = []
    for state in truth["states"]:
        idx = state["state_index"]
        for species, block in SPECIES_BLOCKS:
            v = arrays[f"state{idx}_{species}_v"]
            for suffix, label, field in (("original", "original", "actual_temperature_ev"),
                                         ("raw_inverse", "raw_signed", "temperature_ev"),
                                         ("clipnorm", "clipped_normalized", "actual_temperature_ev")):
                f = arrays[f"state{idx}_{species}_{suffix}"]
                norm = np.trapezoid(f, v)
                mean = np.trapezoid(v*f, v)/norm
                temperature = config[block]["mass_kg"]/ELEMENTARY_CHARGE*np.trapezoid((v-mean)**2*f, v)/norm
                expected = state["species"][species][label][field]
                assert np.isclose(temperature, expected, rtol=2e-12, atol=1e-9)
        a, b = arrays[f"state{idx}_original_PV"], arrays[f"state{idx}_clipnorm_PV"]
        l2 = np.linalg.norm(b-a)/np.linalg.norm(a)
        assert np.isclose(l2, state["PV_clipnorm_all_species_relative_l2_vs_original_PV"], atol=1e-14)
        verified.append({"state": idx, "spectrum_l2": float(l2), "all_nine_moments_verified": True})
    # One independent public-PV call verifies the saved 209/78 original spectrum.
    kwargs = {f"{prefix}_{s}": torch.from_numpy(arrays[f"state1_{s}_{suffix}"])
              for s in ("e", "i1", "i2") for prefix, suffix in (("v", "v"), ("f", "original"))}
    x = arrays["wavelength_nm"]
    pv = OTS_spectrum_PV(torch.from_numpy(x)*1e-9, config["ss"], config["k_sh"], config["E_i0"],
        config["lm_0"], 3.5e26, config["Z_i1"], config["N_i1"], config["m_i1"], config["Z_i2"],
        config["N_i2"], config["m_i2"], config["V"], **kwargs, u_e=config["u_e"], u_i1=config["u_i1"],
        u_i2=config["u_i2"], fourier_modes=config["fourier_modes"]).numpy()
    delta = float(np.max(np.abs(pv-arrays["state1_original_PV"])))
    assert delta < 1e-12
    sens = np.load(HERE / "measurement_sensitivity/spectra.npz")
    broad = _gaussian_convolve(x, pv, .057)
    broad /= np.max(np.abs(broad))
    response_delta = float(np.max(np.abs(broad-sens["state1_gaussian057"])))
    assert response_delta < 1e-12
    old_hashes = json.loads((controlled_dir / "results.json").read_text())["sha256_before"]
    current_hashes = {p: sha(ROOT/p) for p in old_hashes}
    assert current_hashes == old_hashes
    result["independent_verification"] = {"states": verified, "new_PV_calls": 1,
        "state1_PV_max_absolute_difference": delta, "existing_convolution_max_absolute_difference": response_delta,
        "protected_hashes_match_first_audit": True, "sha256": current_hashes}
    (HERE / "input_and_verification_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({"sampling_support": result["current_default_sampling_support"], "verification": result["independent_verification"]}, indent=2))

if __name__ == "__main__":
    main()
