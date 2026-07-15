#!/usr/bin/env python3
"""Reproduce the four panels of paper Figure 3 from a trained paper model.

Panels (a) and (b) use fixed test-set indices.  Panels (c) and (d) use the
out-of-distribution super-Gaussian EVDFs with m=3.5 and m=5 named in the
paper.  The script writes one combined 2x2 figure, all four individual panels,
and numerical panel metrics.
"""

from __future__ import print_function

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pyarrow.parquet as pq
import torch

from phyEqns.ots_superGaussian import OTS_spectrum_sg
from train_paper_fig3 import (
    FOURIER_MODES,
    INPUT_DIM,
    OUTPUT_DIM,
    PaperOTSModel,
    SPEED_OF_LIGHT,
    inverse_fourier_matrix,
)


ELEMENTARY_CHARGE = 1.602176634e-19
BOLTZMANN_CONSTANT = 1.380649e-23
ELECTRON_MASS = 9.10938356e-31
PROTON_MASS = 1.67262192369e-27


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-indices", default="0,1")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def choose_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_test_data(path):
    values = pq.read_table(str(path)).to_pandas().values
    values = values[:, : INPUT_DIM + OUTPUT_DIM].astype(np.float32, copy=False)
    metadata = values[0]
    physical = values[1:]
    return (
        torch.from_numpy(physical[:, :INPUT_DIM].copy()),
        torch.from_numpy(physical[:, INPUT_DIM:].copy()),
        torch.from_numpy(metadata[:INPUT_DIM].copy()),
        torch.from_numpy(metadata[INPUT_DIM : INPUT_DIM + FOURIER_MODES].copy()),
    )


def projected_super_gaussian(velocity, exponent, electron_temperature_eV=1370.0):
    """Numerically project the isotropic 3-D distribution used by OTS_spectrum_sg."""
    dtype = velocity.dtype
    thermal_velocity = math.sqrt(
        electron_temperature_eV * ELEMENTARY_CHARGE / ELECTRON_MASS
    )
    a_value = 2.0 * math.gamma(5.0 / exponent) / (
        3.0 * math.gamma(3.0 / exponent)
    )
    normalization = exponent / (4.0 * math.pi * math.gamma(3.0 / exponent))
    normalization *= (a_value / (2.0 * thermal_velocity**2)) ** 1.5
    radial = torch.linspace(
        0.0, 20.0 * thermal_velocity, 4000, dtype=dtype
    )
    speed = torch.sqrt(velocity[:, None].square() + radial[None, :].square())
    distribution_3d = normalization * torch.exp(
        -(
            a_value * speed.square() / (2.0 * thermal_velocity**2)
        ) ** (exponent / 2.0)
    )
    projected = 2.0 * math.pi * torch.trapz(
        radial[None, :] * distribution_3d, radial, dim=1
    )
    integral = torch.trapz(projected, velocity)
    if not torch.isfinite(integral) or integral <= 0.0:
        raise RuntimeError("Invalid projected super-Gaussian normalization")
    return projected / integral


def super_gaussian_spectrum(wavelengths_m, exponent):
    dtype = wavelengths_m.dtype
    scattering_angle = math.radians(123.0)
    scattering_direction = torch.tensor(
        [math.cos(scattering_angle), math.sin(scattering_angle), 0.0], dtype=dtype
    )
    incident_direction = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    polarization = torch.tensor([0.0, 0.0, 1.0], dtype=dtype)
    electron_temperature_k = 1370.0 * ELEMENTARY_CHARGE / BOLTZMANN_CONSTANT
    ion_temperature_k = 171.3 * ELEMENTARY_CHARGE / BOLTZMANN_CONSTANT
    zero_drift = torch.zeros(3, dtype=dtype)
    bands = (slice(0, 579), slice(579, 883), slice(883, 1224))
    spectra = []
    for band in bands:
        spectra.append(
            OTS_spectrum_sg(
                wavelengths_m[band],
                scattering_direction,
                incident_direction,
                polarization,
                351.0e-9,
                1.81e26,
                electron_temperature_k,
                1.0,
                0.55,
                torch.tensor(PROTON_MASS, dtype=dtype),
                6.0,
                0.45,
                torch.tensor(12.0 * PROTON_MASS, dtype=dtype),
                ion_temperature_k,
                1.0e-9,
                exponent,
                zero_drift,
                zero_drift,
                zero_drift,
            ).real
        )
    spectrum = torch.cat(spectra)
    if tuple(spectrum.shape) != (INPUT_DIM,) or not torch.isfinite(spectrum).all():
        raise RuntimeError("Invalid super-Gaussian spectrum for m={}".format(exponent))
    return spectrum.float()


def inverse_evdf(coefficients, k_grid, velocity):
    matrix = inverse_fourier_matrix(k_grid, velocity)
    return coefficients.matmul(matrix)


def panel_metrics(target, prediction, velocity):
    difference = prediction - target
    residual = difference.square().sum()
    total = (target - target.mean()).square().sum().clamp(min=1.0e-30)
    relative_l2 = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
        target
    ).clamp(min=1.0e-30)
    negative = torch.trapz(torch.clamp(-prediction, min=0.0), velocity)
    absolute = torch.trapz(prediction.abs(), velocity).clamp(min=1.0e-30)
    return {
        "mae": difference.abs().mean().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "r2": (1.0 - residual / total).item(),
        "relative_l2": relative_l2.item(),
        "negative_mass_fraction": (negative / absolute).item(),
    }


def style_panel(axis, velocity, truth, reconstruction, panel_label, title):
    axis.plot(
        velocity / SPEED_OF_LIGHT,
        truth,
        color="#0072BD",
        linewidth=2.2,
        label="Truth EVDF",
    )
    axis.plot(
        velocity / SPEED_OF_LIGHT,
        reconstruction,
        color="#D95319",
        linewidth=2.0,
        label="Reconstructed EVDF",
    )
    axis.set_xlim(-0.4, 0.4)
    visible = (velocity / SPEED_OF_LIGHT >= -0.4) & (
        velocity / SPEED_OF_LIGHT <= 0.4
    )
    visible_values = torch.cat((truth[visible], reconstruction[visible]))
    lower = min(0.0, visible_values.min().item() * 1.10)
    upper = max(2.0e-8, visible_values.max().item() * 1.10)
    axis.set_ylim(lower, upper)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-9, -9))
    axis.yaxis.set_major_formatter(formatter)
    axis.set_xlabel("Electron velocity [c]")
    axis.set_ylabel("EVDF [s m$^{-1}$]")
    axis.set_title(title)
    axis.text(
        0.03,
        0.95,
        "({})".format(panel_label),
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontweight="bold",
        fontsize=12,
    )
    axis.grid(True, alpha=0.18)


def save_individual(path, velocity, truth, prediction, panel_label, title):
    figure, axis = plt.subplots(figsize=(5.2, 4.3))
    style_panel(axis, velocity, truth, prediction, panel_label, title)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def self_test():
    velocity = torch.linspace(-0.8, 0.8, 1000, dtype=torch.float64) * SPEED_OF_LIGHT
    for exponent in (3.5, 5.0):
        evdf = projected_super_gaussian(velocity, exponent)
        integral = torch.trapz(evdf, velocity).item()
        assert abs(integral - 1.0) < 1.0e-6
        assert torch.isfinite(evdf).all()
    print("PAPER_FIG3_SELF_TEST_PASSED")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    indices = [int(item.strip()) for item in args.test_indices.split(",")]
    if len(indices) != 2 or min(indices) < 0:
        raise ValueError("--test-indices must contain exactly two non-negative indices")
    device = choose_device(args.device)
    try:
        checkpoint = torch.load(
            str(args.model), map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(str(args.model), map_location="cpu")
    if checkpoint.get("architecture") != [INPUT_DIM, 5000, 5000, OUTPUT_DIM]:
        raise ValueError("Checkpoint is not the paper Figure-3 architecture")
    model = PaperOTSModel()
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    test_spectra, test_coefficients, wavelengths_nm, k_grid = load_test_data(
        args.test_data
    )
    if max(indices) >= len(test_spectra):
        raise IndexError("Requested test index exceeds the test dataset")
    dense_velocity = (
        torch.linspace(-0.8, 0.8, 1000, dtype=torch.float32) * SPEED_OF_LIGHT
    )
    selected_spectra = test_spectra[indices]
    selected_coefficients = test_coefficients[indices]
    with torch.no_grad():
        selected_prediction = model(selected_spectra.to(device)).cpu()
    test_truth = inverse_evdf(selected_coefficients, k_grid, dense_velocity)
    test_reconstruction = inverse_evdf(selected_prediction, k_grid, dense_velocity)

    wavelengths_m = wavelengths_nm.double() * 1.0e-9
    super_exponents = (3.5, 5.0)
    super_spectra = torch.stack(
        [super_gaussian_spectrum(wavelengths_m, exponent) for exponent in super_exponents]
    )
    with torch.no_grad():
        super_prediction = model(super_spectra.to(device)).cpu()
    super_truth = torch.stack(
        [
            projected_super_gaussian(dense_velocity.double(), exponent).float()
            for exponent in super_exponents
        ]
    )
    super_reconstruction = inverse_evdf(super_prediction, k_grid, dense_velocity)

    truths = [test_truth[0], test_truth[1], super_truth[0], super_truth[1]]
    reconstructions = [
        test_reconstruction[0],
        test_reconstruction[1],
        super_reconstruction[0],
        super_reconstruction[1],
    ]
    labels = ("a", "b", "c", "d")
    titles = (
        "Test-set EVDF (index {})".format(indices[0]),
        "Test-set EVDF (index {})".format(indices[1]),
        "Super-Gaussian EVDF (m=3.5)",
        "Super-Gaussian EVDF (m=5)",
    )
    filenames = (
        "fig3a_test_evdf.png",
        "fig3b_test_evdf.png",
        "fig3c_supergaussian_m3p5.png",
        "fig3d_supergaussian_m5.png",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 8.2))
    for axis, truth, prediction, label, title, filename in zip(
        axes.flat, truths, reconstructions, labels, titles, filenames
    ):
        style_panel(axis, dense_velocity, truth, prediction, label, title)
        save_individual(
            args.output_dir / filename,
            dense_velocity,
            truth,
            prediction,
            label,
            title,
        )
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    figure.suptitle("Reproduction of paper Figure 3", y=0.995, fontsize=14)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    combined_path = args.output_dir / "paper_fig3_reproduction.png"
    figure.savefig(combined_path, dpi=300)
    plt.close(figure)

    panel_summary = {}
    for label, title, truth, prediction, filename in zip(
        labels, titles, truths, reconstructions, filenames
    ):
        panel_summary[label] = {
            "title": title,
            "file": filename,
            "metrics": panel_metrics(truth, prediction, dense_velocity),
        }
    summary = {
        "model": str(args.model),
        "test_data": str(args.test_data),
        "test_indices_zero_based": indices,
        "combined_figure": combined_path.name,
        "panels": panel_summary,
        "super_gaussian": {
            "exponents": list(super_exponents),
            "truth": "isotropic 3-D super-Gaussian projected to one velocity dimension",
            "spectrum": "phyEqns.ots_superGaussian.OTS_spectrum_sg",
            "electron_temperature_eV": 1370.0,
        },
        "limitations": [
            "The paper does not publish the random seeds or test-row identities for panels (a) and (b).",
            "The paper names m=3.5 and m=5 but does not publish every plotting parameter; the released source-code super-Gaussian convention is used.",
        ],
    }
    metrics_path = args.output_dir / "paper_fig3_metrics.json"
    metrics_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("PAPER_FIG3_RESULT {}".format(json.dumps(summary, sort_keys=True)), flush=True)


if __name__ == "__main__":
    main()
