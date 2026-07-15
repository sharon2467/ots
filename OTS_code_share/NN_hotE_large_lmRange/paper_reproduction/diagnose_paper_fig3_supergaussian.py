#!/usr/bin/env python3
"""Compare analytic and Fourier OTS spectra for Figure-3 super-Gaussians.

The paper training set is generated through ``OTS_spectrum_fft``, while the
released super-Gaussian helper uses a separate analytic response.  This script
feeds both spectra to the same frozen checkpoint so that forward-model mismatch
is not confused with neural-network generalization error.
"""

from __future__ import print_function

import argparse
import json
from pathlib import Path
import sys

TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parent
for search_path in (PROJECT_ROOT, TASK_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import torch

from generate_paper_dataset import physics_parameters, spectrum_band
from phyEqns.ots_fourier import ft_x_n
from reproduce_paper_fig3 import (
    INPUT_DIM,
    PaperOTSModel,
    SPEED_OF_LIGHT,
    choose_device,
    inverse_evdf,
    load_test_data,
    panel_metrics,
    projected_super_gaussian,
    super_gaussian_spectrum,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    return parser.parse_args()


def load_checkpoint(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def relative_l2(reference, candidate):
    denominator = torch.linalg.vector_norm(reference).clamp(min=1.0e-30)
    return (torch.linalg.vector_norm(candidate - reference) / denominator).item()


def fourier_super_gaussian_spectrum(wavelengths_m, velocity, exponent):
    truth = projected_super_gaussian(velocity, exponent)
    k_grid, coefficients = ft_x_n(velocity, truth, 80)
    physics = physics_parameters(dtype=torch.float64)
    bands = (slice(0, 579), slice(579, 883), slice(883, INPUT_DIM))
    spectra = torch.cat(
        tuple(
            spectrum_band(wavelengths_m[band], k_grid, coefficients, physics)
            for band in bands
        )
    )
    packed_coefficients = torch.cat((coefficients.real, coefficients.imag)).float()
    return spectra.real.float(), k_grid.float(), packed_coefficients


def main():
    args = parse_args()
    device = choose_device(args.device)
    _, _, wavelengths_nm, metadata_k = load_test_data(args.test_data)
    checkpoint = load_checkpoint(args.model)

    if not torch.allclose(checkpoint["k_grid"].float(), metadata_k.float()):
        raise RuntimeError("Checkpoint and test-data K grids differ")

    model = PaperOTSModel()
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    velocity = torch.linspace(-0.8, 0.8, 1000, dtype=torch.float64) * SPEED_OF_LIGHT
    dense_velocity = torch.linspace(-0.4, 0.4, 1000, dtype=torch.float32) * SPEED_OF_LIGHT
    wavelengths_m = wavelengths_nm.double() * 1.0e-9
    results = {}

    for exponent in (3.5, 5.0):
        analytic_spectrum = super_gaussian_spectrum(wavelengths_m, exponent)
        fourier_spectrum, k_grid, truth_coefficients = fourier_super_gaussian_spectrum(
            wavelengths_m, velocity, exponent
        )
        with torch.no_grad():
            predictions = model(
                torch.stack((analytic_spectrum, fourier_spectrum)).to(device)
            ).cpu()

        truth_evdf = inverse_evdf(truth_coefficients[None, :], k_grid, dense_velocity)[0]
        analytic_evdf = inverse_evdf(predictions[0:1], k_grid, dense_velocity)[0]
        fourier_evdf = inverse_evdf(predictions[1:2], k_grid, dense_velocity)[0]
        band_slices = (slice(0, 579), slice(579, 883), slice(883, INPUT_DIM))

        results[str(exponent)] = {
            "spectrum_relative_l2_total": relative_l2(
                fourier_spectrum, analytic_spectrum
            ),
            "spectrum_relative_l2_by_band": [
                relative_l2(fourier_spectrum[band], analytic_spectrum[band])
                for band in band_slices
            ],
            "analytic_input_evdf_metrics": panel_metrics(
                truth_evdf, analytic_evdf, dense_velocity
            ),
            "fourier_input_evdf_metrics": panel_metrics(
                truth_evdf, fourier_evdf, dense_velocity
            ),
            "prediction_coefficient_relative_l2": {
                "analytic_input": relative_l2(truth_coefficients, predictions[0]),
                "fourier_input": relative_l2(truth_coefficients, predictions[1]),
            },
        }

    payload = {
        "checkpoint": str(args.model),
        "test_data": str(args.test_data),
        "comparison": results,
        "interpretation": (
            "The Fourier spectrum is the training-forward-model control; the "
            "analytic spectrum is the released super-Gaussian helper path."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
