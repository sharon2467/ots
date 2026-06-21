"""Unified evaluation entry point for the OTS inversion model.

Two data sources are supported:

1. ``parquet`` reads spectra and Fourier labels from ``dataset_test.parquet``.
2. ``synthetic`` creates EVDFs in memory, runs the OTS forward model, and
   evaluates the neural network without writing a temporary dataset.

The summary figure compares reconstructed EVDFs for a batch. Passing
``--detail-index`` also creates a spectrum/Fourier/EVDF single-sample view.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from model import load_model
from phyEqns.ots_fourier import OTS_spectrum_fft, ft_x_n, ift_k
from readData import readData_dataset


SCRIPT_DIR = Path(__file__).resolve().parent

ELEMENTARY_CHARGE = 1.602176634e-19
BOLTZMANN_CONSTANT = 1.380649e-23
SPEED_OF_LIGHT = 2.99792458e8
PROTON_MASS = 1.67262192369e-27

ION_BAND = slice(0, 579)
RED_ELECTRON_BAND = slice(579, 883)
BLUE_ELECTRON_BAND = slice(883, 1224)
FOURIER_MODES = 80


@dataclass
class EvaluationBatch:
    """Spectra, targets, and grids required by the evaluation pipeline."""

    spectra: torch.Tensor
    coefficients: torch.Tensor
    wavelengths_m: torch.Tensor
    k_grid: torch.Tensor
    labels: list[str]


@dataclass
class OtsPhysics:
    """Physical parameters used to create synthetic OTS spectra."""

    scattering_direction: torch.Tensor
    incident_direction: torch.Tensor
    incident_polarization: torch.Tensor
    probe_wavelength_m: float
    electron_density_m3: float
    electron_temperature_k: float
    ion_temperature_k: float
    ion_1_charge: float
    ion_1_fraction: float
    ion_1_mass_kg: float
    ion_2_charge: float
    ion_2_fraction: float
    ion_2_mass_kg: float
    plasma_volume_m3: float
    electron_drift_m_s: torch.Tensor
    ion_1_drift_m_s: torch.Tensor
    ion_2_drift_m_s: torch.Tensor

    @classmethod
    def default(cls) -> "OtsPhysics":
        scattering_angle = math.radians(123.0)
        zero_drift = torch.zeros(3)
        return cls(
            scattering_direction=torch.tensor(
                [math.cos(scattering_angle), math.sin(scattering_angle), 0.0]
            ),
            incident_direction=torch.tensor([1.0, 0.0, 0.0]),
            incident_polarization=torch.tensor([0.0, 0.0, 1.0]),
            probe_wavelength_m=351e-9,
            electron_density_m3=1.81e26,
            electron_temperature_k=1370.0
            * ELEMENTARY_CHARGE
            / BOLTZMANN_CONSTANT,
            ion_temperature_k=171.3
            * ELEMENTARY_CHARGE
            / BOLTZMANN_CONSTANT,
            ion_1_charge=1.0,
            ion_1_fraction=0.55,
            ion_1_mass_kg=PROTON_MASS,
            ion_2_charge=6.0,
            ion_2_fraction=0.45,
            ion_2_mass_kg=12.0 * PROTON_MASS,
            plasma_volume_m3=(1e-3) ** 3,
            electron_drift_m_s=zero_drift.clone(),
            ion_1_drift_m_s=zero_drift.clone(),
            ion_2_drift_m_s=zero_drift.clone(),
        )

    def spectrum(
        self,
        wavelengths_m: torch.Tensor,
        k_grid: torch.Tensor,
        fourier_coefficients: torch.Tensor,
    ) -> torch.Tensor:
        return OTS_spectrum_fft(
            wavelengths_m,
            self.scattering_direction,
            self.incident_direction,
            self.incident_polarization,
            self.probe_wavelength_m,
            self.electron_density_m3,
            self.electron_temperature_k,
            self.ion_1_charge,
            self.ion_1_fraction,
            self.ion_1_mass_kg,
            self.ion_2_charge,
            self.ion_2_fraction,
            self.ion_2_mass_kg,
            self.ion_temperature_k,
            self.plasma_volume_m3,
            k_grid,
            fourier_coefficients,
            self.electron_drift_m_s,
            self.ion_1_drift_m_s,
            self.ion_2_drift_m_s,
        )


def default_wavelength_grid() -> torch.Tensor:
    """Return the three wavelength bands used by the trained network."""

    ion = torch.linspace(349.0, 353.0, 579) * 1e-9
    red = torch.linspace(375.3552, 599.7064, 304) * 1e-9
    blue = torch.linspace(100.1144, 339.6504, 341) * 1e-9
    return torch.cat((ion, red, blue))


def load_parquet_batch(path: Path, start: int, count: int) -> EvaluationBatch:
    """Load metadata and physical samples from the legacy Parquet layout."""

    metadata_spectra, metadata_coefficients = readData_dataset(
        str(path), 0, 1
    )
    spectra, coefficients = readData_dataset(
        str(path), start, start + count
    )
    if len(spectra) == 0:
        raise ValueError(
            f"No samples were read from {path} at physical row {start}."
        )

    wavelengths_m = metadata_spectra[0] * 1e-9
    k_grid = metadata_coefficients[0, :FOURIER_MODES]
    labels = [f"Parquet row {start + index}" for index in range(len(spectra))]
    return EvaluationBatch(
        spectra=spectra.float(),
        coefficients=coefficients.float(),
        wavelengths_m=wavelengths_m.float(),
        k_grid=k_grid.float(),
        labels=labels,
    )


def normalize_evdf(evdf: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    integral = torch.trapz(evdf, velocity)
    if not torch.isfinite(integral) or integral <= 0:
        raise ValueError("Synthetic EVDF has an invalid normalization integral.")
    return evdf / integral


def random_uniform(
    low: float, high: float, generator: torch.Generator
) -> float:
    sample = torch.rand((), generator=generator).item()
    return low + (high - low) * sample


def make_synthetic_evdf(
    velocity: torch.Tensor,
    profile: str,
    index: int,
    count: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, str]:
    """Create one EVDF for a random or controlled synthetic test."""

    if profile == "width-scan":
        width_fraction = torch.linspace(0.01, 0.20, count)[index].item()
        width = width_fraction * SPEED_OF_LIGHT
        evdf = torch.exp(-(velocity**2) / width**2)
        return normalize_evdf(evdf, velocity), f"width={width_fraction:.3f}c"

    if profile == "hot-electron-scan":
        center_fraction = torch.linspace(-0.51, 0.51, count)[index].item()
        background_width = 0.08 * SPEED_OF_LIGHT
        hot_width = 0.06 * SPEED_OF_LIGHT
        hot_center = center_fraction * SPEED_OF_LIGHT
        evdf = torch.exp(-(velocity**2) / background_width**2)
        evdf += 0.5 * torch.exp(
            -((velocity - hot_center) ** 2) / hot_width**2
        )
        return (
            normalize_evdf(evdf, velocity),
            f"hot center={center_fraction:.3f}c",
        )

    evdf = torch.zeros_like(velocity)
    for _ in range(3):
        center = random_uniform(-0.06, 0.06, generator) * SPEED_OF_LIGHT
        sigma = random_uniform(0.01, 0.05, generator) * SPEED_OF_LIGHT
        height = random_uniform(0.2, 2.5, generator)
        evdf += height * torch.exp(
            -((velocity - center) ** 2) / (2.0 * sigma**2)
        )

    hot_center = random_uniform(-0.4, 0.4, generator) * SPEED_OF_LIGHT
    hot_sigma = random_uniform(0.01, 0.05, generator) * SPEED_OF_LIGHT
    hot_height = random_uniform(0.001, 0.045, generator)
    evdf += hot_height * torch.exp(
        -((velocity - hot_center) ** 2) / (2.0 * hot_sigma**2)
    )
    return normalize_evdf(evdf, velocity), f"random sample {index + 1}"


def generate_synthetic_batch(
    count: int, profile: str, seed: int
) -> EvaluationBatch:
    """Generate spectra and labels directly from the OTS forward model."""

    physics = OtsPhysics.default()
    wavelengths_m = default_wavelength_grid()
    velocity = (
        torch.linspace(-0.8, 0.8, 1000) * SPEED_OF_LIGHT
    )
    generator = torch.Generator().manual_seed(seed)

    spectra: list[torch.Tensor] = []
    coefficients: list[torch.Tensor] = []
    labels: list[str] = []
    k_grid: torch.Tensor | None = None

    for index in range(count):
        evdf, label = make_synthetic_evdf(
            velocity, profile, index, count, generator
        )
        sample_k_grid, sample_coefficients = ft_x_n(
            velocity, evdf, FOURIER_MODES
        )
        if k_grid is None:
            k_grid = sample_k_grid

        spectrum = torch.cat(
            (
                physics.spectrum(
                    wavelengths_m[ION_BAND],
                    sample_k_grid,
                    sample_coefficients,
                ),
                physics.spectrum(
                    wavelengths_m[RED_ELECTRON_BAND],
                    sample_k_grid,
                    sample_coefficients,
                ),
                physics.spectrum(
                    wavelengths_m[BLUE_ELECTRON_BAND],
                    sample_k_grid,
                    sample_coefficients,
                ),
            )
        )
        target = torch.cat(
            (sample_coefficients.real, sample_coefficients.imag)
        )
        spectra.append(spectrum.float())
        coefficients.append(target.float())
        labels.append(label)

    if k_grid is None:
        raise ValueError("Synthetic sample count must be positive.")

    return EvaluationBatch(
        spectra=torch.stack(spectra),
        coefficients=torch.stack(coefficients),
        wavelengths_m=wavelengths_m,
        k_grid=k_grid.float(),
        labels=labels,
    )


def reconstruct_evdf(
    k_grid: torch.Tensor, coefficients: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    mode_count = coefficients.numel() // 2
    complex_coefficients = (
        coefficients[:mode_count]
        + 1j * coefficients[mode_count : 2 * mode_count]
    )
    velocity, evdf = ift_k(k_grid, complex_coefficients)
    return velocity.real, evdf.real


def evaluate(
    batch: EvaluationBatch, model_path: Path
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run inference and calculate coefficient- and EVDF-domain errors."""

    model = load_model(str(model_path))
    model.eval()
    with torch.no_grad():
        predictions = model(batch.spectra).cpu()

    coefficient_error = predictions - batch.coefficients
    true_evdfs: list[torch.Tensor] = []
    predicted_evdfs: list[torch.Tensor] = []
    for target, prediction in zip(batch.coefficients, predictions):
        _, true_evdf = reconstruct_evdf(batch.k_grid, target)
        _, predicted_evdf = reconstruct_evdf(batch.k_grid, prediction)
        true_evdfs.append(true_evdf)
        predicted_evdfs.append(predicted_evdf)

    evdf_error = torch.stack(predicted_evdfs) - torch.stack(true_evdfs)
    metrics = {
        "coefficient_l1": coefficient_error.abs().mean().item(),
        "coefficient_mse": coefficient_error.square().mean().item(),
        "evdf_l1": evdf_error.abs().mean().item(),
        "evdf_mse": evdf_error.square().mean().item(),
    }
    return predictions, metrics


def plot_summary(
    batch: EvaluationBatch,
    predictions: torch.Tensor,
    output_path: Path,
) -> None:
    """Plot true and predicted EVDFs for all selected samples."""

    sample_count = len(batch.spectra)
    columns = min(3, sample_count)
    rows = math.ceil(sample_count / columns)
    figure = plt.figure(figsize=(5 * columns, 3.5 * rows))

    for index, (target, prediction) in enumerate(
        zip(batch.coefficients, predictions)
    ):
        velocity, true_evdf = reconstruct_evdf(batch.k_grid, target)
        _, predicted_evdf = reconstruct_evdf(batch.k_grid, prediction)
        axis = figure.add_subplot(rows, columns, index + 1)
        axis.plot(
            velocity / SPEED_OF_LIGHT,
            true_evdf,
            label="Ground truth",
            linewidth=2,
        )
        axis.plot(
            velocity / SPEED_OF_LIGHT,
            predicted_evdf,
            label="Prediction",
            linestyle="--",
        )
        axis.set_title(batch.labels[index])
        axis.set_xlabel("Velocity / c")
        axis.set_ylabel("EVDF")
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_detail(
    batch: EvaluationBatch,
    predictions: torch.Tensor,
    sample_index: int,
    output_path: Path,
) -> None:
    """Plot one sample's spectra, Fourier coefficients, and EVDF."""

    if sample_index < 0 or sample_index >= len(batch.spectra):
        raise IndexError(
            f"detail index {sample_index} is outside the selected batch "
            f"of {len(batch.spectra)} samples"
        )

    spectrum = batch.spectra[sample_index]
    target = batch.coefficients[sample_index]
    prediction = predictions[sample_index]
    velocity, true_evdf = reconstruct_evdf(batch.k_grid, target)
    _, predicted_evdf = reconstruct_evdf(batch.k_grid, prediction)

    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    bands = (
        (ION_BAND, "Ion feature"),
        (RED_ELECTRON_BAND, "Red-shifted electron feature"),
        (BLUE_ELECTRON_BAND, "Blue-shifted electron feature"),
    )
    for axis, (band, title) in zip(axes[0], bands):
        axis.plot(batch.wavelengths_m[band] * 1e9, spectrum[band])
        axis.set_title(title)
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Normalized intensity")
        axis.grid(True, alpha=0.3)

    axes[1, 0].plot(
        target[:FOURIER_MODES], label="Ground truth", linewidth=2
    )
    axes[1, 0].plot(
        prediction[:FOURIER_MODES], label="Prediction", linestyle="--"
    )
    axes[1, 0].set_title("Fourier coefficients: real")
    axes[1, 0].set_xlabel("Mode index")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(
        target[FOURIER_MODES:], label="Ground truth", linewidth=2
    )
    axes[1, 1].plot(
        prediction[FOURIER_MODES:], label="Prediction", linestyle="--"
    )
    axes[1, 1].set_title("Fourier coefficients: imaginary")
    axes[1, 1].set_xlabel("Mode index")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    axes[1, 2].plot(
        velocity / SPEED_OF_LIGHT,
        true_evdf,
        label="Ground truth",
        linewidth=2,
    )
    axes[1, 2].plot(
        velocity / SPEED_OF_LIGHT,
        predicted_evdf,
        label="Prediction",
        linestyle="--",
    )
    axes[1, 2].set_title("Reconstructed EVDF")
    axes[1, 2].set_xlabel("Velocity / c")
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()

    figure.suptitle(batch.labels[sample_index])
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the OTS inversion model on stored or generated data."
    )
    parser.add_argument(
        "--source",
        choices=("parquet", "synthetic"),
        default="parquet",
        help="Read a Parquet test set or generate samples in memory.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=SCRIPT_DIR / "dataset_test.parquet",
        help="Legacy Parquet test dataset used by --source parquet.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=SCRIPT_DIR / "OTS_Te_model.pickle",
        help="Trained ModelOTS checkpoint.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=6,
        help="Number of samples to evaluate.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="First physical Parquet row; row 0 contains metadata.",
    )
    parser.add_argument(
        "--synthetic-profile",
        choices=("random", "width-scan", "hot-electron-scan"),
        default="random",
        help="EVDF family used by --source synthetic.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for synthetic random samples.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "evaluation.png",
        help="Batch EVDF comparison figure.",
    )
    parser.add_argument(
        "--detail-index",
        type=int,
        default=None,
        help="Optional zero-based index inside the selected batch.",
    )
    parser.add_argument(
        "--detail-output",
        type=Path,
        default=SCRIPT_DIR / "evaluation_detail.png",
        help="Detailed single-sample figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive.")
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    if args.source == "parquet":
        if args.start_index < 1:
            raise ValueError(
                "--start-index must be at least 1 because row 0 is metadata."
            )
        if not args.data_path.is_file():
            raise FileNotFoundError(f"Parquet test set not found: {args.data_path}")
        batch = load_parquet_batch(
            args.data_path, args.start_index, args.samples
        )
    else:
        batch = generate_synthetic_batch(
            args.samples, args.synthetic_profile, args.seed
        )

    predictions, metrics = evaluate(batch, args.model_path)
    plot_summary(batch, predictions, args.output)
    print(f"Evaluated {len(batch.spectra)} samples from {args.source}.")
    for name, value in metrics.items():
        print(f"{name}: {value:.6e}")
    print(f"Summary figure: {args.output}")

    if args.detail_index is not None:
        plot_detail(
            batch, predictions, args.detail_index, args.detail_output
        )
        print(f"Detail figure: {args.detail_output}")


if __name__ == "__main__":
    main()
