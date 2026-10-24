"""Two positive-VDF tail perturbations; no training, clipping or inverse.

Reuse saved original Gaussian states and their saved baseline PV spectra.
Exactly one existing PV call per state, only H changes. Constructed physical-VDF
perturbations are not samples from the existing sampler or D4 predictions.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path('/workspace/ots')
sys.path.insert(0, str(ROOT / 'OTS_code_share/NN_hotE_large_lmRange'))
from data import _physical_config_fig1_fig11, normalize_distribution, distribution_moments
from phyEqns.ots_full_fourier import OTS_spectrum_PV

torch.set_num_threads(2)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def relative_l2(a, b):
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b))

def main():
    input_path = HERE / 'controlled_arrays.npz'
    input_hash = digest(input_path)
    provenance = json.loads((HERE / 'results.json').read_text())
    for name, expected in provenance['sha256_before'].items():
        assert digest(ROOT / name) == expected, name
    baseline = np.load(input_path)
    config = _physical_config_fig1_fig11()
    assert len(config['wavelength_bands_nm']) == 1
    wavelength = torch.tensor(baseline['wavelength_nm'], dtype=torch.float64)
    assert torch.equal(wavelength, config['wavelength_bands_nm'][0])
    fraction = .018
    density = 3.5e26
    mass = config['ion1_dis']['mass_kg']
    record = {
        'scope': 'Two isolated, constructed positive physical-VDF perturbations; no clipping, NN, inverse, fitting, sampler calls or training. Not existing-sampler draws or D4 predictions.',
        'limitations': 'Direct VDF-space sensitivity construction; not a claim that these arrays are exactly representable by the M80 inverse or that any D4 output contains them. The normalized IAW-window spectrum is compared, not absolute/broadband power.',
        'baseline_arrays_sha256_before': input_hash,
        'source_git_head': provenance['git_head'],
        'baseline_source_sha256': {k: v for k, v in provenance['sha256_before'].items() if k.endswith('.py')},
        'density_m3': density,
        'tail_mixture_mass': fraction,
        'tail_centers_vmax': [-.95, .95], 'tail_sigma_vmax': .02,
        'construction': 'g=normalize(exp(-0.5*((v-0.95V)/(0.02V))^2)+exp(-0.5*((v+0.95V)/(0.02V))^2)); f=normalize((1-p)*f0+p*g), p=0.018. Normalize uses existing normalize_distribution.',
        'PV_calls': 0, 'states': [],
    }
    arrays = {'wavelength_nm': wavelength.numpy()}
    for index, te, ti in [(1, 209., 78.), (2, 1392., 912.)]:
        distributions = {species: (torch.tensor(baseline[f'state{index}_{species}_v'], dtype=torch.float64),
                                   torch.tensor(baseline[f'state{index}_{species}_original'], dtype=torch.float64))
                         for species in ['e', 'i1', 'i2']}
        v, f0 = distributions['i1']
        assert torch.equal(v, torch.as_tensor(config['ion1_dis']['velocity_grid_m_s']))
        vmax = float(v.abs().max())
        width = .02 * vmax
        tail = normalize_distribution(v, torch.exp(-.5 * ((v - .95 * vmax) / width).square())
                                      + torch.exp(-.5 * ((v + .95 * vmax) / width).square()), 'toy H tail')
        changed_h = normalize_distribution(v, (1 - fraction) * f0 + fraction * tail, 'toy H mixture')
        assert bool(torch.all(changed_h >= 0))
        distributions['i1'] = (v, changed_h)
        kwargs = {f'{prefix}_{species}': values[j]
                  for species, values in distributions.items()
                  for j, prefix in enumerate(['v', 'f'])}
        changed_spectrum = OTS_spectrum_PV(
            wavelength * 1e-9, config['ss'], config['k_sh'], config['E_i0'], config['lm_0'], density,
            config['Z_i1'], config['N_i1'], config['m_i1'], config['Z_i2'], config['N_i2'],
            config['m_i2'], config['V'], **kwargs, u_e=config['u_e'], u_i1=config['u_i1'],
            u_i2=config['u_i2'], fourier_modes=config['fourier_modes'])
        record['PV_calls'] += 1
        original_spectrum = torch.tensor(baseline[f'state{index}_original_PV'], dtype=torch.float64)
        original_moments = distribution_moments(v, f0, mass)
        changed_moments = distribution_moments(v, changed_h, mass)
        tail_moments = distribution_moments(v, tail, mass)
        row = {
            'baseline_state_index': index, 'Te_ev': te, 'Ti_ev': ti,
            'H_vmax_m_s': vmax, 'tail_sigma_m_s': width,
            'H_original': original_moments, 'H_perturbed': changed_moments,
            'H_tail_alone': tail_moments,
            'H_temperature_change_ev': changed_moments['actual_temperature_ev'] - original_moments['actual_temperature_ev'],
            'H_temperature_ratio': changed_moments['actual_temperature_ev'] / original_moments['actual_temperature_ev'],
            'H_VDF_relative_l2': relative_l2(changed_h, f0),
            'spectrum_relative_l2': relative_l2(changed_spectrum, original_spectrum),
            'H_norm': float(torch.trapz(changed_h, v)), 'H_min_f': float(changed_h.min()),
            'H_negative_mass': float(torch.trapz(torch.relu(-changed_h), v)),
            'H_endpoint_f': [float(changed_h[0]), float(changed_h[-1])],
            'electrons_unchanged': bool(torch.equal(distributions['e'][1], torch.tensor(baseline[f'state{index}_e_original']))),
            'carbon_unchanged': bool(torch.equal(distributions['i2'][1], torch.tensor(baseline[f'state{index}_i2_original']))),
        }
        record['states'].append(row)
        arrays[f'state{index}_H_v'] = v.numpy()
        arrays[f'state{index}_H_tail_only'] = tail.numpy()
        arrays[f'state{index}_H_positive_mixture'] = changed_h.numpy()
        arrays[f'state{index}_perturbed_PV'] = changed_spectrum.numpy()
        print(json.dumps(row, indent=2))
    assert record['PV_calls'] == 2
    assert digest(input_path) == input_hash
    for name, expected in provenance['sha256_before'].items():
        assert digest(ROOT / name) == expected, name
    record['baseline_arrays_sha256_after'] = digest(input_path)
    record['original_source_data_model_and_baseline_arrays_unchanged'] = True
    (HERE / 'positive_tail_results.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(HERE / 'positive_tail_arrays.npz', **arrays)

if __name__ == '__main__':
    main()
