"""Three isolated known-truth representation controls, never training data.

No Parquet, registry, NN, training, or dataset generator. Existing public sampler
VDF methods, transform, inverse, moment and PV APIs are reused unchanged.
"""
import json
from pathlib import Path
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, '/workspace/ots/OTS_code_share/NN_hotE_large_lmRange')
from audit_representation import signed_diagnostics, sha
from data import _physical_config_fig1_fig11, Fig1_Fig11_sampler, SPECIES_BLOCKS, distribution_moments, normalize_distribution
from evaluation import reconstruct_real_evdf
from phyEqns.ots_full_fourier import ft_x_n, OTS_spectrum_PV

torch.set_num_threads(2)

def relative_l2(a, b):
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b))

def pv(config, distributions, density):
    kwargs = {f'{prefix}_{species}': values[j]
              for species, values in distributions.items()
              for j, prefix in enumerate(['v', 'f'])}
    return torch.cat([OTS_spectrum_PV(
        band * 1e-9, config['ss'], config['k_sh'], config['E_i0'], config['lm_0'], density,
        config['Z_i1'], config['N_i1'], config['m_i1'], config['Z_i2'], config['N_i2'],
        config['m_i2'], config['V'], **kwargs, u_e=config['u_e'], u_i1=config['u_i1'],
        u_i2=config['u_i2'], fourier_modes=config['fourier_modes'])
        for band in config['wavelength_bands_nm']])

def main():
    anchor = Path('/workspace/ots-cloud-setup/data/paper_experimental_points.csv')
    config = _physical_config_fig1_fig11()
    config['points_csv'] = str(anchor)
    config['sampling']['shape_probabilities'] = [1., 0., 0.]
    sampler = Fig1_Fig11_sampler(np.random.default_rng(20261023), config)
    records = {'scope': 'Three standalone Gaussian known-truth diagnostic states, not existing-row replay or new training datasets; no NN.',
               'anchor_path': str(anchor), 'anchor_sha256': sha(anchor),
               'density_m3': 3.5e26, 'states': []}
    arrays = {'wavelength_nm': config['wavelength_bands_nm'][0].numpy()}
    for state_index, (te, ti) in enumerate([(100., 60.), (209., 78.), (1392., 912.)]):
        sampler.parameters = {'expected': {s: {'temperature_ev': t, 'mean_velocity_m_s': 0.}
                                         for s, t in [('e', te), ('i1', ti), ('i2', ti)]},
                              'electron_density_m3': records['density_m3']}
        truth = {'e': sampler.sample_electron_vdf(), 'i1': sampler.sample_ion1_vdf(), 'i2': sampler.sample_ion2_vdf()}
        clipped = {}
        state = {'state_index': state_index, 'Te_ev': te, 'Ti_ev': ti, 'species': {}}
        for species, key in SPECIES_BLOCKS:
            v, f = truth[species]
            k, fc = ft_x_n(v, f, config['fourier_modes'])
            label = torch.cat((fc.real, fc.imag)).reshape(1, -1)
            _, reconstructed = reconstruct_real_evdf(label, k, velocity=v)
            raw = reconstructed[0]
            physical = normalize_distribution(v, torch.relu(raw), species)
            clipped[species] = (v, physical)
            mass = config[key]['mass_kg']
            original_moments = distribution_moments(v, f, mass)
            raw_moments = signed_diagnostics(v, raw, mass)
            clip_moments = distribution_moments(v, physical, mass)
            state['species'][species] = {
                'original': original_moments, 'raw_signed': raw_moments,
                'clipped_normalized': clip_moments,
                'raw_VDF_relative_l2_vs_original': relative_l2(raw, f),
                'clipnorm_VDF_relative_l2_vs_original': relative_l2(physical, f),
                'raw_temperature_bias_ev': raw_moments['temperature_ev'] - original_moments['actual_temperature_ev'],
                'clipnorm_temperature_bias_ev': clip_moments['actual_temperature_ev'] - original_moments['actual_temperature_ev'],
            }
            arrays[f'state{state_index}_{species}_v'] = v.numpy()
            arrays[f'state{state_index}_{species}_original'] = f.numpy()
            arrays[f'state{state_index}_{species}_raw_inverse'] = raw.numpy()
            arrays[f'state{state_index}_{species}_clipnorm'] = physical.numpy()
        original_spectrum = pv(config, truth, records['density_m3'])
        reconstructed_spectrum = pv(config, clipped, records['density_m3'])
        state['PV_clipnorm_all_species_relative_l2_vs_original_PV'] = relative_l2(reconstructed_spectrum, original_spectrum)
        state['PV_raw_signed_not_called_reason'] = 'Existing public PV API requires nonnegative, normalized distributions; raw inverse has negative lobes.'
        arrays[f'state{state_index}_original_PV'] = original_spectrum.numpy()
        arrays[f'state{state_index}_clipnorm_PV'] = reconstructed_spectrum.numpy()
        records['states'].append(state)
        print('state', te, ti, 'PV relative-L2', state['PV_clipnorm_all_species_relative_l2_vs_original_PV'])
        for species, r in state['species'].items():
            print(species, 'L2', r['raw_VDF_relative_l2_vs_original'], 'T truth/raw/clip', r['original']['actual_temperature_ev'], r['raw_signed']['temperature_ev'], r['clipped_normalized']['actual_temperature_ev'])
    (HERE / 'controlled_results.json').write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(HERE / 'controlled_arrays.npz', **arrays)

if __name__ == '__main__':
    main()
