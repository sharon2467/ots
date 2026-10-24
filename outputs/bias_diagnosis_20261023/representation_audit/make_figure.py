"""Plot saved diagnostic arrays; no additional physical calculations/calls."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
r = json.loads((HERE / 'controlled_results.json').read_text())
a = np.load(HERE / 'controlled_arrays.npz')
fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
v = a['state0_i1_v']
mass = 1.67262192369e-27
for suffix, label, color in [('original', 'Original Gaussian', 'black'), ('raw_inverse', 'Exact-label inverse', '#3479b8'), ('clipnorm', 'Clipped + normalized', '#dd7132')]:
    f = a['state0_i1_' + suffix]
    ax[0, 0].plot(v / v.max(), f * v.max(), color=color, label=label, lw=1.2)
    density = mass / 1.602176634e-19 * v**2 * f / np.trapezoid(f, v)
    cumulative = np.r_[0., np.cumsum((density[1:] + density[:-1]) / 2 * np.diff(v))]
    ax[0, 1].plot(v / v.max(), cumulative, color=color, label=label)
ax[0, 0].set_yscale('symlog', linthresh=1e-4)
ax[0, 0].set(xlabel=r'$v/v_{max}$', ylabel=r'$v_{max} f(v)$', title='A. Hydrogen target, Ti = 60 eV; M = 80')
ax[0, 0].legend(fontsize=8)
ax[0, 1].set(xlabel=r'$v/v_{max}$', ylabel='Cumulative signed second moment (eV)', title='B. Clipping removes cancellation in v²-weighted tails')
lm = a['wavelength_nm']
ax[1, 0].plot(lm, a['state0_original_PV'], color='black', label='Original f → PV')
ax[1, 0].plot(lm, a['state0_clipnorm_PV'], color='#dd7132', label='Exact label → inverse → clipnorm → PV')
ax[1, 0].set(xlabel='Wavelength (nm)', ylabel='Normalized intensity', title='C. All species, same ne = 3.5e26 m⁻³; spectrum L2 = 14.16%')
ax[1, 0].legend(fontsize=8)
species = ['e', 'i1', 'i2']
pos = np.arange(3)
states = r['states'][0]['species']
for offset, key, label, color in [(-.24, 'original', 'Original', 'black'), (0., 'raw_signed', 'Raw signed inverse', '#3479b8'), (.24, 'clipped_normalized', 'Clipped + normalized', '#dd7132')]:
    field = 'temperature_ev' if key == 'raw_signed' else 'actual_temperature_ev'
    ax[1, 1].bar(pos + offset, [states[s][key][field] for s in species], .23, label=label, color=color)
ax[1, 1].set(xticks=pos, xticklabels=['Electron', 'Hydrogen', 'Carbon'], ylabel='Temperature (eV)', title='D. Tiny VDF L2 does not ensure accurate clipped moments')
ax[1, 1].legend(fontsize=8)
for panel in ax.flat:
    panel.grid(alpha=.18)
    panel.title.set_fontsize(10)
fig.suptitle('No neural network: Fourier representation and postprocessing control', fontsize=13)
fig.savefig(HERE / 'representation_control.png', dpi=180)
fig.savefig(HERE / 'representation_control.pdf')
print(HERE / 'representation_control.png')
