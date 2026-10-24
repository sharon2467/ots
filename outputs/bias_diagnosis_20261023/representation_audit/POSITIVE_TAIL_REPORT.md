# Positive-tail sensitivity counterexample

Two constructed, everywhere-positive H VDF perturbations produce approximately **1.8 keV temperature increases**, while H VDF relative-L2 is **2.33–3.27%** and normalized IAW spectrum relative-L2 is **0.80–1.47%**. No clipping, inverse transform or neural network is involved. Thus a large second moment can coexist with a small VDF/spectrum discrepancy even when negative sidelobes and clipping are absent.

| Baseline Te / Ti | Original H T | Perturbed H T | H temperature change | H VDF relative-L2 | Normalized spectrum relative-L2 |
|---|---:|---:|---:|---:|---:|
| 209 / 78 eV | 78.000 eV | 1882.008 eV | +1804.008 eV | 2.3264% | 1.4685% |
| 1392 / 912 eV | 912.000 eV | 2700.996 eV | +1788.996 eV | 3.2660% | 0.80225% |

The baseline arrays and baseline spectra come directly from the previously saved `controlled_arrays.npz`, states 1 and 2. The two modified H distributions are constructed by

`g(v) = normalize[exp(−(v−0.95V)²/(2(0.02V)²)) + exp(−(v+0.95V)²/(2(0.02V)²))]`

`f_H(v) = normalize[0.982 f_H,original(v) + 0.018 g(v)]`,

using unchanged `data.normalize_distribution` for both normalizations. Here `V = vmax = 3.2632767e6 m/s`; tail sigma is `6.5265534e4 m/s`. The mixture coefficient 0.018 denotes final H distribution mass, not peak height. Both components integrate to one, and the final normalization retains their intended weights to roundoff. Electron and carbon distributions remain identical, element by element, to their saved positive originals.

Both modified H distributions integrate to exactly 1 at displayed precision, have zero negative mass and means within 3e-12 m/s of zero. The existing positive-density `distribution_moments` computes all temperatures. Tail-only T is 100300.654 eV, so the second-moment relation is directly visible:

`T_H,mix = 0.982 T_H,original + 0.018 × 100300.654 eV`.

This relation explains the large temperature increase without any negative values or clipping. For the colder baseline, adding the positive distant tails raises a 78 eV physical second-moment temperature to approximately 1.88 keV.

Exactly **two** calls to unchanged `OTS_spectrum_PV` were made, one for each perturbed state. Baseline spectra were read from the saved arrays, without repeating their PV calls. Geometry, species densities/composition, original velocity grids, 319-point wavelength window and `ne = 3.5e26 m^-3` match the baseline. Relative-L2 uses the same unweighted Euclidean norm convention as the earlier diagnostics. The spectrum is peak-normalized by the existing API, so these percentages describe the saved IAW spectral window and shape; they do not compare absolute or broadband power.

The remote Gaussian tails are truncated at the original grid endpoints and have nonzero endpoint values (approximately 2.432e-9 s/m). This is a deliberate finite-support construction. The original PV implementation retains its hard-boundary terms; no smoothing, endpoint zeroing or replacement physics was introduced. No phase velocity equaled a singular endpoint in these two completed calls.

This is a mathematical sensitivity example in positive VDF space. The constructed remote-tail family is **not** the existing sampler's actual distribution family, an existing data row, or a D4 prediction. It has not been shown to lie exactly in the image of the M80 inverse. Consequently, it establishes that temperature sensitivity is not exclusively a clipping phenomenon, while it does not attribute the actual D4 H temperature result to such tails or quantify that model's error. The earlier Gaussian inverse control establishes a separate clipping mechanism; both mechanisms should be distinguished when interpreting a model output.

Reproducible script: [positive_tail_sensitivity.py](positive_tail_sensitivity.py). Detailed results: [positive_tail_results.json](positive_tail_results.json). New toy arrays only: `positive_tail_arrays.npz`. Hashes verify the original source, datasets, model and baseline diagnostic arrays remained unchanged. No additional states, fitting, training or dataset generation were performed.

The NPZ arrays named here are external numerical inputs/results and are not included in this public code-and-report branch.
