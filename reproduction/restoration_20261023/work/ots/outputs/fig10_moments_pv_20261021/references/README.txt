Fig11 references

fig11_experimental_points.csv: 54 existing square readings, 27 Te/27 Ti; all original fields retained.

fig11_simulation_ink_envelopes.csv: native-image saturated color-ink envelopes only. derived_ink_midpoint_ev is an image midpoint, NOT original simulation centerline or numeric truth. No distinct center curve is established. Shaded regions represent spatial inhomogeneity, not statistical error. Stroke thickness and marker occlusion affect bounds; missing columns remain absent. See selection overlay and audit JSON.

Fig10b=blue without probe, Fig10c=red with probe. Common Ti does not give separate TH/TC. Fig11 has no flow curve; prose small flow does not mean exactly zero. 3.5e20 cm^-3 is fully-ionized fill baseline only.

Reproduce from the project root with the configured analysis environment:
python -B outputs/fig10_moments_pv_20261021/references/prepare_fig11_references.py
Map the script’s external input paths to the matching paper and CSV before running.
