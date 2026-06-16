# -*- coding: utf-8 -*-
"""
Quick smoke test for the OTS inversion pipeline.

This script intentionally uses a tiny synthetic dataset and a much shorter
training schedule than `main.py`. It verifies that the following chain works:
EVDF -> Fourier coefficients -> OTS spectrum -> DNN prediction -> inverse
Fourier reconstruction.

Differences from the paper/full training run:
- Only 20 train samples and 5 test samples are generated.
- The EVDF helper uses 3 background Gaussian components and no explicit
  hot-electron enhancement.
- The loss reported here is coefficient-domain L1, matching `train.py`;
  the paper text describes EVDF-domain MSE for the full experiment.
"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch, math, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phyEqns.ots_fourier import OTS_spectrum_fft, ft_x, ft_x_n, ift_k, ift_k_x
from model import ModelOTS, load_model
from train import train, Loss_func

# --- physical constants ---
e = torch.tensor(1.602176634e-19)
k_B = torch.tensor(1.380649e-23)
c = torch.tensor(2.99792458e8)
m_e = torch.tensor(9.10938356e-31)
m_p = torch.tensor(1.67262192369e-27)

# --- OTS geometry and plasma parameters ---
an_s = 123/180*math.pi
ss = torch.tensor([math.cos(an_s), math.sin(an_s), 0.0])
k_sh = torch.tensor([1.0, 0.0, 0.0])
E_i0 = torch.tensor([0.0, 0.0, 1.0])
lm_0 = 351e-9
Te_eV = 1.37e3
Ti_eV = 0.223e3
n_e = 1.81e26

Z_i1, N_i1 = 1, 0.55
Z_i2, N_i2 = 6, 0.45
m_i1, m_i2 = m_p, m_p*12

u_e = torch.zeros(3) * c
u_i1 = torch.zeros(3) * c
u_i2 = torch.zeros(3) * c

# Wavelength grids for the three independently normalized input bands:
# ion feature, red-shifted electron feature, and blue-shifted electron feature.
x1 = torch.linspace(349, 353, 579) * 1e-9
x2 = torch.linspace(375.3552, 599.7064, 304) * 1e-9
x3 = torch.linspace(100.1144, 339.6504, 341) * 1e-9

def random_gaussians(v):
    """Generate a normalized 3-Gaussian EVDF on the supplied velocity grid."""
    f_v = torch.zeros_like(v)
    for _ in range(3):
        center = np.random.uniform(-0.06, 0.06) * c
        sigma = np.random.uniform(0.01, 0.05) * c
        height = np.random.uniform(0.2, 2.5)
        f_v += height * torch.exp(-((v - center)**2) / (2 * sigma**2))
    f_v /= torch.trapz(f_v, v)
    return f_v

def generate_one_sample(vxs):
    """
    Generate one spectrum/label pair.

    Returns:
        I_0:   1224-point concatenated OTS spectrum.
        label: 160 Fourier coefficients stored as [Re(F_e), Im(F_e)].
    """
    T_e = Te_eV * e / k_B
    T_i = Ti_eV * e / k_B
    V = 1e-9

    f_es = random_gaussians(vxs)
    K2, F_e = ft_x_n(vxs, f_es, 80)

    y1 = OTS_spectrum_fft(x1, ss, k_sh, E_i0, lm_0, n_e, T_e, Z_i1, N_i1,
                          m_i1, Z_i2, N_i2, m_i2, T_i, V, K2, F_e, u_e, u_i1, u_i2)
    y2 = OTS_spectrum_fft(x2, ss, k_sh, E_i0, lm_0, n_e, T_e, Z_i1, N_i1,
                          m_i1, Z_i2, N_i2, m_i2, T_i, V, K2, F_e, u_e, u_i1, u_i2)
    y3 = OTS_spectrum_fft(x3, ss, k_sh, E_i0, lm_0, n_e, T_e, Z_i1, N_i1,
                          m_i1, Z_i2, N_i2, m_i2, T_i, V, K2, F_e, u_e, u_i1, u_i2)

    I_0 = torch.cat((y1, y2, y3), 0)
    # The model predicts the low-pass EVDF Fourier representation directly.
    label = torch.cat((torch.real(F_e), torch.imag(F_e)), 0)
    return I_0, label

print("=" * 50)
print("Step 1: Generating small dataset (20 train + 5 test)")
print("=" * 50)

vxs = torch.linspace(-1, 1, 1000) * 0.8 * c
N_train, N_test = 20, 5

t0 = time.time()
train_xs, train_ys = [], []
for i in range(N_train):
    x, y = generate_one_sample(vxs)
    train_xs.append(x)
    train_ys.append(y)
    if (i+1) % 5 == 0:
        print(f"  Generated {i+1}/{N_train} training samples...")

test_xs, test_ys = [], []
for i in range(N_test):
    x, y = generate_one_sample(vxs)
    test_xs.append(x)
    test_ys.append(y)

train_xs = torch.stack(train_xs).float()
train_ys = torch.stack(train_ys).float()
test_xs = torch.stack(test_xs).float()
test_ys = torch.stack(test_ys).float()

print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Train: {train_xs.shape} -> {train_ys.shape}")
print(f"  Test:  {test_xs.shape} -> {test_ys.shape}")

print()
print("=" * 50)
print("Step 2: Training model (200+400+800 = 1400 iters)")
print("=" * 50)

model = ModelOTS()
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

t0 = time.time()
train(train_xs, train_ys, model, loss_rate=1e-4, N=200)
train(train_xs, train_ys, model, loss_rate=1e-5, N=400)
train(train_xs, train_ys, model, loss_rate=1e-6, N=800)
print(f"  Training done in {time.time()-t0:.1f}s")

print()
print("=" * 50)
print("Step 3: Evaluation")
print("=" * 50)

with torch.no_grad():
    train_pred = model(train_xs)
    test_pred = model(test_xs)
    # Smoke-test metric: same coefficient-domain L1 used by train.py.
    train_loss = torch.nn.L1Loss()(train_pred, train_ys)
    test_loss = torch.nn.L1Loss()(test_pred, test_ys)

print(f"  Train loss (L1): {train_loss.item():.6f}")
print(f"  Test loss  (L1): {test_loss.item():.6f}")

print()
print("=" * 50)
print("Step 4: Visualizing f(v) reconstruction")
print("=" * 50)

sample_idx = 0
ys_true = test_ys[sample_idx].numpy()
ys_pred = test_pred[sample_idx].numpy()
Nk = 80

Fe_true = torch.tensor(ys_true[:Nk]) + 1j*torch.tensor(ys_true[Nk:])
Fe_pred = torch.tensor(ys_pred[:Nk]) + 1j*torch.tensor(ys_pred[Nk:])

# Reconstruct f(v) from the predicted and true Fourier coefficients.
K2 = torch.linspace(-1, 1, Nk) * (2*math.pi/(2*0.8*c)) * Nk/2
v_recon, fv_true = ift_k(K2, Fe_true)
_, fv_pred = ift_k(K2, Fe_pred)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(test_xs[sample_idx].numpy(), alpha=0.7, linewidth=0.5)
axes[0].set_title("Input Spectrum")
axes[0].set_xlabel("Wavelength index")
axes[0].set_ylabel("I (normalized)")

axes[1].plot(ys_true[:Nk], "b-", label="Re (true)", alpha=0.7)
axes[1].plot(ys_pred[:Nk], "b--", label="Re (pred)", alpha=0.7)
axes[1].plot(ys_true[Nk:], "r-", label="Im (true)", alpha=0.7)
axes[1].plot(ys_pred[Nk:], "r--", label="Im (pred)", alpha=0.7)
axes[1].set_title("Fourier Coefficients")
axes[1].legend(fontsize=7)

axes[2].plot(v_recon.numpy(), fv_true.numpy(), "k-", label="True f(v)")
axes[2].plot(v_recon.numpy(), fv_pred.numpy(), "r--", label="Reconstructed")
axes[2].set_title("f(v) Reconstruction")
axes[2].set_xlabel("v (m/s)")
axes[2].legend(fontsize=7)

plt.tight_layout()
plt.savefig("quick_test_result.png", dpi=150)
print("  Saved: quick_test_result.png")
print()
print("=" * 50)
print("ALL DONE - Pipeline verified!")
print("=" * 50)
