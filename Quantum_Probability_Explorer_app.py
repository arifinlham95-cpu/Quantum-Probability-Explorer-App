"""
Quantum Probability Explorer
Streamlit app (single-file): Quantum_Probability_Explorer_app.py

Purpose:
- Interactive visualization for 1D wavefunctions (infinite square well eigenstates and their superpositions).
- Shows real(ψ), imag(ψ), and probability density |ψ|^2, plus normalization and expectation values <x>, <p>.
- Designed for teaching materi fungsi gelombang (SMA/undergraduate intro).

Requirements (requirements.txt):
streamlit
numpy
scipy
matplotlib
"""

# --- App code starts here ---
import streamlit as st
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

st.set_page_config(page_title="Quantum Probability Explorer", layout="wide")

# Constants (dimensionless units)
hbar = 1.0
m = 1.0

# Helper functions
def phi_infinite_well(x, L, n):
    """Eigenfunction for infinite square well [0, L]."""
    return np.sqrt(2.0 / L) * np.sin(n * np.pi * x / L)

def E_infinite_well(n, L):
    """Energy eigenvalues for infinite square well [0, L]."""
    return (n ** 2 * np.pi ** 2 * hbar ** 2) / (2.0 * m * L ** 2)

def expectation_x(x, prob):
    return np.trapz(x * prob, x)

# ==============================
# UI LAYOUT
# ==============================
st.title("Quantum Probability Explorer — Fungsi Gelombang (Kotak Tak Hingga)")
st.markdown("Aplikasi interaktif untuk mengeksplor fungsi gelombang ψ(x,t) dan probabilitas |ψ|² dalam **infinite square well**. "
            "Satuan natural: ħ = 1, m = 1.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Pengaturan Sistem")
    L = st.slider("Panjang kotak L", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
    x_points = st.slider("Jumlah titik x (resolusi)", 200, 2000, 800)
    x = np.linspace(0, L, x_points)

    mode = st.radio("Pilih mode:", ["Eigenstate tunggal", "Superposisi"])

    if mode == "Eigenstate tunggal":
        n_val = st.slider("Pilih bilangan kuantum n", 1, 8, 1)
        max_basis = n_val
        coeffs = np.array([1.0 if i == (n_val-1) else 0.0 for i in range(max_basis)], dtype=complex)

    else:  # Superposisi
        max_basis = st.slider("Jumlah basis (N)", 1, 8, 4)
        st.markdown("**Koefisien kompleks untuk masing-masing basis** — atur amplitudo & fase lalu normalisasi otomatis.")
        amps, phases = [], []
        for n in range(1, max_basis + 1):
            st.write(f"Basis n = {n}")
            col_a, col_p = st.columns([1, 2])
            with col_a:
                a = st.slider(f"Amplitudo a_{n}", 0.0, 1.0, 0.0, key=f"a{n}")
            with col_p:
                ph = st.slider(f"Fase φ_{n} (rad)", 0.0, 2 * np.pi, 0.0, key=f"p{n}")
            amps.append(a)
            phases.append(ph)

        if np.allclose(amps, 0.0):
            amps[0] = 1.0  # fallback biar gak kosong

        coeffs = np.array([amps[i] * np.exp(1j * phases[i]) for i in range(max_basis)])
        norm = np.sqrt(np.sum(np.abs(coeffs) ** 2))
        coeffs = coeffs / (norm if norm != 0 else 1.0)

        st.write("Koefisien normalisasi (cₙ):")
        for i, c in enumerate(coeffs, start=1):
            st.write(f"c_{i} = {c:.3f}")

    st.subheader("Waktu")
    t = st.slider("Waktu t (unit)", 0.0, 10.0, 0.0, step=0.01)

with col2:
    st.header("Informasi & Statistik")
    st.markdown("- Sistem: Infinite square well pada domain [0, L]\n- Basis: eigenfungsi sinus")
    st.markdown("### Energi setiap basis")
    energies = [E_infinite_well(n, L) for n in range(1, max_basis + 1)]
    for n, E in enumerate(energies, start=1):
        st.write(f"E_{n} = {E:.4f}")

# ==============================
# BANGUN ψ(x,t)
# ==============================
psi_xt = np.zeros_like(x, dtype=complex)
for n in range(1, max_basis + 1):
    En = E_infinite_well(n, L)
    psi_xt += coeffs[n-1] * phi_infinite_well(x, L, n) * np.exp(-1j * En * t / hbar)

# Normalisasi
norm = integrate.trapz(np.abs(psi_xt)**2, x)
psi_xt /= np.sqrt(norm)

# Probabilitas
prob_density = np.abs(psi_xt) ** 2

# Ekspektasi
x_expect = expectation_x(x, prob_density)

dx = x[1] - x[0]
psi_x_deriv = np.gradient(psi_xt, dx)
p_psi = -1j * hbar * psi_x_deriv
p_expect = np.trapz(np.conj(psi_xt) * p_psi, x).real

# ==============================
# OUTPUT
# ==============================
st.subheader("Hasil Perhitungan")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Normalisasi ∫|ψ|² dx", f"{norm:.6f}")
col_b.metric("⟨x⟩", f"{x_expect:.4f}")
col_c.metric("⟨p⟩", f"{p_expect:.4f}")

# Plot
fig, ax = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
ax[0].plot(x, psi_xt.real)
ax[0].set_ylabel('Re(ψ)')
ax[1].plot(x, psi_xt.imag)
ax[1].set_ylabel('Im(ψ)')
ax[2].plot(x, prob_density)
ax[2].set_ylabel('|ψ|²')
ax[2].set_xlabel('x')
plt.tight_layout()
st.pyplot(fig)

# --- End of file ---
