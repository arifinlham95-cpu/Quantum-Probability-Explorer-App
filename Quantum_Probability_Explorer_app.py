"""
Quantum Probability Explorer
Streamlit app (single-file): Quantum_Probability_Explorer_app.py

README (top of file):

Purpose:
- Interactive visualization for 1D wavefunctions (infinite square well eigenstates and their superpositions).
- Shows real(ψ), imag(ψ), and probability density |ψ|^2, plus normalization and expectation values <x>, <p>.
- Designed for teaching materi fungsi gelombang (SMA/undergraduate intro).

Requirements (put into requirements.txt):
streamlit
numpy
scipy
matplotlib

How to run locally:
1. Create virtual environment (recommended):
   python -m venv .venv
   source .venv/bin/activate   # mac/linux
   .venv\Scripts\activate     # windows
2. Install requirements:
   pip install -r requirements.txt
3. Save this file as Quantum_Probability_Explorer_app.py
4. Run:
   streamlit run Quantum_Probability_Explorer_app.py

Simple GitHub + Streamlit Cloud deploy steps:
1. Create a new GitHub repo (e.g., Quantum-Probability-Explorer).
2. Add this file and a requirements.txt (see above). Also add a short README.md and .gitignore.
3. Commit & push:
   git init
   git add .
   git commit -m "Initial commit: Quantum Probability Explorer Streamlit app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo>.git
   git push -u origin main
4. Deploy to Streamlit Cloud (https://share.streamlit.io):
   - Sign in with GitHub, choose your repo and branch, and start the app.
   - Alternatively, use other hosting (Heroku, Railway) but Streamlit Cloud is easiest for Streamlit apps.

Notes for educators:
- Units: code uses atomic-style units with hbar=1 and m=1 so students focus on shape and probabilities.
- You can change to physical units by modifying hbar and m.

Possible extensions:
- Add other potentials (harmonic oscillator numeric solver using finite-difference + eigensolver).
- Show interactive decomposition of a Gaussian wavepacket into eigenstates (Fourier coefficients).
- Add explanation text and exercises for LMS.

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
    # eigenfunction for infinite square well [0, L]
    return np.sqrt(2.0 / L) * np.sin(n * np.pi * x / L)

def E_infinite_well(n, L):
    return (n ** 2 * np.pi ** 2 * hbar ** 2) / (2.0 * m * L ** 2)

def expectation_x(x, prob):
    return np.trapz(x * prob, x)

def expectation_p_from_basis(coeffs, n_vals, L):
    # For infinite well basis, <p> in stationary basis is 0 for real coefficients
    # But with complex phases we can estimate via derivative in x-space (numerical)
    return None

# UI layout
st.title("Quantum Probability Explorer — Fungsi Gelombang (Sum of Infinite-Well Eigenstates)")
st.markdown("Aplikasi interaktif untuk mengeksplor probabilitas gelombang dalam kotak tak hingga. Units: hbar=1, m=1.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Pengaturan Sistem")
    L = st.slider("Panjang kotak L", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
    x_points = st.slider("Jumlah titik x (resolusi)", 200, 2000, 800)
    x = np.linspace(0, L, x_points)

    st.subheader("Mode basis")
    max_basis = st.slider("Jumlah basis (N) untuk superposisi", 1, 8, 4)

    st.markdown("**Koefisien kompleks untuk masing-masing basis** — atur amplitudo & fase lalu normalisasi otomatis.")
    amps = []
    phases = []
    for n in range(1, max_basis + 1):
        st.write(f"Basis n = {n}")
        col_a, col_p = st.columns([1, 2])
        with col_a:
            a = st.slider(f"Amplitudo a_{n}", 0.0, 1.0, 0.0, key=f"a{n}")
        with col_p:
            ph = st.slider(f"Fase φ_{n} (rad)", 0.0, 2 * np.pi, 0.0, key=f"p{n}")
        amps.append(a)
        phases.append(ph)

    # If all amplitudes are zero, default to n=1
    if np.allclose(amps, 0.0):
        amps[0] = 1.0

    # build complex coefficients and normalize
    coeffs = np.array([amps[i] * np.exp(1j * phases[i]) for i in range(max_basis)])
    norm = np.sqrt(np.sum(np.abs(coeffs) ** 2))
    if norm == 0:
        norm = 1.0
    coeffs = coeffs / norm

    st.write("Koefisien normalisasi (c_n):")
    for i, c in enumerate(coeffs, start=1):
        st.write(f"c_{i} = {c:.3f}")

    st.subheader("Waktu")
    t = st.slider("Waktu t (unit)", 0.0, 10.0, 0.0, step=0.01)
    animate = st.checkbox("Animasi (auto-step waktu saat menggeser)")

with col2:
    st.header("Informasi & Statistik")
    st.markdown("- Sistem: Infinite square well pada domain [0, L]\n- Basis: eigenfungsi sinus")
    st.markdown("### Energi setiap basis")
    energies = [E_infinite_well(n, L) for n in range(1, max_basis + 1)]
    for n, E in enumerate(energies, start=1):
        st.write(f"E_{n} = {E:.4f}")

# Build psi(x, t)
phis = np.array([phi_infinite_well(x, L, n) for n in range(1, max_basis + 1)])
Es = np.array([E_infinite_well(n, L) for n in range(1, max_basis + 1)])

# time evolution factor for each component
time_factors = np.exp(-1j * Es * t / hbar)

# psi(x,t) = sum c_n phi_n(x) e^{-i E_n t / hbar}
psi_xt = np.tensordot(coeffs * time_factors, phis, axes=(0, 0))
prob_density = np.abs(psi_xt) ** 2

# normalization check
norm_xt = np.trapz(prob_density, x)

# expectation values
x_expect = expectation_x(x, prob_density)

# numerical momentum expectation via derivative (−i ħ ∂/∂x)
# compute derivative using central differences
dx = x[1] - x[0]
psi_x_deriv = np.gradient(psi_xt, dx)
p_psi = -1j * hbar * psi_x_deriv
p_expect = integrate.trapz(np.conj(psi_xt) * p_psi, x).real

# Display values
st.subheader("Hasil perhitungan")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Normalisasi ∫|ψ|^2 dx", f"{norm_xt:.6f}")
col_b.metric("⟨x⟩", f"{x_expect:.4f}")
col_c.metric("⟨p⟩", f"{p_expect:.4f}")

# Plots
fig, ax = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
ax[0].plot(x, psi_xt.real)
ax[0].set_ylabel('Re(ψ)')
ax[1].plot(x, psi_xt.imag)
ax[1].set_ylabel('Im(ψ)')
ax[2].plot(x, prob_density)
ax[2].set_ylabel('|ψ|^2')
ax[2].set_xlabel('x')
plt.tight_layout()

st.pyplot(fig)

st.markdown("---")
st.subheader("Instruksi untuk guru / catatan pembelajaran")
st.markdown("1. Minta siswa coba mengatur amplitudo/fase c_n untuk melihat bagaimana pola |ψ|^2 berubah (interferensi konstruktif/destruktif).\n2. Tunjukkan bahwa untuk eigenstates tunggal probabilitas stabil (waktu-independen).\n3. Tanyakan bagaimana ⟨x⟩ dan ⟨p⟩ berubah bila koefisien kompleks berbeda fase.")

st.markdown("---")
st.markdown("Jika Anda ingin saya tambahkan fitur: harmonic oscillator numerik, dekomposisi Gaussian, atau export data CSV untuk LMS, beri tahu saya — saya akan tambahkan ke aplikasi ini.")

# End of file

