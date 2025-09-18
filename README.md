# Quantum-Probability-Explorer-App
Streamlit app for exploring quantum wavefunctions and probability distributions in an infinite square well.
# Quantum Probability Explorer

Aplikasi interaktif berbasis **Streamlit** untuk mengeksplorasi fungsi gelombang kuantum 1D pada **kotak tak hingga (infinite square well)**.  
Didesain untuk mendukung pembelajaran konsep **probabilitas kuantum** dan **fungsi gelombang** pada mahasiswa atau siswa tingkat lanjut.

---

## ✨ Fitur
- Visualisasi **fungsi gelombang** ψ(x,t) sebagai kombinasi linear dari eigenstate kotak tak hingga.
- Plot interaktif:
  - Bagian real Re(ψ)
  - Bagian imajiner Im(ψ)
  - Densitas probabilitas |ψ|²
- Pengaturan **amplitudo** dan **fase** masing-masing basis.
- Evolusi waktu ψ(x,t) dengan slider waktu t.
- Perhitungan numerik:
  - Normalisasi ∫ |ψ|² dx
  - Nilai harapan posisi ⟨x⟩
  - Nilai harapan momentum ⟨p⟩
- Tampilan energi eigenstate (Eₙ).

---

## 🛠️ Instalasi Lokal

1. Clone repositori:
   ```bash
   git clone https://github.com/<username>/Quantum-Probability-Explorer.git
   cd Quantum-Probability-Explorer

