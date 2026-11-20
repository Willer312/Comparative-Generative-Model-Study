# Studi Komparatif Empiris Model Generatif (D6421 - Sesi 07)

Repositori resmi untuk Tugas Take-Home Deep Learning (D6421) — Studi komparatif empiris antara **Vanilla Autoencoder, Sparse Autoencoder, Variational Autoencoder (VAE), Vanilla GAN, DCGAN, dan DDPM**, disertai pembuktian empiris batasan **MLP Classifier** untuk *image generation*.

## Identitas Mahasiswa
* **Nama:** Willian Yehezkiel Alvin
* **NIM:** 2802419811
* **Kelas:** LA01
* **Dataset Benchmark:** Fashion-MNIST (28x28 Grayscale, Split 80/10/10, Seed = 42)

## Struktur Repositori
* `models/` — Definisi arsitektur PyTorch untuk MLP Classifier dan 6 model generatif.
* `utils/` — Utilitas *random seed*, *standardized data loader* (80/10/10), metrik evaluasi (MSE, PSNR, SSIM, FID, Latency), dan generator visualisasi.
* `train.py` — Skrip pelatihan terstandarisasi untuk seluruh model.
* `evaluate.py` — Skrip pengujian kuantitatif (Tabel IV), pembuktian MLP Classifier (Bagian 2), dan pembuatan Figur 1–5.

## Cara Menjalankan (Reproducibility)
1. **Instalasi Dependensi:**
   ```bash
   pip install -r requirements.txt