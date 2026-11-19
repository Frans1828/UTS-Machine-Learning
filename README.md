# UTS Machine Learning - Kelompok 16

> **Hands-on Machine Learning: Regression, Classification, and Clustering**
> Ujian Tengah Semester (UTS) - Program Studi Teknik Komputer, Telkom University

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Identitas Anggota Kelompok

| No | Nama Lengkap | NIM | Kelas | Tanggung Jawab Utama |
|:---:|:---|:---:|:---:|:---|
| 1 | **Fransisco Sitepu** | **101032300004** | **TK 47 03** | **Task 1 (Regression) & Task 3 (Clustering)** |
| 2 | **Ahmad Ghazali El-Faruqi HSB** | **101032330080** | **TK 47 03** | **Task 2 (Classification)** |

---

## Struktur Repositori

```plaintext
UTS-Machine-Learning/
├── laporan.pdf                   # Laporan lengkap UTS menjawab Task 1–3
├── kontribusi.md                 # Rincian pembagian tugas & pengungkapan alat AI
├── prediksi_regresi.csv          # Hasil prediksi Task 1 (id_paket, p50, p90)
├── prediksi_klasifikasi.csv      # Hasil prediksi Task 2 (id_transaksi, prob, tandai)
├── cluster.csv                   # Hasil klasterisasi Task 3 (id_pelanggan, cluster)
├── README.md                     # Ringkasan proyek dan arsitektur repositori
└── kode/                         # Seluruh skrip pipeline dan modul implementasi
    ├── task1_regression.py       # Pipeline Task 1 (Regresi Estimasi Waktu Pengiriman)
    ├── task2_classification.py   # Pipeline Task 2 (Klasifikasi Deteksi Fraud)
    ├── laporan_task2.py          # Generator halaman Task 2 di laporan.pdf dari hasil_task2/
    ├── hasil_task2/              # Semua tabel, grafik & ringkasan angka Task 2
    ├── task3_clustering.py       # Pipeline Task 3 (Klasterisasi Segmentasi Pelanggan)
    ├── run_all.py                # Skrip master untuk reproduksi penuh satu perintah
    └── README.md                 # Petunjuk instalasi dependencies & eksekusi kode
```

---

## Ringkasan Eksekutif & Hasil Utama

### Task 1: Estimasi Waktu Pengiriman Paket (Regression)
- **Target**: Memprediksi `durasi_jam` paket September–Oktober 2025 dalam bentuk median ($P50$) dan batas atas 90% ($P90$) saat paket diambil kurir.
- **Audit Data & Kebocoran (Data Leakage)**:
  - Ditemukan kebocoran data fatal pada fitur `jumlah_scan`: rata-rata data latih sebesar **9.65 scan** (karena dicatat pasca-paket sampai), sedangkan pada saat *pickup* (data uji) hanya **1.50 scan**.
  - Eksperimen menunjukkan model yang memanfaatkan `jumlah_scan` mengalami lonjakan error drastis (MAE meledak dari **18.74** menjadi **30.48 jam** saat diuji pada kondisi data realistis). Model bersih tanpa `jumlah_scan` memberikan performa stabil dengan MAE **21.50 jam**.
  - Fitur `berat_kg` yang hilang (5.56%) diimputasi dengan median.
- **Validasi Berbasis Waktu**:
  - Pelatihan pada Januari–Juni 2025 (181.396 baris) dan pengujian pada Juli–Agustus 2025 (68.604 baris) untuk merefleksikan horizon prediksi 2 bulan mendatang.
- **Model & Ketidakpastian**:
  - Model terbaik menggunakan *LightGBM Quantile Regression* ($q=0.5$ dan $q=0.9$).
  - Baseline *Ridge (log-target)* mencatat MAE 28.27 jam, sedangkan LightGBM P50 mencapai Pinball Loss **10.75** (MAE 21.50 jam).
  - Cakupan empiris awal $P90$ sebesar 87.24% dikalibrasi menggunakan metode *Conformal Prediction* (offset $+3.96$ jam) sehingga mencapai **cakupan tepat 90.00%** dengan Pinball Loss **11.31**.
  - Kategori baru pada data uji (`mitra_kurir = 'K13'`) ditangani dengan mulus melalui *LightGBM categorical fallback*.
  - File output: `prediksi_regresi.csv` (60.000 baris, terverifikasi $P90 \ge P50$ 100%).

### Task 2: Deteksi Transaksi Fraud (Classification)
- **Target**: Memprediksi probabilitas `is_fraud` transaksi September–Oktober 2025 dan menentukan transaksi yang ditandai (`tandai = 1`). Data latih 403.077 transaksi (prevalensi fraud **0,65%**), data uji 106.674 transaksi.
- **Audit Label & Fitur**:
  - *Label lag*: kematangan label 3 minggu terakhir Agustus hanya **68% → 52% → 14%** dari tingkat fraud normal (fraud belum dilaporkan saat data ditarik).
  - *Kebocoran* `flag_review_manual` (hasil tinjauan manual setelah transaksi; 79,9% fraud ber-flag 1): PR-AUC semu **0,926**, tetapi saat flag belum ada turun ke **0,774** dan biayanya lebih buruk dari model jujur. Kolom dibuang.
  - `v15` adalah proksi waktu (naik linear tiap bulan, AUC 0,51) → dibuang; hanya 7 dari 49 fitur anonim `v**` yang informatif dipertahankan.
- **Validasi Berbasis Waktu**: Fold A (latih Feb–Jun, validasi Juli) dan Fold B (latih Feb–Jul, validasi 1–10 Agustus). Split acak terlalu optimis (PR-AUC 0,855 vs 0,787). Simulasi label lag menunjukkan strategi terbaik adalah **membuang 3 minggu terakhir yang labelnya belum matang** (model final dilatih 1 Feb–10 Agu).
- **Fitur Perilaku** (tanpa informasi masa depan): riwayat pengguna (rata-rata/maks nominal sebelumnya, merchant/kategori/kanal baru) dan riwayat merchant (jumlah & nominal transaksi 24 jam/7 hari/30 hari, pengguna unik, umur merchant di data). Baseline logistic regression PR-AUC 0,615 → LightGBM + fitur perilaku **0,787** (Fold A) / **0,531** (Fold B).
- **Ketidakseimbangan Kelas**: dibandingkan tanpa penanganan, *class weighting*, *random undersampling* 1:20, *random oversampling* ×20, plus *threshold tuning*. Resampling/weighting tidak konsisten menaikkan PR-AUC dan merusak kalibrasi; yang paling menentukan biaya adalah ambang berbasis biaya.
- **Pergeseran Pola**: fraud QRIS melonjak dari 10–16 kasus/bulan menjadi **92 kasus di Agustus** — merchant QRIS baru (median umur 19 hari) yang menguras saldo akun lama. Adaptasi: fitur riwayat merchant, bobot *recency* (half-life 90 hari, PR-AUC Fold B 0,531 → 0,549), dan kalibrasi out-of-time.
- **Kalibrasi & Keputusan**: kalibrasi isotonic (+ tie-break agar urutan skor tetap) dilatih pada Juli, diuji pada Agustus. Aturan: tandai jika $P(\text{fraud}) \times \text{nominal} \ge 25.000$. Di Fold B: tidak tandai Rp 138,5 jt, tandai semua Rp 435,9 jt, **aturan biaya Rp 39,3 jt**.
- **Analisis Galat**: kuat pada pola lama (kartu tidak hadir, perangkat baru, lokasi jauh; PR-AUC > 0,8); kelemahan utama pada pola baru QRIS merchant muda (recall ~0,45).
- **File output**: `prediksi_klasifikasi.csv` (106.674 transaksi) dan seluruh tabel/grafik di `kode/hasil_task2/`.

### Task 3: Segmentasi Pelanggan E-Commerce (Clustering)
- **Dataset**: 120.000 pelanggan, ringkasan 90 hari terakhir.
- **Prapemrosesan Khusus**:
  - Variabel komposisional (16 proporsi kategori belanja & 4 proporsi waktu belanja yang berjumlah 1) ditransformasi menggunakan *Centered Log-Ratio* (CLR).
  - Variabel bernilai moneter diskalakan secara non-linear dengan $\ln(1+x)$.
  - Nilai hilang pada `rata2_keranjang` (6.736 baris) terjadi karena `frekuensi_90h == 0` (pelanggan inaktif) dan diimputasi dengan 0 serta ditambahkan indikator biner `is_inactive_90h`.
- **Seleksi Fitur Ilmiah**:
  - Fitur anonim `n01`–`n15` terbukti merupakan *synthetic noise* acak (distribusi seragam $U[0, 100]$, $U[-500, 500]$, dll.). Fitur-fitur ini disaring demi meningkatkan kualitas klaster.
- **Deteksi Anomali**:
  - Algoritma *Isolation Forest* mengidentifikasi **1.800 pelanggan (1.50%)** sebagai anomali ekstrem (pembeli grosir/bot/frekuensi ekstrem) dan diberi label `-1`.
- **Evaluasi Jumlah Klaster ($K=5$)**:
  - Dievaluasi menggunakan skor *Silhouette* (0.062), *Davies-Bouldin* (3.18), dan *Calinski-Harabasz* (1064.7).
  - Uji stabilitas *bootstrap subsampling* mencatat nilai rata-rata *Adjusted Rand Index* (ARI) yang kuat (**0.53+**).
- **Profil 5 Segmen Pelanggan**:
  1. **Segment 0 (32.9% - Deal Seekers / Flash Sale Chasers)**: Sensitif diskon, pengguna voucher 70%, usia muda (~27 tahun), kategori fashion.
  2. **Segment 1 (4.4% - Inactive / Churn-Risk Dormant)**: Frekuensi & belanja 90 hari nol, perlu kampanye *win-back* re-aktivasi.
  3. **Segment 2 (15.9% - High-Value Loyal Champions)**: Frekuensi tinggi (25.7 transaksi), belanja tertinggi (Rp 9.29 Juta), retensi tinggi.
  4. **Segment 3 (15.5% - App Power Browsers)**: Aktivitas sesi aplikasi luar biasa tinggi (**27.7 sesi/minggu**), konversi tinggi, pembeli impulsif.
  5. **Segment 4 (29.8% - High-Ticket Mature Planners)**: Nilai keranjang terbesar (Rp 408.000), usia matang (~38 tahun), pembeli terencana kategori elektronik & perabot.
- **File output**: `cluster.csv` (120.000 baris, termasuk anomali `-1`).

---

## Reproduksi Hasil (How to Reproduce)

Untuk mereproduksi seluruh eksperimen dan memastikan seluruh angka dalam laporan dapat dihasilkan kembali dalam satu perintah:

```bash
# Dari root directory repositori
python kode/run_all.py
```
