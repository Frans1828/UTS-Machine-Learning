# UTS Machine Learning - Kelompok 16

> **Hands-on Machine Learning: Regression, Classification, and Clustering**
> Ujian Tengah Semester (UTS) - Program Studi Teknik Komputer, Telkom University

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Identitas Anggota Kelompok

| No | Nama Lengkap | NIM | Kelas | Tanggung Jawab Utama |
|:---:|:---|:---:|:---:|:---|
| 1 | **Fransisco Sitepu** | **101032300004** | **TK 47 03** | **Task 1 (Regression) & Task 3 (Clustering)** |
| 2 | **Ahmad Ghazali El-Faruqi HSB** | **101032330080** | **TK 47 03** | **Task 2 (Classification); perbaikan Task 1 & 3; generator laporan** |

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
    ├── task3_clustering.py       # Pipeline Task 3 (Klasterisasi Segmentasi Pelanggan)
    ├── laporan.py                # Membangun seluruh laporan.pdf dari hasil_task1-3/
    ├── laporan_task2.py          # Bagian Task 2 laporan + gaya dokumen
    ├── hasil_task1/ hasil_task2/ hasil_task3/   # Semua tabel, grafik & ringkasan angka laporan
    ├── run_all.py                # Skrip master untuk reproduksi penuh satu perintah
    └── README.md                 # Petunjuk instalasi dependencies & eksekusi kode
```

---

## Ringkasan Eksekutif & Hasil Utama

### Task 1: Estimasi Waktu Pengiriman Paket (Regression)
- **Target**: `durasi_jam` paket September–Oktober 2025 sebagai median (P50) dan batas atas 90% (P90) saat paket diambil kurir.
- **Audit Data**:
  - *Kebocoran* `jumlah_scan` (rata-rata 9,65 di data latih vs 1,50 saat pickup): MAE semu **10,89 jam** melonjak ke **21,60 jam** bila scan diisi seperti saat pickup, lebih buruk dari model jujur (**13,77 jam**). Kolom dibuang.
  - *Label bersatuan menit*: 370 durasi > 1.000 jam (maks 25.452) — bila dibagi 60, mediannya cocok dengan median normal per layanan → dikoreksi, bukan dipotong.
  - Sentinel `jarak_km` (0 & 9999) → missing; `berat_kg` hilang 5,56% dan tidak acak (59% kargo) → indikator; `f30` proksi waktu → dibuang.
- **Validasi**: latih Jan–Jun → uji Jul–Agu (horizon 2 bulan), tuning hanya di Jan–Apr → Mei–Jun. Random 5-fold CV sedikit optimis (MAE 13,27 vs 13,77 jam).
- **Model**: baseline kuantil per layanan×pulau (MAE 20,97) → Ridge target log (15,37) → **LightGBM quantile tuned (13,77; pinball q0,5 6,886; q0,9 4,471)**.
- **Kategori**: `kota_tujuan` (405 kategori) native LightGBM + smoothing; simulasi kurir tak dikenal (pengganti K13) hanya menambah MAE 0,24 jam.
- **Kalibrasi conformal (CQR)**: offset dari Juli (+5,40 jam), diuji di Agustus → cakupan P90 **86,96% → 90,80%**.
- **Analisis**: galat per layanan/pulau/kurir (kurir K07 cakupan P90 hanya 68%), SHAP & partial dependence (layanan, jarak, kota tujuan).
- **File output**: `prediksi_regresi.csv` (60.000 baris, P90 ≥ P50), semua angka di `kode/hasil_task1/`.

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
- **Prapemrosesan**: log1p (variabel uang/hitungan), logit (persen), CLR untuk 16 porsi kategori & 4 porsi jam, one-hot perangkat, imputasi `rata2_keranjang` (NaN hanya saat frekuensi 0), winsor 0,5/99,5% + standardisasi.
- **Seleksi fitur dengan bukti**: `n01–n15` independen dari semua fitur domain (|ρ| maks 0,02) → noise; fitur redundan (belanja = frekuensi × keranjang, versi 30 hari) dibuang; relevansi η² konsisten di K=5..8 → 31 fitur. Silhouette 0,041 → 0,067 dan stabilitas 0,845 → 0,961 dibanding memakai semua kolom.
- **Anomali**: Isolation Forest dengan ambang titik siku → **2.462 pelanggan (2,05%)** berlabel `-1` (pola bot: median 158,6 sesi/minggu, konversi 0,01).
- **Jumlah klaster**: K=3..10 dinilai dengan silhouette, CH, DB, BIC, dan stabilitas bootstrap; aturan "K terbesar dengan ARI ≥ 0,95" → **K = 7** (K=8 turun ke 0,80).
- **Metode**: K-Means vs Gaussian Mixture vs Agglomerative Ward; K-Means terpilih (stabilitas 0,961, sepakat dengan Ward ARI 0,842). Stabilitas final 10 bootstrap: **ARI 0,967 ± 0,004**.
- **7 segmen**: Pemburu Voucher/Flash Sale (22,7%), Dorman (18,9%), Keluarga Ibu & Bayi (15,6%), Premium Keranjang Besar (12,2%), Paylater & Sering Retur (9,8%), Pelanggan Setia Konversi Tinggi (9,7%), Member Baru (9,0%) — masing-masing dengan aksi pemasaran di laporan.
- **File output**: `cluster.csv` (120.000 baris, `-1` = anomali), semua angka di `kode/hasil_task3/`.

---

## Reproduksi Hasil (How to Reproduce)

Untuk mereproduksi seluruh eksperimen dan memastikan seluruh angka dalam laporan dapat dihasilkan kembali dalam satu perintah:

```bash
# Dari root directory repositori
python kode/run_all.py
```
