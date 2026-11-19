# Panduan Reproduksi Kode - UTS Machine Learning

**Kelompok 16:**
- **Fransisco Sitepu** (NIM: 101032300004) - Task 1 & Task 3
- **Ahmad Ghazali El-Faruqi HSB** (NIM: 101032330080) - Task 2

---

## 1. Persyaratan Lingkungan (Environment Requirements)

- **Python**: 3.10+ atau 3.11+
- **Library Utama**:
  ```bash
  pip install pandas numpy scikit-learn lightgbm scipy fpdf2 pypdf matplotlib
  ```

---

## 2. Satu Perintah Reproduksi (One-Command Reproducibility)

Sesuai instruksi ujian: *"One command must reproduce every number in the report."*

Jalankan perintah berikut dari folder root repositori:

```bash
python kode/run_all.py
```

Perintah di atas secara otomatis akan:
1. Memuat seluruh dataset dari direktori data lokal.
2. Menjalankan **Task 1** (Audit kebocoran `jumlah_scan`, validasi berbasis waktu Jan–Jun vs Jul–Agu, pelatihan model kuantil LightGBM P50 dan P90, kalibrasi *Conformal Prediction*, dan menghasilkan `prediksi_regresi.csv`).
3. Menjalankan **Task 2** (audit label lag & kebocoran `flag_review_manual`, validasi berbasis waktu, fitur perilaku pengguna/merchant tanpa informasi masa depan, perbandingan penanganan imbalance, analisis pergeseran pola, kalibrasi out-of-time, aturan keputusan berbasis biaya, analisis galat; menghasilkan `prediksi_klasifikasi.csv`, seluruh tabel/grafik di `kode/hasil_task2/`, lalu memperbarui bagian Task 2 di `laporan.pdf` lewat `kode/laporan_task2.py`).
4. Menjalankan **Task 3** (Transformasi data komposisional *Centered Log-Ratio* (CLR), deteksi anomali *Isolation Forest*, *K-Means* $K=5$, uji stabilitas *Adjusted Rand Index*, dan menghasilkan `cluster.csv`).

---

## 3. Menjalankan Script Secara Terpisah (Per-Task Execution)

Jika ingin menjalankan tugas tertentu secara independen:

### Task 1: Regresi Estimasi Waktu Pengiriman
```bash
python kode/task1_regression.py
```
- Output: `prediksi_regresi.csv` (kolom: `id_paket`, `p50`, `p90`)

### Task 2: Klasifikasi Deteksi Fraud
```bash
python kode/task2_classification.py            # opsional: path folder data sebagai argumen
python kode/laporan_task2.py                   # perbarui halaman Task 2 di laporan.pdf dari hasil di atas
```
- Output: `prediksi_klasifikasi.csv` (kolom: `id_transaksi`, `prob`, `tandai`)
- Semua angka laporan Task 2: `kode/hasil_task2/ringkasan_task2.json` dan tabel `kode/hasil_task2/*.csv`
- Grafik: `kode/hasil_task2/fig_task2_*.png`
- Waktu eksekusi ± 7 menit (2 CPU), seed 42, deterministik.

### Task 3: Klasterisasi Segmentasi Pelanggan
```bash
python kode/task3_clustering.py
```
- Output: `cluster.csv` (kolom: `id_pelanggan`, `cluster`)

---

## 4. Random Seeds & Konsistensi Numerik

Semua eksperimen dan model menggunakan seed tetap untuk menjamin determinisme:
- `random_state = 42` (Model training, split, subsampling)
- Validasi berbasis waktu memastikan tidak ada *look-ahead bias* dan tidak terjadi *data leakage*.
