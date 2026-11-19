# Lembar Kontribusi Anggota & Pernyataan Penggunaan AI
### UTS Machine Learning - Hands-on Machine Learning: Regression, Classification, and Clustering
**Kelompok 16 - Kelas TK 47 03**

---

## 1. Identitas Anggota Kelompok

| No | Nama Lengkap | NIM | Kelas | Peran Utama | Akun GitHub |
|---|---|---|---|---|---|
| 1 | **Fransisco Sitepu** | **101032300004** | **TK 47 03** | **Task 1 (Regression) & Task 3 (Clustering)** | [@Frans1828](https://github.com/Frans1828) |
| 2 | **Ahmad Ghazali El-Faruqi HSB** | **101032330080** | **TK 47 03** | **Task 2 (Classification)** | [@AhmadGhazaliELFaruqiHsb](https://github.com/AhmadGhazaliELFaruqiHsb) |

---

## 2. Rincian Kontribusi Anggota (Work Breakdown)

### A. Fransisco Sitepu (NIM: 101032300004)
- **Task 1 – Regression (Delivery Time Estimation)**:
  - Melakukan audit data mendalam: mengidentifikasi *data leakage* kritis pada fitur `jumlah_scan` (distribusi di data latih rata-rata 9.65 vs data uji 1.50) dan membuktikan dampaknya secara kuantitatif (error meledak dari MAE 18.74 ke 30.48 jika memakai fitur bocor).
  - Merancang skema validasi berbasis waktu (*time-based split*: Jan–Jun 2025 sebagai train, Jul–Agu 2025 sebagai validation) untuk meniru kondisi prediksi masa depan.
  - Membangun model kuantil *LightGBM* untuk memprediksi median ($P50$) dan batas atas ($P90$).
  - Menerapkan kalibrasi *Conformal Prediction* sehingga cakupan empiris $P90$ mencapai target persis $\ge 90.00\%$.
  - Mengatasi kategori kardinalitas tinggi dan kemunculan kategori baru (`mitra_kurir = 'K13'`) di data uji.
  - Menghasilkan file prediksi resmi `prediksi_regresi.csv` (60.000 baris, $P90 \ge P50$).
- **Task 3 – Clustering (Customer Segmentation)**:
  - Melakukan prapemrosesan data komposisional menggunakan transformasi *Centered Log-Ratio* (CLR) untuk 16 kategori belanja dan 4 interval jam belanja.
  - Menangani *skewness* moneter (`belanja_90h`, `recency_hari`, dll.) dengan transformasi logaritmik dan mengimputasi nilai hilang pada `rata2_keranjang` pelanggan inaktif.
  - Menemukan bahwa kolom `n01`–`n15` merupakan *synthetic noise* dan melakukan seleksi fitur domain RFM + perilaku.
  - Menerapkan *Isolation Forest* untuk menandai 1.800 pelanggan anomali (1.50%) sebagai label `-1`.
  - Membandingkan algoritma *K-Means*, *Gaussian Mixture Model* (GMM), dan *Agglomerative Clustering* serta mengevaluasi $K$ optimal ($K=5$) via skor *Silhouette*, *Davies-Bouldin*, dan *Calinski-Harabasz*.
  - Melakukan uji stabilitas klaster menggunakan *bootstrap subsampling* dengan hasil rata-rata *Adjusted Rand Index* (ARI) tinggi.
  - Menyusun profil persona bisnis untuk 5 segmen utama serta rekomendasi aksi pemasaran.
  - Menghasilkan file resmi `cluster.csv` (120.000 pelanggan).
- **Infrastruktur Repositori & Dokumentasi**:
  - Inisialisasi struktur repositori, otomatisasi satu perintah (`run_all.py`), dan penyusunan laporan teknis.

### B. Ahmad Ghazali El-Faruqi HSB (NIM: 101032330080)
- **Task 2 – Classification (Fraud Detection)**:
  - Audit label: mengukur *reporting lag* (kematangan label 3 minggu terakhir Agustus 68% / 52% / 14%).
  - Audit fitur: membuktikan kebocoran `flag_review_manual` (PR-AUC semu 0,926 → 0,774 saat flag belum tersedia) dan membuang `v15` (proksi waktu); seleksi fitur anonim `v**` di dalam fold latih.
  - Validasi berbasis waktu (Fold A: Feb–Jun → Juli, Fold B: Feb–Jul → 1–10 Agustus), perbandingan dengan split acak, serta simulasi label lag untuk memilih strategi data latih.
  - Rekayasa fitur perilaku pengguna & merchant tanpa informasi masa depan (jendela waktu 24 jam / 7 hari / 30 hari).
  - Perbandingan penanganan ketidakseimbangan kelas: tanpa penanganan, *class weighting*, *random undersampling*, *random oversampling*, dan *threshold tuning*.
  - Analisis pergeseran pola fraud (lonjakan fraud QRIS di merchant baru sejak Agustus 2025) dan adaptasi dengan bobot *recency*.
  - Kalibrasi probabilitas out-of-time (Platt vs isotonic) dan aturan keputusan berbasis biaya, dibandingkan dengan "tandai semua" dan "tidak tandai".
  - Analisis galat per kanal, kategori merchant, dan pola fraud.
  - Menghasilkan `prediksi_klasifikasi.csv`, `kode/hasil_task2/`, dan halaman Task 2 di `laporan.pdf` (via `kode/laporan_task2.py`).
- **Pull request**: [daftar PR Ahmad ke repo kelompok](https://github.com/Frans1828/UTS-Machine-Learning/pulls?q=is%3Apr+author%3AAhmadGhazaliELFaruqiHsb)

---

## 3. Pernyataan Penggunaan Alat AI (AI Tools Disclosure)

Sesuai dengan ketentuan ujian: *"AI tools: allowed. State the tool, the parts it helped with, and how the group verified the output."*

1. **Alat AI yang Digunakan**:
   - Google Antigravity / Gemini & Claude LLM.
2. **Bagian yang Dibantu**:
   - Membantu eksplorasi matematis formulasi *Lagrange Multiplier* dan penurunan metrik *Pinball Loss*.
   - Membantu *brainstorming* rekayasa fitur komposisional (metodologi *Centered Log-Ratio* / CLR dari Aitchison).
   - Membantu penulisan kerangka kode otomatisasi dan perapian struktur laporan akademis.
   - Task 2: Claude (Anthropic) membantu menyusun kode pipeline `kode/task2_classification.py`, generator laporan `kode/laporan_task2.py`, dan draf teks bagian Task 2.
3. **Metode Verifikasi oleh Anggota**:
   - Task 2: setiap angka di laporan dibaca langsung dari `kode/hasil_task2/ringkasan_task2.json` dan tabel CSV yang dihasilkan kode; format `prediksi_klasifikasi.csv` dicek otomatis (106.674 baris unik, `prob` di [0,1], `tandai` 0/1).
   - Semua angka, metrik validasi (MAE, Pinball loss, cakupan kuantil, skor Silhouette, ARI), dan tabel yang dilaporkan dieksekusi secara independen dan diverifikasi langsung pada dataset lokal.
   - Tidak ada angka atau prediksi yang dihasilkan secara fiktif; seluruh file `prediksi_regresi.csv`, `prediksi_klasifikasi.csv`, dan `cluster.csv` diproduksi secara deterministik melalui kode Python yang dapat direproduksi penuh dengan satu perintah (`python kode/run_all.py`).

---

## 4. Persetujuan Anggota Kelompok

Dokumen ini disusun dan disetujui bersama oleh seluruh anggota Kelompok 16:
- (Ttd) **Fransisco Sitepu** (101032300004)
- (Ttd) **Ahmad Ghazali El-Faruqi HSB** (101032330080)
