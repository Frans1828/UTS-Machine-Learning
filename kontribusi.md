# Lembar Kontribusi Anggota & Pernyataan Penggunaan AI
### UTS Machine Learning - Hands-on Machine Learning: Regression, Classification, and Clustering
**Kelompok 16 - Kelas TK 47 03**

---

## 1. Identitas Anggota Kelompok

| No | Nama Lengkap | NIM | Kelas | Peran Utama | Akun GitHub |
|---|---|---|---|---|---|
| 1 | **Fransisco Sitepu** | **101032300004** | **TK 47 03** | **Task 1 (Regression) & Task 3 (Clustering)** | [@Frans1828](https://github.com/Frans1828) |
| 2 | **Ahmad Ghazali El-Faruqi HSB** | **101032330080** | **TK 47 03** | **Task 2 (Classification); perbaikan Task 1 & 3; generator laporan** | [@AhmadGhazaliELFaruqiHsb](https://github.com/AhmadGhazaliELFaruqiHsb) |

---

## 2. Rincian Kontribusi Anggota (Work Breakdown)

### A. Fransisco Sitepu (NIM: 101032300004)
- **Inisialisasi repositori kelompok** dan versi awal seluruh pipeline (commit `b3d0267`): struktur repo, `run_all.py`, laporan versi awal.
- **Task 1 – Regression (versi awal)**:
  - Mengidentifikasi kebocoran `jumlah_scan` (rata-rata 9,65 di data latih vs 1,50 saat pickup) dan membuangnya dari model.
  - Merancang validasi berbasis waktu (latih Jan–Jun 2025, validasi Jul–Agu 2025).
  - Membangun baseline Ridge (target log) dan LightGBM quantile P50/P90 dengan offset conformal.
  - Menangani kategori baru `mitra_kurir = 'K13'` lewat kategori native LightGBM.
- **Task 3 – Clustering (versi awal)**:
  - Transformasi CLR untuk 16 porsi kategori dan 4 porsi jam, log1p untuk variabel miring, one-hot perangkat.
  - Menemukan bahwa `n01`–`n15` adalah noise dan mengeluarkannya.
  - Isolation Forest untuk label anomali `-1` dan K-Means sebagai metode utama, beserta profil persona segmen.
- Meminta perbaikan Task 1 & Task 3 agar seluruh angka laporan dihasilkan oleh kode (dikerjakan pada PR Ahmad, lihat bagian B).
- **Pull request**: [daftar PR Frans ke repo kelompok](https://github.com/Frans1828/UTS-Machine-Learning/pulls?q=is%3Apr+author%3AFrans1828)

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
- **Perbaikan Task 1 – Regression** (atas permintaan Fransisco):
  - Menulis ulang `kode/task1_regression.py` agar semua poin a–f dihitung oleh kode: dampak kebocoran `jumlah_scan`, koreksi label bersatuan menit (370 baris > 1.000 jam), sentinel `jarak_km`, `berat_kg` hilang tidak acak, `f30` proksi waktu.
  - Random 5-fold CV vs split waktu, tuning LightGBM di dalam lipatan latih, baseline + 2 model lanjutan (Ridge log-target, LightGBM quantile).
  - Simulasi kategori baru (kurir dikeluarkan dari data latih), kalibrasi conformal dengan data kalibrasi (Juli) terpisah dari evaluasi (Agustus), galat per layanan/pulau/kurir, SHAP & partial dependence.
- **Perbaikan Task 3 – Clustering** (atas permintaan Fransisco):
  - Menulis ulang `kode/task3_clustering.py`: bukti noise `n01–n15`, seleksi fitur dengan η² & redundansi, anomali dengan ambang knee, pemilihan K dengan silhouette/CH/DB/BIC/stabilitas bootstrap (K = 7), perbandingan 3 metode, stabilitas final, profil & aksi otomatis.
- **Laporan & reproduksibilitas**: `kode/laporan.py` membangun seluruh `laporan.pdf` dari `kode/hasil_task1-3/`; `run_all.py` diperbarui.
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
   - Perbaikan Task 1 & Task 3: Claude (Anthropic) membantu menulis ulang `kode/task1_regression.py`, `kode/task3_clustering.py`, `kode/laporan.py`, dan draf teks laporan Task 1 & 3.
3. **Metode Verifikasi oleh Anggota**:
   - Task 1–3: setiap angka di laporan dibaca langsung dari `kode/hasil_task1-3/` (JSON & CSV hasil kode) oleh `kode/laporan.py`; format ketiga file prediksi dicek otomatis di kode (jumlah baris, ID unik, `p90 >= p50`, `prob` di [0,1], `tandai` 0/1); `run_all.py` dijalankan ulang dari awal dan md5 file prediksi identik.
   - Semua angka, metrik validasi (MAE, Pinball loss, cakupan kuantil, skor Silhouette, ARI), dan tabel yang dilaporkan dieksekusi secara independen dan diverifikasi langsung pada dataset lokal.
   - Tidak ada angka atau prediksi yang dihasilkan secara fiktif; seluruh file `prediksi_regresi.csv`, `prediksi_klasifikasi.csv`, dan `cluster.csv` diproduksi secara deterministik melalui kode Python yang dapat direproduksi penuh dengan satu perintah (`python kode/run_all.py`).

---

## 4. Persetujuan Anggota Kelompok

Dokumen ini disusun dan disetujui bersama oleh seluruh anggota Kelompok 16:
- (Ttd) **Fransisco Sitepu** (101032300004)
- (Ttd) **Ahmad Ghazali El-Faruqi HSB** (101032330080)

Persetujuan dicatat melalui pull request masing-masing anggota di GitHub. Isi pernyataan penggunaan AI di bagian 3 sudah dibaca dan disetujui oleh kedua anggota.
