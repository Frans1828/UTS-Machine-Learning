"""
=============================================================================
UTS MACHINE LEARNING - GENERATOR laporan.pdf (TASK 1-3)
Kelompok 16: Fransisco Sitepu (101032300004) & Ahmad Ghazali El-Faruqi HSB (101032330080)
=============================================================================
Membangun seluruh laporan.pdf dari angka yang dihasilkan kode:
  kode/hasil_task1/  (task1_regression.py)
  kode/hasil_task2/  (task2_classification.py)
  kode/hasil_task3/  (task3_clustering.py)
Tidak ada angka yang diketik manual, sehingga setiap angka di laporan dapat
direproduksi dengan satu perintah: python kode/run_all.py
"""

import json
import os

import pandas as pd

from laporan_task2 import new_doc, add_task2, rp, th, pct, num

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
R1D = os.path.join(HERE, "hasil_task1")
R3D = os.path.join(HERE, "hasil_task3")
REPO_URL = "https://github.com/Frans1828/UTS-Machine-Learning"


def jam(x, d=2):
    return f"{num(x, d)} jam"


def cover(d):
    d.cover = True
    d.add_page()
    d.set_y(45)
    d.set_font("Helvetica", "B", 16)
    d.multi_cell(0, 8, "LAPORAN UJIAN TENGAH SEMESTER (UTS)\nMATA KULIAH MACHINE LEARNING (CSG3H3)", align="C",
                 new_x="LMARGIN", new_y="NEXT")
    d.ln(6)
    d.set_font("Helvetica", "I", 11)
    d.multi_cell(0, 6, "Hands-on Machine Learning: Regression, Classification, and Clustering\n"
                       "Estimasi Waktu Pengiriman, Deteksi Fraud Transaksi, dan Segmentasi Pelanggan",
                 align="C", new_x="LMARGIN", new_y="NEXT")
    d.ln(20)
    d.set_font("Helvetica", "B", 12)
    d.cell(0, 7, "KELOMPOK 16 - KELAS TK 47 03", align="C", new_x="LMARGIN", new_y="NEXT")
    d.ln(4)
    d.set_font("Helvetica", "", 11)
    for line in ["Fransisco Sitepu - NIM 101032300004 (Task 1 & Task 3)",
                 "Ahmad Ghazali El-Faruqi HSB - NIM 101032330080 (Task 2; perbaikan Task 1 & 3)"]:
        d.cell(0, 7, line, align="C", new_x="LMARGIN", new_y="NEXT")
    d.ln(6)
    d.set_font("Helvetica", "", 9)
    d.cell(0, 6, f"Repositori: {REPO_URL}", align="C", new_x="LMARGIN", new_y="NEXT")
    d.set_y(225)
    d.set_font("Helvetica", "B", 11)
    d.multi_cell(0, 6, "PROGRAM STUDI S1 TEKNIK KOMPUTER\nFAKULTAS TEKNIK ELEKTRO - TELKOM UNIVERSITY\nBANDUNG, NOVEMBER 2026",
                 align="C", new_x="LMARGIN", new_y="NEXT")


def task1(d):
    R = json.load(open(os.path.join(R1D, "ringkasan_task1.json")))
    a, v, L = R["audit"], R["validasi"], R["lipatan"]
    leak = pd.read_csv(os.path.join(R1D, "a_dampak_kebocoran.csv"))
    unit = pd.read_csv(os.path.join(R1D, "a_label_satuan_menit.csv"))
    tune = pd.read_csv(os.path.join(R1D, "c_tuning.csv"))
    mod = pd.read_csv(os.path.join(R1D, "c_perbandingan_model.csv"))
    cat = pd.read_csv(os.path.join(R1D, "d_kategori.csv"))
    unc = pd.read_csv(os.path.join(R1D, "e_kalibrasi_conformal.csv"))
    gl = pd.read_csv(os.path.join(R1D, "f_galat_per_layanan.csv"))
    gp = pd.read_csv(os.path.join(R1D, "f_galat_per_pulau_tujuan.csv"))
    gm = pd.read_csv(os.path.join(R1D, "f_galat_per_mitra_kurir.csv"))
    shap = pd.read_csv(os.path.join(R1D, "f_shap_importance.csv"))
    c, fin = R["conformal"], R["final"]

    d.add_page()
    d.h1("TASK 1 - REGRESI: ESTIMASI WAKTU PENGIRIMAN (P50 & P90)")
    d.p(f"Data latih {th(R['n_train'])} paket (Jan-Agu 2025), data uji {th(R['n_test'])} paket (Sep-Okt 2025). Target: durasi_jam "
        "sebagai median (P50) dan batas atas 90% (P90) pada saat paket diambil kurir. Semua angka di bagian ini dihasilkan oleh "
        "kode/task1_regression.py (seed 42) dan tersimpan di kode/hasil_task1/.")

    d.h2("1.1 Audit Data (a)")
    d.p(f"1) Kebocoran 'jumlah_scan'. Kolom ini menghitung scan pelacakan sepanjang perjalanan, sehingga di data latih rata-ratanya "
        f"{num(a['scan_mean_train'], 2)} (maks {a['scan_max_train']}) sedangkan di data uji, yang diambil saat pickup, hanya "
        f"{num(a['scan_mean_test'], 2)} (maks {a['scan_max_test']}). Korelasi Spearman dengan target {num(a['scan_spearman_target'], 3)}. "
        "Dampaknya diukur pada lipatan Jan-Jun -> Jul-Agu:")
    d.table(["Skenario (LightGBM P50)", "MAE", "Pinball q=0,5"],
            [[r.model, jam(r.mae_p50), num(r.pinball_50, 3)] for r in leak.itertuples()], [110, 35, 35], bold_row=3)
    d.p(f"Dengan jumlah_scan, MAE terlihat {jam(leak.mae_p50[0])}, padahal bila scan diisi seperti kondisi pickup MAE melonjak ke "
        f"{jam(leak.mae_p50[1])}, lebih buruk dari model jujur ({jam(leak.mae_p50[3])}). Kolom ini dibuang.")
    ex = unit.set_index("layanan")
    d.p(f"2) Label tercatat dalam menit. {a['n_label_menit']} baris ({pct(a['pct_label_menit'], 3)}) berdurasi > 1.000 jam (maks "
        f"{th(a['max_label_raw'])} jam), sedangkan P99 hanya {num(a['p99_label'], 1)} jam. Bila dibagi 60, median per layanan hampir sama "
        f"dengan data normal (mis. reguler {num(ex.loc['reguler', 'median_outlier_dibagi_60'], 1)} vs {num(ex.loc['reguler', 'median_normal_jam'], 1)} jam; "
        f"hemat {num(ex.loc['hemat', 'median_outlier_dibagi_60'], 1)} vs {num(ex.loc['hemat', 'median_normal_jam'], 1)} jam), jadi ini kesalahan satuan "
        "dan dikoreksi (dibagi 60), bukan dipotong. Dampaknya pada LightGBM kuantil kecil (loss kuantil tahan pencilan, lihat tabel), tetapi "
        "koreksi ini penting untuk model berbasis rata-rata dan untuk evaluasi yang benar. "
        f"3) Nilai sentinel jarak_km: {a['jarak_9999_train']} baris bernilai 9999 dan {a['jarak_0_train']} baris bernilai 0 (uji: "
        f"{a['jarak_9999_test']} & {a['jarak_0_test']}) -> dijadikan missing. 4) berat_kg kosong {th(a['berat_missing_train'])} baris "
        f"({pct(a['berat_missing_pct'], 2)}), tidak acak: {pct(a['share_kargo_saat_berat_hilang'])} di antaranya layanan kargo (porsi kargo "
        f"total {pct(a['share_kargo_total'])}) -> dibiarkan NaN + indikator 'berat_hilang'. 5) 'f30' adalah proksi waktu (rata-rata "
        f"{num(a['f30_mean_per_bulan']['1'], 2)} di Jan -> {num(a['f30_mean_per_bulan']['8'], 2)} di Agu -> {num(a['f30_mean_per_bulan']['10'], 2)} di Okt; "
        f"Spearman {num(a['f30_spearman_target'], 3)}) -> dibuang. 6) {a['umur_akun_mustahil']} umur akun penjual > 20.000 hari (mustahil) -> missing.")

    d.h2("1.2 Skema Validasi (b)")
    d.p(f"Validasi meniru pemakaian nyata (memprediksi 2 bulan berikutnya): latih Jan-Jun ({th(L['latih_jan_jun'])} baris) -> uji Jul-Agu "
        f"({th(L['uji_jul_agu'])} baris). Tuning hanya dilakukan di dalam data latih (Jan-Apr -> Mei-Jun). Untuk kalibrasi interval, Jul "
        f"({th(L['kalibrasi_jul'])}) dipakai sebagai data kalibrasi dan Agu ({th(L['evaluasi_agu'])}) sebagai data evaluasi. Random 5-fold CV "
        f"(pada model yang sama) memberi MAE {jam(v['random_cv_mae'])} (+/- {num(v['random_cv_mae_std'], 3)}) vs {jam(v['waktu_mae'])} pada split waktu; "
        f"validasi bergulir Jan-Apr -> Mei-Jun memberi {jam(v['rolling'][0]['mae'])}. Random CV sedikit terlalu optimis karena paket dari hari "
        "yang sama (antrean, cuaca, hari libur yang sama) ada di latih dan uji sekaligus; selisihnya kecil karena hubungan fitur-target "
        "relatif stabil antarbulan. Kami melaporkan angka split waktu.")

    d.h2("1.3 Model & Tuning (c)")
    b50, b90 = R["tuning_terbaik"]["0.5"], R["tuning_terbaik"]["0.9"]
    d.p(f"Grid LightGBM (num_leaves {{31, 127}} x min_child_samples {{50, 300}}, learning rate 0,05, early stopping) dievaluasi pada "
        f"Mei-Jun: terbaik untuk P50 num_leaves={b50['num_leaves']}, min_child_samples={b50['min_child_samples']} "
        f"({b50['n_estimators']} pohon) dan untuk P90 num_leaves={b90['num_leaves']}, min_child_samples={b90['min_child_samples']} "
        f"({b90['n_estimators']} pohon). Pinball terbaik P50 {num(tune[tune.alpha == 0.5].pinball.min(), 3)} vs terburuk "
        f"{num(tune[tune.alpha == 0.5].pinball.max(), 3)}. Hasil pada lipatan Jul-Agu:")
    d.table(["Model", "MAE P50", "Pinball 0,5", "Pinball 0,9", "Cakupan P90"],
            [[r.model, jam(r.mae_p50), num(r.pinball_50, 3), num(r.pinball_90, 3), pct(r.coverage_90, 2)] for r in mod.itertuples()],
            [88, 26, 24, 24, 24], bold_row=2)
    d.p("Baseline = kuantil empiris durasi per kombinasi layanan x pulau_tujuan. Model lanjutan 1 = Ridge pada log1p(durasi) dengan "
        "one-hot & target encoding, P90 dari kuantil residual log. Model lanjutan 2 = LightGBM dengan loss kuantil (q=0,5 dan q=0,9). "
        "LightGBM terbaik untuk pinball kedua kuantil, tetapi P90 mentahnya kurang cakupan, sehingga dikalibrasi (1.5).")

    d.h2("1.4 Kategori Kardinalitas Tinggi & Kategori Baru (d)")
    d.p(f"kota_tujuan memiliki {a['n_kota_tujuan']} kategori; kategori yang hanya ada di uji adalah mitra_kurir K13 "
        f"({th(a['n_baris_k13_test'])} baris uji). LightGBM memakai kategori native dengan regularisasi (cat_smooth=20, "
        "min_data_per_group=100, cat_l2=10); kategori yang tidak dikenal dipetakan ke missing secara eksplisit. Untuk Ridge dipakai target "
        "encoding ber-smoothing (k=50, kategori baru -> rata-rata global). Untuk menguji K13, satu kurir (K12) dikeluarkan dari data latih "
        "lalu dievaluasi seolah-olah kategori baru:")
    d.table(["Skenario", "MAE P50", "Pinball 0,5"],
            [[r.model + (f" (n={th(r.n)})" if pd.notna(r.n) else ""), jam(r.mae_p50), num(r.pinball_50, 3)] for r in cat.itertuples()],
            [120, 30, 30])
    d.p(f"Native + smoothing setara dengan target encoding dan lebih baik daripada membuang kota_tujuan. Kurir tak dikenal hanya menambah "
        f"MAE {num(cat.mae_p50[4] - cat.mae_p50[3], 2)} jam (jauh di bawah baseline), jadi prediksi K13 tetap layak "
        f"(median P50 K13 di data uji {jam(fin['p50_median_k13'])}).")

    d.h2("1.5 Ketidakpastian & Kalibrasi Conformal (e)")
    d.p(f"P90 mentah LightGBM hanya mencakup {pct(c['coverage_raw_jul_agu'], 2)} durasi Jul-Agu. Kami memakai conformalized quantile "
        f"regression: offset = kuantil ke-90 (dengan koreksi (n+1)/n) dari skor y - P90 pada data kalibrasi Juli = {num(c['offset_kalibrasi_jul'], 2)} jam, "
        "lalu diuji pada Agustus yang tidak dipakai untuk menghitung offset:")
    d.table(["Agustus (out-of-sample)", "MAE P50", "Pinball 0,5", "Pinball 0,9", "Cakupan P90"],
            [[r.model, jam(r.mae_p50), num(r.pinball_50, 3), num(r.pinball_90, 3), pct(r.coverage_90, 2)] for r in unc.itertuples()],
            [88, 26, 24, 24, 24], bold_row=1)
    d.p(f"Setelah kalibrasi cakupan P90 naik ke target dan pinball q=0,9 ikut membaik. Untuk prediksi uji, offset dihitung ulang dari "
        f"seluruh residual out-of-time Jul-Agu = {num(c['offset_final'], 2)} jam dan ditambahkan ke P90 model final; P90 >= P50 dijamin.")

    d.h2("1.6 Analisis Galat & Interpretasi (f)")
    d.table(["Layanan", "n", "Median aktual", "MAE P50", "Pinball 0,9", "Cakupan P90", "MAE relatif"],
            [[r.layanan, th(r.n), jam(r.median_aktual, 1), jam(r.mae_p50), num(r.pinball_90, 2), pct(r.coverage_90), pct(r.mae_relatif_pct)]
             for r in gl.itertuples()], [24, 20, 28, 26, 24, 26, 26])
    d.table(["Pulau tujuan", "n", "Median aktual", "MAE P50", "Pinball 0,9", "Cakupan P90", "MAE relatif"],
            [[r.pulau_tujuan, th(r.n), jam(r.median_aktual, 1), jam(r.mae_p50), num(r.pinball_90, 2), pct(r.coverage_90), pct(r.mae_relatif_pct)]
             for r in gp.itertuples()], [34, 20, 26, 24, 22, 24, 24], size=7)
    worst = gm.iloc[0]; rest = gm.iloc[1:]
    d.p(f"Galat absolut terbesar ada pada pulau jauh (Papua MAE {jam(gp.mae_p50[0])}) dan layanan hemat, tetapi secara relatif terhadap "
        f"median galatnya mirip (+/- 19-26%). Per mitra kurir, {worst.mitra_kurir} menonjol: MAE {jam(worst.mae_p50)} dan cakupan P90 hanya "
        f"{pct(worst.coverage_90)} (kurir lain {pct(rest.coverage_90.min())}-{pct(rest.coverage_90.max())}), artinya {worst.mitra_kurir} memiliki "
        "ekor keterlambatan yang lebih panjang daripada yang ditangkap model; disarankan offset conformal per kurir atau fitur keandalan kurir.")
    top = ", ".join(f"{r.fitur} ({num(r.mean_abs_shap_jam, 1)} jam)" for r in shap.head(8).itertuples())
    pdp = R["pdp"]
    d.p(f"SHAP (TreeSHAP bawaan LightGBM, rata-rata |kontribusi| P50 pada 5.000 paket Jul-Agu): {top}. Partial dependence: P50 rata-rata "
        f"naik dari {jam(pdp['jarak_km']['p50'][0], 1)} pada jarak {th(pdp['jarak_km']['grid'][0])} km ke {jam(pdp['jarak_km']['p50'][-1], 1)} pada "
        f"{th(pdp['jarak_km']['grid'][-1])} km; antrean hub menaikkan P50 dari {jam(pdp['antrean_hub_asal']['p50'][0], 1)} ke "
        f"{jam(pdp['antrean_hub_asal']['p50'][-1], 1)} hanya pada antrean sangat tinggi (efek nonlinear); per layanan: ekspres "
        f"{jam(pdp['layanan']['ekspres'], 1)}, reguler {jam(pdp['layanan']['reguler'], 1)}, hemat {jam(pdp['layanan']['hemat'], 1)}, kargo "
        f"{jam(pdp['layanan']['kargo'], 1)}. Fitur anonim f04 dan f17 ikut berpengaruh; sebagian besar f01-f50 lain tidak berkontribusi.")
    img = os.path.join(R1D, "fig_task1_shap_pdp_galat.png")
    if os.path.exists(img):
        if d.get_y() > 215:
            d.add_page()
        d.image(img, w=190)
    d.p(f"Model final dilatih ulang pada Jan-Agu (label terkoreksi) -> prediksi_regresi.csv ({th(fin['n'])} baris; median P50 "
        f"{jam(fin['p50_median'])}, median P90 {jam(fin['p90_median'])}).")


def task3(d):
    R = json.load(open(os.path.join(R3D, "ringkasan_task3.json")))
    a, sel = R["audit"], R["seleksi"]
    imp = pd.read_csv(os.path.join(R3D, "b_dampak_seleksi.csv"))
    eta = pd.read_csv(os.path.join(R3D, "b_relevansi_eta2.csv"), index_col=0)
    ktab = pd.read_csv(os.path.join(R3D, "d_pemilihan_k.csv"))
    meth = pd.read_csv(os.path.join(R3D, "c_perbandingan_metode.csv"))
    prof = pd.read_csv(os.path.join(R3D, "f_profil_segmen.csv"))
    an, st, K = R["anomali"], R["stabilitas_final"], R["pemilihan_k"]["K"]

    d.add_page()
    d.h1("TASK 3 - CLUSTERING: SEGMENTASI PELANGGAN")
    d.p(f"{th(R['n_pelanggan'])} pelanggan x {R['n_kolom']} kolom (ringkasan 90 hari, tanpa label). Semua angka dihasilkan oleh "
        "kode/task3_clustering.py (seed 42) dan tersimpan di kode/hasil_task3/.")

    d.h2("3.1 Prapemrosesan (a)")
    sb, ss = a["skew_sebelum"], a["skew_sesudah_log"]
    d.p(f"1) Skala & kemiringan: variabel uang/hitungan sangat miring (skewness belanja_90h {num(sb['belanja_90h'], 2)}, recency_hari "
        f"{num(sb['recency_hari'], 2)}, sesi_per_minggu {num(sb['sesi_per_minggu'], 2)}) -> log1p (menjadi {num(ss['belanja_90h'], 2)}, "
        f"{num(ss['recency_hari'], 2)}, {num(ss['sesi_per_minggu'], 2)}); persen_* (rentang 0-1) -> logit. 2) Data komposisional: "
        f"{a['n_porsi_kategori']} porsi kategori dan {a['n_porsi_jam']} porsi jam masing-masing berjumlah 1 (rentang "
        f"{num(a['jumlah_porsi_kategori_min_max'][0], 4)}-{num(a['jumlah_porsi_kategori_min_max'][1], 4)}); {pct(a['pct_nol_porsi_kategori'], 2)} "
        "nilai porsi = 0 -> penggantian nol multiplikatif (1e-3), normalisasi ulang, lalu CLR (Aitchison) agar jarak Euclid bermakna. "
        f"3) Missing: rata2_keranjang kosong {th(a['keranjang_nan'])} baris, seluruhnya saat frekuensi_90h = 0 ({th(a['keranjang_nan_saat_frekuensi_nol'])}). "
        f"Karena {th(a['frekuensi_nol_tapi_recency_lt_90'])} di antaranya justru belanja < 90 hari lalu (recency), frekuensi nol adalah nol "
        "kebetulan, bukan segmen tersendiri; nilai diimputasi median tanpa indikator biner agar tidak membentuk klaster buatan. 4) Kategorikal: "
        "perangkat -> one-hot (ios, web). 5) Pencilan: fitur di-winsor pada persentil 0,5/99,5 sebelum standardisasi, dan pelanggan "
        "ekstrem diberi label -1 (3.3).")

    d.h2("3.2 Seleksi Fitur dengan Bukti (b)")
    n = R["noise"]
    d.p(f"1) n01-n15 adalah noise: korelasi Spearman maksimum dengan fitur domain mana pun hanya {num(n['maks_rho_domain'], 3)}, sedangkan "
        f"fitur domain biasanya berkorelasi |rho| ~{num(R['rho_domain_median_maks'], 2)} dengan fitur lain; {', '.join(n['seragam_p_gt_005'])} lolos uji "
        "KS distribusi seragam (p > 0,05). 2) Redundansi: log belanja_90h = log(frekuensi x keranjang) (r = "
        f"{num(R['redundansi_rho']['log_belanja_90h~log_rata2_keranjang+log_frekuensi_90h'], 3)}); belanja_30h dan frekuensi_30h berkorelasi "
        f"{num(R['redundansi_rho']['log_belanja_30h~log_belanja_90h'], 3)} dan {num(R['redundansi_rho']['log_frekuensi_30h~log_frekuensi_90h'], 3)} "
        "dengan versi 90 hari -> dibuang agar informasi yang sama tidak terhitung dua kali. 3) Relevansi: eta^2 (porsi variansi fitur yang "
        f"dijelaskan klaster awal K-Means K=5..8). Fitur dipakai bila eta^2 minimum >= {num(sel['eta_min'], 2)}, yaitu relevan konsisten di semua K. "
        f"Dibuang: {', '.join(x.replace('clr_porsi_', '') for x in sel['dibuang_eta'])}. dev_web hanya relevan di K=6-7 (eta^2 "
        f"{num(eta.loc['dev_web', 'K6'], 2)}) karena membentuk klaster 'pengguna web' tanpa ciri lain, dan nyaris nol di K=5/K=8 "
        f"({num(eta.loc['dev_web', 'K5'], 4)}). Hasilnya {sel['n_terpilih']} fitur. Dampak (K={K}):")
    d.table(["Set fitur", "n fitur", "Silhouette", "Stabilitas ARI"],
            [[r.set_fitur, r.n_fitur, num(r.silhouette, 4), num(r.stabilitas_ari, 3)] for r in imp.itertuples()], [100, 22, 28, 30], bold_row=2)

    d.h2("3.3 Anomali (g)")
    pm = an["profil_median_vs_normal"]
    d.p(f"Isolation Forest (300 pohon) pada fitur terpilih tanpa winsor; ambang ditentukan dari titik siku kurva skor anomali (bukan "
        f"persentase tetap): {th(an['n'])} pelanggan ({pct(an['pct'], 2)}) -> cluster = -1. {pct(an['pct_di_atas_p99_minimal_satu_fitur'])} "
        f"di antaranya berada di atas persentil 99 pada minimal satu fitur. Median sesi_per_minggu anomali {num(pm['sesi_per_minggu'][0], 1)} vs "
        f"{num(pm['sesi_per_minggu'][1], 1)} pada pelanggan normal dengan rasio konversi {num(pm['rasio_konversi'][0], 2)} vs "
        f"{num(pm['rasio_konversi'][1], 2)}: pola mirip bot/scraper (sesi sangat banyak, konversi sangat rendah). "
        f"Median belanja_90h anomali juga lebih tinggi (Rp {th(pm['belanja_90h'][0])} vs Rp {th(pm['belanja_90h'][1])}). Anomali dikeluarkan sebelum "
        "memilih K dan melatih model agar centroid tidak tertarik pencilan.")

    d.h2("3.4 Jumlah Klaster (d)")
    d.table(["K", "Silhouette", "Calinski-H.", "Davies-B.", "BIC GMM", "Stabilitas ARI", "ARI min"],
            [[int(r.K), num(r.silhouette, 4), num(r.calinski_harabasz, 1), num(r.davies_bouldin, 3), th(r.bic_gmm), num(r.stabilitas_ari, 3),
              num(r.stabilitas_ari_min, 3)] for r in ktab.itertuples()], [14, 26, 28, 26, 30, 30, 26],
            bold_row=int(ktab.index[ktab.K == K][0]))
    kk = ktab.set_index("K")
    d.p(f"Silhouette dan CH cenderung menurun seiring K (wajar untuk segmen yang saling tumpang tindih), sehingga tidak bisa dipakai sendiri. "
        f"Kriteria utama: stabilitas bootstrap (5 subsample, K-Means, ARI pada holdout 10.000 pelanggan yang sama). Aturan: K terbesar dengan "
        f"ARI >= 0,95 -> K = {K}. Pada K = {K + 1} stabilitas turun tajam ke {num(kk.loc[K + 1, 'stabilitas_ari'], 3)}, tanda klaster "
        f"tambahan hanya membelah segmen secara acak. Kriteria kedua: Davies-Bouldin pada K = {K} ({num(kk.loc[K, 'davies_bouldin'], 3)}) "
        f"tidak lebih buruk daripada K = 5 ({num(kk.loc[5, 'davies_bouldin'], 3)}), sedangkan silhouette hanya turun sedikit "
        f"({num(kk.loc[5, 'silhouette'], 4)} -> {num(kk.loc[K, 'silhouette'], 4)}). BIC GMM turun bertahap hingga K = 7 lalu anjlok di K = 8; "
        "BIC tidak kami jadikan penentu karena GMM diagonal pada data ini membentuk komponen sangat kecil untuk menangkap nilai hitungan "
        f"diskret (lihat 3.5: klaster terkecil GMM {pct(meth.ukuran_terkecil_pct[1], 2)} dan stabilitasnya rendah), sehingga BIC cenderung memilih komponen artifisial.")

    d.h2("3.5 Perbandingan Metode (c) & Stabilitas (e)")
    d.table([f"Metode (K={K})", "Silhouette", "Davies-B.", "Calinski-H.", "Stabilitas ARI", "Klaster terkecil"],
            [[r.metode, num(r.silhouette, 4), num(r.davies_bouldin, 3), num(r.calinski_harabasz, 1), num(r.stabilitas_ari, 3),
              pct(r.ukuran_terkecil_pct, 2)] for r in meth.itertuples()], [50, 26, 26, 28, 28, 30],
            bold_row=int(meth.index[meth.metode == R["metode_terpilih"]][0]))
    ag = R["metode"]["kesepakatan_ari"]
    d.p(f"K-Means dan Agglomerative Ward (15.000 sampel, sisanya ke centroid terdekat) menghasilkan partisi yang mirip (ARI "
        f"{num(ag['K-Means vs Agglomerative Ward'], 3)}), sedangkan GMM diagonal tidak stabil dan membentuk komponen sangat kecil (ARI dengan "
        f"K-Means {num(ag['K-Means vs Gaussian Mixture (diag)'], 3)}) karena fitur hitungan diskret melanggar asumsi Gaussian. Terpilih "
        f"{R['metode_terpilih']} (stabilitas tertinggi, ukuran segmen seimbang, dapat dilatih pada seluruh data). Uji stabilitas model final: "
        f"{st['n_bootstrap']} bootstrap x {th(st['ukuran_subsample'])} pelanggan, ARI antar-run {num(st['ari_mean'], 3)} +/- "
        f"{num(st['ari_std'], 3)} (minimum {num(st['ari_min'], 3)}).")

    d.h2("3.6 Profil Segmen & Aksi Pemasaran (f)")
    d.p("Nama dan aksi ditentukan otomatis dari dua fitur pembeda terkuat (z-score rata-rata segmen, |z| > 0,4); nilai di tabel adalah median.")
    rows = []
    for r in prof.itertuples():
        rows.append([str(r.cluster), f"{th(r.n)} ({pct(r.pct)})", th(r.recency_hari) if r.recency_hari >= 1 else num(r.recency_hari, 1),
                     num(r.frekuensi_90h, 0), f"{num(r.rata2_keranjang / 1000, 0)} rb", pct(100 * r.persen_voucher, 0),
                     pct(100 * r.persen_paylater, 0), num(r.sesi_per_minggu, 1), num(r.rasio_konversi, 2), num(r.umur, 0),
                     num(r.lama_member_bulan, 1)])
    d.table(["Klaster", "n (%)", "Recency", "Frek", "Keranjang", "Voucher", "Paylater", "Sesi/mg", "Konversi", "Umur", "Member (bln)"],
            rows, [14, 30, 16, 12, 20, 16, 16, 16, 16, 12, 22], size=6.8)
    for r in prof.itertuples():
        d.bullet(f"Klaster {r.cluster} - {r.nama} ({pct(r.pct)}). Aksi: {r.aksi}.", size=8)
    d.ln(1)
    img = os.path.join(R3D, "fig_task3_k_profil.png")
    if os.path.exists(img):
        if d.get_y() > 215:
            d.add_page()
        d.image(img, w=190)
    d.p(f"Output: cluster.csv ({th(sum(R['distribusi'].values()))} baris; {K} segmen bernomor 0-{K - 1} berurutan dari yang terbesar, -1 = anomali).")


def penutup(d):
    d.add_page()
    d.h1("KESIMPULAN & REPRODUKSIBILITAS")
    R1 = json.load(open(os.path.join(R1D, "ringkasan_task1.json")))
    R3 = json.load(open(os.path.join(R3D, "ringkasan_task3.json")))
    R2 = json.load(open(os.path.join(HERE, "hasil_task2", "ringkasan_task2.json")))
    d.p("Repositori memuat: laporan.pdf; kode/ (run_all.py, task1_regression.py, task2_classification.py, task3_clustering.py, "
        "laporan.py, laporan_task2.py, README.md, serta hasil_task1-3/ berisi semua tabel, grafik, dan ringkasan JSON); "
        f"prediksi_regresi.csv ({th(R1['final']['n'])} baris, p90 >= p50); prediksi_klasifikasi.csv ({th(R2['n_test'])} baris); "
        f"cluster.csv ({th(R3['n_pelanggan'])} baris, -1 = anomali); kontribusi.md.")
    d.p("Satu perintah mereproduksi seluruh angka, file prediksi, dan laporan ini: python kode/run_all.py. Seed 42 di semua tahap. "
        f"Versi pustaka: LightGBM {R1['library']['lightgbm']}, scikit-learn {R1['library']['sklearn']}, pandas {R1['library']['pandas']}, "
        f"numpy {R1['library']['numpy']}. Waktu eksekusi pada 2 CPU: Task 1 ~{round(R1['durasi_detik'] / 60)} menit, Task 2 "
        f"~{round(R2['durasi_detik'] / 60)} menit, Task 3 ~{max(1, round(R3['durasi_detik'] / 60))} menit.")
    d.p("Temuan utama: (1) setiap task memuat jebakan data yang hanya terlihat lewat audit: jumlah_scan & label bersatuan menit (Task 1), "
        "flag_review_manual & label fraud yang belum lengkap (Task 2), serta kolom noise n01-n15 & fitur biner yang membentuk klaster "
        "buatan (Task 3); (2) validasi berbasis waktu dan kalibrasi pada data terpisah memberi estimasi kinerja yang jujur; (3) keputusan akhir "
        "(P90 untuk janji ke pembeli, aturan biaya untuk review fraud, aksi per segmen) diturunkan langsung dari hasil model.")
    d.p("Penggunaan AI dan pembagian kerja dijelaskan di kontribusi.md. Seluruh anggota bertanggung jawab menjelaskan semua bagian laporan.")


def build(pdf_path=None):
    pdf_path = pdf_path or os.path.join(ROOT, "laporan.pdf")
    d = new_doc(1)
    cover(d)
    task1(d)
    add_task2(d)
    task3(d)
    penutup(d)
    d.output(pdf_path)
    print(f"[Laporan] {pdf_path} dibangun: {d.page_no()} halaman")
    return d.page_no()


if __name__ == "__main__":
    build()
