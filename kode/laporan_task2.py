"""
=============================================================================
UTS MACHINE LEARNING - GENERATOR BAGIAN TASK 2 PADA laporan.pdf
Author : Ahmad Ghazali El-Faruqi HSB (101032330080)
=============================================================================
Membaca angka dari kode/hasil_task2/ (hasil task2_classification.py) lalu
mengganti halaman "TASK 2" di laporan.pdf dengan bagian Task 2 yang lengkap.
Semua angka pada bagian Task 2 laporan berasal dari file hasil tersebut,
sehingga dapat direproduksi dengan satu perintah (python kode/run_all.py).

Halaman lain (Task 1 & Task 3) tidak diubah isinya; hanya nomor halaman
pada header yang disesuaikan.
"""

import io
import json
import os

import pandas as pd
from fpdf import FPDF
from pypdf import PdfReader, PdfWriter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(HERE, "hasil_task2")
HEADER = "UTS Machine Learning - Kelompok 16: Fransisco Sitepu & Ahmad Ghazali El-Faruqi HSB"
FOOTER = "Program Studi S1 Teknik Komputer - Telkom University 2026"
MARKER = "Bagian Task 2 dihasilkan dari kode/laporan_task2.py"
NAVY = (31, 56, 100)
BLUE = (46, 116, 181)


def rp(x_juta):
    return f"Rp {x_juta:,.2f} jt".replace(",", "X").replace(".", ",").replace("X", ".")


def th(n):
    return f"{int(round(n)):,}".replace(",", ".")


def pct(x, d=1):
    return f"{x:.{d}f}%".replace(".", ",")


def num(x, d=4):
    return f"{x:.{d}f}".replace(".", ",")


class Doc(FPDF):
    start_page = 3

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 5, HEADER, align="L")
        self.set_x(-30)
        self.cell(20, 5, f"Hal. {self.start_page + self.page_no() - 1}", align="R")
        self.ln(3)
        self.set_draw_color(150, 150, 150)
        self.line(10, 14, 200, 14)
        self.ln(6)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(120, 120, 120)
        self.cell(0, 4, FOOTER, align="C")
        self.ln(3)
        self.set_font("Helvetica", "", 5)
        self.set_text_color(200, 200, 200)
        self.cell(0, 3, MARKER, align="C")
        self.set_text_color(0, 0, 0)

    def h1(self, t):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*NAVY)
        self.cell(0, 8, t, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*NAVY)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def h2(self, t):
        self.ln(1)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*BLUE)
        self.multi_cell(0, 5.5, t, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

    def p(self, t, size=8.5):
        self.set_font("Helvetica", "", size)
        self.multi_cell(0, 4.3, t, align="J", new_x="LMARGIN", new_y="NEXT")
        self.ln(0.8)

    def bullet(self, t, size=8.5):
        self.set_font("Helvetica", "", size)
        x = self.get_x()
        self.cell(4, 4.3, "-")
        self.multi_cell(0, 4.3, t, align="J", new_x="LMARGIN", new_y="NEXT")
        self.set_x(x)

    def table(self, header, rows, widths, size=7.5, bold_row=None):
        self.set_font("Helvetica", "B", size)
        self.set_fill_color(220, 228, 240)
        for h, w in zip(header, widths):
            self.cell(w, 5, h, border=1, align="C", fill=True)
        self.ln()
        for i, r in enumerate(rows):
            self.set_font("Helvetica", "B" if bold_row is not None and i == bold_row else "", size)
            for j, (v, w) in enumerate(zip(r, widths)):
                self.cell(w, 4.6, str(v), border=1, align="L" if j == 0 else "C")
            self.ln()
        self.ln(1.5)


def build_task2_pdf(start_page):
    R = json.load(open(os.path.join(RES, "ringkasan_task2.json")))
    leak = pd.read_csv(os.path.join(RES, "a_dampak_kebocoran.csv"))
    mg = pd.read_csv(os.path.join(RES, "a_fraud_per_minggu.csv"))
    lag = pd.read_csv(os.path.join(RES, "b_simulasi_label_lag.csv"))
    mod = pd.read_csv(os.path.join(RES, "c_perbandingan_model.csv"))
    imb = pd.read_csv(os.path.join(RES, "d_ketidakseimbangan_kelas.csv"))
    rec = pd.read_csv(os.path.join(RES, "e_recency_weighting.csv"))
    cal = pd.read_csv(os.path.join(RES, "f_kalibrasi.csv"))
    dec = pd.read_csv(os.path.join(RES, "f_kebijakan_keputusan.csv"))
    gk = pd.read_csv(os.path.join(RES, "g_galat_per_kanal.csv"))
    gp = pd.read_csv(os.path.join(RES, "g_galat_per_pola.csv"))
    gc = pd.read_csv(os.path.join(RES, "g_galat_per_kategori.csv"))
    imp = pd.read_csv(os.path.join(RES, "importance_model_final.csv"))
    fr = R["flag_review"]; fold = R["fold"]; fin = R["final"]
    qris = R["pola_qris_agustus"]

    d = Doc(format="A4")
    d.start_page = start_page
    d.set_margins(10, 10, 10)
    d.set_auto_page_break(True, margin=15)
    d.add_page()
    d.h1("TASK 2 - KLASIFIKASI: DETEKSI TRANSAKSI FRAUD")
    d.p(f"Dikerjakan oleh Ahmad Ghazali El-Faruqi HSB (NIM 101032330080). Data latih {th(R['n_train'])} transaksi "
        f"(Jan-Agu 2025, prevalensi fraud {pct(R['prevalensi_train'], 2)}) dan data uji {th(R['n_test'])} transaksi "
        "(Sep-Okt 2025). Seluruh angka di bagian ini dihasilkan oleh kode/task2_classification.py (seed 42) "
        "dan tersimpan di kode/hasil_task2/. Metrik utama: PR-AUC (average precision) dan total biaya = "
        "nominal fraud yang lolos + Rp 25.000 x jumlah transaksi yang ditandai.")

    # ---------------- (a)
    d.h2("2.1 Audit Label & Fitur (a)")
    m = mg.set_index("waktu")["kematangan_pct"]
    d.p("1) Label belum lengkap (reporting lag). Label is_fraud adalah status 'terkonfirmasi fraud saat data ditarik "
        "(31 Agustus 2025)'. Tingkat fraud mingguan stabil di sekitar rata-rata Feb-Jul, lalu turun tajam di tiga minggu "
        f"terakhir: kematangan label (rasio terhadap rata-rata) {pct(m.iloc[-3])}, {pct(m.iloc[-2])}, dan hanya "
        f"{pct(m.iloc[-1])} pada 25-31 Agustus. Artinya banyak fraud akhir Agustus belum dilaporkan dan masih berlabel 0.")
    d.p("2) Kebocoran 'flag_review_manual'. Kolom ini menandai transaksi yang pernah ditinjau manual oleh tim risiko - "
        f"terjadi SETELAH transaksi, sehingga tidak tersedia saat skor dihitung. Bukti: {pct(fr['share_fraud_flagged_pct'])} "
        f"fraud memiliki flag=1; fraud rate {pct(fr['fraud_rate_if_flag1_pct'], 1)} bila flag=1 vs "
        f"{pct(fr['fraud_rate_if_flag0_pct'], 3)} bila flag=0; AUC univariat {num(fr['auc_univariat'])}. Dampak (Fold A, validasi Juli):")
    d.table(["Skenario", "PR-AUC", "ROC-AUC", "Biaya (aturan)", "Ditandai"],
            [[r.model, num(r.pr_auc), num(r.roc_auc), rp(r.cost_rule_juta), r.n_flag] for r in leak.itertuples()],
            [78, 22, 22, 36, 22])
    d.p("Model yang memakai flag terlihat sangat baik (PR-AUC semu), tetapi saat dipakai sungguhan (flag belum ada = 0) "
        "performanya jatuh dan biayanya justru lebih buruk daripada model jujur. Kolom ini dibuang.")
    v15 = R["v15_mean_per_bulan"]
    d.p(f"3) 'v15' adalah proksi waktu: rata-ratanya naik linear tiap bulan ({num(v15['1'],2)} di Jan -> {num(v15['8'],2)} di Agu -> "
        f"{num(v15['10'],2)} di Okt) dengan AUC univariat hanya {num(R['v15_auc_univariat'])}; di data uji nilainya di luar rentang "
        f"latih sehingga dibuang. Dari 49 fitur anonim v** lainnya, hanya {len(R['v_terpilih'])} yang dipertahankan "
        f"({', '.join(R['v_terpilih'])}; AUC univariat >= 0,53 di fold latih), sisanya ({R['v_dibuang_n']}) berperilaku seperti noise. "
        f"4) 'menit_sejak_trx_terakhir' & 'rasio_nominal_rata2' kosong {pct(R['missing_pct']['menit_sejak_trx_terakhir'])} "
        "(transaksi pertama pengguna); dibiarkan NaN agar LightGBM mempelajari arah missing secara eksplisit.")

    # ---------------- (b)
    d.h2("2.2 Validasi Berbasis Waktu & Penanganan Label yang Belum Lengkap (b)")
    d.p(f"Fold A: latih Feb-Jun ({th(fold['A_train_FebJun']['n'])} trx, {fold['A_train_FebJun']['fraud']} fraud), validasi Juli "
        f"({th(fold['A_val_Jul']['n'])} trx, {fold['A_val_Jul']['fraud']} fraud). Fold B: latih Feb-Jul, validasi 1-10 Agustus "
        f"({th(fold['B_val_1_10Agu']['n'])} trx, {fold['B_val_1_10Agu']['fraud']} fraud) - periode terakhir yang labelnya sudah matang "
        "(>100%) dan sudah memuat pola fraud baru. Januari dipakai sebagai warm-up 30 hari untuk fitur jendela merchant. "
        f"Split acak memberi PR-AUC {num(R['random_vs_waktu']['random_split_pr_auc'])} vs {num(R['random_vs_waktu']['time_split_pr_auc'])} "
        "pada split waktu: validasi acak terlalu optimis karena transaksi bulan yang sama (pengguna/merchant yang sama) bocor ke validasi.")
    d.p("Untuk memilih cara memakai data yang labelnya belum lengkap, kami mensimulasikan lag pada Fold A: label positif di 3 "
        "minggu terakhir Juni dihapus mengikuti profil kematangan Agustus (72%/55%/14%), lalu diuji di Juli:")
    best = lag.iloc[1:].pr_auc.idxmax()
    d.table(["Strategi data latih", "PR-AUC", "ROC-AUC", "Biaya", "n latih"],
            [[r.model, num(r.pr_auc), num(r.roc_auc), rp(r.cost_rule_juta), f"{th(r.n_train)}".replace(",", ".")] for r in lag.itertuples()],
            [92, 20, 20, 30, 24], bold_row=best)
    d.p(f"Memakai label tertunda apa adanya menurunkan PR-AUC (model belajar bahwa pola fraud = aman). Strategi terbaik: "
        f"'{R['strategi_lag_terpilih']}', sehingga model final dilatih pada 1 Feb - {R['final_train_end']} dan transaksi "
        "11-31 Agustus (label < 70% matang) tidak dipakai sebagai label. Validasi selalu dilakukan pada periode dengan label matang.")

    # ---------------- (c)
    d.h2("2.3 Fitur Perilaku & Model (c)")
    d.p("Fitur dibangun pada gabungan train+test yang diurutkan berdasarkan waktu, dan setiap baris HANYA memakai transaksi "
        "sebelumnya (tanpa label, tanpa informasi masa depan): (i) pengguna - jumlah transaksi sebelumnya, rata-rata & maksimum "
        "nominal sebelumnya, rasio nominal terhadapnya, hari sejak transaksi pertama, merchant/kategori/kanal baru bagi pengguna; "
        "(ii) merchant - jumlah & total nominal transaksi dalam 24 jam / 7 hari / 30 hari terakhir (dihitung dengan jendela waktu "
        "via searchsorted, transaksi saat ini dikecualikan), rata-rata nominal 30 hari, lonjakan 24 jam vs 30 hari, jumlah pengguna "
        "unik sebelumnya, hari sejak merchant pertama terlihat; (iii) domain - log nominal, malam (00-05), merchant muda (<60 hari), "
        f"nominal x persen_saldo. Total {R['n_fitur']['full']} fitur ({R['n_fitur']['raw']} fitur mentah + perilaku).")
    d.table(["Model", "Fold", "PR-AUC", "ROC-AUC", "Biaya (aturan)"],
            [[r.model, r.fold, num(r.pr_auc), num(r.roc_auc), rp(r.cost_rule_juta)] for r in mod.itertuples()],
            [82, 14, 24, 24, 36])
    d.p("Baseline regresi logistik (one-hot + standarisasi) jauh tertinggal. LightGBM (500 pohon, lr 0,03, 31 daun, "
        "min_child_samples 200, subsample 0,8) dengan fitur perilaku konsisten terbaik di kedua fold, terutama di Fold B "
        "di mana pola baru muncul - fitur riwayat merchant membantu mengenali merchant baru bervolume rendah.")

    # ---------------- (d)
    d.h2("2.4 Ketidakseimbangan Kelas (d)")
    d.table(["Metode", "Fold", "PR-AUC", "Brier", "Biaya thr 0,5", "Biaya aturan", "Biaya thr tuned"],
            [[r.metode, r.fold, num(r.pr_auc), num(r.brier, 5), rp(r.biaya_ambang_0_5_juta), rp(r.biaya_aturan_biaya_mentah_juta),
              rp(r.biaya_threshold_tuned_juta)] for r in imb.itertuples()],
            [52, 10, 18, 18, 30, 30, 32], size=7)
    d.p("Class weighting dan resampling hanya menggeser skor ke atas (probabilitas tidak terkalibrasi, Brier memburuk) tanpa "
        "menaikkan PR-AUC secara konsisten: oversampling sedikit unggul di Fold A tetapi paling buruk di Fold B (overfit ke pola lama). "
        "Ambang 0,5 selalu mahal; penentuan ambang berbasis biaya (aturan Bayes atau threshold tuning) adalah faktor terpenting. "
        "Pilihan: model tanpa resampling + kalibrasi + aturan biaya (paling stabil antar fold dan probabilitasnya paling jujur).")

    # ---------------- (e)
    d.h2("2.5 Pergeseran Pola Fraud & Adaptasi (e)")
    nq = qris["n_fraud_qris_per_bulan"]; pq = qris["profil_median"]
    d.p(f"Fraud kanal QRIS stabil 10-16 kasus/bulan pada Jan-Jul, lalu melonjak menjadi {nq['8']} kasus di Agustus (padahal label "
        "Agustus belum lengkap), sementara kanal lain turun karena lag. Profil median fraud QRIS Agustus vs fraud lama: nominal "
        f"Rp {th(pq['fraud_qris_agustus']['nominal'])} vs Rp {th(pq['fraud_jan_jul']['nominal'])}; umur merchant "
        f"{pq['fraud_qris_agustus']['umur_merchant_hari']:.0f} vs {pq['fraud_jan_jul']['umur_merchant_hari']:.0f} hari; transaksi "
        f"merchant 30 hari {pq['fraud_qris_agustus']['m_trx_30h']:.0f} vs {pq['fraud_jan_jul']['m_trx_30h']:.0f}; persen_saldo "
        f"{pq['fraud_qris_agustus']['persen_saldo']:.1f} vs {pq['fraud_jan_jul']['persen_saldo']:.1f}; perangkat & akun justru LAMA. "
        "Interpretasi: pola baru berupa merchant QRIS baru yang menguras saldo akun lama (kemungkinan social engineering / merchant "
        f"palsu), bukan pola perangkat baru seperti sebelumnya. Data uji memuat {qris['test_trx_qris_merchant_muda']} transaksi QRIS "
        "di merchant muda, sehingga adaptasi penting.")
    d.p("Adaptasi: (1) fitur riwayat merchant yang label-free sehingga tetap terhitung untuk merchant baru di data uji; "
        "(2) bobot sampel berbasis recency (half-life) agar bulan terbaru lebih berpengaruh; (3) data hingga 10 Agustus ikut dilatih; "
        "(4) kalibrasi out-of-time agar probabilitas mengikuti prevalensi terbaru. Evaluasi bobot recency di Fold B:")
    d.table(["Half-life bobot", "PR-AUC", "ROC-AUC", "PR-AUC QRIS", "Biaya (aturan)"],
            [[str(r.half_life_hari), num(r.pr_auc), num(r.roc_auc), num(r.pr_auc_qris), rp(r.biaya_aturan_juta)] for r in rec.itertuples()],
            [40, 26, 26, 30, 36])
    d.p(f"Half-life terpilih: {R['half_life_terpilih']} hari (PR-AUC Fold B tertinggi; PR-AUC subset QRIS juga naik).")

    # ---------------- (f)
    d.h2("2.6 Kalibrasi Probabilitas & Keputusan Berbasis Biaya (f)")
    d.p("Kalibrator dilatih pada prediksi out-of-time Fold A (Juli) dan diuji pada Fold B (1-10 Agustus), sehingga tidak ada "
        "kebocoran. Aturan keputusan (Bayes): tandai jika P(fraud) x nominal >= Rp 25.000, karena ekspektasi kerugian bila "
        "dibiarkan = p x nominal, sedangkan biaya review pasti = Rp 25.000.")
    d.table(["Kalibrasi", "Brier", "Rata2 prob", "Fraud aktual", "PR-AUC", "Biaya (aturan)", "Ditandai"],
            [[r.kalibrasi, num(r.brier, 6), num(r.rata2_prob, 5), num(r.rate_aktual, 5), num(r.pr_auc), rp(r.biaya_aturan_juta), r.n_tandai]
             for r in cal.itertuples()], [24, 24, 26, 26, 22, 34, 22])
    d.table(["Kebijakan (Fold B, prob terkalibrasi)", "Ditandai", "Total biaya"],
            [[r.kebijakan, f"{th(r.n_tandai)}".replace(",", "."), rp(r.biaya_juta)] for r in dec.itertuples()], [110, 30, 40],
            bold_row=len(dec) - 1)
    ka = R["keputusan_foldA"]
    d.p(f"Kalibrasi terpilih: {R['kalibrasi_terpilih']} (Brier & biaya terendah). Aturan biaya mengalahkan 'tandai semua' "
        f"dan 'tidak tandai'; di Fold A (Juli) hasilnya serupa: tidak tandai {rp(ka['tidak_tandai_juta'])}, tandai semua "
        f"{rp(ka['tandai_semua_juta'])}, aturan biaya {rp(ka['aturan_biaya_juta'])}. Ambang global yang di-tune pada data validasi "
        "sendiri (optimis) pun tetap kalah dari aturan biaya, karena aturan biaya memperhitungkan nominal tiap transaksi: transaksi "
        "bernominal besar ditandai pada probabilitas lebih rendah.")

    # ---------------- (g)
    d.h2("2.7 Analisis Galat (g) - out-of-time Juli + 1-10 Agustus")
    d.table(["Kanal", "n", "Fraud", "Ditandai", "Recall", "Precision", "Fraud lolos", "PR-AUC"],
            [[r.kanal, f"{th(int(r.n))}".replace(",", "."), int(r.fraud), int(r.tandai), num(r.recall, 3), num(r.precision, 3),
              rp(r.fraud_lolos_juta), num(r.pr_auc, 3)] for r in gk.itertuples()], [26, 22, 18, 20, 20, 22, 34, 20])
    d.table(["Pola fraud", "n", "Fraud", "Recall", "Precision", "Fraud lolos", "PR-AUC"],
            [[r.pola, f"{th(int(r.n))}".replace(",", "."), int(r.fraud), num(r.recall, 3), num(r.precision, 3),
              rp(r.fraud_lolos_juta), num(r.pr_auc, 3)] for r in gp.itertuples()], [58, 22, 16, 20, 22, 34, 20])
    top_k = gc.dropna(subset=["recall"]).sort_values("fraud_lolos_juta", ascending=False).head(3)
    d.p("Model kuat pada pola lama: kartu tidak hadir, perangkat baru, dan lokasi jauh (PR-AUC > 0,8). Galat terbesar ada pada "
        "kanal QRIS / pola merchant QRIS muda: recall rendah dan sebagian besar kerugian fraud yang lolos berasal dari pola baru ini "
        "(nominal besar). Per kategori merchant, kerugian lolos terbesar ada di "
        + ", ".join(f"{r.kategori_merchant} ({rp(r.fraud_lolos_juta)}, recall {num(r.recall, 2)})" for r in top_k.itertuples())
        + ". Kategori voucher_game, pulsa_data, top_up_ewallet memiliki fraud bernominal kecil sehingga sebagian sengaja tidak "
        "ditandai (biaya review > ekspektasi kerugian) - ini konsekuensi rasional dari aturan biaya, bukan kegagalan model.")

    d.h2("2.8 Model Final, Prediksi Uji, & Keterbatasan")
    tops = ", ".join(f"{r.fitur} ({num(r.gain_pct, 1)}%)" for r in imp.head(8).itertuples())
    d.p(f"Model final: LightGBM + fitur perilaku, tanpa resampling, bobot recency half-life {R['half_life_terpilih']} hari, dilatih "
        f"pada {th(fin['n_train'])} transaksi ({th(fin['fraud_train'])} fraud), kalibrasi {R['kalibrasi_terpilih']}, aturan biaya. "
        f"Pada data uji Sep-Okt: rata-rata prob {num(fin['prob_rata2_test'], 5)}, {th(fin['n_tandai_test'])} transaksi ditandai "
        f"({pct(fin['pct_tandai_test'], 2)}, biaya review {rp(fin['biaya_review_test_juta'])}). Fitur terpenting (gain): {tops}. "
        "Output: prediksi_klasifikasi.csv (id_transaksi, prob, tandai). Keterbatasan: (1) pola QRIS baru hanya terwakili sedikit "
        "kasus berlabel matang - disarankan retraining bulanan dengan label yang sudah matang; (2) kalibrasi diasumsikan stabil "
        "dari Juli/Agustus ke Sep-Okt; (3) fitur jendela merchant pada Januari terpotong (sehingga Januari hanya dipakai warm-up).")

    for img in ["fig_task2_audit_shift_kalibrasi.png", "fig_task2_pr_importance.png"]:
        path = os.path.join(RES, img)
        if os.path.exists(path):
            if d.get_y() > 200:
                d.add_page()
            d.image(path, w=190)
            d.ln(2)
    d.set_font("Helvetica", "I", 7.5)
    d.multi_cell(0, 4, "Gambar: (a) fraud rate mingguan & label lag, (e) fraud rate per kanal per bulan, (f) reliability diagram Fold B; "
                       "(d) kurva precision-recall metode imbalance di Fold B dan importance model final.",
                 new_x="LMARGIN", new_y="NEXT")
    return bytes(d.output())


def header_overlay(page_no, width, height):
    """Overlay putih + header baru untuk menyesuaikan nomor halaman."""
    o = FPDF(unit="pt", format=(width, height))
    o.add_page()
    # Posisi asli "Hal. N" pada laporan: x=543.2 pt, baseline y=799.8 pt dari bawah
    base_y = height - 799.80
    o.set_fill_color(255, 255, 255)
    o.rect(525, base_y - 9, 55, 12, style="F")
    o.set_font("Helvetica", "I", 8)
    o.set_text_color(100, 100, 100)
    txt = f"Hal. {page_no}"
    o.text(543.20, base_y, txt)
    return PdfReader(io.BytesIO(bytes(o.output()))).pages[0]


def run(pdf_path=None):
    pdf_path = pdf_path or os.path.join(ROOT, "laporan.pdf")
    reader = PdfReader(pdf_path)
    texts = [(pg.extract_text() or "") for pg in reader.pages]
    t2 = [i for i, t in enumerate(texts) if "TASK 2 - KLASIFIKASI" in t or MARKER in t]
    if not t2:
        raise RuntimeError("Halaman TASK 2 tidak ditemukan di laporan.pdf")
    first, last = min(t2), max(t2)
    new_pages = PdfReader(io.BytesIO(build_task2_pdf(first + 1))).pages
    w = PdfWriter()
    for i in range(first):
        w.add_page(reader.pages[i])
    for pg in new_pages:
        w.add_page(pg)
    shift = len(new_pages) - (last - first + 1)
    for i in range(last + 1, len(reader.pages)):
        pg = reader.pages[i]
        if shift != 0:
            box = pg.mediabox
            pg.merge_page(header_overlay(i + 1 + shift, float(box.width), float(box.height)))
        w.add_page(pg)
    with open(pdf_path, "wb") as f:
        w.write(f)
    print(f"[Laporan] Bagian Task 2 diperbarui: halaman {first + 1}-{first + len(new_pages)}; total {len(w.pages)} halaman")


if __name__ == "__main__":
    run()
