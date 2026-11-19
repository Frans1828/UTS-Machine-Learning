"""
=============================================================================
UTS MACHINE LEARNING - TASK 2: CLASSIFICATION (FRAUD DETECTION)
Author : Ahmad Ghazali El-Faruqi HSB (101032330080)
Group  : Kelompok 16 (bersama Fransisco Sitepu, 101032300004)
=============================================================================
Pipeline lengkap Task 2 (a-g sesuai soal UTS):

a. Audit label & fitur
   - Label lag: tingkat fraud per minggu turun tajam di 3 minggu terakhir
     Agustus (label belum lengkap saat data ditarik 31 Agustus 2025).
   - `flag_review_manual` = hasil tinjauan manual SETELAH transaksi -> bocor.
     Dampaknya diukur (PR-AUC semu vs PR-AUC saat fitur tidak tersedia).
   - `v15` adalah proksi waktu (naik linear tiap bulan, tanpa sinyal) -> dibuang.
   - Fitur v01-v50 anonim lain diseleksi berdasarkan AUC univariat di fold latih.
b. Validasi berbasis waktu
   - Fold A: latih Feb-Jun, validasi Jul (label sudah matang).
   - Fold B: latih Feb-Jul, validasi 1-10 Agustus (pola baru mulai muncul).
   - Eksperimen simulasi label lag untuk memilih strategi data Agustus.
c. Fitur perilaku pengguna & merchant (hanya memakai transaksi SEBELUMNYA,
   tanpa label) dihitung pada gabungan train+test yang diurutkan waktu.
d. Ketidakseimbangan kelas: tanpa penanganan, class weighting, random
   undersampling, random oversampling, + threshold tuning berbasis biaya.
e. Pergeseran pola: fraud QRIS di merchant baru muncul sejak Agustus 2025.
   Adaptasi: fitur merchant berbasis riwayat + bobot sampel berbasis recency
   + data Agustus ikut dilatih.
f. Kalibrasi (Platt / isotonic, out-of-time) dan aturan keputusan biaya:
   tandai jika P(fraud) x nominal >= Rp 25.000. Dibandingkan dengan
   "tandai semua" dan "tidak tandai sama sekali".
g. Analisis galat per kanal, kategori merchant, dan pola fraud.

Output:
   - prediksi_klasifikasi.csv  (id_transaksi, prob, tandai)
   - kode/hasil_task2/*.csv, *.json, *.png (semua angka & grafik laporan)

Seed: 42. Satu perintah: python kode/task2_classification.py
"""

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

SEED = 42
REVIEW_COST = 25_000
CAT_COLS = ["kanal", "kategori_merchant", "tipe_perangkat", "email_domain"]
LEAK_COLS = ["flag_review_manual"]          # tersedia hanya setelah transaksi
TIME_PROXY_COLS = ["v15"]                   # naik linear mengikuti bulan
ID_COLS = ["id_transaksi", "waktu", "id_pengguna", "id_merchant"]

LGB_PARAMS = dict(
    n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=200,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
    random_state=SEED, n_jobs=-1, verbose=-1,
)


# ----------------------------------------------------------------------------
# Utilitas
# ----------------------------------------------------------------------------
def log(msg):
    print(f"[Task 2] {msg}", flush=True)


def total_cost(y, nominal, flag):
    """cost = jumlah nominal fraud yang lolos + 25.000 x jumlah transaksi ditandai."""
    y = np.asarray(y); nominal = np.asarray(nominal); flag = np.asarray(flag).astype(bool)
    return float(nominal[(y == 1) & ~flag].sum() + REVIEW_COST * flag.sum())


def cost_rule(prob, nominal, c=REVIEW_COST):
    """Aturan Bayes: tandai jika ekspektasi kerugian >= biaya review."""
    return (np.asarray(prob) * np.asarray(nominal) >= c).astype(int)


def best_threshold(prob, y, nominal):
    """Ambang probabilitas global yang meminimalkan biaya (threshold tuning)."""
    grid = np.unique(np.quantile(prob, np.linspace(0.90, 0.99999, 400)))
    costs = [total_cost(y, nominal, prob >= t) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def metrics(y, prob, nominal, name, extra=None):
    flag = cost_rule(prob, nominal)
    out = dict(
        model=name,
        pr_auc=round(average_precision_score(y, prob), 4),
        roc_auc=round(roc_auc_score(y, prob), 4),
        cost_rule_juta=round(total_cost(y, nominal, flag) / 1e6, 2),
        n_flag=int(flag.sum()),
    )
    if extra:
        out.update(extra)
    return out


def platt_fit(p, y):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return lr


def platt_apply(lr, p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    return lr.predict_proba(z)[:, 1]


def resolve_data_dir(data_dir):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    cands = [data_dir] if data_dir else []
    cands += [
        os.path.join(root, "..", "TUGAS MACHINE LEARNING", "data_uts_ml-20261119T075355Z-1-001", "data_uts_ml"),
        os.path.join(root, "data"), os.path.join(root, "data_uts_ml"),
        os.path.join(root, "..", "data_uts_ml"), "data_uts_ml",
    ]
    for p in cands:
        if p and os.path.exists(os.path.join(p, "klasifikasi_train.csv.gz")):
            return os.path.abspath(p)
    raise FileNotFoundError(f"klasifikasi_train.csv.gz tidak ditemukan di: {cands}")


# ----------------------------------------------------------------------------
# (c) Fitur perilaku tanpa informasi masa depan
# ----------------------------------------------------------------------------
def _window_count_sum(key_codes, t_sec, values, window_sec):
    """Jumlah transaksi & total nilai dalam jendela (t - w, t) per kunci,
    TIDAK termasuk transaksi saat ini. Data harus terurut (kunci, waktu)."""
    big = np.int64(10 ** 11)
    k = key_codes.astype(np.int64) * big + t_sec.astype(np.int64)
    left = np.searchsorted(k, k - window_sec, side="right")
    # posisi transaksi saat ini (pertama dengan kunci & detik yang sama)
    cur = np.searchsorted(k, k, side="left")
    cnt = cur - left
    cs = np.concatenate([[0.0], np.cumsum(values)])
    sm = cs[cur] - cs[left]
    return cnt, sm


def build_behavioral_features(df):
    """df: gabungan train+test. Semua fitur hanya memakai transaksi sebelumnya."""
    df = df.sort_values(["waktu", "id_transaksi"]).reset_index(drop=True)
    t_sec = (df["waktu"].astype("int64") // 10 ** 9).to_numpy()
    nom = df["nominal"].to_numpy(dtype=float)
    feats = pd.DataFrame(index=df.index)

    # --- Riwayat pengguna ----------------------------------------------------
    g = df.groupby("id_pengguna", sort=False)
    n_prev = g.cumcount()
    feats["u_n_prev"] = n_prev
    cum_nom = g["nominal"].cumsum() - df["nominal"]
    feats["u_nom_mean_prev"] = np.where(n_prev > 0, cum_nom / n_prev.replace(0, 1), np.nan)
    feats["u_nom_ratio_prev"] = df["nominal"] / feats["u_nom_mean_prev"]
    prev_max = g["nominal"].cummax().groupby(df["id_pengguna"]).shift(1)
    feats["u_nom_over_max"] = df["nominal"] / prev_max
    first_t = g["waktu"].transform("min")
    feats["u_hari_sejak_pertama"] = (df["waktu"] - first_t).dt.total_seconds() / 86400
    feats["u_merchant_baru"] = (df.groupby(["id_pengguna", "id_merchant"]).cumcount() == 0).astype(int)
    feats["u_kategori_baru"] = (df.groupby(["id_pengguna", "kategori_merchant"]).cumcount() == 0).astype(int)
    feats["u_kanal_baru"] = (df.groupby(["id_pengguna", "kanal"]).cumcount() == 0).astype(int)
    feats["u_n_merchant_prev"] = feats["u_merchant_baru"].groupby(df["id_pengguna"]).cumsum() - feats["u_merchant_baru"]

    # --- Riwayat merchant (jendela waktu, konsisten antara train & test) -------
    order = np.lexsort((t_sec, df["id_merchant"].astype("category").cat.codes.to_numpy()))
    mcode = df["id_merchant"].astype("category").cat.codes.to_numpy()[order]
    ts = t_sec[order]; nv = nom[order]
    for name, w in [("24j", 86400), ("7h", 7 * 86400), ("30h", 30 * 86400)]:
        c, s = _window_count_sum(mcode, ts, nv, w)
        cc = np.empty_like(c); ss = np.empty_like(s)
        cc[order] = c; ss[order] = s
        feats[f"m_trx_{name}"] = cc
        feats[f"m_nominal_{name}"] = ss
    feats["m_nom_rata2_30h"] = feats["m_nominal_30h"] / feats["m_trx_30h"].replace(0, np.nan)
    feats["m_rasio_nominal_30h"] = df["nominal"] / feats["m_nom_rata2_30h"]
    feats["m_lonjakan_24j_vs_30h"] = feats["m_trx_24j"] / (feats["m_trx_30h"] / 30 + 1)
    gm = df.groupby("id_merchant", sort=False)
    feats["m_hari_sejak_pertama"] = (df["waktu"] - gm["waktu"].transform("min")).dt.total_seconds() / 86400
    # pengguna unik merchant sebelum transaksi ini
    first_pair = (df.groupby(["id_merchant", "id_pengguna"]).cumcount() == 0).astype(int)
    feats["m_user_unik_prev"] = first_pair.groupby(df["id_merchant"]).cumsum() - first_pair
    feats["m_share_user_baru_prev"] = feats["m_user_unik_prev"] / (gm.cumcount() + 1)

    # --- Interaksi domain -----------------------------------------------------
    feats["nominal_log"] = np.log1p(df["nominal"])
    feats["hari_minggu"] = df["waktu"].dt.dayofweek
    feats["malam"] = df["jam"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    feats["merchant_muda"] = (df["umur_merchant_hari"] < 60).astype(int)
    feats["nominal_x_persen_saldo"] = feats["nominal_log"] * df["persen_saldo"]

    return pd.concat([df, feats], axis=1)


# ----------------------------------------------------------------------------
# Model helpers
# ----------------------------------------------------------------------------
def to_lgb_frame(df, cols):
    X = df[cols].copy()
    for c in CAT_COLS:
        if c in X:
            X[c] = pd.Categorical(X[c], categories=sorted(CAT_LEVELS[c]))
    return X


CAT_LEVELS = {}


def fit_lgb(X, y, w=None, **kw):
    params = dict(LGB_PARAMS); params.update(kw)
    m = lgb.LGBMClassifier(**params)
    m.fit(X, y, sample_weight=w, categorical_feature=[c for c in CAT_COLS if c in X.columns])
    return m


def recency_weight(t, ref, half_life_days):
    if half_life_days is None:
        return np.ones(len(t))
    age = (ref - t).dt.total_seconds().to_numpy() / 86400
    return 0.5 ** (age / half_life_days)


def fit_logreg(Xtr, ytr, Xva, num_cols):
    """Baseline sederhana: regresi logistik + one-hot kategori."""
    med = Xtr[num_cols].median()
    Ztr = Xtr[num_cols].fillna(med)
    Zva = Xva[num_cols].fillna(med)
    for Z in (Ztr, Zva):
        for c in ["nominal", "saldo_sebelum", "nominal_24j", "jarak_lokasi_km", "volume_merchant_30h",
                  "menit_sejak_trx_terakhir"]:
            if c in Z:
                Z[c] = np.log1p(Z[c].clip(lower=0))
    dtr = pd.get_dummies(Xtr[CAT_COLS].astype(str)); dva = pd.get_dummies(Xva[CAT_COLS].astype(str)).reindex(columns=dtr.columns, fill_value=0)
    sc = StandardScaler().fit(Ztr)
    A = np.hstack([sc.transform(Ztr), dtr.to_numpy(float)])
    B = np.hstack([sc.transform(Zva), dva.to_numpy(float)])
    A = np.clip(A, -10, 10); B = np.clip(B, -10, 10)
    lr = LogisticRegression(C=0.1, max_iter=2000).fit(A, ytr)
    return lr.predict_proba(B)[:, 1]


# ----------------------------------------------------------------------------
# Pipeline utama
# ----------------------------------------------------------------------------
def run_task2(data_dir=None, output_dir=None):
    t0 = time.time()
    data_dir = resolve_data_dir(data_dir)
    here = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(here, ".."))
    res_dir = os.path.join(here, "hasil_task2")
    os.makedirs(res_dir, exist_ok=True)
    np.random.seed(SEED)
    R = {}  # semua angka laporan

    log(f"Memuat data dari {data_dir}")
    tr = pd.read_csv(os.path.join(data_dir, "klasifikasi_train.csv.gz"), parse_dates=["waktu"])
    te = pd.read_csv(os.path.join(data_dir, "klasifikasi_test.csv.gz"), parse_dates=["waktu"])
    for c in CAT_COLS:
        CAT_LEVELS[c] = set(tr[c].astype(str)) | set(te[c].astype(str))
    R["n_train"], R["n_test"] = len(tr), len(te)
    R["prevalensi_train"] = round(tr["is_fraud"].mean() * 100, 3)

    # ======================================================================
    # (a) AUDIT LABEL & FITUR
    # ======================================================================
    log("(a) Audit label: tingkat fraud per bulan & per minggu")
    bulanan = tr.groupby(tr["waktu"].dt.to_period("M")).agg(n=("is_fraud", "size"), fraud=("is_fraud", "sum"),
                                                           rate_pct=("is_fraud", lambda s: round(100 * s.mean(), 3)))
    bulanan.to_csv(os.path.join(res_dir, "a_fraud_per_bulan.csv"))
    mingguan = tr[tr["waktu"] >= "2025-06-02"].groupby(tr["waktu"].dt.to_period("W")).agg(
        n=("is_fraud", "size"), rate_pct=("is_fraud", lambda s: round(100 * s.mean(), 3)))
    base_rate = tr[(tr["waktu"] >= "2025-02-01") & (tr["waktu"] < "2025-08-01")]["is_fraud"].mean()
    mingguan["kematangan_pct"] = (mingguan["rate_pct"] / (100 * base_rate) * 100).round(1)
    mingguan.to_csv(os.path.join(res_dir, "a_fraud_per_minggu.csv"))
    R["label_lag_minggu"] = {str(k): v for k, v in mingguan["kematangan_pct"].items()}

    # flag_review_manual vs label
    ct = pd.crosstab(tr["flag_review_manual"], tr["is_fraud"])
    R["flag_review"] = dict(
        rate_train_pct=round(100 * tr["flag_review_manual"].mean(), 2),
        rate_test_pct=round(100 * te["flag_review_manual"].mean(), 2),
        fraud_rate_if_flag1_pct=round(100 * tr.loc[tr.flag_review_manual == 1, "is_fraud"].mean(), 2),
        fraud_rate_if_flag0_pct=round(100 * tr.loc[tr.flag_review_manual == 0, "is_fraud"].mean(), 3),
        share_fraud_flagged_pct=round(100 * ct.loc[1, 1] / ct[1].sum(), 1),
        auc_univariat=round(roc_auc_score(tr["is_fraud"], tr["flag_review_manual"]), 4),
    )
    # v15 = proksi waktu
    v15_bulan = pd.concat([tr, te]).groupby(pd.concat([tr, te])["waktu"].dt.month)["v15"].mean().round(3)
    R["v15_mean_per_bulan"] = v15_bulan.to_dict()
    R["v15_auc_univariat"] = round(max(roc_auc_score(tr.is_fraud, tr.v15), 1 - roc_auc_score(tr.is_fraud, tr.v15)), 4)
    R["missing_pct"] = {c: round(100 * tr[c].isna().mean(), 2) for c in ["menit_sejak_trx_terakhir", "rasio_nominal_rata2"]}

    # ======================================================================
    # (c) FITUR PERILAKU
    # ======================================================================
    log("(c) Membangun fitur perilaku pengguna & merchant (hanya masa lalu)")
    te2 = te.copy(); te2["is_fraud"] = -1
    al = build_behavioral_features(pd.concat([tr, te2], ignore_index=True))
    al["is_test"] = (al["is_fraud"] == -1).astype(int)

    beh_cols = [c for c in al.columns if c.startswith(("u_", "m_")) or c in
                ["nominal_log", "hari_minggu", "malam", "merchant_muda", "nominal_x_persen_saldo"]]
    raw_num = [c for c in tr.select_dtypes("number").columns
               if c not in ["is_fraud"] + LEAK_COLS + TIME_PROXY_COLS and not c.startswith("v")]
    v_all = [c for c in tr.columns if c.startswith("v") and c not in TIME_PROXY_COLS]

    # Periode (Januari = warm-up 30 hari untuk fitur jendela merchant)
    L = al[al.is_test == 0]
    T = al[al.is_test == 1].copy()
    foldA_tr = L[(L.waktu >= "2025-02-01") & (L.waktu < "2025-07-01")]
    foldA_va = L[(L.waktu >= "2025-07-01") & (L.waktu < "2025-08-01")]
    foldB_tr = L[(L.waktu >= "2025-02-01") & (L.waktu < "2025-08-01")]
    foldB_va = L[(L.waktu >= "2025-08-01") & (L.waktu < "2025-08-11")]
    R["fold"] = {k: dict(n=len(d), fraud=int(d.is_fraud.sum())) for k, d in
                 [("A_train_FebJun", foldA_tr), ("A_val_Jul", foldA_va), ("B_train_FebJul", foldB_tr), ("B_val_1_10Agu", foldB_va)]}

    # Seleksi fitur anonim v** di dalam fold latih A saja
    v_auc = {c: max(roc_auc_score(foldA_tr.is_fraud, foldA_tr[c]), 1 - roc_auc_score(foldA_tr.is_fraud, foldA_tr[c])) for c in v_all}
    v_keep = sorted([c for c, a in v_auc.items() if a >= 0.53])
    R["v_terpilih"] = {c: round(v_auc[c], 3) for c in v_keep}
    R["v_dibuang_n"] = len(v_all) - len(v_keep)

    FEAT_RAW = CAT_COLS + raw_num + v_keep
    FEAT_FULL = FEAT_RAW + beh_cols
    FEAT_LEAK = FEAT_FULL + LEAK_COLS
    R["n_fitur"] = dict(raw=len(FEAT_RAW), full=len(FEAT_FULL))

    def run(cols, trd, vad, w=None, **kw):
        m = fit_lgb(to_lgb_frame(trd, cols), trd.is_fraud.to_numpy(), w, **kw)
        return m, m.predict_proba(to_lgb_frame(vad, cols))[:, 1]

    # ---- Dampak kebocoran flag_review_manual (Fold A) ----------------------
    log("(a) Mengukur dampak kebocoran flag_review_manual")
    yA, nA = foldA_va.is_fraud.to_numpy(), foldA_va.nominal.to_numpy()
    m_leak, p_leak = run(FEAT_LEAK, foldA_tr, foldA_va)
    va_noflag = foldA_va.copy(); va_noflag["flag_review_manual"] = 0
    p_leak_real = m_leak.predict_proba(to_lgb_frame(va_noflag, FEAT_LEAK))[:, 1]
    m_full_A, p_full_A = run(FEAT_FULL, foldA_tr, foldA_va)
    leak_tab = [
        metrics(yA, p_leak, nA, "Dengan flag_review_manual (semu)"),
        metrics(yA, p_leak_real, nA, "Model bocor saat flag belum ada (=0)"),
        metrics(yA, p_full_A, nA, "Tanpa flag_review_manual (jujur)"),
    ]
    pd.DataFrame(leak_tab).to_csv(os.path.join(res_dir, "a_dampak_kebocoran.csv"), index=False)
    R["kebocoran"] = leak_tab

    # ======================================================================
    # (b) VALIDASI WAKTU + PERBANDINGAN RANDOM CV + SIMULASI LABEL LAG
    # ======================================================================
    log("(b) Membandingkan validasi waktu vs acak & simulasi label lag")
    rng = np.random.RandomState(SEED)
    pool = pd.concat([foldA_tr, foldA_va])
    idx = rng.permutation(len(pool)); cut = int(0.8 * len(pool))
    r_tr, r_va = pool.iloc[idx[:cut]], pool.iloc[idx[cut:]]
    _, p_rand = run(FEAT_FULL, r_tr, r_va)
    R["random_vs_waktu"] = dict(
        random_split_pr_auc=round(average_precision_score(r_va.is_fraud, p_rand), 4),
        time_split_pr_auc=round(average_precision_score(yA, p_full_A), 4),
    )

    # Simulasi lag: hapus label positif di 3 minggu terakhir Juni sesuai profil kematangan Agustus
    mat = {1: 0.72, 2: 0.55, 3: 0.14}   # minggu ke-3 dari akhir, ke-2, terakhir (lihat a_fraud_per_minggu.csv)
    sim = foldA_tr.copy()
    end = pd.Timestamp("2025-07-01")
    wk = np.select([sim.waktu >= end - pd.Timedelta(days=7), sim.waktu >= end - pd.Timedelta(days=14),
                    sim.waktu >= end - pd.Timedelta(days=21)], [3, 2, 1], 0)
    keep_p = np.select([wk == 3, wk == 2, wk == 1], [mat[3], mat[2], mat[1]], 1.0)
    flip = (sim.is_fraud.to_numpy() == 1) & (rng.rand(len(sim)) > keep_p)
    sim.loc[flip, "is_fraud"] = 0
    lag_rows = []
    for name, d in [("Label lengkap (oracle)", foldA_tr),
                    ("Label tertunda, dipakai apa adanya", sim),
                    ("Label tertunda, buang 3 minggu terakhir", sim[wk == 0]),
                    ("Label tertunda, buang 1 minggu terakhir", sim[wk < 3]),
                    ("Label tertunda, 3 minggu terakhir: simpan positif, buang negatif",
                     sim[(wk == 0) | (sim.is_fraud.to_numpy() == 1)])]:
        _, p = run(FEAT_FULL, d, foldA_va)
        lag_rows.append(metrics(yA, p, nA, name, dict(n_train=len(d))))
    pd.DataFrame(lag_rows).to_csv(os.path.join(res_dir, "b_simulasi_label_lag.csv"), index=False)
    R["simulasi_lag"] = lag_rows
    R["simulasi_lag_label_dihapus"] = int(flip.sum())

    # ======================================================================
    # (c)+(d) MODEL, FITUR PERILAKU, KETIDAKSEIMBANGAN KELAS
    # ======================================================================
    log("(c,d) Baseline, fitur perilaku, dan perbandingan penanganan imbalance")
    yB, nB = foldB_va.is_fraud.to_numpy(), foldB_va.nominal.to_numpy()
    rows = []
    for fold, trd, vad in [("A", foldA_tr, foldA_va), ("B", foldB_tr, foldB_va)]:
        y, nom = vad.is_fraud.to_numpy(), vad.nominal.to_numpy()
        num_for_lr = [c for c in raw_num + v_keep]
        p_lr = fit_logreg(trd, trd.is_fraud.to_numpy(), vad, num_for_lr)
        rows.append(metrics(y, p_lr, nom, "Baseline: Logistic Regression (fitur mentah)", dict(fold=fold)))
        _, p_raw = run(FEAT_RAW, trd, vad)
        rows.append(metrics(y, p_raw, nom, "LightGBM fitur mentah", dict(fold=fold)))
        _, p_full = run(FEAT_FULL, trd, vad)
        rows.append(metrics(y, p_full, nom, "LightGBM + fitur perilaku", dict(fold=fold)))
    model_tab = pd.DataFrame(rows)
    model_tab.to_csv(os.path.join(res_dir, "c_perbandingan_model.csv"), index=False)
    R["model"] = rows

    imb_rows = []
    imb_preds = {}
    for fold, trd, vad in [("A", foldA_tr, foldA_va), ("B", foldB_tr, foldB_va)]:
        y, nom = vad.is_fraud.to_numpy(), vad.nominal.to_numpy()
        ytr = trd.is_fraud.to_numpy()
        spw = (ytr == 0).sum() / (ytr == 1).sum()
        variants = {}
        variants["1. Tanpa penanganan"] = run(FEAT_FULL, trd, vad)[1]
        variants["2. Class weighting (spw=neg/pos)"] = run(FEAT_FULL, trd, vad, scale_pos_weight=spw)[1]
        # Random undersampling negatif ke rasio 1:20
        rs = np.random.RandomState(SEED)
        pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
        neg_s = rs.choice(neg, size=min(len(neg), 20 * len(pos)), replace=False)
        variants["3. Random undersampling 1:20"] = run(FEAT_FULL, trd.iloc[np.sort(np.r_[pos, neg_s])], vad)[1]
        # Random oversampling positif x20
        pos_o = rs.choice(pos, size=20 * len(pos), replace=True)
        variants["4. Random oversampling x20"] = run(FEAT_FULL, trd.iloc[np.r_[neg, pos_o]], vad)[1]
        for name, p in variants.items():
            thr, c_thr = best_threshold(p, y, nom)
            imb_rows.append(dict(
                fold=fold, metode=name,
                pr_auc=round(average_precision_score(y, p), 4), roc_auc=round(roc_auc_score(y, p), 4),
                brier=round(brier_score_loss(y, p), 5),
                biaya_ambang_0_5_juta=round(total_cost(y, nom, p >= 0.5) / 1e6, 2),
                biaya_aturan_biaya_mentah_juta=round(total_cost(y, nom, cost_rule(p, nom)) / 1e6, 2),
                biaya_threshold_tuned_juta=round(c_thr / 1e6, 2), threshold_tuned=round(thr, 4),
            ))
            imb_preds[(fold, name)] = p
    imb_tab = pd.DataFrame(imb_rows)
    imb_tab.to_csv(os.path.join(res_dir, "d_ketidakseimbangan_kelas.csv"), index=False)
    R["imbalance"] = imb_rows

    # ======================================================================
    # (e) PERGESERAN POLA
    # ======================================================================
    log("(e) Analisis pergeseran pola fraud & adaptasi recency weighting")
    Lm = L.copy(); Lm["bulan"] = Lm.waktu.dt.month
    shift_kanal = Lm.pivot_table(index="bulan", columns="kanal", values="is_fraud", aggfunc="mean").mul(100).round(3)
    shift_kanal.to_csv(os.path.join(res_dir, "e_fraud_rate_kanal_per_bulan.csv"))
    fr = Lm[Lm.is_fraud == 1]
    profil = pd.DataFrame({
        "fraud_qris_agustus": fr[(fr.bulan == 8) & (fr.kanal == "qris")][["nominal", "umur_merchant_hari", "volume_merchant_30h",
                                                                          "persen_saldo", "umur_perangkat_hari", "umur_akun_hari", "m_trx_30h"]].median(),
        "fraud_jan_jul": fr[fr.bulan < 8][["nominal", "umur_merchant_hari", "volume_merchant_30h", "persen_saldo",
                                           "umur_perangkat_hari", "umur_akun_hari", "m_trx_30h"]].median(),
        "non_fraud": Lm[Lm.is_fraud == 0][["nominal", "umur_merchant_hari", "volume_merchant_30h", "persen_saldo",
                                           "umur_perangkat_hari", "umur_akun_hari", "m_trx_30h"]].median(),
    }).round(1)
    profil.to_csv(os.path.join(res_dir, "e_profil_pola_baru.csv"))
    R["pola_qris_agustus"] = dict(
        n_fraud_qris_per_bulan=fr[fr.kanal == "qris"].groupby("bulan").size().to_dict(),
        profil_median=profil.to_dict(),
        test_trx_qris_merchant_muda=int(((T.kanal == "qris") & (T.umur_merchant_hari < 60)).sum()),
        train_trx_qris_merchant_muda_feb_jul=int(((foldB_tr.kanal == "qris") & (foldB_tr.umur_merchant_hari < 60)).sum()),
    )
    # Pergeseran pentingnya fitur: importance Fold A vs model latih s.d. 24 Agu
    # Adaptasi: recency weighting dievaluasi di Fold B (validasi Agustus awal)
    rec_rows = []
    ref = pd.Timestamp("2025-08-01")
    for hl in [None, 90, 45]:
        w = recency_weight(foldB_tr.waktu, ref, hl)
        _, p = run(FEAT_FULL, foldB_tr, foldB_va, w=w)
        q = (foldB_va.kanal == "qris").to_numpy()
        rec_rows.append(dict(
            half_life_hari="tanpa bobot" if hl is None else hl,
            pr_auc=round(average_precision_score(yB, p), 4), roc_auc=round(roc_auc_score(yB, p), 4),
            pr_auc_qris=round(average_precision_score(yB[q], p[q]), 4) if yB[q].sum() else None,
            biaya_aturan_juta=round(total_cost(yB, nB, cost_rule(p, nB)) / 1e6, 2)))
    pd.DataFrame(rec_rows).to_csv(os.path.join(res_dir, "e_recency_weighting.csv"), index=False)
    R["recency"] = rec_rows

    # ======================================================================
    # (f) KALIBRASI & KEPUTUSAN BIAYA
    # ======================================================================
    log("(f) Kalibrasi probabilitas out-of-time & aturan keputusan biaya")
    # Pilih half-life terbaik di Fold B (PR-AUC), konfigurasi tanpa resampling.
    best_rec = max(rec_rows, key=lambda r: r["pr_auc"])
    HL = None if best_rec["half_life_hari"] == "tanpa bobot" else int(best_rec["half_life_hari"])
    R["half_life_terpilih"] = best_rec["half_life_hari"]

    # Kalibrator dilatih pada Fold A (Jul), diuji pada Fold B (1-10 Agu) -> out-of-time murni
    wA = recency_weight(foldA_tr.waktu, pd.Timestamp("2025-07-01"), HL)
    _, pA = run(FEAT_FULL, foldA_tr, foldA_va, w=wA)
    wB = recency_weight(foldB_tr.waktu, ref, HL)
    _, pB = run(FEAT_FULL, foldB_tr, foldB_va, w=wB)
    platt = platt_fit(pA, yA)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pA, yA)
    cal_rows = []
    # Isotonic menghasilkan fungsi tangga (banyak prob kembar) yang merusak urutan skor & PR-AUC.
    # Tie-break: campur 1e-4 x skor mentah -> urutan identik dengan skor mentah, kalibrasi praktis sama.
    def iso_apply(p):
        return (1 - 1e-4) * iso.predict(p) + 1e-4 * p
    cal_preds = {"Mentah": pB, "Platt": platt_apply(platt, pB), "Isotonic (+tie-break)": iso_apply(pB)}
    for name, p in cal_preds.items():
        cal_rows.append(dict(kalibrasi=name, brier=round(brier_score_loss(yB, p), 6),
                             rata2_prob=round(float(p.mean()), 5), rate_aktual=round(float(yB.mean()), 5),
                             pr_auc=round(average_precision_score(yB, p), 4),
                             biaya_aturan_juta=round(total_cost(yB, nB, cost_rule(p, nB)) / 1e6, 2),
                             n_tandai=int(cost_rule(p, nB).sum())))
    pd.DataFrame(cal_rows).to_csv(os.path.join(res_dir, "f_kalibrasi.csv"), index=False)
    R["kalibrasi"] = cal_rows
    best_cal = min(cal_rows, key=lambda r: r["biaya_aturan_juta"])["kalibrasi"]
    R["kalibrasi_terpilih"] = best_cal

    # Reliability table (desil) untuk laporan
    pc = cal_preds[best_cal]
    bins = pd.qcut(pc, 10, duplicates="drop")
    rel = pd.DataFrame({"p": pc, "y": yB}).groupby(bins, observed=True).agg(prob_rata2=("p", "mean"), fraud_aktual=("y", "mean"), n=("y", "size"))
    rel.to_csv(os.path.join(res_dir, "f_reliability_foldB.csv"))

    # Perbandingan kebijakan keputusan (Fold B, prob terkalibrasi)
    thr, c_thr = best_threshold(pc, yB, nB)
    dec = [
        dict(kebijakan="Tidak tandai sama sekali", n_tandai=0, biaya_juta=round(total_cost(yB, nB, np.zeros_like(yB)) / 1e6, 2)),
        dict(kebijakan="Tandai semua transaksi", n_tandai=len(yB), biaya_juta=round(total_cost(yB, nB, np.ones_like(yB)) / 1e6, 2)),
        dict(kebijakan="Ambang prob 0.5", n_tandai=int((pc >= 0.5).sum()), biaya_juta=round(total_cost(yB, nB, pc >= 0.5) / 1e6, 2)),
        dict(kebijakan=f"Ambang prob global tuned ({thr:.4f}, optimis)", n_tandai=int((pc >= thr).sum()), biaya_juta=round(c_thr / 1e6, 2)),
        dict(kebijakan="Aturan biaya: p x nominal >= 25.000", n_tandai=int(cost_rule(pc, nB).sum()),
             biaya_juta=round(total_cost(yB, nB, cost_rule(pc, nB)) / 1e6, 2)),
    ]
    pd.DataFrame(dec).to_csv(os.path.join(res_dir, "f_kebijakan_keputusan.csv"), index=False)
    R["keputusan_foldB"] = dec
    # Kebijakan yang sama di Fold A (prob terkalibrasi in-sample hanya sebagai referensi)
    pA_c = platt_apply(platt, pA) if best_cal == "Platt" else (iso_apply(pA) if best_cal.startswith("Isotonic") else pA)
    R["keputusan_foldA"] = dict(
        tidak_tandai_juta=round(total_cost(yA, nA, np.zeros_like(yA)) / 1e6, 2),
        tandai_semua_juta=round(total_cost(yA, nA, np.ones_like(yA)) / 1e6, 2),
        aturan_biaya_juta=round(total_cost(yA, nA, cost_rule(pA_c, nA)) / 1e6, 2),
        n_tandai=int(cost_rule(pA_c, nA).sum()))

    # ======================================================================
    # (g) ANALISIS GALAT
    # ======================================================================
    log("(g) Analisis galat per kanal, kategori, dan pola fraud")
    va = foldB_va.copy(); va["prob"] = pc; va["tandai"] = cost_rule(pc, nB)
    vaA = foldA_va.copy(); vaA["prob"] = pA_c; vaA["tandai"] = cost_rule(pA_c, nA)
    ev = pd.concat([vaA, va])  # Jul + 1-10 Agu, semua out-of-time
    ev["pola"] = np.select(
        [(ev.kanal == "qris") & (ev.umur_merchant_hari < 60),
         ev.kartu_tidak_hadir == 1,
         ev.perangkat_baru == 1,
         ev.jarak_lokasi_km > 100,
         ev.malam == 1],
        ["QRIS merchant muda (<60 hari)", "Kartu tidak hadir (CNP)", "Perangkat baru",
         "Lokasi jauh (>100 km)", "Transaksi malam (00-05)"], "Lainnya")

    def breakdown(col):
        def f(d):
            fr_ = d.is_fraud == 1
            return pd.Series(dict(
                n=len(d), fraud=int(fr_.sum()), tandai=int(d.tandai.sum()),
                recall=round(d.loc[fr_, "tandai"].mean(), 3) if fr_.any() else np.nan,
                precision=round(d.loc[d.tandai == 1, "is_fraud"].mean(), 3) if d.tandai.any() else np.nan,
                fraud_lolos_juta=round(d.loc[fr_ & (d.tandai == 0), "nominal"].sum() / 1e6, 2),
                biaya_review_juta=round(REVIEW_COST * d.tandai.sum() / 1e6, 2),
                pr_auc=round(average_precision_score(d.is_fraud, d.prob), 3) if 0 < fr_.sum() < len(d) else np.nan,
            ))
        return ev.groupby(col).apply(f).sort_values("fraud", ascending=False)

    for col, fn in [("kanal", "g_galat_per_kanal.csv"), ("kategori_merchant", "g_galat_per_kategori.csv"), ("pola", "g_galat_per_pola.csv")]:
        b = breakdown(col); b.to_csv(os.path.join(res_dir, fn))
        R[f"galat_{col}"] = b.reset_index().to_dict(orient="records")

    # ======================================================================
    # MODEL FINAL & PREDIKSI TEST
    # ======================================================================
    # Strategi data label-tertunda dipilih dari simulasi (PR-AUC tertinggi di Fold A)
    best_lag = max([r for r in lag_rows if "tertunda" in r["model"]], key=lambda r: r["pr_auc"])
    R["strategi_lag_terpilih"] = best_lag["model"]
    log(f"Melatih model final (strategi label lag: {best_lag['model']})")
    immature_start = pd.Timestamp("2025-08-11")   # kematangan label < 70% sejak minggu ini
    final_end = pd.Timestamp("2025-09-01")
    base = L[(L.waktu >= "2025-02-01") & (L.waktu < final_end)]
    late = (base.waktu >= immature_start).to_numpy()
    if "simpan positif" in best_lag["model"]:
        final_tr = base[~late | (base.is_fraud.to_numpy() == 1)]
    elif "3 minggu" in best_lag["model"]:
        final_tr = base[~late]
    elif "1 minggu" in best_lag["model"]:
        final_tr = base[base.waktu < pd.Timestamp("2025-08-25")]
    else:
        final_tr = base
    R["final_train_end"] = str(final_tr.waktu.max().date())
    wF = recency_weight(final_tr.waktu, final_tr.waktu.max().normalize() + pd.Timedelta(days=1), HL)
    final_model = fit_lgb(to_lgb_frame(final_tr, FEAT_FULL), final_tr.is_fraud.to_numpy(), wF)
    p_test_raw = final_model.predict_proba(to_lgb_frame(T, FEAT_FULL))[:, 1]
    if best_cal == "Platt":
        p_test = platt_apply(platt, p_test_raw)
    elif best_cal.startswith("Isotonic"):
        p_test = iso_apply(p_test_raw)
    else:
        p_test = p_test_raw
    p_test = np.clip(p_test, 0.0, 1.0)
    tandai = cost_rule(p_test, T.nominal.to_numpy())
    R["final"] = dict(n_train=len(final_tr), fraud_train=int(final_tr.is_fraud.sum()),
                      prob_rata2_test=round(float(p_test.mean()), 5), n_tandai_test=int(tandai.sum()),
                      pct_tandai_test=round(100 * tandai.mean(), 2),
                      biaya_review_test_juta=round(REVIEW_COST * tandai.sum() / 1e6, 2))

    imp = pd.DataFrame({"fitur": FEAT_FULL, "gain": final_model.booster_.feature_importance("gain")})
    imp["gain_pct"] = (100 * imp.gain / imp.gain.sum()).round(2)
    imp = imp.sort_values("gain", ascending=False)
    imp.to_csv(os.path.join(res_dir, "importance_model_final.csv"), index=False)
    R["top_fitur"] = imp.head(15)[["fitur", "gain_pct"]].to_dict(orient="records")

    sub = pd.DataFrame({"id_transaksi": T.id_transaksi.to_numpy(), "prob": p_test, "tandai": tandai})
    sub = sub.set_index("id_transaksi").loc[te.id_transaksi].reset_index()   # urutan sama dengan file test
    assert len(sub) == len(te) and sub.prob.between(0, 1).all() and sub.tandai.isin([0, 1]).all()
    out_csv = os.path.join(output_dir, "prediksi_klasifikasi.csv")
    sub.to_csv(out_csv, index=False, float_format="%.8g")
    log(f"Tersimpan: {out_csv} ({len(sub)} baris, {int(tandai.sum())} ditandai)")

    # ======================================================================
    # GRAFIK
    # ======================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve

        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        mg = mingguan.reset_index(); ax[0].bar(range(len(mg)), mg.rate_pct, color="#4C72B0")
        ax[0].axhline(100 * base_rate, color="r", ls="--", label="rata2 Feb-Jul")
        ax[0].set_xticks(range(len(mg))); ax[0].set_xticklabels([str(p.start_time.date())[5:] for p in mg.waktu], rotation=90, fontsize=7)
        ax[0].set_title("(a) Fraud rate mingguan: label lag"); ax[0].set_ylabel("%"); ax[0].legend()
        shift_kanal.plot(ax=ax[1], marker="o"); ax[1].set_title("(e) Fraud rate per kanal per bulan"); ax[1].set_ylabel("%")
        for name, p in [("Mentah", cal_preds["Mentah"]), (best_cal, pc)]:
            b = pd.DataFrame({"p": p, "y": yB}).groupby(pd.qcut(p, 10, duplicates="drop"), observed=True).mean()
            ax[2].plot(b.p, b.y, marker="o", label=name)
        mx = max(0.05, float(pc.max()))
        ax[2].plot([0, mx], [0, mx], "k--", lw=1); ax[2].set_xscale("symlog", linthresh=1e-3); ax[2].set_yscale("symlog", linthresh=1e-3)
        ax[2].set_title("(f) Reliability (Fold B)"); ax[2].set_xlabel("prob prediksi"); ax[2].set_ylabel("fraud aktual"); ax[2].legend()
        plt.tight_layout(); plt.savefig(os.path.join(res_dir, "fig_task2_audit_shift_kalibrasi.png"), dpi=130); plt.close()

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for name in ["1. Tanpa penanganan", "2. Class weighting (spw=neg/pos)", "3. Random undersampling 1:20", "4. Random oversampling x20"]:
            pr, rc, _ = precision_recall_curve(yB, imb_preds[("B", name)])
            ax[0].plot(rc, pr, label=name, lw=1)
        ax[0].set_title("(d) Kurva PR Fold B"); ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision"); ax[0].legend(fontsize=7)
        top = imp.head(15)[::-1]; ax[1].barh(top.fitur, top.gain_pct, color="#55A868"); ax[1].set_title("Importance (gain %) model final")
        plt.tight_layout(); plt.savefig(os.path.join(res_dir, "fig_task2_pr_importance.png"), dpi=130); plt.close()
    except Exception as e:  # grafik opsional
        log(f"Grafik dilewati: {e}")

    R["library"] = dict(lightgbm=lgb.__version__, pandas=pd.__version__, numpy=np.__version__,
                        sklearn=__import__("sklearn").__version__, seed=SEED)
    R["durasi_detik"] = round(time.time() - t0, 1)
    with open(os.path.join(res_dir, "ringkasan_task2.json"), "w") as f:
        json.dump(R, f, indent=2, default=str, ensure_ascii=False)
    log(f"Selesai dalam {R['durasi_detik']} detik. Ringkasan: {res_dir}/ringkasan_task2.json")
    return R


if __name__ == "__main__":
    import sys
    run_task2(sys.argv[1] if len(sys.argv) > 1 else None)
