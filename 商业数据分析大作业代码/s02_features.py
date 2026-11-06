# -*- coding: utf-8 -*-
"""s02_features.py —— 声学特征工程 + 质检指标 + 成本/价值派生字段。

产物：
  data/features_asset.csv   535 行 × N 维声学特征 + 质检 + 成本价值字段
  data/features_block.csv   535×150 = 80250 行，100 ms 时间块明细（时序/热力图用）
  data/redo.sqlite          asset_ledger / features_asset / features_block 三表（复用与验证）
  data/audit_features.json  特征一致性、质检结论、成本口径对账
"""
from __future__ import annotations
import io, json, sqlite3, sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.fft import dct

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cfg

EPS = 1e-12
WIN = int(cfg.SR_EXPECTED * cfg.FRAME_MS / 1000)      # 100
HOP = int(cfg.SR_EXPECTED * cfg.HOP_MS / 1000)        # 40
NFREQ = cfg.N_FFT // 2 + 1                            # 65
FREQS = np.fft.rfftfreq(cfg.N_FFT, 1.0 / cfg.SR_EXPECTED)
BLOCK = int(cfg.SR_EXPECTED * cfg.BLOCK_MS / 1000)    # 400
NBLOCKS = 150


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_mels=cfg.N_MELS, n_fft=cfg.N_FFT, sr=cfg.SR_EXPECTED,
                   fmin=cfg.MEL_FMIN, fmax=cfg.MEL_FMAX):
    lo, hi = hz_to_mel(fmin), hz_to_mel(fmax)
    pts = mel_to_hz(np.linspace(lo, hi, n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            if c > l:
                fb[m - 1, k] = (k - l) / (c - l)
        for k in range(c, r):
            if r > c:
                fb[m - 1, k] = (r - k) / (r - c)
    return fb


MEL_FB = mel_filterbank()
DCT_M = dct(np.eye(cfg.N_MFCC, cfg.N_MELS), type=2, axis=0, norm="ortho")  # (13,26)


def frame_matrix(x: np.ndarray) -> np.ndarray:
    n = 1 + (len(x) - WIN) // HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx]


def autocorr_peak(env: np.ndarray, lag_lo: int, lag_hi: int) -> tuple[float, float]:
    """在 [lag_lo, lag_hi] 内找归一化自相关峰；返回 (bpm, 峰值)。env 采样率 100 Hz。"""
    e = env - env.mean()
    denom = float((e * e).sum()) + EPS
    lag_hi = min(lag_hi, len(e) - 2)
    if lag_hi <= lag_lo:
        return np.nan, np.nan
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    seg = ac[lag_lo:lag_hi + 1] / denom
    if len(seg) == 0:
        return np.nan, np.nan
    k = int(np.argmax(seg))
    lag = lag_lo + k
    bpm = 60.0 * 100.0 / lag if lag > 0 else np.nan
    return float(bpm), float(seg[k])


def asset_features(x: np.ndarray) -> dict:
    fr = frame_matrix(x)                                    # (1498,100)
    win = np.hanning(WIN)[None, :]
    spec = np.abs(np.fft.rfft(fr * win, n=cfg.N_FFT, axis=1)) ** 2   # 功率谱 (1498,65)
    rms = np.sqrt((fr ** 2).mean(axis=1) + EPS)
    rms_db = 20 * np.log10(rms + EPS)
    zcr = (np.abs(np.diff(np.sign(fr), axis=1)) > 0).mean(axis=1)

    p = spec.sum(axis=1) + EPS
    centroid = (spec * FREQS[None, :]).sum(axis=1) / p
    var = (spec * (FREQS[None, :] - centroid[:, None]) ** 2).sum(axis=1) / p
    bandwidth = np.sqrt(var + EPS)
    cum = np.cumsum(spec, axis=1) / p[:, None]
    rolloff85 = np.array([FREQS[min(int(np.searchsorted(cum[i], 0.85)), NFREQ - 1)] for i in range(len(cum))])
    rolloff95 = np.array([FREQS[min(int(np.searchsorted(cum[i], 0.95)), NFREQ - 1)] for i in range(len(cum))])
    geo = np.exp(np.log(spec + EPS).mean(axis=1))
    flatness = geo / (spec.mean(axis=1) + EPS)
    pn = spec / p[:, None]
    entropy = -(pn * np.log(pn + EPS)).sum(axis=1) / np.log(NFREQ)
    slope = np.polyfit(FREQS, spec.mean(axis=0), 1)[0]
    flux = float(np.sqrt((np.diff(np.sqrt(spec), axis=0) ** 2).sum(axis=1).mean() + EPS))

    mel = np.log(np.maximum(spec @ MEL_FB.T, EPS))
    mfcc = mel @ DCT_M.T                                     # (1498,13)

    bands = {}
    for (lo, hi), nm in zip(cfg.BANDS_HZ, cfg.BAND_NAMES):
        m = (FREQS >= lo) & (FREQS < hi)
        bands[nm] = float(spec[:, m].sum(axis=1).mean() / (p.mean() + EPS))

    sil_rel = float((rms_db < np.percentile(rms_db, 95) - cfg.SILENCE_REL_DB).mean())
    hr, hr_pk = autocorr_peak(rms, int(cfg.HR_LAG_SEC[0] * 100), int(cfg.HR_LAG_SEC[1] * 100))
    rr, rr_pk = autocorr_peak(rms, int(cfg.RR_LAG_SEC[0] * 100), int(cfg.RR_LAG_SEC[1] * 100))
    env = rms / (rms.max() + EPS)
    silent = rms_db < cfg.SILENCE_DB

    f = {
        "rms": float(rms.mean()), "peak": float(np.abs(x).max()),
        "crest_factor": float(np.abs(x).max() / (rms.mean() + EPS)),
        "mean_abs": float(np.abs(x).mean()),
        "waveform_factor": float(rms.mean() / (np.abs(x).mean() + EPS)),
        "kurtosis": float(pd.Series(x).kurt()), "skewness": float(pd.Series(x).skew()),
        "zcr_mean": float(zcr.mean()), "zcr_std": float(zcr.std()),
        "dynamic_range_db": float(rms_db.max() - rms_db.min()),
        "snr_db": float(np.percentile(rms_db, 95) - np.percentile(rms_db, 10)),
        "centroid": float(centroid.mean()), "centroid_std": float(centroid.std()),
        "bandwidth": float(bandwidth.mean()), "rolloff85": float(rolloff85.mean()),
        "rolloff95": float(rolloff95.mean()), "flatness": float(flatness.mean()),
        "entropy": float(entropy.mean()), "entropy_std": float(entropy.std()),
        "slope": float(slope), "flux": flux, "dominant_freq": float(FREQS[int(np.argmax(spec.mean(axis=0)))]),
        "env_crest": float(env.max() / (env.mean() + EPS)), "env_cv": float(rms.std() / (rms.mean() + EPS)),
        "heart_rate_bpm": hr, "hr_autocorr_peak": hr_pk,
        "resp_rate_bpm": rr, "rr_autocorr_peak": rr_pk,
        "silence_ratio": sil_rel,
        "silence_ratio_abs": float(silent.mean()),
        "mfcc_std_mean": float(mfcc.std(axis=0).mean()),
    }
    f.update(bands)
    f["band_low_high_ratio"] = f["b1_20_150"] / (f["b5_1200_2000"] + EPS)
    for i in range(cfg.N_MFCC):
        f[f"mfcc{i+1}_mean"] = float(mfcc[:, i].mean())
        f[f"mfcc{i+1}_std"] = float(mfcc[:, i].std())
    return f


def qc_features(x: np.ndarray) -> dict:
    fr = frame_matrix(x)
    rms = np.sqrt((fr ** 2).mean(axis=1) + EPS)
    rms_db = 20 * np.log10(rms + EPS)
    return {
        "dc_offset": float(x.mean()),
        "clip_ratio": float((np.abs(x) > cfg.CLIP_LEVEL).mean()),
        "zero_ratio": float((x == 0).mean()),
        "frame_silence_ratio": float((rms_db < cfg.SILENCE_DB).mean()),
        "peak_db": float(20 * np.log10(np.abs(x).max() + EPS)),
    }


def block_table(asset_id: str, x: np.ndarray) -> pd.DataFrame:
    b = x[: NBLOCKS * BLOCK].reshape(NBLOCKS, BLOCK)
    rms = np.sqrt((b ** 2).mean(axis=1) + EPS)
    rms_db = 20 * np.log10(rms + EPS)
    zcr = (np.abs(np.diff(np.sign(b), axis=1)) > 0).mean(axis=1)
    spec = np.abs(np.fft.rfft(b * np.hanning(BLOCK)[None, :], n=512, axis=1)) ** 2
    fr = np.fft.rfftfreq(512, 1.0 / cfg.SR_EXPECTED)
    p = spec.sum(axis=1) + EPS
    cent = (spec * fr[None, :]).sum(axis=1) / p
    pn = spec / p[:, None]
    ent = -(pn * np.log(pn + EPS)).sum(axis=1) / np.log(len(fr))
    return pd.DataFrame({
        "asset_id": asset_id, "block_idx": np.arange(NBLOCKS),
        "t_start_sec": np.round(np.arange(NBLOCKS) * cfg.BLOCK_MS / 1000, 3),
        "block_rms": rms, "block_rms_db": rms_db, "block_zcr": zcr,
        "block_centroid": cent, "block_entropy": ent,
        "block_energy_ratio": rms / (rms.max() + EPS),
        "is_silent": (rms_db < cfg.SILENCE_DB).astype(int),
    })


def norm01(s: pd.Series) -> pd.Series:
    lo, hi = float(s.min()), float(s.max())
    return (s - lo) / (hi - lo) if hi > lo else s * 0.0


def main() -> None:
    ledger = pd.read_csv(cfg.DATA / "asset_ledger.csv")
    blobs: dict[str, bytes] = {}
    for pkg, i in (("HS", 0), ("LS", 1), ("Mix", 2)):
        with zipfile.ZipFile(cfg.RAW / cfg.ZIP_FILES[i]) as z:
            for info in z.infolist():
                n = Path(info.filename).name
                if info.is_dir() or "__MACOSX" in info.filename or not n.lower().endswith(".wav"):
                    continue
                blobs[f"{cfg.ZIP_FILES[i]}/{n}"] = z.read(info)

    feat_rows, block_rows, fmt_issues = [], [], []
    for k, r in enumerate(ledger.itertuples(index=False), start=1):
        blob = blobs[r.path_key]
        sr, x = wavfile.read(io.BytesIO(blob))
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != cfg.SR_EXPECTED or x.dtype != np.int16 or len(x) != 60000:
            fmt_issues.append({"asset_id": r.asset_id, "sample_rate": int(sr),
                               "dtype": str(x.dtype), "n_samples": int(len(x))})
        xf = x.astype(np.float64) / 32768.0
        d = {"asset_id": r.asset_id, "path_key": r.path_key, "sha256": r.sha256}
        d.update(qc_features(xf))
        d.update(asset_features(xf))
        feat_rows.append(d)
        block_rows.append(block_table(r.asset_id, xf))
        if k % 100 == 0:
            print(f"  特征提取 {k}/{len(ledger)}")

    F = pd.DataFrame(feat_rows)
    B = pd.concat(block_rows, ignore_index=True)

    # ---------------- 质检判定 ----------------
    F["qc_dc_fail"] = (F["dc_offset"].abs() > cfg.DC_MAX).astype(int)
    F["qc_clip_fail"] = (F["clip_ratio"] > cfg.CLIP_RATIO_MAX).astype(int)
    # 静音超标用相对口径 silence_ratio；绝对口径 frame_silence_ratio 只作“低电平”诊断，不参与判定
    F["qc_silence_fail"] = (F["silence_ratio"] > cfg.SILENCE_RATIO_MAX).astype(int)
    F["qc_low_level_flag"] = (F["frame_silence_ratio"] > cfg.SILENCE_RATIO_MAX).astype(int)
    F["qc_pass"] = ((F["qc_dc_fail"] + F["qc_clip_fail"] + F["qc_silence_fail"]) == 0).astype(int)

    # ---------------- 质量分 / 信息量分 / 稀缺度 / 价值指数 ----------------
    snr_n = norm01(F["snr_db"].clip(cfg.SNR_FLOOR_DB, cfg.SNR_CEIL_DB))
    F["quality_score"] = (60 * snr_n + 20 * (1 - F["silence_ratio"])
                          + 20 * (1 - (F["clip_ratio"] / cfg.CLIP_RATIO_MAX).clip(0, 1))).round(2)
    # 信息量 = 谱熵 50% + MFCC 时长波动 30% + 等效带宽 20%，权重和为 1，直接得 0~100 分
    F["info_score"] = (100 * (0.50 * norm01(F["entropy"]) + 0.30 * norm01(F["mfcc_std_mean"])
                              + 0.20 * norm01(F["bandwidth"]))).round(2)

    L = ledger.copy()
    L["category"] = np.where(L["cohort"] == "lung_pure", L["lung_type_zh"],
                             np.where(L["cohort"] == "heart_pure", L["heart_type_zh"],
                                      L["heart_type_zh"] + "×" + L["lung_type_zh"]))
    cnt = L[L["in_model"] == 1].groupby("category")["asset_id"].transform("size")
    L["category_supply"] = cnt
    nmax = float(L.loc[L["in_model"] == 1, "category_supply"].max())
    L["scarcity_score"] = np.where(L["in_model"] == 1,
                                   (100 * (1 - (L["category_supply"] - 1) / max(nmax - 1, 1))).round(2), 0.0)
    M = L[["asset_id", "category", "category_supply", "scarcity_score"]].merge(F, on="asset_id", how="right")
    assert len(M) == len(F)
    F = M
    # 复用度：同一指纹在几个包里被重复使用（1 次=0 分，>=4 次=100 分，封顶避免长尾主导）
    F = F.merge(ledger[["asset_id", "group_size"]], on="asset_id", how="left")
    F["reuse_score"] = ((F["group_size"].clip(upper=4) - 1) / 3 * 100).round(2)
    F["value_index"] = (cfg.W_QUALITY * F["quality_score"] + cfg.W_INFO * F["info_score"]
                        + cfg.W_SCARCITY * F["scarcity_score"] + cfg.W_REUSE * F["reuse_score"]).round(2)

    # ---------------- 成本模型 ----------------
    F = F.merge(ledger[["asset_id", "n_bytes", "cohort", "modality", "heart_abnormal",
                        "lung_abnormal", "batch_no", "batch_label", "is_primary_copy", "in_model"]],
                on="asset_id", how="left")
    storage = F["n_bytes"] / 1024 ** 3 * cfg.STORAGE_YUAN_PER_GB_MONTH * cfg.STORAGE_MONTHS
    label_min = (cfg.LABEL_MIN_BASE
                 + cfg.LABEL_MIN_ABNORMAL_EXTRA * ((F["heart_abnormal"] == 1).astype(int)
                                                   | (F["lung_abnormal"] == 1).astype(int))
                 + cfg.LABEL_MIN_MIXED_EXTRA * (F["modality"] == "mixed").astype(int))
    F["label_minutes"] = label_min
    F["storage_cost"] = storage.round(4)
    F["label_cost"] = (label_min * cfg.LABELER_YUAN_PER_MIN).round(4)
    F["review_cost"] = (label_min * cfg.PHYSICIAN_YUAN_PER_MIN * cfg.PHYSICIAN_REVIEW_RATE).round(4)
    F["qc_cost"] = round(cfg.QC_AUTO_MIN * cfg.LABELER_YUAN_PER_MIN
                         + cfg.QC_SAMPLE_RATE * cfg.QC_MANUAL_MIN * cfg.PHYSICIAN_YUAN_PER_MIN, 4)
    F["total_cost"] = (F["storage_cost"] + F["label_cost"] + F["review_cost"] + F["qc_cost"]).round(4)
    F["value_cost_ratio"] = (F["value_index"] / F["total_cost"]).round(3)
    F["cost_per_minute_audio"] = (F["total_cost"] / (F["duration_sec"] / 60)).round(4) if "duration_sec" in F else np.nan

    F = F.merge(ledger[["asset_id", "duration_sec", "sample_rate"]], on="asset_id", how="left")
    F["cost_per_minute_audio"] = (F["total_cost"] / (F["duration_sec"] / 60)).round(4)

    F.round(6).to_csv(cfg.DATA / "features_asset.csv", index=False, encoding="utf-8-sig")
    B.round(6).to_csv(cfg.DATA / "features_block.csv", index=False, encoding="utf-8-sig")

    # ---------------- SQLite 导出（要求 2.1.5）----------------
    db = cfg.DATA / "redo.sqlite"
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    ledger.to_sql("asset_ledger", con, index=False)
    F.to_sql("features_asset", con, index=False)
    B.to_sql("features_block", con, index=False)
    con.commit()

    # ---------------- 一致性校验：重复副本特征必须完全相同 ----------------
    key_cols = ["rms", "peak", "centroid", "bandwidth", "entropy", "zcr_mean",
                "silence_ratio", "silence_ratio_abs", "mfcc1_mean", "mfcc7_std",
                "heart_rate_bpm", "resp_rate_bpm", "snr_db", "dynamic_range_db"]
    g = F.groupby("sha256")[key_cols].nunique().max().max()
    dup_consistent = bool(g == 1)

    audit = {
        "feature_rows": int(len(F)), "feature_cols": int(F.shape[1]),
        "block_rows": int(len(B)), "block_cols": int(B.shape[1]),
        "acoustic_and_qc_dims": int(len(feat_rows[0]) - 3),
        "derived_business_fields": ["category", "category_supply", "scarcity_score", "reuse_score",
                                    "quality_score", "info_score", "value_index", "label_minutes",
                                    "storage_cost", "label_cost", "review_cost", "qc_cost",
                                    "total_cost", "value_cost_ratio", "cost_per_minute_audio"],
        "duplicate_feature_consistency": dup_consistent,
        "format_issues": fmt_issues,
        "qc": {"pass": int(F["qc_pass"].sum()), "dc_fail": int(F["qc_dc_fail"].sum()),
               "clip_fail": int(F["qc_clip_fail"].sum()),
               "silence_fail": int(F["qc_silence_fail"].sum()),
               "low_level_flag": int(F["qc_low_level_flag"].sum()),
               "silence_ratio_mean": round(float(F["silence_ratio"].mean()), 4),
               "frame_silence_ratio_mean": round(float(F["frame_silence_ratio"].mean()), 4),
               "pass_rate": round(float(F["qc_pass"].mean()), 4)},
        "qc_in_model": {"pass": int(F.loc[F.in_model == 1, "qc_pass"].sum()),
                        "total": int(F.loc[F.in_model == 1].shape[0])},
        "sqlite_tables": ["asset_ledger", "features_asset", "features_block"],
        "cost_total_yuan": round(float(F.loc[F.is_primary_copy == 1, "total_cost"].sum()), 2),
        "cost_wasted_on_duplicates_yuan": round(float(F.loc[F.is_primary_copy == 0, "total_cost"].sum()), 2),
        "value_index_mean": round(float(F["value_index"].mean()), 2),
        "hr_median": float(F["heart_rate_bpm"].median()),
        "hr_range": [round(float(F["heart_rate_bpm"].min()), 1), round(float(F["heart_rate_bpm"].max()), 1)],
        "rr_median": float(F["resp_rate_bpm"].median()),
        "rr_range": [round(float(F["resp_rate_bpm"].min()), 1), round(float(F["resp_rate_bpm"].max()), 1)],
    }
    con.execute("SELECT COUNT(*) FROM features_block")
    con.close()
    (cfg.DATA / "audit_features.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
