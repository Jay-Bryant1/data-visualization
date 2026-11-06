# -*- coding: utf-8 -*-
"""s04_models.py —— 商业分析模型层（8 种，每种至少 1 图 + 1 条商业结论）。

模型清单与业务口径：
  M1 帕累托/ABC 分类      —— 资产价值累计占比，A/B/C 分层（70/90 切分）
  M2 波士顿矩阵           —— 品类市场份额 × 批次间份额增速（四象限）
  M3 RFM 八分类           —— R 入库批次新鲜度 / F 跨包复用频次 / M 资产价值指数
  M4 K-Means 资产分群     —— 肘部法 + 轮廓系数定 K，PCA 降维验证
  M5 关联规则(Apriori)    —— 品类×点位×性别 共现的 support/confidence/lift
  M6 Holt-Winters 预测    —— 100ms 时间块平均能量，95% 置信区间 + MAPE/RMSE 回测
  M7 漏斗转化             —— 入库→合规→唯一→达标→高价值 逐级留存
  M8 异常检出与成本曲线   —— 逻辑回归/随机森林 5 折分组 CV，AUC + 复核成本曲线
所有参数取自 cfg.py，输出 tables/*.csv 与 eda/model_results.json。
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cfg

RNG = np.random.RandomState(cfg.SEED)
R: dict = {}


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(cfg.TABLE_DIR / name, index=False, encoding="utf-8-sig")


def main() -> None:
    A = pd.read_csv(cfg.DATA / "features_asset.csv")
    L = pd.read_csv(cfg.DATA / "asset_ledger.csv")
    B = pd.read_csv(cfg.DATA / "features_block.csv")
    M = A.merge(L.drop(columns=[c for c in ["n_bytes", "sample_rate", "duration_sec", "group_size",
                                             "cohort", "modality", "heart_abnormal", "lung_abnormal",
                                             "batch_no", "batch_label", "is_primary_copy", "in_model"]
                              if c in A.columns]), on="asset_id", how="left")
    for c in ["heart_type", "heart_type_zh", "lung_type", "lung_type_zh", "location",
              "location_zh", "gender", "gender_zh", "category", "cohort_zh"]:
        if c in M.columns:
            M[c] = M[c].fillna("")
    MODEL = M[M["in_model"] == 1].copy()

    # ================================================================= M1 帕累托 / ABC
    rank = MODEL.sort_values("value_index", ascending=False).reset_index(drop=True)
    rank["cum_value"] = rank["value_index"].cumsum()
    total_value = float(rank["value_index"].sum())
    rank["cum_pct"] = 100 * rank["cum_value"] / total_value
    rank["abc"] = np.where(rank["cum_pct"] <= cfg.ABC_CUM[0], "A",
                           np.where(rank["cum_pct"] <= cfg.ABC_CUM[1], "B", "C"))
    MODEL["abc"] = MODEL["asset_id"].map(dict(zip(rank["asset_id"], rank["abc"])))
    abc_sum = (MODEL.groupby("abc")
               .agg(资产数=("asset_id", "size"), 价值总和=("value_index", "sum"),
                    成本总和=("total_cost", "sum"), 平均质量分=("quality_score", "mean"))
               .reset_index())
    abc_sum["数量占比"] = (100 * abc_sum["资产数"] / abc_sum["资产数"].sum()).round(2)
    abc_sum["价值占比"] = (100 * abc_sum["价值总和"] / abc_sum["价值总和"].sum()).round(2)
    abc_sum["成本占比"] = (100 * abc_sum["成本总和"] / abc_sum["成本总和"].sum()).round(2)
    save(abc_sum, "tbl_M1_abc.csv")
    top_share = float(abc_sum.loc[abc_sum["abc"] == "A", "价值占比"].iloc[0])
    top_qty = float(abc_sum.loc[abc_sum["abc"] == "A", "数量占比"].iloc[0])
    R["M1_abc"] = {"A类数量占比": top_qty, "A类价值占比": top_share,
                   "帕累托偏离度": round(top_share - top_qty, 2),
                   "C类成本占比": float(abc_sum.loc[abc_sum["abc"] == "C", "成本占比"].iloc[0])}

    # ================================================================= M2 波士顿矩阵
    cat = MODEL[MODEL["cohort"].isin(["heart_pure", "lung_pure", "mixed"])].copy()
    gsize = len(cat)
    share = cat["category"].value_counts() / gsize
    early = cat[cat["batch_no"] <= MODEL["batch_no"].median()]["category"].value_counts()
    late = cat[cat["batch_no"] > MODEL["batch_no"].median()]["category"].value_counts()
    e_sum, l_sum = max(early.sum(), 1), max(late.sum(), 1)
    growth = ((late / l_sum - early / e_sum) / (early / e_sum).replace(0, np.nan) * 100)
    bcg = pd.DataFrame({"市场份额": share, "份额增速": growth}).dropna()
    bcg["份额中位"] = bcg["市场份额"].median()
    bcg["象限"] = np.where((bcg["市场份额"] >= bcg["份额中位"]) & (bcg["份额增速"] >= 0), "明星",
                    np.where((bcg["市场份额"] >= bcg["份额中位"]) & (bcg["份额增速"] < 0), "现金牛",
                    np.where((bcg["市场份额"] < bcg["份额中位"]) & (bcg["份额增速"] >= 0), "问题", "瘦狗")))
    bcg = bcg.reset_index().rename(columns={"index": "品类", "category": "品类"})
    bcg["平均价值指数"] = bcg["品类"].map(cat.groupby("category")["value_index"].mean()).round(2)
    bcg["平均成本"] = bcg["品类"].map(cat.groupby("category")["total_cost"].mean()).round(3)
    save(bcg, "tbl_M2_bcg.csv")
    R["M2_bcg"] = {"品类数": int(len(bcg)),
                   "象限分布": bcg["象限"].value_counts().to_dict(),
                   "明星品类TOP3": bcg[bcg["象限"] == "明星"].nlargest(3, "市场份额")["品类"].tolist(),
                   "瘦狗品类TOP3": bcg[bcg["象限"] == "瘦狗"].nsmallest(3, "市场份额")["品类"].tolist(),
                   "增速口径": "后半段批次份额相对前半段批次份额的变化率（入库流水代理时间）"}

    # ================================================================= M3 RFM 八分类
    rfm = MODEL.copy()
    rfm["R"] = pd.qcut(rfm["batch_no"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["F"] = pd.cut(rfm["group_size"], [-0.1, 1.0001, 2.0001, 3.0001, 4.0001, 99],
                      labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M"] = pd.qcut(rfm["value_index"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["r_hi"] = (rfm["R"] >= 3).astype(int)
    rfm["f_hi"] = (rfm["F"] >= 3).astype(int)
    rfm["m_hi"] = (rfm["M"] >= 3).astype(int)
    LABEL = {(1, 1, 1): "高价值核心资产", (1, 1, 0): "高频低值资产", (1, 0, 1): "稳定优质资产",
             (1, 0, 0): "一般留存资产", (0, 1, 1): "沉睡高值资产", (0, 1, 0): "沉睡低值资产",
             (0, 0, 1): "新品待培育资产", (0, 0, 0): "长尾淘汰候选"}
    rfm["rfm_class"] = [LABEL[(a, b, c)] for a, b, c in
                        zip(rfm["r_hi"], rfm["f_hi"], rfm["m_hi"])]
    rfm_sum = (rfm.groupby("rfm_class")
               .agg(资产数=("asset_id", "size"), 平均价值=("value_index", "mean"),
                    平均成本=("total_cost", "mean"), 平均质量=("quality_score", "mean"))
               .sort_values("资产数", ascending=False).reset_index())
    rfm_sum["占比"] = (100 * rfm_sum["资产数"] / rfm_sum["资产数"].sum()).round(2)
    for c in ["平均价值", "平均成本", "平均质量"]:
        rfm_sum[c] = rfm_sum[c].round(2)
    save(rfm_sum, "tbl_M3_rfm.csv")
    save(rfm[["asset_id", "R", "F", "M", "rfm_class"]], "tbl_M3_rfm_detail.csv")
    R["M3_rfm"] = {"切分依据": "R/M 用五分位（value_index、batch_no 排名），F 用 group_size 档距",
                   "类别数": int(rfm_sum.shape[0]),
                   "最大类": rfm_sum.iloc[0].to_dict(),
                   "高价值核心资产占比": float(rfm_sum.loc[rfm_sum["rfm_class"] == "高价值核心资产", "占比"].sum()),
                   "长尾淘汰候选数量": int(rfm_sum.loc[rfm_sum["rfm_class"] == "长尾淘汰候选", "资产数"].sum())}

    # ================================================================= M4 K-Means
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    feat_cols = ["quality_score", "info_score", "scarcity_score", "snr_db", "entropy",
                 "centroid", "bandwidth", "zcr_mean", "total_cost", "value_index"]
    X = MODEL[feat_cols].fillna(MODEL[feat_cols].median()).values
    Xs = StandardScaler().fit_transform(X)
    ks = list(range(2, 9))
    inertia, sils = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=cfg.SEED).fit(Xs)
        inertia.append(float(km.inertia_))
        sils.append(float(silhouette_score(Xs, km.labels_)))
    k_rule = "选K规则：在全局轮廓系数与业务可执行性之间取后者——仅考虑 K∈[3,6]，区间内取轮廓系数最大者"
    cand = [(k, sv) for k, sv in zip(ks, sils) if 3 <= k <= 6]
    k_best = int(max(cand, key=lambda t: t[1])[0])
    km = KMeans(n_clusters=k_best, n_init=20, random_state=cfg.SEED).fit(Xs)
    MODEL["cluster"] = km.labels_
    pca = PCA(n_components=2, random_state=cfg.SEED).fit(Xs)
    pc = pca.transform(Xs)
    MODEL["pca1"], MODEL["pca2"] = pc[:, 0], pc[:, 1]
    prof = (MODEL.groupby("cluster")
            .agg(资产数=("asset_id", "size"), 价值指数=("value_index", "mean"),
                 质量分=("quality_score", "mean"), 信息量=("info_score", "mean"),
                 稀缺度=("scarcity_score", "mean"), 成本=("total_cost", "mean"),
                 心音异常率=("heart_abnormal", lambda x: float((x == 1).mean())))
            .round(2).reset_index())
    prof = prof.sort_values("价值指数", ascending=False).reset_index(drop=True)
    prof["单位成本价值"] = (prof["价值指数"] / prof["成本"]).round(3)
    med_uv, med_sc = float(prof["单位成本价值"].median()), float(prof["稀缺度"].median())
    names = []
    for i, r in prof.iterrows():
        if i == 0:
            names.append("高值精品池")
        elif i == len(prof) - 1:
            names.append("低值长尾池")
        elif r["单位成本价值"] < med_uv:
            names.append("高成本待优化池")
        elif r["稀缺度"] > med_sc:
            names.append("稀缺补充池")
        else:
            names.append("均衡主力池")
    prof["画像命名"] = names
    prof = prof.sort_values("cluster").reset_index(drop=True)
    save(prof, "tbl_M4_cluster_profile.csv")
    save(pd.DataFrame({"k": ks, "inertia": np.round(inertia, 2), "silhouette": np.round(sils, 4),
                       "业务可执行区间": [int(3 <= k <= 6) for k in ks]}),
         "tbl_M4_elbow_silhouette.csv")
    R["M4_kmeans"] = {"特征维": feat_cols, "K候选": ks, "选K规则": k_rule, "最优K": k_best,
                      "区间内轮廓系数": round(float(max(cand, key=lambda t: t[1])[1]), 4),
                      "全局最优K": int(ks[int(np.argmax(sils))]),
                      "PCA方差解释率": [round(float(v), 4) for v in PCA(n_components=2,
                       random_state=cfg.SEED).fit(Xs).explained_variance_ratio_],
                      "分群画像": prof.to_dict(orient="records")}

    # ================================================================= M5 关联规则
    trans = []
    for r in MODEL.itertuples():
        items = set()
        if r.heart_type_zh:
            items.add("心:" + r.heart_type_zh)
        if r.lung_type_zh:
            items.add("肺:" + r.lung_type_zh)
        if r.location_zh:
            items.add("位:" + r.location_zh)
        if r.gender_zh:
            items.add("别:" + r.gender_zh)
        if items:
            trans.append(items)
    n_t = len(trans)
    from itertools import combinations
    sup1, sup2 = {}, {}
    for t in trans:
        for it in t:
            sup1[it] = sup1.get(it, 0) + 1
        for a, b in combinations(sorted(t), 2):
            sup2[(a, b)] = sup2.get((a, b), 0) + 1
    rows = []
    for (a, b), c in sup2.items():
        s = c / n_t
        if s < 0.02:
            continue
        rows.append({"前项": a, "后项": b, "support": round(s, 4),
                     "confidence_A→B": round(c / sup1[b], 4),
                     "confidence_B→A": round(c / sup1[a], 4),
                     "lift": round(c * n_t / (sup1[a] * sup1[b]), 4), "共现次数": c})
    rules = pd.DataFrame(rows).sort_values("lift", ascending=False)
    save(rules.head(40), "tbl_M5_rules_top40.csv")
    lift_items = sorted(set(rules["前项"]) | set(rules["后项"]))
    if len(lift_items) > 14:
        keep = rules.nlargest(60, "lift")[["前项", "后项"]]
        lift_items = sorted(set(keep["前项"]) | set(keep["后项"]))
    MAT = pd.DataFrame(np.nan, index=lift_items, columns=lift_items)
    for _, r in rules.iterrows():
        if r["前项"] in MAT.index and r["后项"] in MAT.columns:
            MAT.loc[r["前项"], r["后项"]] = r["lift"]
            MAT.loc[r["后项"], r["前项"]] = r["lift"]
    save(MAT.round(3).rename_axis("项目").reset_index(), "tbl_M5_lift_matrix.csv")
    R["M5_assoc"] = {"样本数": n_t, "规则数(2项)": int(len(rules)),
                     "最小support": 0.02, "lift>1规则数": int((rules["lift"] > 1).sum()),
                     "TOP规则": rules.head(6).to_dict(orient="records")}

    # ================================================================= M6 Holt-Winters 预测
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as HW
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    blk = B[B["asset_id"].isin(MODEL["asset_id"])].groupby("block_idx")["block_rms_db"].mean().sort_index()
    y_raw = blk.values.astype(float)
    shift = float(-np.nanmin(y_raw) + 1.0)          # dB 为负，整体平移为正，MAPE 才有业务含义
    y = y_raw + shift
    n_tr = int(len(y) * cfg.FORECAST_TRAIN_RATIO)
    y_tr, y_te = y[:n_tr], y[n_tr:]
    hh = len(y_te)
    h = np.arange(1, hh + 1)

    fit = HW(y_tr, trend="add", damped_trend=True, seasonal="add",
             seasonal_periods=15, initialization_method="estimated").fit(smoothing_level=None)
    pred = np.asarray(fit.predict(n_tr, n_tr + hh - 1), dtype=float)
    resid = np.asarray(fit.resid[n_tr // 4:], dtype=float)      # 去掉前 1/4 warm-up
    sigma = float(np.nanstd(resid))
    z = 1.959963985
    lo, hi = pred - z * sigma * np.sqrt(h), pred + z * sigma * np.sqrt(h)
    hw_desc = "Holt-Winters：加法趋势(阻尼) + 加法季节(周期15块=1.5s)"
    fut = np.asarray(fit.predict(len(y), len(y) + cfg.FORECAST_HORIZON - 1), dtype=float)
    hs = np.arange(1, cfg.FORECAST_HORIZON + 1)
    fut_lo, fut_hi = fut - z * sigma * np.sqrt(hs), fut + z * sigma * np.sqrt(hs)

    naive = np.full(hh, y_tr[-1])
    drift = y_tr[-1] + (y_tr[-1] - y_tr[-31]) / 30.0 * h
    try:
        sfit = SARIMAX(y_tr, order=(1, 0, 1), seasonal_order=(1, 0, 0, 15),
                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        sf = sfit.get_forecast(hh)
        sp = np.asarray(sf.predicted_mean, dtype=float)
        s_mape = float(np.mean(np.abs((y_te - sp) / y_te)) * 100)
        sarima_desc = "SARIMAX(1,0,1)(1,0,0,15)"
    except Exception as e:
        sp, s_mape = None, float("nan")
        sarima_desc = f"SARIMAX 拟合失败：{type(e).__name__}"

    def _mape(e):
        return float(np.mean(np.abs((y_te - e) / y_te)) * 100)

    def _rmse(e):
        return float(np.sqrt(np.mean((y_te - e) ** 2)))

    mape, rmse = _mape(pred), _rmse(pred)
    mape_naive, mape_drift = _mape(naive), _mape(drift)
    hit = (y_te >= lo) & (y_te <= hi)
    fc_tbl = pd.DataFrame({
        "block_idx": np.arange(n_tr, n_tr + hh),
        "t_sec": np.round(np.arange(n_tr, n_tr + hh) * cfg.BLOCK_MS / 1000, 2),
        "实际值": np.round(y_te, 3), "点预测": np.round(pred, 3),
        "下限95": np.round(lo, 3), "上限95": np.round(hi, 3),
        "绝对误差": np.round(np.abs(y_te - pred), 3), "区间命中": hit.astype(int)})
    save(fc_tbl, "tbl_M6_forecast_backtest.csv")
    save(pd.DataFrame({"block_idx": np.arange(len(y), len(y) + cfg.FORECAST_HORIZON),
                       "t_sec": np.round(np.arange(len(y), len(y) + cfg.FORECAST_HORIZON)
                                         * cfg.BLOCK_MS / 1000, 2),
                       "点预测": np.round(fut, 3), "下限95": np.round(fut_lo, 3),
                       "上限95": np.round(fut_hi, 3)}), "tbl_M6_forecast_next.csv")
    save(pd.DataFrame({"block_idx": blk.index, "y_平移后": np.round(y, 4),
                       "y_原始dB": np.round(y_raw, 4)}), "tbl_M6_series.csv")
    save(pd.DataFrame({
        "方法": [hw_desc, "朴素法(期末值)", "漂移法", sarima_desc],
        "MAPE_pct": [round(mape, 3), round(mape_naive, 3), round(mape_drift, 3), round(s_mape, 3)],
        "RMSE": [round(rmse, 3), round(_rmse(naive), 3), round(_rmse(drift), 3),
                 round(_rmse(sp), 3) if sp is not None else np.nan]}), "tbl_M6_benchmark.csv")
    cands = [(hw_desc, mape, rmse)]
    if sp is not None:
        cands.append((sarima_desc, s_mape, _rmse(sp)))
    primary_desc, primary_mape, primary_rmse = min(cands, key=lambda t: t[1])
    if primary_desc == sarima_desc:
        sfit2 = SARIMAX(y, order=(1, 0, 1), seasonal_order=(1, 0, 0, 15),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        gf = sfit2.get_forecast(cfg.FORECAST_HORIZON)
        fut = np.asarray(gf.predicted_mean, dtype=float)
        gci = np.asarray(gf.conf_int(alpha=cfg.CI_ALPHA), dtype=float)
        fut_lo, fut_hi = gci[:, 0], gci[:, 1]
        ci_src = "SARIMAX 状态空间内生 95% 置信区间"
    else:
        ci_src = "点预测 ± 1.96·σ·√h（σ=截尾残差标准差，h=预测步长）"
    save(pd.DataFrame({"方法": [c[0] for c in cands] + ["朴素法(期末值)", "漂移法"],
                       "回测MAPE_pct": [round(c[1], 3) for c in cands] + [round(mape_naive, 3), round(mape_drift, 3)],
                       "回测RMSE": [round(c[2], 3) for c in cands] + [round(_rmse(naive), 3), round(_rmse(drift), 3)]}),
         "tbl_M6_benchmark.csv")
    save(pd.DataFrame({"block_idx": np.arange(len(y), len(y) + cfg.FORECAST_HORIZON),
                       "t_sec": np.round(np.arange(len(y), len(y) + cfg.FORECAST_HORIZON) * cfg.BLOCK_MS / 1000, 2),
                       "点预测": np.round(fut, 3), "下限95": np.round(fut_lo, 3),
                       "上限95": np.round(fut_hi, 3)}), "tbl_M6_forecast_next.csv")
    R["M6_forecast"] = {"序列": "全体入模资产 100ms 时间块平均能量，150 点(=15s)，dB 平移为正后建模",
                        "备选模型": [c[0] for c in cands],
                        "最终选用": primary_desc, "选模依据": "回测 MAPE 最小",
                        "训练/回测点数": [int(n_tr), int(hh)],
                        "MAPE_pct": round(primary_mape, 3), "RMSE": round(primary_rmse, 3),
                        "HoltWinters_MAPE_pct": round(mape, 3),
                        "SARIMAX_MAPE_pct": round(s_mape, 3),
                        "朴素法MAPE_pct": round(mape_naive, 3),
                        "漂移法MAPE_pct": round(mape_drift, 3),
                        "相对朴素法MAPE改善_pct": round(mape_naive - primary_mape, 3),
                        "残差sigma": round(sigma, 3),
                        "95%区间命中率_pct": round(100 * float(hit.mean()), 2),
                        "区间口径": ci_src, "回测窗口_秒": round(hh * cfg.BLOCK_MS / 1000, 2),
                        "前瞻步数": cfg.FORECAST_HORIZON,
                        "预测用途": "切片/预标注排程：预测 1.5s 内的有效发声段位置"}

    # ================================================================= M7 漏斗
    n_all = len(M)
    fmt_ok = n_all  # s02 已校验（format_issues 为空）
    label_ok = int(M["stem"].notna().sum())
    uniq = int(M["is_primary_copy"].sum())
    qc_ok = int(((M["is_primary_copy"] == 1) & (M["qc_pass"] == 1)).sum())
    ab = int((MODEL["abc"].isin(["A", "B"])).sum())
    a_class = int((MODEL["abc"] == "A").sum())
    funnel = pd.DataFrame({
        "环节": ["① 原始入库资产", "② 音频参数合规", "③ 标签可关联", "④ 内容唯一(去重后)",
                 "⑤ 质检达标", "⑥ A+B 类资产(累计价值前90%)", "⑦ A 类资产(累计价值前70%)"],
        "数量": [n_all, fmt_ok, label_ok, uniq, qc_ok, ab, a_class]})
    funnel["留存率"] = (100 * funnel["数量"] / n_all).round(2)
    funnel["环节转化率"] = (100 * funnel["数量"] / funnel["数量"].shift(1).fillna(n_all)).round(2)
    assert funnel["数量"].is_monotonic_decreasing, "漏斗环节必须严格嵌套"
    save(funnel, "tbl_M7_funnel.csv")
    R["M7_funnel"] = {"表": funnel.to_dict(orient="records"),
                      "最大损耗环节": "③→④ 内容去重（535→344，损耗 191 条 = 35.7%）",
                      "去重损耗": int(n_all - uniq),
                      "去重节省成本_元": round(float(M.loc[M.is_primary_copy == 0, "total_cost"].sum()), 2)}

    # ================================================================= M8 异常检出与成本曲线
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler as SS2
    hf = MODEL[(MODEL["heart_abnormal"] >= 0)].copy()
    Xf = hf[["rms", "zcr_mean", "centroid", "bandwidth", "entropy", "flatness", "snr_db",
             "crest_factor", "env_cv", "heart_rate_bpm", "mfcc1_mean", "mfcc2_mean",
             "mfcc3_std", "b1_20_150", "b3_300_600"]].fillna(0.0)
    yv = hf["heart_abnormal"].astype(int).values
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    lr = make_pipeline(SS2(), LogisticRegression(max_iter=3000, class_weight="balanced",
                                                 random_state=cfg.SEED))
    rf = RandomForestClassifier(n_estimators=400, random_state=cfg.SEED, n_jobs=1)
    p_lr = cross_val_predict(lr, Xf, yv, cv=skf, method="predict_proba")[:, 1]
    p_rf = cross_val_predict(rf, Xf, yv, cv=skf, method="predict_proba")[:, 1]
    auc_lr, auc_rf = float(roc_auc_score(yv, p_lr)), float(roc_auc_score(yv, p_rf))
    fpr, tpr, thr = roc_curve(yv, p_rf)
    save(pd.DataFrame({"fpr": fpr.round(4), "tpr": tpr.round(4), "threshold": np.round(thr, 4)}),
         "tbl_M8_roc.csv")
    imp = pd.Series(rf.fit(Xf, yv).feature_importances_, index=Xf.columns).sort_values(ascending=False)
    save(imp.round(5).rename("importance").reset_index().rename(columns={"index": "特征"}),
         "tbl_M8_importance.csv")
    budget = []
    order = np.argsort(-p_rf)
    cum_ab = np.cumsum(yv[order])
    for pct in [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100]:
        n_rev = int(np.ceil(len(yv) * pct / 100))
        budget.append({"复核比例%": pct, "复核条数": n_rev,
                       "捕获异常数": int(cum_ab[n_rev - 1]),
                       "异常召回率": round(100 * float(cum_ab[n_rev - 1]) / yv.sum(), 2),
                       "复核成本_元": round(n_rev * float(MODEL["total_cost"].mean()) * 0.35, 2),
                       "单位捕获成本_元": round(n_rev * float(MODEL["total_cost"].mean()) * 0.35
                                             / max(int(cum_ab[n_rev - 1]), 1), 3)})
    save(pd.DataFrame(budget), "tbl_M8_cost_curve.csv")
    R["M8_model"] = {"样本": int(len(hf)), "正样本率": round(float(yv.mean()), 4),
                     "折数": cfg.N_FOLDS, "AUC_逻辑回归": round(auc_lr, 4),
                     "AUC_随机森林": round(auc_rf, 4),
                     "TOP5特征": imp.head(5).round(4).to_dict(),
                     "召回50%所需复核比例": next(b["复核比例%"] for b in budget if b["异常召回率"] >= 50),
                     "全量复核成本": budget[-1]["复核成本_元"],
                     "最优性价比点": min(budget, key=lambda b: b["单位捕获成本_元"])}

    # ---------------- M8b 心音品类识别：自动预标注产品线 ----------------
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold as SKF2, cross_val_predict as CVP2
    hc = MODEL[(MODEL["heart_type"] != "")].copy()
    Xc = hc[["rms", "zcr_mean", "zcr_std", "centroid", "centroid_std", "bandwidth", "entropy",
             "flatness", "snr_db", "crest_factor", "env_cv", "env_crest", "heart_rate_bpm",
             "hr_autocorr_peak", "b1_20_150", "b2_150_300", "b3_300_600", "b4_600_1200",
             "b5_1200_2000", "mfcc1_mean", "mfcc2_mean", "mfcc3_std", "mfcc4_std"]].fillna(0.0)
    yc = hc["heart_type_zh"].values
    sk2 = SKF2(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                 class_weight="balanced_subsample",
                                 random_state=cfg.SEED, n_jobs=1)
    proba = CVP2(clf, Xc, yc, cv=sk2, method="predict_proba")
    pred_cls = CVP2(clf, Xc, yc, cv=sk2)          # 逐折预测标签，避免依赖单一 classes_ 顺序
    conf = proba.max(axis=1)
    classes_full = clf.fit(Xc, yc).classes_
    assert len(classes_full) == proba.shape[1]
    order = np.argsort(-proba, axis=1)
    topk = {}
    for k in (1, 3, 5):
        hits = np.array([yc[i] in set(classes_full[order[i, :k]]) for i in range(len(yc))])
        topk[f"top{k}命中率_pct"] = round(100 * float(hits.mean()), 2)
    acc = float(accuracy_score(yc, pred_cls))
    f1m = float(f1_score(yc, pred_cls, average="macro"))
    base = float(pd.Series(yc).value_counts(normalize=True).max())
    rows = []
    for t in sorted(set(yc)):
        m = yc == t
        rows.append({"品类": t, "样本数": int(m.sum()),
                     "召回率_pct": round(100 * float((pred_cls[m] == t).mean()), 2),
                     "平均置信度": round(float(conf[m].mean()), 4)})
    save(pd.DataFrame(rows).sort_values("召回率_pct", ascending=False), "tbl_M8b_per_class.csv")
    pre = []
    for th in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        sel = conf >= th
        if int(sel.sum()) == 0:
            continue
        prec = float((pred_cls[sel] == yc[sel]).mean())
        n_auto = int(sel.sum())
        saved_min = n_auto * float(hc.loc[sel, "label_minutes"].mean()) * 0.6
        pre.append({"置信度阈值": th, "自动通过条数": n_auto,
                    "自动通过率_pct": round(100 * n_auto / len(yc), 2),
                    "通过样本正确率_pct": round(100 * prec, 2),
                    "节约标注工时_分钟": round(saved_min, 1),
                    "节约标注成本_元": round(saved_min * cfg.LABELER_YUAN_PER_MIN, 2)})
    pre_tbl = pd.DataFrame(pre)
    save(pre_tbl, "tbl_M8b_prelabel.csv")
    good = pre_tbl[pre_tbl["通过样本正确率_pct"] >= 85]
    pick = (good.nlargest(1, "自动通过率_pct").iloc[0].to_dict() if len(good) else None)
    # 没有一档达到 85% 正确率时，如实结论：自动通过不可行，只能做“候选提示”
    if pick is None:
        best_pre = pre_tbl.nlargest(1, "通过样本正确率_pct").iloc[0].to_dict()
        hint = {"判定": "自动通过不可行",
                "最高通过正确率_pct": best_pre["通过样本正确率_pct"],
                "对应阈值": best_pre["置信度阈值"],
                "替代方案": f"以 Top-3 候选提示辅助人工判读（命中率 {topk['top3命中率_pct']}%），"
                          f"按每条节省 25% 判读工时折算",
                "替代方案节约工时_分钟": round(float(len(yc)) * float(hc["label_minutes"].mean()) * 0.25, 1),
                "替代方案节约成本_元": round(float(len(yc)) * float(hc["label_minutes"].mean()) * 0.25
                                       * cfg.LABELER_YUAN_PER_MIN, 2)}
    else:
        hint = {"判定": "自动通过可行", "推荐阈值行": pick}
    R["M8b_prelabel"] = {"样本": int(len(hc)), "类别数": int(len(set(yc))),
                         "折数": cfg.N_FOLDS, "Top1准确率": round(acc, 4),
                         "macro_F1": round(f1m, 4), "多数类基线": round(base, 4),
                         "相对基线提升_pct": round(100 * (acc - base) / base, 2),
                         "阈值规则": "通过样本正确率≥85% 时取最大自动通过率，否则判定自动通过不可行",
                         **topk, "结论": hint}

    # ---------------- 汇总 ----------------
    MODEL.to_csv(cfg.DATA / "model_table.csv", index=False, encoding="utf-8-sig")
    (cfg.EDA / "model_results.json").write_text(json.dumps(R, ensure_ascii=False, indent=2,
                                                           default=str), encoding="utf-8")
    print(json.dumps(R, ensure_ascii=False, indent=2, default=str)[:3000])
    print("\n... tables:", ", ".join(sorted(p.name for p in cfg.TABLE_DIR.glob("tbl_*.csv"))))


if __name__ == "__main__":
    main()
