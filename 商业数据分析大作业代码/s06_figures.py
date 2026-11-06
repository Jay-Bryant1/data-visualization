# -*- coding: utf-8 -*-
"""s06_figures.py —— 业务模型类图表 fig19–fig27，与 s04 的 8 个模型一一对应。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cfg
from plotstyle import PALETTE, ACCENT, load, tbl, finish, style_ax, footer

A, L, B, R = load()
MOD = A[A["in_model"] == 1].copy()


def fig19():
    d = MOD.sort_values("value_index", ascending=False).reset_index(drop=True)
    tot = float(d["value_index"].sum())
    d["cum"] = 100 * d["value_index"].cumsum() / tot
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.6, 6.2), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2.4, 1]))
    colmap = {"A": ACCENT["bad"], "B": ACCENT["warn"], "C": ACCENT["navy"]}
    ax.bar(range(len(d)), d["value_index"], color=[colmap[a] for a in d["abc"]], width=1.0)
    ax.set_ylabel("单条资产价值指数")
    ax.set_title("帕累托图｜344 条独立资产的价值排序与 ABC 分层", pad=12, fontweight="bold")
    for cut, lab in [(cfg.ABC_CUM[0], f"A｜累计{cfg.ABC_CUM[0]:.0f}%"),
                     (cfg.ABC_CUM[1], f"B｜累计{cfg.ABC_CUM[1]:.0f}%")]:
        i = int(np.searchsorted(d["cum"].values, cut))
        ax.axvline(i, color="#333", ls="--", lw=1.1)
        ax.text(i - 2, d["value_index"].max() * 0.92, lab, ha="right", fontsize=8.4, color="#333")
    ax2.plot(range(len(d)), d["cum"], color=PALETTE[0], lw=2.2, label="实际累计价值占比")
    ax2.plot([0, len(d) - 1], [0, 100], ls=":", color="#999", lw=1.4, label="绝对均匀参照线")
    ax2.axhline(80, ls="--", color=ACCENT["good"], lw=1.2)
    i80 = int(np.searchsorted(d["cum"].values, 80))
    ax2.annotate(f"累计 80% 价值需 {i80+1} 条（占 {100*(i80+1)/len(d):.1f}%）",
                 (i80, 80), xytext=(i80 - 150, 52), fontsize=9, color=ACCENT["good"],
                 arrowprops=dict(arrowstyle="->", color=ACCENT["good"]))
    ax2.set_ylabel("累计价值占比 %"); ax2.set_xlabel("资产序号（按价值降序）")
    ax2.set_ylim(0, 102); ax2.legend(loc="lower right", fontsize=8.6)
    ax2.grid(axis="x", visible=False)
    footer(fig, f"结论：A 类以 {R['M1_abc']['A类数量占比']:.1f}% 的数量承载 {R['M1_abc']['A类价值占比']:.1f}% 的价值，"
                f"偏离度仅 {R['M1_abc']['帕累托偏离度']:.1f} 个百分点 → 未形成二八结构，资产同质化严重")
    finish(fig, "fig19_帕累托图_ABC价值分层.png")



def fig20():
    d = tbl("tbl_M2_bcg.csv")
    d["x"] = d["市场份额"] * 100
    med = float(d.loc[d["x"] > 0, "x"].median())
    fig, ax = plt.subplots(figsize=(10.0, 6.4))
    cmap = {"明星": ACCENT["bad"], "现金牛": ACCENT["navy"], "问题": ACCENT["warn"], "瘦狗": "#7F7F7F"}
    ax.set_xscale("log")
    ax.axvspan(0.8, med, ymin=0.5, color=ACCENT["warn"], alpha=0.07)
    ax.axvspan(med, 120, ymin=0.5, color=ACCENT["bad"], alpha=0.07)
    ax.axvspan(0.8, med, ymax=0.5, color="#7F7F7F", alpha=0.07)
    ax.axvspan(med, 120, ymax=0.5, color=ACCENT["navy"], alpha=0.07)
    for q, sub in d.groupby("象限"):
        ax.scatter(sub["x"], sub["份额增速"], s=sub["平均价值指数"] * 5.5 + 30, color=cmap[q],
                   alpha=0.7, edgecolor="white", lw=0.7, label=f"{q} n={len(sub)}")
    placed = []
    cands = pd.concat([d.nlargest(6, "x"), d.nlargest(4, "份额增速"),
                       d.nsmallest(3, "份额增速")]).drop_duplicates()
    cands = cands.reindex(cands["份额增速"].abs().sort_values(ascending=False).index)  # 极值优先占位
    for _, r in cands.iterrows():                  # 贪心避让：过密标签跳过，避免糊成一团
        lx, ly = np.log10(max(r["x"], 1e-9)), r["份额增速"]
        if any(abs(lx - px) < 0.22 and abs(ly - py) < 25 for px, py in placed):
            continue
        placed.append((lx, ly))
        dy = -14 if r["份额增速"] > 200 else 4       # 顶部气泡标签下移，避开象限角标
        ax.annotate(r["品类"], (r["x"], r["份额增速"]), textcoords="offset points", xytext=(6, dy),
                    fontsize=7.2, color="#333")
    ax.axhline(0, color="#444", lw=1.1)
    ax.axvline(med, color="#444", lw=1.1)
    ax.set_xlim(0.8, 120)
    ax.text(0.995, 0.90, "明星（高份额·正增长）", transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color=ACCENT["bad"], fontweight="bold")
    ax.text(0.005, 0.90, "问题（低份额·正增长）", transform=ax.transAxes, ha="left", va="top",
            fontsize=10, color=ACCENT["warn"], fontweight="bold")
    ax.text(0.005, 0.015, "瘦狗（低份额·负增长）", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, color="#7F7F7F", fontweight="bold")
    ax.text(0.995, 0.015, "现金牛（高份额·负增长）", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color=ACCENT["navy"], fontweight="bold")
    style_ax(ax, "波士顿矩阵｜品类市场份额（对数轴）× 批次间份额增速（气泡=平均价值指数）",
             xlabel="市场份额 %（log）", ylabel="份额增速 %")
    ax.legend(loc="center right", fontsize=8.4)
    footer(fig, "口径：份额=品类资产量/344；增速=后半段批次份额相对前半段的变化率（入库流水代理时间，"
                f"非真实日历时间，属可解释性折中）；纵向分界线=有效品类份额中位数 {med:.1f}%")
    finish(fig, "fig20_波士顿矩阵_品类四象限.png")


def fig21():
    d = tbl("tbl_M3_rfm_detail.csv")
    classes = sorted(d["rfm_class"].unique())
    cidx = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(classes)}
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), sharex=True, sharey=True)
    for ax, f_level in zip(axes.flat, [1, 2, 3, 4]):
        sub = d[d["F"] == f_level]
        for cl, ss in sub.groupby("rfm_class"):
            jx = ss["M"] + np.random.RandomState(abs(hash(cl)) % 999).uniform(-0.14, 0.14, len(ss))
            jy = ss["R"] + np.random.RandomState(abs(hash(cl + "r")) % 999).uniform(-0.14, 0.14, len(ss))
            ax.scatter(jx, jy, s=34, color=cidx[cl], alpha=0.62, edgecolor="white", lw=0.4,
                       label=cl if ax is axes.flat[0] else None)
        ax.set_title(f"F={f_level}（复用频次档）｜n={len(sub)}", fontsize=9.5)
        ax.set_xticks(range(1, 6)); ax.set_yticks(range(1, 6))
        ax.axvline(2.5, color="#BBB", lw=0.9); ax.axhline(2.5, color="#BBB", lw=0.9)
        ax.text(5.35, 0.6, "高M低R" if f_level <= 2 else "", fontsize=7, color="#999")
    axes[1][0].set_xlabel("M｜价值指数五分位"); axes[1][1].set_xlabel("M｜价值指数五分位")
    axes[0][0].set_ylabel("R｜批次新鲜度五分位"); axes[1][0].set_ylabel("R｜批次新鲜度五分位")
    fig.suptitle("RFM 八分类｜按复用频次 F 分面的 R×M 分布", fontsize=13, fontweight="bold", y=0.97)
    axes[0][1].legend(fontsize=7.6, loc="upper left", ncol=1, framealpha=0.9)
    footer(fig, f"切分依据：R/M 取五分位（batch_no 排名、value_index 排名），F 按 group_size 档距 1/2/3/4+；"
                f"高价值核心资产占比 {R['M3_rfm']['高价值核心资产占比']:.2f}%，长尾淘汰候选 "
                f"{R['M3_rfm']['长尾淘汰候选数量']} 条")
    finish(fig, "fig21_RFM八分类_资产分层气泡.png")

def fig22():
    prof = tbl("tbl_M4_cluster_profile.csv")
    es = tbl("tbl_M4_elbow_silhouette.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.8), gridspec_kw=dict(width_ratios=[1, 1.5]))
    ax1.plot(es["k"], es["inertia"], "o-", color=PALETTE[0], lw=2, label=" inertia（肘部）")
    ax1.set_xlabel("聚类数 K"); ax1.set_ylabel("Inertia", color=PALETTE[0])
    ax1b = ax1.twinx()
    ax1b.plot(es["k"], es["silhouette"], "s--", color=ACCENT["bad"], lw=2, label="轮廓系数")
    ax1b.set_ylabel("Silhouette", color=ACCENT["bad"])
    ax1.axvspan(3, 6, color="#EEEEEE", alpha=0.8)
    ax1.text(4.5, 1150, "业务可执行区间\nK∈[3,6]", ha="center", fontsize=8, color="#666")
    ax1.set_ylim(1000, 2350)
    ax1.set_title("选 K 依据（A）｜肘部法 + 轮廓系数", pad=10, fontweight="bold")
    ax1b.grid(False)
    for i, (c, sub) in enumerate(MOD.groupby("cluster")):
        ax2.scatter(sub["pca1"], sub["pca2"], s=22, color=PALETTE[i % len(PALETTE)], alpha=0.62,
                    edgecolor="white", lw=0.4,
                    label=f"{c} {prof.loc[prof['cluster']==c,'画像命名'].iloc[0]} n={len(sub)}")
    cen = MOD.groupby("cluster")[["pca1", "pca2"]].mean()
    ax2.scatter(cen["pca1"], cen["pca2"], marker="X", s=190, color="black", zorder=5, label="聚类中心")
    style_ax(ax2, f"PCA 投影（B）｜K={R['M4_kmeans']['最优K']} 分群结果",
             xlabel=f"PC1（{100*R['M4_kmeans']['PCA方差解释率'][0]:.1f}%）",
             ylabel=f"PC2（{100*R['M4_kmeans']['PCA方差解释率'][1]:.1f}%）")
    ax2.legend(fontsize=8, loc="best")
    footer(fig, f"选 K 规则：{R['M4_kmeans']['选K规则']}；区间内轮廓系数 {R['M4_kmeans']['区间内轮廓系数']}")
    finish(fig, "fig22_KMeans_选K与PCA分群.png")



def fig23():
    rules = tbl("tbl_M5_rules_top40.csv").head(12).iloc[::-1]
    full = pd.read_csv(cfg.TABLE_DIR / "tbl_M5_lift_matrix.csv", index_col=0)
    items = sorted(set(rules["前项"]) | set(rules["后项"]))[:12]
    lift = full.loc[[i for i in full.index if i in items], [i for i in full.columns if i in items]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.8), gridspec_kw=dict(width_ratios=[1.1, 1]))
    y = np.arange(len(rules))
    ax1.barh(y, rules["lift"], color=[ACCENT["bad"] if v > 2 else ACCENT["warn"] if v > 1.3 else PALETTE[4]
                                      for v in rules["lift"]], height=0.66)
    ax1.axvline(1.0, color="#333", lw=1.3)
    ax1.set_yticks(y, [f"{a} → {b}" for a, b in zip(rules["前项"], rules["后项"])], fontsize=7.6)
    for yi, (v, sp) in zip(y, zip(rules["lift"], rules["support"])):
        ax1.text(v + 0.03, yi, f"lift {v:.2f}｜sup {sp:.3f}", va="center", fontsize=7.2, color="#444")
    ax1.set_xlabel("提升度 lift（=1 为独立）"); ax1.grid(axis="y", visible=False)
    ax1.set_title("关联规则（A）｜Top12 高提升度组合", pad=10, fontweight="bold")
    hm = sns.heatmap(lift.astype(float), cmap="YlOrRd", ax=ax2, annot=lift.shape[0] <= 12,
                     fmt=".2f", annot_kws={"fontsize": 5.8}, linewidths=0.5, linecolor="white",
                     cbar_kws={"label": "lift"}, vmin=0, vmax=float(np.nanmax(lift.values)))
    ax2.set_title("提升度矩阵（B）｜Top12 高频项（空白=未达 support 阈值）", pad=10, fontweight="bold",
                  fontsize=11)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=52, ha="right", fontsize=6.8)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=6.8)
    footer(fig, f"参数：最小 support=0.02（业务依据：单条规则至少覆盖 "
                f"{int(0.02*R['M5_assoc']['样本数'])} 条资产才值得排产），共 {R['M5_assoc']['规则数(2项)']} 条 "
                f"2 项规则，lift>1 的 {R['M5_assoc']['lift>1规则数']} 条")
    finish(fig, "fig23_关联规则_提升度.png")

def fig24():
    ser = tbl("tbl_M6_series.csv")
    bt = tbl("tbl_M6_forecast_backtest.csv")
    nx = tbl("tbl_M6_forecast_next.csv")
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    ax.plot(ser["block_idx"] * 0.1, ser["y_平移后"], color="#333", lw=1.4, label="实际平均能量（训练段+回测段）")
    ax.fill_between(bt["block_idx"] * 0.1, bt["下限95"], bt["上限95"], color=PALETTE[4], alpha=0.22,
                    label="回测 95% 置信区间")
    ax.plot(bt["block_idx"] * 0.1, bt["点预测"], color=PALETTE[4], lw=1.6, ls="-", marker="o", ms=3,
            label="回测点预测")
    ax.fill_between(nx["block_idx"] * 0.1, nx["下限95"], nx["上限95"], color=ACCENT["bad"], alpha=0.22,
                    label="前瞻 1.5s 的 95% 置信区间")
    ax.plot(nx["block_idx"] * 0.1, nx["点预测"], color=ACCENT["bad"], lw=1.8, ls="--", marker="s", ms=3.4,
            label="前瞻点预测")
    b0 = float(ser["block_idx"].iloc[int(len(ser) * 0.8)])
    ax.axvline(b0 * 0.1, color="#888", ls=":", lw=1.3)
    ax.text(b0 * 0.1 + 0.05, ser["y_平移后"].max() * 0.98, "训练/回测分界（80%）", fontsize=8, color="#666")
    style_ax(ax, "时间序列预测｜能量包络的点预测与 95% 置信区间",
             xlabel="录音内时间（秒）", ylabel="平均能量（dB 平移为正）")
    ax.legend(fontsize=8.2, loc="lower right", ncol=2)
    footer(fig, f"最终模型：{R['M6_forecast']['最终选用']}；回测 MAPE={R['M6_forecast']['MAPE_pct']}%、"
                f"RMSE={R['M6_forecast']['RMSE']}，朴素法 {R['M6_forecast']['朴素法MAPE_pct']}%、"
                f"漂移法 {R['M6_forecast']['漂移法MAPE_pct']}% → 相对朴素法改善 "
                f"{R['M6_forecast']['相对朴素法MAPE改善_pct']} 个百分点")
    finish(fig, "fig24_时间序列预测_置信区间与回测.png")



def fig25():
    f = tbl("tbl_M7_funnel.csv")
    vals = f["数量"].values
    maxv = float(vals.max())
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    gap = 0.36
    for i, (nm, v) in enumerate(zip(f["环节"], vals)):
        w = max(v / maxv, 0.30)
        w_next = max(vals[i + 1] / maxv, 0.30) if i + 1 < len(vals) else w * 0.9
        y0 = len(vals) - i - 1
        pts = [((1 - w) / 2, y0 + gap / 2), ((1 + w) / 2, y0 + gap / 2),
               ((1 + w_next) / 2, y0 - gap / 2), ((1 - w_next) / 2, y0 - gap / 2)]
        ax.add_patch(Polygon(pts, closed=True, facecolor=PALETTE[i % len(PALETTE)], alpha=0.88,
                             edgecolor="white", lw=1.2))
        txt = f"{nm}｜{int(v)} 条（{100*v/maxv:.1f}%）"
        w_mid = (w + w_next) / 2                     # 梯形中线宽度才是文字可用宽度
        tw = 0.015 * len(txt)
        if w_mid > tw + 0.04:
            ax.text(0.5, y0, txt, ha="center", va="center", color="white", fontsize=9.2,
                    fontweight="bold")
        else:
            ax.text((1 + w) / 2 + 0.02, y0, f"{nm}｜{int(v)} 条", ha="left", va="center", color="#333",
                    fontsize=9.2, fontweight="bold")
        if i:
            ax.text(1.34, y0, f"转化 {100*v/vals[i-1]:.1f}%", va="center", fontsize=8.4, color="#555")
            loss = int(vals[i - 1] - v)
            ax.text(-0.03, y0, f"损耗 {loss}" if loss else "无损耗", ha="right", va="center",
                    fontsize=8.4, color=ACCENT["bad"] if loss else "#999")
    ax.set_xlim(-0.25, 1.62); ax.set_ylim(-0.8, len(vals) - 0.2)
    ax.axis("off")
    ax.set_title("漏斗转化图｜数据资产投产全链路", pad=12, fontweight="bold")
    footer(fig, f"最大损耗环节：内容去重（535→344，−191 条 / −35.7%），对应浪费成本 "
                f"{R['M7_funnel']['去重节省成本_元']:.2f} 元；⑥⑦ 为价值分层筛选而非质量淘汰")
    finish(fig, "fig25_漏斗图_资产投产转化.png")


def fig26():
    roc = tbl("tbl_M8_roc.csv")
    cost = tbl("tbl_M8_cost_curve.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 5.0))
    ax1.plot(roc["fpr"], roc["tpr"], color=ACCENT["bad"], lw=2.2,
             label=f"随机森林 AUC={R['M8_model']['AUC_随机森林']:.3f}")
    ax1.plot([0, 1], [0, 1], ls="--", color="#999", lw=1.2, label="随机猜测基线 AUC=0.500")
    style_ax(ax1, f"ROC（A）｜心音异常检出（{R['M8_model']['样本']} 条，正样本率 "
                  f"{R['M8_model']['正样本率']:.1%}）", xlabel="假正率 FPR", ylabel="真正率 TPR")
    ax1.legend(fontsize=8.4, loc="lower right")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.02)
    ax2.plot(cost["复核比例%"], cost["异常召回率"], "o-", color=PALETTE[0], lw=2.2, label="异常召回率")
    ax2b = ax2.twinx()
    ax2b.plot(cost["复核比例%"], cost["单位捕获成本_元"], "s--", color=ACCENT["bad"], lw=2,
              label="单位捕获成本（元/条）")
    best = cost.loc[cost["单位捕获成本_元"].idxmin()]
    ax2.scatter([best["复核比例%"]], [best["异常召回率"]], s=170, color=ACCENT["warn"], zorder=5,
                edgecolor="white", label=f"性价比拐点：复核 {best['复核比例%']:.0f}% → 召回 {best['异常召回率']:.1f}%")
    style_ax(ax2, "复核预算曲线（B）｜抽检比例 vs 异常召回", xlabel="人工复核比例 %", ylabel="异常召回率 %")
    ax2b.set_ylabel("单位捕获成本（元）"); ax2b.grid(False); ax2.set_xlim(0, 100)
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=7.8, loc="upper left")
    footer(fig, f"关键结论：浅层声学特征 AUC 仅 {R['M8_model']['AUC_随机森林']:.3f}（低于随机），"
                f"逻辑回归 {R['M8_model']['AUC_逻辑回归']:.3f} → 不支持“自动异常筛查”作为商业卖点")
    finish(fig, "fig26_ROC与复核成本曲线.png")

def fig27():
    per = tbl("tbl_M8b_per_class.csv")
    pre = tbl("tbl_M8b_prelabel.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 5.0), gridspec_kw=dict(width_ratios=[1.25, 1]))
    y = np.arange(len(per))
    ax1.barh(y, per["召回率_pct"], color=[ACCENT["good"] if v >= 40 else PALETTE[4] if v >= 25 else ACCENT["bad"]
                                         for v in per["召回率_pct"]], height=0.68)
    ax1.set_yticks(y, per["品类"], fontsize=8.4)
    for yi, (v, n) in zip(y, zip(per["召回率_pct"], per["样本数"])):
        ax1.text(v + 0.8, yi, f"{v:.1f}%（n={n}）", va="center", fontsize=7.6, color="#444")
    ax1.axvline(100 / R["M8b_prelabel"]["类别数"], ls=":", color="#666", lw=1.3)
    ax1.text(100 / R["M8b_prelabel"]["类别数"] + 0.5, -0.4, "随机基线 10.0%", fontsize=7.6, color="#666")
    ax1.set_xlabel("Top-1 召回率 %"); ax1.grid(axis="y", visible=False)
    ax1.set_title(f"心音品类识别（A）｜10 类 Top-1 召回（准确率 "
                  f"{100*R['M8b_prelabel']['Top1准确率']:.1f}%，基线 {100*R['M8b_prelabel']['多数类基线']:.1f}%）",
                  pad=10, fontweight="bold", fontsize=11)
    ax2.plot(pre["置信度阈值"], pre["自动通过率_pct"], "o-", color=PALETTE[0], lw=2.2, label="自动通过率")
    ax2.plot(pre["置信度阈值"], pre["通过样本正确率_pct"], "s--", color=ACCENT["bad"], lw=2.2,
             label="通过样本正确率")
    ax2.axhline(85, ls=":", color=ACCENT["good"], lw=1.4, label="可用门槛 85%")
    for _, r in pre.iterrows():
        ax2.annotate(f"{r['通过样本正确率_pct']:.0f}%", (r["置信度阈值"], r["通过样本正确率_pct"]),
                     textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7.6, color="#555")
    style_ax(ax2, "预标注可行性（B）｜置信度阈值扫描", xlabel="置信度阈值", ylabel="百分比 %")
    ax2.set_ylim(0, 92); ax2.set_xlim(0.28, 0.62)
    ax2.legend(fontsize=8.4, loc="center right")
    footer(fig, f"结论：{R['M8b_prelabel']['结论']['判定']}——最高通过正确率仅 "
                f"{R['M8b_prelabel']['结论']['最高通过正确率_pct']}%，改为 Top-3 候选提示"
                f"（命中率 {R['M8b_prelabel']['top3命中率_pct']}%，Top-5 {R['M8b_prelabel']['top5命中率_pct']}%）")
    finish(fig, "fig27_预标注模型_品类识别与阈值扫描.png")


def main():
    print("业务模型类图表：")
    for f in [fig19, fig20, fig21, fig22, fig23, fig24, fig25, fig26, fig27]:
        f()
    print("完成，共 9 张 →", cfg.FIG)


if __name__ == "__main__":
    main()


