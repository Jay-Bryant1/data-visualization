# -*- coding: utf-8 -*-
"""s05_figures.py —— 模板复现类图表 fig01–fig18。

对照《Excel数据可视化——从图表到数据大屏》的典型模板，用 Matplotlib/Seaborn 忠实复现：
图表类型、布局结构、配色、标签样式、图例位置、标题设计均按模板要点还原。
统一输出 150dpi PNG，命名规则 figNN_中文主题.png。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cfg
from plotstyle import PALETTE, ACCENT, load, tbl, finish, style_ax, footer, cbar_format

A, L, B, R = load()
MOD = A[A["in_model"] == 1].copy()
HEART = MOD[MOD["cohort"].isin(["heart_pure", "mixed"])].copy()
LUNG = MOD[MOD["cohort"].isin(["lung_pure", "mixed"])].copy()


def fig01():
    c = HEART.groupby("heart_type_zh")["cohort"].value_counts().unstack(fill_value=0)
    c = c.reindex(columns=["heart_pure", "mixed"]).fillna(0)
    c = c.loc[c.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(c))
    ax.bar(x - 0.2, c["heart_pure"], 0.4, label="纯心音队列", color=PALETTE[0])
    ax.bar(x + 0.2, c["mixed"], 0.4, label="混合队列", color=PALETTE[3])
    for xi, (a, b) in zip(x, zip(c["heart_pure"], c["mixed"])):
        ax.text(xi - 0.2, a + 1, int(a), ha="center", fontsize=8)
        ax.text(xi + 0.2, b + 1, int(b), ha="center", fontsize=8)
    style_ax(ax, "簇状柱形图｜心音品类资产数量（纯音 vs 混合）", ylabel="资产条数")
    ax.set_xticks(x, c.index, rotation=28, ha="right")
    ax.legend(loc="upper right")
    footer(fig, "数据口径：344 条独立录音中的 251 条含心音标签；模板要点：同轴双系列并置、数值标签外置、类别按总量降序")
    finish(fig, "fig01_簇状柱形图_心音品类资产量.png")


def fig02():
    g = MOD.groupby(["modality_zh", "gender_zh"]).size().unstack(fill_value=0)
    for col in ["男", "女"]:
        if col not in g:
            g[col] = 0
    g = g[["男", "女"]].sort_values("男", ascending=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    y = np.arange(len(g))
    ax.barh(y, g["男"], 0.62, label="男", color=PALETTE[0])
    ax.barh(y, g["女"], 0.62, left=g["男"], label="女", color=PALETTE[4])
    for yi, (m, f) in zip(y, zip(g["男"], g["女"])):
        ax.text(m / 2, yi, f"{m:.0f}", ha="center", va="center", color="w", fontsize=9)
        ax.text(m + f / 2, yi, f"{f:.0f}", ha="center", va="center", color="w", fontsize=9)
        ax.text(m + f + 2, yi, f"合计 {m+f:.0f}", va="center", fontsize=8.5, color="#444")
    style_ax(ax, "堆积条形图｜模态 × 性别构成", xlabel="资产条数")
    ax.set_yticks(y, g.index)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    footer(fig, "模板要点：水平堆积、段内反白数值、末段外置合计、去掉纵向网格干扰")
    finish(fig, "fig02_堆积条形图_模态性别构成.png")



def fig03():
    """百分比堆积柱：来源包 × 模态（5 类），体现三包的真实构成差异。"""
    ct = pd.crosstab(L["source_package"], L["modality_zh"])
    order = ["心音", "肺音", "心肺混合", "心音源轨", "肺音源轨"]
    ct = ct[[c for c in order if c in ct.columns]]
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    xs = [cfg.PACKAGE_ZH[i] for i in ct.index]
    bottom = np.zeros(len(ct))
    for i, col in enumerate(pct.columns):
        v = pct[col].values
        ax.bar(xs, v, 0.5, bottom=bottom, label=f"{col}（{int(ct[col].sum())}条）",
               color=PALETTE[i % len(PALETTE)])
        for xi, (b, vv) in enumerate(zip(bottom, v)):
            if vv > 5.5:
                ax.text(xi, b + vv / 2, f"{vv:.1f}%", ha="center", va="center",
                        color="white", fontsize=8.4, fontweight="bold")
        bottom += v
    style_ax(ax, "百分比堆积柱形图｜来源包的模态构成", ylabel="占比 %")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.6)
    footer(fig, "模板要点：统一柱宽、段内反白百分比、图例外置；HS/LS 为单模态包，Mix 为 1:1:1 三轨结构（各 145 条）")
    finish(fig, "fig03_百分比堆积柱_来源包模态构成.png")

def fig04():
    g = LUNG.groupby("lung_type_zh").agg(数量=("asset_id", "size"), 平均价值=("value_index", "mean"),
                                         平均成本=("total_cost", "mean")).sort_values("数量", ascending=False)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(g))
    ax.bar(x, g["数量"], 0.56, color=PALETTE[0], label="资产数量（左轴）")
    for xi, v in zip(x, g["数量"]):
        ax.text(xi, v + 2, int(v), ha="center", fontsize=8.5)
    ax2 = ax.twinx()
    ax2.plot(x, g["平均价值"], "o-", color=ACCENT["bad"], lw=2.2, ms=7, label="平均价值指数（右轴）")
    ax2.plot(x, g["平均成本"] * 10, "s--", color=ACCENT["good"], lw=1.8, ms=6,
             label="平均成本×10（右轴，元×10）")
    for xi, v in zip(x, g["平均价值"]):
        ax2.annotate(f"{v:.1f}", (xi, v), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8, color=ACCENT["bad"])
    style_ax(ax, "柱线组合图（双轴）｜肺音品类供给量与资产价值", ylabel="资产条数")
    ax2.set_ylabel("价值指数 / 成本×10")
    ax.set_xticks(x, g.index, rotation=22, ha="right")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3,
              fontsize=8, frameon=False)
    footer(fig, "模板要点：主次轴刻度分离、右轴系列用标记+虚线区分、避免柱线同色")
    finish(fig, "fig04_双轴柱线图_肺音品类供给与价值.png")


def fig05():
    g = MOD.groupby("batch_label").agg(平均质量分=("quality_score", "mean"),
                                       平均信息量=("info_score", "mean"),
                                       平均价值指数=("value_index", "mean"),
                                       平均成本=("total_cost", "mean")).sort_index()
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    markers = ["o", "s", "^", "D"]
    for i, col in enumerate(g.columns):
        ax.plot(g.index, g[col], marker=markers[i], lw=2.1, ms=6, color=PALETTE[i * 2 % len(PALETTE)], label=col)
    ax.axvspan(2.5, 3.5, color="#EEEEEE", alpha=0.6)
    ax.text(3, 78, "批次切换\n(跨包入库)", ha="center", fontsize=8, color="#777")
    style_ax(ax, "多系列折线图｜入库批次的资产质量与价值走势", ylabel="均值（分）")
    ax.legend(ncol=2, loc="lower right")
    footer(fig, f"模板要点：等距分类轴、标记点区分系列、异常区间底纹标注。选K规则与批次口径见 cfg.BATCH_SIZE={cfg.BATCH_SIZE}")
    finish(fig, "fig05_多折线图_批次质量与价值走势.png")


def fig06():
    blk = tbl("tbl_M6_series.csv")
    y = blk["y_原始dB"].values
    sd = (B[B["asset_id"].isin(MOD["asset_id"])].groupby("block_idx")["block_rms_db"].std().sort_index().values)
    t = blk["block_idx"] * cfg.BLOCK_MS / 1000
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.fill_between(t, y - sd, y + sd, color=PALETTE[0], alpha=0.22, label="±1 标准差区间")
    ax.plot(t, y, color=PALETTE[0], lw=1.6, label="均值能量包络")
    ax.fill_between(t, y, y - sd, color=PALETTE[0], alpha=0.30)
    ax.plot(t, y - sd, color=PALETTE[0], lw=0.8, ls=":")
    style_ax(ax, "面积图｜15 秒录音的平均能量包络（100ms 时间块）", xlabel="时间（秒）", ylabel="平均能量（dB）")
    ax.legend(loc="upper right")
    footer(fig, "模板要点：渐变式面积填充、边界细线、X 轴用连续时间而非序号")
    finish(fig, "fig06_面积图_能量包络.png")


def _donut(ax, values, names, colors, startangle=90, width=0.42,
           inside_count=True, outside_count=False):
    """环式饼图：百分比标签锁在环带中线（白色粗体），类别名走外侧并带引导线。

    原实现用 autopct 默认 pctdistance=0.6，落点正好压在环带内缘上——
    文字一半在圆心的白色孔洞里（白字白底不可见），一半被扇区边缘切掉。
    这里把百分比固定在环带径向中线上，外侧标签按 y 排序做纵向避让。
    """
    total = float(np.sum(values))
    wedges, _ = ax.pie(values, colors=colors, startangle=startangle,
                       counterclock=False, radius=1.0,
                       wedgeprops=dict(width=width, edgecolor="w", lw=1.6))
    r_mid = 1.0 - width / 2.0                 # 环带中线的半径
    for w, v in zip(wedges, values):
        ang = np.deg2rad((w.theta1 + w.theta2) / 2.0)
        pct = v / total * 100.0
        label = f"{pct:.1f}%" + (f"\n{v} 条" if inside_count else "")
        ax.text(r_mid * np.cos(ang), r_mid * np.sin(ang), label,
                ha="center", va="center", color="white", fontsize=8.8,
                fontweight="bold", linespacing=1.35, zorder=6)

    left, right = [], []
    for w, name, v in zip(wedges, names, values):
        ang = np.deg2rad((w.theta1 + w.theta2) / 2.0)
        txt = f"{name}\n{v} 条" if outside_count else name
        (right if np.cos(ang) >= 0 else left).append([ang, np.sin(ang), txt])

    gap = 0.20                                # 标签最小纵向间距（数据坐标）
    for group, sign in ((right, 1), (left, -1)):
        group.sort(key=lambda t: t[1])
        for i in range(1, len(group)):
            if group[i][1] - group[i - 1][1] < gap:
                group[i][1] = group[i - 1][1] + gap
        if group and group[-1][1] > 1.02:     # 越界则整体回压
            shift = group[-1][1] - 1.02
            for item in group:
                item[1] -= shift
        for ang, y, txt in group:
            ax.annotate(txt, xy=(1.02 * np.cos(ang), 1.02 * np.sin(ang)),
                        xytext=(1.22 * sign, y), textcoords="data",
                        ha="left" if sign > 0 else "right", va="center",
                        fontsize=9.2, color="#333333", linespacing=1.35,
                        arrowprops=dict(arrowstyle="-", color="#9A9A9A",
                                        lw=0.8, shrinkA=2, shrinkB=2))
    ax.set_xlim(-1.82, 1.82)
    ax.set_ylim(-1.22, 1.22)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.axis("off")
    # 收紧上下留白：饼图与底部说明之间不留空白带
    ax.figure.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04)


def fig07():
    s = L["source_package"].value_counts()
    zh = [cfg.PACKAGE_ZH[k] for k in s.index]
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    _donut(ax, s.values, zh, PALETTE[:len(s)], startangle=90, width=0.42,
           inside_count=True, outside_count=False)
    ax.text(0, 0, f"入库资产\n{s.sum()} 条", ha="center", va="center",
            fontsize=11, fontweight="bold", linespacing=1.4)
    ax.set_title("圆环饼图｜来源包资产构成", pad=10, fontweight="bold")
    footer(fig, "模板要点：环式饼图 + 圆心汇总指标 + 内外双标签（百分比与绝对量）")
    finish(fig, "fig07_圆环饼图_来源包资产构成.png")


def fig08():
    s = MOD["cohort_zh"].value_counts()
    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    _donut(ax, s.values, s.index, [PALETTE[0], PALETTE[1], PALETTE[3]],
           startangle=110, width=0.45, inside_count=False, outside_count=True)
    ax.text(0, 0, f"独立录音\n{s.sum()} 条", ha="center", va="center",
            fontsize=10.5, fontweight="bold", linespacing=1.4)
    ax.set_title("环形图｜去重后的队列结构（重做版三分口径）", pad=10, fontweight="bold")
    footer(fig, "口径要点：heart_pure + lung_pure + mixed = 344 = sha256 独立录音数，可与去重结果逐条对账")
    finish(fig, "fig08_环形图_队列结构.png")


def fig09():
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for i, (coh, sub) in enumerate(MOD.groupby("cohort")):
        ax.scatter(sub["quality_score"], sub["info_score"], s=26, alpha=0.72,
                   color=PALETTE[i * 2 % len(PALETTE)], edgecolor="w", lw=0.5,
                   label=f"{cfg.MODALITY_ZH.get(coh, coh)} n={len(sub)}")
    k, b = np.polyfit(MOD["quality_score"], MOD["info_score"], 1)
    xs = np.linspace(MOD["quality_score"].min(), MOD["quality_score"].max(), 50)
    ax.plot(xs, k * xs + b, ls="--", color="#333", lw=1.6, label=f"OLS 拟合 y={k:.2f}x+{b:.1f}")
    r = float(MOD["quality_score"].corr(MOD["info_score"]))
    ax.text(0.03, 0.96, f"Pearson r = {r:.3f}", transform=ax.transAxes, va="top", fontsize=9.5,
            bbox=dict(fc="white", ec="#BBB", alpha=0.9))
    style_ax(ax, "散点图 + 趋势线｜质量分与信息量", xlabel="质量分（0–100）", ylabel="信息量分（0–100）")
    ax.legend(loc="lower right", fontsize=8.5)
    footer(fig, "模板要点：分系列着色、白描边避免重叠、相关系数嵌入左上角说明框")
    finish(fig, "fig09_散点图_质量分与信息量.png")



def fig10():
    """气泡图：性价比 × 价值指数 × 稀缺度，ABC 分层着色。"""
    d = MOD.copy()
    d["roi"] = (d["value_index"] / d["total_cost"]).round(3)
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    for i, (lab, sub) in enumerate(d.groupby("abc")):
        ax.scatter(sub["roi"], sub["value_index"], s=sub["scarcity_score"] * 3.2 + 14,
                   c=[{0: ACCENT["bad"], 1: ACCENT["warn"], 2: ACCENT["navy"]}[i]],
                   alpha=0.6, edgecolor="white", lw=0.6, label=f"{lab} 类 n={len(sub)}")
    ax.axhline(float(d["value_index"].median()), color="#888", ls=":", lw=1.2)
    ax.axvline(float(d["roi"].median()), color="#888", ls=":", lw=1.2)
    ax.text(0.015, 0.965, "① 低性价比·高价值：压缩成本", transform=ax.transAxes, fontsize=8.4, color="#C0504D")
    ax.text(0.62, 0.965, "② 高性价比·高价值：加大投产", transform=ax.transAxes, fontsize=8.4, color="#2E8B8B")
    ax.text(0.015, 0.045, "③ 低性价比·低价值：淘汰候选", transform=ax.transAxes, fontsize=8.4, color="#666")
    ax.text(0.62, 0.045, "④ 高性价比·低价值：复采增值", transform=ax.transAxes, fontsize=8.4, color="#1F4E79")
    style_ax(ax, "气泡图｜单位成本价值比 × 价值指数（气泡=稀缺度，颜色=ABC 分层）",
             xlabel="价值成本比（价值指数 / 全成本，元）", ylabel="资产价值指数")
    ax.legend(loc="center right")
    footer(fig, "模板要点：气泡面积线性映射第三维、中位数十字线切四象限并在象限内直接写运营动作")
    finish(fig, "fig10_气泡图_价值成本比与价值指数.png")

def fig11():
    cols = ["quality_score", "info_score", "scarcity_score", "value_index", "total_cost",
            "snr_db", "entropy", "centroid", "bandwidth", "zcr_mean", "heart_rate_bpm",
            "silence_ratio", "dynamic_range_db", "crest_factor"]
    C = MOD[cols].corr(method="pearson")
    mask = np.triu(np.ones_like(C, dtype=bool))
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    sns.heatmap(C, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                center=0, square=True, linewidths=0.6, linecolor="white",
                cbar_kws={"label": "Pearson 相关系数"}, ax=ax,
                annot_kws={"fontsize": 7.6})
    ax.set_title("热力图｜核心指标相关矩阵（下三角）", pad=14, fontweight="bold")
    strong = [(a, b, round(float(C.loc[a, b]), 3)) for a in cols for b in cols
              if a < b and abs(C.loc[a, b]) >= 0.8]
    ax.set_xlabel("")
    footer(fig, "模板要点：对角空白（|r|=1 无信息）、双向色标以 0 为中心、强相关对（|r|≥0.8）另表登记"
                + f"；本矩阵强相关对 {len(strong)} 组")
    finish(fig, "fig11_热力图_核心指标相关矩阵.png")


def fig12():
    order = LUNG.groupby("lung_type_zh")["entropy"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    data = [LUNG.loc[LUNG["lung_type_zh"] == k, "entropy"].dropna().values for k in order]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops=dict(color="#C0504D", lw=1.8),
                    whiskerprops=dict(color="#666"), capprops=dict(color="#666"),
                    flierprops=dict(marker="o", ms=3.5, mfc="#999", mec="none"))
    for i, p in enumerate(bp["boxes"]):
        p.set_facecolor(PALETTE[i % len(PALETTE)]); p.set_alpha(0.72); p.set_edgecolor("#555")
    for i, d in enumerate(data):
        ax.scatter(np.random.RandomState(cfg.SEED + i).normal(i + 1, 0.055, len(d)), d, s=7,
                   color="#333", alpha=0.42, zorder=3)
    ax.set_xticks(range(1, len(order) + 1), order, rotation=18, ha="right")
    style_ax(ax, "箱线图 + 抖动散点｜各肺音品类的谱熵分布", ylabel="谱熵（归一化）")
    footer(fig, "模板要点：箱体填色 + 中位红粗线 + 原始散点抖动显示样本量，避免只看中位数误判")
    finish(fig, "fig12_箱线图_肺音品类谱熵分布.png")


def fig13():
    cats = ["正常心音", "心房颤动", "心动过速", "第三心音S3", "收缩中期杂音"]
    sub = HEART[HEART["heart_type_zh"].isin(cats)]
    data = [sub.loc[sub["heart_type_zh"] == c, "heart_rate_bpm"].dropna().values for c in cats]
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    vp = ax.violinplot(data, showmeans=True, showextrema=False, widths=0.86)
    for i, b in enumerate(vp["bodies"]):
        b.set_facecolor(PALETTE[i % len(PALETTE)]); b.set_alpha(0.72); b.set_edgecolor("#555")
    for i, d in enumerate(data):
        q = np.percentile(d, [25, 50, 75])
        ax.plot([i + 1] * 3, q, "-", color="#C0504D", lw=2.4, alpha=0.9, solid_capstyle="round")
        ax.scatter(np.full(len(d), i + 1) + np.random.RandomState(i).uniform(-0.03, 0.03, len(d)),
                   d, s=5, color="#222", alpha=0.35)
    ax.axhspan(60, 100, color="#2E8B8B", alpha=0.07)
    ax.text(5.45, 80, "临床参考\n正常心率\n60–100", fontsize=7.8, color="#2E8B8B", va="center")
    ax.set_xticks(range(1, len(cats) + 1), cats, rotation=14, ha="right")
    style_ax(ax, "小提琴图｜典型心音品类的心率分布", ylabel="估计心率（bpm）")
    footer(fig, f"心率提取：帧能量序列自相关在 {cfg.HR_LAG_SEC[0]}–{cfg.HR_LAG_SEC[1]}s 搜索峰值 → {cfg.HR_LAG_SEC[0]*60:.0f}–{cfg.HR_LAG_SEC[1]*60:.0f} bpm")
    finish(fig, "fig13_小提琴图_心音品类心率分布.png")


def fig14():
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    v = MOD["value_index"].dropna()
    ax.hist(v, bins=24, density=True, color=PALETTE[0], alpha=0.55, edgecolor="white", label="直方图（24 bin）")
    xs = np.linspace(v.min(), v.max(), 300)
    bw = 1.06 * float(np.nanstd(v)) * len(v) ** (-0.2)
    kde = np.mean([np.exp(-0.5 * ((xs - x) / bw) ** 2) for x in v], axis=0) / (bw * np.sqrt(2 * np.pi))
    ax.plot(xs, kde, color=ACCENT["bad"], lw=2.2, label=f"KDE 核密度（bw={bw:.2f}）")
    for lab, col in [("均值", ACCENT["warn"]), ("中位数", ACCENT["good"])]:
        x0 = float(v.mean()) if lab == "均值" else float(v.median())
        ax.axvline(x0, ls="--", color=col, lw=1.5, label=f"{lab} {x0:.1f}")
    style_ax(ax, "直方图 + 核密度｜资产价值指数分布", xlabel="价值指数", ylabel="概率密度")
    ax.legend(loc="upper left", fontsize=8.5)
    footer(fig, f"偏度 {float(v.skew()):.3f}、峰度 {float(pd.Series(v).kurt()):.3f}；按 quality_thresholds 判定为"
                + ("轻微偏态" if abs(float(v.skew())) < 0.5 else "中度偏态"))
    finish(fig, "fig14_直方图核密度_资产价值指数.png")



def fig15():
    """瀑布图：从 535 条入库到 210 条 A 类的逐级净流失（严格按 tbl_M7_funnel）。"""
    f = tbl("tbl_M7_funnel.csv")
    names = ["原始\n入库", "参数\n合规", "标签\n关联", "内容\n去重", "质检\n达标",
             "A+B\n类", "A\n类"]
    vals = f["数量"].tolist()
    steps = [vals[0]] + [vals[i] - vals[i - 1] for i in range(1, len(vals))]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    cum = 0
    for i, (nm, st) in enumerate(zip(names, steps)):
        if i == 0:
            ax.bar(i, st, 0.62, color=PALETTE[0])
            ax.text(i, st + 10, f"{st}", ha="center", fontsize=9, fontweight="bold")
            cum = st
        elif st == 0:
            ax.bar(i, cum, 0.62, color=PALETTE[0], alpha=0.55)
            ax.text(i, cum + 10, f"{cum}", ha="center", fontsize=9, fontweight="bold")
            ax.text(i, cum / 2, "无损耗", ha="center", va="center", color="white", fontsize=8)
        elif st < 0:
            ax.bar(i, st, 0.62, bottom=cum, color=ACCENT["bad"])
            ax.text(i, cum + st / 2, f"{st}", ha="center", va="center", color="white",
                    fontsize=9, fontweight="bold")
            ax.text(i, cum + 10, f"→{cum + st}", ha="center", fontsize=8.2, color="#555")
            cum += st
        else:
            ax.bar(i, st, 0.62, bottom=cum, color=ACCENT["good"])
            ax.text(i, cum + st / 2, f"+{st}", ha="center", va="center", color="white", fontsize=9)
            cum += st
        if i:
            ax.plot([i - 1 + 0.31, i - 0.31], [cum, cum], color="#999", lw=0.9, ls="--")
    ax.set_xticks(range(len(names)), names, fontsize=9)
    style_ax(ax, "瀑布图｜资产从入库到 A 类的逐级净流失", ylabel="资产条数")
    ax.set_ylim(0, 600)
    footer(fig, f"最大损耗环节：内容去重 −{vals[0]-vals[3]} 条（{(vals[0]-vals[3])/vals[0]*100:.1f}%），"
                f"对应浪费成本 {R['M7_funnel']['去重节省成本_元']:.2f} 元")
    finish(fig, "fig15_瀑布图_资产去重与分层净流失.png")

def fig16():
    prof = tbl("tbl_M4_cluster_profile.csv")
    metrics = ["价值指数", "质量分", "信息量", "稀缺度", "心音异常率"]
    X = prof[metrics].values.astype(float)
    Xn = (X - X.min(0)) / (np.ptp(X, axis=0) + 1e-9)
    ang = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(7.6, 6.4), subplot_kw=dict(polar=True))
    for i, row in enumerate(Xn):
        rr = np.concatenate([row, row[:1]])
        ax.plot(ang, rr, lw=2, color=PALETTE[i % len(PALETTE)], label=f"{prof.iloc[i]['cluster']} {prof.iloc[i]['画像命名']}")
        ax.fill(ang, rr, color=PALETTE[i % len(PALETTE)], alpha=0.10)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels([f"{m}\n(原始均值 {prof[m].mean():.1f})" for m in metrics], fontsize=8.5)
    ax.set_yticklabels([])
    ax.set_title("雷达图｜K-Means 分群画像对比（各维归一化）", pad=22, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.12, 0.5), fontsize=8.5)
    footer(fig, f"K={R['M4_kmeans']['最优K']}；选 K 规则：{R['M4_kmeans']['选K规则']}")
    finish(fig, "fig16_雷达图_KMeans分群画像.png")



def fig17():
    """树形图：列宽=队列资产量，行高=队列内品类占比 → 矩形面积严格等于资产条数。"""
    g = (MOD.groupby(["cohort_zh", "category"]).size().rename("n").reset_index()
         .sort_values("n", ascending=False))
    total = int(g["n"].sum())
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    coh_tot = g.groupby("cohort_zh")["n"].sum().sort_values(ascending=False)
    x0 = 0.0
    for ci, (coh, cw_n) in enumerate(coh_tot.items()):
        cw = 100 * cw_n / total
        sub = g[g["cohort_zh"] == coh].copy()
        sub["hh"] = 100 * sub["n"] / cw_n
        yv = 0.0
        for _, r in sub.iterrows():
            hh = float(r["hh"])
            ax.add_patch(Rectangle((x0, yv), cw, max(hh - 0.28, 0.1),
                                   facecolor=PALETTE[ci % len(PALETTE)],
                                   alpha=0.42 + 0.46 * min(float(r["n"]) / max(sub["n"].max(), 1), 1.0),
                                   edgecolor="white", lw=1.0))
            if hh > 4.2 and cw > 6.5:
                ax.text(x0 + cw / 2, yv + hh / 2, f"{r['category']}\n{int(r['n'])}",
                        ha="center", va="center", fontsize=7.2)
            yv += hh
        ax.text(x0 + cw / 2, -3.0, f"{coh}\n{int(cw_n)} 条 / {100*cw_n/total:.1f}%", ha="center",
                va="top", fontsize=8.4, fontweight="bold", color="#333")
        x0 += cw
    ax.set_xlim(0, 100)
    ax.set_ylim(-14, 100)
    ax.axis("off")
    ax.set_title("树形图｜矩形面积 = 资产条数（列=队列，行=品类）", pad=12, fontweight="bold")
    footer(fig, "模板要点：两级层次面积编码、色相区分一级节点、同组内按数量降序、面积过小的格子省略标签")
    finish(fig, "fig17_树形图_队列与品类结构.png")

def fig18():
    ct = pd.crosstab(HEART["location_zh"], HEART["heart_type_zh"])
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index,
                ct.sum(axis=0).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    for i, yl in enumerate(ct.index):
        for j, xl in enumerate(ct.columns):
            v = int(ct.loc[yl, xl])
            r = np.sqrt(v / max(int(ct.values.max()), 1))
            ax.add_patch(plt.Circle((j, i), 0.44 * r + 0.05, color=PALETTE[min(int(r * 4), 4)],
                                     alpha=0.85 if v else 0.15))
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                        color="white" if v >= 4 else "#333")
    ax.set_xticks(range(len(ct.columns)), ct.columns, rotation=30, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(ct.index)), ct.index, fontsize=9)
    ax.set_xlim(-0.7, len(ct.columns) - 0.3); ax.set_ylim(-0.7, len(ct.index) - 0.3)
    ax.grid(False)
    ax.set_aspect("equal")
    style_ax(ax, "矩阵气泡图｜心音听诊点位 × 心音品类共现数量", ylabel="听诊点位", xlabel="心音品类")
    footer(fig, "模板要点：气泡面积=√数量（面积感知线性）、零值保留空心位、行列表头按总量排序")
    finish(fig, "fig18_矩阵气泡图_听诊点位与心音品类.png")


def main():
    print("模板复现类图表：")
    for f in [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08,
              fig09, fig10, fig11, fig12, fig13, fig14, fig15, fig16, fig17, fig18]:
        f()
    print("完成，共 18 张 →", cfg.FIG)


if __name__ == "__main__":
    main()


