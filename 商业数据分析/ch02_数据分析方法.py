# -*- coding: utf-8 -*-
r"""
《商业数据分析》教材重构 —— 第2章 数据分析方法
================================================================================
原教材实现: Excel / Power BI (含 12 个分析范式)
本文件实现: Python 单文件版 (pandas + numpy + matplotlib), 可直接运行
================================================================================

运行::

    C:\Users\dell\anaconda3\envs\dataviz\python.exe -X utf8 ch02_数据分析方法.py

输出: 与脚本同目录的 PNG 图片 (300dpi)

本章的 12 种分析方法
--------------------
基础范式 (6)
  1. 波士顿矩阵   增长-份额二维矩阵, 定位"明星/金牛/问题/瘦狗"
  2. RFM 模型     最近消费 / 消费频次 / 消费金额, 做客户分层
  3. 用户忠诚度   用 R 与 F 划分"潜在用户"与"易流失用户"
  4. 同期群分析   按注册月份分组, 追踪留存衰减
  5. 漏斗图       逐层转化率, 定位流失环节
  6. 相关分析     两个连续变量的关系强度与方向

引申方法 (6)
  7. 趋势分析     时间序列的水平、波动与预警
  8. 对比分析     同类对象横向对比, 含排序与雷达图
  9. 构成分析     整体中各部分的占比与结构变化
 10. 分布分析     连续变量的集中趋势、离散程度与异常值
 11. 关系分析     分类变量与连续变量之间的关系
 12. 桑基图       流量在多级页面之间的流转路径

方法论上的共同点
----------------
每种方法都遵循"业务问题 -> 数据口径 -> 计算 -> 图形表达"的顺序;
Python 版本把 Excel 中散落在辅助列、数据透视表里的中间步骤显式写成代码,
使口径可复核、可复用。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent      # 本章交付目录
ROOT = HERE.parent                          # 项目根目录 (教材原文所在位置)
CH2 = ROOT / "第2章 数据分析方法"
BASE1 = CH2 / "1、分析的基础范式" / "data-无答案"
BASE2 = CH2 / "2、由基础分析范式引申出的分析方法" / "data-无答案"
FIG = HERE
FIG.mkdir(parents=True, exist_ok=True)

if sys.platform == "win32":
    _lib_bin = Path(sys.prefix) / "Library" / "bin"
    if _lib_bin.is_dir() and str(_lib_bin).lower() not in os.environ.get("PATH", "").lower():
        os.environ["PATH"] = f"{_lib_bin}{os.pathsep}{os.environ.get('PATH', '')}"


def rule(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ============================================================================ #
# 1. 绘图主题
# ============================================================================ #

PALETTE = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1",
           "#76B7B2", "#EDC948", "#9C755F", "#BAB0AC", "#FF9DA7"]
COLOR = {"primary": "#4E79A7", "accent": "#F28E2B", "positive": "#59A14F",
         "negative": "#E15759", "neutral": "#BAB0AC", "highlight": "#B07AA1",
         "muted": "#8FAFC9", "grid": "#E4E7EB", "text": "#2F3B45",
         "text_soft": "#7A8794", "bg": "#FFFFFF", "bg_soft": "#F7F9FB"}

SEQ_CMAP = LinearSegmentedColormap.from_list(
    "biz_seq", ["#EAF1F8", "#A9C4DF", "#6E97C2", "#4E79A7", "#2C4A66"])
matplotlib.colormaps.register(SEQ_CMAP, force=True)

_CJK = ["Microsoft YaHei", "微软雅黑", "Source Han Sans SC", "Noto Sans CJK SC",
        "SimHei", "黑体", "DengXian", "SimSun"]


def cjk_font_name(fallback="DejaVu Sans") -> str:
    try:
        from matplotlib import font_manager
        have = {f.name for f in font_manager.fontManager.ttflist}
        for n in _CJK:
            if n in have:
                return n
    except Exception:
        pass
    return fallback


def setup_style(dpi=300, figsize=(11, 6.2)) -> str:
    font = cjk_font_name()
    plt.rcParams.update({
        "font.sans-serif": [font, "DejaVu Sans"], "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "figure.dpi": 110, "savefig.dpi": dpi, "figure.figsize": figsize,
        "figure.facecolor": COLOR["bg"], "savefig.facecolor": COLOR["bg"],
        "savefig.bbox": "tight", "savefig.pad_inches": 0.18,
        "font.size": 11, "axes.titlesize": 15, "axes.titleweight": "bold",
        "axes.labelsize": 12, "axes.labelcolor": COLOR["text"],
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10.5,
        "figure.titlesize": 17, "figure.titleweight": "bold",
        "axes.edgecolor": COLOR["grid"], "axes.linewidth": 1.0, "axes.labelpad": 8,
        "axes.titlepad": 12, "axes.titlelocation": "left",
        "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "grid.color": COLOR["grid"], "grid.linewidth": 0.8, "grid.alpha": 0.9,
        "xtick.color": COLOR["text_soft"], "ytick.color": COLOR["text_soft"],
        "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.handlelength": 1.4,
        "legend.borderaxespad": 0.0,
        "lines.linewidth": 2.2, "lines.markersize": 6, "patch.linewidth": 0,
        "image.cmap": "biz_seq",
    })
    return font


def style_axes_clean(ax) -> None:
    for s in ("left", "bottom"):
        ax.spines[s].set_color(COLOR["grid"])


def pct(v, _=None) -> str:
    return f"{v:.0%}"


def wan(v, _=None) -> str:
    v = float(v)
    if abs(v) >= 1e8:
        return f"{v / 1e8:,.1f}亿"
    if abs(v) >= 1e4:
        return f"{v / 1e4:,.0f}万"
    return f"{v:,.0f}"


def annotate_bars(ax, values, bars=None, fmt="{:,.0f}", offset=0.02,
                  fontsize=10, color=None):
    bars = list(bars) if bars is not None else list(ax.patches)
    values = [float(v) for v in values]
    finite = [v for v in values if np.isfinite(v)] or [0.0]
    span = (max(finite) - min(finite + [0.0])) or 1.0
    for b, v in zip(bars, values):
        if not np.isfinite(v):
            continue
        ax.text(b.get_x() + b.get_width() / 2, v + span * offset, fmt.format(v),
                ha="center", va="bottom", fontsize=fontsize, fontweight="bold",
                color=color or COLOR["text_soft"], clip_on=False)


def annotate_hbars(ax, values, fmt="{:,.0f}", pad=0.015, fontsize=10):
    values = [float(v) for v in values]
    finite = [v for v in values if np.isfinite(v)] or [0.0]
    span = (max(finite) - min(finite + [0.0])) or 1.0
    for b, v in zip(ax.patches[-len(values):], values):
        if not np.isfinite(v):
            continue
        ax.text(v + span * pad, b.get_y() + b.get_height() / 2, fmt.format(v),
                va="center", ha="left", fontsize=fontsize, fontweight="bold",
                color=COLOR["text"])


def add_source(fig, text, y=0.0):
    fig.text(0.005, y, text, fontsize=9, color=COLOR["text_soft"],
             ha="left", va="top")


def save_fig(fig, name):
    p = FIG / f"{name}.png"
    fig.savefig(p, format="png")
    plt.close(fig)
    return p


def note_box(ax, x, y, text, face=COLOR["bg_soft"], edge=COLOR["grid"],
             ha="left", va="top", fontsize=10):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize,
            color=COLOR["text"], linespacing=1.7,
            bbox=dict(boxstyle="round,pad=0.45", fc=face, ec=edge, lw=1.0))
# ============================================================================ #
# 2. 基础范式
# ============================================================================ #

def m_1_bcg() -> pd.DataFrame:
    """① 波士顿矩阵: 销售增长率 x 相对市场份额, 气泡 = 销售额。"""
    rule("① 波士顿矩阵")
    df = pd.read_excel(BASE1 / "1-1波士顿矩阵.xlsx", sheet_name="销售表")
    df["销售额"] = df["单价"] * df["销量"]
    df["月"] = df["销售日期"].dt.month

    mon = df.pivot_table(index="品牌", columns="月", values="销售额", aggfunc="sum")
    mon = mon.reindex(columns=[7, 8], fill_value=0)
    growth = (mon[8] - mon[7]) / mon[7].replace(0, np.nan)
    share = mon.sum(axis=1) / mon.sum(axis=1).sum()

    # 相对市场份额 = 自身份额 / 最大竞争对手份额
    rows = []
    for b in share.index:
        rivals = share.drop(b)
        rows.append(share[b] / rivals.max() if rivals.max() > 0 else np.nan)
    rel = pd.Series(rows, index=share.index)

    bcg = pd.DataFrame({"销售额": mon.sum(axis=1), "市场增长率": growth,
                        "相对市场份额": rel})
    bcg = bcg.dropna(subset=["市场增长率", "相对市场份额"]).sort_values("销售额")
    avg_growth = (mon[8].sum() - mon[7].sum()) / mon[7].sum()

    def label(r):
        high_g, high_s = r["市场增长率"] >= avg_growth, r["相对市场份额"] >= 1
        if high_g and high_s:
            return "明星"
        if high_g and not high_s:
            return "问题"
        if not high_g and high_s:
            return "金牛"
        return "瘦狗"

    bcg["类型"] = bcg.apply(label, axis=1)
    print(bcg.round(4).to_string())
    print(f"\n市场整体 7->8 月增长率: {avg_growth:+.2%} (四象限的纵轴分界)")

    fig, ax = plt.subplots(figsize=(11.5, 7.6))
    cmap = {"明星": COLOR["positive"], "金牛": COLOR["primary"],
            "问题": COLOR["accent"], "瘦狗": COLOR["negative"]}
    size = (bcg["销售额"] / bcg["销售额"].max()) * 2200 + 120
    for t, sub in bcg.groupby("类型"):
        ax.scatter(sub["相对市场份额"], sub["市场增长率"],
                   s=(sub["销售额"] / bcg["销售额"].max()) * 2200 + 120,
                   color=cmap[t], alpha=0.75, edgecolors="white", linewidths=1.6,
                   label=f"{t} ({len(sub)})", zorder=3)
    for b, r in bcg.iterrows():
        ax.annotate(b, (r["相对市场份额"], r["市场增长率"]),
                    textcoords="offset points", xytext=(0, 15), ha="center",
                    fontsize=10.5, fontweight="bold", color=COLOR["text"])

    ax.axvline(1.0, color=COLOR["text_soft"], lw=1.2, ls="--")
    ax.axhline(avg_growth, color=COLOR["text_soft"], lw=1.2, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("相对市场份额 (对数轴, 1.0 = 与最大竞争对手持平)")
    ax.set_ylabel("7→8 月销售增长率")
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    # 气泡会占满左上角, 因此图例放到坐标区上方, 并留出上下余量
    g_lo, g_hi = bcg["市场增长率"].min(), bcg["市场增长率"].max()
    ax.set_ylim(g_lo - (g_hi - g_lo) * 0.22 - 0.005, g_hi + (g_hi - g_lo) * 0.22 + 0.005)
    ax.set_title("波士顿矩阵: 14 个品牌的增长-份额定位", pad=44)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=4,
              columnspacing=1.6)
    for txt, x, y, ha, va in [("明星", 0.97, 0.96, "right", "top"),
                              ("问题", 0.03, 0.96, "left", "top"),
                              ("金牛", 0.97, 0.04, "right", "bottom"),
                              ("瘦狗", 0.03, 0.04, "left", "bottom")]:
        ax.text(x, y, txt, transform=ax.transAxes, fontsize=13, alpha=0.28,
                fontweight="bold", ha=ha, va=va, color=cmap[txt])
    style_axes_clean(ax)
    add_source(fig, "数据来源: 1-1波士顿矩阵.xlsx 销售表 (9,284 条, 2021-07/08)  |  "
                    "气泡面积 = 品牌销售额", y=0.005)
    save_fig(fig, "ch02_01_波士顿矩阵")
    return bcg


def m_2_rfm() -> pd.DataFrame:
    """② RFM 模型: 用户分层的 8 类客户。"""
    rule("② RFM 模型")
    df = pd.read_excel(BASE1 / "1-2RFM模型.xlsx", sheet_name="原始数据")
    df["销售日期"] = pd.to_datetime(df["销售日期"])
    snap = df["销售日期"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("用户编号").agg(
        最近消费=("销售日期", "max"),
        消费频次=("流水号", "nunique"),
        消费金额=("销售金额", "sum")).reset_index()
    rfm["R"] = (snap - rfm["最近消费"]).dt.days
    rfm.rename(columns={"消费频次": "F", "消费金额": "M"}, inplace=True)

    def score(s, asc=True):
        return pd.qcut(s.rank(method="first", ascending=asc), 5,
                       labels=[1, 2, 3, 4, 5]).astype(int)

    rfm["R分"] = score(rfm["R"], asc=False)   # 最近消费越近分越高
    rfm["F分"] = score(rfm["F"])
    rfm["M分"] = score(rfm["M"])

    def seg(r):
        hi_f, hi_m = r["F分"] >= 3, r["M分"] >= 3
        if r["R分"] >= 3:
            return ("重要价值" if hi_f and hi_m else
                    "重要发展" if hi_f else "重要保持" if hi_m else "重要挽留")
        return ("一般价值" if hi_f and hi_m else
                "一般发展" if hi_f else "一般保持" if hi_m else "一般挽留")

    rfm["客户类型"] = rfm.apply(seg, axis=1)
    stat = (rfm.groupby("客户类型")
            .agg(客户数=("用户编号", "size"), 平均R=("R", "mean"),
                 平均F=("F", "mean"), 平均M=("M", "mean"),
                 总金额=("M", "sum"))
            .sort_values("客户数", ascending=False))
    stat["人数占比"] = stat["客户数"] / stat["客户数"].sum()
    stat["金额占比"] = stat["总金额"] / stat["总金额"].sum()
    print(stat.round(2).to_string())
    print(f"\n用户数 {len(rfm):,}, 快照日 {snap.date()}, 人均消费 {rfm['M'].mean():,.1f} 元")

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.4), width_ratios=[0.95, 1.15],
                             gridspec_kw={"wspace": 0.30})
    ax = axes[0]
    order = stat.sort_values("客户数")
    ax.barh(order.index, order["客户数"], height=0.62,
            color=[COLOR["accent"] if i.startswith("重要") else COLOR["muted"]
                   for i in order.index])
    annotate_hbars(ax, order["客户数"].values, fmt="{:,.0f}", pad=0.012)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{name}\n金额占比 {r['金额占比']:.0%}"
                        for name, r in order.iterrows()], fontsize=9.5)
    ax.set_xlim(0, order["客户数"].max() * 1.24)
    ax.set_title("① 八类客户的规模与金额贡献")
    ax.set_xlabel("客户数")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    style_axes_clean(ax)

    ax = axes[1]
    seg_colors = {"重要价值": COLOR["positive"], "重要发展": COLOR["accent"],
                  "重要保持": COLOR["primary"], "重要挽留": COLOR["highlight"],
                  "一般价值": COLOR["muted"], "一般发展": COLOR["neutral"],
                  "一般保持": COLOR["text_soft"], "一般挽留": COLOR["negative"]}
    for t, sub in rfm.groupby("客户类型"):
        ax.scatter(sub["R"], sub["F"], s=np.clip(sub["M"] / 40, 8, 260),
                   alpha=0.45, color=seg_colors.get(t, COLOR["neutral"]),
                   label=t, edgecolors="none")
    ax.set_xlabel("R: 距最近一次消费的天数 (越小越好)")
    ax.set_ylabel("F: 消费频次 (单)")
    ax.set_title("② R-F 分布 (气泡 = 消费金额)")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    style_axes_clean(ax)

    fig.suptitle("RFM 模型: 用 3 个指标把 7,178 名用户分成 8 类",
                 y=1.02, x=0.02, ha="left")
    add_source(fig, "数据来源: 1-2RFM模型.xlsx 原始数据 (65,989 行明细 / 26,735 张流水)  |  "
                    "R/F/M 各按五分位打分, 以 3 分为高/低分界", y=0.005)
    save_fig(fig, "ch02_02_RFM模型")
    return rfm

def m_3_loyalty() -> pd.DataFrame:
    """③ 用户忠诚度: 用 R 与 F 圈出"潜在用户"与"易流失用户"。"""
    rule("③ 用户忠诚度模型")
    df = pd.read_excel(BASE1 / "1-3用户忠诚度模型.xlsx", sheet_name="销售流水表")
    df["消费日期"] = pd.to_datetime(df["消费日期"])
    snap = df["消费日期"].max() + pd.Timedelta(days=1)
    u = df.groupby("会员编号").agg(
        最近消费=("消费日期", "max"), 订单数=("订单编号", "nunique"),
        消费金额=("消费金额", "sum")).reset_index()
    u["R"] = (snap - u["最近消费"]).dt.days

    r_med, f_med = u["R"].median(), u["订单数"].median()
    u["类型"] = np.where((u["R"] <= r_med) & (u["订单数"] <= f_med), "潜在用户",
                         np.where((u["R"] > r_med) & (u["订单数"] > f_med),
                                  "易流失用户", "普通用户"))
    print(u["类型"].value_counts().to_string())
    print(f"\nR 中位数 {r_med:.0f} 天, F 中位数 {f_med:.0f} 单, 会员 {len(u):,} 人")

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {"潜在用户": COLOR["positive"], "易流失用户": COLOR["negative"],
              "普通用户": COLOR["muted"]}
    for t, sub in u.groupby("类型"):
        ax.scatter(sub["R"], sub["订单数"], s=46, alpha=0.55, color=colors[t],
                   label=f"{t} ({len(sub)})", edgecolors="white", linewidths=0.4)
    ax.axvline(r_med, color=COLOR["text_soft"], ls="--", lw=1.2)
    ax.axhline(f_med, color=COLOR["text_soft"], ls="--", lw=1.2)
    ax.set_xlabel("R: 距最近一次消费的天数")
    ax.set_ylabel("F: 累计订单数")
    ax.set_title("用户忠诚度分层: R-F 四象限")
    ax.legend(loc="upper right")
    ax.set_ylim(0, u["订单数"].max() * 1.12)
    note_box(ax, 0.02, 0.97, "左上 = 高频但久未消费 → 易流失, 优先召回",
             face="#FDECEC", edge=COLOR["negative"], fontsize=10.5)
    note_box(ax, 0.02, 0.06, "左下 = 近期且低频 → 潜在用户, 重点培育",
             face="#EDF7EC", edge=COLOR["positive"], va="bottom", fontsize=10.5)
    style_axes_clean(ax)
    add_source(fig, "数据来源: 1-3用户忠诚度模型.xlsx 销售流水表 (80,111 行 / 5,001 单 / 990 位会员)  |  "
                    "以 R、F 的中位数为分界", y=0.005)
    save_fig(fig, "ch02_03_用户忠诚度")
    return u


def m_4_cohort() -> pd.DataFrame:
    """④ 同期群分析: 按注册日分组, 追踪逐日留存。"""
    rule("④ 同期群分析")
    df = pd.read_excel(BASE1 / "1-4同期群分析.xlsx", sheet_name="用户登录明细")
    df["注册日期"] = pd.to_datetime(df["注册日期"]).dt.normalize()
    df["登录时点"] = pd.to_datetime(df["登录时点"])
    df["群组"] = df["注册日期"].dt.strftime("%m-%d")
    df["第几天"] = (df["登录时点"].dt.normalize() - df["注册日期"]).dt.days
    df = df[(df["第几天"] >= 0) & (df["第几天"] <= 14)]

    size = df.groupby("群组")["用户编号"].nunique()
    active = (df.groupby(["群组", "第几天"])["用户编号"].nunique()
                .unstack(fill_value=0).reindex(columns=range(0, 15), fill_value=0))
    ret = active.div(size, axis=0)

    print("各群组注册人数:")
    print(size.to_string())
    print(f"\n次日留存率均值: {ret[1].mean():.1%}   7 日留存率均值: {ret[7].mean():.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.0), width_ratios=[1.45, 1.0],
                             gridspec_kw={"wspace": 0.22})
    ax = axes[0]
    im = ax.imshow(ret.values, cmap=SEQ_CMAP, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(15))
    ax.set_xticklabels([f"D{i}" for i in range(15)])
    ax.set_yticks(range(len(ret)))
    ax.set_yticklabels([f"{c} (n={size[c]})" for c in ret.index], fontsize=9.5)
    ax.grid(False)
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            v = ret.values[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 0.55 else COLOR["text"])
    ax.set_title("① 同期群留存矩阵 (按注册日分组)")
    ax.set_xlabel("注册后第几天")
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.03)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(pct))
    cb.outline.set_visible(False)

    ax = axes[1]
    for i, (c, row) in enumerate(ret.iterrows()):
        ax.plot(row.index, row.values, marker="o", ms=3.5, lw=1.6,
                color=PALETTE[i % len(PALETTE)], alpha=0.85, label=c)
    ax.set_title("② 各群组留存曲线")
    ax.set_xlabel("注册后第几天")
    ax.set_ylabel("留存率")
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_xticks(range(15))
    ax.legend(loc="upper right", ncol=2, fontsize=8.5, title="注册日",
              title_fontsize=9)
    style_axes_clean(ax)

    fig.suptitle("同期群分析: 不同注册批次的首日留存相近, 衰减节奏也相近",
                 y=1.02, x=0.02, ha="left")
    add_source(fig, "数据来源: 1-4同期群分析.xlsx 用户登录明细 (13,102 行 / 1,001 位用户)  |  "
                    "留存率 = 当天登录人数 / 该群组注册人数", y=0.005)
    save_fig(fig, "ch02_04_同期群分析")
    return ret


def m_5_funnel() -> None:
    """⑤ 漏斗图: 用户行为漏斗 + AARRR 运营漏斗。"""
    rule("⑤ 漏斗图")
    demo = pd.read_excel(BASE1 / "2-1漏斗图.xlsx", sheet_name="演示")
    demo.columns = [str(c).strip() for c in demo.columns]
    act = pd.DataFrame({"阶段": ["浏览", "收藏", "加购", "购买"],
                        "人数": [100, 90, 80, 80]})
    aarrr = pd.DataFrame({"阶段": ["获取", "激活", "留存", "收益", "推荐"],
                          "人数": [100, 80, 60, 40, 30]})
    print(demo.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), width_ratios=[1, 1],
                             gridspec_kw={"wspace": 0.24})

    def draw(ax, df, title):
        y = np.arange(len(df))[::-1]
        vals = df["人数"].values.astype(float)
        maxv = vals.max()
        for i, (yy, v, name) in enumerate(zip(y, vals, df["阶段"])):
            half = v / maxv / 2
            ax.add_patch(Rectangle((0.5 - half, yy - 0.36), half * 2, 0.72,
                                   facecolor=PALETTE[i % len(PALETTE)], alpha=0.88,
                                   edgecolor="white", linewidth=1.5))
            ax.text(0.5, yy, f"{name}  {v:,.0f}", ha="center", va="center",
                    color="white", fontsize=12, fontweight="bold")
            if i > 0:
                step = v / vals[i - 1]
                ax.text(0.5 + half + 0.03, yy, f"{step:.0%}", ha="left",
                        va="center", fontsize=10, color=COLOR["text_soft"])
        ax.set_xlim(0, 1.22)
        ax.set_ylim(-0.7, len(df) - 0.3)
        ax.axis("off")
        ax.set_title(title)

    draw(axes[0], act, "① 用户行为漏斗 (浏览 → 购买)")
    draw(axes[1], aarrr, "② AARRR 运营行为漏斗")
    axes[0].text(1.18, 3, f"整体转化 {act['人数'].iloc[-1] / act['人数'].iloc[0]:.0%}",
                 fontsize=11, color=COLOR["negative"], fontweight="bold",
                 ha="right", va="top")
    axes[1].text(1.18, 4, f"整体转化 {aarrr['人数'].iloc[-1] / aarrr['人数'].iloc[0]:.0%}",
                 fontsize=11, color=COLOR["negative"], fontweight="bold",
                 ha="right", va="top")

    fig.suptitle("漏斗图: 逐层转化率, 用于定位流失最严重的环节",
                 y=1.02, x=0.02, ha="left")
    add_source(fig, "数据来源: 2-1漏斗图.xlsx 演示表 / 练习表  |  "
                    "箭头右侧为该层相对上一层的转化率", y=0.005)
    save_fig(fig, "ch02_05_漏斗图")


def m_6_corr() -> pd.DataFrame:
    """⑥ 相关分析: 连续变量的散点 + 回归 + 相关系数。"""
    rule("⑥ 相关分析")
    df = pd.read_excel(BASE1 / "3-1相关分析.xlsx", sheet_name="原始数据")
    x, y = df.iloc[:, 1].astype(float), df.iloc[:, 0].astype(float)
    r = np.corrcoef(x, y)[0, 1]
    k, b = np.polyfit(x, y, 1)
    print(f"样本量 {len(df)}, Pearson r = {r:.4f}, 回归 y = {k:.3f}x + {b:.1f}")

    fig, ax = plt.subplots(figsize=(10.5, 7))
    ax.scatter(x, y, s=40, color=COLOR["primary"], alpha=0.55,
               edgecolors="white", linewidths=0.4, label="样本")
    xs = np.linspace(x.min(), x.max(), 60)
    ax.plot(xs, k * xs + b, color=COLOR["negative"], lw=2.4, ls="--",
            label=f"拟合线 y = {k:.2f}x + {b:.1f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("相关分析: 散点图 + 线性拟合")
    ax.legend(loc="upper left")
    note_box(ax, 0.97, 0.95, f"Pearson r = {r:.3f}\n决定系数 R² = {r ** 2:.3f}\n"
                             f"样本量 n = {len(df)}",
             ha="right", fontsize=11)
    style_axes_clean(ax)
    add_source(fig, "数据来源: 3-1相关分析.xlsx 原始数据", y=0.005)
    save_fig(fig, "ch02_06_相关分析")
    return pd.DataFrame({"r": [r], "k": [k], "b": [b], "n": [len(df)]})

# ============================================================================ #
# 3. 引申分析方法
# ============================================================================ #

def block(raw: pd.DataFrame, key: str, ncols: int = 3) -> pd.DataFrame:
    """切出以 key 为表头的区块。

    教材的这些 xlsx 把多张图表的数据堆在同一张表里, 且标题未必在第 0 列,
    因此这里统一委托给 block_at (它会扫描所有列定位标题)。
    """
    out = block_at(raw, key, ncols)
    if out.empty:
        raise ValueError(f"未找到数据区块: {key!r}")
    return out


def m_7_trend() -> None:
    """⑦ 趋势分析: 水平、波动与预警。"""
    rule("⑦ 趋势分析")
    raw = pd.read_excel(BASE2 / "1趋势分析.xlsx", sheet_name="趋势图", header=None)
    yearly = block(raw, "year", 2).astype({"year": int, "订单量（单位：个）": float})
    weekly = block(raw, "周", 4)
    weekly.columns = ["周", "时长", "均值", "预警值"]
    weekly = weekly.astype({"时长": float, "均值": float, "预警值": float})
    monthly = block(raw, "月", 3)
    monthly.columns = ["月", "订单量", "环比"]
    print(yearly.to_string(index=False))
    print(weekly.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8),
                             gridspec_kw={"wspace": 0.28})
    ax = axes[0]
    ax.plot(yearly["year"], yearly["订单量（单位：个）"], marker="o",
            color=COLOR["primary"], lw=2.4)
    ax.fill_between(yearly["year"], yearly["订单量（单位：个）"],
                    color=COLOR["primary"], alpha=0.15)
    lo = yearly["订单量（单位：个）"].idxmin()
    ax.scatter([yearly.loc[lo, "year"]], [yearly.loc[lo, "订单量（单位：个）"]],
               s=110, color=COLOR["negative"], zorder=5, edgecolor="white", lw=1.5)
    ax.annotate(f"低点 {int(yearly.loc[lo, 'year'])} 年\n"
                f"{yearly.loc[lo, '订单量（单位：个）']:,.0f} 个",
                (yearly.loc[lo, "year"], yearly.loc[lo, "订单量（单位：个）"]),
                textcoords="offset points", xytext=(0, -46), ha="center",
                fontsize=10, color=COLOR["negative"], fontweight="bold")
    ax.set_title("① 近 10 年订单量趋势")
    ax.set_xlabel("年份")
    ax.set_ylabel("订单量 (个)")
    ax.set_xticks(yearly["year"])
    ax.tick_params(axis="x", rotation=45)
    style_axes_clean(ax)

    ax = axes[1]
    xw = np.arange(len(weekly))
    ax.plot(xw, weekly["时长"], marker="o", color=COLOR["primary"], label="周均时长")
    ax.axhline(weekly["均值"].iloc[0], color=COLOR["accent"], ls="--", lw=1.8,
               label=f"11 周均值 {weekly['均值'].iloc[0]:.1f}h")
    ax.axhline(weekly["预警值"].iloc[0], color=COLOR["negative"], ls=":", lw=2.0,
               label=f"预警线 {weekly['预警值'].iloc[0]:.0f}h")
    over = weekly[weekly["时长"] > weekly["预警值"].iloc[0]]
    ax.scatter(xw[over.index], over["时长"], s=110, color=COLOR["negative"],
               zorder=5, edgecolor="white", lw=1.5)
    for i in over.index:
        ax.annotate(f"超预警 {weekly.loc[i, '时长']:.0f}h",
                    (i, weekly.loc[i, "时长"]), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=9.5,
                    color=COLOR["negative"], fontweight="bold")
    ax.set_xticks(xw)
    ax.set_xticklabels(weekly["周"], fontsize=9)
    ax.set_title("② 周均处理时长: 均值与预警线")
    ax.set_ylabel("小时")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, weekly["时长"].max() * 1.20)
    style_axes_clean(ax)

    ax = axes[2]
    months = monthly["月"].astype(str)
    vals = monthly["订单量"].astype(float)
    bars = ax.bar(months, vals, color=COLOR["primary"], width=0.62)
    annotate_bars(ax, vals, bars=bars, fmt="{:,.0f}", offset=0.02, fontsize=8.5)
    ax2 = ax.twinx()
    ax2.plot(months, monthly["环比"].astype(float), color=COLOR["accent"],
             marker="o", lw=2.0)
    ax2.axhline(0, color=COLOR["text_soft"], lw=0.8, ls=":")
    ax2.yaxis.set_major_formatter(FuncFormatter(pct))
    ax2.set_ylabel("环比增幅", color=COLOR["accent"])
    ax2.tick_params(axis="y", colors=COLOR["accent"])
    ax2.grid(False)
    ax.set_title("③ 月度订单量与环比")
    ax.set_xlabel("月份")
    ax.set_ylabel("订单量 (个)")
    ax.set_ylim(0, vals.max() * 1.22)
    style_axes_clean(ax)

    fig.suptitle("趋势分析: 看水平、看波动、也看拐点", y=1.03, x=0.02, ha="left")
    add_source(fig, "数据来源: 1趋势分析.xlsx 趋势图 (三个独立区块)", y=0.005)
    save_fig(fig, "ch02_07_趋势分析")


def m_8_compare() -> None:
    """⑧ 对比分析: 排序条形 + 雷达图。"""
    rule("⑧ 对比分析")
    raw = pd.read_excel(BASE2 / "2对比分析.xlsx", sheet_name="对比分析", header=None)
    prod = block(raw, "产品类型", 2)
    prod.columns = ["产品", "订单量"]
    prod["订单量"] = prod["订单量"].astype(float)
    radar = block(raw, "某顾客画像", 2)
    radar.columns = ["维度", "得分"]
    radar["得分"] = radar["得分"].astype(float)
    print(prod.sort_values("订单量", ascending=False).head(5).to_string(index=False))
    print(radar.to_string(index=False))

    fig = plt.figure(figsize=(16.5, 6.4))
    ax = fig.add_subplot(1, 2, 1)
    p = prod.sort_values("订单量")
    ax.barh(p["产品"], p["订单量"], height=0.66, color=COLOR["primary"])
    annotate_hbars(ax, p["订单量"].values, fmt="{:,.0f}", pad=0.012, fontsize=9)
    ax.set_xlim(0, p["订单量"].max() * 1.22)
    ax.set_title("① 各产品订单量 (排序后)")
    ax.set_xlabel("订单量 (个)")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    style_axes_clean(ax)

    ax = fig.add_subplot(1, 2, 2, projection="polar")
    ang = np.linspace(0, 2 * np.pi, len(radar), endpoint=False).tolist()
    vals = radar["得分"].tolist()
    ang += ang[:1]
    vals += vals[:1]
    ax.plot(ang, vals, color=COLOR["primary"], lw=2.2)
    ax.fill(ang, vals, color=COLOR["primary"], alpha=0.22)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(radar["维度"], fontsize=11)
    ax.set_ylim(0, max(radar["得分"]) * 1.15)
    ax.set_title("② 客户画像: 六维得分雷达图", pad=26)
    ax.grid(color=COLOR["grid"])

    fig.suptitle("对比分析: 横向排序看差距, 雷达图看结构",
                 y=1.02, x=0.02, ha="left")
    add_source(fig, "数据来源: 2对比分析.xlsx 对比分析", y=0.005)
    save_fig(fig, "ch02_08_对比分析")


def m_9_composition() -> None:
    """⑨ 构成分析: 占比 (饼图) + 结构变化 (堆叠柱 / 面积)。"""
    rule("⑨ 构成分析")
    raw = pd.read_excel(BASE2 / "3构成分析.xlsx", sheet_name="构成分析", header=None)
    part = block(raw, "产品类型", 2)
    part.columns = ["产品", "销售量"]
    part["销售量"] = part["销售量"].astype(float)
    yearly = block(raw, "年/产品类型", 4)
    yearly = yearly.set_index(yearly.columns[0]).astype(float)
    print(part.to_string(index=False))
    print(yearly.to_string())
    share = part.set_index("产品")["销售量"]
    share = share / share.sum()

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.0),
                             gridspec_kw={"wspace": 0.30})
    ax = axes[0]
    wedges, _, autotexts = ax.pie(
        share.values, labels=share.index, autopct="%.1f%%", startangle=90,
        colors=PALETTE[:len(share)], pctdistance=0.76,
        wedgeprops=dict(width=0.44, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=11))
    for t in autotexts:
        t.set_color("white")
        t.set_fontweight("bold")
    ax.text(0, 0, f"{part['销售量'].sum():,.0f}\n总销量", ha="center", va="center",
            fontsize=14, fontweight="bold", color=COLOR["text"])
    ax.set_title("① 产品销量占比")

    ax = axes[1]
    x = np.arange(len(yearly.index))
    bottom = np.zeros(len(yearly.index))
    for i, c in enumerate(yearly.columns):
        v = yearly[c].values
        ax.bar(x, v, bottom=bottom, width=0.6, label=c, color=PALETTE[i])
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels(yearly.index.astype(int))
    ax.set_title("② 各年份产品销量结构")
    ax.set_xlabel("年份")
    ax.set_ylabel("销量")
    ax.legend(loc="upper right", ncol=1, fontsize=9)
    style_axes_clean(ax)

    ax = axes[2]
    for i, c in enumerate(yearly.columns):
        ax.plot(yearly.index.astype(int), yearly[c].values, marker="o",
                color=PALETTE[i], label=c, lw=2.0)
    ax.set_title("③ 各产品销量的年度走势")
    ax.set_xlabel("年份")
    ax.set_ylabel("销量")
    ax.legend(loc="upper right", ncol=1, fontsize=9)
    style_axes_clean(ax)

    fig.suptitle("构成分析: 同一份数据, 占比看结构、堆叠看变化",
                 y=1.02, x=0.02, ha="left")
    add_source(fig, "数据来源: 3构成分析.xlsx 构成分析", y=0.005)
    save_fig(fig, "ch02_09_构成分析")

def m_10_distribution() -> None:
    """⑩ 分布分析: 直方图看形态, 箱线图看异常值。"""
    rule("⑩ 分布分析")
    raw = pd.read_excel(BASE2 / "4分布分析.xlsx", sheet_name="分布分析", header=None)
    # 原表最左侧有整列为空, 先丢掉全空列再取前两列 (时点 / 门店客流量)
    raw = raw.dropna(axis=1, how="all")
    d = raw.iloc[:, :2].copy()
    d.columns = ["时点", "客流量"]
    d["客流量"] = pd.to_numeric(d["客流量"], errors="coerce")
    d["时点"] = pd.to_datetime(d["时点"], errors="coerce", format="mixed")
    d = d.dropna()
    if d.empty:
        raise ValueError("分布分析: 未解析到有效的时点/客流量数据")
    d["小时"] = d["时点"].dt.hour
    print(f"样本 {len(d)} 个半小时时段, 客流均值 {d['客流量'].mean():.1f}, "
          f"中位数 {d['客流量'].median():.1f}, 标准差 {d['客流量'].std():.1f}")

    q1, q3 = d["客流量"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper, lower = q3 + 1.5 * iqr, q1 - 1.5 * iqr
    n_out = int(((d["客流量"] > upper) | (d["客流量"] < lower)).sum())

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.2), width_ratios=[1, 1.1],
                             gridspec_kw={"wspace": 0.24})
    ax = axes[0]
    ax.hist(d["客流量"], bins=34, color=COLOR["primary"], alpha=0.85,
            edgecolor="white", linewidth=0.8)
    ax.axvline(d["客流量"].mean(), color=COLOR["negative"], lw=2.0, ls="--",
               label=f"均值 {d['客流量'].mean():.0f}")
    ax.axvline(d["客流量"].median(), color=COLOR["accent"], lw=2.0, ls=":",
               label=f"中位数 {d['客流量'].median():.0f}")
    ax.set_title("① 门店客流量分布")
    ax.set_xlabel("半小时时段客流量 (人)")
    ax.set_ylabel("时段数")
    ax.legend(loc="upper right")
    note_box(ax, 0.02, 0.97,
             f"Q1 {q1:.0f} / Q3 {q3:.0f}, IQR {iqr:.0f}\n"
             f"上限 {upper:.0f}, 异常时段 {n_out} 个",
             fontsize=10)
    style_axes_clean(ax)

    ax = axes[1]
    hours = sorted(d["小时"].unique())
    data = [d.loc[d["小时"] == h, "客流量"].values for h in hours]
    bp = ax.boxplot(data, positions=hours, widths=0.62, patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker="o", markersize=3.5, alpha=0.5,
                                    markerfacecolor=COLOR["negative"],
                                    markeredgecolor="none"),
                    medianprops=dict(color=COLOR["accent"], lw=2),
                    boxprops=dict(facecolor="#DCE7F2", edgecolor=COLOR["primary"]),
                    whiskerprops=dict(color=COLOR["primary"]),
                    capprops=dict(color=COLOR["primary"]))
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h}时" for h in hours], rotation=45, ha="right",
                       fontsize=9)
    ax.set_title("② 按小时分组的客流量箱线图")
    ax.set_xlabel("小时")
    ax.set_ylabel("客流量 (人)")
    style_axes_clean(ax)

    fig.suptitle("分布分析: 先看形态与集中趋势, 再看离散程度与异常值",
                 y=1.02, x=0.02, ha="left")
    add_source(fig, "数据来源: 4分布分析.xlsx 分布分析 (半小时粒度客流量)", y=0.005)
    save_fig(fig, "ch02_10_分布分析")


def block_at(raw: pd.DataFrame, key: str, ncols: int) -> pd.DataFrame:
    """在任意位置定位以 key 为表头的区块。"""
    arr = raw.values
    for i in range(len(arr)):
        for j in range(arr.shape[1] - ncols + 1):
            if str(arr[i, j]).strip() == key:
                header = [str(x).strip() for x in arr[i][j:j + ncols]]
                rows, k = [], i + 1
                while k < len(arr) and pd.notna(arr[k, j]):
                    rows.append(arr[k][j:j + ncols])
                    k += 1
                return pd.DataFrame(rows, columns=header)
    return pd.DataFrame()


def m_11_relation() -> None:
    """⑪ 关系分析: 分类变量之间的转化差异 + 两个连续变量的关系。"""
    rule("⑪ 关系分析")
    raw = pd.read_excel(BASE2 / "5关系分析.xlsx", sheet_name="关系图", header=None)
    cat = block_at(raw, "地区/人数", 3)
    cat.columns = ["营销活动", "下单", "未下单"]
    cat[["下单", "未下单"]] = cat[["下单", "未下单"]].astype(float)
    cat["下单率"] = cat["下单"] / (cat["下单"] + cat["未下单"])

    yy = block_at(raw, "产出收益", 2)
    if not yy.empty:
        yy.columns = ["收益", "成本"]
        yy = yy.apply(pd.to_numeric, errors="coerce").dropna()
    print(cat.to_string(index=False))
    if not yy.empty:
        r = np.corrcoef(yy["成本"], yy["收益"])[0, 1]
        print(f"\n投入成本 vs 产出收益: r = {r:.4f}, n = {len(yy)}")

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.2), width_ratios=[1, 1.15],
                             gridspec_kw={"wspace": 0.26})
    ax = axes[0]
    x = np.arange(len(cat))
    w = 0.36
    b1 = ax.bar(x - w / 2, cat["下单"], width=w, label="下单", color=COLOR["primary"])
    b2 = ax.bar(x + w / 2, cat["未下单"], width=w, label="未下单", color=COLOR["muted"])
    annotate_bars(ax, cat["下单"].values, bars=b1, fmt="{:,.0f}", offset=0.02, fontsize=9.5)
    annotate_bars(ax, cat["未下单"].values, bars=b2, fmt="{:,.0f}", offset=0.02, fontsize=9.5)
    for i, v in enumerate(cat["下单率"]):
        ax.text(i, (cat["下单"].iloc[i] + cat["未下单"].iloc[i]) * 1.06,
                f"下单率 {v:.0%}", ha="center", fontsize=11,
                color=COLOR["positive"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cat["营销活动"])
    ax.set_ylim(0, max(cat["下单"] + cat["未下单"]) * 1.26)
    ax.set_title("① 分类变量: 不同营销活动的下单情况")
    ax.set_ylabel("客户数")
    ax.legend(loc="upper right", ncol=2)
    style_axes_clean(ax)

    ax = axes[1]
    ax.scatter(yy["成本"], yy["收益"], s=42, color=COLOR["primary"], alpha=0.55,
               edgecolors="white", linewidths=0.4)
    k, b = np.polyfit(yy["成本"], yy["收益"], 1)
    xs = np.linspace(yy["成本"].min(), yy["成本"].max(), 50)
    ax.plot(xs, k * xs + b, color=COLOR["negative"], lw=2.2, ls="--")
    ax.set_title("② 连续变量: 投入成本与产出收益")
    ax.set_xlabel("投入成本")
    ax.set_ylabel("产出收益")
    note_box(ax, 0.03, 0.97, f"Pearson r = {r:.3f}\n样本 n = {len(yy)}",
             fontsize=10.5)
    style_axes_clean(ax)

    fig.suptitle("关系分析: 分类看差异, 连续看相关",
                 y=1.02, x=0.02, ha="left")
    add_source(fig, "数据来源: 5关系分析.xlsx 关系图", y=0.005)
    save_fig(fig, "ch02_11_关系分析")


def m_12_sankey() -> None:
    """⑫ 桑基图: 页面之间的流量流转。"""
    rule("⑫ 桑基图")
    df = pd.read_excel(BASE2 / "6桑吉图.xlsx", sheet_name="Sheet1")
    df.columns = [str(c).strip() for c in df.columns]
    df["流量"] = pd.to_numeric(df["流量"], errors="coerce")
    df = df.dropna(subset=["来源", "目标", "流量"])
    print(df[["来源", "目标", "流量"]].to_string(index=False))

    # --- 分层: 从"首页"开始做 BFS ---
    adj = {}
    for _, r in df.iterrows():
        adj.setdefault(r["来源"], []).append(r["目标"])
    level, frontier = {"首页": 0}, ["首页"]
    while frontier:
        nxt = []
        for n in frontier:
            for m in adj.get(n, []):
                if m not in level:
                    level[m] = level[n] + 1
                    nxt.append(m)
        frontier = nxt
    for n in set(df["来源"]) | set(df["目标"]):
        level.setdefault(n, 0)

    inflow = df.groupby("目标")["流量"].sum()
    outflow = df.groupby("来源")["流量"].sum()
    total = df.groupby("来源")["流量"].sum().max()

    # --- 布局 ---
    levels = {}
    for n, lv in level.items():
        levels.setdefault(lv, []).append(n)
    ypos = {}
    for lv, nodes in levels.items():
        nodes.sort(key=lambda n: -(inflow.get(n, 0) + outflow.get(n, 0)))
        span = sum(inflow.get(n, 0) + outflow.get(n, 0) for n in nodes)
        cur = 1.0
        for n in nodes:
            h = (inflow.get(n, 0) + outflow.get(n, 0)) / span * 0.86
            ypos[n] = (cur, cur - h)
            cur -= h

    fig, ax = plt.subplots(figsize=(14, 8))
    nlev = max(levels) + 1
    xpos = {lv: 0.06 + i * (0.88 / max(nlev - 1, 1)) for i, lv in enumerate(sorted(levels))}
    bw = 0.018

    # 节点
    for n, (y0, y1) in ypos.items():
        x = xpos[level[n]]
        ax.add_patch(Rectangle((x - bw / 2, y1), bw, y0 - y1,
                               facecolor=COLOR["primary"], alpha=0.85,
                               edgecolor="white", linewidth=1.2, zorder=4))
        ha = "left" if level[n] < nlev - 1 else "right"
        dx = 0.012 if ha == "left" else -0.012
        ax.text(x + dx, (y0 + y1) / 2, n, ha=ha, va="center", fontsize=10.5,
                color=COLOR["text"], zorder=5)

    # 连线 (贝塞尔带)
    used_out = {n: ypos[n][0] for n in ypos}
    used_in = {n: ypos[n][0] for n in ypos}
    for _, r in df.sort_values(["来源", "目标"]).iterrows():
        s, t, v = r["来源"], r["目标"], r["流量"]
        hs = v / (inflow.get(s, 0) + outflow.get(s, 0)) * (ypos[s][0] - ypos[s][1])
        ht = v / (inflow.get(t, 0) + outflow.get(t, 0)) * (ypos[t][0] - ypos[t][1])
        ys0, ys1 = used_out[s], used_out[s] - hs
        yt0, yt1 = used_in[t], used_in[t] - ht
        used_out[s] -= hs
        used_in[t] -= ht
        x0, x1 = xpos[level[s]] + bw / 2, xpos[level[t]] - bw / 2
        xm = (x0 + x1) / 2
        verts = [(x0, ys0), (xm, ys0), (xm, yt0), (x1, yt0),
                 (x1, yt1), (xm, yt1), (xm, ys1), (x0, ys1), (x0, ys0)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=COLOR["primary"],
                               alpha=0.22, edgecolor="none", zorder=2))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.04)
    ax.axis("off")
    ax.set_title("桑基图: 网站页面的流量流转路径", pad=18)
    add_source(fig, "数据来源: 6桑吉图.xlsx Sheet1  |  带宽度 = 流量", y=0.01)
    save_fig(fig, "ch02_12_桑基图")


# ============================================================================ #
# 4. 主入口
# ============================================================================ #

def main() -> int:
    font = setup_style()
    print("=" * 78)
    print("《商业数据分析》第2章 数据分析方法 —— 12 种分析范式的 Python 实现")
    print("=" * 78)
    print(f"Python    : {sys.version.split()[0]}")
    print(f"中文字体  : {font}")
    print(f"图片输出  : {FIG}  (PNG, 300dpi)")

    for d in (BASE1, BASE2):
        if not d.exists():
            print(f"\n[错误] 找不到教材目录: {d}")
            return 2

    m_1_bcg()
    m_2_rfm()
    m_3_loyalty()
    m_4_cohort()
    m_5_funnel()
    m_6_corr()
    m_7_trend()
    m_8_compare()
    m_9_composition()
    m_10_distribution()
    m_11_relation()
    m_12_sankey()

    rule("小结")
    print("""
12 种方法可归为四类动作:
  定位  —— 波士顿矩阵 (增长-份额)、漏斗图 (转化瓶颈)
  分层  —— RFM、用户忠诚度、同期群
  描述  —— 趋势、对比、构成、分布
  解释  —— 相关分析、关系分析、桑基图 (流转路径)

Python 迁移的价值:
  所有中间口径 (分位点、增长率、留存分母、转化率) 都写在代码里,
  Excel 中藏在辅助列和数据透视表里的假设, 现在可以逐行复核。
""")
    print("图片已输出到:", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
