# -*- coding: utf-8 -*-
r"""
《商业数据分析》教材重构 —— 第5章 业务数据分析
================================================================================
原教材实现: Power BI 业务看板 + WPS 归因分析 (Excel)
本文件实现: Python 单文件版 (pandas + matplotlib), 可直接运行
================================================================================

运行::

    C:\Users\dell\anaconda3\envs\dataviz\python.exe -X utf8 ch05_业务数据分析.py

输出: 与脚本同目录的 PNG 图片 (300dpi)

两个案例
--------
案例A  银行理财产品销售看板
       原 Power BI 看板由 6 个视觉对象组成:
         cardVisual(MTD销售目标达成率) / decompositionTreeVisual(分解树)
         clusteredColumnChart(不同理财品类的销售占比)
         lineClusteredColumnComboChart(客单及期限均值分布)
         WordCloud x2(选择本行的原因 / 选择产品原因)
       业务约束: 本月目标 2500 万, 当前为 29 日时点数据, 共 31 天。

案例B  贷款申请量归因分析
       业务问题: 2021 年 5 月贷款申请量环比下降, 下滑原因是什么?
       方法: 渠道贡献度 = 该渠道环比差额 / 整体环比差额
       原报告结论: 理财经理柜台 58.97%、手机银行 48.72%、自助机 -7.69%

迁移要点
--------
* 词云需要额外依赖, 这里改用**排序条形图**表达同一信息 —— 名次与频次都更精确,
  而且可以直接读出数值, 比词云更适合教材。
* 原看板的"分解树"是一种交互式下钻, Python 版用**分组汇总 + 逐层拆解**表达。
* 归因分析的关键是"分母": 贡献度必须用整体差额做分母, 各渠道贡献度之和为 1。
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
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent      # 本章交付目录
ROOT = HERE.parent                          # 项目根目录 (教材原文所在位置)
CH5 = ROOT / "第5章 业务数据分析"
BI_DIR = CH5 / "第5章  PowerBI业务看板案例-银行案例"
ATTR_DIR = CH5 / "第5章 WPS归因分析案例-银行案例"
FIG = HERE
FIG.mkdir(parents=True, exist_ok=True)

TARGET = 25_000_000          # 本月销售目标 2500 万
DAYS_TOTAL, DAYS_NOW = 31, 29

if sys.platform == "win32":
    _lib = Path(sys.prefix) / "Library" / "bin"
    if _lib.is_dir() and str(_lib).lower() not in os.environ.get("PATH", "").lower():
        os.environ["PATH"] = f"{_lib}{os.pathsep}{os.environ.get('PATH', '')}"


def rule(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


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


def style_axes_clean(ax):
    for s in ("left", "bottom"):
        ax.spines[s].set_color(COLOR["grid"])


def pct(v, _=None):
    return f"{v:.0%}"


def money_wan(v, _=None):
    v = float(v)
    return f"{v / 1e4:,.0f}万" if abs(v) >= 1e4 else f"{v:,.0f}"


def annotate_bars(ax, values, bars=None, fmt="{:,.0f}", offset=0.02, fontsize=10,
                  color=None):
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
            color=COLOR["text"], linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.45", fc=face, ec=edge, lw=1.0))

# ============================================================================ #
# 案例A: 银行理财产品销售看板
# ============================================================================ #

def load_bi():
    """读取销售数据与三张说明表 (维度名称解码)。"""
    d = pd.read_excel(BI_DIR / "银行案例数据 - 学员用.xlsx", sheet_name="销售数据")
    risk = pd.read_excel(BI_DIR / "银行案例数据 - 学员用.xlsx", sheet_name="风险偏好说明表")
    cat = pd.read_excel(BI_DIR / "银行案例数据 - 学员用.xlsx", sheet_name="品类说明表")
    edu = pd.read_excel(BI_DIR / "银行案例数据 - 学员用.xlsx", sheet_name="学历说明表")
    d["风险"] = d["风险编号"].map(dict(zip(risk.iloc[:, 0], risk.iloc[:, 1])))
    d["品类"] = d["品类编号"].map(dict(zip(cat.iloc[:, 0], cat.iloc[:, 1])))
    d["学历"] = d["学历编号"].map(dict(zip(edu.iloc[:, 0], edu.iloc[:, 1])))
    return d


def a1_kpi(d: pd.DataFrame) -> dict:
    """图 ch05_01: KPI 看板 —— 对应原 cardVisual + MTD 销售目标达成率。"""
    rule("案例A · ① KPI 看板 (MTD 销售目标达成率)")
    total = float(d["购买金额"].sum())
    rate = total / TARGET
    avg = float(d["购买金额"].mean())
    time_rate = DAYS_NOW / DAYS_TOTAL
    forecast = total / DAYS_NOW * DAYS_TOTAL
    print(f"累计销售 {total:,.0f} 元, 目标 {TARGET:,.0f}, 达成率 {rate:.2%}")
    print(f"时间进度 {time_rate:.2%}, 月末线性预测 {forecast:,.0f} 元 ({forecast/TARGET:.2%})")

    fig = plt.figure(figsize=(15, 7.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.78], hspace=0.42, wspace=0.18)

    cards = [("累计销售额", f"{total/1e4:,.0f}万", f"目标 {TARGET/1e4:,.0f}万", COLOR["primary"]),
             ("目标达成率", f"{rate:.1%}", f"时间进度 {time_rate:.1%}", COLOR["accent"]),
             ("成交客户数", f"{len(d):,}", "人", COLOR["positive"]),
             ("客单价", f"{avg/1e4:.2f}万", f"平均期限 {d['期限'].mean():.1f} 年", COLOR["highlight"])]
    for i, (label, value, sub, col) in enumerate(cards):
        ax = fig.add_subplot(gs[0, i])
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.02, 0.06), 0.96, 0.88, transform=ax.transAxes,
                                   facecolor=COLOR["bg_soft"], edgecolor=COLOR["grid"],
                                   linewidth=1.0, clip_on=False))
        ax.add_patch(plt.Rectangle((0.02, 0.06), 0.05, 0.88, transform=ax.transAxes,
                                   facecolor=col, edgecolor="none", clip_on=False))
        ax.text(0.11, 0.78, label, transform=ax.transAxes, fontsize=11.5,
                color=COLOR["text_soft"])
        ax.text(0.11, 0.50, value, transform=ax.transAxes, fontsize=26,
                fontweight="bold", color=col)
        ax.text(0.11, 0.20, sub, transform=ax.transAxes, fontsize=10,
                color=COLOR["text_soft"])

    # 目标进度 + 分日累计
    ax = fig.add_subplot(gs[1, :2])
    ax.barh([0], [1], color=COLOR["grid"], height=0.46)
    ax.barh([0], [rate], color=COLOR["primary"], height=0.46)
    ax.axvline(time_rate, color=COLOR["negative"], lw=2.0, ls="--")
    ax.text(time_rate, 0.34, f"时间进度 {time_rate:.1%}", color=COLOR["negative"],
            ha="center", fontsize=10, fontweight="bold")
    ax.text(rate, -0.34, f"销售达成 {rate:.1%}", color=COLOR["primary"],
            ha="right", fontsize=10.5, fontweight="bold")
    ax.set_xlim(0, 1.16)
    ax.set_ylim(-0.7, 0.7)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_title("目标进度: 销售略快于时间进度", fontsize=12)
    ax.grid(False)
    for s in ("left", "bottom", "top", "right"):
        ax.spines[s].set_visible(False)

    ax = fig.add_subplot(gs[1, 2:])
    ax.axis("off")
    ax.text(0.02, 0.86, "月末达成预测", fontsize=12, color=COLOR["text_soft"])
    ax.text(0.02, 0.56, f"{forecast/1e4:,.0f} 万", fontsize=24, fontweight="bold",
            color=COLOR["positive"] if forecast >= TARGET else COLOR["negative"])
    ax.text(0.02, 0.24,
            f"按 29 日日均 {total/DAYS_NOW/1e4:.1f} 万线性外推至 31 日\n"
            f"预计达成 {forecast/TARGET:.1%}, "
            + ("可完成目标" if forecast >= TARGET else "仍有缺口"),
            fontsize=10.5, color=COLOR["text"], linespacing=1.7)

    fig.suptitle("案例A · 银行理财产品销售看板 (29 日时点)", y=0.99, x=0.02, ha="left")
    add_source(fig, "数据来源: 银行案例数据-学员用.xlsx 销售数据 (440 条)  |  目标 2,500 万, "
                    "本月 31 天, 当前为第 29 天", y=0.005)
    save_fig(fig, "ch05_01_销售看板_KPI")
    return {"total": total, "rate": rate, "forecast": forecast}


def a2_category(d: pd.DataFrame) -> pd.DataFrame:
    """图 ch05_02: 品类销售占比 + 风险等级结构 (对应 clusteredColumnChart)。"""
    rule("案例A · ② 品类与风险结构")
    g = (d.groupby("品类", as_index=False)
           .agg(销售额=("购买金额", "sum"), 客户数=("购买金额", "size"))
           .sort_values("销售额"))
    g["占比"] = g["销售额"] / g["销售额"].sum()
    risk = d.pivot_table(index="品类", columns="风险", values="购买金额",
                         aggfunc="sum").fillna(0).reindex(g["品类"])
    print(g.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.4), width_ratios=[1.1, 1.0],
                             gridspec_kw={"wspace": 0.30})
    ax = axes[0]
    ax.barh(g["品类"], g["销售额"] / 1e4, height=0.66, color=COLOR["primary"])
    annotate_hbars(ax, g["销售额"].values / 1e4, fmt="{:,.0f}万", pad=0.012, fontsize=9.5)
    for i, (_, r) in enumerate(g.iterrows()):
        ax.text(g["销售额"].max() / 1e4 * 1.20, i, f"{r['占比']:.1%}",
                va="center", ha="right", fontsize=9.5, color=COLOR["text_soft"])
    ax.set_xlim(0, g["销售额"].max() / 1e4 * 1.26)
    ax.set_title("① 不同理财品类的销售额")
    ax.set_xlabel("销售额 (万元)")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    style_axes_clean(ax)

    ax = axes[1]
    x = np.arange(len(risk.index))
    bottom = np.zeros(len(risk.index))
    for i, c in enumerate(risk.columns):
        v = risk[c].values / 1e4
        ax.bar(x, v, bottom=bottom, width=0.62, label=str(c), color=PALETTE[i])
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels(risk.index, rotation=32, ha="right", fontsize=9.5)
    ax.set_title("② 各品类的风险等级构成")
    ax.set_ylabel("销售额 (万元)")
    ax.legend(loc="upper right", ncol=1, fontsize=9)
    style_axes_clean(ax)

    fig.suptitle("品类结构: 单一品类占比过半, 风险等级高度集中",
                 y=1.02, x=0.02, ha="left")
    add_source(fig, "数据来源: 销售数据 + 品类说明表 + 风险偏好说明表  |  "
                    "对应原看板 clusteredColumnChart", y=0.005)
    save_fig(fig, "ch05_02_品类与风险结构")
    return g


def a3_ticket_term(d: pd.DataFrame) -> pd.DataFrame:
    """图 ch05_03: 客单价与期限均值 (对应 lineClusteredColumnComboChart)。"""
    rule("案例A · ③ 客单价与期限")
    g = (d.groupby("品类", as_index=False)
           .agg(客单价=("购买金额", "mean"), 平均期限=("期限", "mean"),
                销售额=("购买金额", "sum"))
           .sort_values("销售额", ascending=False))
    print(g.round(2).to_string(index=False))

    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    x = np.arange(len(g))
    bars = ax.bar(x, g["客单价"] / 1e4, width=0.58, color=COLOR["primary"],
                  label="客单价 (万元)")
    annotate_bars(ax, g["客单价"].values / 1e4, bars=bars, fmt="{:,.1f}", offset=0.02)
    ax.set_xticks(x)
    ax.set_xticklabels(g["品类"], rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("客单价 (万元)")
    ax.set_ylim(0, g["客单价"].max() / 1e4 * 1.22)
    ax2 = ax.twinx()
    ax2.plot(x, g["平均期限"], color=COLOR["accent"], marker="o", lw=2.4,
             label="平均期限 (年)")
    for xi, v in zip(x, g["平均期限"]):
        ax2.annotate(f"{v:.1f}年", (xi, v), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=9.5,
                     color=COLOR["accent"], fontweight="bold")
    ax2.set_ylabel("平均期限 (年)", color=COLOR["accent"])
    ax2.tick_params(axis="y", colors=COLOR["accent"])
    ax2.set_ylim(0, g["平均期限"].max() * 1.35)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    ax.set_title("客单价与平均期限: 按品类")
    style_axes_clean(ax)
    add_source(fig, "数据来源: 销售数据  |  对应原看板 lineClusteredColumnComboChart", y=0.005)
    save_fig(fig, "ch05_03_客单价与期限")
    return g

def a4_profile(d: pd.DataFrame) -> None:
    """图 ch05_04: 客户画像 (年龄 / 性别 / 收入 / 学历)。"""
    rule("案例A · ④ 客户画像")
    age = d["年龄"].value_counts().reindex(
        ["18-30岁", "31-40岁", "41-50岁", "50-60岁", "61岁以上"]).dropna()
    income = d["家庭年收入"].value_counts().reindex(
        ["5万以下", "5万-10万", "10万-20万", "20万-50万", "50万-100万", "100万以上"]).dropna()
    edu = d["学历"].value_counts()
    print("年龄:\n", age.to_string())
    print("\n收入:\n", income.to_string())

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.8),
                             gridspec_kw={"wspace": 0.34})
    ax = axes[0]
    bars = ax.bar(age.index, age.values, color=COLOR["primary"], width=0.62)
    annotate_bars(ax, age.values, bars=bars, fmt="{:,.0f}", offset=0.02, fontsize=9)
    ax.set_title("① 年龄")
    ax.set_ylabel("客户数")
    ax.tick_params(axis="x", rotation=32)
    ax.set_ylim(0, age.max() * 1.20)
    style_axes_clean(ax)

    ax = axes[1]
    sex = d["性别"].value_counts()
    wedges, _, autotexts = ax.pie(
        sex.values, labels=sex.index, autopct="%.1f%%", startangle=90,
        colors=[COLOR["highlight"], COLOR["primary"]],
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        pctdistance=0.78, textprops=dict(fontsize=11))
    for t in autotexts:
        t.set_color("white"); t.set_fontweight("bold")
    ax.text(0, 0, f"{sex.sum():,}\n客户", ha="center", va="center",
            fontsize=13, fontweight="bold", color=COLOR["text"])
    ax.set_title("② 性别")

    ax = axes[2]
    ax.barh(income.index, income.values, height=0.62, color=COLOR["accent"])
    annotate_hbars(ax, income.values, fmt="{:,.0f}", pad=0.015, fontsize=9)
    ax.set_xlim(0, income.max() * 1.24)
    ax.set_title("③ 家庭年收入")
    ax.set_xlabel("客户数")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    style_axes_clean(ax)

    ax = axes[3]
    ax.barh(edu.index, edu.values, height=0.62, color=COLOR["positive"])
    annotate_hbars(ax, edu.values, fmt="{:,.0f}", pad=0.015, fontsize=9)
    ax.set_xlim(0, edu.max() * 1.24)
    ax.set_title("④ 学历")
    ax.set_xlabel("客户数")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    style_axes_clean(ax)

    fig.suptitle("客户画像: 31-50 岁、中等收入、本科学历为主",
                 y=1.02, x=0.02, ha="left")
    add_source(fig, "数据来源: 销售数据 + 学历说明表 (440 位客户)  |  "
                    "替代原看板的分解树, 用维度分布表达客群结构", y=0.005)
    save_fig(fig, "ch05_04_客户画像")


def a5_reason(d: pd.DataFrame) -> None:
    """图 ch05_05: 选择本行/产品的原因 (替代原看板的两个词云)。"""
    rule("案例A · ⑤ 选择原因")
    bank = d["选择本行的原因"].value_counts()
    prod = d["购买目的"].value_counts()
    print("选择本行:", bank.to_dict())
    print("购买目的:", prod.to_dict())

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2), width_ratios=[1, 1.15],
                             gridspec_kw={"wspace": 0.42})
    ax = axes[0]
    b = bank.sort_values()
    ax.barh(b.index, b.values, height=0.62, color=COLOR["primary"])
    annotate_hbars(ax, b.values, fmt="{:,.0f}", pad=0.014, fontsize=10)
    ax.set_xlim(0, b.max() * 1.26)
    ax.set_title("① 选择本行的原因")
    ax.set_xlabel("客户数")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    share = b.iloc[-1] / b.sum()
    note_box(ax, 0.97, 0.06, f"「{b.index[-1]}」占 {share:.0%}",
             ha="right", va="bottom", face="#FDEDD8", edge=COLOR["accent"])
    style_axes_clean(ax)

    ax = axes[1]
    p = prod.sort_values()
    ax.barh(p.index, p.values, height=0.62, color=COLOR["accent"])
    annotate_hbars(ax, p.values, fmt="{:,.0f}", pad=0.014, fontsize=10)
    ax.set_xlim(0, p.max() * 1.24)
    ax.set_title("② 购买目的 (选择产品的原因)")
    ax.set_xlabel("客户数")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    style_axes_clean(ax)

    fig.suptitle("客户选择动因: 收益与风险并重, 而非单一追求高收益",
                 y=1.02, x=0.02, ha="left")
    add_source(fig, "数据来源: 销售数据  |  原看板用词云表达, 这里改用排序条形图, "
                    "名次与频次更精确、可直接读数", y=0.005)
    save_fig(fig, "ch05_05_选择原因")


# ============================================================================ #
# 案例B: 贷款申请量归因分析
# ============================================================================ #

def load_attr() -> pd.DataFrame:
    a = pd.read_excel(ATTR_DIR / "归因分析数据.xlsx", sheet_name="贷款申请表")
    a["申请日期"] = pd.to_datetime(a["申请日期"])
    a["年月"] = a["申请日期"].dt.to_period("M")
    return a


def b1_trend(a: pd.DataFrame) -> pd.DataFrame:
    """图 ch05_06: 贷款申请量月度趋势与环比。"""
    rule("案例B · ① 月度趋势")
    piv = (a.pivot_table(index="年月", columns="申请渠道",
                         values="申请书编号（主键）", aggfunc="count")
             .fillna(0).astype(int))
    piv["总计"] = piv.sum(axis=1)
    piv["环比"] = piv["总计"].pct_change()
    print(piv.to_string())

    fig, ax = plt.subplots(figsize=(13, 6.2))
    labels = [str(i) for i in piv.index]
    bars = ax.bar(labels, piv["总计"], width=0.58, color=COLOR["primary"])
    annotate_bars(ax, piv["总计"].values, bars=bars, fmt="{:,.0f}", offset=0.02,
                  fontsize=10.5)
    ax.set_ylim(0, piv["总计"].max() * 1.24)
    ax.set_ylabel("贷款申请量 (笔)")
    ax.set_xlabel("")
    ax2 = ax.twinx()
    ax2.plot(labels, piv["环比"], color=COLOR["negative"], marker="o", lw=2.2)
    for i, v in enumerate(piv["环比"]):
        if np.isfinite(v):
            ax2.annotate(f"{v:+.1%}", (i, v), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9.5,
                         color=COLOR["negative"], fontweight="bold")
    ax2.axhline(0, color=COLOR["text_soft"], lw=0.8, ls=":")
    ax2.yaxis.set_major_formatter(FuncFormatter(pct))
    ax2.set_ylabel("环比变化率", color=COLOR["negative"])
    ax2.tick_params(axis="y", colors=COLOR["negative"])
    ax2.grid(False)
    ax.set_title("贷款申请量: 近 7 个月趋势与环比")
    style_axes_clean(ax)
    add_source(fig, "数据来源: 归因分析数据.xlsx 贷款申请表 (1,597 条)  |  "
                    "对应原报告图 1-1", y=0.005)
    save_fig(fig, "ch05_06_申请量趋势")
    return piv


def b2_channel(piv: pd.DataFrame) -> None:
    """图 ch05_07: 各渠道申请构成 (对应原报告图 1-2)。"""
    rule("案例B · ② 渠道构成")
    ch = [c for c in piv.columns if c not in ("总计", "环比")]
    share = piv[ch].div(piv["总计"], axis=0)
    print("渠道占比:\n", share.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2), width_ratios=[1.35, 1.0],
                             gridspec_kw={"wspace": 0.26})
    ax = axes[0]
    x = np.arange(len(piv))
    bottom = np.zeros(len(piv))
    for i, c in enumerate(ch):
        v = piv[c].values
        ax.bar(x, v, bottom=bottom, width=0.62, label=c, color=PALETTE[i])
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in piv.index], rotation=30, ha="right")
    ax.set_title("① 各渠道申请量 (堆叠)")
    ax.set_ylabel("申请量 (笔)")
    ax.legend(loc="upper left", ncol=1, fontsize=9)
    style_axes_clean(ax)

    ax = axes[1]
    for i, c in enumerate(ch):
        ax.plot([str(j) for j in piv.index], share[c].values, marker="o",
                color=PALETTE[i], label=c, lw=2.0)
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_title("② 各渠道占比变化")
    ax.set_ylabel("占比")
    ax.legend(loc="upper right", ncol=1, fontsize=9)
    ax.set_ylim(0, 1)
    style_axes_clean(ax)

    fig.suptitle("渠道构成: 手机银行与柜台贡献约九成申请量",
                 y=1.02, x=0.02, ha="left")
    add_source(fig, "数据来源: 归因分析数据.xlsx 贷款申请表  |  对应原报告图 1-2", y=0.005)
    save_fig(fig, "ch05_07_渠道构成")


def b3_attribution(piv: pd.DataFrame) -> pd.DataFrame:
    """图 ch05_08: 渠道贡献度 (对应原报告图 1-3)。"""
    rule("案例B · ③ 归因贡献度")
    m4, m5 = piv.loc[piv.index[-2]], piv.loc[piv.index[-1]]
    ch = [c for c in piv.columns if c not in ("总计", "环比")]
    diff = m5[ch] - m4[ch]
    total_diff = int(m5["总计"] - m4["总计"])
    contrib = diff / total_diff
    res = pd.DataFrame({"上月": m4[ch], "本月": m5[ch], "差额": diff,
                        "贡献度": contrib}).sort_values("贡献度")
    print(res.round(4).to_string())
    print(f"\n整体: {m4['总计']} -> {m5['总计']} ({total_diff:+d}, {total_diff/m4['总计']:+.1%})")

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.2), width_ratios=[1, 1.05],
                             gridspec_kw={"wspace": 0.32})
    ax = axes[0]
    vals = res["贡献度"].values
    ax.barh(res.index, vals, height=0.58,
            color=[COLOR["negative"] if v > 0 else COLOR["positive"] for v in vals])
    annotate_hbars(ax, vals, fmt="{:+.2%}", pad=0.02, fontsize=11)
    ax.set_xlim(min(vals.min() * 1.5, -0.1), max(vals.max() * 1.35, 0.7))
    ax.axvline(0, color=COLOR["text_soft"], lw=1.0)
    ax.set_title("① 各渠道对整体下滑的贡献度")
    ax.set_xlabel("贡献度 (渠道差额 / 整体差额)")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(FuncFormatter(pct))
    style_axes_clean(ax)

    ax = axes[1]
    labels = ["上月\n2021-04", "手机银行", "理财经理\n(柜台)", "自助机", "本月\n2021-05"]
    steps = [m4["总计"], -diff["手机银行"], -diff["理财经理（柜台）"], -diff["自助机"], m5["总计"]]
    # 用"减少量"画瀑布, 视觉上更符合"下滑"的叙事
    base = m4["总计"]
    bottoms, heights, colors = [], [], []
    cur = base
    for i, (lab, dlt) in enumerate(zip(labels, steps)):
        if i in (0, len(labels) - 1):
            bottoms.append(0); heights.append(dlt); colors.append(COLOR["primary"])
            cur = dlt if i == 0 else cur
        else:
            bottoms.append(cur - abs(dlt)); heights.append(abs(dlt))
            colors.append(COLOR["negative"] if dlt > 0 else COLOR["positive"])
            cur -= dlt
    ax.bar(range(len(labels)), heights, bottom=bottoms, width=0.62, color=colors)
    for i, (b, h) in enumerate(zip(bottoms, heights)):
        if i in (0, len(labels) - 1):
            txt = f"{h:,.0f}"
        else:
            # steps 存的是"减少量", 为负表示该渠道反而增加了申请量
            txt = f"{h:,.0f}" if steps[i] > 0 else f"+{h:,.0f}"
        ax.text(i, b + h + 6, txt, ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=COLOR["text"])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, base * 1.16)
    ax.set_title("② 申请量下降的渠道归因 (瀑布图)")
    ax.set_ylabel("申请量 (笔)")
    style_axes_clean(ax)

    fig.suptitle("归因分析: 柜台 58.97% + 手机银行 48.72%, 自助机反向抵消 -7.69%",
                 y=1.02, x=0.02, ha="left")
    add_source(fig, "数据来源: 归因分析数据.xlsx 贷款申请表  |  "
                    "贡献度之和为 1, 对应原报告图 1-3", y=0.005)
    save_fig(fig, "ch05_08_归因贡献度")
    return res


# ============================================================================ #
# 主入口
# ============================================================================ #

def main() -> int:
    font = setup_style()
    print("=" * 78)
    print("《商业数据分析》第5章 业务数据分析 —— 银行看板 + 归因分析")
    print("=" * 78)
    print(f"Python    : {sys.version.split()[0]}")
    print(f"中文字体  : {font}")
    print(f"图片输出  : {FIG}  (PNG, 300dpi)")

    for f in (BI_DIR / "银行案例数据 - 学员用.xlsx",
              ATTR_DIR / "归因分析数据.xlsx"):
        if not f.exists():
            print(f"\n[错误] 找不到教材文件: {f}")
            return 2

    d = load_bi()
    rule("案例A · 业务背景")
    print("银行理财产品销售增长乏力; 本月目标 2,500 万; 当前为 29 日时点数据 (共 31 天)。")
    kpi = a1_kpi(d)
    a2_category(d)
    a3_ticket_term(d)
    a4_profile(d)
    a5_reason(d)

    a = load_attr()
    rule("案例B · 业务背景")
    print("某区域性商业银行 2021 年 5 月贷款申请量环比下降, 需说明下滑原因。")
    piv = b1_trend(a)
    b2_channel(piv)
    res = b3_attribution(piv)

    rule("结论")
    print(f"""
案例A 银行理财看板
  1. 29 日累计销售 {kpi['total']/1e4:,.0f} 万元, 达成率 {kpi['rate']:.1%},
     略快于时间进度 {DAYS_NOW/DAYS_TOTAL:.1%}
  2. 按日均线性外推, 月末约 {kpi['forecast']/1e4:,.0f} 万元
     ({kpi['forecast']/TARGET:.1%} 目标), {'可以达标' if kpi['forecast'] >= TARGET else '仍有缺口'}
  3. 客群高度集中: 31-50 岁、家庭年收入 10-20 万、本科学历;
     选择本行的首要原因是"预期年化收益率高"

案例B 贷款申请归因
  1. 5 月申请量 {piv['总计'].iloc[-1]} 笔, 环比 {piv['环比'].iloc[-1]:+.1%}
  2. 下滑主要由两个渠道贡献:
     理财经理(柜台) {res.loc['理财经理（柜台）','贡献度']:+.2%}、
     手机银行 {res.loc['手机银行','贡献度']:+.2%}
     自助机 {res.loc['自助机','贡献度']:+.2%} (反向抵消)
  3. 归因结论指向"渠道迁移未完成": 柜台与 APP 同时下滑, 自助机未承接住
  4. 对应两项策略 + A/B 测试设计 (见原报告):
     自助机宣传/协调流程、APP 首页广告位
""")
    print("图片已输出到:", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
