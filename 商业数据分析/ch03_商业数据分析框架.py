# -*- coding: utf-8 -*-
r"""
《商业数据分析》教材重构 —— 第3章 商业数据分析框架
================================================================================
原教材实现: **Excel** (3 个案例, 每个案例均为 .xlsx)
本文件实现: Python 单文件版 (pandas + matplotlib), 可直接运行
================================================================================

运行方式::

    C:\Users\dell\anaconda3\envs\dataviz\python.exe -X utf8 ch03_商业数据分析框架.py

输出: 与脚本同目录的 PNG 图片 (300dpi)

教材的三段式分析框架
--------------------
每个案例 (xlsx) 都按同一套框架组织, 本脚本沿用同样的骨架:

    业务需求  ->  原始数据  ->  准备区 (数据加工/透视)  ->  报表界面 (图表)
    问题的界定     事实与口径     计算与分析口径            结论与表达

三个案例
--------
案例1 收入趋势分析     某线上超市 2017-2020 年各省份月度销售 (1,187 行)
                       需求: 判断收入是否放缓, 若放缓, 定位问题点
案例2 财务费用分析     某零售企业 2020年6月/7月财务费用
                       需求: 找出影响财务费用"大幅增长"的问题点
案例3 电商精准营销     某电商平台 2023年1月客户数据 (3,926 人)
                       需求: 在男性/女性中二选一, 给出具体促销方向

迁移中发现并修正的三处口径问题 (原 Excel 的"准备区"环节)
--------------------------------------------------------
1. **同比公式分母用错** (案例1)
   原表: =(当年-上年)/当年      正解: =(当年-上年)/上年
   后果: 3 月同比 +48.0% 被低估 (真值 +92.4%); 四川 11 月出现 -821.7%,
   跌幅超过 -100% 在逻辑上不成立。

2. **总费用与三大期间费用不是同一口径** (案例2)
   总费用 6->7 月增长 78,600 元, 但"三大期间费用"只增长 14,350 元,
   剩余 64,250 元 (81.7%) 无法由三大费用解释; 且 6 月总费用 (1,560,000)
   反而小于三大费用之和 (1,564,900)。

3. **年龄段透视的基数错误** (案例3)
   原"女性消费客户的年龄段分布"图的数据合计 1,555 人, 恰好等于男性总数;
   而女性真实人数为 2,371 人。该透视表与同案例第 3 步 (50岁以上女性 918 人)
   自相矛盾。本脚本按原始数据重算。

教学价值: 这三个问题都不是"代码写错", 而是**分析口径错误** ——
不报错、能出图、结论看似合理, 却是错的。这正是"准备区"环节最需要校验的地方。
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
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

# ============================================================================ #
# 0. 路径与运行环境
# ============================================================================ #

HERE = Path(__file__).resolve().parent      # 本章交付目录
ROOT = HERE.parent                          # 项目根目录 (教材原文所在位置)
CH3_DIR = ROOT / "第3章 商业数据分析框架"
FIG = HERE
FIG.mkdir(parents=True, exist_ok=True)

CASE1 = CH3_DIR / "案例1：收入趋势分析"
CASE2 = CH3_DIR / "案例2：财务费用分析"
CASE3 = CH3_DIR / "案例3：电商平台的精准营销"


def xlsx(case_dir: Path, keyword: str) -> Path:
    """在案例目录中按关键词定位工作簿。"""
    hits = [p for p in case_dir.glob("*.xlsx") if keyword in p.name]
    if not hits:
        raise FileNotFoundError(f"{case_dir} 下找不到含 '{keyword}' 的 xlsx")
    return hits[0]


#: 原始数据工作簿 (教材单独提供, 用于核对)
RAW1 = xlsx(CASE1, "原始数据")
#: 完整案例工作簿 (含 业务需求/原始数据/准备区/报表界面 四个工作表)
FULL1 = xlsx(CASE1, "完整")
FULL2 = xlsx(CASE2, "完整")
FULL3 = xlsx(CASE3, "完整")

# conda 环境未激活时, matplotlib 依赖的 DLL 位于 <env>\Library\bin, 不在默认
# 搜索路径中, 导出 PNG 时会直接崩溃 (Windows fatal exception 0xc06d007f)。
if sys.platform == "win32":
    _lib_bin = Path(sys.prefix) / "Library" / "bin"
    if _lib_bin.is_dir() and str(_lib_bin).lower() not in os.environ.get("PATH", "").lower():
        os.environ["PATH"] = f"{_lib_bin}{os.pathsep}{os.environ.get('PATH', '')}"


def rule(title: str) -> None:
    """打印分节标题, 便于对照教材的四个工作表阅读日志。"""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
# ============================================================================ #
# 1. 绘图主题 (内置, 与第1章脚本保持同一套视觉规范)
# ============================================================================ #

PALETTE = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1",
           "#76B7B2", "#EDC948", "#9C755F", "#BAB0AC", "#FF9DA7"]

COLOR = {
    "primary": "#4E79A7", "accent": "#F28E2B", "positive": "#59A14F",
    "negative": "#E15759", "neutral": "#BAB0AC", "highlight": "#B07AA1",
    "muted": "#8FAFC9", "grid": "#E4E7EB", "text": "#2F3B45",
    "text_soft": "#7A8794", "bg": "#FFFFFF", "bg_soft": "#F7F9FB",
}

SEQ_CMAP = LinearSegmentedColormap.from_list(
    "biz_seq", ["#EAF1F8", "#A9C4DF", "#6E97C2", "#4E79A7", "#2C4A66"])
matplotlib.colormaps.register(SEQ_CMAP, force=True)

_CJK_CANDIDATES = ["Microsoft YaHei", "微软雅黑", "Source Han Sans SC",
                   "Noto Sans CJK SC", "SimHei", "黑体", "DengXian", "SimSun"]


def cjk_font_name(fallback: str = "DejaVu Sans") -> str:
    """返回本机可用的第一个中文字体名。"""
    try:
        from matplotlib import font_manager
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in _CJK_CANDIDATES:
            if name in available:
                return name
    except Exception:
        pass
    return fallback


def setup_style(dpi: int = 300, figsize: tuple[float, float] = (11, 6.2)) -> str:
    """应用全局绘图主题: 中文正体 / 语义配色 / 去噪坐标轴 / 高清导出。"""
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
    """淡化左/下轴线, 让网格线承担视觉引导。"""
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR["grid"])


def thousands(x, _pos=None) -> str:
    """千分位 / 万 / 亿 自适应。"""
    if abs(x) >= 1e8:
        return f"{x / 1e8:,.1f}亿"
    if abs(x) >= 1e4:
        return f"{x / 1e4:,.0f}万"
    return f"{x:,.0f}"


def currency(x, _pos=None) -> str:
    """人民币金额, 自动切换 元 / 万。"""
    if abs(x) >= 1e4:
        return f"¥{x / 1e4:,.0f}万"
    return f"¥{x:,.0f}"


def wan(x) -> str:
    """纯千分位 (不切换单位), 用于数值区间较窄、切换单位会产生重复刻度的场景。"""
    return f"{x:,.0f}"


def apply_axis_format(ax, y=None, x=None) -> None:
    """给坐标轴套格式化函数。

    fmt 可取: "thousands"(万/亿自适应) / "currency" / "plain"(纯千分位) / None。
    注意: 类别轴上必须显式传 None, 否则数值刻度定位器会覆盖类别名。
    """
    table = {"thousands": thousands, "currency": currency,
             "plain": wan, None: None}
    if table.get(y) is not None:
        ax.yaxis.set_major_formatter(FuncFormatter(table[y]))
    if table.get(x) is not None:
        ax.xaxis.set_major_formatter(FuncFormatter(table[x]))
# ------------------------------- 绘图工具 --------------------------------- #

def annotate_bars(ax, values=None, bars=None, fmt="{:,.0f}", offset=0.015,
                  inside=False, color=None, fontsize=10, fontweight="bold"):
    """在柱状图上标注数值。values 必须与柱高使用同一单位。"""
    bars = list(bars) if bars is not None else list(ax.patches)
    if values is None:
        values = [b.get_height() for b in bars]
    values = [float(v) for v in values]
    finite = [v for v in values if np.isfinite(v)] or [0.0]
    span = (max(finite) - min(finite + [0.0])) or 1.0
    for bar, val in zip(bars, values):
        if not np.isfinite(val):
            continue
        x = bar.get_x() + bar.get_width() / 2
        if inside:
            ax.text(x, bar.get_y() + val / 2, fmt.format(val), ha="center",
                    va="center", fontsize=fontsize, color="white",
                    fontweight=fontweight, clip_on=False)
        else:
            ax.text(x, val + span * offset, fmt.format(val), ha="center",
                    va="bottom", fontsize=fontsize,
                    color=color or COLOR["text_soft"],
                    fontweight=fontweight, clip_on=False)


def annotate_hbars(ax, values, fmt="{:,.0f}", pad=0.012, color=None,
                   fontsize=10, fontweight="bold"):
    """在横向条形图右侧标注数值。"""
    values = [float(v) for v in values]
    finite = [v for v in values if np.isfinite(v)] or [0.0]
    span = (max(finite) - min(finite + [0.0])) or 1.0
    for bar, val in zip(ax.patches[-len(values):], values):
        if not np.isfinite(val):
            continue
        ax.text(val + span * pad, bar.get_y() + bar.get_height() / 2,
                fmt.format(val), va="center", ha="left", fontsize=fontsize,
                color=color or COLOR["text"], fontweight=fontweight)


def add_source(fig, text: str, y: float = 0.0) -> None:
    """在图片底部添加数据来源注释 (教材图必备)。"""
    fig.text(0.005, y, text, fontsize=9, color=COLOR["text_soft"],
             ha="left", va="top")


def save_fig(fig, name: str, out_dir: Path = FIG) -> list[Path]:
    """保存 PNG 图片 (300dpi)。"""
    p = out_dir / f"{name}.png"
    fig.savefig(p, format="png")
    plt.close(fig)
    return [p]


def note_box(ax, x, y, text, face=COLOR["bg_soft"], edge=COLOR["grid"],
             ha="left", va="top", fontsize=10.5):
    """在坐标区内放一个浅色说明框。"""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize,
            color=COLOR["text"], linespacing=1.7,
            bbox=dict(boxstyle="round,pad=0.5", fc=face, ec=edge, lw=1.0))
# ============================================================================ #
# 2. 案例1: 收入趋势分析
# ============================================================================ #

def load_case1() -> pd.DataFrame:
    """读取案例1 原始数据 (订单年份, 订单月份, 销售省份, 销售额)。"""
    df = pd.read_excel(RAW1, sheet_name="原始数据")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def case1_overview(df: pd.DataFrame) -> pd.DataFrame:
    """图 ch03_01: 年度趋势 + 2020 年月度趋势 (对应原报表第 1、2 张图)。"""
    ann = df.groupby("订单年份")["销售额"].sum()
    month = (df[df["订单年份"] == 2020].groupby("订单月份")["销售额"].sum())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.0), width_ratios=[1, 1.3],
                             gridspec_kw={"wspace": 0.22})

    # ① 年度销售趋势 (原 lineChart, 这里用柱 + 增长率标签, 信息量更高)
    ax = axes[0]
    years = ann.index.astype(int)
    bars = ax.bar(years, ann.values / 1e4, color=COLOR["primary"], width=0.55)
    annotate_bars(ax, ann.values / 1e4, bars=bars, fmt="{:,.0f}万", offset=0.02)
    growth = ann.pct_change()
    # 柱宽有限, 标签只放百分比; "同比"的口径由左图注与数据来源说明交代
    for x, yv, g in zip(years, ann.values / 1e4, growth.values):
        if np.isfinite(g):
            ax.text(x, yv * 0.5, f"{g:+.1%}", ha="center", va="center",
                    fontsize=12.5, color="white", fontweight="bold")
    ax.set_ylim(0, ann.max() / 1e4 * 1.22)
    ax.set_xticks(years)
    ax.set_title("① 年度销售额趋势")
    ax.set_xlabel("年份")
    ax.set_ylabel("销售额 (万元)")
    style_axes_clean(ax)
    note_box(ax, 0.03, 0.96,
             f"4 年累计 {ann.sum() / 1e4:,.0f} 万元\n"
             f"逐年增速 {growth.iloc[1]:.1%} → {growth.iloc[3]:.1%}, 未见放缓",
             face="#EDF7EC", edge=COLOR["positive"])

    # ② 2020 年月度趋势 (原 lineChart)
    ax = axes[1]
    months = month.index.astype(int)
    ax.fill_between(months, month.values / 1e4, color=COLOR["primary"], alpha=0.18)
    ax.plot(months, month.values / 1e4, marker="o", color=COLOR["primary"], lw=2.2)
    lo = int(month.idxmin())
    hi = int(month.idxmax())
    for m, tag in [(lo, "最低"), (hi, "最高")]:
        ax.scatter([m], [month[m] / 1e4], s=90, color=COLOR["negative"], zorder=5,
                   edgecolor="white", linewidth=1.4)
        ax.annotate(f"{tag} {m}月\n{month[m] / 1e4:,.0f}万",
                    (m, month[m] / 1e4), textcoords="offset points",
                    xytext=(0, 16 if m == hi else -32), ha="center", fontsize=10,
                    fontweight="bold", color=COLOR["text"])
    ax.set_xticks(months)
    ax.set_xticklabels([f"{m}月" for m in months])
    ax.set_ylim(0, month.max() / 1e4 * 1.25)
    ax.set_title("② 2020 年月度销售额走势")
    ax.set_xlabel("月份")
    ax.set_ylabel("销售额 (万元)")
    style_axes_clean(ax)

    fig.suptitle("案例1 · 收入趋势: 整体在增长, 但个别月份出现异常",
                 y=1.02, x=0.02, ha="left")
    add_source(fig, "数据来源: 案例1：收入趋势分析原始数据.xlsx (1,187 行, 31 个省份, 2017-2020)  |  "
                    "对应原报表第 1、2 张图", y=0.005)
    save_fig(fig, "ch03_01_收入趋势总览")
    return ann


def case1_yoy(df: pd.DataFrame) -> pd.DataFrame:
    """图 ch03_02: 2020 年同比增长率 (含原表公式修正) + 11 月各省份同比。"""
    p = (df[df["订单年份"].isin([2019, 2020])]
         .pivot_table(index="订单月份", columns="订单年份",
                      values="销售额", aggfunc="sum"))
    p["同比_正确"] = (p[2020] - p[2019]) / p[2019]      # 正解: 分母用上年
    p["同比_原表"] = (p[2020] - p[2019]) / p[2020]      # 原表: 分母用当年

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 6.2), width_ratios=[1.15, 1.0],
                             gridspec_kw={"wspace": 0.30})

    # ① 同比增长率: 正确口径 vs 原表口径
    ax = axes[0]
    months = p.index.astype(int)
    w = 0.38
    ax.bar(months - w / 2, p["同比_正确"].values, width=w, label="正确口径 (分母=上年)",
           color=COLOR["primary"])
    ax.bar(months + w / 2, p["同比_原表"].values, width=w, label="原表口径 (分母=当年)",
           color=COLOR["muted"])
    for m in months:
        v = p.loc[m, "同比_正确"]
        if v < 0:      # 负值标签放进柱内, 避免与下方注解打架
            ax.text(m, v / 2, f"{v:.0%}", ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
        else:
            ax.text(m, v + 0.03, f"{v:.0%}", ha="center", va="bottom",
                    fontsize=9, color=COLOR["text"])
    ax.axhline(0, color=COLOR["text_soft"], lw=1.0)
    ax.set_xticks(months)
    ax.set_xticklabels([f"{m}月" for m in months])
    ax.set_ylim(-0.30, 1.12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("① 2020 年各月同比增长率: 口径修正前后对比")
    ax.set_xlabel("月份")
    ax.set_ylabel("同比增长率")
    ax.legend(loc="upper left", ncol=1)
    style_axes_clean(ax)
    note_box(ax, 0.985, 0.965,
             "11 月是 2020 年唯一的负增长月份\n两种口径的结论方向一致",
             face="#FDECEC", edge=COLOR["negative"], ha="right")
    worst = int(p["同比_正确"].idxmin())
    ax.annotate(f"{worst} 月\n{p.loc[worst, '同比_正确']:.1%}",
                xy=(worst, p.loc[worst, "同比_正确"] * 0.9),
                xytext=(worst + 0.75, -0.235), fontsize=10.5,
                color=COLOR["negative"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLOR["negative"], lw=1.8))

    # ② 11 月各省份同比 (原 barChart)
    ax = axes[1]
    n = (df[(df["订单月份"] == 11) & (df["订单年份"].isin([2019, 2020]))]
         .pivot_table(index="销售省份", columns="订单年份",
                      values="销售额", aggfunc="sum"))
    n = n[n[2019].notna()].fillna({2020: 0})
    n["同比"] = (n[2020] - n[2019]) / n[2019]
    n = n.sort_values("同比")
    # 只画下滑的省份: 增长侧存在 +11462% 这类"小基数"极值, 会把其余省份压成一条线,
    # 而这幅图的分析目标是"找出拖累整体的问题省份", 因此聚焦下滑侧。
    show = n[n["同比"] < 0]
    labels = [f"{i}" + ("*" if n.loc[i, 2020] == 0 else "") for i in show.index]
    vals = show["同比"].values
    ax.barh(labels, vals, height=0.64, color=COLOR["negative"])
    annotate_hbars(ax, vals, fmt="{:+.0%}", pad=0.022, fontsize=9.5)
    ax.set_xlim(vals.min() * 1.32, 0.08)
    ax.set_title(f"② 2020 年 11 月同比下滑的省份 (共 {len(show)} 个)")
    ax.set_xlabel("同比增长率")
    ax.set_ylabel("省份")
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    style_axes_clean(ax)
    grow = n[n["同比"] >= 0]
    # 放到左上角: 上部柱子短, 左侧留白足够, 不会压住底部的长柱
    note_box(ax, 0.02, 0.97,
             f"31 个省份中 {len(show)} 个下滑、{len(grow)} 个增长\n"
             f"增长侧最高为甘肃 +11462% (2019年11月基数仅 157 元)",
             ha="left", va="top", fontsize=9.5)

    fig.suptitle("案例1 · 定位问题点: 11 月负增长, 四川/吉林/内蒙古跌幅最深",
                 y=1.02, x=0.02, ha="left")
    add_source(fig, "数据来源: 案例1 原始数据  |  * 表示该省份 2020 年 11 月无销售记录  |  "
                    "原表公式 =(当年-上年)/当年, 本脚本已修正为 /上年", y=0.005)
    save_fig(fig, "ch03_02_同比增长率")
    return p, n

def case1_drilldown(df: pd.DataFrame) -> None:
    """图 ch03_03: 下钻到四川省, 看各月同比 (对应原报表第 5 张图)。"""
    sc = (df[(df["销售省份"] == "四川") & (df["订单年份"].isin([2019, 2020]))]
          .pivot_table(index="订单月份", columns="订单年份",
                       values="销售额", aggfunc="sum").fillna(0))
    sc["同比"] = (sc[2020] - sc[2019]) / sc[2019].replace(0, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.0), width_ratios=[1.25, 1.0],
                             gridspec_kw={"wspace": 0.24})

    ax = axes[0]
    months = sc.index.astype(int)
    w = 0.38
    bars19 = ax.bar(months - w / 2, sc[2019].values / 1e4, width=w,
                    label="2019 年", color=COLOR["muted"])
    bars20 = ax.bar(months + w / 2, sc[2020].values / 1e4, width=w,
                    label="2020 年", color=COLOR["primary"])
    annotate_bars(ax, sc[2020].values / 1e4, bars=bars20, fmt="{:,.1f}",
                  offset=0.02, fontsize=8)
    ab = int(sc["同比"].idxmin())
    # 用背景高亮条标出异常月份, 比画一条横穿全图的箭头更干净
    ax.add_patch(Rectangle((ab - 0.5, 0), 1.0, sc[2019].max() / 1e4 * 1.5,
                           facecolor=COLOR["negative"], alpha=0.09, zorder=0))
    ax.set_xticks(months)
    ax.set_xticklabels([f"{m}月" for m in months])
    ax.set_ylim(0, sc[2019].max() / 1e4 * 1.50)
    note_box(ax, 0.02, 0.98,
             f"{ab} 月几乎归零: {sc.loc[ab, 2020] / 1e4:,.1f} 万 vs 上年 "
             f"{sc.loc[ab, 2019] / 1e4:,.1f} 万 ({sc.loc[ab, '同比']:.0%})",
             face="#FDECEC", edge=COLOR["negative"], ha="left", va="top",
             fontsize=10.5)
    ax.set_title("① 四川省 2019 vs 2020 各月销售额")
    ax.set_xlabel("月份")
    ax.set_ylabel("销售额 (万元)")
    ax.legend(loc="upper right", ncol=2)      # 左上已被说明框占用
    style_axes_clean(ax)

    ax = axes[1]
    vals = sc["同比"].values
    y_lo, y_hi = -1.15, 3.30        # 统一用百分比表达, 并截断 12 月的基数极值
    bars = ax.bar(months, vals, width=0.6,
                  color=[COLOR["negative"] if v < 0 else COLOR["positive"] for v in vals])
    for b, v in zip(bars, vals):
        if not np.isfinite(v):
            continue
        xc = b.get_x() + b.get_width() / 2
        if v > y_hi:                # 超出坐标范围, 星号标注, 真值见数据来源
            ax.text(xc, y_hi * 0.97, f"{v:+.0%}*", ha="right", va="top",
                    fontsize=9.5, color=COLOR["negative"], fontweight="bold")
        elif v < 0:
            ax.text(xc, v - 0.06, f"{v:+.0%}", ha="center", va="top",
                    fontsize=8.5, color=COLOR["text"])
        else:
            ax.text(xc, v + 0.06, f"{v:+.0%}", ha="center", va="bottom",
                    fontsize=8.5, color=COLOR["text"])
    ax.axhline(0, color=COLOR["text_soft"], lw=1.0)
    ax.set_xticks(months)
    ax.set_xticklabels([f"{m}月" for m in months])
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("② 四川省各月同比增长率")
    ax.set_xlabel("月份")
    ax.set_ylabel("同比增长率")
    style_axes_clean(ax)

    fig.suptitle("案例1 · 下钻: 四川 11 月销售几乎归零, 12 月大幅补偿",
                 y=1.02, x=0.02, ha="left")
    add_source(fig, "数据来源: 案例1 原始数据  |  对应原报表第 5 张图  |  "
                    "* 12 月同比 +3080% 超出坐标范围 (2019 年 12 月基数仅 682 元)", y=0.005)
    save_fig(fig, "ch03_03_四川下钻")


def run_case1() -> dict:
    """案例1 完整流程: 业务需求 -> 原始数据 -> 准备区 -> 报表界面。"""
    rule("案例1 · 业务需求")
    print("现提供某线上超市平台 2017-2020 年各省份每月销售数据, 请对该平台的收入趋势进行分析,")
    print("并得出该平台的销售状况; 若收入出现放缓, 要求通过数据分析找到收入放缓的问题点。")

    rule("案例1 · 原始数据")
    df = load_case1()
    print(f"行数 {len(df):,}, 省份 {df['销售省份'].nunique()} 个, "
          f"年份 {sorted(df['订单年份'].unique())}, 月份 {df['订单月份'].min()}-{df['订单月份'].max()}")

    rule("案例1 · 准备区 ① 年度与月度趋势")
    ann = case1_overview(df)
    print(ann.to_string())

    rule("案例1 · 准备区 ② 同比增长率")
    p, n = case1_yoy(df)
    out = p.copy()
    out.index.name = "月份"
    print(out.round(4).to_string())
    print(f"\n[口径修正] 11 月同比: 原表 {p.loc[11, '同比_原表']:.2%} "
          f"-> 正确 {p.loc[11, '同比_正确']:.2%}")
    print(f"[口径修正] 3 月同比: 原表 {p.loc[3, '同比_原表']:.2%} "
          f"-> 正确 {p.loc[3, '同比_正确']:.2%}")
    print(f"[口径修正] 四川 11 月: 原表 -821.7% (跌幅超过 -100%, 逻辑不成立) "
          f"-> 正确 {n.loc['四川', '同比']:.2%}")
    print(f"2020 年唯一负增长月份: {int(p['同比_正确'].idxmin())} 月 "
          f"({p['同比_正确'].min():.2%})")
    print("11 月跌幅最大的 5 个省份 (正确口径):")
    print(n.sort_values("同比").head(5).round(4).to_string())

    rule("案例1 · 准备区 ③ 下钻到四川省")
    case1_drilldown(df)
    sub = df[(df["销售省份"] == "四川") & (df["订单年份"] == 2020)]
    print(f"四川 2020 年 10 月销售额: "
          f"{sub[sub['订单月份'] == 10]['销售额'].sum():,}")
    print(f"四川 2020 年 11 月销售额: "
          f"{sub[sub['订单月份'] == 11]['销售额'].sum():,}")
    print(f"四川 2020 年 12 月销售额: "
          f"{sub[sub['订单月份'] == 12]['销售额'].sum():,}")

    rule("案例1 · 结论")
    print(f"""
1. 平台整体收入并未放缓: 2017-2020 年逐年增长
   {ann.iloc[0]:,} -> {ann.iloc[1]:,} -> {ann.iloc[2]:,} -> {ann.iloc[3]:,} 元
   年均增速 {(ann.iloc[3] / ann.iloc[0]) ** (1 / 3) - 1:.1%}

2. 问题点在"结构性异常"而非"整体放缓"
   - 2020 年 11 月是全年唯一负增长月份 ({p.loc[11, '同比_正确']:.1%})
   - 跌幅最深的省份: {', '.join(n.sort_values('同比').head(5).index.tolist())}

3. 下钻到四川: 11 月销售额从上年 18,019 元塌陷到 1,955 元 (下滑 89.2%),
   12 月又反弹到 21,686 元, 属于异常波动而非趋势性下滑,
   建议核查 11 月的库存/供应/数据上报情况。

4. 原 Excel 的同比公式 =(当年-上年)/当年 分母用错, 应改为 /上年;
   该错误会让"跌幅超过 -100%"(如四川 -821.7%)这种不可能的结果出现。
""")
    return {"ann": ann, "yoy": p, "prov": n}

# ============================================================================ #
# 3. 案例2: 财务费用分析
# ============================================================================ #

def load_case2() -> dict:
    """读取案例2 的三张原始数据表。"""
    book = pd.read_excel(FULL2, sheet_name=None)
    total = book["总费用数据-原始数据"].dropna(how="all").dropna(axis=1, how="all")
    periods = book["财务三大期间费用-原始数据"].dropna(how="all")
    manage = book["管理费用-原始数据"].dropna(how="all")
    for d in (periods, manage):
        d.iloc[:, 2] = d.iloc[:, 2].astype(str).str.strip()   # 原数据含前导空格
    return {"total": total, "periods": periods, "manage": manage}


def case2_overview(data: dict) -> dict:
    """图 ch03_04: 总费用对比 + 三大期间费用环比差额。"""
    total, periods = data["total"], data["periods"]

    def value(a, b=None):
        return total.iloc[a, 2]

    t6, t7 = value(0), value(1)
    piv = periods.pivot_table(index=periods.columns[2], columns=periods.columns[1],
                              values=periods.columns[3], aggfunc="sum")
    piv["差额"] = piv[7] - piv[6]
    piv["环比"] = piv["差额"] / piv[6]
    piv = piv.sort_values("差额")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.1), width_ratios=[0.85, 1.15],
                             gridspec_kw={"wspace": 0.26})

    # ① 总财务费用
    ax = axes[0]
    bars = ax.bar(["6月", "7月"], [t6 / 1e4, t7 / 1e4], width=0.5,
                  color=[COLOR["primary"], COLOR["accent"]])
    annotate_bars(ax, [t6 / 1e4, t7 / 1e4], bars=bars, fmt="{:,.1f}万", offset=0.03)
    ax.set_ylim(0, t7 / 1e4 * 1.28)
    ax.set_title("① 总财务费用: 6 月 vs 7 月")
    ax.set_xlabel("")
    ax.set_ylabel("金额 (万元)")
    # 箭头落在柱顶数值标签之上, 避免两者叠在一起
    ax.annotate(f"环比 +{(t7 - t6) / 1e4:,.1f} 万元\n{(t7 / t6 - 1):+.2%}",
                xy=(0.96, t7 / 1e4 * 1.055), xytext=(0.24, t7 / 1e4 * 1.155),
                fontsize=11, color=COLOR["negative"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLOR["negative"], lw=1.8))
    style_axes_clean(ax)

    # ② 三大期间费用差额 (原 barChart)
    ax = axes[1]
    y = np.arange(len(piv))
    vals = piv["差额"].values
    ax.barh([str(i) for i in piv.index], vals, height=0.55,
            color=[COLOR["positive"] if v > 0 else COLOR["negative"] for v in vals])
    annotate_hbars(ax, vals, fmt="{:+,.0f}", pad=0.03, fontsize=10.5)
    span = max(abs(vals).max(), 1)
    ax.set_xlim(min(vals.min() * 1.5, -span * 0.35), span * 1.45)
    ax.axvline(0, color=COLOR["text_soft"], lw=1.0)
    ax.set_title("② 三大期间费用 7 月较 6 月的差额")
    ax.set_xlabel("环比差额 (元)")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    style_axes_clean(ax)

    fig.suptitle("案例2 · 总费用增长 7.86 万元, 但三大期间费用只解释了 1.44 万元",
                 y=1.03, x=0.02, ha="left")
    add_source(fig, "数据来源: 案例2 完整案例.xlsx  总费用数据 / 财务三大期间费用数据  |  "
                    "对应原报表第 1、2 张图", y=0.005)
    save_fig(fig, "ch03_04_财务费用总览")

    return {"t6": t6, "t7": t7, "piv": piv}


def case2_management(data: dict, ov: dict) -> dict:
    """图 ch03_05: 管理费用明细的环比增长率 vs 绝对额 + 增长归因瀑布图。"""
    manage = data["manage"]
    piv = manage.pivot_table(index=manage.columns[2], columns=manage.columns[1],
                             values=manage.columns[3], aggfunc="sum")
    piv["差额"] = piv[7] - piv[6]
    piv["环比"] = piv["差额"] / piv[6]
    piv = piv.sort_values("差额")

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4),
                             gridspec_kw={"wspace": 0.34})

    # ① 环比增长率 (原 barChart: 管理费用7月环比增长分析)
    ax = axes[0]
    rate = piv.sort_values("环比")
    vals = rate["环比"].values
    ax.barh([str(i) for i in rate.index], vals, height=0.6,
            color=[COLOR["negative"] if v < 0 else COLOR["positive"] for v in vals])
    annotate_hbars(ax, vals, fmt="{:+.2%}", pad=0.02, fontsize=9.5)
    ax.set_xlim(min(vals.min() * 1.6, -0.005), max(vals.max() * 1.5, 0.02))
    ax.axvline(0, color=COLOR["text_soft"], lw=1.0)
    ax.set_title("① 看增长率: 咨询费 / 培训与发展 最高")
    ax.set_xlabel("环比增长率")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    style_axes_clean(ax)

    # ② 绝对差额
    ax = axes[1]
    vals = piv["差额"].values
    ax.barh([str(i) for i in piv.index], vals, height=0.6,
            color=[COLOR["negative"] if v < 0 else COLOR["positive"] for v in vals])
    annotate_hbars(ax, vals, fmt="{:+,.0f}", pad=0.03, fontsize=9.5)
    ax.set_xlim(min(vals.min() * 1.6, -1500), max(vals.max() * 1.45, 4000))
    ax.axvline(0, color=COLOR["text_soft"], lw=1.0)
    ax.set_title("② 看绝对额: 咨询费 +10,000 元居首")
    ax.set_xlabel("环比差额 (元)")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    style_axes_clean(ax)

    # ③ 增长归因瀑布: 把"总费用为什么涨了"拆成"三大费用解释的部分 + 未解释缺口"
    ax = axes[2]
    d_total = (ov["t7"] - ov["t6"]) / 1e4
    d_period = (ov["piv"][7].sum() - ov["piv"][6].sum()) / 1e4
    d_gap = d_total - d_period
    steps = [("6月\n总费用", ov["t6"] / 1e4, COLOR["primary"], "合计"),
             ("三大期间费用\n增长", d_period, COLOR["positive"], "增量"),
             ("未解释缺口", d_gap, COLOR["negative"], "增量"),
             ("7月\n总费用", ov["t7"] / 1e4, COLOR["primary"], "合计")]
    bottoms, heights, colors, labels = [], [], [], []
    base = 0.0
    for name, val, col, kind in steps:
        if kind == "合计":
            bottoms.append(0.0)
            heights.append(val)
            base = val
        else:
            bottoms.append(base)
            heights.append(val)
            base += val
        colors.append(col)
        labels.append(name)
    for i in range(len(steps) - 1):        # 连接线, 让瀑布关系一眼可读
        top = bottoms[i] + heights[i] if steps[i][3] == "合计" else bottoms[i] + heights[i]
        ax.plot([i + 0.31, i + 1 - 0.31], [top, top], ls=":", lw=1.2,
                color=COLOR["text_soft"], zorder=0)
    ax.bar(range(len(steps)), heights, bottom=bottoms, width=0.62, color=colors)
    for i, (h, b) in enumerate(zip(heights, bottoms)):
        txt = f"{h:,.1f}万" if steps[i][3] == "合计" else f"{h:+,.2f}万"
        ax.text(i, b + h + max(heights) * 0.03, txt, ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=COLOR["text"])
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, (bottoms[-1] + heights[-1]) * 1.20)
    ax.set_title("③ 总费用增长归因")
    ax.set_ylabel("金额 (万元)")
    style_axes_clean(ax)
    note_box(ax, 0.5, 0.30,
             f"总增长 {d_total:,.2f} 万元中,\n"
             f"三大期间费用只解释 {d_period / d_total:.1%}",
             face="#FDECEC", edge=COLOR["negative"], ha="center", va="top",
             fontsize=10)

    fig.suptitle("案例2 · 下钻: 两种视角结论不同, 且存在 6.43 万元未解释缺口",
                 y=1.02, x=0.02, ha="left")
    add_source(fig, "数据来源: 案例2 完整案例.xlsx  管理费用数据 / 总费用数据 / 财务三大期间费用数据  |  "
                    "对应原报表第 3 张图 (原表只做了增长率视角)", y=0.005)
    save_fig(fig, "ch03_05_管理费用下钻")
    return piv

def run_case2() -> dict:
    """案例2 完整流程。"""
    rule("案例2 · 业务需求")
    print("现提供某零售消费企业 2020 年 6 月、7 月财务费用数据, 请对该企业的财务期间费用进行分析,")
    print("要求通过数据分析找到影响财务费用大幅增长的问题点。")

    rule("案例2 · 原始数据")
    data = load_case2()
    for k, v in data.items():
        print(f"\n[{k}] {v.shape}")
        print(v.to_string(index=False))

    rule("案例2 · 准备区 ① 总费用 + ② 三大期间费用")
    r = case2_overview(data)
    print(f"\n总费用: 6月 {r['t6']:,} -> 7月 {r['t7']:,}  ({r['t7'] - r['t6']:+,})")
    print(r["piv"].round(4).to_string())

    rule("案例2 · 准备区 ③ 管理费用明细")
    piv = case2_management(data, r)
    print(piv.round(4).to_string())

    # ---- 口径一致性校验 (原表未做) ----
    rule("案例2 · 口径校验 (原 Excel 未做的一步)")
    s6, s7 = r["piv"][6].sum(), r["piv"][7].sum()
    gap6, gap7 = r["t6"] - s6, r["t7"] - s7
    gap_growth = (r["t7"] - r["t6"]) - (s7 - s6)
    print(f"三大期间费用合计: 6月 {s6:,.0f} / 7月 {s7:,.0f}")
    print(f"总费用数据      : 6月 {r['t6']:,.0f} / 7月 {r['t7']:,.0f}")
    print(f"两者差额        : 6月 {gap6:+,.0f} / 7月 {gap7:+,.0f}")
    print(f"总费用增长 {r['t7'] - r['t6']:+,.0f} 元, 三大费用增长 {s7 - s6:+,.0f} 元, "
          f"未解释 {gap_growth:+,.0f} 元 ({gap_growth / (r['t7'] - r['t6']):.1%})")

    rule("案例2 · 结论")
    top_rate = piv["环比"].sort_values(ascending=False).head(2)
    print(f"""
1. 表面上"财务费用大幅增长": 总费用 6->7 月 {r['t7'] - r['t6']:+,.0f} 元
   ({(r['t7'] - r['t6']) / r['t6']:+.2%})

2. 但三大期间费用的口径对不上:
   - 总费用增长 {r['t7'] - r['t6']:+,.0f} 元, 三大费用只增长 {s7 - s6:+,.0f} 元
   - 未解释部分 {gap_growth:+,.0f} 元, 占增长的 {gap_growth / (r['t7'] - r['t6']):.1%}
   - 且 6 月总费用 ({r['t6']:,.0f}) 小于三大费用之和 ({s6:,.0f}), 口径本身自相矛盾
   -> 首要动作是核对"总费用数据"的取数范围, 而不是直接解释增长

3. 三大期间费用内部: 管理费用 +{r['piv'].loc['管理费用', '差额']:,.0f} 元是唯一显著增长项,
   销售费用 +{r['piv'].loc['销售费用', '差额']:,.0f} 元基本持平, 财务费用 {r['piv'].loc['财务费用', '差额']:,.0f} 元

4. 管理费用下钻的两种视角给出不同答案:
   - 增长率视角: {', '.join(f"{i} {v:+.2%}" for i, v in top_rate.items())}
   - 绝对额视角: 咨询费 +10,000 元、培训与发展 +5,000 元
   -> 教材原表只做了增长率视角, 会漏掉"绝对额大但增长率一般"的项目;
      实践中应两个视角同时看
""")
    return {"total": r, "manage": piv, "gap": gap_growth}


# ============================================================================ #
# 4. 案例3: 电商平台的精准营销
# ============================================================================ #

def load_case3() -> pd.DataFrame:
    """读取案例3 原始数据 (3,926 行, 一行 = 一位客户)。"""
    df = pd.read_excel(FULL3, sheet_name="原始数据")
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    return df


def case3_profile(df: pd.DataFrame) -> dict:
    """图 ch03_06: 性别分布 + 女性年龄段分布 (并附原表口径对比)。"""
    gender = df["性别"].value_counts()
    fem = df[df["性别"] == "Female"]
    age = (fem["客户年龄段（计算）"].value_counts()
           .reindex(["18-25岁", "26-35岁", "36-49岁", "50岁以上"]))
    # 原表"女性消费客户的年龄段分布"图给出的数值 (合计 1,555)
    origin = pd.Series({"18-25岁": 265, "26-35岁": 313,
                        "36-49岁": 412, "50岁以上": 565})

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), width_ratios=[0.95, 1.25],
                             gridspec_kw={"wspace": 0.24})

    # ① 性别分布 (原 pieChart)
    ax = axes[0]
    sizes = [gender["Female"], gender["Male"]]
    wedges, _, autotexts = ax.pie(
        sizes, labels=["女性", "男性"], autopct=lambda p: f"{p:.2f}%\n({p * sum(sizes) / 100:,.0f} 人)",
        colors=[COLOR["highlight"], COLOR["primary"]], startangle=90,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
        pctdistance=0.79, textprops=dict(fontsize=11, color=COLOR["text"]))
    for t in autotexts:
        t.set_color("white")
        t.set_fontsize(10.5)
        t.set_fontweight("bold")
    ax.text(0, 0.08, f"{sum(sizes):,}", ha="center", va="center", fontsize=20,
            fontweight="bold", color=COLOR["text"])
    ax.text(0, -0.16, "总客户数", ha="center", va="center", fontsize=11,
            color=COLOR["text_soft"])
    ax.set_title("① 购买客户的性别分布")

    # ② 女性年龄段分布 (原 pieChart, 但原表数据口径有误)
    ax = axes[1]
    x = np.arange(len(age))
    w = 0.38
    bars = ax.bar(x - w / 2, age.values, width=w, label="按原始数据重算 (合计 2,371)",
                  color=COLOR["primary"])
    annotate_bars(ax, age.values, bars=bars, fmt="{:,.0f}", offset=0.02)
    bars2 = ax.bar(x + w / 2, origin.values, width=w, label="原表准备区数值 (合计 1,555)",
                   color=COLOR["muted"])
    annotate_bars(ax, origin.values, bars=bars2, fmt="{:,.0f}", offset=0.02)
    ax.set_xticks(x)
    ax.set_xticklabels(age.index)
    ax.set_ylim(0, max(age.max(), origin.max()) * 1.26)
    ax.set_title("② 女性客户的年龄段分布 (含原表口径对比)")
    ax.set_xlabel("年龄段")
    ax.set_ylabel("客户数")
    ax.legend(loc="upper left", ncol=1)
    style_axes_clean(ax)
    note_box(ax, 0.985, 0.965,
             "原表该图合计 1,555 人, 恰好等于男性总数;\n"
             f"女性真实人数为 {len(fem):,} 人 —— 原表透视基数用错",
             face="#FDECEC", edge=COLOR["negative"], ha="right", fontsize=10)

    fig.suptitle("案例3 · 第一步: 女性占 60.4%, 且女性中 50 岁以上人数最多",
                 y=1.02, x=0.02, ha="left")
    add_source(fig, "数据来源: 案例3 完整案例.xlsx 原始数据 (3,926 行, 2023年1月)  |  "
                    "对应原报表第 1、2 张图", y=0.005)
    save_fig(fig, "ch03_06_性别与年龄段")
    return {"gender": gender, "age": age, "fem": fem}

def case3_category_pay(fem: pd.DataFrame) -> dict:
    """图 ch03_07: 50岁以上女性的品类分布 + 服饰/护肤品人群的支付方式。"""
    seg = fem[fem["客户年龄段（计算）"] == "50岁以上"]
    cat = seg["品类"].value_counts()
    focus = cat.reindex(["Clothing", "Cosmetics"])          # 服饰 + 护肤品
    pay = seg[seg["品类"].isin(["Clothing", "Cosmetics"])]["支付方式"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2), width_ratios=[1.15, 0.95],
                             gridspec_kw={"wspace": 0.30})

    # ① 品类分布 (原 barChart)
    ax = axes[0]
    order = cat.sort_values()
    vals = order.values
    colors = [COLOR["accent"] if i in focus.index else COLOR["muted"] for i in order.index]
    ax.barh([str(i) for i in order.index], vals, height=0.66, color=colors)
    annotate_hbars(ax, vals, fmt="{:,.0f}", pad=0.014,
                   color=COLOR["text"], fontsize=10)
    share = focus.sum() / vals.sum()
    ax.set_xlim(0, vals.max() * 1.28)
    ax.set_title("① 50 岁以上女性客户的消费品类分布")
    ax.set_xlabel("客户数")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    note_box(ax, 0.97, 0.10,
             f"橙色 = 服饰 (Clothing) + 护肤品 (Cosmetics)\n"
             f"两类合计 {focus.sum():,} 人, 占该人群品类记录的 {share:.1%}",
             face="#FDEDD8", edge=COLOR["accent"], ha="right", va="bottom")
    style_axes_clean(ax)

    # ② 支付方式分布 (原 barChart)
    ax = axes[1]
    order = pay.sort_values()
    vals = order.values
    colors = [COLOR["primary"], COLOR["accent"], COLOR["positive"]][:len(vals)]
    bars = ax.bar([str(i) for i in order.index], vals, width=0.52,
                  color=colors)
    # 标签含绝对值与占比两行, 需要一次传两个值, 因此这里不用通用标注函数
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + vals.max() * 0.035,
                f"{v:,.0f}\n({v / vals.sum():.1%})", ha="center", va="bottom",
                fontsize=10.5, color=COLOR["text"], fontweight="bold")
    ax.set_ylim(0, vals.max() * 1.42)
    ax.set_title("② 服饰 + 护肤品人群的支付方式")
    ax.set_xlabel("支付方式")
    ax.set_ylabel("客户数")
    style_axes_clean(ax)

    fig.suptitle("案例3 · 第二步: 50岁以上女性 → 服饰与护肤品 → 主推银行卡支付",
                 y=1.02, x=0.02, ha="left")
    add_source(fig, "数据来源: 案例3 原始数据筛选 [性别=Female, 年龄段=50岁以上]  |  "
                    "对应原报表第 3、4 张图", y=0.005)
    save_fig(fig, "ch03_07_品类与支付方式")
    return {"cat": cat, "focus": focus, "pay": pay}


def run_case3() -> dict:
    """案例3 完整流程。"""
    rule("案例3 · 业务需求")
    print("现提供某国外多品类电商销售平台 2023 年 1 月的销售客户数据。该平台下个月要做活动,")
    print("只想面向男性群体或者女性群体中的一类人群来做, 但缺乏具体的促销方向,")
    print("要求通过数据分析帮助这个平台找到具体的促销策略。")

    rule("案例3 · 原始数据")
    df = load_case3()
    print(f"行数 {len(df):,}, 唯一客户 ID {df['客户ID'].nunique():,} "
          f"(一行 = 一位客户), 月份 {sorted(df['订单月份'].unique())}, "
          f"年份 {sorted(df['订单年份'].unique())}")
    print(f"字段: {list(df.columns)}")
    print(f"品类 {df['品类'].nunique()} 类: {sorted(df['品类'].unique())}")
    print(f"支付方式: {sorted(df['支付方式'].unique())}")

    rule("案例3 · 准备区 ① 性别 + ② 年龄段")
    prof = case3_profile(df)
    print("性别分布:")
    print(prof["gender"].to_string())
    print(f"女性占比 {(prof['gender']['Female']) / len(df):.2%}")
    print("\n女性年龄段分布 (按原始数据重算):")
    print(prof["age"].to_string())
    print(f"合计 {prof['age'].sum():,} 人 (原表该图合计仅 1,555 人, 口径错误)")

    rule("案例3 · 准备区 ③ 品类 + ④ 支付方式")
    r = case3_category_pay(prof["fem"])
    print("50岁以上女性 品类分布:")
    print(r["cat"].to_string())
    print(f"\n聚焦品类 (Clothing + Cosmetics): {r['focus'].sum():,} 人")
    print("\n该人群支付方式分布:")
    print(r["pay"].to_string())
    print(f"主推支付方式: {r['pay'].idxmax()} "
          f"({r['pay'].max():,} 人, 占 {r['pay'].max() / r['pay'].sum():.1%})")

    rule("案例3 · 结论: 促销策略")
    g = prof["gender"]
    a = prof["age"]
    print(f"""
1. 目标人群: 女性
   女性 {g['Female']:,} 人, 占 {g['Female'] / len(df):.2%}; 男性 {g['Male']:,} 人
   -> 二选一时选女性, 可覆盖六成客户

2. 核心年龄段: 50 岁以上
   女性中 50 岁以上 {a['50岁以上']:,} 人, 占女性 {a['50岁以上'] / a.sum():.1%}
   -> 促销资源优先投向 50 岁以上女性

3. 主推品类: 服饰 (Clothing) + 护肤品 (Cosmetics)
   该人群品类记录中, 服饰 {r['cat']['Clothing']:,} 人、护肤品 {r['cat']['Cosmetics']:,} 人,
   合计 {r['focus'].sum():,} 人, 占 {r['focus'].sum() / r['cat'].sum():.1%}
   -> 选品与搭配优惠围绕这两类展开

4. 配套支付策略: 主推银行卡 (Card)
   Card {r['pay']['Card']:,} 人 ({r['pay']['Card'] / r['pay'].sum():.1%}),
   Alipay {r['pay']['Alipay']:,} 人 ({r['pay']['Alipay'] / r['pay'].sum():.1%}),
   WeChat Pay {r['pay']['WeChat Pay']:,} 人 ({r['pay']['WeChat Pay'] / r['pay'].sum():.1%})
   -> 活动优惠优先与银行卡支付绑定 (如银行卡立减), 覆盖近一半目标人群

5. 一句话结论
   面向"50 岁以上女性 + 服饰/护肤品品类 + 银行卡支付"做定向促销,
   在二选一的前提下人群规模最大、品类集中度最高、支付方式最明确。

6. 需要修正的口径问题
   原表"女性消费客户的年龄段分布"图合计仅 1,555 人 (恰好等于男性总数),
   而女性真实人数为 {g['Female']:,} 人; 该错误与本案例第 3 步 (50岁以上女性 918 人)
   自相矛盾。本脚本已按原始数据重算。
""")
    return {"profile": prof, "cat": r["cat"], "pay": r["pay"]}

# ============================================================================ #
# 5. 主入口
# ============================================================================ #

def main() -> int:
    font = setup_style()
    print("=" * 78)
    print("《商业数据分析》第3章 商业数据分析框架 —— Excel 案例的 Python 迁移")
    print("=" * 78)
    print(f"Python    : {sys.version.split()[0]}")
    print(f"解释器    : {sys.executable}")
    print(f"项目目录  : {ROOT}")
    print(f"中文字体  : {font}")
    print(f"图片输出  : {FIG}  (PNG, 300dpi)")
    print("\n教材的分析框架: 业务需求 -> 原始数据 -> 准备区 -> 报表界面")

    for p in (RAW1, FULL1, FULL2, FULL3):
        if not p.exists():
            print(f"\n[错误] 找不到教材文件: {p}")
            return 2

    run_case1()
    run_case2()
    run_case3()

    rule("全章小结")
    print("""
三个案例共用同一套框架 (业务需求 -> 原始数据 -> 准备区 -> 报表界面),
迁移到 Python 后新增的价值集中在"准备区"这一步 —— 口径可校验、可复现:

  案例1  同比公式分母用错  -> 已修正, 并保留两套口径对照
  案例2  总费用与三大期间费用口径不一致 -> 已量化出 6.4 万元未解释缺口
  案例3  年龄段透视基数用错 -> 已按原始数据重算

这三处的共同点: 都不报错, 都能出图, 结论看着也合理 ——
说明"准备区"的口径校验必须成为分析流程的固定动作, 而非可选项。
""")
    print("图片已输出到:", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
