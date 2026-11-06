"""综合数据一页式展板：把 22 张年报压缩成一块看板。

版式分四层，从上到下依次回答「总量—结构—分化—民族地区」四个问题：

1. KPI 条：5 个可独立引用的数字（法人单位、企业占比、三产占比、乡级区划、粮食占有量），
   每张卡带一条迷你面积图，只示方向不示刻度；
2. 主视觉：左侧三次产业法人单位堆叠面积（看结构怎么挪的），
   右侧企业法人单位构成环形（看主体是谁）；
3. 底部四格：民族自治地方与两个剧烈分化的工业品，用小倍数保持同一套读法。

图表类型全部落在既定规范内：面积图 / 堆叠面积 / 环形图 / 真时间轴散点，
不使用柱形、条形、折线。

运行：
    python 统计局代码/综合/panel.py
    python 统计局代码/综合/panel.py --output <路径>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stj_core import StjTable, data_root, parse_table  # noqa: E402

CODE_DIR: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = data_root(CODE_DIR) / CODE_DIR.name      # 年度数据/综合
DEFAULT_OUT: Final[Path] = DATA_DIR / "综合数据展板.png"

# ---------------------------------------------------------------- 视觉常量

_TEXT: Final[str] = "#132A31"
_MUTED: Final[str] = "#5E7075"
_FAINT: Final[str] = "#93A1A4"
_RULE: Final[str] = "#DDE1E6"
_CARD: Final[str] = "#FBFCFC"
_ACCENT: Final[str] = "#2C7A7B"
_DEEP: Final[str] = "#0F4C52"
_WARM: Final[str] = "#C97B3C"
_SAND: Final[str] = "#E4B363"
_TEAL: Final[str] = "#3C887E"

# 页边距与横向栅格
_L: Final[float] = 0.042
_R: Final[float] = 0.958
_SPAN: Final[float] = _R - _L


def _style() -> None:
    """统一中文字体与画布底色。"""
    mpl.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


# ---------------------------------------------------------------- 数据层


def _load(name: str) -> StjTable:
    """读取 年度数据/综合/<name>.csv。"""
    return parse_table(DATA_DIR / f"{name}.csv")


def _change(series: pd.Series, base_year: int) -> float | None:
    """末年相对指定基年的变化率；基数非正时返回 None（百分比无意义）。"""
    years = list(series.index)
    if base_year not in years:
        return None
    first, last = series[base_year], series[years[-1]]
    if pd.isna(first) or pd.isna(last) or first <= 0:
        return None
    return float(last / first - 1)


def _kpi_bundle() -> list[dict[str, object]]:
    """汇总 5 张 KPI 卡需要的数字，避免在绘图代码里散落业务口径。"""
    industry = _load("法人单位数（三次产业）")
    org = _load("法人单位数（按机构类型）")
    admin = _load("行政区划数")
    agri = _load("人均主要农产品占有量")

    # 三次产业三行相加即法人单位总数，与「按机构类型」表的各行合计一致（已交叉核对）
    total = industry.wide.sum(axis=0)
    tertiary = industry.wide.loc["第三产业法人单位数"]
    third_share = tertiary / total * 100

    org_total = org.wide.sum(axis=0)
    enterprise = org.wide.loc["企业法人单位数"]
    ent_share = enterprise / org_total * 100

    towns = admin.wide.loc["乡级区划数"]
    township = admin.wide.loc["镇数"] + admin.wide.loc["街道数"]
    town_share = (township / towns * 100).dropna()

    grain = agri.wide.loc["粮食人均占有量"]

    return [
        {
            "label": "法人单位数",
            "value": f"{total.iloc[-1] / 1e4:,.0f}",
            "unit": "万个",
            "delta": _change(total, 2016),
            "delta_prefix": "较 2016 年",
            "series": total,
            "color": _ACCENT,
            "note": f"{total.index[-1]} 年",
        },
        {
            "label": "企业法人占比",
            "value": f"{ent_share.iloc[-1]:.1f}",
            "unit": "%",
            "delta": float(ent_share.iloc[-1] - ent_share.iloc[0]) / 100,
            "delta_prefix": "较 2016 年",
            "series": ent_share,
            "color": _DEEP,
            "note": f"企业 {enterprise.iloc[-1] / 1e4:,.0f} 万个",
        },
        {
            "label": "第三产业法人占比",
            "value": f"{third_share.iloc[-1]:.1f}",
            "unit": "%",
            "delta": float(third_share.iloc[-1] - third_share.iloc[0]) / 100,
            "delta_prefix": "较 2016 年",
            "series": third_share,
            "color": _TEAL,
            "note": "结构持续向服务化",
        },
        {
            "label": "乡级区划数",
            "value": f"{towns.iloc[-1]:,.0f}",
            "unit": "个",
            "delta": None,
            "delta_prefix": "",
            "series": towns,
            "color": _SAND,
            "note": f"镇与街道合计占 {town_share.iloc[-1]:.1f}%",
        },
        {
            "label": "粮食人均占有量",
            "value": f"{grain.iloc[-1]:,.0f}",
            "unit": "公斤",
            "delta": _change(grain, 2016),
            "delta_prefix": "较 2016 年",
            "series": grain,
            "color": _WARM,
            "note": "首次突破 500 公斤",
        },
    ]


# ---------------------------------------------------------------- 绘图层


def _mini_area(ax: plt.Axes, series: pd.Series, color: str) -> None:
    """卡片内的迷你面积图：只保留形状，不给任何刻度。

    坐标轴一律去掉——迷你图的作用是让「涨还是跌、陡还是缓」一眼可见，
    加上刻度反而会诱导读者去比对不同量纲的数字。
    """
    values = series.dropna()
    if len(values) < 2:
        ax.axis("off")
        return
    x = np.arange(len(values))
    ax.fill_between(x, 0, values.to_numpy(), color=color, alpha=0.18, linewidth=0)
    ax.plot(x, values.to_numpy(), color=color, linewidth=1.4, solid_capstyle="round")
    ax.plot([x[-1]], [values.iloc[-1]], marker="o", markersize=3.4, color=color, zorder=3)
    ax.set_xlim(0, len(values) - 1)
    top = float(values.max())
    bottom = min(0.0, float(values.min()))
    span = top - bottom or 1.0
    ax.set_ylim(bottom - span * 0.18, top + span * 0.22)
    ax.axis("off")


def _kpi_card(fig: plt.Figure, rect: list[float], item: dict[str, object]) -> None:
    """画一张 KPI 卡：左侧文字、右侧迷你面积图。"""
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=_CARD,
                  edgecolor="#ECEEF1", linewidth=0.9, clip_on=False)
    )
    # 顶部一道同色细线：只作视觉锚点，各卡等长，不会被读成柱形
    ax.add_patch(
        Rectangle((0, 0.982), 1, 0.018, transform=ax.transAxes,
                  facecolor=item["color"], edgecolor="none", clip_on=False)
    )

    ax.text(0.058, 0.905, str(item["label"]), fontsize=9.4, color=_MUTED,
            ha="left", va="top")
    value = ax.text(0.052, 0.565, str(item["value"]), fontsize=24, color=_TEXT,
                    fontweight="bold", ha="left", va="baseline")
    # 单位紧贴数字右缘：数字位数不定（501 / 38,723），固定偏移会把「万个」推得忽远忽近，
    # 干脆让单位跟随数字文本的实际包围盒。
    ax.annotate(str(item["unit"]), xycoords=value, xy=(1, 0), xytext=(5, 1),
                textcoords="offset points", fontsize=9.6, color=_MUTED, va="baseline")

    delta = item["delta"]
    if delta is not None:
        arrow = "▲" if float(delta) >= 0 else "▼"
        color = _ACCENT if float(delta) >= 0 else _WARM
        prefix = str(item["delta_prefix"])
        body = f"{abs(float(delta)) * 100:.1f}"
        tail = "个百分点" if "占比" in str(item["label"]) else "%"
        ax.text(0.058, 0.305, f"{prefix} {arrow} {body}{tail}".strip(),
                fontsize=8.6, color=color, ha="left", va="baseline")
    ax.text(0.058, 0.105, str(item["note"]), fontsize=8.2, color=_FAINT,
            ha="left", va="baseline")

    spark = fig.add_axes([rect[0] + rect[2] * 0.575, rect[1] + rect[3] * 0.34,
                          rect[2] * 0.365, rect[3] * 0.40])
    _mini_area(spark, item["series"], str(item["color"]))


def _stack_axes(fig: plt.Figure, rect: list[float], industry: StjTable) -> None:
    """三次产业法人单位数的堆叠面积图（主视觉左）。"""
    ax = fig.add_axes(rect)
    years = industry.years
    rows = [
        ("第一产业法人单位数", "第一产业", _SAND),
        ("第二产业法人单位数", "第二产业", _DEEP),
        ("第三产业法人单位数", "第三产业", _ACCENT),
    ]
    series = [industry.wide.loc[name] for name, _label, _c in rows]
    ax.stackplot(years, *[s.to_numpy() for s in series],
                 colors=[c for _n, _l, c in rows], edgecolor="white", linewidth=1.1)

    total = industry.wide.sum(axis=0)
    ax.set_xlim(min(years) - 0.3, max(years) + 2.9)
    ax.set_ylim(0, float(total.max()) * 1.12)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=8.6, color=_MUTED)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v / 1e4:,.0f}"))
    ax.tick_params(axis="y", labelsize=8.4, colors=_MUTED, length=0)
    ax.tick_params(axis="x", length=3, color="#C7CBCF")
    ax.set_ylabel("法人单位数（万个）", fontsize=9, color=_MUTED, labelpad=6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#C7CBCF")
    ax.spines["bottom"].set_bounds(min(years) - 0.3, max(years))

    last = years[-1]
    cum = 0.0
    for name, label, color in rows:
        value = float(industry.wide.loc[name, last])
        mid = cum + value / 2
        share = value / float(total[last]) * 100
        ax.text(last + 0.35, mid, f"{label}  {share:.1f}%", fontsize=9.4,
                color=color, va="center", ha="left",
                fontweight="bold" if share >= 10 else "normal")
        cum += value

    ax.annotate(
        f"合计 {total[last] / 1e4:,.0f} 万个",
        xy=(last, float(total[last])),
        xytext=(last + 0.35, float(total[last]) * 1.055),
        fontsize=9.2, color=_TEXT, va="center", ha="left", fontweight="bold",
    )

    # 标题与副说明分两行放：pad 留到 34pt 才够，否则 12.4pt 的标题会压住 8.8pt 的副说明
    ax.set_title("法人单位数按三次产业的结构迁移", fontsize=12.4, color=_TEXT,
                 loc="left", pad=34, fontweight="bold")
    ax.text(0, 1.052, "每一层是该产业法人单位数的绝对量，浅色分层即结构变化的来源",
            transform=ax.transAxes, fontsize=8.8, color="#7C8288", ha="left", va="bottom")

    for y_tick in ax.get_yticks():
        if y_tick > 0:
            ax.hlines(y_tick, min(years) - 0.3, last, color="#EDEFF1", linewidth=0.9, zorder=0)


def _donut_axes(fig: plt.Figure, rect: list[float], ent: StjTable) -> None:
    """企业法人单位构成的环形图（主视觉右）。

    环形而非饼：中间的圆孔留给总数与占比，两者是这张图最该被读到的两个数。
    类别合并到 6 项——再细下去每段都薄到量不出厚度，读数只能靠标签堆叠。
    """
    ax = fig.add_axes(rect)
    year = ent.years[-1]
    raw = ent.wide[year].dropna()

    groups = {
        "有限责任公司": raw.get("有限责任公司法人单位数", 0.0),
        "个人独资企业": raw.get("个人独资企业法人单位数", 0.0),
        "合伙企业": raw.get("合伙企业法人单位数", 0.0),
        "港澳台投资": raw.get("港澳台投资企业法人单位数", 0.0),
        "外商投资": raw.get("外商投资企业法人单位数", 0.0),
        "股份/非公司等": raw.get("股份有限公司法人单位数", 0.0)
        + raw.get("非公司企业法人单位数", 0.0)
        + raw.get("其他内资企业法人单位数", 0.0),
    }
    labels = list(groups)
    values = np.array([float(v) for v in groups.values()])
    total_ent = float(raw.get("企业法人单位数", values.sum()))
    colors = [_DEEP, _ACCENT, "#7FB3D5", _SAND, _WARM, "#B8BEC6"]

    wedges, _texts = ax.pie(
        values, colors=colors, startangle=90, counterclock=False,
        wedgeprops={"width": 0.34, "edgecolor": "white", "linewidth": 1.6},
    )
    ax.set_aspect("equal")

    shares = values / total_ent * 100
    # 直接标注占比最大的三项，其余进右侧图例；环上标满会挤成一圈数字
    for wedge, label, share in zip(wedges, labels, shares):
        if share < 3:
            continue
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        radius = 0.83
        ax.text(radius * np.cos(angle), radius * np.sin(angle),
                f"{label}\n{share:.1f}%", fontsize=8.8, color="white",
                ha="center", va="center", fontweight="bold", linespacing=1.5)

    ax.text(0, 0.10, f"{total_ent / 1e4:,.0f}", fontsize=25, color=_TEXT,
            ha="center", va="center", fontweight="bold")
    ax.text(0, -0.10, "万个企业法人", fontsize=9.6, color=_MUTED, ha="center", va="center")
    ax.text(0, -0.30, f"{year} 年", fontsize=8.8, color=_FAINT, ha="center", va="center")

    # 图例放在圆环右下，避免与右上角的标题区抢位置
    legend_labels = [f"{lb} {sh:.1f}%" for lb, sh in zip(labels, shares) if sh < 3]
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=c)
        for c, lb, sh in zip(colors, labels, shares) if sh < 3
    ]
    if handles:
        # 图例压在环下方，位置要卡在底部四小图标题之上（约 0.32），否则两行图例
        # 会盖住「人均发电量…」的说明文字
        ax.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.12),
                  ncol=2, frameon=False, fontsize=8.0, handlelength=0.9,
                  handleheight=0.9, columnspacing=1.2, labelcolor=_MUTED)

    ax.set_title("企业法人单位构成（按登记注册类型）", fontsize=12.4, color=_TEXT,
                 loc="left", pad=26, fontweight="bold", x=-0.14)


def _small_panel(
    fig: plt.Figure,
    rect: list[float],
    years: list[int],
    series: pd.Series,
    title: str,
    color: str,
    unit: str,
    note: str = "",
    dot_mode: bool = False,
    show_pct: bool = True,
) -> None:
    """底部小格：面积图或真时间轴散点，右上角统一标首末变化率。

    Args:
        fig: 画布。
        rect: 轴位置 [x, y, w, h]（figure 坐标）。
        years: 年份序列。
        series: 指标值（index 为年份）。
        title: 小图标题。
        color: 主色。
        unit: 物理单位。
        note: 标题上方的补充说明。
        dot_mode: True 时画真时间轴散点（观测年份稀疏且不等距）。
        show_pct: 是否标首末变化率；存在口径断点时须关闭，否则会把断点读成真实增减。
    """
    ax = fig.add_axes(rect)
    values = series.dropna()
    max_v = float(values.max()) if len(values) else 1.0
    min_v = float(values.min()) if len(values) else 0.0
    top = max_v * 1.22 if max_v > 0 else 1.0
    bottom = min_v * 1.28 if min_v < 0 else 0.0

    if dot_mode:
        xs = np.array([y for y in years if y in values.index], dtype=float)
        ys = values.to_numpy(dtype=float)
        ax.plot(xs, ys, marker="o", markersize=6.4, color=color, linestyle="none",
                markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        # 左右各留 0.75 年的空白：首末点贴边时，居中标签会越出坐标区压到纵轴刻度上
        left, right = min(years) - 0.75, max(years) + 0.75
        for x, y in zip(xs, ys):
            frac = (x - left) / (right - left)
            if frac < 0.22:
                ha, dx = "left", 4
            elif frac > 0.78:
                ha, dx = "right", -4
            else:
                ha, dx = "center", 0
            ax.annotate(f"{y:,.0f}", xy=(x, y), xytext=(dx, 9), textcoords="offset points",
                        ha=ha, va="bottom", fontsize=8.0, color=_TEXT)
        ax.set_xticks(years)
        ax.set_xticklabels([str(y) for y in years], fontsize=8.2, color=_MUTED)
        ax.set_xlim(left, right)
    else:
        ax.fill_between(years, 0, values.to_numpy(dtype=float), color=color, alpha=0.16,
                        linewidth=0)
        ax.plot(years, values.to_numpy(dtype=float), color=color, linewidth=1.6)
        gaps = [y for y in range(min(years), max(years) + 1) if y not in values.index]
        for y in gaps:
            ax.axvspan(y - 0.5, y + 0.5, color="#F1F2F4", zorder=0, linewidth=0)
        step = 2 if len(years) > 6 else 1
        ax.set_xticks(years[::step])
        ax.set_xticklabels([str(y) for y in years[::step]], fontsize=8.2, color=_MUTED)
        ax.set_xlim(min(years) - 0.4, max(years) + 0.4)

    ax.set_ylim(bottom, top)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
    # 不再按量级压成「10万」：这里的单位（亿元、万人）本身已是完整名词，
    # 轴上写 10万 会让 120,399 亿元被读成 12 万亿元，读者要乘一次才敢下判断。
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(axis="y", labelsize=7.8, colors=_MUTED, length=0, pad=2)
    ax.tick_params(axis="x", length=2.5, color="#C7CBCF")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#C7CBCF")
    if bottom < 0:
        ax.axhline(0, color="#9AA0A6", linewidth=0.9, zorder=2)

    ax.set_title(f"{title}（{unit}）", fontsize=10.2, color=_TEXT, loc="left",
                 pad=10, fontweight="bold")
    if note:
        # 说明放在标题之上，两者间距约 15pt；同高会直接把标题盖掉
        ax.text(0, 1.115, note, transform=ax.transAxes, fontsize=8.0, color=_FAINT,
                ha="left", va="bottom")

    if show_pct and len(values) >= 3 and values.iloc[0] > 0:
        pct = float(values.iloc[-1] / values.iloc[0] - 1)
        ax.text(0.995, 1.035, f"{pct:+.1%}", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=9.2, fontweight="bold",
                color=_ACCENT if pct >= 0 else _WARM)


# ---------------------------------------------------------------- 组装


def build_panel(out_path: Path) -> Path:
    """绘制整块展板并写盘。

    Args:
        out_path: 输出 PNG 路径。

    Returns:
        写出的文件路径。

    Raises:
        ValueError: 数据文件缺失或表结构不符合预期。
    """
    _style()
    industry = _load("法人单位数（三次产业）")
    ent = _load("企业法人单位构成")
    agri = _load("人均主要农产品占有量")
    minzu_gdp = _load("民族自治地方地区生产总值")
    minzu_pop = _load("民族自治地方人口")

    fig = plt.figure(figsize=(16, 11.2), dpi=200)

    # —— 标题区
    fig.text(_L, 0.972, "中国综合统计概览", fontsize=27, color=_TEXT,
             fontweight="bold", ha="left", va="top")
    fig.text(_L + 0.001, 0.931,
             "法人单位与产业结构 · 行政区划 · 人均农产品占有量 · 民族自治地方　"
             "|　数据区间 2016–2025 年（各表起止不同，见图内标注）",
             fontsize=10.6, color=_MUTED, ha="left", va="top")
    fig.text(_R, 0.966, "国家统计局 · 年度数据", fontsize=9.4, color=_FAINT,
             ha="right", va="top")
    fig.text(_R, 0.945, "图表由 stj 脚本生成，数值未做插补", fontsize=8.4, color=_FAINT,
             ha="right", va="top")
    fig.add_artist(plt.Line2D([_L, _R], [0.912, 0.912], color=_RULE, linewidth=1.1))

    # —— KPI 条
    cards = _kpi_bundle()
    gap = 0.0125
    card_w = (_SPAN - gap * (len(cards) - 1)) / len(cards)
    for i, item in enumerate(cards):
        _kpi_card(fig, [_L + i * (card_w + gap), 0.762, card_w, 0.135], item)

    # —— 主视觉
    _stack_axes(fig, [_L, 0.385, _SPAN * 0.605, 0.315], industry)
    _donut_axes(fig, [_L + _SPAN * 0.655 + 0.02, 0.398, _SPAN * 0.325, 0.285], ent)

    # —— 底部四格
    small_gap = 0.016
    small_w = (_SPAN - small_gap * 3) / 4
    small_y, small_h = 0.052, 0.255

    gdp = minzu_gdp.wide.loc["民族自治地方国内生产总值"]
    _small_panel(
        fig, [_L + 0 * (small_w + small_gap), small_y, small_w, small_h],
        minzu_gdp.years, gdp, "民族自治地方地区生产总值", _ACCENT, "亿元",
        note="仅 2020、2023、2024 年公布，按真实年份落位",
        dot_mode=True,
    )

    pop = minzu_pop.wide.loc["民族自治地方总人口数"]
    _small_panel(
        fig, [_L + 1 * (small_w + small_gap), small_y, small_w, small_h],
        minzu_pop.years, pop, "民族自治地方总人口", _DEEP, "万人",
        note="2016 年与 2017 年起口径不同，跨年不可直接比较",
        show_pct=False,  # 口径断点处算变化率等于把统计调整报成人口减少
    )

    cloth = agri.wide.loc["人均布产量"]
    _small_panel(
        fig, [_L + 2 * (small_w + small_gap), small_y, small_w, small_h],
        agri.years, cloth, "人均布产量", _WARM, "米",
        note="纺织产能外迁，九年间降幅最深",
    )

    power = agri.wide.loc["人均发电量"]
    _small_panel(
        fig, [_L + 3 * (small_w + small_gap), small_y, small_w, small_h],
        agri.years, power, "人均发电量", _TEAL, "千瓦时",
        note="能源供给能力持续扩张",
    )

    # —— 页脚
    fig.add_artist(plt.Line2D([_L, _R], [0.036, 0.036], color=_RULE, linewidth=1.0))
    fig.text(
        _L, 0.020,
        "数据来源：国家统计局 · 年度数据（综合）。法人单位数 2004 年起不含国际组织，2018、2023 年经经济普查修订，"
        "两次修订处存在口径性回落；企业法人占比的分母为机构类型表各行合计，与三次产业表总数一致。"
        "民族自治地方含 5 个自治区、30 个自治州、120 个自治县（旗）。",
        fontsize=7.6, color=_FAINT, ha="left", va="bottom",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成综合数据一页式展板")
    parser.add_argument("--output", default=str(DEFAULT_OUT), help="输出 PNG 路径")
    args = parser.parse_args()
    path = build_panel(Path(args.output))
    print(f"已生成：{path.resolve()}")


if __name__ == "__main__":
    main()
