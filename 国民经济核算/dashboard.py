"""国民经济核算 · 数据驾驶舱（一页式展板）。

版式参照驾驶舱类看板：顶部标题 + 年份滑轨 + KPI 卡条 + 速读条 + 主副图 + 底部三联。
配色统一为深青 / 薄荷 / 沙色，逐表图与展板共用一套色板。

图表约束（项目约定，优先级高于参考图）：
- 不用柱形、条形、折线及其变体。参考图里的柱状与折线在这里换成
  面积图、100% 堆叠面积图、真时间轴圆点；信息量与读数方式不变。
- 面积图一律零基线；含负值的数列纵轴下探并画零线。
- 不插补、不连断档：只在真实观测年份落点。

用法（任意目录执行均可）：
    python 统计局代码/国民经济核算/dashboard.py
输出：
    年度数据/国民经济核算/国民经济核算数据驾驶舱.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stj_core import StjTable, data_root, parse_table  # noqa: E402

# ── 色板（与 charts.py 同步）
_INK = "#132A31"
_MUT = "#5E7075"
_FAINT = "#93A1A4"
_LINE = "#E6EBEB"
_CARD = "#FBFCFC"
_DEEP = "#0F4C52"
_MAIN = "#2C7A7B"
_MINT = "#BFDCD6"
_MINT2 = "#8FC0B6"
_BLUE = "#5E8CA6"
_SAND = "#E4B363"
_WARN = "#C97B3C"

_FIG_W, _FIG_H = 16.0, 9.6
_L, _R = 0.022, 0.978
_SPAN = _R - _L

YEAR = 2025


def _setup() -> None:
    """统一字体与画布参数。"""
    mpl.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _rounded(
    fig: plt.Figure,
    rect: list[float],
    *,
    face: str = "white",
    edge: str = _LINE,
    radius: float = 0.012,
    lw: float = 1.0,
    shadow: bool = False,
    zorder: float = -2,
) -> None:
    """在图坐标里画一个圆角矩形（卡片底）。

    卡片必须画在坐标区**之下**（zorder 取负）：figure 级 patch 默认 zorder=1，
    比 axes 的 0 高，会把里面的图整块盖住。浮层提示卡反过来要压在最上面，显式传 9。

    Args:
        fig: 画布。
        rect: [x, y, w, h]，figure 坐标。
        face: 填充色。
        edge: 描边色。
        radius: 圆角半径（figure 坐标）。
        lw: 描边宽度。
        shadow: 是否带一层极淡的投影，用来压住浮层卡片。
        zorder: 层级；默认在坐标区之下。
    """
    if shadow:
        fig.patches.append(FancyBboxPatch(
            (rect[0], rect[1] - 0.004), rect[2], rect[3],
            boxstyle=f"round,pad=0,rounding_size={radius}",
            transform=fig.transFigure, facecolor="#0F4C52", alpha=0.06,
            edgecolor="none", zorder=zorder - 0.1,
        ))
    fig.patches.append(FancyBboxPatch(
        (rect[0], rect[1]), rect[2], rect[3],
        boxstyle=f"round,pad=0,rounding_size={radius}",
        transform=fig.transFigure, facecolor=face, edgecolor=edge,
        linewidth=lw, zorder=zorder,
    ))


def _panel(
    fig: plt.Figure,
    rect: list[float],
    title: str,
    sub: str = "",
    tag: str = "",
) -> plt.Axes:
    """画一张图表卡片，返回卡片内部的绘图区。

    Args:
        fig: 画布。
        rect: 卡片 [x, y, w, h]。
        title: 卡片标题。
        sub: 标题下方小字。
        tag: 右上角标签（如「全部增速」「结构」）。

    Returns:
        卡片内的坐标区。
    """
    _rounded(fig, rect, face=_CARD, edge=_LINE)
    fig.text(rect[0] + 0.013, rect[1] + rect[3] - 0.026, title,
             fontsize=11.4, color=_INK, fontweight="bold", va="top")
    if sub:
        fig.text(rect[0] + 0.013, rect[1] + rect[3] - 0.049, sub,
                 fontsize=8.0, color=_FAINT, va="top")
    if tag:
        fig.text(rect[0] + rect[2] - 0.013, rect[1] + rect[3] - 0.026, tag,
                 fontsize=7.8, color=_FAINT, va="top", ha="right")
    top_pad = 0.072 if sub else 0.050
    return fig.add_axes([
        rect[0] + 0.014, rect[1] + 0.028,
        rect[2] - 0.028, rect[3] - top_pad - 0.028,
    ])


def _clean(ax: plt.Axes, *, x_ticks: list[int] | None = None,
           step: int = 2, label_size: float = 7.2) -> None:
    """统一去掉边框与刻度线，只留底部一条浅轴。"""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#D8DEDE")
    ax.tick_params(axis="y", labelsize=label_size, colors=_MUT, length=0, pad=2)
    ax.tick_params(axis="x", labelsize=label_size, colors=_MUT, length=0, pad=3)
    if x_ticks is not None:
        ticks = x_ticks[::step] if len(x_ticks) > 7 else list(x_ticks)
        # 末年必须出现在刻度上，但别硬加一格：隔年取点若已排到 2024，
        # 再补 2025 会让两个标签贴在一起，直接把 2024 顶掉更干净
        if ticks[-1] != x_ticks[-1]:
            if x_ticks[-1] - ticks[-1] < 1.5:
                ticks = ticks[:-1] + [x_ticks[-1]]
            else:
                ticks = ticks + [x_ticks[-1]]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(y) for y in ticks])


def _grid(ax: plt.Axes, top: float, bottom: float = 0.0) -> None:
    """只画水平浅网格，避免竖向线条把面积切成柱状。"""
    for t in ax.get_yticks():
        if bottom < t < top:
            ax.axhline(t, color="#EDF1F1", linewidth=0.8, zorder=0)


def _header(fig: plt.Figure) -> None:
    """顶部标题区：图标 + 主副标题 + 右侧两个胶囊标签。"""
    _rounded(fig, [_L, 0.923, 0.031, 0.052], face=_DEEP, edge="none", radius=0.010)
    # 图标用同心圆环，避免再画柱状符号
    fig.add_artist(Circle((_L + 0.0155, 0.949), 0.0092, transform=fig.transFigure,
                          facecolor="none", edgecolor="white", linewidth=1.6, zorder=3))
    fig.add_artist(Circle((_L + 0.0155, 0.949), 0.0034, transform=fig.transFigure,
                          facecolor="white", edgecolor="none", zorder=3))

    fig.text(_L + 0.040, 0.958, "国民经济核算 · 数据驾驶舱", fontsize=18,
             color=_INK, fontweight="bold", va="center")
    fig.text(_L + 0.040, 0.930, "2016 – 2025 年 · 生产法 / 支出法 / 三次产业 · 国家统计局年度数据",
             fontsize=9.2, color=_MUT, va="center")

    # 右侧标签：当前年份实心，口径为描边
    _rounded(fig, [0.858, 0.928, 0.052, 0.040], face=_DEEP, edge=_DEEP, radius=0.008)
    fig.text(0.884, 0.948, f"{YEAR} 年", fontsize=8.8, color="white",
             ha="center", va="center", fontweight="bold")
    _rounded(fig, [0.916, 0.928, 0.060, 0.040], face="white", edge="#D8DEDE", radius=0.008)
    fig.text(0.946, 0.948, "当年价", fontsize=8.8, color=_MUT, ha="center", va="center")


def _year_rail(fig: plt.Figure, years: list[int], current: int) -> None:
    """年份滑轨：底轨 + 已走过段 + 年份刻度 + 当前点。"""
    y = 0.876
    left, right = _L + 0.006, 0.905
    span = right - left
    fig.add_artist(plt.Line2D([left, right], [y, y], transform=fig.transFigure,
                              color="#E3E8E8", linewidth=3.0, solid_capstyle="round"))
    frac = (current - years[0]) / (years[-1] - years[0])
    x_now = left + span * frac
    fig.add_artist(plt.Line2D([left, x_now], [y, y], transform=fig.transFigure,
                              color=_DEEP, linewidth=3.0, solid_capstyle="round"))
    for yr in years:
        x = left + span * (yr - years[0]) / (years[-1] - years[0])
        passed = yr <= current
        fig.add_artist(plt.Line2D([x, x], [y - 0.008, y + 0.008],
                                  transform=fig.transFigure,
                                  color=_DEEP if passed else "#D8DEDE", linewidth=1.2))
        fig.text(x, y - 0.024, str(yr), fontsize=7.4, ha="center", va="top",
                 color=_MUT if passed else _FAINT,
                 fontweight="bold" if yr == current else "normal")
    fig.add_artist(Circle((x_now, y), 0.0072, transform=fig.transFigure,
                          facecolor=_DEEP, edgecolor="white", linewidth=2.4, zorder=4))
    fig.text(0.918, y, "年度", fontsize=7.8, color=_FAINT, va="center")


def _kpi(fig: plt.Figure, rect: list[float], item: dict[str, object]) -> None:
    """一张 KPI 卡：左上标签、右上口径标签、数值 + 单位、底行同比与较上年。

    底行两端对齐：左边「同比 +x%」随数字宽度变化，右边「较上年 ±x pp」贴卡右缘。
    早先按固定偏移排，长一点的「+5.0pp」会直接压到「较上年」上。
    """
    _rounded(fig, rect, face=_CARD, edge=_LINE, radius=0.010)
    x0, y0, w, h = rect
    fig.text(x0 + 0.012, y0 + h - 0.016, str(item["label"]), fontsize=9.0,
             color=_MUT, va="top")
    fig.text(x0 + w - 0.012, y0 + h - 0.016, str(item["tag"]), fontsize=7.6,
             color=_FAINT, va="top", ha="right")

    fig.text(x0 + 0.012, y0 + 0.030, str(item["value"]), fontsize=23,
             color=_INK, fontweight="bold", va="bottom")
    fig.text(x0 + 0.012 + float(item["value_w"]), y0 + 0.033, str(item["unit"]),
             fontsize=8.8, color=_MUT, va="bottom")

    fig.text(x0 + 0.012, y0 + 0.011, "同比", fontsize=8.0, color=_FAINT, va="bottom")
    fig.text(x0 + 0.038, y0 + 0.011, str(item["delta"]), fontsize=8.6, va="bottom",
             color=_MAIN if float(item["delta_v"]) >= 0 else _WARN, fontweight="bold")
    fig.text(x0 + w - 0.012, y0 + 0.011, str(item["base"]), fontsize=8.0,
             va="bottom", ha="right", color=_FAINT)


def _legend(fig: plt.Figure, rect: list[float], labels: list[str],
            colors: list[str]) -> None:
    """卡片标题行右端的图例：一排「色点 + 名称」，右对齐。"""
    x = rect[0] + rect[2] - 0.012
    y = rect[1] + rect[3] - 0.026
    for name, color in zip(reversed(labels), reversed(colors)):
        fig.text(x, y, name, fontsize=7.4, color=_MUT, ha="right", va="center")
        cx = x - 0.0028 - len(name) * 0.0072
        fig.add_artist(Circle((cx, y), 0.0030, transform=fig.transFigure,
                              facecolor=color, edgecolor="white", linewidth=0.6, zorder=3))
        x = cx - 0.0115


def _callout(ax: plt.Axes, x: float, y: float, text: str) -> None:
    """在图内落一个小的白底提示框，用引线连到数据点。

    比在画布上浮一张大卡片稳：跟着数据走，画到哪都不会压住曲线主体。
    """
    ax.annotate(
        text, xy=(x, y), xytext=(-6, 26), textcoords="offset points",
        ha="right", va="bottom", fontsize=8.0, color=_INK, linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.42", facecolor="white",
                  edgecolor="#DCE4E4", linewidth=1.0),
        arrowprops=dict(arrowstyle="-", color="#C6D2D2", linewidth=0.9,
                        shrinkA=2, shrinkB=6),
        annotation_clip=False, zorder=8,
    )


def _brief(fig: plt.Figure, lead: str, body: str) -> None:
    """速读条：左侧竖条 + 加粗年份 + 正文（自动折两行）。"""
    rect = [_L, 0.663, _SPAN, 0.056]
    _rounded(fig, rect, face="#F4F8F8", edge="#E4ECEC", radius=0.007)
    fig.add_artist(Rectangle((rect[0] + 0.004, rect[1] + 0.010), 0.0035, rect[3] - 0.020,
                             transform=fig.transFigure, facecolor=_DEEP, edgecolor="none"))
    fig.text(rect[0] + 0.014, rect[1] + rect[3] - 0.014, lead, fontsize=8.8,
             color=_DEEP, fontweight="bold", va="top")

    # 按画布宽度手工折行：优先在标点处断开，避免把「52.0% / 15.3%」劈成两半
    per_line = 112
    lines: list[str] = []
    rest = body
    while rest:
        if len(rest) <= per_line:
            lines.append(rest)
            break
        window = rest[:per_line + 12]
        cut = max(window.rfind(ch) for ch in "，；。、")
        cut = cut + 1 if cut > per_line - 40 else per_line
        lines.append(rest[:cut])
        rest = rest[cut:]
    for i, line in enumerate(lines[:2]):
        fig.text(rect[0] + 0.064, rect[1] + rect[3] - 0.014 - i * 0.021, line,
                 fontsize=8.4, color=_INK, va="top")


def _main_gdp(ax: plt.Axes, years: list[int], gdp: pd.Series, growth: pd.Series) -> None:
    """主图：GDP 现价面积（左轴，万亿元）+ 同比增速圆点（右轴，%）。"""
    ys = gdp.reindex(years) / 1e4
    ax.fill_between(years, 0, ys, color=_MINT, linewidth=0, zorder=1)
    ax.plot(years, ys, color=_MAIN, linewidth=1.6, zorder=2)
    # 末年加一圈强调环，替代参考图里的「选中柱」
    ax.plot([years[-1]], [ys.iloc[-1]], marker="o", markersize=9.5, markerfacecolor="none",
            markeredgecolor=_DEEP, markeredgewidth=1.6, zorder=6)

    ax.set_ylim(0, float(ys.max()) * 1.26)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _grid(ax, float(ys.max()) * 1.26)
    _clean(ax, x_ticks=years, step=1, label_size=7.4)
    ax.set_xlim(years[0] - 0.4, years[-1] + 0.4)
    ax.tick_params(axis="x", pad=4)

    # 增速：只落点，不连线（连成折线即违反约束）
    ax2 = ax.twinx()
    g = growth.reindex(years)
    ax2.plot(years, g, marker="o", markersize=5.0, color=_SAND,
             markeredgecolor="white", markeredgewidth=1.2, linestyle="none", zorder=5)
    ax2.set_ylim(min(0.0, float(g.min()) - 1), float(g.max()) + 2.4)
    ax2.yaxis.set_major_locator(MaxNLocator(4))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.tick_params(axis="y", labelsize=7.2, colors=_SAND, length=0, pad=2)
    for side in ("top", "right", "left", "bottom"):
        ax2.spines[side].set_visible(False)


def _growth_band(ax: plt.Axes, years: list[int], growth: pd.Series) -> None:
    """副图：增速围绕 5% 的偏离带，向下为放缓、向上为加快。"""
    ys = growth.reindex(years).to_numpy(dtype=float)
    base = 5.0
    ax.axhline(base, color="#B9C4C4", linewidth=1.0, linestyle=(0, (3, 3)), zorder=3)
    above = ys >= base
    ax.fill_between(years, base, ys, where=above, color=_MINT2, alpha=0.55,
                    linewidth=0, interpolate=True, zorder=1)
    ax.fill_between(years, base, ys, where=~above, color=_WARN, alpha=0.45,
                    linewidth=0, interpolate=True, zorder=1)
    ax.plot(years, ys, marker="o", markersize=3.2, color=_DEEP, linestyle="none", zorder=4)

    lo, hi = float(np.nanmin(ys)), float(np.nanmax(ys))
    ax.set_ylim(min(base, lo) - 1.2, max(base, hi) + 1.4)
    ax.set_xlim(years[0] - 0.4, years[-1] + 0.4)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    _grid(ax, max(base, hi) + 1.4, min(base, lo) - 1.2)
    _clean(ax, x_ticks=years, step=2, label_size=7.2)


def _stack_100(
    ax: plt.Axes,
    years: list[int],
    frame: pd.DataFrame,
    labels: list[str],
    colors: list[str],
) -> None:
    """100% 堆叠面积：三层份额从下往上累加，层间画白线分隔。

    右端只标百分比，层名交给卡片顶部的图例——名字写在右端会顶出卡片边界。

    Args:
        ax: 坐标区。
        years: 年份。
        frame: 行=层名、列=年份的份额矩阵（每列合计 100）。
        labels: 层名（自下而上）。
        colors: 各层填充色。
    """
    base = np.zeros(len(years))
    for name, color in zip(labels, colors):
        vals = frame.loc[name].reindex(years).to_numpy(dtype=float)
        top = base + vals
        ax.fill_between(years, base, top, color=color, linewidth=0, zorder=1)
        ax.plot(years, top, color="white", linewidth=1.0, zorder=2)
        base = top

    ax.set_ylim(0, 100)
    ax.set_xlim(years[0] - 0.3, years[-1] + 1.05)
    ax.set_yticks([0, 50, 100])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    _clean(ax, x_ticks=years, step=2, label_size=7.2)

    cum = np.cumsum([float(frame.loc[n].reindex(years).iloc[-1]) for n in labels])
    bottoms = np.concatenate([[0.0], cum[:-1]])
    for top_v, bottom_v in zip(cum, bottoms):
        mid = (top_v + bottom_v) / 2
        ax.text(years[-1] + 0.35, mid, f"{top_v - bottom_v:.1f}%", fontsize=8.4,
                color=_DEEP if top_v > 40 else _INK, va="center", ha="left",
                fontweight="bold")


def _mini_areas(fig: plt.Figure, rect: list[float], years: list[int],
                frame: pd.DataFrame, labels: list[str], colors: list[str]) -> None:
    """伪小倍数：同一张卡片里竖排三格迷你面积图。

    迷你图画成 sparkline：不画刻度、不画轴线，只留面积 + 末点 + 左上角名称与末年读数。
    三格共用一张卡片，任何一根刻度都会互相压住。
    """
    n = len(labels)
    pad = 0.010
    h = (rect[3] - pad * (n - 1)) / n
    for i, (name, color) in enumerate(zip(labels, colors)):
        ax = fig.add_axes([rect[0], rect[1] + (n - 1 - i) * (h + pad), rect[2], h])
        vals = frame.loc[name].reindex(years).to_numpy(dtype=float)
        ax.fill_between(years, 0, vals, color=color, alpha=0.26, linewidth=0)
        ax.plot(years, vals, color=color, linewidth=1.3, zorder=2)
        ax.plot([years[-1]], [vals[-1]], marker="o", markersize=4.0, color=color,
                markeredgecolor="white", markeredgewidth=1.0, zorder=3)

        top = float(np.nanmax(vals)) * 1.45
        ax.set_ylim(0, top)
        ax.set_xlim(years[0] - 0.3, years[-1] + 0.3)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        if i == n - 1:
            ticks = list(years[::2])
            if ticks[-1] != years[-1]:
                ticks = ticks[:-1] + [years[-1]] if years[-1] - ticks[-1] < 1.5 else ticks + [years[-1]]
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(y) for y in ticks], fontsize=6.8, color=_MUT)
            ax.tick_params(axis="x", length=0, pad=3)

        ax.text(0.004, 0.94, name, transform=ax.transAxes, fontsize=7.6, color=color,
                ha="left", va="top", fontweight="bold")
        ax.text(0.996, 0.94, f"{vals[-1]:.1f}%", transform=ax.transAxes, fontsize=7.8,
                color=color, ha="right", va="top", fontweight="bold")


def build() -> Path:
    """组装驾驶舱并保存 PNG。"""
    _setup()
    root = data_root(Path(__file__).resolve().parent.parent) / "国民经济核算"

    def load(name: str) -> StjTable:
        return parse_table(root / f"{name}.csv")

    gdp_t = load("01-国内生产总值与三次产业增加值")
    share_t = load("04-三次产业构成")
    contrib_t = load("09-三次产业贡献率")
    demand_t = load("12-三大需求贡献率与拉动")
    index_t = load("02-国民经济主要指标指数（上年=100）")

    years = gdp_t.years
    gdp = gdp_t.wide.loc["国内生产总值"]
    growth = index_t.wide.loc["国内生产总值指数"] - 100.0

    share = pd.DataFrame(
        {
            "第一产业": share_t.wide.loc["三次产业构成-第一产业增加值"],
            "第二产业": share_t.wide.loc["三次产业构成-第二产业增加值"],
            "第三产业": share_t.wide.loc["三次产业构成-第三产业增加值"],
        }
    ).T
    contrib = pd.DataFrame(
        {
            "第一产业": contrib_t.wide.loc["第一产业对GDP的贡献率"],
            "第二产业": contrib_t.wide.loc["第二产业对GDP的贡献率"],
            "第三产业": contrib_t.wide.loc["第三产业对GDP的贡献率"],
        }
    ).T
    speeds = pd.DataFrame(
        {
            "第一产业": index_t.wide.loc["第一产业增加值指数"] - 100.0,
            "第二产业": index_t.wide.loc["第二产业增加值指数"] - 100.0,
            "第三产业": index_t.wide.loc["第三产业增加值指数"] - 100.0,
        }
    ).T

    consumption = float(demand_t.wide.loc["最终消费支出对国内生产总值增长贡献率"][YEAR])
    capital = float(demand_t.wide.loc["资本形成总额对国内生产总值增长贡献率"][YEAR])
    netexp = float(demand_t.wide.loc["货物和服务净出口对国内生产总值增长贡献率"][YEAR])
    cons_prev = float(demand_t.wide.loc["最终消费支出对国内生产总值增长贡献率"][YEAR - 1])
    cap_prev = float(demand_t.wide.loc["资本形成总额对国内生产总值增长贡献率"][YEAR - 1])
    net_prev = float(demand_t.wide.loc["货物和服务净出口对国内生产总值增长贡献率"][YEAR - 1])

    gdp_y = float(gdp[YEAR])
    gdp_g = float(growth[YEAR])
    gdp_g_prev = float(growth[YEAR - 1])
    third = float(share.loc["第三产业"][YEAR])
    third_prev = float(share.loc["第三产业"][YEAR - 1])
    pc = float(gdp_t.wide.loc["人均国内生产总值"][YEAR])

    fig = plt.figure(figsize=(_FIG_W, _FIG_H), dpi=200)
    fig.patch.set_facecolor("white")

    _header(fig)
    _year_rail(fig, years, YEAR)

    # ── KPI 条
    cards = [
        {"label": "GDP 总量", "tag": "现价", "value": f"{gdp_y / 1e4:,.2f}", "unit": "万亿元",
         "value_w": 0.072, "delta": f"{gdp_g:+.1f}%", "delta_v": gdp_g, "delta_w": 0.070,
         "base": f"较上年 {gdp_g - gdp_g_prev:+.1f}pp"},
        {"label": "第三产业占比", "tag": "结构", "value": f"{third:.1f}", "unit": "%",
         "value_w": 0.040, "delta": f"{third - third_prev:+.1f}pp", "delta_v": third - third_prev,
         "delta_w": 0.048, "base": f"贡献率 {float(contrib.loc['第三产业'][YEAR]):.1f}%"},
        {"label": "最终消费贡献率", "tag": "需求", "value": f"{consumption:.1f}", "unit": "%",
         "value_w": 0.040, "delta": f"{consumption - cons_prev:+.1f}pp",
         "delta_v": consumption - cons_prev, "delta_w": 0.050,
         "base": f"拉动 {float(demand_t.wide.loc['最终消费支出对国内生产总值增长拉动'][YEAR]):.1f}pp"},
        {"label": "资本形成贡献率", "tag": "需求", "value": f"{capital:.1f}", "unit": "%",
         "value_w": 0.040, "delta": f"{capital - cap_prev:+.1f}pp",
         "delta_v": capital - cap_prev, "delta_w": 0.050,
         "base": f"拉动 {float(demand_t.wide.loc['资本形成总额对国内生产总值增长拉动'][YEAR]):.1f}pp"},
        {"label": "净出口贡献率", "tag": "需求", "value": f"{netexp:.1f}", "unit": "%",
         "value_w": 0.040, "delta": f"{netexp - net_prev:+.1f}pp",
         "delta_v": netexp - net_prev, "delta_w": 0.050,
         "base": f"拉动 {float(demand_t.wide.loc['货物和服务净出口对国内生产总值增长拉动'][YEAR]):.1f}pp"},
    ]
    gap = 0.011
    card_w = (_SPAN - gap * (len(cards) - 1)) / len(cards)
    for i, item in enumerate(cards):
        _kpi(fig, [_L + i * (card_w + gap), 0.742, card_w, 0.102], item)

    _brief(
        fig,
        f"{YEAR} 速读",
        f"GDP {gdp_y / 1e4:,.2f} 万亿元，同比 +{gdp_g:.1f}%；人均 {pc:,.0f} 元。"
        f"产业结构继续向服务迁移，第三产业占比 {third:.1f}%（较上年 {third - third_prev:+.1f}pp）、"
        f"贡献率 {float(contrib.loc['第三产业'][YEAR]):.1f}%。需求侧最终消费 / 资本形成 / 净出口贡献率"
        f" {consumption:.1f}% / {capital:.1f}% / {netexp:.1f}%，资本形成贡献率较上年 {capital - cap_prev:+.1f}pp，"
        f"净出口升至 {netexp:.1f}%（{netexp - net_prev:+.1f}pp），外需对增长的支撑明显强于上年。",
    )

    # ── 主图：GDP 现价 + 增速
    main_rect = [_L, 0.335, _SPAN * 0.612, 0.300]
    ax_main = _panel(fig, main_rect,
                     "GDP 现价与同比增速", "面积＝GDP（万亿元）　圆点＝同比增速（右轴，%）",
                     tag="当年价 · 年度")
    _main_gdp(ax_main, years, gdp, growth)
    _callout(ax_main, years[-1], gdp[YEAR] / 1e4,
             f"{YEAR} 年  {gdp_y / 1e4:,.2f} 万亿元\n同比 +{gdp_g:.1f}%　人均 {pc:,.0f} 元")

    # ── 副图：增速围绕 5% 的偏离
    ax_g = _panel(fig, [_L + _SPAN * 0.624, 0.335, _SPAN * 0.376, 0.300],
                  "增长动能：增速与 5% 的偏离", "高于 5% 为加快（青），低于为放缓（橙）",
                  tag="全部增速")
    _growth_band(ax_g, years, growth)

    # ── 底部三联
    y_b, h_b = 0.048, 0.252
    w_b = (_SPAN - gap * 2) / 3

    rect_s = [_L, y_b, w_b, h_b]
    ax_s = _panel(fig, rect_s, "三次产业结构", "占 GDP 比重，自下而上累加至 100%")
    _legend(fig, rect_s, ["第一产业", "第二产业", "第三产业"], [_SAND, _MINT2, _DEEP])
    _stack_100(ax_s, years, share, ["第一产业", "第二产业", "第三产业"],
               [_SAND, _MINT2, _DEEP])

    rect_v = [_L + w_b + gap, y_b, w_b, h_b]
    # 这张卡片的绘图区由三个迷你坐标区拼成，外层坐标区必须关掉，
    # 否则会露出一套 0.0–1.0 的默认刻度压在迷你图上
    ax_v = _panel(fig, rect_v, "三次产业增速", "各产业增加值指数减 100，三格独立纵轴",
                  tag="比上年")
    ax_v.set_axis_off()
    _mini_areas(fig, [rect_v[0] + 0.014, y_b + 0.020, w_b - 0.030, h_b - 0.100],
                years, speeds, ["第一产业", "第二产业", "第三产业"], [_SAND, _MINT2, _DEEP])

    rect_c = [_L + 2 * (w_b + gap), y_b, w_b, h_b]
    ax_c = _panel(fig, rect_c, "三次产业贡献率", "对 GDP 增长的贡献，三项合计 100%")
    _legend(fig, rect_c, ["第一产业", "第二产业", "第三产业"], [_SAND, _MINT2, _DEEP])
    _stack_100(ax_c, years, contrib, ["第一产业", "第二产业", "第三产业"],
               [_SAND, _MINT2, _DEEP])

    fig.text(
        _L, 0.016,
        "数据来源：国家统计局 · 年度数据（国民经济核算）。GDP 与三次产业增加值为当年价；"
        "增速按不变价指数（上年=100）折算；三大需求贡献率合计为 100%，个别年份单项可为负。"
        "全部数值未做插补，图中面积均为零基线。",
        fontsize=6.8, color=_FAINT, va="bottom",
    )

    out = root / "国民经济核算数据驾驶舱.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("已生成：" + str(build()))
