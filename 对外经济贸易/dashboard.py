"""对外经济贸易 · 数据驾驶舱（一页式展板）。

版式参照驾驶舱类看板：顶部标题 + 年份滑轨 + KPI 卡条 + 速读条 + 主副图 + 底部三联。
配色统一为深青 / 薄荷 / 沙色，逐表图与展板共用一套色板。

图表约束（项目约定，优先级高于参考图）：
- 不用柱形、条形、折线及其变体。参考图里的柱状与折线在这里换成
  面积图、100% 堆叠面积图、真时间轴圆点；信息量与读数方式不变。
- 面积图一律零基线；含负值的数列纵轴下探并画零线。
- 不插补、不连断档：只在真实观测年份落点。

用法（任意目录执行均可）：
    python 统计局代码/对外经济贸易/dashboard.py
输出：
    年度数据/对外经济贸易/对外经济贸易数据驾驶舱.png
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

from charts import short_names  # noqa: E402
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

    fig.text(_L + 0.040, 0.958, "对外经济贸易 · 数据驾驶舱", fontsize=18,
             color=_INK, fontweight="bold", va="center")
    fig.text(_L + 0.040, 0.930, "2016 – 2025 年 · 进出口 / 贸易差额 / 区域结构 / 利用外资 / 对外投资 · 国家统计局年度数据",
             fontsize=9.2, color=_MUT, va="center")

    # 右侧标签：当前年份实心，口径为描边
    _rounded(fig, [0.858, 0.928, 0.052, 0.040], face=_DEEP, edge=_DEEP, radius=0.008)
    fig.text(0.884, 0.948, f"{YEAR} 年", fontsize=8.8, color="white",
             ha="center", va="center", fontweight="bold")
    _rounded(fig, [0.916, 0.928, 0.060, 0.040], face="white", edge="#D8DEDE", radius=0.008)
    fig.text(0.946, 0.948, "年末", fontsize=8.8, color=_MUT, ha="center", va="center")


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



def _main_trade(ax: plt.Axes, years: list[int], exports: pd.Series,
                imports: pd.Series, balance: pd.Series) -> None:
    """主图：出口 / 进口堆叠面积（万亿元）+ 贸易差额圆点（右轴）。

    出口与进口相加即进出口总额，堆叠的顶边就是总额曲线；差额单独挂右轴用圆点，
    三项读数在一张图里互不遮挡。面积一律零基线。
    """
    ex = exports.reindex(years) / 1e4
    im = imports.reindex(years) / 1e4
    top = ex + im
    ax.fill_between(years, 0, ex, color=_MINT, linewidth=0, zorder=1)
    ax.fill_between(years, ex, top, color=_MINT2, alpha=0.70, linewidth=0, zorder=1)
    ax.plot(years, top, color=_DEEP, linewidth=1.6, zorder=3)
    ax.plot(years, ex, color=_MAIN, linewidth=1.0, zorder=3)
    ax.set_ylim(0, float(top.max()) * 1.24)
    ax.set_xlim(years[0] - 0.4, years[-1] + 0.4)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _grid(ax, float(top.max()) * 1.24)
    _clean(ax, x_ticks=years, step=2, label_size=7.2)
    ax.tick_params(axis="x", pad=4)

    ax2 = ax.twinx()
    bal = balance.reindex(years) / 1e4
    ax2.plot(years, bal, marker="o", markersize=4.8, color=_SAND,
             markeredgecolor="white", markeredgewidth=1.1, linestyle="none", zorder=5)
    ax2.set_ylim(0, float(np.nanmax(bal)) * 1.30)
    ax2.yaxis.set_major_locator(MaxNLocator(4))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.1f}"))
    ax2.tick_params(axis="y", labelsize=7.0, colors=_SAND, length=0, pad=2)
    for side in ("top", "right", "left", "bottom"):
        ax2.spines[side].set_visible(False)


def _partner_dots(ax: plt.Axes, series: pd.Series) -> None:
    """卡片内的伙伴排名：一行一个经济体，点落在量级上，数值贴点右侧。"""
    names = [_short_partner(n) for n in series.index][::-1]
    values = (series.to_numpy(dtype=float) / 1e4)[::-1]
    ys = np.arange(len(names), dtype=float)
    top = float(values.max())
    ax.set_xlim(0, top * 1.30)
    ax.set_ylim(-0.7, len(names) - 0.3)
    for name, value, y in zip(names, values, ys):
        ax.plot([0, value], [y, y], color="#E3E9E9", linewidth=2.2,
                solid_capstyle="round", zorder=1)
        ax.plot([value], [y], marker="o", markersize=5.6, color=_MAIN,
                markeredgecolor="white", markeredgewidth=0.9, zorder=2)
        ax.text(value + top * 0.03, y, f"{value:,.0f}", fontsize=7.0,
                color=_INK, va="center", ha="left")
        ax.text(-0.025, y, name, transform=ax.get_yaxis_transform(), fontsize=7.0,
                color=_MUT, va="center", ha="right", clip_on=False)
    ax.set_xticks([])
    ax.set_yticks([])
    # 坐标区缩进后，axes 的白色底会在卡片上留一条色差带；设成透明与卡片同色
    ax.patch.set_alpha(0.0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def _short_partner(name: str) -> str:
    """把「中国同韩国进出口总额」压成「韩国」。"""
    for cut in ("中国同", "进出口总额"):
        name = name.replace(cut, "")
    return name


def _rank_dots(fig: plt.Figure, ax: plt.Axes, series: pd.Series, unit: str,
               fmt: str = "{:,.0f}") -> None:
    """卡片内的一列排名：一行一个类别，点落在量级上，数值贴点右侧。

    十来个行业挤进一张卡片，用点图排完序比小倍数矩阵好读：谁高谁低一眼可见，
    也不必为每格单独配一根纵轴。
    """
    names = list(series.index)[::-1]
    values = series.to_numpy(dtype=float)[::-1]
    ys = np.arange(len(names), dtype=float)
    top = float(values.max())
    ax.set_xlim(0, top * 1.34)
    ax.set_ylim(-0.7, len(names) - 0.3)

    for name, value, y in zip(names, values, ys):
        ax.plot([0, value], [y, y], color="#E3E9E9", linewidth=2.2,
                solid_capstyle="round", zorder=1)
        ax.plot([value], [y], marker="o", markersize=5.6, color=_MAIN,
                markeredgecolor="white", markeredgewidth=0.9, zorder=2)
        ax.text(value + top * 0.035, y, fmt.format(value) + unit, fontsize=7.2,
                color=_INK, va="center", ha="left")
        # 名称写在坐标区左侧的留白里（x 用轴比例、y 用数据值），
        # 不能用负的数据坐标：xlim 从 0 起，负值会被裁掉或甩进隔壁卡片
        ax.text(-0.025, y, name, transform=ax.get_yaxis_transform(), fontsize=7.6,
                color=_MUT, va="center", ha="right", clip_on=False)

    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


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
    # 份额小的层（大洋洲 4.4%、非洲 4.8%）中点只差 4 个百分点，8.4pt 的字会叠在一起；
    # 把相邻标注撑到至少 6.8 个百分点，顶部顶到 99% 时整体下移
    mids = [(t + b) / 2 for t, b in zip(cum, bottoms)]
    for i in range(1, len(mids)):
        mids[i] = max(mids[i], mids[i - 1] + 6.8)
    over = mids[-1] - 99.0
    if over > 0:
        mids = [m - over for m in mids]
        for i in range(len(mids) - 2, -1, -1):
            mids[i] = min(mids[i], mids[i + 1] - 6.8)
    for (top_v, bottom_v), mid in zip(zip(cum, bottoms), mids):
        # 标注落在末端年份之后的空白里，不在色带上，所以一律用深色；
        # 曾按底色亮度改成白字，结果白字写在了白底上，整行百分比消失
        ax.text(years[-1] + 0.35, mid, f"{top_v - bottom_v:.1f}%", fontsize=8.4,
                color=_DEEP, va="center", ha="left", fontweight="bold")


def _mini_areas(fig: plt.Figure, rect: list[float], years: list[int],
                frame: pd.DataFrame, labels: list[str], colors: list[str],
                fmt: str = "{:.1f}%") -> None:
    """伪小倍数：同一张卡片里竖排三格迷你面积图。

    迷你图画成 sparkline：不画刻度、不画轴线，只留面积 + 末点 + 左上角名称与末年读数。
    三格共用一张卡片，任何一根刻度都会互相压住。
    """
    n = len(labels)
    # 每个标签单独占一条横带，写在面板之上：留在面板里时，曲线的顶边会紧贴文字基线，
    # 读起来像给名称划了一道删除线
    label_h, pad = 0.0118, 0.006
    h = (rect[3] - label_h * n - pad * (n - 1)) / n
    for i, (name, color) in enumerate(zip(labels, colors)):
        bottom = rect[1] + (n - 1 - i) * (h + pad + label_h) + label_h
        ax = fig.add_axes([rect[0], bottom, rect[2], h])
        vals = frame.loc[name].reindex(years).to_numpy(dtype=float)
        ax.fill_between(years, 0, vals, color=color, alpha=0.26, linewidth=0)
        ax.plot(years, vals, color=color, linewidth=1.3, zorder=2)
        ax.plot([years[-1]], [vals[-1]], marker="o", markersize=4.0, color=color,
                markeredgecolor="white", markeredgewidth=1.0, zorder=3)

        # 1.45 倍余量在迷你格里只隔 6pt，读数会压到末端折线上；抬到 1.85 倍才分得开
        top = float(np.nanmax(vals)) * 2.30
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

        # 名称与读数写在面板上方的标签带里（figure 坐标），不再压进绘图区
        fig.text(rect[0] + 0.003, bottom + h + 0.0015, name, fontsize=7.4, color=color,
                 ha="left", va="bottom", fontweight="bold")
        fig.text(rect[0] + 0.30 * rect[2], bottom + h + 0.0015, fmt.format(vals[-1]),
                 fontsize=7.4, color=color, ha="left", va="bottom", fontweight="bold")
        ax.patch.set_alpha(0.0)


def build() -> Path:
    """组装对外经济贸易驾驶舱并保存 PNG。"""
    _setup()
    root = data_root(Path(__file__).resolve().parent.parent) / "对外经济贸易"

    def load(name: str) -> StjTable:
        return parse_table(root / f"{name}.csv")

    trade_t = load("01-进出口总额（人民币）")
    fdi_t = load("33-实际利用外资额（外商直接投资）")
    odi_t = load("56-中国对世界直接投资流量")
    project_t = load("61-对世界承包工程完成营业额")

    region_files = [
        ("亚洲", "06-同亚洲进出口总额"),
        ("非洲", "07-同非洲进出口总额"),
        ("欧洲", "08-同欧洲进出口总额"),
        ("拉丁美洲", "09-同拉丁美洲进出口总额"),
        ("北美洲", "10-同北美洲进出口总额"),
        ("大洋洲及太平洋群岛", "11-同大洋洲及太平洋群岛进出口总额"),
    ]
    regions: dict[str, pd.Series] = {}
    asia_t = None
    for label, name in region_files:
        table = load(name)
        if asia_t is None:
            asia_t = table
        total_key = f"中国同{label}进出口总额"
        if total_key in table.wide.index:
            regions[label] = table.wide.loc[total_key]

    years = trade_t.years
    exports = trade_t.wide.loc["出口总额 (人民币)"]
    imports = trade_t.wide.loc["进口总额 (人民币)"]
    balance = trade_t.wide.loc["进出口差额 (人民币)"]
    total = trade_t.wide.loc["进出口总额 (人民币)"]

    ex_y, ex_prev = float(exports[YEAR]), float(exports[YEAR - 1])
    im_y, im_prev = float(imports[YEAR]), float(imports[YEAR - 1])
    bal_y, bal_prev = float(balance[YEAR]), float(balance[YEAR - 1])
    tot_y, tot_prev = float(total[YEAR]), float(total[YEAR - 1])

    fdi = fdi_t.wide.loc["实际使用外商直接投资金额"].dropna()
    fdi_year = int(fdi.index[-1])
    fdi_y, fdi_prev = float(fdi.iloc[-1]), float(fdi.iloc[-2])

    # 伙伴排名用「最近一个有完整明细的年份」：2025 只更新了总量，国别还是空的
    partner_year = int(asia_t.wide[2024].notna().sum() and 2024)
    partner_series = (
        asia_t.wide[partner_year].dropna()
        .drop(labels=[k for k in asia_t.wide.index if k.endswith("亚洲进出口总额")], errors="ignore")
        .sort_values(ascending=False).head(8)
    )

    region_frame = pd.DataFrame(regions)
    region_frame = region_frame.dropna(how="all")
    region_share = (region_frame.div(region_frame.sum(axis=1), axis=0) * 100).T
    region_share.columns = [int(c) for c in region_share.columns]
    # 只保留六洲齐全的年份：2024 年大洋洲的合计行没有发布（各分项有数），
    # 若照常按已发布五洲归一，这一年会被悄悄重标成 100%，与其余年份不可比；
    # 用分项求和补又比合计行少 0.04%（若干小岛国为空白），属于插补，不做。
    dropped_years = [int(c) for c in region_share.columns if region_share[c].isna().any()]
    region_share = region_share.loc[:, region_share.notna().all()]
    region_note = (
        f"　{ '、'.join(map(str, dropped_years)) } 年大洋洲合计未发布，该年不绘入"
        if dropped_years else ""
    )

    mini = pd.DataFrame({
        "实际利用外资": fdi / 1e4,
        "对外直接投资": odi_t.wide.loc["中国对世界直接投资流量"] / 1e4,
        "承包工程营业额": project_t.wide.loc["中国对世界承包工程完成营业额"] / 1e4,
    }).T
    mini_years = list(odi_t.years)

    region_last = int(region_share.columns[-1])
    asia_share = float(region_share.loc["亚洲", region_last])

    fig = plt.figure(figsize=(_FIG_W, _FIG_H), dpi=200)
    fig.patch.set_facecolor("white")

    _header(fig)
    _year_rail(fig, years, YEAR)

    cards = [
        {"label": "进出口总额", "tag": "总量", "value": f"{tot_y / 1e4:,.2f}", "unit": "万亿元",
         "value_w": 0.062, "delta": f"{(tot_y / tot_prev - 1) * 100:+.1f}%",
         "delta_v": tot_y - tot_prev, "base": f"较上年 {tot_y - tot_prev:+,.0f} 亿元"},
        {"label": "出口", "tag": "出口", "value": f"{ex_y / 1e4:,.2f}", "unit": "万亿元",
         "value_w": 0.058, "delta": f"{(ex_y / ex_prev - 1) * 100:+.1f}%",
         "delta_v": ex_y - ex_prev, "base": f"占总额 {ex_y / tot_y:.1%}"},
        {"label": "进口", "tag": "进口", "value": f"{im_y / 1e4:,.2f}", "unit": "万亿元",
         "value_w": 0.058, "delta": f"{(im_y / im_prev - 1) * 100:+.1f}%",
         "delta_v": im_y - im_prev, "base": f"占总额 {im_y / tot_y:.1%}"},
        {"label": "贸易差额", "tag": "差额", "value": f"{bal_y / 1e4:,.2f}", "unit": "万亿元",
         "value_w": 0.058, "delta": f"{(bal_y / bal_prev - 1) * 100:+.1f}%",
         "delta_v": bal_y - bal_prev, "base": f"相当于进口 {bal_y / im_y:.1%}"},
        {"label": "实际使用外资", "tag": f"外资 {fdi_year}", "value": f"{fdi_y / 1e4:,.0f}", "unit": "亿美元",
         "value_w": 0.052, "delta": f"{(fdi_y / fdi_prev - 1) * 100:+.1f}%",
         "delta_v": fdi_y - fdi_prev, "base": f"对外投资 {float(odi_t.wide.loc['中国对世界直接投资流量'].dropna().iloc[-1]) / 1e4:,.0f} 亿美元"},
    ]
    gap = 0.011
    card_w = (_SPAN - gap * (len(cards) - 1)) / len(cards)
    for i, item in enumerate(cards):
        _kpi(fig, [_L + i * (card_w + gap), 0.742, card_w, 0.102], item)

    _brief(
        fig,
        f"{YEAR} 速读",
        f"货物进出口 {tot_y / 1e4:,.2f} 万亿元，比上年 {(tot_y / tot_prev - 1) * 100:+.1f}%；"
        f"出口 {ex_y / 1e4:,.2f} 万亿元（{(ex_y / ex_prev - 1) * 100:+.1f}%）、"
        f"进口 {im_y / 1e4:,.2f} 万亿元（{(im_y / im_prev - 1) * 100:+.1f}%），"
        f"顺差 {bal_y / 1e4:,.2f} 万亿元。{region_last} 年亚洲占进出口 {asia_share:.1f}%。"
        f"{fdi_year} 年实际使用外商直接投资 {fdi_y / 1e4:,.0f} 亿美元（{(fdi_y / fdi_prev - 1) * 100:+.1f}%），"
        f"对外直接投资 {float(odi_t.wide.loc['中国对世界直接投资流量'].dropna().iloc[-1]) / 1e4:,.0f} 亿美元。",
    )

    rect_m = [_L, 0.335, _SPAN * 0.600, 0.300]
    ax_main = _panel(fig, rect_m, "进出口总额与贸易差额",
                     "面积＝出口 / 进口（万亿元）　圆点＝差额（右轴，万亿元）")
    _legend(fig, rect_m, ["出口", "进口"], [_MINT, _MINT2])
    _main_trade(ax_main, years, exports, imports, balance)

    rect_r = [_L + _SPAN * 0.612, 0.335, _SPAN * 0.388, 0.300]
    ax_r = _panel(fig, rect_r, "进出口的区域结构",
                  "占当年进出口总额，自下而上累加至 100%" + region_note)
    _legend(fig, rect_r, ["亚洲", "欧洲", "北美洲", "拉丁美洲", "非洲", "大洋洲"],
            [_DEEP, _MAIN, _MINT, _MINT2, _SAND, "#B9C4C4"])
    _stack_100(ax_r, list(region_share.columns), region_share,
               ["亚洲", "欧洲", "北美洲", "拉丁美洲", "非洲", "大洋洲及太平洋群岛"],
               [_DEEP, _MAIN, _MINT, _MINT2, _SAND, "#B9C4C4"])

    y_b, h_b = 0.048, 0.252
    w_b = (_SPAN - gap * 2) / 3

    rect_o = [_L, y_b, w_b, h_b]
    ax_o = _panel(fig, rect_o, "对外经济合作", "亿美元　三格独立纵轴")
    ax_o.set_axis_off()
    # 底部留 0.045 给末格的年份刻度，否则 2016…2024 会被卡片下沿切掉
    _mini_areas(fig, [rect_o[0] + 0.014, y_b + 0.045, w_b - 0.030, h_b - 0.135],
                mini_years, mini, list(mini.index), [_DEEP, _MINT2, _SAND],
                fmt="{:,.0f} 亿")

    rect_p = [_L + w_b + gap, y_b, w_b, h_b]
    ax_p = _panel(fig, rect_p, f"{partner_year} 年亚洲前 8 个贸易伙伴", "亿美元")
    ax_p.set_position([rect_p[0] + 0.052, rect_p[1] + 0.034,
                       rect_p[2] - 0.062, rect_p[3] - 0.122])
    _partner_dots(ax_p, partner_series)

    rect_s = [_L + 2 * (w_b + gap), y_b, w_b, h_b]
    ax_s = _panel(fig, rect_s, "出口与进口的比重", "占当年进出口总额")
    _legend(fig, rect_s, ["出口", "进口"], [_DEEP, _MINT2])
    ex_share = pd.DataFrame({"出口": exports / total * 100, "进口": imports / total * 100}).T
    _stack_100(ax_s, years, ex_share, ["出口", "进口"], [_DEEP, _MINT2])

    fig.text(
        _L, 0.016,
        "数据来源：国家统计局 · 年度数据（对外经济贸易）。进出口为海关统计，2025 年国别明细尚未发布，"
        "伙伴排名取 2024 年；2024 年大洋洲进出口合计未发布，区域结构图只画到六洲齐全的年份。"
        "实际使用外资与对外直接投资为国家外汇管理局口径，单位亿美元。"
        "区域结构按六大洲合计口径测算，与总额的差额为「国别（地区）不详」部分。全部数值未做插补。",
        fontsize=6.8, color=_FAINT, va="bottom",
    )

    out = root / "对外经济贸易数据驾驶舱.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("已生成：" + str(build()))
