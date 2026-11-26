"""固定资产投资和房地产 · 数据驾驶舱（一页式展板）。

版式参照驾驶舱类看板：顶部标题 + 年份滑轨 + KPI 卡条 + 速读条 + 主副图 + 底部三联。
配色统一为深青 / 薄荷 / 沙色，逐表图与展板共用一套色板。

图表约束（项目约定，优先级高于参考图）：
- 不用柱形、条形、折线及其变体。参考图里的柱状与折线在这里换成
  面积图、100% 堆叠面积图、真时间轴圆点；信息量与读数方式不变。
- 面积图一律零基线；含负值的数列纵轴下探并画零线。
- 不插补、不连断档：只在真实观测年份落点。

用法（任意目录执行均可）：
    python 统计局代码/固定资产投资和房地产/dashboard.py
输出：
    年度数据/固定资产投资和房地产/固定资产投资和房地产数据驾驶舱.png
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

    fig.text(_L + 0.040, 0.958, "固定资产投资和房地产 · 数据驾驶舱", fontsize=18,
             color=_INK, fontweight="bold", va="center")
    fig.text(_L + 0.040, 0.930, "2016 – 2025 年 · 投资总量 / 三次产业 / 到位资金 / 房地产开发 / 销售 · 国家统计局年度数据",
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



def _main_property(ax: plt.Axes, years: list[int], frame: pd.DataFrame,
                   labels: list[str], colors: list[str]) -> None:
    """主图：房地产开发投资按用途堆叠（亿元），顶边即开发投资总额。

    四类用途相加就是房地产开发投资额，堆叠后顶边直接是总额曲线：
    「开发投资从 14.8 万亿掉到 8.3 万亿」和「住宅占大头」两件事在同一张图里读出。
    """
    base = np.zeros(len(years))
    for name, color in zip(labels, colors):
        vals = frame.loc[name].reindex(years).to_numpy(dtype=float)
        top = base + vals
        ax.fill_between(years, base, top, color=color, linewidth=0, zorder=1)
        ax.plot(years, top, color="white", linewidth=1.0, zorder=2)
        base = top

    total = base
    ax.plot(years, total, color=_DEEP, linewidth=1.6, zorder=3)
    ax.set_ylim(0, float(total.max()) * 1.18)
    ax.set_xlim(years[0] - 0.4, years[-1] + 0.4)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.1f}"))
    _grid(ax, float(total.max()) * 1.18)
    _clean(ax, x_ticks=years, step=2, label_size=7.2)
    ax.tick_params(axis="x", pad=4)


def _sales_dots(ax: plt.Axes, years: list[int], area: pd.Series, price: pd.Series) -> None:
    """副图：销售面积（面积图、零基线）+ 平均销售价格（右轴圆点，不连线）。

    销售面积与均价量纲差三个数量级，共用一根纵轴会把均价压成贴地的一条线；
    面积留在主轴，均价挂右轴用圆点，两个读数各自可读。
    """
    a = area.reindex(years) / 1e4
    ax.fill_between(years, 0, a, color=_MINT, linewidth=0, zorder=1)
    ax.plot(years, a, color=_MAIN, linewidth=1.3, zorder=2)
    ax.set_ylim(0, float(np.nanmax(a)) * 1.22)
    ax.set_xlim(years[0] - 0.4, years[-1] + 0.4)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.1f}"))
    _grid(ax, float(np.nanmax(a)) * 1.22)
    _clean(ax, x_ticks=years, step=2, label_size=7.2)

    ax2 = ax.twinx()
    p = price.reindex(years)
    ax2.plot(years, p, marker="o", markersize=4.6, color=_SAND,
             markeredgecolor="white", markeredgewidth=1.1, linestyle="none", zorder=5)
    ax2.set_ylim(float(np.nanmin(p)) * 0.92, float(np.nanmax(p)) * 1.05)
    ax2.yaxis.set_major_locator(MaxNLocator(4))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax2.tick_params(axis="y", labelsize=7.0, colors=_SAND, length=0, pad=2)
    for side in ("top", "right", "left", "bottom"):
        ax2.spines[side].set_visible(False)


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
        # 一行排不下就缩字号：八行卡片的行高只够一行字，折行必然压到下一行
        ax.text(-0.025, y, name, transform=ax.get_yaxis_transform(), fontsize=6.7,
                color=_MUT, va="center", ha="right", clip_on=False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_alpha(0.0)   # 坐标区缩进后别留一条白色色差带
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


def _top_industries(table: StjTable, k: int = 8) -> tuple[int, pd.Series]:
    """取最后一个有完整行业明细的年份，返回 (年份, 前 k 个行业)。

    平均工资表里 2025 年只更新了总量行，行业明细还停在 2024 年；若直接取末列，
    排名卡上只剩一行「合计」。这里按「该年非空行数 ≥ k+1」回溯，同时剔掉总量行——
    它与各行业之和相差不到 1%（或名字是各行业名的后缀），留着会把其余七个行业压成一条线。

    Args:
        table: 分行业表。
        k: 需要的行业数。

    Returns:
        (年份, 降序排列的前 k 个行业)。
    """
    first = table.indicators[0]
    others = table.indicators[1:]
    is_total = len(others) > 0 and all(n.endswith(first) for n in others)
    # 行业名在卡片里只有 2.2in 宽，「信息传输、计算机服务和软件业城镇非私营单位就业人员」
    # 这种 24 字的名字会直接顶出卡片；剥掉公共后缀只留行业名
    short = dict(zip(table.indicators, short_names(table.indicators)))
    for year in reversed(table.years):
        column = table.wide[year].dropna()
        if is_total and first in column.index:
            column = column.drop(first)
        if len(column) >= k:
            top = column.sort_values(ascending=False).head(k)
            return int(year), top.rename(index=short)
    year = int(table.years[-1])
    return year, table.wide[year].dropna().sort_values(ascending=False).head(k)


def _bottom_industries(table: StjTable, k: int = 5) -> tuple[int, pd.Series]:
    """取最后一个有完整行业明细的年份里增速最低的 k 个行业（降幅最大）。"""
    for year in reversed(table.years):
        column = table.wide[year].dropna()
        if len(column) >= 30:
            return int(year), column.sort_values(ascending=True).head(k)
    year = int(table.years[-1])
    return year, table.wide[year].dropna().sort_values(ascending=True).head(k)


def _decline_years(series: pd.Series) -> int:
    """连续下降的年数（从末年往前数）。"""
    vals = series.dropna()
    count = 0
    for i in range(len(vals) - 1, 0, -1):
        if vals.iloc[i] < vals.iloc[i - 1]:
            count += 1
        else:
            break
    return count


def _first_negative_note(growth: pd.Series) -> str:
    """判断当年增速是否为序列里第一次转负，返回可接在「比上年下降 x%」后面的短句。

    原文写的是「为 2020 年以来首次转负」，但 2020 年增速是 +2.6%，并没有转负；
    2025 年才是 2016 年有记录以来唯一的负增长。这类断言必须从数据里数出来，
    不能凭印象写——读者一核对就会发现年份对不上。
    """
    vals = growth.dropna()
    if len(vals) == 0 or float(vals.iloc[-1]) >= 0:
        return ""
    earlier = [float(v) for v in vals.iloc[:-1]]
    if earlier and all(v >= 0 for v in earlier):
        return f"，为 {int(vals.index[0])} 年有记录以来首次转负"
    if earlier and all(v < 0 for v in earlier):
        return "，已连续多年负增长"
    return ""


def build() -> Path:
    """组装固定资产投资和房地产驾驶舱并保存 PNG。"""
    _setup()
    root = data_root(Path(__file__).resolve().parent.parent) / "固定资产投资和房地产"

    def load(name: str) -> StjTable:
        return parse_table(root / f"{name}.csv")

    total_t = load("01-全社会固定资产投资")
    private_t = load("03-民间固定资产投资")
    property_t = load("36-房地产开发投资额")
    area_t = load("40-新建商品房销售面积")
    value_t = load("41-新建商品房销售额")
    fund_t = load("37-房地产开发企业本年实际到位资金")
    price_t = load("42-新建商品房平均销售价格")
    industry_t = load("09-按行业分固定资产投资增速")

    years = total_t.years
    total = total_t.wide.loc["全社会固定资产投资"]
    growth = total_t.wide.loc["全社会固定资产投资比上年增长"]
    private = private_t.wide.loc["民间固定资产（不含农户）投资额"]
    prop_total = property_t.wide.loc["房地产开发投资额"]
    sale_area = area_t.wide.loc["新建商品房销售面积"]
    sale_value = value_t.wide.loc["新建商品房销售额"]
    price = price_t.wide.loc["新建商品房平均销售价格"]

    purpose = pd.DataFrame({
        "住宅": property_t.wide.loc["房地产开发住宅投资额"] / 1e4,
        "办公楼": property_t.wide.loc["房地产开发办公楼投资额"] / 1e4,
        "商业营业用房": property_t.wide.loc["房地产开发商业营业用房投资额"] / 1e4,
        "其他": (prop_total
                 - property_t.wide.loc["房地产开发住宅投资额"]
                 - property_t.wide.loc["房地产开发办公楼投资额"]
                 - property_t.wide.loc["房地产开发商业营业用房投资额"]) / 1e4,
    }).T

    fund_abs = pd.DataFrame({
        "国内贷款": fund_t.wide.loc["房地产开发企业国内贷款"],
        "自筹资金": fund_t.wide.loc["房地产开发企业自筹资金"],
        "定金及预收款": fund_t.wide.loc["房地产开发企业定金及预收款"],
        "个人按揭贷款": fund_t.wide.loc["房地产开发企业个人按揭贷款"],
        "其他": (fund_t.wide.loc["房地产开发企业本年实际到位资金"]
                 - fund_t.wide.loc["房地产开发企业国内贷款"]
                 - fund_t.wide.loc["房地产开发企业自筹资金"]
                 - fund_t.wide.loc["房地产开发企业定金及预收款"]
                 - fund_t.wide.loc["房地产开发企业个人按揭贷款"]),
    })
    # wide.loc[名] 返回以年份为索引的序列，所以 fund_abs 是「年 × 分项」；
    # 按行求和得当年合计，逐行相除后转置成「分项 × 年」供堆叠图使用
    funds = (fund_abs.div(fund_abs.sum(axis=1), axis=0) * 100).T
    funds.columns = [int(c) for c in funds.columns]

    price_frame = pd.DataFrame({
        "住宅": price_t.wide.loc["住宅商品房平均销售价格"],
        "办公楼": price_t.wide.loc["办公楼商品房平均销售价格"],
        "商业营业用房": price_t.wide.loc["商业营业用房平均销售价格"],
    }).T

    latest, ind_rank = _top_industries(industry_t, k=8)
    low_year, ind_low = _bottom_industries(industry_t, k=5)

    tot_y, tot_prev = float(total[YEAR]), float(total[YEAR - 1])
    prop_y, prop_prev = float(prop_total[YEAR]), float(prop_total[YEAR - 1])
    area_y, area_prev = float(sale_area[YEAR]), float(sale_area[YEAR - 1])
    value_y, value_prev = float(sale_value[YEAR]), float(sale_value[YEAR - 1])
    # 均价表比总量表晚一年发布，缺 2025 就退回最近一个有数年份，不能当 0 处理
    price_last = int(price.dropna().index[-1])
    price_y, price_prev = float(price[price_last]), float(price[price_last - 1])
    priv_y, priv_prev = float(private[YEAR]), float(private[YEAR - 1])
    priv_share = priv_y / tot_y * 100
    prop_share = prop_y / tot_y * 100
    growth_y = float(growth[YEAR])

    fig = plt.figure(figsize=(_FIG_W, _FIG_H), dpi=200)
    fig.patch.set_facecolor("white")

    _header(fig)
    _year_rail(fig, years, YEAR)

    cards = [
        {"label": "全社会固定资产投资", "tag": "总量", "value": f"{tot_y / 1e4:,.1f}", "unit": "万亿元",
         "value_w": 0.070, "delta": f"{growth_y:+.1f}%", "delta_v": growth_y,
         "base": f"较上年 {tot_y - tot_prev:+,.0f} 亿元"},
        {"label": "民间投资占比", "tag": "结构", "value": f"{priv_share:.1f}", "unit": "%",
         "value_w": 0.040, "delta": f"{(priv_y / priv_prev - 1) * 100:+.1f}%",
         "delta_v": priv_y - priv_prev, "base": f"民间 {priv_y / 1e4:,.1f} 万亿元"},
        {"label": "房地产开发投资", "tag": "房地产", "value": f"{prop_y / 1e4:,.2f}", "unit": "万亿元",
         "value_w": 0.058, "delta": f"{(prop_y / prop_prev - 1) * 100:+.1f}%",
         "delta_v": prop_y - prop_prev, "base": f"占投资 {prop_share:.1f}%"},
        {"label": "商品房销售面积", "tag": "销售", "value": f"{area_y / 1e4:,.2f}", "unit": "亿㎡",
         "value_w": 0.052, "delta": f"{(area_y / area_prev - 1) * 100:+.1f}%",
         "delta_v": area_y - area_prev, "base": f"销售额 {value_y / 1e4:,.2f} 万亿元"},
        {"label": "商品房平均售价", "tag": f"价格 {price_last}", "value": f"{price_y:,.0f}", "unit": "元/㎡",
         "value_w": 0.056, "delta": f"{(price_y / price_prev - 1) * 100:+.1f}%",
         "delta_v": price_y - price_prev, "base": f"销售额 {value_y / 1e4:,.2f} 万亿元"},
    ]
    gap = 0.011
    card_w = (_SPAN - gap * (len(cards) - 1)) / len(cards)
    for i, item in enumerate(cards):
        _kpi(fig, [_L + i * (card_w + gap), 0.742, card_w, 0.102], item)

    _brief(
        fig,
        f"{YEAR} 速读",
        f"全社会固定资产投资 {tot_y / 1e4:,.2f} 万亿元，比上年{'下降' if growth_y < 0 else '增长'} "
        f"{abs(growth_y):.1f}%{_first_negative_note(growth)}。房地产开发投资 {prop_y / 1e4:,.2f} 万亿元"
        f"（{(prop_y / prop_prev - 1) * 100:+.1f}%），已连续 {_decline_years(prop_total)} 年下降，"
        f"占全部投资 {prop_share:.1f}%。商品房销售面积 {area_y / 1e4:,.2f} 亿平方米、"
        f"销售额 {value_y / 1e4:,.2f} 万亿元，{price_last} 年均价 {price_y:,.0f} 元／平方米"
        f"（{(price_y / price_prev - 1) * 100:+.1f}%）。到位资金中定金及预收款占 "
        f"{float(funds.loc['定金及预收款', YEAR]):.1f}%，自筹 {float(funds.loc['自筹资金', YEAR]):.1f}%。",
    )

    rect_m = [_L, 0.335, _SPAN * 0.600, 0.300]
    ax_main = _panel(fig, rect_m, "房地产开发投资：按用途",
                     "万亿元／年　面积自下而上堆叠，顶边＝开发投资总额")
    _legend(fig, rect_m, ["住宅", "办公楼", "商业营业用房", "其他"], [_DEEP, _MINT, _MINT2, _SAND])
    _main_property(ax_main, years, purpose, ["住宅", "办公楼", "商业营业用房", "其他"],
                   [_DEEP, _MINT, _MINT2, _SAND])

    rect_s = [_L + _SPAN * 0.612, 0.335, _SPAN * 0.388, 0.300]
    ax_s = _panel(fig, rect_s, "销售面积与均价",
                  "面积＝亿平方米（左轴）　圆点＝元／平方米（右轴）")
    # 右轴读数画在坐标区外，卡片右边缘只留 0.014；四位数刻度会伸出圆角边框，
    # 把绘图区右沿内收 0.026 才有落脚处
    _pos = ax_s.get_position()
    ax_s.set_position([_pos.x0, _pos.y0, _pos.width - 0.026, _pos.height])
    _sales_dots(ax_s, years, sale_area, price)

    y_b, h_b = 0.048, 0.252
    w_b = (_SPAN - gap * 2) / 3

    rect_f = [_L, y_b, w_b, h_b]
    ax_f = _panel(fig, rect_f, "开发到位资金结构", "占本年到位资金合计，自下而上累加至 100%")
    _legend(fig, rect_f, ["国内贷款", "自筹", "定金及预收款", "按揭", "其他"],
            [_SAND, _MINT2, _DEEP, _MAIN, _MINT])
    _stack_100(ax_f, years, funds, ["国内贷款", "自筹资金", "定金及预收款", "个人按揭贷款", "其他"],
               [_SAND, _MINT2, _DEEP, _MAIN, _MINT])

    rect_p = [_L + w_b + gap, y_b, w_b, h_b]
    ax_p = _panel(fig, rect_p, "商品房均价：按用途", "元／平方米　三格独立纵轴", tag="")
    ax_p.set_axis_off()
    _mini_areas(fig, [rect_p[0] + 0.014, y_b + 0.045, w_b - 0.030, h_b - 0.135],
                price_t.years, price_frame, list(price_frame.index), [_DEEP, _SAND, _MINT2],
                fmt="{:,.0f} 元")

    rect_i = [_L + 2 * (w_b + gap), y_b, w_b, h_b]
    ax_i = _panel(fig, rect_i, f"{latest} 年行业投资增速前 8", "%，按增速排序")
    ax_i.set_position([rect_i[0] + 0.162, rect_i[1] + 0.030,
                       rect_i[2] - 0.176, rect_i[3] - 0.100])
    _rank_dots(fig, ax_i, ind_rank, "%", fmt="{:+,.1f}")

    fig.text(
        _L, 0.016,
        "数据来源：国家统计局 · 年度数据（固定资产投资和房地产）。固定资产投资为全社会口径，"
        "2011 年起统计起点为计划总投资 500 万元以上项目；房地产开发投资为名义值，"
        "商品房均价＝销售额÷销售面积。全部数值未做插补。",
        fontsize=6.8, color=_FAINT, va="bottom",
    )

    out = root / "固定资产投资和房地产数据驾驶舱.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("已生成：" + str(build()))
