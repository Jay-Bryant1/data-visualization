"""人口 · 数据驾驶舱（一页式展板）。

版式参照驾驶舱类看板：顶部标题 + 年份滑轨 + KPI 卡条 + 速读条 + 主副图 + 底部三联。
配色统一为深青 / 薄荷 / 沙色，逐表图与展板共用一套色板。

图表约束（项目约定，优先级高于参考图）：
- 不用柱形、条形、折线及其变体。参考图里的柱状与折线在这里换成
  面积图、100% 堆叠面积图、真时间轴圆点；信息量与读数方式不变。
- 面积图一律零基线；含负值的数列纵轴下探并画零线。
- 不插补、不连断档：只在真实观测年份落点。

用法（任意目录执行均可）：
    python 统计局代码/人口/dashboard.py
输出：
    年度数据/人口/人口数据驾驶舱.png
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

    fig.text(_L + 0.040, 0.958, "人口 · 数据驾驶舱", fontsize=18,
             color=_INK, fontweight="bold", va="center")
    fig.text(_L + 0.040, 0.930, "2016 – 2025 年 · 总量 / 城乡 / 年龄结构 / 自然变动 · 国家统计局年度数据",
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



def _main_pop(ax: plt.Axes, years: list[int], urban: pd.Series, rural: pd.Series) -> None:
    """主图：城乡人口堆叠面积（万人）+ 右轴城镇化率圆点。

    城镇与乡村人口相加即总人口，堆叠的顶端就是总人口曲线；
    总量收缩与「城镇继续增、乡村继续减」两条事实能在一张图里同时读出来。
    """
    u = urban.reindex(years) / 1e4
    r = rural.reindex(years) / 1e4
    top = u + r
    ax.fill_between(years, 0, u, color=_MINT, linewidth=0, zorder=1)
    ax.fill_between(years, u, top, color=_MINT2, alpha=0.55, linewidth=0, zorder=1)
    ax.plot(years, top, color=_DEEP, linewidth=1.6, zorder=3)
    ax.plot(years, u, color=_MAIN, linewidth=1.1, zorder=3)

    ax.set_ylim(0, float(top.max()) * 1.22)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _grid(ax, float(top.max()) * 1.22)
    _clean(ax, x_ticks=years, step=1, label_size=7.4)
    ax.set_xlim(years[0] - 0.4, years[-1] + 0.4)
    ax.tick_params(axis="x", pad=4)

    ax2 = ax.twinx()
    rate = (urban / (urban + rural) * 100).reindex(years)
    ax2.plot(years, rate, marker="o", markersize=5.0, color=_SAND,
             markeredgecolor="white", markeredgewidth=1.2, linestyle="none", zorder=5)
    ax2.plot([years[-1]], [rate.iloc[-1]], marker="o", markersize=9.0, markerfacecolor="none",
             markeredgecolor=_WARN, markeredgewidth=1.5, zorder=6)
    ax2.set_ylim(float(rate.min()) - 3.0, float(rate.max()) + 3.0)
    ax2.yaxis.set_major_locator(MaxNLocator(4))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.tick_params(axis="y", labelsize=7.2, colors=_SAND, length=0, pad=2)
    for side in ("top", "right", "left", "bottom"):
        ax2.spines[side].set_visible(False)


def _vital_dots(ax: plt.Axes, years: list[int], birth: pd.Series,
                death: pd.Series, natural: pd.Series) -> None:
    """副图：出生率 / 死亡率 / 自然增长率，只落点不连线。

    三条序列在同一 ‰ 量纲上，圆点的相对位置就是出生与死亡的此消彼长；
    自然增长率的负值区间铺一层浅橙，把「人口负增长」这段直接标出来。
    """
    n = natural.reindex(years).to_numpy(dtype=float)
    ax.axhline(0, color="#B9C4C4", linewidth=1.0, linestyle=(0, (3, 3)), zorder=3)
    ax.fill_between(years, 0, n, where=n < 0, color=_WARN, alpha=0.28,
                    linewidth=0, interpolate=True, zorder=1)
    ax.fill_between(years, 0, n, where=n >= 0, color=_MINT2, alpha=0.35,
                    linewidth=0, interpolate=True, zorder=1)

    series = [
        ("出生率", birth, _MAIN),
        ("死亡率", death, _SAND),
        ("自然增长率", natural, _DEEP),
    ]
    for name, s, color in series:
        vals = s.reindex(years)
        ax.plot(years, vals, marker="o", markersize=4.4, color=color,
                markeredgecolor="white", markeredgewidth=1.0, linestyle="none",
                zorder=5, label=name)

    lo = float(np.nanmin([birth.reindex(years).min(), death.reindex(years).min(), n.min()]))
    hi = float(np.nanmax([birth.reindex(years).max(), death.reindex(years).max(), n.max()]))
    ax.set_ylim(lo - 1.6, hi + 1.8)
    ax.set_xlim(years[0] - 0.4, years[-1] + 0.4)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}‰"))
    _grid(ax, hi + 1.8, lo - 1.6)
    _clean(ax, x_ticks=years, step=2, label_size=7.2)

    # 末年三条线的读数直接标在点旁，省掉一次「对齐刻度」的工夫
    for _, s, color in series:
        v = float(s.reindex(years).iloc[-1])
        ax.annotate(f"{v:+.2f}‰" if v < 0 else f"{v:.2f}‰", xy=(years[-1], v),
                    xytext=(5, 6), textcoords="offset points", fontsize=7.2,
                    color=color, fontweight="bold", ha="left")


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
    """组装人口驾驶舱并保存 PNG。"""
    _setup()
    root = data_root(Path(__file__).resolve().parent.parent) / "人口"

    def load(name: str) -> StjTable:
        return parse_table(root / f"{name}.csv")

    total_t = load("01-年末总人口与城乡构成")
    vital_t = load("02-人口出生率、死亡率与自然增长率")
    age_t = load("03-人口年龄构成与抚养比")

    years = total_t.years
    total = total_t.wide.loc["年末总人口"]
    urban = total_t.wide.loc["城镇人口"]
    rural = total_t.wide.loc["乡村人口"]

    birth = vital_t.wide.loc["人口出生率"]
    death = vital_t.wide.loc["人口死亡率"]
    natural = vital_t.wide.loc["人口自然增长率"]

    young = age_t.wide.loc["0-14岁人口"]
    working = age_t.wide.loc["15-64岁人口"]
    aged = age_t.wide.loc["65岁及以上人口"]
    age_frame = pd.DataFrame({
        "0-14 岁": young / total * 100,
        "15-64 岁": working / total * 100,
        "65 岁及以上": aged / total * 100,
    }).T
    depend = pd.DataFrame({
        "总抚养比": age_t.wide.loc["总抚养比"],
        "少儿抚养比": age_t.wide.loc["少儿抚养比"],
        "老年抚养比": age_t.wide.loc["老年抚养比"],
    }).T

    tot_y = float(total[YEAR])
    tot_prev = float(total[YEAR - 1])
    rate_y = float(urban[YEAR]) / tot_y * 100
    rate_prev = float(urban[YEAR - 1]) / tot_prev * 100
    aged_y = float(aged[YEAR]) / tot_y * 100
    aged_prev = float(aged[YEAR - 1]) / tot_prev * 100
    birth_y, birth_prev = float(birth[YEAR]), float(birth[YEAR - 1])
    death_y = float(death[YEAR])
    nat_y, nat_prev = float(natural[YEAR]), float(natural[YEAR - 1])
    neg_years = int((natural <= 0).sum())

    fig = plt.figure(figsize=(_FIG_W, _FIG_H), dpi=200)
    fig.patch.set_facecolor("white")

    _header(fig)
    _year_rail(fig, years, YEAR)

    cards = [
        {"label": "年末总人口", "tag": "年末", "value": f"{tot_y:,.0f}", "unit": "万人",
         "value_w": 0.078, "delta": f"{(tot_y / tot_prev - 1) * 100:+.2f}%",
         "delta_v": tot_y - tot_prev, "base": f"较上年 {tot_y - tot_prev:+,.0f} 万人"},
        {"label": "城镇化率", "tag": "城乡结构", "value": f"{rate_y:.1f}", "unit": "%",
         "value_w": 0.040, "delta": f"{rate_y - rate_prev:+.2f}pp", "delta_v": rate_y - rate_prev,
         "base": f"城镇 {float(urban[YEAR]):,.0f} 万人"},
        {"label": "人口出生率", "tag": "自然变动", "value": f"{birth_y:.2f}", "unit": "‰",
         "value_w": 0.040, "delta": f"{birth_y - birth_prev:+.2f}‰", "delta_v": birth_y - birth_prev,
         "base": f"死亡率 {death_y:.2f}‰"},
        {"label": "自然增长率", "tag": "自然变动", "value": f"{nat_y:.2f}", "unit": "‰",
         "value_w": 0.052, "delta": f"{nat_y - nat_prev:+.2f}‰", "delta_v": nat_y - nat_prev,
         "base": f"连续 {neg_years} 年为负"},
        {"label": "65 岁及以上占比", "tag": "年龄结构", "value": f"{aged_y:.1f}", "unit": "%",
         "value_w": 0.040, "delta": f"{aged_y - aged_prev:+.2f}pp", "delta_v": aged_y - aged_prev,
         "base": f"老年抚养比 {float(depend.loc['老年抚养比'][YEAR]):.1f}%"},
    ]
    gap = 0.011
    card_w = (_SPAN - gap * (len(cards) - 1)) / len(cards)
    for i, item in enumerate(cards):
        _kpi(fig, [_L + i * (card_w + gap), 0.742, card_w, 0.102], item)

    _brief(
        fig,
        f"{YEAR} 速读",
        f"年末总人口 {tot_y:,.0f} 万人，比上年减少 {abs(tot_y - tot_prev):,.0f} 万人"
        f"（{tot_y / tot_prev - 1:+.2%}）。城镇化率 {rate_y:.1f}%（较上年 {rate_y - rate_prev:+.2f}pp），"
        f"城镇人口 {float(urban[YEAR]):,.0f} 万人、乡村 {float(rural[YEAR]):,.0f} 万人。"
        f"出生率 {birth_y:.2f}‰、死亡率 {death_y:.2f}‰，自然增长率 {nat_y:.2f}‰，"
        f"已连续 {neg_years} 年为负。65 岁及以上占 {aged_y:.1f}%、老年抚养比 "
        f"{float(depend.loc['老年抚养比'][YEAR]):.1f}%，少儿抚养比降至 "
        f"{float(depend.loc['少儿抚养比'][YEAR]):.1f}%。",
    )

    # 主图：城乡人口堆叠 + 城镇化率
    rect_m = [_L, 0.335, _SPAN * 0.612, 0.300]
    ax_main = _panel(fig, rect_m,
                     "总人口与城乡构成",
                     "年末口径　面积＝城镇 / 乡村人口（亿人）　圆点＝城镇化率（右轴，%）")
    _legend(fig, rect_m, ["城镇", "乡村"], [_MAIN, _MINT2])
    _main_pop(ax_main, years, urban, rural)
    _callout(ax_main, years[-1], float((urban + rural)[YEAR]) / 1e4,
             f"{YEAR} 年  {tot_y:,.0f} 万人\n城镇化率 {rate_y:.1f}%　城镇 {float(urban[YEAR]):,.0f} 万人")

    ax_v = _panel(fig, [_L + _SPAN * 0.624, 0.335, _SPAN * 0.376, 0.300],
                  "人口自然变动", "‰　圆点＝逐年观测值　浅橙带＝自然增长率为负的区间")
    _legend(fig, [_L + _SPAN * 0.624, 0.335, _SPAN * 0.376, 0.300],
            ["出生率", "死亡率", "自然增长率"], [_MAIN, _SAND, _DEEP])
    _vital_dots(ax_v, years, birth, death, natural)

    y_b, h_b = 0.048, 0.252
    w_b = (_SPAN - gap * 2) / 3

    rect_a = [_L, y_b, w_b, h_b]
    ax_a = _panel(fig, rect_a, "年龄结构", "占年末总人口比重，自下而上累加至 100%")
    _legend(fig, rect_a, ["0-14 岁", "15-64 岁", "65 岁及以上"], [_SAND, _MINT2, _DEEP])
    _stack_100(ax_a, years, age_frame, ["0-14 岁", "15-64 岁", "65 岁及以上"],
               [_SAND, _MINT2, _DEEP])

    rect_d = [_L + w_b + gap, y_b, w_b, h_b]
    ax_d = _panel(fig, rect_d, "抚养比", "每 100 名劳动年龄人口负担的非劳动年龄人口",
                  tag="%")
    ax_d.set_axis_off()
    _mini_areas(fig, [rect_d[0] + 0.014, y_b + 0.020, w_b - 0.030, h_b - 0.100],
                years, depend, ["总抚养比", "少儿抚养比", "老年抚养比"],
                [_DEEP, _SAND, _WARN])

    rect_u = [_L + 2 * (w_b + gap), y_b, w_b, h_b]
    ax_u = _panel(fig, rect_u, "城乡人口比重", "城镇 / 乡村占年末总人口")
    _legend(fig, rect_u, ["城镇", "乡村"], [_DEEP, _MINT2])
    urban_share = pd.DataFrame({"城镇": urban / total * 100, "乡村": rural / total * 100}).T
    _stack_100(ax_u, years, urban_share, ["城镇", "乡村"], [_DEEP, _MINT2])

    fig.text(
        _L, 0.016,
        "数据来源：国家统计局 · 年度数据（人口）。总人口为年末全国人口，含现役军人；城乡按常住人口划分。"
        "出生率、死亡率、自然增长率均为千分率；年龄结构与抚养比来自年度人口变动情况抽样调查。"
        "全部数值未做插补，图中面积均为零基线。",
        fontsize=6.8, color=_FAINT, va="bottom",
    )

    out = root / "人口数据驾驶舱.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("已生成：" + str(build()))
