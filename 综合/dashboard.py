"""综合 · 数据驾驶舱（一页式展板）。

版式与「国民经济核算/dashboard.py」保持同一套：顶部标题 + 年份滑轨 + KPI 卡条 +
速读条 + 主副图 + 底部三联。配色同源，两份驾驶舱并排看是一套东西。

综合分类没有 GDP 这类总量指标，主角换成**法人单位**：
总量翻倍、结构向服务业迁移、行政建制的「乡」被「镇/街道」替代。

图表约束（项目约定，优先级高于参考图）：
- 不用柱形、条形、折线及其变体；排名类信息用哑铃点图（两点 + 细连线），
  不是条形图——条形的长度编码数量，这里的位置编码数量。
- 面积图一律零基线；100% 堆叠用于构成。
- 不插补、不连断档：机构类型表缺 2018、2023 年，涉及它的同比一律写明「较 2022」。

用法（任意目录执行均可）：
    python 统计局代码/综合/dashboard.py
输出：
    年度数据/综合/综合数据驾驶舱.png
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

# ── 色板（与 charts.py、国民经济核算驾驶舱同步）
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

YEAR = 2024          # 法人单位、人均产量等主体表的末年
RAIL_END = 2025      # 行政区划表到 2025 年，滑轨右端取全部表的最末年


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
        shadow: 是否带一层极淡的投影，压住浮层卡片。
        zorder: 层级；默认在坐标区之下。
    """
    if shadow:
        fig.patches.append(FancyBboxPatch(
            (rect[0], rect[1] - 0.004), rect[2], rect[3],
            boxstyle=f"round,pad=0,rounding_size={radius}",
            transform=fig.transFigure, facecolor=_DEEP, alpha=0.06,
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
    *,
    left: float = 0.014,
    right: float = 0.014,
) -> plt.Axes:
    """画一张图表卡片，返回卡片内部的绘图区。

    Args:
        fig: 画布。
        rect: 卡片 [x, y, w, h]。
        title: 卡片标题。
        sub: 标题下方小字。
        tag: 右上角标签。
        left: 绘图区距卡片左缘的距离。纵轴出现「4,500」这类四位数加千分位时需要
            加大，否则首字会被卡片边缘切掉。
        right: 绘图区距卡片右缘的距离。带右轴（twinx）的图必须加大，
            否则右轴刻度会落到卡片外面。

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
        rect[0] + left, rect[1] + 0.028,
        rect[2] - left - right, rect[3] - top_pad - 0.028,
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
    fig.add_artist(Circle((_L + 0.0155, 0.949), 0.0040, transform=fig.transFigure,
                          facecolor=_SAND, edgecolor="none", zorder=3))

    fig.text(_L + 0.040, 0.958, "综合 · 数据驾驶舱", fontsize=18,
             color=_INK, fontweight="bold", va="center")
    fig.text(_L + 0.040, 0.930,
             "2016 – 2025 年 · 法人单位 / 行政区划 / 人均农产品 / 民族自治地方 · 国家统计局年度数据",
             fontsize=9.2, color=_MUT, va="center")

    _rounded(fig, [0.858, 0.928, 0.052, 0.040], face=_DEEP, edge=_DEEP, radius=0.008)
    fig.text(0.884, 0.948, f"{YEAR} 年", fontsize=8.8, color="white",
             ha="center", va="center", fontweight="bold")
    _rounded(fig, [0.916, 0.928, 0.060, 0.040], face="white", edge="#D8DEDE", radius=0.008)
    fig.text(0.946, 0.948, "法人单位", fontsize=8.8, color=_MUT, ha="center", va="center")


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
    """一张 KPI 卡：左上标签、右上口径标签、数值 + 单位、底行变化与参照。

    底行两端对齐：左边短语随数字宽度变化，右边参照贴卡右缘。按固定偏移排时，
    长一点的「+5.0pp」会直接压到右半句上。
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

    fig.text(x0 + 0.012, y0 + 0.011, str(item["delta_label"]), fontsize=8.0,
             color=_FAINT, va="bottom")
    fig.text(x0 + 0.048, y0 + 0.011, str(item["delta"]), fontsize=8.6, va="bottom",
             color=_MAIN if float(item["delta_v"]) >= 0 else _WARN, fontweight="bold")
    fig.text(x0 + w - 0.012, y0 + 0.011, str(item["base"]), fontsize=8.0,
             va="bottom", ha="right", color=_FAINT)


def _legend(fig: plt.Figure, rect: list[float], labels: list[str],
            colors: list[str], *, size: float = 7.4) -> None:
    """卡片标题行右端的图例：一排「色点 + 名称」，右对齐。"""
    x = rect[0] + rect[2] - 0.012
    y = rect[1] + rect[3] - 0.026
    for name, color in zip(reversed(labels), reversed(colors)):
        fig.text(x, y, name, fontsize=size, color=_MUT, ha="right", va="center")
        cx = x - 0.0028 - len(name) * 0.0072 * (size / 7.4)
        fig.add_artist(Circle((cx, y), 0.0030, transform=fig.transFigure,
                              facecolor=color, edgecolor="white", linewidth=0.6, zorder=3))
        x = cx - 0.0115


def _callout(ax: plt.Axes, x: float, y: float, text: str,
             offset: tuple[int, int] = (-6, 26), ha: str = "right") -> None:
    """在图内落一个小的白底提示框，用引线连到数据点。

    比在画布上浮一张大卡片稳：跟着数据走，画到哪都不会压住主体。
    """
    ax.annotate(
        text, xy=(x, y), xytext=offset, textcoords="offset points",
        ha=ha, va="bottom", fontsize=8.0, color=_INK, linespacing=1.5,
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

    # 按画布宽度手工折行：优先在标点处断开，避免把「76.0% / 88.9%」劈成两半。
    # 96 字是这条速读条右缘的安全上限，112 会让第二行顶出卡片。
    per_line = 96
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


def _main_units(ax: plt.Axes, years: list[int], units: pd.Series,
                growth: pd.Series) -> None:
    """主图：法人单位总数（万个）面积 + 同比增速圆点（右轴，%）。"""
    ys = units.reindex(years) / 1e4
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


def _stack_100(
    ax: plt.Axes,
    years: list[int],
    frame: pd.DataFrame,
    labels: list[str],
    colors: list[str],
    *,
    right_label: bool = True,
    pct_size: float = 8.4,
) -> None:
    """100% 堆叠面积：各层份额从下往上累加，层间画白线分隔。

    右端只标百分比，层名交给卡片顶部的图例——名字写在右端会顶出卡片边界。

    Args:
        ax: 坐标区。
        years: 年份。
        frame: 行=层名、列=年份的份额矩阵（每列合计约 100）。
        labels: 层名（自下而上）。
        colors: 各层填充色。
        right_label: 是否在右端标出末年份额。
        pct_size: 右端百分比字号。
    """
    base = np.zeros(len(years))
    for name, color in zip(labels, colors):
        vals = frame.loc[name].reindex(years).to_numpy(dtype=float)
        top = base + vals
        ax.fill_between(years, base, top, color=color, linewidth=0, zorder=1)
        ax.plot(years, top, color="white", linewidth=1.0, zorder=2)
        base = top

    ax.set_ylim(0, 100)
    ax.set_xlim(years[0] - 0.3, years[-1] + (1.05 if right_label else 0.3))
    ax.set_yticks([0, 50, 100])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    _clean(ax, x_ticks=years, step=2, label_size=7.2)

    if not right_label:
        return
    cum = np.cumsum([float(frame.loc[n].reindex(years).iloc[-1]) for n in labels])
    bottoms = np.concatenate([[0.0], cum[:-1]])
    for top_v, bottom_v in zip(cum, bottoms):
        mid = (top_v + bottom_v) / 2
        ax.text(years[-1] + 0.32, mid, f"{top_v - bottom_v:.1f}%", fontsize=pct_size,
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
        ax.text(0.996, 0.94, f"{vals[-1]:+.1f}%", transform=ax.transAxes, fontsize=7.8,
                color=color, ha="right", va="top", fontweight="bold")


def _rank_dumbbell(
    ax: plt.Axes,
    names: list[str],
    first: np.ndarray,
    last: np.ndarray,
) -> None:
    """行业门类排名的哑铃点图：左端＝首年，右端＝末年，中间一道细连线。

    用点的位置而非条的长度编码数量，避开「条形图」禁令；连线只标记同一类别的两个
    时点，不承担趋势表达，所以不构成折线图。

    卡片只有约 1.5 英寸高，一格不到 0.25 英寸：数值标签必须与点同一行（放上方会
    直接压住下一格的类别名），类别名只保留 6 个。
    """
    n = len(names)
    ys = np.arange(n)[::-1]
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=7.6, color=_MUT)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", labelsize=7.0, colors=_MUT, length=0, pad=3)

    # 后端点的数值标签只需约 0.4 英寸；留到 1.5 倍会让右半张卡片空着
    right = float(np.nanmax(last)) * 1.22
    for y, a, b in zip(ys, first, last):
        ax.plot([a, b], [y, y], color="#DCE5E5", linewidth=1.3, zorder=1)
        ax.plot([a], [y], marker="o", markersize=4.0, color="#C9D6D6",
                markeredgecolor="white", markeredgewidth=0.7, zorder=2)
        ax.plot([b], [y], marker="o", markersize=5.0, color=_DEEP,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.text(b + right * 0.035, y, f"{b / 1e4:,.0f}", fontsize=7.2, color=_INK,
                ha="left", va="center")

    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(0, right)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e4:,.0f}万"))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#D8DEDE")


def build() -> Path:
    """组装驾驶舱并保存 PNG。"""
    _setup()
    root = data_root(Path(__file__).resolve().parent.parent) / "综合"

    def load(name: str) -> StjTable:
        return parse_table(root / f"{name}.csv")

    ind_t = load("法人单位数（三次产业）")        # 2016–2024 完整
    org_t = load("法人单位数（按机构类型）")      # 缺 2018、2023
    div_t = load("行政区划数")                    # 2016–2025
    agr_t = load("人均主要农产品占有量")          # 2016–2024
    sec_t = load("法人单位数（按行业门类）")      # 2016–2024

    years = ind_t.years
    total = ind_t.wide.sum(axis=0)

    # ── KPI 与速读用到的数字，全部由表实时算出
    t_now, t_prev = float(total[YEAR]), float(total[YEAR - 1])
    g_now = t_now / t_prev - 1
    g_start = t_now / float(total[years[0]]) - 1

    ent_now = float(org_t.wide.loc["企业法人单位数"][YEAR])
    org_tot_now = float(org_t.wide[YEAR].sum())
    ent_share = ent_now / org_tot_now
    ent_prev = float(org_t.wide.loc["企业法人单位数"][2022]) / float(org_t.wide[2022].sum())

    third = float(ind_t.wide.loc["第三产业法人单位数"][YEAR]) / t_now * 100
    third_prev = float(ind_t.wide.loc["第三产业法人单位数"][YEAR - 1]) / t_prev * 100

    town = float(div_t.wide.loc["镇数"][RAIL_END])
    township_total = float(div_t.wide.loc["乡级区划数"][RAIL_END])
    township_prev = float(div_t.wide.loc["乡级区划数"][YEAR])
    township_low = float(div_t.wide.loc["乡级区划数"][div_t.years[0]])

    grain = float(agr_t.wide.loc["粮食人均占有量"][YEAR])
    grain_prev = float(agr_t.wide.loc["粮食人均占有量"][YEAR - 1])
    power = float(agr_t.wide.loc["人均发电量"][YEAR])
    power_start = float(agr_t.wide.loc["人均发电量"][agr_t.years[0]])
    cloth = float(agr_t.wide.loc["人均布产量"][YEAR])
    cloth_start = float(agr_t.wide.loc["人均布产量"][agr_t.years[0]])

    fig = plt.figure(figsize=(_FIG_W, _FIG_H), dpi=200)
    fig.patch.set_facecolor("white")

    _header(fig)
    _year_rail(fig, list(range(years[0], RAIL_END + 1)), YEAR)

    # ── KPI 条
    cards = [
        {"label": "法人单位总数", "tag": "2024", "value": f"{t_now / 1e4:,.0f}",
         "unit": "万个", "value_w": 0.052, "delta_label": "同比",
         "delta": f"{g_now:+.1%}", "delta_v": g_now,
         "base": f"较 2016 年 {g_start:+.1%}"},
        {"label": "企业法人占比", "tag": "机构类型", "value": f"{ent_share * 100:.1f}",
         "unit": "%", "value_w": 0.040, "delta_label": "较 2022",
         "delta": f"{(ent_share - ent_prev) * 100:+.1f}pp", "delta_v": ent_share - ent_prev,
         "base": f"{ent_now / 1e4:,.0f} 万个"},
        {"label": "第三产业占比", "tag": "产业", "value": f"{third:.1f}",
         "unit": "%", "value_w": 0.040, "delta_label": "同比",
         "delta": f"{third - third_prev:+.1f}pp", "delta_v": third - third_prev,
         "base": f"{float(ind_t.wide.loc['第三产业法人单位数'][YEAR]) / 1e4:,.0f} 万个"},
        {"label": "乡级区划", "tag": "2025", "value": f"{township_total:,.0f}",
         "unit": "个", "value_w": 0.052, "delta_label": "较上年",
         "delta": f"{township_total - township_prev:+.0f} 个",
         "delta_v": township_total - township_prev,
         "base": f"镇 {town:,.0f} / 乡 {float(div_t.wide.loc['乡数'][RAIL_END]):,.0f}"},
        {"label": "人均粮食占有量", "tag": "农业", "value": f"{grain:,.0f}",
         "unit": "公斤", "value_w": 0.052, "delta_label": "同比",
         "delta": f"{grain / grain_prev - 1:+.1%}", "delta_v": grain - grain_prev,
         "base": f"人均发电量 {power:,.0f} 度"},
    ]
    gap = 0.011
    card_w = (_SPAN - gap * (len(cards) - 1)) / len(cards)
    for i, item in enumerate(cards):
        _kpi(fig, [_L + i * (card_w + gap), 0.742, card_w, 0.102], item)

    # 正文必须压进两行（约 190 字）：_brief 只渲染 lines[:2]，超出的部分会被静默丢弃，
    # 第一版就因此丢掉了「人均布产量」整句。
    _brief(
        fig,
        f"{YEAR} 速读",
        f"法人单位 {t_now / 1e4:,.0f} 万个，同比 {g_now:+.1%}，为 2016 年的 "
        f"{t_now / float(total[years[0]]):.2f} 倍；企业法人占 {ent_share * 100:.1f}%，"
        f"第三产业占 {third:.1f}%。乡级建制中「乡」持续退出：乡数由 "
        f"{float(div_t.wide.loc['乡数'][div_t.years[0]]):,.0f} 降至 "
        f"{float(div_t.wide.loc['乡数'][RAIL_END]):,.0f}，街道增至 "
        f"{float(div_t.wide.loc['街道数'][RAIL_END]):,.0f}。人均粮食 {grain:,.0f} 公斤、"
        f"发电量 {power:,.0f} 度创新高，人均布产量却由 {cloth_start:.1f} 米降至 {cloth:.1f} 米。",
    )

    # ── 主图：法人单位总量 + 增速
    main_rect = [_L, 0.335, _SPAN * 0.612, 0.300]
    growth = total.pct_change() * 100
    ax_main = _panel(fig, main_rect, "法人单位总数与同比增速",
                     "面积＝单位数（万个）　圆点＝增速（右轴，%）　2023 年为普查修订",
                     tag="全部单位 · 年度", left=0.034, right=0.040)
    _main_units(ax_main, years, total, growth)
    _callout(ax_main, YEAR, float(total[YEAR]) / 1e4,
             f"{YEAR} 年  {t_now / 1e4:,.0f} 万个\n同比 {g_now:+.1%}　企业 {ent_now / 1e4:,.0f} 万个",
             offset=(-12, 22))
    # 2023 年那次回落是第五次全国经济普查后的数据修订，不是真实收缩。
    # 只加一个不带框的小标注：主图已经有一个提示卡，再放一张会直接压在它身上。
    if 2023 in years:
        ax_main.annotate(
            "普查修订", xy=(2023, float(total[2023]) / 1e4), xytext=(0, -13),
            textcoords="offset points", ha="center", va="top", fontsize=7.4,
            color=_WARN, zorder=7,
        )

    # ── 副图：乡级建制的构成迁移
    div_years = div_t.years
    div_share = pd.DataFrame({
        "镇": div_t.wide.loc["镇数"] / div_t.wide.loc["乡级区划数"] * 100,
        "乡": div_t.wide.loc["乡数"] / div_t.wide.loc["乡级区划数"] * 100,
        "街道": div_t.wide.loc["街道数"] / div_t.wide.loc["乡级区划数"] * 100,
    }).T
    rect_d = [_L + _SPAN * 0.624, 0.335, _SPAN * 0.376, 0.300]
    # 这张卡片右上角留给图例，不再放 tag：两个都贴右缘会叠成一行
    ax_d = _panel(fig, rect_d, "乡级建制：街道与镇替代乡",
                  "乡级区划的构成（%），其余为苏木、民族乡等")
    _legend(fig, rect_d, ["镇", "乡", "街道"], [_MINT2, _SAND, _DEEP], size=7.2)
    _stack_100(ax_d, div_years, div_share, ["镇", "乡", "街道"],
               [_MINT2, _SAND, _DEEP], pct_size=7.8)

    # ── 底部三联
    y_b, h_b = 0.048, 0.252
    w_b = (_SPAN - gap * 2) / 3

    share = pd.DataFrame({
        "第一产业": ind_t.wide.loc["第一产业法人单位数"],
        "第二产业": ind_t.wide.loc["第二产业法人单位数"],
        "第三产业": ind_t.wide.loc["第三产业法人单位数"],
    }).T
    share = share / share.sum(axis=0) * 100

    rect_s = [_L, y_b, w_b, h_b]
    ax_s = _panel(fig, rect_s, "法人单位的三次产业结构", "占全部法人单位比重，累加至 100%")
    _legend(fig, rect_s, ["一产", "二产", "三产"], [_SAND, _MINT2, _DEEP], size=7.2)
    _stack_100(ax_s, years, share, ["第一产业", "第二产业", "第三产业"],
               [_SAND, _MINT2, _DEEP])

    speeds = pd.DataFrame({
        "第一产业": ind_t.wide.loc["第一产业法人单位数"].pct_change() * 100,
        "第二产业": ind_t.wide.loc["第二产业法人单位数"].pct_change() * 100,
        "第三产业": ind_t.wide.loc["第三产业法人单位数"].pct_change() * 100,
    }).T
    rect_v = [_L + w_b + gap, y_b, w_b, h_b]
    # 这张卡片的绘图区由三个迷你坐标区拼成，外层坐标区必须关掉，
    # 否则会露出一套 0.0–1.0 的默认刻度压在迷你图上
    ax_v = _panel(fig, rect_v, "三次产业法人单位增速", "同比（%），三格独立纵轴",
                  tag="比上年")
    ax_v.set_axis_off()
    _mini_areas(fig, [rect_v[0] + 0.014, y_b + 0.020, w_b - 0.030, h_b - 0.100],
                years, speeds, ["第一产业", "第二产业", "第三产业"], [_SAND, _MINT2, _DEEP])

    # ── 行业门类：首末年哑铃对照
    sec_names = {
        "批发和零售业法人单位数": "批发零售",
        "租赁和商务服务业法人单位数": "租赁商务服务",
        "制造业法人单位数": "制造业",
        "建筑业法人单位数": "建筑业",
        "科学研究和技术服务业法人单位数": "科研技术服务",
        "农、林、牧、渔业法人单位数": "农林牧渔",
        "信息传输、软件和信息技术服务业法人单位数": "信息传输软件",
        "公共管理、社会保障和社会组织法人单位数": "公共管理社保",
        "房地产业法人单位数": "房地产业",
    }
    col_last, col_first = sec_t.years[-1], sec_t.years[0]
    ranked = (sec_t.wide.loc[list(sec_names)].drop(labels=["法人单位数"], errors="ignore")
              .sort_values(col_last, ascending=False))
    # 只放前六：卡片高 1.5 英寸，八格会让类别名、数值标签和下一行挤在一起
    top = ranked.head(6)
    rect_r = [_L + 2 * (w_b + gap), y_b, w_b, h_b]
    ax_r = _panel(fig, rect_r, "法人单位的行业分布",
                  f"{col_last} 年单位数前六行业门类（万个）", left=0.052)
    _legend(fig, rect_r, [f"{col_first} 年", f"{col_last} 年"], ["#C9D6D6", _DEEP], size=7.0)
    _rank_dumbbell(
        ax_r,
        [sec_names[i] for i in top.index],
        top[col_first].to_numpy(dtype=float),
        top[col_last].to_numpy(dtype=float),
    )

    fig.text(
        _L, 0.016,
        "数据来源：国家统计局 · 年度数据（综合）。法人单位为年末数，2016–2024 年；行政区划至 2025 年底；"
        "人均产量按年平均人口计算。机构类型表缺 2018、2023 年，相关同比统一标注为「较 2022」。"
        "2023 年法人单位数回落系第五次全国经济普查后的数据修订，非真实收缩，图中已单独标注。",
        fontsize=6.8, color=_FAINT, va="bottom",
    )

    out = root / "综合数据驾驶舱.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("已生成：" + str(build()))
