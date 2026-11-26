"""就业人员和工资 · 数据驾驶舱（一页式展板）。

版式参照驾驶舱类看板：顶部标题 + 年份滑轨 + KPI 卡条 + 速读条 + 主副图 + 底部三联。
配色统一为深青 / 薄荷 / 沙色，逐表图与展板共用一套色板。

图表约束（项目约定，优先级高于参考图）：
- 不用柱形、条形、折线及其变体。参考图里的柱状与折线在这里换成
  面积图、100% 堆叠面积图、真时间轴圆点；信息量与读数方式不变。
- 面积图一律零基线；含负值的数列纵轴下探并画零线。
- 不插补、不连断档：只在真实观测年份落点。

用法（任意目录执行均可）：
    python 统计局代码/就业人员和工资/dashboard.py
输出：
    年度数据/就业人员和工资/就业人员和工资数据驾驶舱.png
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

    fig.text(_L + 0.040, 0.958, "就业人员和工资 · 数据驾驶舱", fontsize=18,
             color=_INK, fontweight="bold", va="center")
    fig.text(_L + 0.040, 0.930, "2016 – 2025 年 · 总量 / 三次产业 / 城乡 / 工资 / 失业 · 国家统计局年度数据",
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



def _main_industry(ax: plt.Axes, years: list[int], frame: pd.DataFrame,
                   labels: list[str], colors: list[str]) -> None:
    """主图：三次产业就业人员堆叠面积（万人）+ 总量边界线。

    三个产业相加即就业人员总量，堆叠后顶边就是总量曲线：总量收缩、第三产业的
    份额继续抬升这两件事，在同一张图里同时读得出来，不必再画第二条总量轴。
    """
    base = np.zeros(len(years))
    tops: list[np.ndarray] = []
    for name, color in zip(labels, colors):
        vals = frame.loc[name].reindex(years).to_numpy(dtype=float) / 1e4
        top = base + vals
        ax.fill_between(years, base, top, color=color, linewidth=0, zorder=1)
        ax.plot(years, top, color="white", linewidth=1.0, zorder=2)
        tops.append(top)
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


def _wage_dots(ax: plt.Axes, years: list[int], frame: pd.DataFrame,
               labels: list[str], colors: list[str]) -> None:
    """副图：三类单位平均工资，只落点不连线。

    非私营、私营、外商投资三档工资本身就是「同一年的三个水平」，
    连成线会让人误以为在追同一个主体的轨迹；只落点，横向读差距、纵向读趋势。
    末点旁直接标读数，省掉一次对刻度的工夫。
    """
    for name, color in zip(labels, colors):
        s = frame.loc[name].reindex(years) / 1e4
        ax.plot(years, s, marker="o", markersize=4.4, color=color,
                markeredgecolor="white", markeredgewidth=0.9, linestyle="none",
                zorder=4, label=name)
        last = s.dropna()
        if len(last):
            ax.annotate(f"{float(last.iloc[-1]):.1f}万", xy=(last.index[-1], float(last.iloc[-1])),
                        xytext=(6, 3), textcoords="offset points", fontsize=6.8,
                        color=color, fontweight="bold", ha="left")

    hi = float(np.nanmax(frame.loc[labels].reindex(columns=years).to_numpy())) / 1e4
    ax.set_ylim(0, hi * 1.16)
    ax.set_xlim(years[0] - 0.4, years[-1] + 1.0)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}万"))
    _grid(ax, hi * 1.16)
    _clean(ax, x_ticks=years, step=2, label_size=7.2)


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


def build() -> Path:
    """组装就业人员和工资驾驶舱并保存 PNG。"""
    _setup()
    root = data_root(Path(__file__).resolve().parent.parent) / "就业人员和工资"

    def load(name: str) -> StjTable:
        return parse_table(root / f"{name}.csv")

    total_t = load("02-就业人员与三次产业构成")
    urban_t = load("03-就业人员与城乡构成")
    wage_t = load("18-城镇非私营单位就业人员平均工资")
    private_t = load("28-城镇私营单位平均工资（按行业分）")
    foreign_t = load("26-外商投资单位平均工资（按行业分）")
    ind_emp_t = load("06-城镇非私营单位就业人员（按行业分）")
    ind_wage_t = load("21-城镇非私营单位平均工资（按行业分）")

    years = total_t.years
    total = total_t.wide.loc["就业人员"]
    first = total_t.wide.loc["第一产业就业人员"]
    second = total_t.wide.loc["第二产业就业人员"]
    third = total_t.wide.loc["第三产业就业人员"]
    industry = pd.DataFrame({
        "第一产业": first,
        "第二产业": second,
        "第三产业": third,
    }).T

    urban = urban_t.wide.loc["城镇就业人员"]
    rural = urban_t.wide.loc["乡村就业人员"]

    wages = pd.DataFrame({
        "城镇非私营单位": wage_t.wide.loc["城镇非私营单位就业人员平均工资"],
        "城镇私营单位": private_t.wide.loc["城镇私营单位就业人员平均工资"],
        "外商投资单位": foreign_t.wide.loc["外商投资单位就业人员平均工资"],
    }).T

    tot_y = float(total[YEAR])
    tot_prev = float(total[YEAR - 1])
    urban_y = float(urban[YEAR])
    urban_share = urban_y / tot_y * 100
    urban_share_prev = float(urban[YEAR - 1]) / tot_prev * 100
    third_share = float(third[YEAR]) / tot_y * 100
    third_share_prev = float(third[YEAR - 1]) / tot_prev * 100
    wage_y = float(wage_t.wide.loc["城镇非私营单位就业人员平均工资", YEAR])
    wage_prev = float(wage_t.wide.loc["城镇非私营单位就业人员平均工资", YEAR - 1])
    private_y = float(private_t.wide.loc["城镇私营单位就业人员平均工资"].dropna().iloc[-1])
    private_last = int(private_t.wide.loc["城镇私营单位就业人员平均工资"].dropna().index[-1])

    latest_ind, emp_rank = _top_industries(ind_emp_t)
    latest_wage_ind, wage_rank = _top_industries(ind_wage_t)

    unemp_t = load("29-城镇登记失业人数与登记失业率")
    unemp = unemp_t.wide.loc["城镇登记失业人数"]
    unemp_y, unemp_prev = float(unemp[YEAR]), float(unemp[YEAR - 1])

    # 连续下降年数：从末年往前数，一旦出现回升就停
    downs, vals = 0, total.dropna()
    for i in range(len(vals) - 1, 0, -1):
        if vals.iloc[i] < vals.iloc[i - 1]:
            downs += 1
        else:
            break

    fig = plt.figure(figsize=(_FIG_W, _FIG_H), dpi=200)
    fig.patch.set_facecolor("white")

    _header(fig)
    _year_rail(fig, years, YEAR)

    cards = [
        {"label": "就业人员", "tag": "总量", "value": f"{tot_y:,.0f}", "unit": "万人",
         "value_w": 0.070, "delta": f"{(tot_y / tot_prev - 1) * 100:+.2f}%",
         "delta_v": tot_y - tot_prev, "base": f"较上年 {tot_y - tot_prev:+,.0f} 万人"},
        {"label": "城镇就业占比", "tag": "城乡", "value": f"{urban_share:.1f}", "unit": "%",
         "value_w": 0.040, "delta": f"{urban_share - urban_share_prev:+.2f}pp",
         "delta_v": urban_share - urban_share_prev, "base": f"城镇 {urban_y:,.0f} 万人"},
        {"label": "第三产业就业占比", "tag": "结构", "value": f"{third_share:.1f}", "unit": "%",
         "value_w": 0.040, "delta": f"{third_share - third_share_prev:+.2f}pp",
         "delta_v": third_share - third_share_prev, "base": f"第三 {float(third[YEAR]):,.0f} 万人"},
        {"label": "非私营单位平均工资", "tag": "工资", "value": f"{wage_y:,.0f}", "unit": "元",
         "value_w": 0.078, "delta": f"{(wage_y / wage_prev - 1) * 100:+.2f}%",
         "delta_v": wage_y - wage_prev, "base": f"私营 {private_y:,.0f} 元（{private_last}）"},
        {"label": "城镇登记失业人数", "tag": "失业", "value": f"{unemp_y:,.0f}", "unit": "万人",
         "value_w": 0.048, "delta": f"{(unemp_y / unemp_prev - 1) * 100:+.1f}%",
         "delta_v": unemp_y - unemp_prev, "base": "登记失业率已停发"},
    ]
    gap = 0.011
    card_w = (_SPAN - gap * (len(cards) - 1)) / len(cards)
    for i, item in enumerate(cards):
        _kpi(fig, [_L + i * (card_w + gap), 0.742, card_w, 0.102], item)

    _brief(
        fig,
        f"{YEAR} 速读",
        f"就业人员 {tot_y:,.0f} 万人，比上年减少 {abs(tot_y - tot_prev):,.0f} 万人"
        f"（{tot_y / tot_prev - 1:+.2%}），连续第 {downs} 年下降。"
        f"城镇就业 {urban_y:,.0f} 万人、占 {urban_share:.1f}%（{urban_share - urban_share_prev:+.2f}pp）；"
        f"第三产业就业 {float(third[YEAR]):,.0f} 万人、占 {third_share:.1f}%。"
        f"城镇非私营单位平均工资 {wage_y:,.0f} 元（{wage_y / wage_prev - 1:+.1%}），"
        f"私营单位 {private_y:,.0f} 元，两者相差 {wage_y / private_y:.2f} 倍。",
    )

    rect_m = [_L, 0.335, _SPAN * 0.600, 0.300]
    ax_main = _panel(fig, rect_m, "就业人员与三次产业构成",
                     "面积＝三次产业就业人员（亿人），顶边为就业人员总量")
    _legend(fig, rect_m, ["第一产业", "第二产业", "第三产业"], [_SAND, _MINT2, _DEEP])
    _main_industry(ax_main, years, industry, ["第一产业", "第二产业", "第三产业"],
                   [_SAND, _MINT2, _DEEP])

    rect_w = [_L + _SPAN * 0.612, 0.335, _SPAN * 0.388, 0.300]
    ax_w = _panel(fig, rect_w, "平均工资：三类单位",
                  "元／年　年末口径　圆点＝逐年观测值，不连线")
    _legend(fig, rect_w, ["非私营", "私营", "外商投资"], [_DEEP, _MINT2, _SAND])
    _wage_dots(ax_w, years, wages, list(wages.index), [_DEEP, _MINT2, _SAND])

    y_b, h_b = 0.048, 0.252
    w_b = (_SPAN - gap * 2) / 3

    rect_a = [_L, y_b, w_b, h_b]
    ax_a = _panel(fig, rect_a, "城乡就业结构", "占就业人员总量，自下而上累加至 100%")
    _legend(fig, rect_a, ["城镇", "乡村"], [_DEEP, _MINT2])
    share = pd.DataFrame({"城镇": urban / total * 100, "乡村": rural / total * 100}).T
    _stack_100(ax_a, years, share, ["城镇", "乡村"], [_DEEP, _MINT2])

    rect_e = [_L + w_b + gap, y_b, w_b, h_b]
    ax_e = _panel(fig, rect_e, f"{latest_ind} 年就业人数前 8 行业",
                  "城镇非私营单位，万人")
    ax_e.set_position([rect_e[0] + 0.150, rect_e[1] + 0.030,
                       rect_e[2] - 0.165, rect_e[3] - 0.100])
    _rank_dots(fig, ax_e, emp_rank, "")

    rect_g = [_L + 2 * (w_b + gap), y_b, w_b, h_b]
    ax_g = _panel(fig, rect_g, f"{latest_wage_ind} 年平均工资前 8 行业",
                  "城镇非私营单位，元／年")
    ax_g.set_position([rect_g[0] + 0.158, rect_g[1] + 0.030,
                       rect_g[2] - 0.173, rect_g[3] - 0.100])
    _rank_dots(fig, ax_g, wage_rank, "")

    fig.text(
        _L, 0.016,
        "数据来源：国家统计局 · 年度数据（就业人员和工资）。就业人员为年末数，按常住地划分城乡；"
        "工资为城镇非私营／私营／外商投资单位就业人员年平均工资，2025 年私营与外商投资单位数据尚未发布。"
        "城镇登记失业率自 2022 年起停止发布。全部数值未做插补。",
        fontsize=6.8, color=_FAINT, va="bottom",
    )

    out = root / "就业人员和工资数据驾驶舱.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("已生成：" + str(build()))
