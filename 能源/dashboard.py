"""能源 · 数据驾驶舱（一页式展板）。

版式参照驾驶舱类看板：顶部标题 + 年份滑轨 + KPI 卡条 + 速读条 + 主副图 + 底部三联。
配色统一为深青 / 薄荷 / 沙色，逐表图与展板共用一套色板。

图表约束（项目约定，优先级高于参考图）：
- 不用柱形、条形、折线及其变体。生产与消费用面积图，结构用 100% 堆叠面积，
  行业排名用克利夫兰点图，产量趋势用迷你面积。
- 面积图一律零基线；缺口（消费−生产）挂右轴用圆点。
- 不插补、不连断档：只在真实观测年份落点。

用法（任意目录执行均可）：
    python 统计局代码/能源/dashboard.py
输出：
    年度数据/能源/能源数据驾驶舱.png
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
    """在图坐标里画一个圆角矩形（卡片底）。"""
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
    """画一张图表卡片，返回卡片内部的绘图区。"""
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


def _lift(ax: plt.Axes, pad: float = 0.034) -> None:
    """把绘图区整体上提，给卡片底部的图例带让位（顶边不动，不挤压标题）。"""
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 + pad, pos.width, pos.height - pad])


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
    fig.add_artist(Circle((_L + 0.0155, 0.949), 0.0092, transform=fig.transFigure,
                          facecolor="none", edgecolor="white", linewidth=1.6, zorder=3))
    fig.add_artist(Circle((_L + 0.0155, 0.949), 0.0034, transform=fig.transFigure,
                          facecolor="white", edgecolor="none", zorder=3))

    fig.text(_L + 0.040, 0.958, "能源 · 数据驾驶舱", fontsize=18,
             color=_INK, fontweight="bold", va="center")
    fig.text(_L + 0.040, 0.930, "2016 – 2025 年 · 能源生产与消费 / 品种结构 / 分行业消费 / 电力产量 · 国家统计局年度数据",
             fontsize=9.2, color=_MUT, va="center")

    _rounded(fig, [0.858, 0.928, 0.052, 0.040], face=_DEEP, edge=_DEEP, radius=0.008)
    fig.text(0.884, 0.948, f"{YEAR} 年", fontsize=8.8, color="white",
             ha="center", va="center", fontweight="bold")
    _rounded(fig, [0.916, 0.928, 0.060, 0.040], face="white", edge="#D8DEDE", radius=0.008)
    fig.text(0.946, 0.948, "标煤口径", fontsize=8.2, color=_MUT, ha="center", va="center")


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
    """一张 KPI 卡：左上标签、右上口径标签、数值 + 单位、底行同比与较上年。"""
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

    fig.text(x0 + 0.012, y0 + 0.011, str(item.get("delta_label", "同比")),
             fontsize=8.0, color=_FAINT, va="bottom")
    fig.text(x0 + 0.038, y0 + 0.011, str(item["delta"]), fontsize=8.6, va="bottom",
             color=_MAIN if float(item["delta_v"]) >= 0 else _WARN, fontweight="bold")
    fig.text(x0 + w - 0.012, y0 + 0.011, str(item["base"]), fontsize=8.0,
             va="bottom", ha="right", color=_FAINT)


def _legend(fig: plt.Figure, rect: list[float], labels: list[str],
            colors: list[str], edges: list[str] | None = None) -> None:
    """卡片标题行右端的图例：一排「色点 + 名称」，右对齐。

    面积图的色点必须用**填充色**而不是描边色：生产面积是浅薄荷、边线是中青，
    图例若跟着边线走，读者会把深青的消费线当成生产面积。
    浅色点再把描边换成同色系深一档，免得在白底上看不见。
    """
    x = rect[0] + rect[2] - 0.012
    y = rect[1] + rect[3] - 0.026
    edge_list = list(edges) if edges else ["white"] * len(labels)
    for name, color, edge in zip(reversed(labels), reversed(colors), reversed(edge_list)):
        fig.text(x, y, name, fontsize=7.4, color=_MUT, ha="right", va="center")
        cx = x - 0.0028 - len(name) * 0.0072
        fig.add_artist(Circle((cx, y), 0.0030, transform=fig.transFigure,
                              facecolor=color, edgecolor=edge, linewidth=0.8, zorder=3))
        x = cx - 0.0115


def _legend_strip(fig: plt.Figure, rect: list[float], labels: list[str],
                  colors: list[str], *, y_off: float = 0.020,
                  size: float = 6.9) -> None:
    """卡片底部的横排图例（层数多时塞不进标题行）。"""
    y = rect[1] + y_off
    step = rect[2] / len(labels)
    for i, (name, color) in enumerate(zip(labels, colors)):
        x = rect[0] + 0.010 + i * step
        fig.add_artist(Circle((x + 0.0030, y), 0.0028, transform=fig.transFigure,
                              facecolor=color, edgecolor="white", linewidth=0.6, zorder=3))
        fig.text(x + 0.0086, y, name, fontsize=size, color=_MUT, va="center")


def _callout(ax: plt.Axes, x: float, y: float, text: str) -> None:
    """在图内落一个小的白底提示框，用引线连到数据点。"""
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


def _main_supply(ax: plt.Axes, years: list[int], cons: pd.Series,
                 prod: pd.Series) -> None:
    """主图：能源消费 / 一次能源生产两层面积（亿吨标准煤），中间夹出供需缺口带。

    消费垫底、生产压在上面，两者之间的沙色带就是「消费−生产」。
    不用第二根纵轴画圆点：缺口与两层面积同为亿吨标准煤，直接夹出来读，
    不必在两套刻度之间来回换算，也不会把缺口点误读成第三根曲线。
    """
    c = cons.reindex(years) / 1e4
    p = prod.reindex(years) / 1e4
    ax.fill_between(years, 0, p, color=_MINT2, alpha=0.85, linewidth=0, zorder=2)
    ax.fill_between(years, p, c, color=_SAND, alpha=0.55, linewidth=0, zorder=3)
    ax.plot(years, c, color=_DEEP, linewidth=1.6, zorder=4)
    ax.plot(years, p, color=_MAIN, linewidth=1.2, zorder=4)
    top = float(np.nanmax(c))
    ax.set_ylim(0, top * 1.26)
    ax.set_xlim(years[0] - 0.4, years[-1] + 0.4)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _grid(ax, top * 1.26)
    _clean(ax, x_ticks=years, step=2, label_size=7.2)
    ax.tick_params(axis="x", pad=4)


def _rank_dots(ax: plt.Axes, series: pd.Series, unit: str,
               fmt: str = "{:,.0f}") -> None:
    """卡片内的一列排名：一行一个类别，点落在量级上，数值贴点右侧。"""
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
        ax.text(-0.025, y, name, transform=ax.get_yaxis_transform(), fontsize=7.4,
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
    """100% 堆叠面积：各层份额自下而上累加，层间画白线分隔，右端只标百分比。"""
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
    mids = [(t + b) / 2 for t, b in zip(cum, bottoms)]
    # 标注最小间距随坐标区高度走：矮卡片上 6.8pp 只折合 6pt，8.4pt 的字必然叠在一起
    h_in = ax.get_position().height * ax.figure.get_figheight()
    step = max(6.8, 100 * 1.45 * 8.4 / (h_in * 72))
    for i in range(1, len(mids)):
        mids[i] = max(mids[i], mids[i - 1] + step)
    over = mids[-1] - 99.0
    if over > 0:
        mids = [m - over for m in mids]
        for i in range(len(mids) - 2, -1, -1):
            mids[i] = min(mids[i], mids[i + 1] - step)
    for (top_v, bottom_v), mid in zip(zip(cum, bottoms), mids):
        ax.text(years[-1] + 0.35, mid, f"{top_v - bottom_v:.1f}%", fontsize=8.4,
                color=_DEEP, va="center", ha="left", fontweight="bold")


def _mini_areas(fig: plt.Figure, rect: list[float], years: list[int],
                frame: pd.DataFrame, labels: list[str], colors: list[str],
                fmts: list[str]) -> None:
    """伪小倍数：竖排若干格迷你面积图，每格可带自己的读数格式（量纲各不相同）。

    迷你图画成 sparkline：不画刻度、不画轴线，只留面积 + 末点；
    名称写在面板上方的标签带里，读数右对齐到面板右缘，长名称不会撞上读数。
    """
    n = len(labels)
    label_h, pad = 0.0118, 0.006
    h = (rect[3] - label_h * n - pad * (n - 1)) / n
    for i, (name, color, fmt) in enumerate(zip(labels, colors, fmts)):
        bottom = rect[1] + (n - 1 - i) * (h + pad + label_h) + label_h
        ax = fig.add_axes([rect[0], bottom, rect[2], h])
        vals = frame.loc[name].reindex(years).to_numpy(dtype=float)
        ax.fill_between(years, 0, vals, color=color, alpha=0.26, linewidth=0)
        ax.plot(years, vals, color=color, linewidth=1.3, zorder=2)
        ax.plot([years[-1]], [vals[-1]], marker="o", markersize=4.0, color=color,
                markeredgecolor="white", markeredgewidth=1.0, zorder=3)

        ax.set_ylim(0, float(np.nanmax(vals)) * 2.30)
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

        fig.text(rect[0] + 0.003, bottom + h + 0.0015, name, fontsize=7.4, color=color,
                 ha="left", va="bottom", fontweight="bold")
        fig.text(rect[0] + rect[2] - 0.002, bottom + h + 0.0015, fmt.format(vals[-1]),
                 fontsize=7.4, color=color, ha="right", va="bottom", fontweight="bold")
        ax.patch.set_alpha(0.0)


def _short_industry(name: str) -> str:
    """把「化学原料及化学制品制造业能源消费总量」压成「化学原料及化学制品」。"""
    for cut in ("能源消费总量", "制造业", "业"):
        name = name.replace(cut, "")
    return name


def build() -> Path:
    """组装能源驾驶舱并保存 PNG。"""
    _setup()
    root = data_root(Path(__file__).resolve().parent.parent) / "能源"

    def load(name: str) -> StjTable:
        return parse_table(root / f"{name}.csv")

    cons_t = load("14-能源消费总量及构成")
    prod_t = load("11-一次能源生产总量及构成")
    ind_t = load("15-分行业能源消费总量")
    out_t = load("37-主要能源产品产量")

    years = cons_t.years
    cons = cons_t.wide.loc["能源消费总量"]
    coal_share = cons_t.wide.loc["煤炭占能源消费总量的比重"]
    prod = prod_t.wide.loc["一次能源生产总量"]

    cons_y, cons_prev = float(cons[YEAR]), float(cons[YEAR - 1])
    prod_y, prod_prev = float(prod[YEAR]), float(prod[YEAR - 1])
    coal_y, coal_prev = float(coal_share[YEAR]), float(coal_share[YEAR - 1])

    coal_out = out_t.wide.loc["原煤产量"]
    power_out = out_t.wide.loc["发电量"]
    coal_out_y, coal_out_prev = float(coal_out[YEAR]), float(coal_out[YEAR - 1])
    power_y, power_prev = float(power_out[YEAR]), float(power_out[YEAR - 1])

    cons_struct = pd.DataFrame({
        "煤炭": cons_t.wide.loc["煤炭占能源消费总量的比重"],
        "石油": cons_t.wide.loc["石油占能源消费总量的比重"],
        "天然气": cons_t.wide.loc["天然气占能源消费总量的比重"],
        "一次电力及其他": cons_t.wide.loc["一次电力及其他能源占能源总量的比重"],
    }).T
    prod_struct = pd.DataFrame({
        "原煤": prod_t.wide.loc["原煤占一次能源生产总量的比重"],
        "原油": prod_t.wide.loc["原油占一次能源生产总量的比重"],
        "天然气": prod_t.wide.loc["天然气占一次能源生产总量的比重"],
        "一次电力及其他": prod_t.wide.loc["一次电力及其他能源占一次能源生产总量的比重"],
    }).T

    # 行业层级有嵌套（工业 ⊃ 制造业 ⊃ 各制造大类），直接排名会把汇总行和明细混在一起；
    # 这里只取互不包含的门类与大类，再按 2023 年消费量排序
    picked = ["化学原料及化学制品制造业", "居民生活", "黑色金属冶炼及压延加工业",
              "交通运输、仓储和邮政业", "石油、煤炭及其他燃料加工业", "非金属矿物制品业",
              "电力、热力、燃气及水生产和供应业", "批发和零售业、住宿和餐饮业"]
    ind_year = 2023
    ind_map = {f"{p}能源消费总量" if p != "居民生活" else "居民生活能源消费总量": p
               for p in picked}
    ind_series = pd.Series(
        {short: float(ind_t.wide.at[raw, ind_year]) for raw, short in ind_map.items()
         if raw in ind_t.wide.index},
    ).sort_values(ascending=False)

    out_series = pd.DataFrame({
        "原煤（亿吨）": out_t.wide.loc["原煤产量"] / 1e4,
        "原油（亿吨）": out_t.wide.loc["原油产量"] / 1e4,
        "天然气（亿立方米）": out_t.wide.loc["天然气产量"],
        "发电量（万亿千瓦时）": out_t.wide.loc["发电量"] / 1e4,
    }).T

    fig = plt.figure(figsize=(_FIG_W, _FIG_H), dpi=200)
    fig.patch.set_facecolor("white")

    _header(fig)
    _year_rail(fig, years, YEAR)

    cards = [
        {"label": "能源消费总量", "tag": "消费端", "value": f"{cons_y / 1e4:,.2f}",
         "unit": "亿吨标准煤", "value_w": 0.062,
         "delta": f"{(cons_y / cons_prev - 1) * 100:+.1f}%",
         "delta_v": cons_y - cons_prev, "base": f"较上年 {cons_y - cons_prev:+,.0f} 万吨"},
        {"label": "一次能源生产总量", "tag": "生产端", "value": f"{prod_y / 1e4:,.2f}",
         "unit": "亿吨标准煤", "value_w": 0.062,
         "delta": f"{(prod_y / prod_prev - 1) * 100:+.1f}%",
         "delta_v": prod_y - prod_prev, "base": f"相当于消费的 {prod_y / cons_y:.1%}"},
        {"label": "原煤产量", "tag": "实物量", "value": f"{coal_out_y / 1e4:,.2f}",
         "unit": "亿吨", "value_w": 0.058,
         "delta": f"{(coal_out_y / coal_out_prev - 1) * 100:+.1f}%",
         "delta_v": coal_out_y - coal_out_prev,
         "base": f"占一次能源生产 {float(prod_t.wide.at['原煤占一次能源生产总量的比重', YEAR]):.1f}%"},
        {"label": "发电量", "tag": "全口径", "value": f"{power_y / 1e4:,.2f}",
         "unit": "万亿千瓦时", "value_w": 0.062,
         "delta": f"{(power_y / power_prev - 1) * 100:+.1f}%",
         "delta_v": power_y - power_prev, "base": f"较上年 {power_y - power_prev:+,.0f} 亿千瓦时"},
        {"label": "煤炭消费占比", "tag": "占消费总量", "value": f"{coal_y:.1f}",
         "unit": "%", "value_w": 0.040,
         "delta": f"{coal_y - coal_prev:+.1f}pp", "delta_v": coal_y - coal_prev,
         "base": f"2016 年为 {float(coal_share[2016]):.1f}%"},
    ]
    gap_px = 0.011
    card_w = (_SPAN - gap_px * (len(cards) - 1)) / len(cards)
    for i, item in enumerate(cards):
        _kpi(fig, [_L + i * (card_w + gap_px), 0.742, card_w, 0.102], item)

    _brief(
        fig,
        f"{YEAR} 速读",
        f"能源消费总量 {cons_y / 1e4:,.2f} 亿吨标准煤，比上年 {(cons_y / cons_prev - 1) * 100:+.1f}%；"
        f"一次能源生产 {prod_y / 1e4:,.2f} 亿吨标准煤（{(prod_y / prod_prev - 1) * 100:+.1f}%），"
        f"相当于消费的 {prod_y / cons_y:.1%}。煤炭占消费 {coal_y:.1f}%，比上年 {coal_y - coal_prev:+.1f} 个百分点；"
        f"原煤产量 {coal_out_y / 1e4:,.2f} 亿吨（{(coal_out_y / coal_out_prev - 1) * 100:+.1f}%），"
        f"发电量 {power_y / 1e4:,.2f} 万亿千瓦时（{(power_y / power_prev - 1) * 100:+.1f}%）。",
    )

    rect_m = [_L, 0.335, _SPAN * 0.600, 0.300]
    ax_main = _panel(fig, rect_m, "能源消费与一次能源生产",
                     "面积＝消费 / 生产（亿吨标准煤）　沙色带＝消费减生产（缺口）")
    _legend(fig, rect_m, ["消费", "生产", "缺口"], [_DEEP, _MINT2, _SAND],
            edges=["white", _MAIN, "white"])
    _main_supply(ax_main, years, cons, prod)
    _callout(ax_main, YEAR, float(cons[YEAR]) / 1e4,
             f"{YEAR} 年消费 {cons_y / 1e4:,.2f} 亿吨标准煤\n生产 {prod_y / 1e4:,.2f} 亿吨")

    rect_r = [_L + _SPAN * 0.612, 0.335, _SPAN * 0.388, 0.300]
    ax_r = _panel(fig, rect_r, "能源消费的品种结构",
                  "占能源消费总量，自下而上累加至 100%")
    _legend_strip(fig, rect_r, ["煤炭", "石油", "天然气", "一次电力及其他"],
                  [_DEEP, _MAIN, _MINT, _SAND], y_off=0.016, size=6.8)
    _lift(ax_r)
    _stack_100(ax_r, years, cons_struct, cons_struct.index.tolist(),
               [_DEEP, _MAIN, _MINT, _SAND])

    y_b, h_b = 0.048, 0.252
    w_b = (_SPAN - gap_px * 2) / 3

    rect_o = [_L, y_b, w_b, h_b]
    ax_o = _panel(fig, rect_o, "主要能源产品产量", "四格独立纵轴，均为实物量")
    ax_o.set_axis_off()
    _mini_areas(fig, [rect_o[0] + 0.014, y_b + 0.044, w_b - 0.030, h_b - 0.132],
                years, out_series, list(out_series.index),
                [_DEEP, _SAND, _MINT2, _MAIN],
                fmts=["{:.2f}", "{:.2f}", "{:,.0f}", "{:.2f}"])

    rect_p = [_L + w_b + gap_px, y_b, w_b, h_b]
    ax_p = _panel(fig, rect_p, f"{ind_year} 年能源消费前 8 个行业",
                  "万吨标准煤　门类与大类，不含工业/制造业合计")
    ax_p.set_position([rect_p[0] + 0.112, rect_p[1] + 0.034,
                       rect_p[2] - 0.124, rect_p[3] - 0.128])
    _rank_dots(ax_p, ind_series, "")

    rect_s = [_L + 2 * (w_b + gap_px), y_b, w_b, h_b]
    ax_s = _panel(fig, rect_s, "一次能源生产的品种结构", "占一次能源生产总量")
    _legend_strip(fig, rect_s, ["原煤", "原油", "天然气", "一次电力及其他"],
                  [_DEEP, _MAIN, _MINT, _SAND], y_off=0.016, size=6.8)
    _lift(ax_s)
    _stack_100(ax_s, years, prod_struct, prod_struct.index.tolist(),
               [_DEEP, _MAIN, _MINT, _SAND])

    fig.text(
        _L, 0.016,
        "数据来源：国家统计局 · 年度数据（能源）。消费与生产统一折算为万吨标准煤，"
        "原煤、原油、天然气、发电量为实物量。供需缺口＝消费−生产，含净进口与库存变动，"
        "不等于净进口。分行业明细发布到 2023 年，2024 年起只更新门类；"
        "汽油、柴油产量发布到 2024 年。全部数值未做插补。",
        fontsize=6.8, color=_FAINT, va="bottom",
    )

    out = root / "能源数据驾驶舱.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("已生成：" + str(build()))
