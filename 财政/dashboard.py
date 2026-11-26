"""财政 · 数据驾驶舱（一页式展板）。

版式参照驾驶舱类看板：顶部标题 + 年份滑轨 + KPI 卡条 + 速读条 + 主副图 + 底部三联。
配色统一为深青 / 薄荷 / 沙色，逐表图与展板共用一套色板。

图表约束（项目约定，优先级高于参考图）：
- 不用柱形、条形、折线及其变体。收支规模用面积图，缺口用真时间轴圆点，
  结构用 100% 堆叠面积，排名用克利夫兰点图。
- 面积图一律零基线；含负值的数列（收支缺口）单独挂右轴并画零线。
- 不插补、不连断档：只在真实观测年份落点。

用法（任意目录执行均可）：
    python 统计局代码/财政/dashboard.py
输出：
    年度数据/财政/财政数据驾驶舱.png
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
        # 末年必须出现在刻度上，但隔年取点排到倒数第二年时直接顶掉更干净
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

    fig.text(_L + 0.040, 0.958, "财政 · 数据驾驶舱", fontsize=18,
             color=_INK, fontweight="bold", va="center")
    fig.text(_L + 0.040, 0.930, "2016 – 2025 年 · 一般公共预算收支 / 税收结构 / 支出功能 / 政府债务 · 国家统计局年度数据",
             fontsize=9.2, color=_MUT, va="center")

    _rounded(fig, [0.858, 0.928, 0.052, 0.040], face=_DEEP, edge=_DEEP, radius=0.008)
    fig.text(0.884, 0.948, f"{YEAR} 年", fontsize=8.8, color="white",
             ha="center", va="center", fontweight="bold")
    _rounded(fig, [0.916, 0.928, 0.060, 0.040], face="white", edge="#D8DEDE", radius=0.008)
    fig.text(0.946, 0.948, "决算", fontsize=8.8, color=_MUT, ha="center", va="center")


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

    # 收支缺口这类「同比」没意义的口径，标签换成「扩大 / 收窄」之外的通用说法
    fig.text(x0 + 0.012, y0 + 0.011, str(item.get("delta_label", "同比")),
             fontsize=8.0, color=_FAINT, va="bottom")
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


def _main_budget(ax: plt.Axes, years: list[int], rev: pd.Series,
                 exp: pd.Series, gap: pd.Series) -> None:
    """主图：支出 / 收入两层面积（万亿元）+ 收支缺口圆点（右轴）。

    支出面积垫底、收入面积压在上面，两层之间的缝就是当年缺口，与右轴圆点互为校验。
    """
    e = exp.reindex(years) / 1e4
    r = rev.reindex(years) / 1e4
    ax.fill_between(years, 0, e, color=_MINT, linewidth=0, zorder=1)
    ax.fill_between(years, 0, r, color=_MINT2, alpha=0.85, linewidth=0, zorder=2)
    ax.plot(years, e, color=_DEEP, linewidth=1.6, zorder=3)
    ax.plot(years, r, color=_MAIN, linewidth=1.2, zorder=3)
    top = float(e.max())
    ax.set_ylim(0, top * 1.26)
    ax.set_xlim(years[0] - 0.4, years[-1] + 0.4)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _grid(ax, top * 1.26)
    _clean(ax, x_ticks=years, step=2, label_size=7.2)
    ax.tick_params(axis="x", pad=4)

    ax2 = ax.twinx()
    g = gap.reindex(years) / 1e4
    ax2.axhline(0, color="#D8DEDE", linewidth=0.9, zorder=1)
    ax2.plot(years, g, marker="o", markersize=4.8, color=_SAND,
             markeredgecolor="white", markeredgewidth=1.1, linestyle="none", zorder=5)
    ax2.set_ylim(float(np.nanmin(g)) * 1.35, 0.0)
    ax2.yaxis.set_major_locator(MaxNLocator(3))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax2.tick_params(axis="y", labelsize=7.0, colors=_SAND, length=0, pad=2)
    for side in ("top", "right", "left", "bottom"):
        ax2.spines[side].set_visible(False)


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
        # 名称写在坐标区左侧留白里：x 用轴比例，负的数据坐标会被 xlim 裁掉
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
    # 份额小的层只有几个百分点，标注必须撑开间距，顶部顶到 99% 时整体下移；
    # 间距随坐标区高度走——矮卡片上 6.8pp 只折合 6pt，8.4pt 的字会叠在一起
    mids = [(t + b) / 2 for t, b in zip(cum, bottoms)]
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
        # 标注落在色带右侧空白里，一律深色；白字曾写在白底上整行消失
        ax.text(years[-1] + 0.35, mid, f"{top_v - bottom_v:.1f}%", fontsize=8.4,
                color=_DEEP, va="center", ha="left", fontweight="bold")


def _legend_strip(fig: plt.Figure, rect: list[float], labels: list[str],
                  colors: list[str], *, y_off: float = 0.020,
                  size: float = 6.9) -> None:
    """卡片底部的横排图例。

    七层堆叠塞不进标题行：标题占掉一半宽度，图例一挤就压到标题上。
    改在图下方铺一条等距图例，右端百分比标注不再需要点色对照。
    """
    y = rect[1] + y_off
    step = rect[2] / len(labels)
    for i, (name, color) in enumerate(zip(labels, colors)):
        x = rect[0] + 0.010 + i * step
        fig.add_artist(Circle((x + 0.0030, y), 0.0028, transform=fig.transFigure,
                              facecolor=color, edgecolor="white", linewidth=0.6, zorder=3))
        fig.text(x + 0.0086, y, name, fontsize=size, color=_MUT, va="center")


def _lift(ax: plt.Axes, pad: float = 0.034) -> None:
    """把绘图区整体上提，给卡片底部的图例带让位。

    图例铺在卡片下沿时，年份刻度数字仍挂在坐标区下方，两者会直接叠在一起；
    上提的同时压缩高度（底边抬高、顶边不动），不会挤压标题。
    """
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 + pad, pos.width, pos.height - pad])


def _short_func(name: str) -> str:
    """把「国家财政社会保障和就业支出」压成「社会保障和就业」。"""
    for cut in ("国家财政", "支出"):
        name = name.replace(cut, "")
    return name


def build() -> Path:
    """组装财政驾驶舱并保存 PNG。"""
    _setup()
    root = data_root(Path(__file__).resolve().parent.parent) / "财政"

    def load(name: str) -> StjTable:
        return parse_table(root / f"{name}.csv")

    budget_t = load("01-一般公共预算收支及增长速度")
    level_t = load("02-全国中央地方一般公共预算收入")
    fund_t = load("07-国家财政支出（按功能）")
    tax_t = load("10-各项税收收入")
    debt_t = load("11-中央财政债务余额")

    years = budget_t.years
    rev = budget_t.wide.loc["一般公共预算收入"]
    exp = budget_t.wide.loc["一般公共预算支出"]
    gap = exp - rev

    rev_y, rev_prev = float(rev[YEAR]), float(rev[YEAR - 1])
    exp_y, exp_prev = float(exp[YEAR]), float(exp[YEAR - 1])
    gap_y, gap_prev = float(gap[YEAR]), float(gap[YEAR - 1])

    tax = tax_t.wide.loc["各项税收"].dropna()
    tax_y = float(tax.iloc[-1])
    tax_prev = float(tax.iloc[-2])

    debt = debt_t.wide.loc["中央财政债务余额"]
    debt_y, debt_prev = float(debt[YEAR]), float(debt[YEAR - 1])

    central_y = float(level_t.wide.loc["中央一般公共预算收入"][YEAR])
    local_y = float(level_t.wide.loc["地方一般公共预算收入"][YEAR])

    # 支出功能明细 2025 年缺 4 项（外交、公共安全、文体、其他），结构图与排名取 2024 年
    func_year = 2024
    func_rows = [n for n in fund_t.wide.index if n != "全国一般公共预算支出"]
    func_series = fund_t.wide.loc[func_rows, func_year].dropna()
    func_series.index = [_short_func(n) for n in func_series.index]
    func_series = func_series.sort_values(ascending=False)

    # 支出结构：取常年占比最高的 6 项，其余按当年总额轧差成「其余各项」
    top6 = ["教育", "社会保障和就业", "农林水事务", "城乡社区事务", "一般公共服务", "医疗卫生"]
    raw6 = ["国家财政教育支出", "国家财政社会保障和就业支出", "国家财政农林水事务支出",
            "国家财政城乡社区事务支出", "国家财政一般公共服务支出", "国家财政医疗卫生支出"]
    total_exp = fund_t.wide.loc["全国一般公共预算支出"]
    struct_years = [y for y in fund_t.years if fund_t.wide.loc[raw6, y].notna().all()]
    struct = pd.DataFrame(index=top6 + ["其余各项"], columns=struct_years, dtype=float)
    for y in struct_years:
        picked = [float(fund_t.wide.at[raw, y]) for raw in raw6]
        for name, value in zip(top6, picked):
            struct.at[name, y] = value
        struct.at["其余各项", y] = float(total_exp[y]) - sum(picked)
    struct = struct.div(struct.sum(axis=0), axis=1) * 100

    # 中央 / 地方收入份额
    level = pd.DataFrame({
        "中央": level_t.wide.loc["中央一般公共预算收入"],
        "地方": level_t.wide.loc["地方一般公共预算收入"],
    }).T
    level_share = level.div(level.sum(axis=0), axis=1) * 100

    # 四税占税收收入的比重：分母用「各项税收」，扣掉营业税停征带来的口径跳变
    tax_share = pd.DataFrame({
        "增值税": tax_t.wide.loc["国内增值税"],
        "企业所得税": tax_t.wide.loc["企业所得税"],
        "个人所得税": tax_t.wide.loc["个人所得税"],
        "消费税": tax_t.wide.loc["国内消费税"],
    }).T
    tax_share = tax_share.div(tax_t.wide.loc["各项税收"], axis=1) * 100
    # 四税只占税收八成，「其余税种」按当年税收总额轧差补上，堆叠才真的到 100%；
    # 少了这一层，视觉上会读成「四税合计就是全部税收」
    tax_share.loc["其他税种"] = 100 - tax_share.sum(axis=0)

    fig = plt.figure(figsize=(_FIG_W, _FIG_H), dpi=200)
    fig.patch.set_facecolor("white")

    _header(fig)
    _year_rail(fig, years, YEAR)

    cards = [
        {"label": "一般公共预算收入", "tag": "全国", "value": f"{rev_y / 1e4:,.2f}",
         "unit": "万亿元", "value_w": 0.062,
         "delta": f"{(rev_y / rev_prev - 1) * 100:+.1f}%",
         "delta_v": rev_y - rev_prev, "base": f"较上年 {rev_y - rev_prev:+,.0f} 亿元"},
        {"label": "一般公共预算支出", "tag": "全国", "value": f"{exp_y / 1e4:,.2f}",
         "unit": "万亿元", "value_w": 0.062,
         "delta": f"{(exp_y / exp_prev - 1) * 100:+.1f}%",
         "delta_v": exp_y - exp_prev, "base": f"较上年 {exp_y - exp_prev:+,.0f} 亿元"},
        {"label": "收支差额", "tag": "缺口", "value": f"{gap_y / 1e4:,.2f}",
         "unit": "万亿元", "value_w": 0.058,
         "delta": f"{gap_y - gap_prev:+,.0f} 亿元", "delta_v": gap_y - gap_prev,
         "delta_label": "较上年", "base": f"相当于支出 {abs(gap_y) / exp_y:.1%}"},
        {"label": "税收收入", "tag": "占收入比", "value": f"{tax_y / 1e4:,.2f}",
         "unit": "万亿元", "value_w": 0.062,
         "delta": f"{(tax_y / tax_prev - 1) * 100:+.1f}%",
         "delta_v": tax_y - tax_prev, "base": f"占一般公共预算收入 {tax_y / rev_y:.1%}"},
        {"label": "中央财政债务余额", "tag": "年末", "value": f"{debt_y / 1e4:,.2f}",
         "unit": "万亿元", "value_w": 0.062,
         "delta": f"{(debt_y / debt_prev - 1) * 100:+.1f}%",
         "delta_v": debt_y - debt_prev, "base": f"较上年 {debt_y - debt_prev:+,.0f} 亿元"},
    ]
    gap_px = 0.011
    card_w = (_SPAN - gap_px * (len(cards) - 1)) / len(cards)
    for i, item in enumerate(cards):
        _kpi(fig, [_L + i * (card_w + gap_px), 0.742, card_w, 0.102], item)

    _brief(
        fig,
        f"{YEAR} 速读",
        f"一般公共预算收入 {rev_y / 1e4:,.2f} 万亿元，比上年 {(rev_y / rev_prev - 1) * 100:+.1f}%；"
        f"支出 {exp_y / 1e4:,.2f} 万亿元（{(exp_y / exp_prev - 1) * 100:+.1f}%），"
        f"缺口 {abs(gap_y) / 1e4:,.2f} 万亿元、相当于支出的 {abs(gap_y) / exp_y:.1%}。"
        f"收入中央占 {central_y / rev_y:.1%}、地方占 {local_y / rev_y:.1%}；"
        f"税收 {tax_y / 1e4:,.2f} 万亿元，占收入 {tax_y / rev_y:.1%}。"
        f"中央财政债务余额 {debt_y / 1e4:,.2f} 万亿元，比上年 {(debt_y / debt_prev - 1) * 100:+.1f}%。",
    )

    rect_m = [_L, 0.335, _SPAN * 0.600, 0.300]
    ax_main = _panel(fig, rect_m, "一般公共预算收支与缺口",
                     "面积＝支出 / 收入（万亿元）　圆点＝收支缺口（右轴，负值为赤字）")
    _legend(fig, rect_m, ["支出", "收入", "缺口"], [_MINT, _MINT2, _SAND])
    _main_budget(ax_main, years, rev, exp, gap)
    _callout(ax_main, YEAR, float(exp[YEAR]) / 1e4,
             f"{YEAR} 年支出 {exp_y / 1e4:,.2f} 万亿元\n收入 {rev_y / 1e4:,.2f} 万亿元")

    rect_r = [_L + _SPAN * 0.612, 0.335, _SPAN * 0.388, 0.300]
    ax_r = _panel(fig, rect_r, "财政支出的功能结构",
                  f"占当年一般公共预算支出　明细至 {max(struct_years)} 年（2025 年 4 项功能未发布）")
    _legend_strip(fig, rect_r, ["教育", "社保就业", "农林水", "城乡社区", "公共服务", "卫生", "其余"],
                  [_DEEP, _MAIN, _MINT2, _MINT, _SAND, "#8FA9A0", "#D5DCDC"], y_off=0.016, size=6.6)
    _lift(ax_r)
    _stack_100(ax_r, struct_years, struct, struct.index.tolist(),
               [_DEEP, _MAIN, _MINT2, _MINT, _SAND, "#8FA9A0", "#D5DCDC"])

    y_b, h_b = 0.048, 0.252
    w_b = (_SPAN - gap_px * 2) / 3

    rect_t = [_L, y_b, w_b, h_b]
    ax_t = _panel(fig, rect_t, "四大税种占税收收入的比重", "四税合计约占税收八成，其余为关税、资源税、房产税等")
    _legend_strip(fig, rect_t, ["增值税", "企业所得税", "个人所得税", "消费税", "其他税种"],
                  [_DEEP, _MAIN, _MINT2, _SAND, "#D5DCDC"], y_off=0.016, size=6.4)
    _lift(ax_t)
    _stack_100(ax_t, years, tax_share, tax_share.index.tolist(),
               [_DEEP, _MAIN, _MINT2, _SAND, "#D5DCDC"])

    rect_p = [_L + w_b + gap_px, y_b, w_b, h_b]
    ax_p = _panel(fig, rect_p, f"{func_year} 年支出功能前 8 项", "亿元")
    ax_p.set_position([rect_p[0] + 0.070, rect_p[1] + 0.034,
                       rect_p[2] - 0.082, rect_p[3] - 0.122])
    _rank_dots(ax_p, func_series.head(8), "")

    rect_s = [_L + 2 * (w_b + gap_px), y_b, w_b, h_b]
    ax_s = _panel(fig, rect_s, "收入的中央与地方分成", "占全国一般公共预算收入")
    _legend(fig, rect_s, ["中央", "地方"], [_DEEP, _MINT])
    _stack_100(ax_s, years, level_share, ["中央", "地方"], [_DEEP, _MINT])

    fig.text(
        _L, 0.016,
        "数据来源：国家统计局 · 年度数据（财政）。一般公共预算收支为决算数，2025 年为预算执行数；"
        "支出功能明细 2025 年缺外交、公共安全、文体传媒、其他四项，结构图与排名取 2024 年。"
        "「其余各项」由当年支出总额减去前六项轧差得到，含国防、科技、交通、环保、债务付息等。"
        "营业税自 2016 年 5 月起停征，故四大税种不含营业税。全部数值未做插补。",
        fontsize=6.8, color=_FAINT, va="bottom",
    )

    out = root / "财政数据驾驶舱.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("已生成：" + str(build()))
