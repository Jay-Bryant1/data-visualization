"""价格指数 · 数据驾驶舱（一页式展板）。

版式与「财政/dashboard.py」同一套：顶部标题 + 年份滑轨 + KPI 卡条 + 速读条 +
主副图 + 底部三联，配色同源，几张驾驶舱并排看是一套东西。

图表约束（项目约定，优先级高于参考图）：
- 不用柱形、条形、折线及其变体。指数用「100 基准偏离带」：带子落在 100 以上就是涨价，
  以下就是跌价，不必先在脑子里做减法；排名用克利夫兰点图；剪刀差用真时间轴圆点。
- 环比指数（上年=100）一律围绕 100 画，不从 0 起——从 0 起会把 97–103 压成一条直线。
- 不插补、不连断档：商品零售价格指数 2022 年后停发、固定资产投资价格指数 2019 年后停发，
  对应位置留空并在脚注写明。

用法（任意目录执行均可）：
    python 统计局代码/价格指数/dashboard.py
输出：
    年度数据/价格指数/价格指数驾驶舱.png
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

from stj_core import data_root, parse_table  # noqa: E402

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
        "savefig.facecolor": "white",
        "axes.edgecolor": "#D8DEDE",
        "axes.linewidth": 0.9,
        "figure.dpi": 110,
    })


def _rounded(fig: plt.Figure, rect: list[float], *, face: str, edge: str,
             radius: float = 0.008) -> None:
    """画一张圆角卡片。"""
    x, y, w, h = rect
    fig.add_artist(FancyBboxPatch(
        (x, y), w, h, transform=fig.transFigure,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=0.9, zorder=0,
    ))


def _panel(fig: plt.Figure, rect: list[float], title: str,
           sub: str = "", tag: str = "") -> plt.Axes:
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

    fig.text(_L + 0.040, 0.958, "价格指数 · 数据驾驶舱", fontsize=18,
             color=_INK, fontweight="bold", va="center")
    fig.text(_L + 0.040, 0.930,
             "2016 – 2025 年 · 居民消费价格 / 工业生产者价格 / 商品零售 / 农产品生产价格 · 国家统计局年度数据",
             fontsize=9.2, color=_MUT, va="center")

    _rounded(fig, [0.858, 0.928, 0.052, 0.040], face=_DEEP, edge=_DEEP, radius=0.008)
    fig.text(0.884, 0.948, f"{YEAR} 年", fontsize=8.8, color="white",
             ha="center", va="center", fontweight="bold")
    _rounded(fig, [0.916, 0.928, 0.060, 0.040], face="white", edge="#D8DEDE", radius=0.008)
    fig.text(0.946, 0.948, "年度", fontsize=8.8, color=_MUT, ha="center", va="center")


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
    """一张 KPI 卡：左上标签、右上口径标签、数值 + 单位、底行同比与补充口径。"""
    _rounded(fig, rect, face=_CARD, edge=_LINE, radius=0.010)
    x0, y0, w, h = rect
    fig.text(x0 + 0.012, y0 + h - 0.016, str(item["label"]), fontsize=9.0,
             color=_MUT, va="top")
    fig.text(x0 + w - 0.012, y0 + h - 0.016, str(item["tag"]), fontsize=7.6,
             color=_FAINT, va="top", ha="right")

    fig.text(x0 + 0.012, y0 + 0.030, str(item["value"]), fontsize=23,
             color=_INK, fontweight="bold", va="bottom")
    fig.text(x0 + 0.012 + float(item["value_w"]), y0 + 0.033, str(item["unit"]),
             fontsize=8.4, color=_MUT, va="bottom")

    fig.text(x0 + 0.012, y0 + 0.011, str(item.get("delta_label", "涨跌")),
             fontsize=8.0, color=_FAINT, va="bottom")
    fig.text(x0 + 0.038, y0 + 0.011, str(item["delta"]), fontsize=8.6, va="bottom",
             color=_MAIN if float(item["delta_v"]) >= 0 else _WARN, fontweight="bold")
    fig.text(x0 + w - 0.012, y0 + 0.011, str(item["base"]), fontsize=8.0,
             va="bottom", ha="right", color=_FAINT)


def _legend(fig: plt.Figure, rect: list[float], labels: list[str],
            colors: list[str], edges: list[str] | None = None) -> None:
    """卡片标题行右端的图例：色点用填充色，浅色点换深一档描边。"""
    x = rect[0] + rect[2] - 0.012
    y = rect[1] + rect[3] - 0.026
    edge_list = list(edges) if edges else ["white"] * len(labels)
    for name, color, edge in zip(reversed(labels), reversed(colors), reversed(edge_list)):
        fig.text(x, y, name, fontsize=7.4, color=_MUT, ha="right", va="center")
        cx = x - 0.0028 - len(name) * 0.0072
        fig.add_artist(Circle((cx, y), 0.0030, transform=fig.transFigure,
                              facecolor=color, edgecolor=edge, linewidth=0.8, zorder=3))
        x = cx - 0.0115


def _callout(ax: plt.Axes, x: float, y: float, text: str, dx: float = -6,
             dy: float = 26) -> None:
    """在图内落一个小的白底提示框，用引线连到数据点。"""
    ax.annotate(
        text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        ha="right", va="bottom", fontsize=8.0, color=_INK, linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.42", facecolor="white",
                  edgecolor="#DCE4E4", linewidth=1.0),
        arrowprops=dict(arrowstyle="-", color="#C6D2D2", linewidth=0.9,
                        shrinkA=2, shrinkB=6),
        annotation_clip=False, zorder=8,
    )


def _brief(fig: plt.Figure, lead: str, body: str) -> None:
    """速读条：左侧竖条 + 加粗年份 + 正文。"""
    rect = [_L, 0.663, _SPAN, 0.056]
    _rounded(fig, rect, face="#F4F8F8", edge="#E4ECEC", radius=0.007)
    fig.add_artist(Rectangle((rect[0] + 0.004, rect[1] + 0.010), 0.0035, rect[3] - 0.020,
                             transform=fig.transFigure, facecolor=_DEEP, edgecolor="none"))
    fig.text(rect[0] + 0.016, rect[1] + rect[3] / 2, lead, fontsize=8.6,
             color=_DEEP, fontweight="bold", va="center")
    fig.text(rect[0] + 0.086, rect[1] + rect[3] / 2, body, fontsize=8.4,
             color=_MUT, va="center")


def _index_band(ax: plt.Axes, years: list[int], series: pd.Series,
                color: str, alpha: float = 0.55) -> None:
    """画一条「相对 100 的偏离带」：高出 100 的部分向上填充，低于 100 的向下填充。

    环比指数只有一个有意义的参照点——100（与上年持平）。用两段填充把「涨/跌」
    直接编成方向，比在 95–110 的折线上找零刻度省事得多。
    """
    y = series.reindex(years).astype(float)
    ax.fill_between(years, 100, y.where(y >= 100, 100), color=color, alpha=alpha,
                    linewidth=0, zorder=2, interpolate=False)
    ax.fill_between(years, 100, y.where(y <= 100, 100), color=_SAND, alpha=0.55,
                    linewidth=0, zorder=2, interpolate=False)
    ax.plot(years, y, color=color, linewidth=1.5, zorder=3)
    ax.plot(years, y, "o", markersize=3.6, color=color, markeredgecolor="white",
            markeredgewidth=0.7, zorder=4)


def _main_index(ax: plt.Axes, years: list[int], cpi: pd.Series,
                ppi: pd.Series, buy: pd.Series) -> None:
    """主图：CPI / PPI / 购进价格指数三条偏离带，100 以下为下跌。"""
    for s, c, a in ((buy, "#9FB8B3", 0.30), (ppi, _SAND, 0.50), (cpi, _MAIN, 0.55)):
        y = s.reindex(years).astype(float)
        ax.fill_between(years, 100, y, color=c, alpha=a, linewidth=0, zorder=2,
                        interpolate=False)
        ax.plot(years, y, color=c if c != _SAND else _WARN, linewidth=1.4, zorder=3)
        ax.plot(years, y, "o", markersize=3.4, color=c if c != _SAND else _WARN,
                markeredgecolor="white", markeredgewidth=0.7, zorder=4)
    ax.axhline(100, color="#9AA6A6", linewidth=1.0, linestyle=(0, (4, 3)), zorder=5)

    vals = pd.concat([cpi, ppi, buy]).dropna()
    lo, hi = min(float(vals.min()), 100.0), max(float(vals.max()), 100.0)
    pad = max((hi - lo) * 0.16, 1.0)
    ax.set_ylim(lo - pad, hi + pad * 1.6)
    # 右端多留一截：末点上的「CPI 100.0」「PPI 97.4」写在点右侧，
    # 只留 0.4 会把标注顶出坐标区和卡片边界
    ax.set_xlim(years[0] - 0.4, years[-1] + 1.1)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _grid(ax, hi + pad)
    _clean(ax, x_ticks=years, step=2)
    ax.tick_params(axis="x", pad=4)


def _rank_dots(ax: plt.Axes, series: pd.Series, unit: str,
               fmt: str = "{:,.1f}") -> None:
    """卡片内的一列排名：一行一个类别，点落在量级上，数值贴点右侧。"""
    names = list(series.index)[::-1]
    values = series.to_numpy(dtype=float)[::-1]
    y = np.arange(len(names), dtype=float)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.6, color=_INK)
    lo, hi = float(values.min()), float(values.max())
    span = max(hi - lo, 1e-6)
    ax.set_xlim(min(lo - span * 0.42, 90), hi + span * 0.42)
    ax.set_ylim(-0.75, len(names) - 0.25)
    ax.axvline(100, color="#C9D3D3", linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
    for v, yy in zip(values, y):
        color = _MAIN if v >= 100 else _WARN
        ax.plot([100, v], [yy, yy], color="#DCDCDC", linewidth=1.1, zorder=1)
        ax.plot([v], [yy], marker="o", markersize=6.4, color=color, zorder=2)
        ax.annotate(fmt.format(v), xy=(v, yy), xytext=(6, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=7.4, color=_INK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#D8DEDE")
    ax.tick_params(axis="x", labelsize=7.0, colors=_MUT, length=2, pad=2)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#EDF1F1", linewidth=0.7)
    ax.set_axisbelow(True)


def _city_rural(ax: plt.Axes, years: list[int], city: pd.Series,
                rural: pd.Series) -> None:
    """城市与农村居民消费价格：两条偏离带叠在同一根 100 基准上。"""
    # 两条同色系面积叠在一起会糊成一片：城市铺面积，农村只画线与点，
    # 这样「农村低于城市」这件事在每一年都能直接看出来
    cy = city.reindex(years).astype(float)
    ax.fill_between(years, 100, cy, color=_MINT2, alpha=0.55, linewidth=0, zorder=2,
                    interpolate=False)
    ax.plot(years, cy, color=_MAIN, linewidth=1.3, zorder=3)
    ax.plot(years, cy, "o", markersize=3.2, color=_MAIN,
            markeredgecolor="white", markeredgewidth=0.7, zorder=4)
    ry = rural.reindex(years).astype(float)
    ax.plot(years, ry, color=_WARN, linewidth=1.5, zorder=5)
    ax.plot(years, ry, "o", markersize=3.6, color=_SAND,
            markeredgecolor="white", markeredgewidth=0.8, zorder=6)
    ax.axhline(100, color="#9AA6A6", linewidth=1.0, linestyle=(0, (4, 3)), zorder=5)
    vals = pd.concat([city, rural]).dropna()
    lo, hi = min(float(vals.min()), 100.0), max(float(vals.max()), 100.0)
    pad = max((hi - lo) * 0.22, 0.8)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(years[0] - 0.4, years[-1] + 0.4)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _grid(ax, hi + pad)
    _clean(ax, x_ticks=years, step=3)


def _scissors(ax: plt.Axes, years: list[int], cpi: pd.Series,
              ppi: pd.Series) -> None:
    """CPI 与 PPI 的剪刀差（PPI 涨幅减 CPI 涨幅，百分点），零线为分界。"""
    gap = (ppi.reindex(years) - 100) - (cpi.reindex(years) - 100)
    ax.axhline(0, color="#9AA6A6", linewidth=1.0, zorder=2)
    pos = gap.where(gap >= 0)
    neg = gap.where(gap <= 0)
    ax.fill_between(years, 0, pos, color=_MAIN, alpha=0.45, linewidth=0, zorder=2,
                    interpolate=False)
    ax.fill_between(years, 0, neg, color=_SAND, alpha=0.55, linewidth=0, zorder=2,
                    interpolate=False)
    ax.plot(years, gap, color=_DEEP, linewidth=1.4, zorder=3)
    ax.plot(years, gap, "o", markersize=3.6, color=_DEEP,
            markeredgecolor="white", markeredgewidth=0.8, zorder=4)
    lo, hi = float(gap.min()), float(gap.max())
    span = max(hi - lo, 1e-6)
    ax.set_ylim(lo - span * 0.28, hi + span * 0.34)
    ax.set_xlim(years[0] - 0.4, years[-1] + 0.4)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+,.0f}"))
    _clean(ax, x_ticks=years, step=3)
    for t in ax.get_yticks():
        if t != 0:
            ax.axhline(t, color="#EDF1F1", linewidth=0.7, zorder=0)


def build() -> Path:
    """拼装整张展板并保存，返回输出路径。"""
    _setup()
    root = data_root(Path(__file__).resolve().parent) / "价格指数"

    allp = parse_table(root / "01-各种价格指数（上年=100）.csv")
    years = [y for y in allp.years if allp.wide[y].notna().any()]
    w = allp.wide
    cpi = w.loc["居民消费价格指数"]
    city = w.loc["城市居民消费价格指数"]
    rural = w.loc["农村居民消费价格指数"]
    ppi = w.loc["工业生产者出厂价格指数"]
    buy = w.loc["工业生产者购进价格指数"]

    # 八大类 CPI：2021–2025 这一段按新分类发布，八张分项表各取首行（大类合计）
    big_names = ["食品烟酒", "衣着", "居住", "生活用品及服务", "交通通信",
                 "教育文化娱乐", "医疗保健", "其他用品及服务"]
    big = {}
    # 必须遍历全部 CSV：八大类里「其他用品及服务」是 10- 开头，
    # 用 "0*.csv" 会把它漏掉，涨幅最高的大类就会错成衣着
    for f in sorted(root.glob("*.csv")):
        stem = f.stem
        if "居民消费价格分类指数（2021-2025）" not in stem or "城市" in stem or "农村" in stem:
            continue
        t = parse_table(f)
        head = str(t.wide.index[0]).replace("类居民消费价格指数", "")
        s = t.wide.iloc[0].dropna()
        if head in big_names and len(s):
            big[head] = float(s.iloc[-1])
    big_series = pd.Series(big).sort_values()

    # PPI 分行业：2025 年涨跌幅的两极各取 6 个
    ppi_t = parse_table(root / "75-工业生产者出厂价格指数（2016-2025）.csv")
    ind = ppi_t.wide[YEAR].dropna()
    ind = ind.drop(index="工业生产者出厂价格指数", errors="ignore")
    # 行业名压到 10 字以内：底部卡片的标签区只有 0.09 个图宽，
    # 原始名字（「文教、工美、体育和娱乐用品制造业」）会顶到画布左沿外
    _SHORT = {
        "煤炭开采和洗选业": "煤炭开采",
        "石油和天然气开采业": "油气开采",
        "石油加工、炼焦和核燃料加工业": "石油加工炼焦",
        "黑色金属冶炼和压延加工业": "黑色金属冶炼",
        "有色金属冶炼和压延加工业": "有色金属冶炼",
        "黑色金属矿采选业": "黑色金属矿采",
        "有色金属矿采选业": "有色金属矿采",
        "化学原料和化学制品制造业": "化学原料制品",
        "化学纤维制造业": "化学纤维",
        "非金属矿物制品业": "非金属矿物",
        "金属制品、机械和设备修理业": "金属制品修理",
        "文教、工美、体育和娱乐用品制造业": "文教工美体育",
        "水的生产和供应业": "水的生产供应",
        "电力、热力生产和供应业": "电力热力",
        "燃气生产和供应业": "燃气生产",
        "医药制造业": "医药制造",
        "汽车制造业": "汽车制造",
        "农副食品加工业": "农副食品",
        "食品制造业": "食品制造",
        "造纸和纸制品业": "造纸纸制品",
        "印刷和记录媒介复制业": "印刷复制",
        "通用设备制造业": "通用设备",
        "专用设备制造业": "专用设备",
        "纺织业": "纺织",
        "橡胶和塑料制品业": "橡胶塑料",
        "家具制造业": "家具制造",
        "仪器仪表制造业": "仪器仪表",
        "其他制造业": "其他制造",
        "废弃资源综合利用业": "废弃资源",
        "金属制品业": "金属制品",
        "烟草制品业": "烟草制品",
        "木材加工和木、竹、藤、棕、草制品业": "木材加工",
        "皮革、毛皮、羽毛及其制品和制鞋业": "皮革制鞋",
        "铁路、船舶、航空航天和其他运输设备制造业": "铁路船舶航空航天",
        "计算机、通信和其他电子设备制造业": "电子设备",
        "电气机械和器材制造业": "电气机械",
        "酒、饮料和精制茶制造业": "酒饮料茶",
        "纺织服装、服饰业": "纺织服装",
        "开采辅助活动": "开采辅助",
    }
    short = {
        n: _SHORT.get(n.replace("工业生产者出厂价格指数", ""),
                      n.replace("工业生产者出厂价格指数", ""))
        for n in ind.index
    }
    ind_s = ind.rename(index=short).sort_values()
    both = pd.concat([ind_s.head(4), ind_s.tail(4)])

    cpi_y = float(cpi[YEAR])
    cpi_prev = float(cpi[YEAR - 1])
    ppi_y = float(ppi[YEAR])
    ppi_prev = float(ppi[YEAR - 1])
    rural_y = float(rural[YEAR])
    city_y = float(city[YEAR])
    # 连续下跌年数：从末年往前数，指数低于 100 的年份
    decline = 0
    for y in reversed(years):
        if pd.notna(ppi[y]) and ppi[y] < 100:
            decline += 1
        else:
            break
    rural_neg = [y for y in years if pd.notna(rural[y]) and rural[y] < 100]
    top_big = max(big.items(), key=lambda kv: kv[1])
    low_big = min(big.items(), key=lambda kv: kv[1])
    low_ind_name = str(ind.idxmin())
    low_ind_val = float(ind.min())
    low_ind_short = short[low_ind_name]
    coal_peak = ppi_t.wide.loc[low_ind_name].dropna().max()

    fig = plt.figure(figsize=(_FIG_W, _FIG_H))
    _header(fig)
    _year_rail(fig, years, YEAR)

    cards = [
        {"label": "居民消费价格指数", "tag": f"{YEAR} 年",
         "value": f"{cpi_y:.1f}", "value_w": 0.070, "unit": "上年=100",
         "delta": f"{(cpi_y - 100):+.1f}%", "delta_v": cpi_y - 100,
         "delta_label": "涨跌", "base": f"较上年 {cpi_y - cpi_prev:+.1f} 点"},
        {"label": "工业生产者出厂价格", "tag": "PPI",
         "value": f"{ppi_y:.1f}", "value_w": 0.064, "unit": "上年=100",
         "delta": f"{(ppi_y - 100):+.1f}%", "delta_v": ppi_y - 100,
         "delta_label": "涨跌", "base": f"连续 {decline} 年下降"},
        {"label": "农村居民消费价格", "tag": "城乡分化",
         "value": f"{rural_y:.1f}", "value_w": 0.064, "unit": "上年=100",
         "delta": f"{(rural_y - 100):+.1f}%", "delta_v": rural_y - 100,
         "delta_label": "涨跌",
         "base": (f"{rural_neg[0]} 年以来首次转负" if rural_neg else "仍为正增长")},
        {"label": f"{YEAR} 年涨幅最高的大类", "tag": top_big[0],
         "value": f"{top_big[1]:.1f}", "value_w": 0.064, "unit": "上年=100",
         "delta": f"{(top_big[1] - 100):+.1f}%", "delta_v": top_big[1] - 100,
         "delta_label": "涨跌", "base": "金银珠宝等拉动"},
        {"label": f"{YEAR} 年跌幅最大的行业", "tag": low_ind_short,
         "value": f"{low_ind_val:.1f}", "value_w": 0.064, "unit": "上年=100",
         "delta": f"{(low_ind_val - 100):+.1f}%", "delta_v": low_ind_val - 100,
         "delta_label": "涨跌", "base": f"{int(ppi_t.wide.loc[low_ind_name].idxmax())} 年峰值 {coal_peak:.0f}"},
    ]
    gap_px = 0.011
    card_w = (_SPAN - gap_px * (len(cards) - 1)) / len(cards)
    for i, item in enumerate(cards):
        _kpi(fig, [_L + i * (card_w + gap_px), 0.742, card_w, 0.102], item)

    _brief(
        fig,
        f"{YEAR} 速读",
        f"居民消费价格 {cpi_y:.1f}，即比上年 {(cpi_y - 100):+.1f}%，涨幅连续两年收窄至零；"
        f"工业生产者出厂价格 {ppi_y:.1f}（{(ppi_y - 100):+.1f}%），已连续 {decline} 年低于 100，"
        f"购进价格 {float(buy[YEAR]):.1f}。八大类里{top_big[0]}涨 {(top_big[1] - 100):+.1f}%、"
        f"{low_big[0]} {(low_big[1] - 100):+.1f}%；农村 CPI {rural_y:.1f} 低于城市 {city_y:.1f}，"
        f"城乡涨幅差 {rural_y - city_y:+.1f} 点。",
    )

    rect_m = [_L, 0.335, _SPAN * 0.600, 0.300]
    ax_main = _panel(fig, rect_m, "消费与工业价格的两条线",
                     "指数＝上年=100　虚线 100 为与上年持平，线下为下跌　阴影为三条指数的偏离")
    _legend(fig, rect_m, ["居民消费", "工业出厂", "工业购进"],
            [_MAIN, _SAND, "#9FB8B3"], edges=[_DEEP, _WARN, _MUT])
    _main_index(ax_main, years, cpi, ppi, buy)
    # 直接在末点旁标数值：2017 年当锚点会把引线框顶出卡片左沿
    ax_main.annotate(f"PPI {ppi_y:.1f}", xy=(YEAR, ppi_y), xytext=(8, -14),
                     textcoords="offset points", fontsize=8.2, color=_WARN,
                     fontweight="bold", ha="left", va="center")
    ax_main.annotate(f"CPI {cpi_y:.1f}", xy=(YEAR, cpi_y), xytext=(8, 8),
                     textcoords="offset points", fontsize=8.2, color=_DEEP,
                     fontweight="bold", ha="left", va="center")
    # 写在右下角空白带里：放在 2021 峰值下方会被沙色面积压住，灰字读不出来
    ax_main.text(0.015, 0.955, "三条线都贴着 100：上游通缩没有传导到消费端",
                 transform=ax_main.transAxes, fontsize=7.8, color=_MUT,
                 ha="left", va="top")

    rect_r = [_L + _SPAN * 0.612, 0.335, _SPAN * 0.388, 0.300]
    ax_r = _panel(fig, rect_r, f"{YEAR} 年八大类的涨跌幅",
                  "居民消费价格分类指数（上年=100），虚线 100 为持平")
    # 八个大类的名字最长 7 字，坐标区不右移会把「其他用品及服务」顶到卡片外
    ax_r.set_position([rect_r[0] + 0.076, rect_r[1] + 0.034,
                       rect_r[2] - 0.090, rect_r[3] - 0.104])
    _rank_dots(ax_r, big_series, "")

    y_b, h_b = 0.048, 0.252
    w_b = (_SPAN - gap_px * 2) / 3

    rect_a = [_L, y_b, w_b, h_b]
    ax_a = _panel(fig, rect_a, f"{YEAR} 年工业行业价格的两极",
                  "出厂价格指数，下为跌幅前四、上为涨幅前四")
    ax_a.set_position([rect_a[0] + 0.088, rect_a[1] + 0.034,
                       rect_a[2] - 0.100, rect_a[3] - 0.104])
    _rank_dots(ax_a, both, "")

    rect_b = [_L + w_b + gap_px, y_b, w_b, h_b]
    ax_b = _panel(fig, rect_b, "城市与农村的居民消费价格", "指数＝上年=100，虚线为持平")
    _legend(fig, rect_b, ["城市", "农村"], [_MINT2, _SAND], edges=[_MAIN, _WARN])
    _city_rural(ax_b, years, city, rural)

    rect_c = [_L + 2 * (w_b + gap_px), y_b, w_b, h_b]
    ax_c = _panel(fig, rect_c, "PPI 与 CPI 的剪刀差", "工业出厂涨幅减居民消费涨幅，百分点")
    _scissors(ax_c, years, cpi, ppi)

    fig.text(
        _L, 0.016,
        "数据来源：国家统计局 · 年度数据（价格指数）。全部为「上年=100」的环比指数，"
        "数值减 100 即当年涨跌幅。商品零售价格分类指数止于 2022 年、固定资产投资价格指数止于 2019 年，"
        "2023 年起不再发布，图中留空未插补；2025 年八大类按 2021 年启用的新分类口径。"
        "八大类之间不可加总（权重未公布），排名只比方向与幅度。",
        fontsize=6.8, color=_FAINT, va="bottom",
    )

    out = root / "价格指数驾驶舱.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("已生成：" + str(build()))
