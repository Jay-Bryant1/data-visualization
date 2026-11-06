"""两种图版：时间序列小倍数面积图、单年构成克利夫兰点图。

选型依据（与既定图表规范一致）：
- 多指标多年份 → 面积图矩阵。每个指标单独一格、独立纵轴，既保留各自的量级，
  又能在同一画布横向比较增速；柱形/折线一律不用。
- 单一年份多类别 → 克利夫兰点图。类别超过 6 个时环形图已无法读数，
  点图按量级排序后最易比较，也不会像条形图那样用面积放大差异。
"""

from __future__ import annotations

from pathlib import Path
from textwrap import wrap as wrap_text

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

from stj_core import StjTable, change_pct, leaf_rows, segments

_TEXT = "#132A31"
_MUTED = "#5E7075"
_GRID = "#EDF1F1"
_ACCENT = "#2C7A7B"
_ACCENT_DOWN = "#9AA6A6"
_INDEX_DOWN = "#C97B3C"   # 环比指数低于 100 的部分（只表示「低于上年」，不套用红绿涨跌）
_GAP = "#F4F6F6"

# 面积填充统一用薄荷色、描边用深青：整块填满深色时的面积图在小倍数里像一堵墙，
# 浅填充 + 细描边既保留「累计量」的读法，也不至于压住格内标注。
_FILL = "#BFDCD6"
_FILL_2 = "#8FC0B6"
_EDGE = "#2C7A7B"
_PANEL_BG = "#F9FBFB"

# 分类色：仅在需要区分「合计 / 分项」等少量角色时使用
_ROLE_COLORS = ["#0F4C52", "#2C7A7B", "#8FC0B6", "#E4B363", "#5E8CA6", "#C97B3C", "#7B8FA1"]


def apply_style() -> None:
    """统一中文字体与全局视觉参数。"""
    mpl.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _common_suffix(names: list[str]) -> str:
    """求所有指标名的最长公共后缀（如「法人单位数」），用于精简小图标题。

    Args:
        names: 指标名列表。

    Returns:
        公共后缀；过短（<3 字）或会吞掉整个名称时返回空串。
    """
    if len(names) < 2:
        return ""
    suffix = names[0]
    for n in names[1:]:
        while not n.endswith(suffix):
            suffix = suffix[1:]
            if not suffix:
                return ""
    return suffix if len(suffix) >= 3 else ""


def _fit_left_margin(fig: plt.Figure, axes: np.ndarray, n: int) -> None:
    """按纵轴刻度文字的实际宽度回收左边距。

    刻度写「2万」时 0.045 的空隙绰绰有余，一旦单位自带万/亿、刻度变成「20,000」，
    最左侧的字符就会被画布边缘切掉，而左边距是按格子数事先算死的，只能事后量。
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xs = []
    for ax in list(axes.ravel())[:n]:
        for tick in ax.get_yticklabels():
            if tick.get_text():
                xs.append(tick.get_window_extent(renderer=renderer).x0 / fig.bbox.width)
    if not xs:
        return
    # x0 为负说明文字已经越出画布左缘，按溢出量把左边距补回来（只增不减）
    overflow = -min(xs)
    if overflow > 0:
        fig.subplots_adjust(left=min(0.32, fig.subplotpars.left + overflow + 0.01))


def _footnote_lines(text: str, fig_w: float) -> list[str]:
    """按画布宽度把脚注折成若干行。

    尾注长短随表而变（民族自治地方那几条尤其长），固定单行会在窄图上被右边缘切掉，
    切掉的正是口径说明，最不该丢。
    """
    return wrap_text(text, width=max(38, int(fig_w * 10.2))) or [text]


def _footnote_height(lines: list[str]) -> float:
    """脚注占用的高度（英寸），用于在画布底部预留空间。"""
    return 0.42 + 0.135 * len(lines)


def _draw_footnote(fig: plt.Figure, text: str, fig_w: float, fig_h: float) -> None:
    """在画布左下角写脚注（已折行）。"""
    lines = _footnote_lines(text, fig_w)
    fig.text(0.045, 0.20 / fig_h, "\n".join(lines), fontsize=7.2, color="#93A1A4",
             ha="left", va="bottom", linespacing=1.55)


def _common_prefix(names: list[str]) -> str:
    """求所有指标名的最长公共前缀（如「民族自治地方」）。

    Args:
        names: 指标名列表。

    Returns:
        公共前缀；短于 4 字时返回空串，避免误剥「规模以上」这类本该保留的限定语。
    """
    if len(names) < 2:
        return ""
    prefix = names[0]
    for n in names[1:]:
        while not n.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix if len(prefix) >= 4 else ""


_DANGLING = "、和及与的对了为 "


def _trim_tail(text: str) -> str:
    """剥掉结尾的连词/助词，避免压出「第一产业对GDP的」这种半截标题。

    剥掉公共后缀「贡献率」后，剩下的「第一产业对GDP的」以「的」收尾，
    单独放进小图标题是个残句；要么把「的」一起削掉，要么整批放弃压缩。
    """
    while text and text[-1] in _DANGLING:
        text = text[:-1]
    return text


def short_names(names: list[str]) -> list[str]:
    """剥离公共前缀/后缀，得到可放进小图标题的短名。

    先剥前缀（「民族自治地方粮食产量」→「粮食产量」），再考虑剥后缀
    （「第一产业法人单位数」→「第一产业」）。每一步都要求：剩余部分至少 2 个字，
    且不会把多个指标剥成同一个名字——否则宁可保留长一点的写法，
    「总人口数」剥成「总」这种标题比不剥更难懂。
    """
    base = list(names)
    prefix = _common_prefix(base)
    if prefix:
        cand = [_trim_tail(n[len(prefix):].lstrip("、和及与 ")) for n in base]
        if all(len(x) >= 2 for x in cand) and len(set(cand)) == len(cand):
            base = cand

    suffix = _common_suffix(base)
    if suffix:
        # 有一行恰好就是「法人单位数」本身时，它是合计行，剥完为空，写作「合计」
        cand = ["合计" if b == suffix else _trim_tail(b[: -len(suffix)]) for b in base]
        # 剥完只剩两个字时要留手：「第一产业贡献率」剥掉「产业贡献率」只剩「第一」，
        # 单独看根本不知道在讲什么；宁可保留完整名称。
        if all(len(x) >= 3 for x in cand) and len(set(cand)) == len(cand):
            return cand
    return base


def _axis_unit(max_value: float, unit: str = "") -> tuple[float, str]:
    """按量级返回 (除数, 单位后缀)，避免纵轴出现一长串 0。

    单位本身以「万」「亿」开头时（万头、亿元…）不再压缩：标题已写明 亿元，
    轴上再标「12万」就变成 12 万亿元，读者要乘两次才敢下结论。
    """
    if unit.startswith(("万", "亿")):
        return 1.0, ""
    if max_value >= 1e8:
        return 1e8, "亿"
    if max_value >= 1e4:
        return 1e4, "万"
    return 1.0, ""


# 长指标名里可以等价替换掉的冗长限定语。只做「去冗余」不做「造缩写」，
# 替换后原意不变；顺序从长到短，保证「对…增长贡献率」整段先被吃掉。
_LABEL_SIMPLIFY: tuple[tuple[str, str], ...] = (
    ("对国内生产总值增长的拉动", "拉动"),
    ("对国内生产总值增长贡献率", "贡献率"),
    ("对国内生产总值增长拉动", "拉动"),
    ("对GDP增长的贡献率", "贡献率"),
    ("对GDP的贡献率", "贡献率"),
    ("对GDP的拉动", "拉动"),
    ("非金融交易资金运用", "资金运用"),
    ("非金融交易资金来源", "资金来源"),
    ("国内生产总值", "GDP"),
)


def compact_labels(names: list[str]) -> list[str]:
    """把过长的指标名压成能放进小图标题的写法。

    统计局的长指标名（「最终消费支出对国内生产总值增长贡献率」21 个字）在 2.9in 宽的
    小图里会一路顶到右上角的变化率标注上，两串字叠在一起谁也读不清。
    压缩只做等价替换；一旦压出重名就整批放弃——重名比长标题更糟。

    Args:
        names: 指标名（通常已经过 :func:`short_names` 处理）。

    Returns:
        等长的标签列表，未缩短的项保持原样。
    """
    simplified: list[str] = []
    for name in names:
        text = name
        for old, new in _LABEL_SIMPLIFY:
            text = text.replace(old, new)
        simplified.append(text)
    if len(set(simplified)) != len(simplified):
        return list(names)
    return [s if len(s) < len(n) else n for s, n in zip(simplified, names)]


def _place_delta(ax: plt.Axes, text: str, title: str, positive: bool) -> None:
    """把「首末变化率」放到面板标题右端；标题过长时改放轴内右上角。

    两者共用面板上方那一条窄带，标题一长就会与变化率叠字。这里按标题的估算宽度
    决定落点：挤得下就并排，挤不下就下沉进轴内——宁可让标注压在留白处，
    也好过两串数字糊成一团。

    Args:
        ax: 目标坐标区。
        text: 已格式化好的变化率文字。
        title: 面板标题（含单位后缀），用于估算宽度。
        positive: 变化率是否为正，决定用强调色还是灰。
    """
    ax_w_pt = ax.get_position().width * ax.figure.get_figwidth() * 72
    # 汉字按 1 em、西文数字按 0.55 em 估算，另留 6% 余量给字距
    est = sum(1.0 if ord(ch) > 0x2E80 else 0.55 for ch in title) * 8.4 * 1.06
    color = _ACCENT if positive else _ACCENT_DOWN
    if est > ax_w_pt * 0.66:
        ax.text(0.995, 0.955, text, transform=ax.transAxes, ha="right", va="top",
                fontsize=7.4, color=color, fontweight="bold", zorder=5)
    else:
        ax.text(0.99, 1.045, text, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.4, color=color, fontweight="bold")



def _value_label(value: float, div: float, suffix: str) -> str:
    """散点旁数值标签：与所在面板的轴刻度同尺度，且保留有效数字。

    轴用「万」时若只留一位小数，16,000 与 17,342 会被写成同一个 1.6 万，
    把真实差异抹掉，因此小数位数按量级放宽到两位。
    """
    if div == 1.0:
        return f"{value:,.0f}" if abs(value) >= 100 or float(value).is_integer() else f"{value:,.1f}"
    scaled = value / div
    decimals = 2 if abs(scaled) < 100 else 0
    text = f"{scaled:,.{decimals}f}"
    return f"{text}{suffix}"


def _tick_decimals(ax: plt.Axes, div: float = 1.0) -> int:
    """按刻度间距决定小数位数。

    拉动、贡献率这类指标只有零点几，刻度是 0 / 0.5 / 1.0；一律按整数格式化会打出
    「0 / 0 / 1」，两个 0 叠在一起，读者会以为轴画错了。间距跨过 1 就用整数，
    否则按 0.1、0.01 逐级放。
    """
    ticks = np.asarray(ax.get_yticks(), dtype=float)
    if len(ticks) < 2 or not div:
        return 0
    gaps = np.abs(np.diff(ticks)) / abs(div)
    positive = gaps[gaps > 1e-12]
    if not len(positive):
        return 0
    gap = float(positive.min())
    if gap >= 1:
        return 0
    return 1 if gap >= 0.1 else 2


def _axis_formatter(decimals: int, div: float, suffix: str) -> FuncFormatter:
    """统一的刻度格式化：压缩量级 + 按需保留小数，0 不挂量级后缀。"""

    def _fmt(v: float, _pos: object) -> str:
        label = f"{v / div:,.{decimals}f}"
        return label if not suffix or v == 0 else f"{label}{suffix}"

    return FuncFormatter(_fmt)


def is_chain_index(unit: str) -> bool:
    """单位是否形如「上年=100」的环比指数。

    注意别把「1978年=100」的定基指数一起抓进来：定基指数从 100 一路涨到几千，
    零基线面积图恰好能表现它的指数式增长，改画偏离带反而看不懂。
    """
    compact = unit.replace(" ", "")
    return "上年" in compact and "100" in compact


def _index_ticks(bottom: float, top: float) -> list[float]:
    """给环比指数挑一组「整数步长 + 必含 100」的纵轴刻度。

    直接交给 MaxNLocator 会挑出 90/105/120 这种步长，读者得先做减法才知道
    105 与 120 差几个点；固定落在 5 或 10 的整数倍上，扫一眼就能对位。
    """
    span = max(top - bottom, 1e-6)
    for step in (1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 25.0, 50.0):
        if span / step <= 5:
            break
    ticks: list[float] = []
    v = float(np.floor(bottom / step) * step)
    while v <= top + 1e-9:
        ticks.append(round(v, 4))
        v += step
    if 100.0 not in ticks:
        ticks.append(100.0)
    return [t for t in ticks if bottom - 1e-9 <= t <= top + 1e-9]


def _draw_index_panel(
    ax: plt.Axes,
    years: list[int],
    series: pd.Series,
    title: str,
    unit: str = "",
) -> None:
    """环比指数专用版式：围绕 100 的偏离带。

    这类表的值全挤在 95–110 之间，从 0 起的面积图会把每格都涂成一块几乎同样的
    实心方块——六格摆在一起，谁也看不出哪年掉下去了。改为以 100 为基准线、
    高出的部分填正向色、低于 100 的部分填反向色，2020 年那种掉头就一眼可见。

    Args:
        ax: 目标坐标区。
        years: 年份序列。
        series: 该指标的数值序列（可含 NaN）。
        title: 面板标题。
        unit: 物理单位（此处恒为「上年=100」），仅用于消歧时拼进标题。
    """
    observed = series.dropna()
    xs = np.array([y for y in years if y in observed.index], dtype=float)
    ys = observed.to_numpy(dtype=float)
    lo, hi = float(ys.min()), float(ys.max())
    # 纵轴留白按波动幅度给：波动小的年份不该被撑满，波动大的年份也不该顶到边框
    pad = max(1.2, (hi - lo) * 0.55)
    bottom, top = min(100.0, lo) - pad, max(100.0, hi) + pad

    # 断档年份仍用浅灰底纹标出，不连线。只标中间的空档：末端还没发布的年份
    # 留白即可，铺一条通栏灰带反而像「面板被切掉一块」。
    present = set(xs)
    for y in years:
        if y not in present and float(min(xs)) < y < float(max(xs)):
            ax.axvspan(y - 0.5, y + 0.5, color=_GAP, zorder=0, linewidth=0)

    ax.axhline(100, color="#9EA3A8", linewidth=0.9, linestyle=(0, (3, 2)), zorder=2)
    above = ys >= 100
    if above.any():
        ax.fill_between(xs, 100, ys, where=above, color=_FILL_2, alpha=0.55,
                        linewidth=0, interpolate=True, zorder=1)
    if (~above).any():
        ax.fill_between(xs, 100, ys, where=~above, color=_INDEX_DOWN, alpha=0.50,
                        linewidth=0, interpolate=True, zorder=1)
    ax.plot(xs, ys, marker="o", markersize=3.0, color="#0F4C52", linestyle="none", zorder=3)

    full_title = f"{title}（{unit}）" if unit else title
    ax.set_title(full_title, fontsize=9.0, color=_TEXT, loc="left", pad=10)
    ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
    ax.set_ylim(bottom, top)
    # 让 MaxNLocator 自己挑步长会给出 90/105/120 这种不着调的刻度，
    # 指数的可读刻度必须是 5 或 10 的整数倍，并且一定要有 100 那条线。
    ax.set_yticks(_index_ticks(bottom, top))
    ax.yaxis.set_major_formatter(_axis_formatter(_tick_decimals(ax), 1.0, ""))
    ax.tick_params(axis="y", labelsize=7.0, colors=_MUTED, length=0, pad=2)
    ax.tick_params(axis="x", labelsize=7.2, colors=_MUTED, length=2, pad=2)
    ax.set_xticks(years[::2] if len(years) > 6 else years)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#D8DEDE")
    for y_tick in ax.get_yticks():
        if bottom < y_tick < top and abs(y_tick - 100) > 1e-9:
            ax.axhline(y_tick, color=_GRID, linewidth=0.6, zorder=0)

    # 环比指数首末相除没有经济含义（「末/首-1」只会得到 -1.7% 这种无意义数字），
    # 改标环比连乘开方得到的年均增速，这才是这张表本身能回答的问题。
    if len(ys) >= 3 and (ys > 0).all():
        growth = float(np.prod(ys / 100.0) ** (1.0 / len(ys)) - 1.0)
        _place_delta(ax, f"年均 {growth:+.1%}", full_title, growth >= 0)
    elif len(ys) < 3:
        ax.text(0.5, 0.06, f"仅 {len(ys)} 个年度有数据", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7.2, color="#93A1A4")


def _draw_panel(
    ax: plt.Axes,
    years: list[int],
    series: pd.Series,
    title: str,
    unit: str = "",
    scale_unit: str = "",
) -> None:
    """在单个坐标区画一条面积图，并标注首末变化率。

    Args:
        ax: 目标坐标区。
        years: 年份序列。
        series: 该指标的数值序列（可含 NaN）。
        title: 面板标题。
        unit: 物理单位；仅在整张图单位不统一时传入，用于消歧。
        scale_unit: 该指标的实际单位，只用于决定纵轴是否压缩量级。
    """
    # 「上年=100」的环比指数走专用版式，零基线面积图对它没有信息量
    if is_chain_index(scale_unit):
        _draw_index_panel(ax, years, series, title, unit)
        return

    segs = segments(series)
    valid = series.dropna()
    max_v = float(valid.max()) if len(valid) else 1.0
    min_v = float(valid.min()) if len(valid) else 0.0
    div, suffix = _axis_unit(max(max_v, abs(min_v)), scale_unit)

    # 一律从 0 起算。曾试过对起伏不足三成的序列放大纵轴，结果 334→333 这种
    # 千分之三的变化会被撑满整格，比「看不出变化」更容易误导，故放弃。
    base, top = 0.0, max_v * 1.16 if max_v > 0 else 1.0
    # 含负值时必须向下让出空间：2020 年最终消费贡献率 -4.3%，若纵轴仍从 0 起，
    # 那一段会被直接裁掉，图上只剩一个空洞，读起来像「那年缺数据」。
    bottom = min_v * 1.25 if min_v < 0 else 0.0

    # 断档年份铺浅灰底纹：宁可留白，也不连线伪造趋势
    present = {years[i] for seg in segs for i in seg}
    for y in years:
        if y not in present:
            ax.axvspan(y - 0.5, y + 0.5, color=_GAP, zorder=0, linewidth=0)

    for seg in segs:
        xs = np.array([years[i] for i in seg], dtype=float)
        ys = series.to_numpy()[seg].astype(float)
        if len(seg) >= 2:
            ax.fill_between(xs, base, ys, color=_FILL, linewidth=0)
            ax.plot(xs, ys, color=_EDGE, linewidth=1.1)
        else:  # 孤立单点无法成面，退化为一个点，明确表示「只有一个观测」
            ax.plot(xs, ys, marker="o", markersize=3.6, color=_EDGE, linestyle="none")

    full_title = f"{title}（{unit}）" if unit else title
    ax.set_title(full_title, fontsize=9.0, color=_TEXT, loc="left", pad=10)
    ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
    ax.set_ylim(bottom, top)
    if bottom < 0:
        # 正负之间的分界要画出来，否则「负值面积」与「正值面积」看起来是同一块
        ax.axhline(0, color="#9AA6A6", linewidth=0.8, zorder=2)

    ax.yaxis.set_major_locator(MaxNLocator(3))

    # 每个非零刻度都带量级后缀：只标最高刻度会出现「10万 / 5 / 0」，
    # 中间的 5 到底是 5 还是 5 万，得靠猜。
    ax.yaxis.set_major_formatter(_axis_formatter(_tick_decimals(ax, div), div, suffix))
    ax.tick_params(axis="y", labelsize=7.0, colors=_MUTED, length=0, pad=2)
    ax.tick_params(axis="x", labelsize=7.2, colors=_MUTED, length=2, pad=2)
    # 年份多时隔年标注，避免小图里刻度挤成一团
    ax.set_xticks(years[::2] if len(years) > 6 else years)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#D8DEDE")
    for y_tick in ax.get_yticks():
        if y_tick != 0 and bottom <= y_tick <= top:
            ax.axhline(y_tick, color=_GRID, linewidth=0.6, zorder=0)

    pct = change_pct(series)
    if pct is not None and len(valid) >= 3:
        _place_delta(ax, f"{pct:+.1%}", full_title, pct >= 0)
    elif len(valid) < 3:
        # 观测过少时不画变化率，避免把两个点之间的偶然差当成趋势
        ax.text(
            0.5, 0.06, f"仅 {len(valid)} 个年度有数据",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.2, color="#93A1A4",
        )


def plot_indicator_grid(
    table: StjTable,
    out_path: str | Path,
    *,
    title: str,
    subtitle: str,
    footnote: str,
    ncols: int = 4,
    dense_panel: bool = False,
) -> Path:
    """把每个指标画成一格面积图，形成小倍数矩阵。

    Args:
        table: 数据表。
        out_path: 输出 PNG 路径。
        title: 总标题。
        subtitle: 副标题。
        footnote: 页脚（来源与口径）。
        ncols: 每行小图数。
        dense_panel: 面板很多时进一步压缩字号。

    Returns:
        输出路径。
    """
    apply_style()
    names = table.indicators
    labels = compact_labels(short_names(names))
    years = table.years
    # 注意：导出文件会把整年缺失的年份整列省掉（如缺 2018、2023），这里保持原有列等距排列，
    # 不补空格年份——补了之后连续段会被切成几根孤立色块，形同柱形图，反而读不出趋势。
    # 断档信息改由副标题「缺 …年」披露。
    # 单位不统一时把单位写进每格标题，否则只在副标题里说明一次
    units = {table.unit_of(n) for n in names}
    show_unit = len(units) > 1
    n = len(names)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))

    # 单个指标时按格子宽度算出来只有 3.4in，副标题和脚注会被切掉，改为整幅宽
    fig_w = 7.6 if n == 1 else 2.95 * ncols + 0.5
    # 只有一行时把画布拔高，否则小图会被压成扁条
    fig_h = 4.3 if nrows == 1 else 1.85 * nrows + 1.6
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=200, squeeze=False)
    # 顶部预留 1.55in 给标题+副标题，否则首行小图标题会顶到副标题上
    foot_h = _footnote_height(_footnote_lines(footnote, fig_w))
    fig.subplots_adjust(left=0.045, right=0.985,
                        top=1 - 1.55 / fig_h, bottom=foot_h / fig_h,
                        hspace=0.78 if nrows > 1 else 0.3, wspace=0.26)
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.set_visible(False)
            continue
        ax.set_facecolor(_PANEL_BG)
        _draw_panel(
            ax, years, table.wide.loc[names[i]], labels[i] if n > 1 else "",
            unit=table.unit_of(names[i]) if show_unit else "",
            scale_unit=table.unit_of(names[i]),
        )

    _fit_left_margin(fig, axes, n)
    fig.text(0.045, 1 - 0.30 / fig_h, title, fontsize=14.5, fontweight="bold",
             color=_TEXT, ha="left", va="top")
    fig.text(0.045, 1 - 0.62 / fig_h, subtitle, fontsize=9.2, color=_MUTED,
             ha="left", va="top")
    _draw_footnote(fig, footnote, fig_w, fig_h)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path


def _draw_point_panel(
    ax: plt.Axes,
    years: list[int],
    series: pd.Series,
    title: str,
    unit: str = "",
    scale_unit: str = "",
) -> None:
    """在单个坐标区画「真时间轴散点」：每个观测一个点，不连线、不填充。

    年份只有三四个且间隔不等时（如 2020、2023、2024），面积图会连成一块斜面，
    等于声称中间年份连续可读；散点按真实年份落位，间隔宽度就是时间宽度，
    缺哪几年一眼能看出来。

    Args:
        ax: 目标坐标区。
        years: 年份序列（升序）。
        series: 该指标的数值序列（可含 NaN）。
        title: 面板标题。
        unit: 物理单位；仅在整张图单位不统一时传入。
        scale_unit: 该指标的实际单位，只用于决定纵轴是否压缩量级。
    """
    observed = series.dropna()
    xs = np.array([y for y in years if y in observed.index], dtype=float)
    ys = observed.to_numpy().astype(float)
    max_v = float(ys.max()) if len(ys) else 1.0
    min_v = float(ys.min()) if len(ys) else 0.0
    div, suffix = _axis_unit(max(max_v, abs(min_v)), scale_unit)

    full_title = f"{title}（{unit}）" if unit else title
    ax.set_title(full_title, fontsize=9.0, color=_TEXT, loc="left", pad=10)

    left, right = min(years) - 0.55, max(years) + 0.55
    # 负值同样不能让零基线吞掉，向下留出一档余量
    bottom = min_v * 1.4 if min_v < 0 else 0.0
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, max_v * 1.34 if max_v > 0 else 1.0)
    if bottom < 0:
        ax.axhline(0, color="#9AA6A6", linewidth=0.8, zorder=2)

    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.plot([x], [y], marker="o", markersize=5.4, color=_EDGE,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        # 相邻两点挨得近（如 2023、2024）时把前一个标签挪到点下方，
        # 否则两个标签会横着挤在一起，数字读串行比不标更糟。
        crowded = i + 1 < len(xs) and (xs[i + 1] - x) <= 1.5
        frac = (x - left) / (right - left)
        if crowded:
            # 挪到点下方时保持居中，否则贴右边缘的标签会横移到前一个点正下方
            ha, dx = "center", 0
        else:
            ha, dx = ("left", 5) if frac < 0.18 else ("right", -5) if frac > 0.82 else ("center", 0)
        ax.annotate(
            _value_label(y, div, suffix),
            xy=(x, y),
            xytext=(dx, -8 if crowded else 6),
            textcoords="offset points",
            ha=ha,
            va="top" if crowded else "bottom",
            fontsize=6.2,
            color=_TEXT if not crowded else _MUTED,
        )

    ax.yaxis.set_major_locator(MaxNLocator(3))

    # 刻度一律带量级后缀（0 除外）：只给最高刻度加「万」会出现「0 / 6 / 12万」，
    # 中间的 6 到底是不是 6 万，读图的人得自己猜。
    ax.yaxis.set_major_formatter(_axis_formatter(_tick_decimals(ax, div), div, suffix))
    ax.tick_params(axis="y", labelsize=7.0, colors=_MUTED, length=0, pad=2)
    ax.tick_params(axis="x", labelsize=7.2, colors=_MUTED, length=2, pad=2)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#D8DEDE")
    for y_tick in ax.get_yticks():
        if y_tick != 0 and bottom <= y_tick <= max_v * 1.34:
            ax.axhline(y_tick, color=_GRID, linewidth=0.6, zorder=0)

    pct = change_pct(series)
    if pct is not None and len(ys) >= 2:
        _place_delta(ax, f"{pct:+.1%}", full_title, pct >= 0)


def plot_indicator_points(
    table: StjTable,
    out_path: str | Path,
    *,
    title: str,
    subtitle: str,
    footnote: str,
    ncols: int = 4,
) -> Path:
    """稀疏年份的小倍数散点矩阵（与面积矩阵同一版式，换成点）。

    适用判据：观测年份 ≤ 4 个。此时面积图的「层厚」信息量极低，
    而连续填充会暗示年份之间连续可得，故改用点。

    Args:
        table: 数据表。
        out_path: 输出 PNG 路径。
        title: 总标题。
        subtitle: 副标题。
        footnote: 页脚。
        ncols: 每行小图数。

    Returns:
        输出路径。
    """
    apply_style()
    names = table.indicators
    labels = compact_labels(short_names(names))
    years = table.years
    units = {table.unit_of(n) for n in names}
    show_unit = len(units) > 1
    n = len(names)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))

    # 单个指标时按格子宽度算出来只有 3.4in，副标题和脚注会被切掉，改为整幅宽
    fig_w = 7.6 if n == 1 else 2.95 * ncols + 0.5
    fig_h = 4.3 if nrows == 1 else 1.85 * nrows + 1.6
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=200, squeeze=False)
    foot_h = _footnote_height(_footnote_lines(footnote, fig_w))
    fig.subplots_adjust(left=0.045, right=0.985,
                        top=1 - 1.55 / fig_h, bottom=foot_h / fig_h,
                        hspace=0.78 if nrows > 1 else 0.3, wspace=0.26)

    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.set_visible(False)
            continue
        ax.set_facecolor(_PANEL_BG)
        _draw_point_panel(
            ax, years, table.wide.loc[names[i]], labels[i] if n > 1 else "",
            unit=table.unit_of(names[i]) if show_unit else "",
            scale_unit=table.unit_of(names[i]),
        )

    _fit_left_margin(fig, axes, n)
    fig.text(0.045, 1 - 0.30 / fig_h, title, fontsize=14.5, fontweight="bold",
             color=_TEXT, ha="left", va="top")
    fig.text(0.045, 1 - 0.62 / fig_h, subtitle, fontsize=9.2, color=_MUTED,
             ha="left", va="top")
    _draw_footnote(fig, footnote, fig_w, fig_h)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path


def plot_single_year_dots(
    table: StjTable,
    out_path: str | Path,
    *,
    title: str,
    subtitle: str,
    footnote: str,
    highlight: tuple[str, ...] = (),
) -> Path:
    """单一年份多类别 → 克利夫兰点图（按量级降序）。

    Args:
        table: 数据表（仅取第一个年份列）。
        out_path: 输出 PNG 路径。
        title: 总标题。
        subtitle: 副标题。
        footnote: 页脚。
        highlight: 需要强调的指标名（用强调色）。

    Returns:
        输出路径。
    """
    apply_style()
    year = table.years[-1]
    raw = table.wide[year].dropna()
    series = leaf_rows(raw).sort_values(ascending=False)
    dropped = len(raw) - len(series)
    labels = dict(zip(table.indicators, short_names(table.indicators)))
    total = float(series.sum())
    unit = table.unit_of(series.index[0])
    div, suffix = _axis_unit(float(series.max()), unit)

    fig_h = 1.0 + 0.42 * len(series)
    fig, ax = plt.subplots(figsize=(9.6, fig_h), dpi=200)
    foot_h = _footnote_height(_footnote_lines(footnote, 9.6))
    fig.subplots_adjust(left=0.30, right=0.87, top=1 - 1.05 / fig_h, bottom=foot_h / fig_h)

    x = series.to_numpy() / div
    y = np.arange(len(series))[::-1]

    for xi, yi, name in zip(x, y, series.index):
        color = "#C97B3C" if name in highlight else _ACCENT
        ax.plot([0, xi], [yi, yi], color="#DCDCDC", linewidth=1.1, zorder=1)
        ax.plot([xi], [yi], marker="o", markersize=7.5, color=color, zorder=2)
        share = series[name] / total * 100
        value_txt = f"{series[name] / div:,.0f}{suffix}" if series[name] / div >= 100 \
            else f"{series[name] / div:,.2f}{suffix}"
        ax.annotate(
            f"{value_txt}　{share:.1f}%",
            xy=(xi, yi),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8.2,
            color=_TEXT,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([labels[n] for n in series.index], fontsize=8.8, color=_TEXT)
    ax.set_xlim(0, max(x) * 1.30)
    ax.set_ylim(-0.7, len(series) - 0.3)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(axis="x", labelsize=7.6, colors=_MUTED, length=2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(f"{unit}（{suffix}）" if suffix else unit, fontsize=7.6, color=_MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#D8DEDE")
    ax.grid(axis="x", color=_GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    fig.text(0.045, 1 - 0.32 / fig_h, title, fontsize=14.5, fontweight="bold",
             color=_TEXT, ha="left", va="top")
    fig.text(0.045, 1 - 0.62 / fig_h, subtitle, fontsize=9.2, color=_MUTED,
             ha="left", va="top")
    _draw_footnote(
        fig,
        footnote + (f"　已剔除 {dropped} 个汇总科目，图中为 {len(series)} 个末级科目。" if dropped else ""),
        9.6,
        fig_h,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path
