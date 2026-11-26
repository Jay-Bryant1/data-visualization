"""两种图版：时间序列小倍数面积图、单年构成克利夫兰点图。

选型依据（与既定图表规范一致）：
- 多指标多年份 → 面积图矩阵。每个指标单独一格、独立纵轴，既保留各自的量级，
  又能在同一画布横向比较增速；柱形/折线一律不用。
- 单一年份多类别 → 克利夫兰点图。类别超过 6 个时环形图已无法读数，
  点图按量级排序后最易比较，也不会像条形图那样用面积放大差异。
"""

from __future__ import annotations

import re
from pathlib import Path
from textwrap import wrap as wrap_text
from typing import Final

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
_HIGHLIGHT = "#C97B3C"   # 需要强调的科目（如人口分类里的「合计」行）

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


def _subtitle_size(text: str, fig_w: float, base: float = 9.2) -> float:
    """副标题按实际长度降字号，避免长句被画布右边缘切掉。

    副标题里堆着年份区间、指标数、单位、缺失说明、指数说明，动辄七十多字；
    固定 9.2pt 时最后一段会在 9.6in 画布外被裁掉，读者看不见「虚线 100 表示持平」
    这类关键读图提示。宁可缩到 7.6pt，也不能丢信息。
    """
    width = sum(1.0 if ord(ch) > 0x2E80 else 0.55 for ch in text) * base
    room = fig_w * 72 * 0.912
    if width <= room:
        return base
    return max(7.6, base * room / width)


def _subtitle_layout(text: str, fig_w: float, base: float = 9.2,
                     max_lines: int = 2) -> tuple[list[str], float]:
    """把副标题折成最多 ``max_lines`` 行，返回 (各行文字, 字号)。

    副标题里挂着年份区间、指标数、单位、缺测说明，指标多、口径杂的表能写到一百多字。
    单行时靠缩字号硬塞，缩到 7.6pt 仍会溢出——被切掉的恰好是「空白即未发布」
    这类最该看见的提示。改为先按「　|　」分段装行，装不下再降字号，一行都不丢。

    Args:
        text: 副标题原文。
        fig_w: 画布宽度（英寸）。
        base: 首选字号。
        max_lines: 最多行数；超过就继续降字号。

    Returns:
        (每行文字列表, 最终字号)。
    """
    sep = "　|　"
    segs = [s for s in text.split(sep) if s]

    def em(s: str) -> float:
        """按「汉字 1 em、西文 0.55 em」估宽。"""
        return sum(1.0 if ord(ch) > 0x2E80 else 0.55 for ch in s)

    for size in (base, base - 0.8, 8.6, 8.2, 7.8, 7.4, 7.0, 6.6):
        budget = fig_w * 72 * 0.912 / size
        lines: list[str] = []
        cur = ""
        for seg in segs:
            cand = f"{cur}{sep}{seg}" if cur else seg
            if not cur or em(cand) <= budget:
                cur = cand
            else:
                lines.append(cur)
                cur = seg
        if cur:
            lines.append(cur)
        if len(lines) <= max_lines and all(em(line) <= budget for line in lines):
            return lines, size
    # 兜底：分段拆不开（分隔符写法不一致）或缩到 6.6pt 仍放不下时按字符硬折，
    # 别让整条副标题单行冲出去被右边缘切掉——切掉的往往是「空白即未发布」这类提示
    hard = wrap_text(text, width=max(24, int(fig_w * 72 * 0.912 / 7.0)))
    if hard:
        return hard[:max_lines], 7.0
    return [text], 6.6


def _draw_subtitle(fig: plt.Figure, text: str, fig_w: float, fig_h: float,
                   *, top_in: float = 0.62, base: float = 9.6,
                   max_lines: int = 2) -> None:
    """按折行结果把副标题写在标题下方。"""
    lines, size = _subtitle_layout(text, fig_w, base, max_lines)
    gap = size * 0.023
    for i, line in enumerate(lines):
        fig.text(0.045, 1 - (top_in + gap * i) / fig_h, line, fontsize=size,
                 color=_MUTED, ha="left", va="top")


def _footnote_lines(text: str, fig_w: float) -> list[str]:
    """按画布宽度把脚注折成若干行。

    尾注长短随表而变（民族自治地方那几条尤其长），固定单行会在窄图上被右边缘切掉，
    切掉的正是口径说明，最不该丢。
    """
    # 7.2pt 中文按每字 0.1 英寸算，每英寸约 10 字；脚注左起 0.045·fig_w，
    # 右侧还要留同样的边，可用宽度只有 0.91·fig_w，再留 5% 余量取 8.7——
    # 按整幅 fig_w 折（系数 10.2）会让末行末尾几个字被右边缘切掉，切掉的又是口径说明。
    return wrap_text(text, width=max(38, int(fig_w * 8.7))) or [text]


def _footnote_height(lines: list[str]) -> float:
    """脚注占用的高度（英寸），用于在画布底部预留空间。

    脚注自画布底部 0.20 英寸起向上排，每行约占 0.135 英寸；
    上方还要给最下排子图的横轴刻度数字留出落脚空间（约 0.22 英寸），
    否则多行尾注会把刻度数字压住——人口分类里三条口径说明的表就撞过。
    """
    return 0.62 + 0.135 * len(lines)


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
    if len(prefix) < 4:
        return ""
    # 「15岁及以上」这类前缀是统计口径，不是可以省掉的重复字：剥掉之后
    # 「15岁及以上人口数」只剩「人口数」，读者会以为那是全年龄人口。
    if any(mark in prefix for mark in ("岁", "以上", "以下", "不满")):
        return ""
    return prefix


_DANGLING = "、和及与的对了为 "


def _trim_tail(text: str) -> str:
    """剥掉结尾的连词/助词，避免压出「第一产业对GDP的」这种半截标题。

    剥掉公共后缀「贡献率」后，剩下的「第一产业对GDP的」以「的」收尾，
    单独放进小图标题是个残句；要么把「的」一起削掉，要么整批放弃压缩。
    """
    while text and text[-1] in _DANGLING:
        text = text[:-1]
    return text


_SCOPE_PARENS: Final[tuple[str, ...]] = ("抽样调查", "人口普查", "普查", "抽样")
_PAREN_TEXT: Final = re.compile(r"[\(（](?P<text>[^\(\)（）]{1,12})[\)）]")


def _drop_scope_paren(name: str) -> str:
    """去掉指标名里「（人口抽样调查）」这类口径括号，图表副标题已交代过。

    一格里放不下「15岁及以上人口数（人口抽样调查）」这种 18 字标题，
    括号里的口径又对每一格都相同，留着只会把标题挤到变化率标注上。
    """
    def _repl(match: re.Match[str]) -> str:
        return "" if any(k in match.group("text") for k in _SCOPE_PARENS) else match.group(0)

    return _PAREN_TEXT.sub(_repl, name).strip()


_ORDINAL_ONLY: Final = re.compile(r"^第[一二三四五六七八九十]$")


def _label_ok(text: str) -> bool:
    """剥完后的短名能否独立示人。

    三个字的门槛会把「教育」「金融业」这类行业名挡在外面，导致整批压缩被否；
    真正要防的是「第一产业贡献率」剥成「第一」这种序数残句，所以单列一条。
    """
    if text == "合计":
        return True
    if _ORDINAL_ONLY.match(text):
        return False
    return len(text) >= 2


_UNITLIKE_PAREN: Final = re.compile(r"\s*[\(（][^\(\)（）]{1,14}[=\／/][^\(\)（）]{0,14}[\)）]\s*$")


def _drop_unit_paren(name: str) -> str:
    """去掉指标名末尾「(美元=100)」这类量纲说明。

    这类括号既不是口径也不是名字的一部分，留着会在剥公共后缀时被切成半截括号，
    标题变成「美元汇率 (美」这种残句。单位已在副标题与纵轴上交代。
    """
    return _UNITLIKE_PAREN.sub("", name).strip()


def _paren_balanced(text: str) -> bool:
    """括号是否两两配对；剥前缀/后缀时若切进括号内，就会剩下半截。"""
    return text.count("(") == text.count(")") and text.count("（") == text.count("）")


def short_names(names: list[str]) -> list[str]:
    """剥离公共前缀/后缀，得到可放进小图标题的短名。

    先剥前缀（「民族自治地方粮食产量」→「粮食产量」），再考虑剥后缀
    （「第一产业法人单位数」→「第一产业」）。每一步都要求：剩余部分至少 2 个字，
    且不会把多个指标剥成同一个名字——否则宁可保留长一点的写法，
    「总人口数」剥成「总」这种标题比不剥更难懂。
    """
    base = [_drop_unit_paren(_drop_scope_paren(n)) for n in names]
    prefix = _common_prefix(base)
    if prefix:
        cand = [_trim_tail(n[len(prefix):].lstrip("、和及与 ")) for n in base]
        if (all(len(x) >= 2 for x in cand) and len(set(cand)) == len(cand)
                and all(_paren_balanced(x) for x in cand)):
            base = cand

    suffix = _common_suffix(base)
    if suffix:
        # 有一行恰好就是「法人单位数」本身时，它是合计行，剥完为空，写作「合计」
        cand = ["合计" if b == suffix else _trim_tail(b[: -len(suffix)]) for b in base]
        # 「城镇就业人员」剥掉「就业人员」只剩「城镇」，单看不知是合计还是城乡之一；
        # 它是表里的总量行，改标「合计」才不会被读成并列的分项。
        if cand[0] in ("城镇", "乡村", "全国") and cand[0] not in cand[1:]:
            cand[0] = "合计"
        # 剥完只剩两个字时要留手：「第一产业贡献率」剥掉「产业贡献率」只剩「第一」，
        # 单独看根本不知道在讲什么；宁可保留完整名称。
        # 同理，剥完落在「15岁及以上」这种口径短语上也是残句，不能留。
        if (
            all(_label_ok(x) for x in cand)
            and all(_paren_balanced(x) for x in cand)
            and len(set(cand)) == len(cand)
            and not any(x.endswith(("岁", "以上", "以下", "口径")) for x in cand)
            # 后缀可能从词中间切开：「能源工业投资 / 煤炭采选业投资」的公共后缀是「业投资」，
            # 剥完首行只剩「能源工」，读起来像残句；落在这些连接字上就放弃剥后缀
            and not any(x.endswith(("工", "业", "的", "和", "及", "与", "、")) for x in cand)
        ):
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
    # 原始表里写作「私营企业城镇就业人员」「个体城镇就业人员」，剥掉后缀后剩下
    # 「私营企业城镇」「个体城镇」，语序拗口；调序后原意不变
    ("私营企业城镇", "城镇私营企业"),
    ("个体城镇", "城镇个体"),
    # 个体户数表里的「城镇就业人数」其实是个体口径，与全国城镇就业人员同名不同义，
    # 不改会被读成 47,535 万人那个数
    ("城镇就业人数", "城镇个体就业人数"),
    # 平均工资指数表里「平均货币工资指数」「平均实际工资指数」各占一行，
    # 「平均」二字在指数里没有额外信息，留着只会把标题顶到隔壁格
    ("平均货币工资指数", "货币工资指数"),
    ("平均实际工资指数", "实际工资指数"),
    ("其他城镇非私营单位", "其他单位"),
    ("城镇非私营集体单位", "非私营集体单位"),
    ("城镇非私营单位", "非私营单位"),
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
    # 调序类替换（「私营企业城镇」→「城镇私营企业」）字数不变，不能按「必须更短」筛
    return [s if s != n else n for s, n in zip(simplified, names)]


def _year_ticks(years: list[int], step: int = 2) -> list[int]:
    """隔年取刻度时把末年补上。

    2016–2023 隔年取到 2020、2022，最后一年 2023 不出现，读者会把 2022 当成年份终点；
    末格距上一个刻度不足一个步长时，用末年顶掉它。
    """
    if len(years) <= 6:
        return list(years)
    ticks = list(years[::step])
    if ticks[-1] != years[-1]:
        ticks = ticks[:-1] + [years[-1]] if len(ticks) > 1 else [years[-1]]
    return ticks


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
        # 落进轴内时贴「空的那半边」：上行序列左下角是线、左上角是空气，
        # 下行序列反过来。若一律贴右上，会和右边一格的同行标注并排，看着像属于隔壁。
        if positive:
            ax.text(0.012, 0.955, text, transform=ax.transAxes, ha="left", va="top",
                    fontsize=7.4, color=color, fontweight="bold", zorder=5)
        else:
            ax.text(0.988, 0.955, text, transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.4, color=color, fontweight="bold", zorder=5)
    else:
        ax.text(0.99, 1.045, text, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.4, color=color, fontweight="bold")



_RATE_UNITS: Final[frozenset[str]] = frozenset({"%", "‰", "百分点"})


def _delta_text(series: pd.Series, unit: str) -> tuple[str, bool] | None:
    """首末变化标注：率类指标写「百分点差」，其余写相对变化率。

    增速从 6.9% 掉到 -3.9%，按相对变化算是 -156.5%，这个数字没有含义也无从核对；
    率类指标本身就在百分数上，读者要的是「掉了 10.8 个百分点」。
    起点为零或负时相对变化率同样失真（-394.6% 那种），统一回落到差值。

    Args:
        series: 该面板的数值序列（可含 NaN）。
        unit: 该指标的单位。

    Returns:
        (标注文字, 是否为正)；观测不足两点时返回 None。
    """
    clean = series.dropna()
    if len(clean) < 2:
        return None
    first, last = float(clean.iloc[0]), float(clean.iloc[-1])
    if unit in _RATE_UNITS:
        diff = last - first
        suffix = "pp" if unit == "%" else unit
        return f"{diff:+.1f}{suffix}", diff >= 0
    if first <= 0:
        return f"{last - first:+,.0f}", last >= first
    pct = change_pct(series)
    if pct is None:
        return None
    return f"{pct:+.1%}", pct >= 0


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

    full_title = _panel_title(ax, title, unit)
    ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
    ax.set_ylim(bottom, top)
    # 让 MaxNLocator 自己挑步长会给出 90/105/120 这种不着调的刻度，
    # 指数的可读刻度必须是 5 或 10 的整数倍，并且一定要有 100 那条线。
    ax.set_yticks(_index_ticks(bottom, top))
    ax.yaxis.set_major_formatter(_axis_formatter(_tick_decimals(ax), 1.0, ""))
    ax.tick_params(axis="y", labelsize=7.0, colors=_MUTED, length=0, pad=2)
    ax.tick_params(axis="x", labelsize=7.2, colors=_MUTED, length=2, pad=2)
    ax.set_xticks(_year_ticks(years))
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


def _panel_title(ax: plt.Axes, title: str, unit: str = "") -> str:
    """写小图标题：指数类不再重复标注「（上年=100）」，过长时按宽度降字号。

    副标题已经交代「虚线 100＝与上年持平」，每个指数格再挂一次「（上年=100）」
    只会把标题顶进右边一格；而平均工资指数这类名字本身仍有二十字，固定 9pt 时
    必然越界——按面板实际宽度估算，缩到放得下为止。

    Returns:
        实际写入的标题文字（供 :func:`_place_delta` 估算宽度）。
    """
    full = title if (unit and is_chain_index(unit)) else (f"{title}（{unit}）" if unit else title)
    width = ax.get_position().width * ax.figure.get_figwidth() * 72
    est = sum(1.0 if ord(ch) > 0x2E80 else 0.55 for ch in full) * 9.0 * 1.04
    size = 9.0 if est <= width * 0.97 else max(7.0, 9.0 * width * 0.97 / est)
    ax.set_title(full, fontsize=size, color=_TEXT, loc="left", pad=10)
    return full


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

    full_title = _panel_title(ax, title, unit)
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
    # 年份多时隔年标注，避免小图里刻度挤成一团；末年一定要出现在刻度里
    ax.set_xticks(_year_ticks(years))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#D8DEDE")
    for y_tick in ax.get_yticks():
        if y_tick != 0 and bottom <= y_tick <= top:
            ax.axhline(y_tick, color=_GRID, linewidth=0.6, zorder=0)

    delta = _delta_text(series, scale_unit)
    if delta is not None and len(valid) >= 3:
        _place_delta(ax, delta[0], full_title, delta[1])
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
    _draw_subtitle(fig, subtitle, fig_w, fig_h)
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

    full_title = _panel_title(ax, title, unit)

    left, right = min(years) - 0.55, max(years) + 0.55
    ax.set_xlim(left, right)
    if is_chain_index(scale_unit):
        # 环比指数（上年=100）不能从 0 起画：95–105 会全挤在顶端一条线上，
        # 波动完全读不出来。改为围绕 100 留边距，并画出 100 基准线。
        lo = min(float(np.nanmin(ys)), 100.0) if len(ys) else 95.0
        hi = max(float(np.nanmax(ys)), 100.0) if len(ys) else 105.0
        pad = max((hi - lo) * 0.45, 1.5)
        bottom, top = lo - pad, hi + pad
        ax.set_ylim(bottom, top)
        ax.axhline(100, color="#9AA6A6", linewidth=0.8, zorder=2)
    else:
        # 负值同样不能让零基线吞掉，向下留出一档余量
        bottom = min_v * 1.4 if min_v < 0 else 0.0
        top = max_v * 1.34 if max_v > 0 else 1.0
        ax.set_ylim(bottom, top)
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
        if y_tick != 0 and bottom <= y_tick <= top:
            ax.axhline(y_tick, color=_GRID, linewidth=0.6, zorder=0)

    delta = _delta_text(series, scale_unit)
    if delta is not None and len(ys) >= 2:
        _place_delta(ax, delta[0], full_title, delta[1])


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
    _draw_subtitle(fig, subtitle, fig_w, fig_h)
    _draw_footnote(fig, footnote, fig_w, fig_h)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path


_ADDITIVE_UNITS: Final = frozenset({
    "人", "万人", "亿人", "户", "万户", "个", "台", "辆", "艘", "张", "座", "所",
    "件", "部", "吨", "万吨", "元", "万元", "亿元",
})


def _is_additive(unit: str) -> bool:
    """该单位下的数值能否相加。

    只有可加量（人口、户数、金额）谈「占合计的比重」才有意义；
    预期寿命 80.88 岁除以三项之和得 34.5%，属于把不能加的数硬加，必须去掉。
    """
    return unit.strip() in _ADDITIVE_UNITS


def _group_by_unit(names: list[str], units: dict[str, str]) -> list[tuple[str, list[str]]]:
    """按单位把指标分组，组序按「先绝对量、后比例/强度」，组内保持传入顺序。"""
    groups: list[tuple[str, list[str]]] = []
    for name in names:
        unit = units[name]
        for group_unit, members in groups:
            if group_unit == unit:
                members.append(name)
                break
        else:
            groups.append((unit, [name]))
    return sorted(groups, key=lambda g: 0 if _is_additive(g[0]) else 1)


def _stated_total(raw: StjTable | None, year: int, names: list[str], unit: str) -> float | None:
    """找这一组指标「在表里明写着的总数」，用于算占比；找不到就返回 None。

    七普那张表把汉族、男性、城镇人口并列在一起，三者互相重叠，
    它们之和并不等于任何口径的总人口，拿这个和当分母算出的 30.1% 是假数。
    只有当原表里存在同单位的合计行、且数值与这组之和吻合时，占比才成立。
    """
    if raw is None:
        return None
    total = 0.0
    for name in names:
        if name not in raw.wide.index:
            return None
        value = raw.wide.at[name, year]
        if pd.isna(value):
            return None
        total += float(value)
    if total <= 0:
        return None
    for name in raw.indicators:
        if name in names or raw.unit_of(name) != unit:
            continue
        value = raw.wide.at[name, year]
        if pd.isna(value):
            continue
        if abs(float(value) - total) <= max(abs(total) * 0.005, 1e-9):
            return float(value)
    return None


def _draw_dot_group(
    ax: plt.Axes,
    names: list[str],
    series: pd.Series,
    labels: dict[str, str],
    unit: str,
    highlight: tuple[str, ...],
    clipped: bool,
    total: float | None = None,
) -> None:
    """画一组同量纲指标的克利夫兰点图（一根横轴、一组标签）。"""
    values = series[names]
    max_v = float(values.max())
    min_v = float(values.min())
    # 单年构成图的每根横轴只服务一组指标，几万人的量级直接写「34,507」比「3.45万」
    # 更好对；只有上到百万级（金额表常见）才压缩，免得标签拉得太长。
    div, suffix = _axis_unit(max_v, unit) if max_v >= 1e6 else (1.0, "")
    x = values.to_numpy(dtype=float) / div
    y = np.arange(len(names), dtype=float)[::-1]

    # 数值远离零且彼此接近时（预期寿命 75–81 岁），从零起画会把所有点挤在一处，
    # 此时截断横轴并在组标题旁注明起点，避免读成「都没有差别」。
    x0 = float(x.min() - (x.max() - x.min()) * 0.6) if clipped else 0.0
    x0 = max(x0, 0.0)
    span = max(x.max() - x0, 1e-6)
    ax.set_xlim(x0, x.max() + span * 0.30)
    ax.set_ylim(-0.7, len(names) - 0.3)

    for xi, yi, name in zip(x, y, names):
        color = _HIGHLIGHT if name in highlight else _ACCENT
        if not clipped:
            ax.plot([x0, xi], [yi, yi], color="#DCDCDC", linewidth=1.1, zorder=1)
        ax.plot([xi], [yi], marker="o", markersize=7.4, color=color, zorder=2)
        value = float(series[name])
        txt = f"{value / div:,.0f}{suffix}" if value / div >= 100 else f"{value / div:,.2f}{suffix}"
        if total:
            txt = f"{txt}　{value / total:.1%}"
        ax.annotate(txt, xy=(xi, yi), xytext=(8, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8.2, color=_TEXT)

    ax.set_yticks(y)
    ax.set_yticklabels([labels.get(n, n) for n in names], fontsize=8.8, color=_TEXT)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    decimals = 1 if max_v / div < 10 else 0
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.{decimals}f}"))
    ax.tick_params(axis="x", labelsize=7.6, colors=_MUTED, length=2)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#D8DEDE")
    ax.grid(axis="x", color=_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_facecolor(_PANEL_BG)
    if clipped:
        ax.text(1.0, 1.06, f"横轴自 {x0 * div:,.0f}{suffix} 起", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.6, color="#93A1A4")


def plot_single_year_dots(
    table: StjTable,
    out_path: str | Path,
    *,
    title: str,
    subtitle: str,
    footnote: str,
    highlight: tuple[str, ...] = (),
    raw_table: StjTable | None = None,
) -> Path:
    """单一年份多类别 → 克利夫兰点图，并按单位分面。

    一张表里常混着「万人」「%」「岁」「人/户」几种量纲（七普主要数据尤其典型），
    塞进同一根横轴等于宣称 105 的性别比和 128631 万人的汉族人口可以直接比大小，
    所以按单位分组，每组一根独立横轴；「占比」只在该组单位可加总时才标。

    Args:
        table: 数据表（仅取第一个年份列）。
        out_path: 输出 PNG 路径。
        title: 总标题。
        subtitle: 副标题。
        footnote: 页脚。
        highlight: 需要强调的指标名（用强调色）。
        raw_table: 剔除恒定科目之前的原表，用来找「明写的总数」以判断占比是否成立。

    Returns:
        输出路径。
    """
    apply_style()
    year = table.years[-1]
    raw = table.wide[year].dropna()
    series = leaf_rows(raw)
    dropped = len(raw) - len(series)
    labels = dict(zip(table.indicators, short_names(table.indicators)))
    units = {n: table.unit_of(n) for n in series.index}
    ordered = list(series.sort_values(ascending=False).index)
    groups = _group_by_unit(ordered, units)

    row_h, gap_h = 0.40, 0.40
    # 只有一组时不画组名：单位已经写在副标题和横轴上，单独一行「岁 · 3 项」只是噪音
    show_header = len(groups) > 1
    header_h = 0.26 if show_header else 0.0
    foot_h = _footnote_height(_footnote_lines(footnote, 9.6))
    body_h = sum(row_h * len(m) for _, m in groups) + (len(groups) - 1) * gap_h + header_h * len(groups)
    fig_h = (1.32 if show_header else 1.15) + body_h + foot_h
    fig = plt.figure(figsize=(9.6, fig_h), dpi=200)

    cursor = foot_h / fig_h + 0.04
    for unit, members in reversed(groups):
        ax_h = row_h * len(members) / fig_h
        ax = fig.add_axes([0.295, cursor, 0.575, ax_h])
        values = series[members]
        clipped = (
            len(members) >= 2
            and not _is_additive(unit)
            and float(values.min()) > 0
            and float(values.min()) / float(values.max()) > 0.55
        )
        share_total = (
            _stated_total(raw_table, year, members, unit) if _is_additive(unit) else None
        )
        _draw_dot_group(ax, members, series, labels, unit, highlight, clipped, share_total)
        if show_header:
            # 组名压在本组面板的上沿：夹在两个面板之间会被读成上一组的注解
            fig.text(0.045, cursor + ax_h, f"{unit or '单位未标注'} · {len(members)} 项",
                     fontsize=8.0, color=_MUTED, ha="left", va="bottom", fontweight="bold")
            cursor += ax_h + (header_h + gap_h) / fig_h
        else:
            cursor += ax_h

    fig.text(0.045, 1 - 0.32 / fig_h, title, fontsize=14.5, fontweight="bold",
             color=_TEXT, ha="left", va="top")
    _draw_subtitle(fig, subtitle, 9.6, fig_h)
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


def _pair_years(table: StjTable, year_from: int | None,
                year_to: int | None) -> tuple[int, int]:
    """挑出「两个可比年份」：缺值最少、且最新的两年。

    分行业明细表常常只有最近两年填了数（2023 年统计分类调整后重新发布），
    用最后两列有数的年份做对比，比把 9 个年份列铺成 20 格面积图好读得多。
    """
    if year_from is not None and year_to is not None:
        return year_from, year_to
    density = {y: float(table.wide[y].notna().mean()) for y in table.years}
    usable = [y for y in table.years if density[y] >= 0.7]
    if len(usable) < 2:
        usable = table.years[-2:]
    return usable[-2], usable[-1]


def plot_year_pair_dumbbell(
    table: StjTable,
    out_path: str | Path,
    *,
    title: str,
    subtitle: str,
    footnote: str,
    year_from: int | None = None,
    year_to: int | None = None,
) -> Path:
    """两个年份的类别对比 → 哑铃点图（一行一个类别，一段连线即一个变化）。

    二十个行业挤进小倍数面积图，每格只有两个观测点，面积会塌成一根竖线，
    读者既看不出量级排序也看不出谁增谁减；换成哑铃点图，一行一个行业，
    横轴是量级、连线是变化方向，一屏就能排完序。

    Args:
        table: 数据表。
        out_path: 输出 PNG 路径。
        title: 总标题。
        subtitle: 副标题。
        footnote: 页脚。
        year_from: 起点年份，缺省取倒数第二个有数的年份。
        year_to: 终点年份，缺省取最后一个有数的年份。

    Returns:
        输出路径。
    """
    apply_style()
    y_from, y_to = _pair_years(table, year_from, year_to)
    both = table.wide[[y_from, y_to]].dropna()
    if both.empty:
        raise ValueError(f"{y_from}/{y_to} 两年没有同时有值的指标")

    units = {table.unit_of(n) for n in both.index}
    unit = units.pop() if len(units) == 1 else ""
    labels_all = dict(zip(table.indicators, short_names(table.indicators)))
    # 「合计」是各行业之和，留在图里会把横轴拉到它的量级，19 个行业全挤在左端一厘米内
    names_all = [n for n in both.index if labels_all.get(n) != "合计"]
    if len(names_all) < 3:
        names_all = list(both.index)
    latest = both.loc[names_all, y_to].sort_values(ascending=False)
    names = list(latest.index)
    labels = labels_all

    values = latest.to_numpy(dtype=float)
    prev = both[y_from].reindex(names).to_numpy(dtype=float)
    max_v = float(values.max())
    div, suffix = _axis_unit(max_v, unit) if max_v >= 1e6 else (1.0, "")

    row_h, head_h = 0.30, 1.56
    foot_h = _footnote_height(_footnote_lines(footnote, 9.6))
    fig_h = head_h + row_h * len(names) + foot_h
    fig = plt.figure(figsize=(9.6, fig_h), dpi=200)

    left, width = 0.335, 0.515
    ax = fig.add_axes([left, foot_h / fig_h + 0.02, width, row_h * len(names) / fig_h])

    ys = np.arange(len(names), dtype=float)[::-1]
    x_new = values / div
    x_old = prev / div
    ax.set_xlim(0, max_v / div * 1.32)
    ax.set_ylim(-0.7, len(names) - 0.3)

    for xi_old, xi_new, yi, raw in zip(x_old, x_new, ys, values):
        ax.plot([xi_old, xi_new], [yi, yi], color="#DCDCDC", linewidth=1.6,
                solid_capstyle="round", zorder=1)
        ax.plot([xi_old], [yi], marker="o", markersize=5.6, markerfacecolor="white",
                markeredgecolor=_FILL_2, markeredgewidth=1.5, zorder=2)
        ax.plot([xi_new], [yi], marker="o", markersize=6.6, color=_ACCENT,
                markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        text = f"{raw / div:,.0f}{suffix}" if raw / div >= 10 else f"{raw / div:,.2f}{suffix}"
        # 标注跟在「两个点里靠右的那个」后面：减项的上一年的点比当年更靠右，
        # 若一律贴当年点写，数字会正好压在上一年的空心圆上
        anchor = max(xi_old, xi_new)
        ax.annotate(text, xy=(anchor, yi), xytext=(7, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=7.8, color=_TEXT)

    ax.set_yticks(ys)
    ax.set_yticklabels([labels.get(n, n) for n in names], fontsize=8.4, color=_TEXT)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    decimals = _tick_decimals(ax, div)
    ax.xaxis.set_major_formatter(_axis_formatter(decimals, div, suffix))
    ax.tick_params(axis="x", labelsize=7.4, colors=_MUTED, length=2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#D8DEDE")
    ax.grid(axis="x", color=_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_facecolor(_PANEL_BG)

    # 变化率单列排在右侧，和数值标签分开，避免两种数字挤在同一行里
    for yi, name in zip(ys, names):
        old_v = float(both.at[name, y_from])
        new_v = float(both.at[name, y_to])
        if old_v <= 0:
            continue
        pct = (new_v / old_v - 1) * 100
        ax.annotate(f"{pct:+.1f}%", xy=(1.012, yi), xycoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=7.8,
                    color=_ACCENT if pct >= 0 else _ACCENT_DOWN,
                    fontweight="bold" if abs(pct) >= 10 else "normal")

    fig.text(0.045, 1 - 0.34 / fig_h, title, fontsize=14.5, fontweight="bold",
             color=_TEXT, ha="left", va="top")
    _draw_subtitle(fig, subtitle, 9.6, fig_h, top_in=0.64)
    _draw_footnote(fig, footnote, 9.6, fig_h)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path


_RATE_HINTS: Final[tuple[str, ...]] = ("增长", "指数", "比重", "率", "占比")


def _is_rate_table(table: StjTable) -> bool:
    """整表是否都是「率」类指标（增速、指数、比重）。

    113 个行业的投资增速铺成 113 格面积图要 55 英寸高，读者滚三屏才能看完；
    这些指标共用同一量纲、围绕 0 上下摆动，用一张「行业 × 年份」矩阵着色反而更清楚。
    """
    units = {table.unit_of(n) for n in table.indicators}
    if units <= {"%", "％"}:
        return True
    return all(any(h in n for h in _RATE_HINTS) for n in table.indicators)


def _heat_cmap_rate() -> mpl.colors.LinearSegmentedColormap:
    """增速用的双向色阶：正为暖（赭红），负为冷（青绿），0 处近白。

    不用默认的红蓝：本项目其余图里「降」用的是青绿、「升」的热度用赭色，
    换一套配色会让同一份数据在两页图里指向相反的含义。
    """
    return mpl.colors.LinearSegmentedColormap.from_list(
        "stj_rate",
        ["#2F6F62", "#7FB2A5", "#E8F0EE", "#F5F9F8", "#EFC9AC", "#C97B3C", "#9E4E2E"],
        N=256,
    )


def _heat_cmap_level() -> mpl.colors.LinearSegmentedColormap:
    """量级用的单向色阶：浅薄荷 → 深青，配合对数刻度读数量级。"""
    return mpl.colors.LinearSegmentedColormap.from_list(
        "stj_level", ["#F4F9F8", "#CFE4DF", "#8FC0B6", "#4E8F87", "#22585C", "#0F3B40"], N=256
    )


def plot_heatmap_matrix(
    table: StjTable,
    out_path: str | Path,
    *,
    title: str,
    subtitle: str,
    footnote: str,
    max_rows: int = 46,
) -> Path:
    """指标多到几十上百个时 → 「指标 × 年份」矩阵热力图。

    大表（分行业投资增速 113 行、HS 章出口 120 行）铺成小倍数矩阵会变成 30 行的画卷，
    单格只有 2 英寸宽，既读不出趋势也没法横向比较。矩阵着色后一屏读完，且两种量纲分开处理：

    - 率类（增速、指数、比重）：颜色就是数值本身，暖色＝增长、冷色＝下降；
    - 量级类（进出口额、投资额）：行已按最新一年排序，颜色若再画绝对值只是把排序重复一遍，
      改画「相对本行首个有数年份的倍数」，同一张图里既读得出谁体量大、也读得出谁在涨。

    Args:
        table: 数据表。
        out_path: 输出 PNG 路径。
        title: 总标题。
        subtitle: 副标题。
        footnote: 页脚。
        max_rows: 每栏最多行数。

    Returns:
        输出路径。
    """
    apply_style()
    years = list(table.years)
    values = table.wide.copy()
    # 末列几乎全空（2025 年只更新了总量行）时，按「有数的最后一列」排序，否则整屏都按 NaN 排
    density = {y: float(values[y].notna().mean()) for y in years}
    sort_year = max((y for y in years if density[y] >= 0.3), default=years[-1])
    names = list(values[sort_year].sort_values(ascending=False, na_position="last").index)

    labels_all = dict(zip(table.indicators, compact_labels(short_names(table.indicators))))
    rate = _is_rate_table(table)
    raw = values.loc[names].to_numpy(dtype=float)

    if rate:
        span = float(np.nanmax(np.abs(raw))) if np.isfinite(raw).any() else 1.0
        span = max(span, 1e-6)
        shades = raw
        norm = mpl.colors.TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)
        cmap = _heat_cmap_rate()
        bar_label = "增速（%）"
        formatter = None
    else:
        # 每行除以本行首个观测值 → 倍数；首值缺失或非正的行（占比、指数）不参与着色
        base = np.full(len(names), np.nan)
        for i, row in enumerate(raw):
            valid = row[np.isfinite(row) & (row > 0)]
            if len(valid):
                base[i] = valid[0]
        # 有些行业对某品种的消费就是 0：除出 0、log2(0) 为 -inf，都在预期之内，
        # 用 errstate 压掉告警（-inf 随后会被 clip 成最深色，本轮已知且无害）
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = raw / base[:, None]
            shades = np.log2(ratio)
        finite = shades[np.isfinite(shades)]
        span = float(np.percentile(np.abs(finite), 92)) if finite.size else 1.0
        span = min(max(span, 0.2), 3.0)
        shades = np.clip(shades, -span, span)
        norm = mpl.colors.TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)
        cmap = _heat_cmap_rate()
        bar_label = "相对本行首个有数年份的倍数（对数）"
        formatter = lambda v, _: f"{2 ** v:,.2g}×"  # noqa: E731
    cmap = cmap.copy()
    cmap.set_bad("#F2F4F4")

    n = len(names)
    ncol_blocks = max(1, int(np.ceil(n / max_rows)))
    per_block = int(np.ceil(n / ncol_blocks))
    row_h = 0.132
    # 标签列宽按**实际最长的行标签**估，而不是写死 2.28 英寸：
    # 「铁路、船舶、航空航天和其他运输设备制造业工业生产者出厂价格指数」有 30 字，
    # 固定宽度装不下，会被画布左缘切掉，只剩「、船舶、航空航天…」这样的残句。
    label_em = max(
        sum(1.0 if ord(ch) > 0x2E80 else 0.55 for ch in labels_all.get(c, c))
        for c in names
    )
    label_w = max(2.28, label_em * 5.9 / 72 + 0.18)
    # 年份列宽也按列数走：10 个年份挤在 1.62 英寸里，旋转 90° 的年份刻度会糊成一片
    block_w = max(1.62, len(years) * 0.19)
    gap_w = 0.30
    # 每一栏的行标签都写在本栏左侧，所以第一栏也要留出一个标签宽度；
    # 第二栏起每栏还要再加「标签 + 色块 + 间隔」一整步，少加一次就会把末栏推出画布右沿。
    # 原来无条件乘 ncol_blocks，单栏图的画布比内容宽出一整个标签位，右侧大片空白。
    fig_w = (label_w + 0.10) + ncol_blocks * block_w + (ncol_blocks - 1) * (label_w + gap_w) + 0.55
    # 标题带高度随副标题行数走：副标题两三行时固定 1.46 英寸会让色标压住末行
    # 热力图的画布被行标签吃掉一大半宽度，副标题按 2 行折放不下，
    # _subtitle_layout 会退化成单行，末段被右边缘切掉；这里放开到 3 行
    sub_lines, _ = _subtitle_layout(subtitle, fig_w, 9.2, max_lines=4)
    head_h = 0.64 + 0.34 * len(sub_lines) + 0.40
    body_h = row_h * per_block
    foot_h = _footnote_height(_footnote_lines(footnote, fig_w))
    fig_h = head_h + body_h + 0.60 + foot_h
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)

    left0 = (label_w + 0.10) / fig_w
    for bi in range(ncol_blocks):
        chunk = names[bi * per_block:(bi + 1) * per_block]
        if not chunk:
            continue
        rows = len(chunk)
        sub = shades[bi * per_block:bi * per_block + rows]
        ax = fig.add_axes([
            left0 + bi * (label_w + block_w + gap_w) / fig_w,
            (foot_h + 0.34) / fig_h,
            block_w / fig_w,
            (row_h * rows) / fig_h,
        ])
        ax.imshow(np.ma.masked_invalid(sub), aspect="auto", cmap=cmap, norm=norm,
                  extent=(-0.5, len(years) - 0.5, rows - 0.5, -0.5),
                  interpolation="nearest")
        ax.set_yticks(np.arange(rows))
        ax.set_yticklabels([labels_all.get(c, c) for c in chunk], fontsize=5.9, color=_TEXT)
        ax.set_xticks(np.arange(len(years)))
        ax.set_xticklabels([str(y) for y in years], fontsize=5.9, color=_MUTED, rotation=90)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.tick_params(axis="x", length=0, pad=2)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        # 单元格之间不画白线：113 行 × 10 列全部描边会糊成一张网格，
        # 只留年份之间的竖向浅线，方便对齐年份
        for x in range(len(years) - 1):
            ax.axvline(x + 0.5, color="white", linewidth=0.35, alpha=0.55, zorder=3)
        if ncol_blocks > 1:
            fig.text(ax.get_position().x0, ax.get_position().y1 + 0.008,
                     f"{bi * per_block + 1}–{bi * per_block + rows}",
                     fontsize=6.4, color=_MUTED, ha="left", va="bottom")

    # 色标放在副标题之下、矩阵之上的那条空带里，右端与画布右沿对齐。
    # 原来钉在 1-1.06/fig_h（标题带正中）且宽度写死 2.10 英寸，
    # 副标题一长就压住末行文字，改成跟着 head_h 走。
    cbar_w = 1.70
    cax = fig.add_axes([(fig_w - 0.30 - cbar_w) / fig_w,
                        1 - (head_h - 0.28) / fig_h, cbar_w / fig_w, 0.011])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.4, colors=_MUTED, length=2)
    if formatter is not None:
        ticks = np.linspace(norm.vmin, norm.vmax, 5)
        cb.set_ticks(ticks)
        cb.ax.set_xticklabels([formatter(t, None) for t in ticks])
    cb.set_label(bar_label, fontsize=6.6, color=_MUTED, labelpad=2)

    fig.text(0.045, 1 - 0.30 / fig_h, title, fontsize=14.5, fontweight="bold",
             color=_TEXT, ha="left", va="top")
    _draw_subtitle(fig, subtitle, fig_w, fig_h, base=9.2, max_lines=4)
    _draw_footnote(fig, footnote, fig_w, fig_h)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path


def _compact_number(value: float) -> str:
    """色标刻度用紧凑写法：1.2万、3.4亿，避免一串零挤满色标。"""
    for cut, unit in ((1e8, "亿"), (1e4, "万")):
        if abs(value) >= cut:
            return f"{value / cut:,.1f}{unit}".replace(".0", "")
    if abs(value) >= 100:
        return f"{value:,.0f}"
    return f"{value:,.2f}".rstrip("0").rstrip(".")
