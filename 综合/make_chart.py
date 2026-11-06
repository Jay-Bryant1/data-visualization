"""法人单位数（按行业门类）的结构演变堆叠面积图。

数据源：年度数据/综合/法人单位数（按行业门类）.csv（19 个行业门类 × 2016–2024）
产出：年度数据/综合/法人单位数结构演变.png

代码在 统计局代码/综合/，与 年度数据/综合/ 镜像对应；数据目录自脚本位置上溯定位，
与当前工作目录无关。运行：
    python 统计局代码/综合/make_chart.py
    python 统计局代码/综合/make_chart.py --input <其它 CSV 路径>

解析与指标计算复用 stj_core；本文件只保留这一种「结构占比」视图的绘图逻辑，
逐指标的常规趋势图见 charts.py / build_all.py。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

# 共享核心（stj_core / charts）与本文件同在本分类目录，直接本地导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stj_core import StjTable, cagr, data_root, parse_stj_csv  # noqa: E402

CODE_DIR: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = data_root(CODE_DIR) / CODE_DIR.name   # 年度数据/综合
_TOTAL_LABEL: Final[str] = "法人单位数"

# ---------------------------------------------------------------- 分析层


def industry_panel(table: StjTable, total_label: str = _TOTAL_LABEL) -> tuple[pd.DataFrame, pd.Series]:
    """返回 (各行业绝对量矩阵, 逐年合计)。"""
    total = table.wide.loc[total_label]
    return table.wide.drop(index=total_label), total


def stacked_top_n(parts: pd.DataFrame, total: pd.Series, top_n: int = 7) -> pd.DataFrame:
    """按末年绝对量取前 top_n 个行业，其余合并为「其他行业」。

    合并而非全部铺开，是为了让每一层的厚度都还在可辨认区间内——
    20 个行业直接堆叠时，末位几层会薄到肉眼无法分辨。
    """
    ranked = parts[parts.columns[-1]].sort_values(ascending=False)
    keep = ranked.head(top_n).index
    other = parts.drop(index=keep).sum(axis=0).rename("其他行业")
    return pd.concat([parts.loc[keep], other.to_frame().T]).T  # 转置为「年份 × 行业」，与绘图口径一致


# ---------------------------------------------------------------- 绘图层

_COLORS: Final[list[str]] = [
    "#0F4C52",  # 占比第一
    "#2C7A7B",
    "#C97B3C",  # 制造业用暖橙，突出「份额逆势下滑」这一反差信息
    "#E4B363",
    "#5E8CA6",
    "#8FC0B6",
    "#7B8FA1",
    "#D5DBDB",  # 其他
]
_TEXT, _MUTED = "#132A31", "#5E7075"

# 行业全称过长会挤压右侧标签区，这里给等义短名（不改动数据）
_SHORT: Final[dict[str, str]] = {
    "批发和零售业法人单位数": "批发和零售业",
    "租赁和商务服务业法人单位数": "租赁和商务服务业",
    "制造业法人单位数": "制造业",
    "建筑业法人单位数": "建筑业",
    "科学研究和技术服务业法人单位数": "科研和技术服务业",
    "农、林、牧、渔业法人单位数": "农林牧渔业",
    "信息传输、软件和信息技术服务业法人单位数": "信息传输/软件服务",
}


def _spread_labels(targets: list[float], min_gap: float) -> list[float]:
    """把右侧标签的纵坐标按最小间距下推，避免窄层标签互相覆盖。

    采用「自下而上的单遍推移」而非等距排布：被挪动的只有冲突的那一层，
    其余标签仍锁在所属色带的中点上，指向关系不至于失真。

    Args:
        targets: 各层标签的理想纵坐标（升序）。
        min_gap: 相邻标签的最小间距（与纵轴同单位）。

    Returns:
        调整后的纵坐标列表，顺序与输入一致。
    """
    adjusted = list(targets)
    for _ in range(3):  # 反复几遍，处理一次推移后又产生的新冲突
        for i in range(1, len(adjusted)):
            if adjusted[i] - adjusted[i - 1] < min_gap:
                adjusted[i] = adjusted[i - 1] + min_gap
    return adjusted


def plot_structure_area(stacked: pd.DataFrame, total: pd.Series, out_path: Path) -> Path:
    """绘制堆叠面积图，右侧直接标注各层名称与末年占比。

    Args:
        stacked: index=年份（升序），columns=行业名，值为绝对量。
        total: 逐年合计，用于定位 2023 年口径性回落。
        out_path: PNG 输出路径。

    Returns:
        写出的文件路径。
    """
    mpl.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    years = stacked.index.to_list()
    names = list(stacked.columns)
    y_max = float(total.max())
    # 轴顶留白两成：给 2023 年的口径说明腾出位置，同时不压缩色带高度
    y_top = y_max * 1.16

    fig, ax = plt.subplots(figsize=(12.4, 6.6), dpi=200)
    fig.subplots_adjust(left=0.062, right=0.775, top=0.795, bottom=0.135)

    ax.stackplot(
        years,
        *[stacked[n].to_numpy() for n in names],
        colors=_COLORS[: len(names)],
        edgecolor="white",
        linewidth=0.7,
    )

    # 右侧直接标注：比图例少一次视线往返；窄层用推移算法防止重叠
    last, last_vals = years[-1], stacked.iloc[-1]
    cum = last_vals.cumsum()
    mids = [cum[n] - last_vals[n] / 2 for n in names]
    shares = [last_vals[n] / last_vals.sum() * 100 for n in names]
    placed = _spread_labels(mids, min_gap=y_top * 0.034)

    for name, y_mid, y_txt, share, color in zip(names, mids, placed, shares, _COLORS):
        if abs(y_txt - y_mid) > 1:
            # 标签被推离色带中心时，补一条细引线，保留指向关系
            ax.plot(
                [last, last + 0.16], [y_mid, y_txt],
                color="#B0B0B0", linewidth=0.7, zorder=3, clip_on=False,
            )
        ax.text(
            last + 0.2,
            y_txt,
            f"{_SHORT.get(name, name)}  {share:.1f}%",
            va="center",
            ha="left",
            fontsize=8.6,
            color=color if name != "其他行业" else _MUTED,
            fontweight="bold" if share >= 10 else "normal",
        )

    # 2023 年回落：说明文字右对齐到回落点左侧，引线极短，不会横穿图面。
    # 仅在数据确实含 2022、2023 两年时才标注，换用别的源文件也不会因此报错。
    dip = 2023 if {2022, 2023} <= set(total.index) else None
    if dip is not None:
        ax.annotate(
            f"{dip} 年 {(total[dip] / total[dip - 1] - 1) * 100:+.1f}%，普查后修订的口径性回落",
            xy=(dip, total[dip]),
            xytext=(dip - 0.35, y_top * 0.985),
            fontsize=8.4,
            color=_MUTED,
            ha="right",
            va="top",
            arrowprops={
                "arrowstyle": "-",
                "color": "#B0B0B0",
                "linewidth": 0.8,
                "shrinkA": 4,
                "shrinkB": 4,
                "connectionstyle": "arc3,rad=0",
            },
        )

    ax.set_xlim(min(years) - 0.15, last + 2.05)
    ax.set_ylim(0, y_top)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=9, color=_MUTED)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e4:,.0f}"))
    ax.tick_params(axis="y", labelsize=9, colors=_MUTED, length=0)
    ax.tick_params(axis="x", length=3, color="#BFBFBF")
    ax.set_ylabel("法人单位数（万个）", fontsize=9.5, color=_MUTED, labelpad=8)

    # 网格只铺到 2024，避免横线穿进右侧标签区
    for y in ax.get_yticks():
        if 0 < y <= y_top:
            ax.hlines(y, min(years) - 0.15, last, color="#E6E6E6", linewidth=0.9, zorder=0)

    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#BFBFBF")
    ax.spines["bottom"].set_color("#BFBFBF")
    ax.spines["bottom"].set_bounds(min(years) - 0.15, last)

    y0, y1 = years[0], years[-1]
    share_m = {n: stacked[n] / total * 100 for n in names}
    fig.text(
        0.062, 0.938,
        "法人单位数八年翻倍，增量集中在批发零售与商务服务",
        fontsize=16.5, fontweight="bold", color=_TEXT, ha="left",
    )
    fig.text(
        0.062, 0.876,
        f"全国法人单位数 {y0}–{y1} 年：{total[y0] / 1e4:,.0f} 万 → {total[y1] / 1e4:,.0f} 万"
        f"（累计 {total[y1] / total[y0] - 1:+.1%}，CAGR {cagr(total[y0], total[y1], y1 - y0):.1%}）"
        f"　|　制造业份额 {share_m[names[2]][y0]:.1f}% → {share_m[names[2]][y1]:.1f}%，降幅居首",
        fontsize=10.2, color=_MUTED, ha="left",
    )
    fig.text(
        0.062, 0.035,
        "数据来源：国家统计局 · 年度数据（按行业分法人单位数）。右侧仅列示 2024 年占比最高的 7 个行业，"
        "其余 12 个门类合并为「其他行业」；2004 年起不含国际组织；2018、2023 年数据经经济普查后修订。",
        fontsize=8, color="#8A8A8A", ha="left", va="bottom",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------- 入口


def main() -> None:
    parser = argparse.ArgumentParser(description="生成行业结构演变堆叠面积图")
    parser.add_argument("--input", default=str(DATA_DIR / "法人单位数（按行业门类）.csv"), help="源 CSV 路径")
    parser.add_argument("--output", default=str(DATA_DIR / "法人单位数结构演变.png"), help="输出 PNG 路径")
    parser.add_argument("--top-n", type=int, default=7, help="单独成层的行业数，其余并入「其他」")
    args = parser.parse_args()

    table = parse_stj_csv(args.input)
    parts, total = industry_panel(table)
    stacked = stacked_top_n(parts, total, top_n=args.top_n)
    path = plot_structure_area(stacked, total, Path(args.output))
    print(f"已生成：{path.resolve()}")


if __name__ == "__main__":
    main()
