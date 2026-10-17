# -*- coding: utf-8 -*-
r"""
《商业数据分析》教材重构 —— 第1章 数据分析思维
================================================================================
原教材实现: Power BI / Excel (.pbix + .xlsx), 本章无 Python / ipynb 内容
本文件实现: Python 单文件版 (pandas + matplotlib + plotly), 可直接运行
================================================================================

运行方式
--------
    C:\Users\dell\anaconda3\envs\dataviz\python.exe -X utf8 ch01_数据分析思维.py

依赖: pandas / numpy / matplotlib / openpyxl (必需), plotly (可选, 用于交互图)
输出: figures/ 下的 PNG 图片 (300dpi), 及可选交互式 HTML

本章两个演示
------------
演示一  关系模型    银行客户与交易数据 (8 张表, 含 105.6 万行交易流水)
                   原 Power BI 报告 2 页视觉: columnChart + stackedAreaChart
演示二  维度模型    商场销售数据表 (1 事实表 + 5 维度表, 雪花型)
                   原 Power BI 报告 2 页视觉: areaChart / lineChart /
                   multiRowCard / keyDriversVisual

迁移原则
--------
不是把 Power BI 的拖拽步骤翻译成代码, 而是还原其业务逻辑与分析口径,
并把原教材用视觉对象"演示"出的观点, 用数据量化出来。

四个可验证的教学点
------------------
1. 全连接产生"空白"组   —— 5,369 名客户中仅 892 人持卡, Power BI 默认左外连接
   生成占 83.4% 的"(空白)"组, 把真实分布在视觉上完全压扁。
2. 关系关联导致重复计数 —— 869 个共管账户使 145 笔贷款被重复归集, 贷款总额从
   1.033 亿虚增到 1.255 亿 (+21.6%)。
3. 维度键不一致         —— 事实表产品编号 a001 与维表 A001 直接关联匹配 0 行,
   数据静默全部丢失。
4. 日期维度是时间智能的前提 —— 独立日期表才能支撑同比、累计与层级下钻。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator

# ============================================================================ #
# 0. 路径与运行环境
# ============================================================================ #

HERE = Path(__file__).resolve().parent      # 本章交付目录
ROOT = HERE.parent                          # 项目根目录 (教材原文所在位置)
CH1_DIR = ROOT / "第1章 数据分析思维"                          # 教材原文目录
RAW_REL = CH1_DIR / "关系模型演示-不建议用poweBI操作关系型数据模型"
RAW_XLSX = CH1_DIR / "维度模型演示" / "商场销售数据表.xlsx"
RAW_SALE = CH1_DIR / "sale.csv"                               # 章节根目录的补充数据
FIG = HERE / "figures"                        # 图片输出目录
FIG.mkdir(parents=True, exist_ok=True)

#: 是否额外生成 Plotly 交互式 HTML。默认关闭, 只产出 PNG 图片;
#: 需要网页版可交互图表时, 改成 True 即可 (需已安装 plotly)。
MAKE_INTERACTIVE_HTML = False

# conda 环境未激活时, matplotlib 依赖的 DLL 位于 <env>\Library\bin, 不在默认
# 搜索路径中, 导出 PNG 时会直接崩溃 (Windows fatal exception 0xc06d007f)。
# 必须在任何绘图动作之前补上该目录。
if sys.platform == "win32":
    _lib_bin = Path(sys.prefix) / "Library" / "bin"
    if _lib_bin.is_dir() and str(_lib_bin).lower() not in os.environ.get("PATH", "").lower():
        os.environ["PATH"] = f"{_lib_bin}{os.pathsep}{os.environ.get('PATH', '')}"


def rule(title: str) -> None:
    """打印分节标题, 便于对照教材章节阅读运行日志。"""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
# ============================================================================ #
# 1. 绘图主题 (内置, 无需外部模块)
# ============================================================================ #

#: 主色板 (分类数据依次取用)
PALETTE = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1",
           "#76B7B2", "#EDC948", "#9C755F", "#BAB0AC", "#FF9DA7"]

#: 语义色 —— 按业务含义取色, 而非装饰
COLOR = {
    "primary": "#4E79A7", "accent": "#F28E2B", "positive": "#59A14F",
    "negative": "#E15759", "neutral": "#BAB0AC", "highlight": "#B07AA1",
    "muted": "#8FAFC9", "grid": "#E4E7EB", "text": "#2F3B45",
    "text_soft": "#7A8794", "bg": "#FFFFFF", "bg_soft": "#F7F9FB",
}

SEQ_CMAP = LinearSegmentedColormap.from_list(
    "biz_seq", ["#EAF1F8", "#A9C4DF", "#6E97C2", "#4E79A7", "#2C4A66"])
matplotlib.colormaps.register(SEQ_CMAP, force=True)

_CJK_CANDIDATES = ["Microsoft YaHei", "微软雅黑", "Source Han Sans SC",
                   "Noto Sans CJK SC", "SimHei", "黑体", "DengXian", "SimSun"]


def cjk_font_name(fallback: str = "DejaVu Sans") -> str:
    """返回本机可用的第一个中文字体名。"""
    try:
        from matplotlib import font_manager
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in _CJK_CANDIDATES:
            if name in available:
                return name
    except Exception:
        pass
    return fallback


def setup_style(dpi: int = 300, figsize: tuple[float, float] = (11, 6.2)) -> str:
    """应用全局绘图主题: 中文正体 / 语义配色 / 去噪坐标轴 / 高清导出。"""
    font = cjk_font_name()
    plt.rcParams.update({
        "font.sans-serif": [font, "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "figure.dpi": 110, "savefig.dpi": dpi, "figure.figsize": figsize,
        "figure.facecolor": COLOR["bg"], "savefig.facecolor": COLOR["bg"],
        "savefig.bbox": "tight", "savefig.pad_inches": 0.18,
        "font.size": 11, "axes.titlesize": 15, "axes.titleweight": "bold",
        "axes.labelsize": 12, "axes.labelcolor": COLOR["text"],
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10.5,
        "figure.titlesize": 17, "figure.titleweight": "bold",
        "axes.edgecolor": COLOR["grid"], "axes.linewidth": 1.0, "axes.labelpad": 8,
        "axes.titlepad": 12, "axes.titlelocation": "left",
        "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "grid.color": COLOR["grid"], "grid.linewidth": 0.8, "grid.alpha": 0.9,
        "xtick.color": COLOR["text_soft"], "ytick.color": COLOR["text_soft"],
        "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.handlelength": 1.4,
        "legend.borderaxespad": 0.0,
        "lines.linewidth": 2.2, "lines.markersize": 6, "patch.linewidth": 0,
        "image.cmap": "biz_seq",
    })
    return font


def style_axes_clean(ax) -> None:
    """淡化左/下轴线, 让网格线承担视觉引导。"""
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR["grid"])


# ------------------------------ 数值格式化 -------------------------------- #

def thousands(x, _pos=None) -> str:
    """千分位 / 万 / 亿 自适应。"""
    if abs(x) >= 1e8:
        return f"{x / 1e8:,.1f}亿"
    if abs(x) >= 1e4:
        return f"{x / 1e4:,.0f}万"
    return f"{x:,.0f}"


def percent(x, _pos=None) -> str:
    return f"{x:.0%}"


def currency(x, _pos=None) -> str:
    """人民币金额, 自动切换 元 / 万 / 亿。"""
    if abs(x) >= 1e8:
        return f"¥{x / 1e8:,.1f}亿"
    if abs(x) >= 1e4:
        return f"¥{x / 1e4:,.0f}万"
    return f"¥{x:,.0f}"


_FORMATTERS = {"thousands": thousands, "percent": percent,
               "currency": currency, None: None}


def apply_axis_format(ax, y="thousands", x=None) -> None:
    """给坐标轴套格式化函数: thousands / percent / currency / None。

    注意: 类别轴上绝不能套用数值刻度定位器, 否则类别名会被 0,1,2... 覆盖,
    因此类别轴必须显式传 y=None 或 x=None。
    """
    if _FORMATTERS.get(y) is not None:
        ax.yaxis.set_major_formatter(FuncFormatter(_FORMATTERS[y]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    if _FORMATTERS.get(x) is not None:
        ax.xaxis.set_major_formatter(FuncFormatter(_FORMATTERS[x]))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
# ------------------------------- 绘图工具 --------------------------------- #

def annotate_bars(ax, values=None, bars=None, fmt="{:,.0f}", offset=0.015,
                  inside=False, color=None, fontsize=10, fontweight="bold"):
    """在柱状图上标注数值。

    注意: values 必须与柱高使用同一单位, 否则标签会落在错误高度。
    """
    bars = list(bars) if bars is not None else list(ax.patches)
    if values is None:
        values = [b.get_height() for b in bars]
    values = [float(v) for v in values]
    finite = [v for v in values if np.isfinite(v)] or [0.0]
    span = (max(finite) - min(finite + [0.0])) or 1.0
    for bar, val in zip(bars, values):
        if not np.isfinite(val):
            continue
        x = bar.get_x() + bar.get_width() / 2
        if inside:
            ax.text(x, bar.get_y() + val / 2, fmt.format(val), ha="center",
                    va="center", fontsize=fontsize, color="white",
                    fontweight=fontweight, clip_on=False)
        else:
            ax.text(x, val + span * offset, fmt.format(val), ha="center",
                    va="bottom", fontsize=fontsize,
                    color=color or COLOR["text_soft"],
                    fontweight=fontweight, clip_on=False)


def annotate_hbars(ax, values, fmt="{:,.0f}", pad=0.012, color=None,
                   fontsize=10, fontweight="bold"):
    """在横向条形图右侧标注数值。"""
    values = [float(v) for v in values]
    finite = [v for v in values if np.isfinite(v)] or [0.0]
    span = (max(finite) - min(finite + [0.0])) or 1.0
    for bar, val in zip(ax.patches[-len(values):], values):
        if not np.isfinite(val):
            continue
        ax.text(val + span * pad, bar.get_y() + bar.get_height() / 2,
                fmt.format(val), va="center", ha="left", fontsize=fontsize,
                color=color or COLOR["text"], fontweight=fontweight)


def legend_above(ax, ncol=2, y=1.02, **kwargs):
    """把图例放到坐标区上方, 避免遮挡数据。"""
    return ax.legend(loc="lower left", bbox_to_anchor=(0, y), ncol=ncol,
                     borderaxespad=0, **kwargs)


def add_source(fig, text: str, y: float = 0.0) -> None:
    """在图片底部添加数据来源注释 (教材图必备)。"""
    fig.text(0.005, y, text, fontsize=9, color=COLOR["text_soft"],
             ha="left", va="top")


def save_fig(fig, name: str, out_dir: Path = FIG,
             formats: tuple[str, ...] = ("png",)) -> list[Path]:
    """保存图片, 默认导出 PNG(300dpi)。需要矢量图时可传 formats=("png", "pdf")。"""
    written = []
    for ext in formats:
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, format=ext)
        written.append(p)
    plt.close(fig)
    return written


def save_html(fig, name: str, out_dir: Path = FIG) -> Path | None:
    """导出 Plotly 交互图为独立 HTML。"""
    try:
        import plotly.io as pio
    except Exception:
        return None
    p = out_dir / f"{name}.html"
    pio.write_html(fig, p, include_plotlyjs="cdn", full_html=True,
                   config={"displaylogo": False, "locale": "zh-CN"})
    return p


def apply_plotly_template() -> bool:
    """注册与 matplotlib 一致的 Plotly 模板; plotly 不可用时返回 False。"""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        from plotly.colors import sample_colorscale
    except Exception:
        return False
    seq = sample_colorscale([[0, "#EAF1F8"], [.25, "#A9C4DF"], [.55, "#6E97C2"],
                             [.8, "#4E79A7"], [1, "#2C4A66"]], [0, .25, .5, .75, 1])
    font = cjk_font_name()
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        font=dict(family=f"{font}, DejaVu Sans", size=13, color=COLOR["text"]),
        title=dict(font=dict(size=18, color=COLOR["text"]), x=0.02, xanchor="left"),
        colorway=PALETTE, paper_bgcolor=COLOR["bg"], plot_bgcolor=COLOR["bg"],
        colorscale=dict(sequential=seq),
        margin=dict(l=75, r=45, t=85, b=65),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=COLOR["grid"],
                   tickfont=dict(size=12, color=COLOR["text_soft"])),
        yaxis=dict(showgrid=True, gridcolor=COLOR["grid"], gridwidth=1,
                   zeroline=False, linecolor=COLOR["grid"], ticks="",
                   tickfont=dict(size=12, color=COLOR["text_soft"])),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right",
                    x=1, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", bordercolor=COLOR["grid"], font_size=12),
    )
    pio.templates["biz_clean"] = tpl
    pio.templates.default = "biz_clean"
    return True
# ============================================================================ #
# 2. 演示一: 关系模型 (银行客户与交易数据)
# ============================================================================ #

# 8 张表的业务含义 (依据教材 PPT《数据库介绍》)
TABLE_META = {
    "district": ("地区人口统计表", "district_id", "地区的人口、经济、犯罪率等宏观统计"),
    "clients": ("客户信息表", "client_id", "客户性别、出生日期、所属地区"),
    "accounts": ("账户表", "account_id", "账户开户地区、日期、结算频度"),
    "disp": ("权限分配表", "disp_id", "客户与账户的权限关系(关系模型的枢纽)"),
    "card": ("信用卡表", "card_id", "账户上发行的信用卡及卡类型"),
    "loans": ("贷款表", "loan_id", "账户的贷款金额、期限、还款状态"),
    "order": ("支付订单表", "order_id", "账户发起的转账指令"),
    "trans": ("交易表", "trans_id", "账户的每一笔交易流水"),
}


def load_relational() -> dict:
    """加载 8 张关系表。

    教材 CSV 为 GBK 编码; 交易流水表的金额含千分位逗号 (如 "1,000"),
    需先去逗号再转数值; 该表 60MB / 105.6 万行, 只载入分析所需列。
    """
    def read(name, **kw):
        return pd.read_csv(RAW_REL / f"{name}.csv", encoding="gbk", **kw)

    T = {n: read(n) for n in ["district", "clients", "accounts", "disp",
                              "card", "loans", "order"]}
    T["trans"] = read("trans", usecols=["trans_id", "account_id", "date", "type",
                                        "operation", "amount", "balance"])
    for col in ("amount", "balance"):
        T["trans"][col] = pd.to_numeric(
            T["trans"][col].astype(str).str.replace(",", "", regex=False),
            errors="coerce")
    return T


def draw_er_diagram(T: dict, n_shared: int) -> None:
    """图 1: 8 张表的 E-R 实体关系图。

    布局要点: accounts 有三张从属表 (loans / order / trans), 若与它直接连三条
    斜线, 连线会穿过中间的实体框。这里用一条竖向导流线 + 三条水平支线替代,
    保证任何连线都不与实体框相交。
    """
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    W, H = 0.21, 0.105
    nodes = {
        "district": (0.500, 0.905, False),
        "clients":  (0.135, 0.605, False),
        "disp":     (0.460, 0.605, True),        # 枢纽表, 高亮
        "accounts": (0.800, 0.605, False),
        "card":     (0.135, 0.265, False),
        "loans":    (0.650, 0.415, False),
        "order":    (0.650, 0.250, False),
        "trans":    (0.650, 0.085, False),
    }

    def line(p1, p2, rad=0.0):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-", lw=1.8,
                                     color=COLOR["neutral"],
                                     connectionstyle=f"arc3,rad={rad}", zorder=1))

    def badge(x, y, text):
        ax.text(x, y, text, fontsize=10, color=COLOR["accent"], fontweight="bold",
                ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.26", fc="white",
                          ec=COLOR["accent"], lw=1.0, alpha=0.97))

    def hlink(a, b, label):
        axx, ayy, _ = nodes[a]
        bxx, byy, _ = nodes[b]
        sgn = np.sign(bxx - axx)
        x1, x2 = axx + W / 2 * sgn, bxx - W / 2 * sgn
        line((x1, ayy), (x2, byy))
        badge((x1 + x2) / 2, ayy, label)

    # --- 主干连接 ---
    line((0.500, 0.905 - H / 2), (0.135, 0.605 + H / 2), rad=0.10)
    badge(0.305, 0.775, "1 : N")
    line((0.500, 0.905 - H / 2), (0.800, 0.605 + H / 2), rad=-0.10)
    badge(0.695, 0.775, "1 : N")
    hlink("clients", "disp", "1 : 1")
    hlink("disp", "accounts", "1 : N")
    line((0.135, 0.605 - H / 2), (0.135, 0.265 + H / 2))
    badge(0.135, 0.435, "1 : 0..1")

    # --- accounts 的从属表: 竖向导流线 + 水平支线 ---
    spine_x = 0.870
    line((spine_x, 0.605 - H / 2), (spine_x, 0.085))
    for nid, lab in [("loans", "1 : 0..1"), ("order", "1 : 0..N"),
                     ("trans", "1 : 0..N")]:
        ny = nodes[nid][1]
        line((spine_x, ny), (nodes[nid][0] + W / 2, ny))
        badge((spine_x + nodes[nid][0] + W / 2) / 2, ny, lab)
    badge(spine_x + 0.030, 0.505, "账户的从属表")

    # --- 实体框 (三行文字用点位偏移, 行距不随画布尺寸变化) ---
    for nid, (x, y, hub) in nodes.items():
        name, pk, _ = TABLE_META[nid]
        ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H,
                                    boxstyle="round,pad=0.008,rounding_size=0.022",
                                    fc="#FDEDD8" if hub else "#EAF1F8",
                                    ec=COLOR["accent"] if hub else COLOR["primary"],
                                    lw=2.0, zorder=3))
        ax.annotate(nid, (x, y), textcoords="offset points", xytext=(0, 13),
                    ha="center", va="center", zorder=4, fontsize=13.5,
                    fontweight="bold", color=COLOR["text"], family="DejaVu Sans")
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, -4),
                    ha="center", va="center", zorder=4, fontsize=11.5,
                    color=COLOR["text"])
        ax.annotate(f"{len(T[nid]):,} 行  |  主键 {pk}", (x, y),
                    textcoords="offset points", xytext=(0, -22),
                    ha="center", va="center", zorder=4, fontsize=9.5,
                    color=COLOR["text_soft"])

    ax.text(0.045, 0.185,
            "说明\n"
            "· disp 是关系模型的枢纽: 客户与账户之间 1:1..N 的桥梁\n"
            f"· {n_shared} 个账户为多人共管, 是后续贷款金额重复计数的根源\n"
            "· 实体框标注: 表名 / 中文名 / 记录数 / 主键",
            fontsize=10.5, color=COLOR["text"], ha="left", va="top", linespacing=1.9,
            bbox=dict(boxstyle="round,pad=0.5", fc=COLOR["bg_soft"],
                      ec=COLOR["grid"], lw=1.0))
    ax.set_title("银行客户与交易数据 —— E-R 实体关系图 (共 8 张表)", pad=20)
    add_source(fig, "数据来源: 教材《数据库介绍》配套数据 (PKDD'99 银行客户与交易数据)  "
                    "|  橙色为关系枢纽表 disp", y=0.012)
    save_fig(fig, "ch01_01_ER图_银行数据")

def draw_blank_group(T: dict, ct: pd.DataFrame, pb_counts: pd.Series,
                     blank_cnt: int) -> None:
    """图 2: 教学点一 —— 全连接产生的「空白」组。

    左图复现 Power BI 的默认行为 (clients 左外连接 card), 右图给出正确口径。
    """
    n_clients = len(T["clients"])
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6), width_ratios=[1, 1.35])

    # 左: Power BI 默认全连接的结果
    ax = axes[0]
    order = ["(空白)", "普通卡", "青年卡", "金卡"]
    vals = [int(pb_counts.get(k, 0)) for k in order]
    colors = [COLOR["muted"]] + [COLOR["primary"]] * 3
    bars = ax.bar(order, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:,}",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color=COLOR["negative"] if v == blank_cnt else COLOR["text"])
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title("Power BI 默认全连接的结果", color=COLOR["negative"])
    ax.set_ylabel("客户数")
    ax.set_xlabel("信用卡类型")
    ax.text(0.97, 0.97,
            f"(空白) 组占 {blank_cnt / n_clients:.0%}\n"
            f"因果链: {n_clients:,} 名客户 → 仅 892 人持卡",
            transform=ax.transAxes, ha="right", va="top", fontsize=10.5,
            color=COLOR["negative"],
            bbox=dict(boxstyle="round,pad=0.45", fc="#FDECEC",
                      ec=COLOR["negative"], lw=1.2))
    style_axes_clean(ax)

    # 右: 正确口径 (内连接, 仅持卡客户)
    ax = axes[1]
    x = np.arange(len(ct.index))
    w = 0.36
    sex_color = {"女": COLOR["highlight"], "男": COLOR["primary"]}
    for i, s in enumerate(ct.columns):
        offset = (i - (len(ct.columns) - 1) / 2) * w
        bars = ax.bar(x + offset, ct[s].values, width=w, label=s,
                      color=sex_color.get(s, PALETTE[i]))
        annotate_bars(ax, ct[s].values, bars=bars, fmt="{:,.0f}", offset=0.02)
    ax.set_xticks(x)
    ax.set_xticklabels(ct.index)
    ax.set_ylim(0, ct.values.max() * 1.20)
    ax.set_title(f"正确口径: 仅统计持卡客户  (n = {int(ct.values.sum()):,})",
                 color=COLOR["positive"])
    ax.set_ylabel("持卡客户数")
    ax.set_xlabel("信用卡类型")
    ax.legend(loc="upper right", ncol=2)
    style_axes_clean(ax)

    fig.suptitle("教学点一: 全连接产生的「空白」组会掩盖真实分布",
                 y=1.03, x=0.02, ha="left")
    add_source(fig, "数据来源: clients.csv / disp.csv / card.csv  |  "
                    "左图为 Power BI 拖拽聚合的默认行为, 右图为正确口径", y=0.005)
    save_fig(fig, "ch01_02_教学点1_空白组陷阱")


def demo_relational() -> dict:
    """演示一: 关系模型 —— Power BI -> Python 迁移。"""
    rule("演示一 · 1. 数据加载 (教材数据为 GBK 编码)")
    T = load_relational()
    scale = pd.DataFrame({
        "表名": [TABLE_META[n][0] for n in TABLE_META],
        "英文名": list(TABLE_META),
        "主键": [TABLE_META[n][1] for n in TABLE_META],
        "记录数": [len(T[n]) for n in TABLE_META],
        "字段数": [T[n].shape[1] for n in TABLE_META],
        "业务含义": [TABLE_META[n][2] for n in TABLE_META],
    })
    print(scale.to_string(index=False))

    # ---------------- 关系基数检验 ---------------- #
    rule("演示一 · 2. 关系基数检验 (关系分析的前提是搞清基数)")
    disp, clients, accounts = T["disp"], T["clients"], T["accounts"]
    card, loans = T["card"], T["loans"]
    n_shared = int((disp.groupby("account_id").size() > 1).sum())

    cardinality = pd.DataFrame([
        ("clients -> disp", "客户 与 权限分配", "1 : 1",
         f"{clients.client_id.nunique():,} 名客户, 每人恰好 1 条权限记录"),
        ("accounts -> disp", "账户 与 权限分配", "1 : 1..N",
         f"{len(accounts):,} 个账户, 其中 {n_shared} 个为多人共管账户 (最多 2 人)"),
        ("disp -> card", "权限分配 与 信用卡", "1 : 0..1",
         f"{len(card):,} 张卡, 仅覆盖 {len(card) / len(disp):.1%} 的权限记录"),
        ("accounts -> loans", "账户 与 贷款", "1 : 0..1",
         f"{len(loans):,} 笔贷款, 一个账户至多 1 笔"),
        ("accounts -> order", "账户 与 支付订单", "1 : 0..N",
         f"{len(T['order']):,} 条指令, 涉及 {T['order'].account_id.nunique():,} 个账户"),
        ("accounts -> trans", "账户 与 交易流水", "1 : 0..N",
         f"{len(T['trans']):,} 笔交易, 涉及 {T['trans'].account_id.nunique():,} 个账户"),
    ], columns=["关系", "业务含义", "基数", "实测说明"])
    print(cardinality.to_string(index=False))

    rule("演示一 · 3. 绘制 E-R 实体关系图")
    draw_er_diagram(T, n_shared)

    # ---------------- 教学点一: 空白组 ---------------- #
    rule("演示一 · 4. 迁移 Power BI 视觉 1: 信用卡类型 x 持卡人性别 (columnChart)")
    card_named = card.rename(columns={"type": "卡类型"})
    disp_named = disp.rename(columns={"type": "权限类型"})

    # Power BI 的默认行为: clients 左外连接 card
    pb_style = (clients.merge(disp_named, on="client_id", how="left")
                        .merge(card_named, on="disp_id", how="left"))
    pb_style["卡类型"] = pb_style["卡类型"].fillna("(空白)")
    blank_cnt = int((pb_style["卡类型"] == "(空白)").sum())
    pb_counts = pb_style["卡类型"].value_counts()
    print("\n[Power BI 默认全连接] 卡类型分组计数 (注意 (空白) 组):")
    print(pb_counts.to_string())
    print(f"-> (空白) 占比 {blank_cnt / len(pb_style):.1%}, 真实分布被严重压扁")

    # 正确做法: 内连接, 只统计真正持卡的客户
    correct = (card_named.merge(disp_named, on="disp_id")
                         .merge(clients, on="client_id"))
    ct = pd.crosstab(correct["卡类型"], correct["sex"]).reindex(
        ["普通卡", "金卡", "青年卡"])
    print("\n[正确做法] 信用卡类型 x 性别 交叉表:")
    print(ct.to_string())
    print(f"-> 有效样本 {int(ct.values.sum()):,} 人, "
          f"占全体客户 {ct.values.sum() / len(clients):.1%}")

    draw_blank_group(T, ct, pb_counts, blank_cnt)
    # ---------------- 教学点二: 重复计数 ---------------- #
    rule("演示一 · 5. 迁移 Power BI 视觉 2: 每年新增贷款金额 x 性别 (stackedAreaChart)")
    loan_clients = pd.DataFrame({"account_id": disp["account_id"],
                                 "client_id": disp["client_id"]}).merge(
        clients[["client_id", "sex"]], on="client_id")

    # 错误做法: 贷款表直接左连接权限表 (共管账户的金额被重复展开)
    wrong = (loans.merge(loan_clients, on="account_id", how="left")
                  .assign(年=lambda d: pd.to_datetime(d["date"]).dt.year))
    # 正确做法: 共管账户的贷款金额按持有人均摊
    n_holders = disp.groupby("account_id").size().rename("持有人数")
    right = wrong.merge(n_holders, on="account_id", how="left")
    right["分摊金额"] = right["amount"] / right["持有人数"]

    w_year = wrong.pivot_table(index="年", columns="sex", values="amount",
                               aggfunc="sum")
    r_year = right.pivot_table(index="年", columns="sex", values="分摊金额",
                               aggfunc="sum")
    print(f"\n贷款总笔数        : {len(loans):,}")
    print(f"经 disp 展开后行数: {len(wrong):,}  "
          f"(膨胀 {len(wrong) - len(loans):,} 行, 均为共管账户)")
    print(f"贷款金额 · 直接展开: {wrong.amount.sum():>14,.0f}  "
          f"(虚增 {wrong.amount.sum() / loans.amount.sum() - 1:+.1%})")
    print(f"贷款金额 · 均摊后  : {right['分摊金额'].sum():>14,.0f}  (与真实总额一致)")

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.1), width_ratios=[1, 1, 0.80],
                             gridspec_kw={"wspace": 0.32})
    years = w_year.index.astype(int)
    sex_color = {"女": COLOR["highlight"], "男": COLOR["primary"]}
    # 两图共用纵轴刻度, 差额才具有可比性
    y_top = max(w_year.values.sum(axis=1).max(),
                r_year.values.sum(axis=1).max()) * 1.30
    for ax, data, title, tag in [
            (axes[0], w_year, "错误口径: 直接左连接 (金额重复)", COLOR["negative"]),
            (axes[1], r_year, "正确口径: 按持有人均摊", COLOR["positive"])]:
        ax.stackplot(years, [data["女"].values, data["男"].values],
                     labels=["女", "男"],
                     colors=[sex_color["女"], sex_color["男"]], alpha=0.88)
        ax.set_title(title, color=tag)
        ax.set_xlabel("贷款发放年份")
        ax.set_ylabel("新增贷款金额")
        ax.set_xticks(years)
        ax.set_ylim(0, y_top)
        apply_axis_format(ax, y="currency")
        ax.legend(loc="upper left", ncol=2)
        style_axes_clean(ax)
        ax.annotate(f"合计 {data.values.sum():,.0f} 元", xy=(0.97, 0.93),
                    xycoords="axes fraction", ha="right", fontsize=10.5,
                    color=COLOR["text"], fontweight="bold")

    ax = axes[2]
    diff = (w_year - r_year).sum(axis=1)
    # 数值 6-7 位, 横向条形图能给出充足的标签空间, 避免文字互相重叠
    diff_w = (diff / 1e4).round(1)
    ax.barh([str(y) for y in years], diff_w.values,
            color=COLOR["negative"], height=0.6)
    annotate_hbars(ax, diff_w.values, fmt="{:.1f}万", pad=0.015, fontsize=9.5)
    ax.set_xlim(0, diff_w.max() * 1.32)
    ax.set_title("虚增金额 = 错误 − 正确", color=COLOR["negative"])
    ax.set_xlabel("虚增金额 (万元)")
    ax.set_ylabel("贷款发放年份")
    ax.grid(axis="y", visible=False)
    style_axes_clean(ax)

    fig.suptitle(f"教学点二: 关系模型关联导致贷款金额重复计数 "
                 f"({wrong.amount.sum() / loans.amount.sum() - 1:+.1%})",
                 y=1.04, x=0.02, ha="left")
    add_source(fig, f"数据来源: loans.csv / disp.csv / clients.csv  |  "
                    f"共管账户({n_shared} 个)使 {len(wrong) - len(loans)} 笔贷款被重复归集",
               y=0.005)
    save_fig(fig, "ch01_03_教学点2_重复计数陷阱")

    # ---------------- 交易表全量分析 ---------------- #
    rule("演示一 · 6. 交易表 (105.6 万行) 全量分析")
    tr = T["trans"].copy()
    tr["年"] = pd.to_datetime(tr["date"]).dt.year
    by_year = tr.groupby("年").agg(交易笔数=("trans_id", "count"),
                                   交易金额=("amount", "sum"),
                                   活跃账户=("account_id", "nunique"))
    print(by_year.to_string())
    ops = tr["operation"].value_counts()
    print("\n交易类型分布:")
    print(ops.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.0), width_ratios=[1.15, 1.0, 0.82],
                             gridspec_kw={"wspace": 0.30})
    years_t = by_year.index.astype(int)

    ax = axes[0]
    vals_w = by_year["交易笔数"].values / 1e4          # 统一换算为"万笔"
    bars = ax.bar(years_t, vals_w, color=COLOR["primary"], width=0.62)
    annotate_bars(ax, vals_w, bars=bars, fmt="{:.1f}万", offset=0.025)
    ax.set_ylim(0, vals_w.max() * 1.18)
    ax.set_title("交易笔数逐年变化")
    ax.set_xlabel("年份")
    ax.set_ylabel("交易笔数 (万笔)")
    ax.set_xticks(years_t)
    style_axes_clean(ax)

    ax = axes[1]
    ax.plot(years_t, by_year["交易金额"].values, marker="o", color=COLOR["accent"])
    for xx, yy in zip(years_t, by_year["交易金额"].values):
        ax.annotate(f"{yy / 1e8:.1f}亿", (xx, yy), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9.5,
                    color=COLOR["text_soft"])
    ax.set_ylim(0, by_year["交易金额"].max() * 1.22)
    ax.set_title("交易金额逐年变化")
    ax.set_xlabel("年份")
    ax.set_ylabel("交易金额")
    ax.set_xticks(years_t)
    apply_axis_format(ax, y="currency")
    style_axes_clean(ax)

    ax = axes[2]
    ops_sorted = ops.sort_values()
    ax.barh([str(c) for c in ops_sorted.index], ops_sorted.values,
            color=COLOR["primary"], height=0.62)
    annotate_hbars(ax, ops_sorted.values / 1e4, fmt="{:.1f}万", pad=0.012,
                   fontsize=9.5)
    ax.set_xlim(0, ops_sorted.max() * 1.28)
    ax.set_title("交易类型分布")
    ax.set_xlabel("交易笔数")
    ax.set_ylabel("")
    # 类别轴必须显式 y=None, 否则 MaxNLocator 会用 0,1,2... 覆盖类别名
    apply_axis_format(ax, y=None, x="thousands")
    ax.grid(axis="y", visible=False)
    style_axes_clean(ax)

    fig.suptitle(f"105.6 万行交易表的 Python 全量分析 (共 {len(tr):,} 笔交易)",
                 y=1.04, x=0.02, ha="left")
    add_source(fig, "数据来源: trans.csv (60MB, 1,056,320 行)  |  "
                    "pandas 全量载入并聚合, 无需抽样", y=0.005)
    save_fig(fig, "ch01_04_交易表全量分析")
    return {"T": T, "tr": tr, "by_year": by_year, "n_shared": n_shared,
            "wrong": wrong, "right": right, "w_year": w_year, "r_year": r_year,
            "blank_cnt": blank_cnt, "n_clients": len(clients),
            "card_holders": int(ct.values.sum())}

# ============================================================================ #
# 3. 演示二: 维度模型 (商场销售数据表)
# ============================================================================ #

FACT_NAME = "产品销售数据表"
DIM_NAMES = ["日期表", "产品信息表", "商铺城市", "品牌", "类别"]


def load_dimensional():
    """读取 Excel 的 6 个工作表, 识别事实表与维度表。"""
    sheets = pd.read_excel(RAW_XLSX, sheet_name=None)
    # 事实表列名带前导空格(Excel 手工录入遗留), 必须先清洗
    sheets[FACT_NAME].columns = [c.strip() for c in sheets[FACT_NAME].columns]
    fact = sheets[FACT_NAME].copy()
    dims = {n: sheets[n].copy() for n in DIM_NAMES}
    return fact, dims, sheets


def draw_star_schema(fact, dims, sheets) -> None:
    """图: 维度模型结构图 (星型模型 + 雪花型扩展)。"""
    fig, ax = plt.subplots(figsize=(13, 8.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    W, H = 0.245, 0.150
    nodes = {
        "品牌": (0.175, 0.845), "产品信息表": (0.175, 0.500), "类别": (0.175, 0.155),
        "日期表": (0.775, 0.845), "产品销售数据表": (0.775, 0.500),
        "商铺城市": (0.775, 0.155),
    }
    roles = {"品牌": "维度表", "产品信息表": "维度表 · 二级", "类别": "维度表",
             "日期表": "维度表", "产品销售数据表": "事实表", "商铺城市": "维度表"}

    def connect(a, b, text, vertical=True):
        axx, ayy = nodes[a]
        bxx, byy = nodes[b]
        if vertical:
            sgn = 1 if byy > ayy else -1
            p1 = (axx, ayy + sgn * H / 2)
            p2 = (bxx, byy - sgn * H / 2)
            mx, my = axx, (p1[1] + p2[1]) / 2
        else:
            sgn = 1 if bxx > axx else -1
            p1 = (axx + sgn * W / 2, ayy)
            p2 = (bxx - sgn * W / 2, byy)
            mx, my = (p1[0] + p2[0]) / 2, ayy
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-", lw=2.0,
                                     color=COLOR["neutral"], zorder=1))
        ax.text(mx, my, text, fontsize=9.5, color=COLOR["accent"],
                fontweight="bold", ha="center", va="center", zorder=5,
                linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.30", fc="white",
                          ec=COLOR["accent"], lw=1.0, alpha=0.97))

    connect("产品信息表", "品牌", "品牌名称\nN : 1")
    connect("产品信息表", "类别", "产品类别\nN : 1")
    connect("产品销售数据表", "日期表", "订单日期\nN : 1")
    connect("产品销售数据表", "商铺城市", "商铺城市\nN : 1")
    connect("产品销售数据表", "产品信息表", "产品编号\nN : 1", vertical=False)

    for nid, (x, y) in nodes.items():
        is_fact = nid == "产品销售数据表"
        ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H,
                                    boxstyle="round,pad=0.010,rounding_size=0.024",
                                    fc="#FDEDD8" if is_fact else "#EAF1F8",
                                    ec=COLOR["accent"] if is_fact else COLOR["primary"],
                                    lw=2.2, zorder=3))
        ax.annotate(nid, (x, y), textcoords="offset points", xytext=(0, 20),
                    ha="center", va="center", zorder=4, fontsize=12.5,
                    fontweight="bold", color=COLOR["text"])
        ax.annotate(roles[nid], (x, y), textcoords="offset points", xytext=(0, 2),
                    ha="center", va="center", zorder=4, fontsize=10.5,
                    color=COLOR["accent"] if is_fact else COLOR["primary"])
        ax.annotate(f"{len(fact) if is_fact else len(sheets[nid]):,} 行",
                    (x, y), textcoords="offset points", xytext=(0, -18),
                    ha="center", va="center", zorder=4, fontsize=10,
                    color=COLOR["text_soft"])

    ax.text(0.50, 0.955, "星型模型 (雪花型扩展): 1 张事实表 + 5 张维度表",
            fontsize=13, fontweight="bold", color=COLOR["text"], ha="center")
    ax.text(0.50, 0.045,
            "事实表粒度 (grain): 一行 = 某日 × 某城市 × 某产品的销售记录   |   "
            "产品信息表是二级维度, 向上再关联 品牌 与 类别",
            fontsize=10.5, color=COLOR["text_soft"], ha="center")
    add_source(fig, "数据来源: 商场销售数据表.xlsx  |  橙色为事实表, 蓝色为维度表  |  "
                    "事实表冗余了 品牌/类别/产品名称 三列, 可同时支持宽表口径", y=0.005)
    save_fig(fig, "ch01_05_维度模型结构图")


def demo_dimensional() -> dict:
    """演示二: 维度模型 —— Power BI + Excel -> Python 迁移。"""
    rule("演示二 · 1. 读取数据模型 (6 个工作表)")
    fact, dims, sheets = load_dimensional()
    dim_product = dims["产品信息表"]

    model = pd.DataFrame(
        [{"表名": FACT_NAME, "角色": "事实表", "记录数": len(fact),
          "字段数": fact.shape[1], "粒度 / 主键": "订单日期 + 商铺城市 + 产品编号"}] +
        [{"表名": n, "角色": "维度表", "记录数": len(dims[n]),
          "字段数": dims[n].shape[1], "粒度 / 主键": str(dims[n].columns[0])}
         for n in DIM_NAMES])
    print(model.to_string(index=False))
    print(f"\n事实表列: {list(fact.columns)}")
    print("注意原列名 ' 销售单价' 带前导空格, 已清洗")

    # ---------------- 教学点三: 键一致性 ---------------- #
    rule("演示二 · 2. 教学点三: 键一致性 (大小写不一致导致关联失败)")
    raw_match = int(fact["产品编号"].isin(dim_product["产品编号"]).sum())
    print(f"事实表产品编号样例: {sorted(fact['产品编号'].unique())[:3]}")
    print(f"维表产品编号样例  : {sorted(dim_product['产品编号'].unique())[:3]}")
    print(f"直接关联          : 匹配 {raw_match:,} / {len(fact):,} 行  "
          f"({raw_match / len(fact):.1%})")

    fact["产品编号_规范"] = fact["产品编号"].astype(str).str.strip().str.upper()
    dim_product["产品编号_规范"] = (dim_product["产品编号"].astype(str)
                                    .str.strip().str.upper())
    norm_match = int(fact["产品编号_规范"].isin(dim_product["产品编号_规范"]).sum())
    print(f"键规范化后关联    : 匹配 {norm_match:,} / {len(fact):,} 行  "
          f"({norm_match / len(fact):.1%})")
    print("-> 键不一致 = 静默失败: 不报错, 但结果为空")

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    labels = ["直接关联 (不处理大小写)", "关联前统一规范化键", "改用事实表冗余的维度列"]
    vals = [raw_match, norm_match, int(fact["品牌名称"].notna().sum())]
    ax.barh(labels, vals, height=0.52,
            color=[COLOR["negative"], COLOR["positive"], COLOR["muted"]])
    annotate_hbars(ax, vals, fmt="{:,.0f} 行", pad=0.02, fontsize=11)
    ax.set_xlim(0, norm_match * 1.30)
    ax.set_xlabel("成功关联到产品维度的行数")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    ax.axvline(norm_match, color=COLOR["positive"], lw=1.4, ls="--",
               alpha=0.75, zorder=0)
    ax.text(0.985, 0.16,
            "事实表键: a001 / b002 ...\n产品维表键: A001 / B002 ...\n"
            "键不一致 = 静默失败, 不报错但结果为空",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5,
            color=COLOR["text"], linespacing=1.7,
            bbox=dict(boxstyle="round,pad=0.5", fc=COLOR["bg_soft"],
                      ec=COLOR["grid"], lw=1.0))
    ax.set_title("教学点三: 维度建模中「键的一致性」决定关联成败")
    style_axes_clean(ax)
    add_source(fig, "数据来源: 商场销售数据表.xlsx  产品销售数据表 / 产品信息表  |  "
                    "本例中 100% 的记录会因大小写不一致而丢失", y=0.01)
    save_fig(fig, "ch01_06_教学点3_键一致性")
    # ---------------- 建立维度模型 ---------------- #
    rule("演示二 · 3. 建立维度模型 (事实表 + 5 张维度表)")
    rel_df = pd.DataFrame([
        ("产品销售数据表", "订单日期", "日期表", "日期", "N : 1", "时间智能"),
        ("产品销售数据表", "商铺城市", "商铺城市", "商铺城市", "N : 1", "城市属性"),
        ("产品销售数据表", "产品编号_规范", "产品信息表", "产品编号_规范", "N : 1",
         "产品属性"),
        ("产品信息表", "品牌名称", "品牌", "品牌名称", "N : 1", "品牌属性(二级维度)"),
        ("产品信息表", "产品类别", "类别", "产品类别", "N : 1", "类别属性(二级维度)"),
    ], columns=["从表", "从表键", "到表", "到表键", "基数", "用途"])
    print(rel_df.to_string(index=False))

    star = (fact
            .merge(dim_product[["产品编号_规范", "产品名称", "品牌名称", "产品类别",
                                "采购单价", "销售单价"]],
                   on="产品编号_规范", how="left", suffixes=("_事实", ""))
            .merge(dims["品牌"], on="品牌名称", how="left")
            .merge(dims["类别"], on="产品类别", how="left")
            .merge(dims["商铺城市"], on="商铺城市", how="left")
            .merge(dims["日期表"], left_on="订单日期", right_on="日期", how="left"))
    print(f"\n事实表 {len(fact):,} 行 -> 展开维度后 {len(star):,} 行 "
          f"({'行数守恒' if len(star) == len(fact) else '行数异常'})")
    print(f"关联后关键字段缺失: 产品名称 {star['产品名称'].isna().sum()} 行, "
          f"所属企业 {star['所属企业'].isna().sum()} 行")

    # 宽表口径 vs 星型模型口径 —— 结果应当完全一致
    wide = star.groupby("产品名称_事实")["销售额"].sum()
    strd = star.groupby("产品名称")["销售额"].sum()
    gap = float((wide.sort_index() - strd.sort_index()).abs().max())
    print(f"宽表口径与星型模型口径差异: {gap:.0f} 元 (应为 0)")

    rule("演示二 · 4. 绘制维度模型结构图")
    draw_star_schema(fact, dims, sheets)

    # ---------------- 销售额趋势 ---------------- #
    rule("演示二 · 5. 迁移 Power BI 视觉: 销售额趋势 (areaChart + lineChart)")
    fact["年"] = fact["订单日期"].dt.year
    fact["月"] = fact["订单日期"].dt.month
    daily = (fact.groupby("订单日期", as_index=False)["销售额"].sum()
                 .sort_values("订单日期"))
    monthly = fact.groupby(["年", "月"], as_index=False)["销售额"].sum()
    print(f"订单日期范围: {fact['订单日期'].min():%Y-%m-%d} ~ "
          f"{fact['订单日期'].max():%Y-%m-%d} "
          f"(共 {fact['订单日期'].nunique()} 个有交易日期)")
    print(f"总销售额: {fact['销售额'].sum():,.0f} 元, "
          f"总销量: {fact['销售数量'].sum():,} 台")

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 9.4), height_ratios=[1, 1.05])
    ax = axes[0]
    ax.fill_between(daily["订单日期"], daily["销售额"], color=COLOR["primary"],
                    alpha=0.22)
    ax.plot(daily["订单日期"], daily["销售额"], color=COLOR["primary"], lw=1.6)
    peak = daily.loc[daily["销售额"].idxmax()]
    trough = daily.loc[daily["销售额"].idxmin()]
    for pt, tag, dy in [(peak, "峰值", 14), (trough, "谷值", -20)]:
        ax.scatter([pt["订单日期"]], [pt["销售额"]], s=80,
                   color=COLOR["negative"], zorder=5, edgecolor="white",
                   linewidth=1.4)
        ax.annotate(f"{tag} {pt['订单日期']:%Y-%m-%d}\n{pt['销售额']:,.0f} 元",
                    (pt["订单日期"], pt["销售额"]), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=9.5,
                    color=COLOR["text"], fontweight="bold")
    ax.set_title("① 日销售额走势 (对应 Power BI 第 1 页 areaChart)")
    ax.set_ylabel("日销售额")
    ax.set_xlabel("")
    apply_axis_format(ax, y="currency")
    ax.margins(y=0.20)
    style_axes_clean(ax)

    ax = axes[1]
    ymax = monthly["销售额"].max()
    for i, (yr, grp) in enumerate(monthly.groupby("年")):
        c = COLOR["primary"] if i == 0 else COLOR["accent"]
        ax.plot(grp["月"], grp["销售额"], marker="o", color=c,
                label=f"{yr} 年", lw=2.4)
        ax.annotate(f"{grp['销售额'].sum() / 1e8:.2f}亿",
                    (grp["月"].iloc[-1], grp["销售额"].iloc[-1]),
                    textcoords="offset points", xytext=(10, 6), fontsize=10,
                    color=c, fontweight="bold")
    piv = monthly.pivot(index="月", columns="年", values="销售额")
    yoy = piv.iloc[:, 1] / piv.iloc[:, 0] - 1
    best_m = int(yoy.idxmax())
    ax.axvline(best_m, color=COLOR["positive"], lw=1.2, ls=":", alpha=0.8)
    ax.annotate(f"{best_m} 月同比 {yoy.max():+.1%}", xy=(best_m, ymax * 0.93),
                xytext=(best_m + 0.35, ymax * 0.93), fontsize=10,
                color=COLOR["positive"], fontweight="bold", va="center")
    ax.set_title("② 月度销售额对比 (对应 Power BI 第 2 页 lineChart 的年/月层次下钻)")
    ax.set_ylabel("月销售额")
    ax.set_xlabel("月份")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([f"{m}月" for m in range(1, 13)])
    ax.set_ylim(0, ymax * 1.18)
    apply_axis_format(ax, y="currency")
    # 图例放在坐标区内部: 1-3 月数值低, 左上角空余, 同时避免压住子图标题
    ax.legend(loc="upper left", ncol=2)
    style_axes_clean(ax)

    fig.suptitle("教学迁移: Power BI 面积图 / 折线图 -> Python 趋势分析",
                 y=1.0, x=0.02, ha="left")
    add_source(fig, "数据来源: 产品销售数据表 (12,960 行)  |  Python 可直接给出同比、"
                    "峰值谷值等 Power BI 需额外度量值才能表达的信息", y=0.005)
    save_fig(fig, "ch01_07_销售额趋势")
    # ---------------- 产品 / 品牌 / 城市 ---------------- #
    rule("演示二 · 6. 迁移 multiRowCard 卡片图: 产品 / 品牌 / 城市结构")
    prod = (star.groupby(["产品名称", "品牌名称", "产品类别"], as_index=False)
                .agg(销售额=("销售额", "sum"), 销量=("销售数量", "sum"))
                .sort_values("销售额"))
    brand_cat = star.pivot_table(index="品牌名称", columns="产品类别",
                                 values="销售额", aggfunc="sum").fillna(0)
    city = star.groupby("商铺城市", as_index=False)["销售额"].sum().sort_values("销售额")

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.4),
                             width_ratios=[1.0, 1.0, 0.92],
                             gridspec_kw={"wspace": 0.34})
    ax = axes[0]
    brand_palette = {"GL": 0, "MD": 1, "HE": 2}
    ax.barh(prod["产品名称"], prod["销售额"] / 1e4, height=0.66,
            color=[PALETTE[brand_palette[b]] for b in prod["品牌名称"]])
    annotate_hbars(ax, prod["销售额"].values / 1e4, fmt="{:,.0f}万", pad=0.014,
                   fontsize=9.5)
    ax.set_xlim(0, prod["销售额"].max() / 1e4 * 1.30)
    ax.set_title("① 各产品销售额 (对应 multiRowCard)")
    ax.set_xlabel("销售额 (万元)")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[i]) for i in range(3)]
    ax.legend(handles, ["GL", "MD", "HE"], loc="lower right", title="品牌",
              title_fontsize=10)
    style_axes_clean(ax)

    ax = axes[1]
    x = np.arange(len(brand_cat.index))
    bottom = np.zeros(len(brand_cat.index))
    for i, c in enumerate(brand_cat.columns):
        v = brand_cat[c].values / 1e4
        ax.bar(x, v, bottom=bottom, width=0.6, label=c, color=PALETTE[i])
        for xi, (b, val) in enumerate(zip(bottom, v)):
            if val > 0:
                ax.text(xi, b + val / 2, f"{val:,.0f}", ha="center", va="center",
                        fontsize=9.5, color="white", fontweight="bold")
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels(brand_cat.index)
    ax.set_title("② 品牌 x 产品类别 销售额构成")
    ax.set_xlabel("品牌")
    ax.set_ylabel("销售额 (万元)")
    ax.set_ylim(0, bottom.max() * 1.16)
    ax.legend(loc="upper right", title="产品类别", title_fontsize=10)
    style_axes_clean(ax)

    ax = axes[2]
    order_c = city.sort_values("销售额")
    ax.barh(order_c["商铺城市"], order_c["销售额"] / 1e4,
            color=COLOR["primary"], height=0.66)
    annotate_hbars(ax, order_c["销售额"].values / 1e4, fmt="{:,.0f}万", pad=0.014,
                   fontsize=9.5)
    ax.set_xlim(0, order_c["销售额"].max() / 1e4 * 1.26)
    ax.set_title("③ 各城市销售额")
    ax.set_xlabel("销售额 (万元)")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    ax.text(0.97, 0.06,
            f"头部城市 {order_c['商铺城市'].iloc[-1]} "
            f"占 {order_c['销售额'].iloc[-1] / order_c['销售额'].sum():.1%}",
            transform=ax.transAxes, ha="right", fontsize=10, color=COLOR["text"],
            bbox=dict(boxstyle="round,pad=0.42", fc=COLOR["bg_soft"],
                      ec=COLOR["grid"], lw=1.0))
    style_axes_clean(ax)

    fig.suptitle("维度模型的多维分析: 产品 / 品牌 / 类别 / 城市",
                 y=1.02, x=0.02, ha="left")
    add_source(fig, "数据来源: 产品销售数据表 关联 产品信息表 / 品牌 / 类别 / 商铺城市  |  "
                    "对应 Power BI 的多卡片图与交叉分析", y=0.005)
    save_fig(fig, "ch01_08_产品_品牌_城市结构")

    # ---------------- 关键影响因素 ---------------- #
    rule("演示二 · 7. 迁移 keyDriversVisual: 关键影响因素分析")
    star["毛利"] = star["销售额"] - star["销售数量"] * star["采购单价"]
    corr = (star[["销售数量", "销售单价", "采购单价", "毛利", "销售额"]]
            .corr()["销售额"].drop("销售额").sort_values())
    print("各数值因素与销售额的相关系数:")
    print(corr.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), width_ratios=[1.25, 1.0],
                             gridspec_kw={"wspace": 0.26})
    ax = axes[0]
    for i, c in enumerate(["冰箱", "电视", "洗衣机"]):
        sub = star[star["产品类别"] == c]
        ax.scatter(sub["销售数量"], sub["销售额"], s=26, alpha=0.55,
                   color=PALETTE[i], label=c, edgecolors="white", linewidths=0.3)
    k, b = np.polyfit(star["销售数量"], star["销售额"], 1)
    xs = np.linspace(star["销售数量"].min(), star["销售数量"].max(), 50)
    ax.plot(xs, k * xs + b, color=COLOR["text"], lw=2.0, ls="--",
            label=f"回归线 (r = {corr['销售数量']:.2f})")
    ax.set_title("① 销售数量 与 销售额 的关系")
    ax.set_xlabel("单笔销售数量 (台)")
    ax.set_ylabel("单笔销售额 (元)")
    apply_axis_format(ax, y="thousands")
    ax.legend(loc="upper left", ncol=2)
    style_axes_clean(ax)

    ax = axes[1]
    v = corr.values
    ax.barh([str(i) for i in corr.index], v, height=0.58,
            color=[COLOR["positive"] if t > 0 else COLOR["negative"] for t in v])
    annotate_hbars(ax, v, fmt="{:+.2f}", pad=0.02, fontsize=10.5)
    ax.set_xlim(min(v.min() * 1.35, -0.2), max(v.max() * 1.28, 1.1))
    ax.axvline(0, color=COLOR["text_soft"], lw=0.9)
    ax.set_title("② 各因素与销售额的相关系数")
    ax.set_xlabel("Pearson 相关系数")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    style_axes_clean(ax)

    fig.suptitle("关键影响因素: 什么在驱动销售额", y=1.02, x=0.02, ha="left")
    add_source(fig, "数据来源: 产品销售数据表  |  "
                    "对应 Power BI 的 keyDriversVisual 关键影响因素图", y=0.005)
    save_fig(fig, "ch01_09_关键影响因素")
    # ---------------- 销售规律热力图 ---------------- #
    rule("演示二 · 8. 销售规律: 月份 x 星期 热力图")
    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    fact["星期"] = fact["订单日期"].dt.dayofweek
    # 用"日均销售额"消除各月天数与星期出现次数的差异
    cells = fact.groupby(["月", "星期"])["销售额"].sum().unstack(fill_value=0)
    days = fact.groupby(["月", "星期"])["订单日期"].nunique().unstack(fill_value=1)
    heat = (cells / days).reindex(index=range(1, 13), columns=range(7), fill_value=0)

    fig, ax = plt.subplots(figsize=(12.5, 7.4))
    im = ax.imshow(heat.values, cmap=SEQ_CMAP, aspect="auto")
    ax.set_xticks(range(7))
    ax.set_xticklabels(weekday_names)
    ax.set_yticks(range(12))
    ax.set_yticklabels([f"{m}月" for m in range(1, 13)])
    ax.grid(False)
    rng_heat = float(np.ptp(heat.values)) or 1.0   # NumPy 2 已移除 ndarray.ptp
    for i in range(12):
        for j in range(7):
            val = heat.values[i, j]
            norm = (val - heat.values.min()) / rng_heat
            ax.text(j, i, f"{val / 1e4:,.0f}", ha="center", va="center",
                    fontsize=9.5, fontweight="bold",
                    color="white" if norm > 0.62 else COLOR["text"])
    ax.set_title("销售规律: 月份 x 星期 的日均销售额 (万元)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.038)
    cb.set_label("日均销售额 (元)")
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x / 1e4:,.0f}万"))
    cb.outline.set_visible(False)
    style_axes_clean(ax)
    add_source(fig, "数据来源: 产品销售数据表  |  "
                    "取日均值以消除各月天数与星期出现次数的差异", y=0.005)
    save_fig(fig, "ch01_10_销售规律热力图")

    stacked = heat.stack()
    best, worst = stacked.idxmax(), stacked.idxmin()
    print(f"日均销售额最高: {best[0]}月 {weekday_names[best[1]]}  {stacked.max():,.0f} 元")
    print(f"日均销售额最低: {worst[0]}月 {weekday_names[worst[1]]}  {stacked.min():,.0f} 元")

    # ---------------- 交互式图表 ---------------- #
    rule("演示二 · 9. 生成交互式 HTML 图表 (对应 Power BI 切片器)")
    if not MAKE_INTERACTIVE_HTML:
        print("MAKE_INTERACTIVE_HTML = False, 跳过 (本次只输出 PNG 图片)")
    elif apply_plotly_template():
        try:
            import plotly.express as px
            m2 = star.copy()
            m2["年"] = m2["订单日期"].dt.year
            agg = m2.groupby(["订单日期", "年"], as_index=False)["销售额"].sum()
            fig1 = px.area(agg, x="订单日期", y="销售额", color="年",
                           title="日销售额走势 (点击图例可按年份筛选)",
                           labels={"销售额": "日销售额 (元)", "订单日期": ""})
            fig1.update_layout(height=520, hovermode="x unified",
                               yaxis_tickformat=",.0f")
            fig1.update_traces(line=dict(width=1.8))
            print("已生成:", save_html(fig1, "ch01_11_交互式_日销售额走势"))

            mm = (star.assign(年=star["订单日期"].dt.year, 月=star["订单日期"].dt.month)
                      .groupby(["年", "月", "品牌名称"], as_index=False)
                      .agg(销售额=("销售额", "sum"), 销量=("销售数量", "sum")))
            fig2 = px.bar(mm, x="月", y="销售额", color="品牌名称", facet_row="年",
                          barmode="stack", height=620,
                          title="月度销售额 · 品牌构成 (按年度分面, 对应切片器联动)",
                          labels={"月": "月份", "销售额": "销售额 (元)"})
            fig2.update_layout(yaxis_tickformat=",.0f")
            print("已生成:", save_html(fig2, "ch01_12_交互式_月度品牌构成"))
        except Exception as e:
            print("Plotly 交互图生成跳过:", type(e).__name__, e)
    else:
        print("未安装 plotly, 跳过交互图 (图片不受影响)")

    # ---------------- 数据质量校验 ---------------- #
    rule("演示二 · 10. 数据质量校验")
    calc = star["销售数量"] * star["销售单价"]
    q = pd.DataFrame({
        "校验项": ["销售额 = 销售数量 x 销售单价", "事实表单价 = 产品维表单价",
                   "事实表产品名称 = 维表产品名称", "事实表日期存在于日期表",
                   "完全重复行"],
        "问题记录数": [
            int((calc.round(2) != star["销售额"].round(2)).sum()),
            int((star["销售单价_事实"] != star["销售单价"]).sum()),
            int((star["产品名称_事实"] != star["产品名称"]).sum()),
            int(star["日期"].isna().sum()),
            int(fact.duplicated().sum())],
    })
    print(q.to_string(index=False))

    return {"fact": fact, "star": star, "dims": dims, "xlsx": RAW_XLSX,
            "corr": corr, "yoy": yoy, "raw_match": raw_match,
            "norm_match": norm_match}


# ============================================================================ #
# 4. 补充案例: sale.csv (年份 x 区域 的销售额与利润)
# ============================================================================ #

def draw_sale_csv() -> pd.DataFrame:
    """补充案例: 章节根目录的 sale.csv。

    教材未提供该数据的说明文档, 因此按通用对比口径出图:
    先看规模 (销售额 / 利润), 再看质量 (利润率), 构成完整的对比视图。

    采用横向分组条形图: 数值 5-6 位, 纵向柱的标签会互相挤压, 横向布局
    才能给标签留出足够空间 (与纵向柱的经典陷阱相同)。
    """
    rule("补充案例 · sale.csv (年份 x 区域)")
    df = pd.read_csv(RAW_SALE, encoding="gbk")
    df["利润率"] = df["profit"] / df["sale"]
    print(df.to_string(index=False))

    regions = ["东", "南", "西", "北"]
    years = sorted(df["year"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.0),
                             gridspec_kw={"wspace": 0.26})
    y = np.arange(len(regions))
    height = 0.26
    panels = [(axes[0], "sale", "① 各区域销售额", "{:,.0f}", "销售额 (元)"),
              (axes[1], "profit", "② 各区域利润", "{:,.0f}", "利润 (元)"),
              (axes[2], "利润率", "③ 各区域利润率", "{:.1%}", "利润率")]

    for ax, col, title, fmt, xlabel in panels:
        peak = float(df[col].max())
        for i, yr in enumerate(years):
            sub = df[df["year"] == yr].set_index("market").reindex(regions)
            vals = sub[col].values.astype(float)
            offset = (i - (len(years) - 1) / 2) * height
            ax.barh(y + offset, vals, height=height, label=f"{yr} 年",
                    color=PALETTE[i])
            annotate_hbars(ax, vals, fmt=fmt, pad=0.012, fontsize=8.5)
        ax.set_yticks(y)
        ax.set_yticklabels(regions)
        ax.invert_yaxis()                       # 东 在最上方
        ax.set_xlim(0, peak * 1.26)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("区域" if ax is axes[0] else "")
        ax.grid(axis="y", visible=False)
        # 该量级下 万/千分位混用会出现重复刻度, 统一用纯千分位
        if col == "利润率":
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        else:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        style_axes_clean(ax)

    # 图例放在图外顶部: 坐标区内任何位置都会压住某一组数据 (四组数据已铺满)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[i]) for i in range(len(years))]
    fig.legend(handles, [f"{yr} 年" for yr in years], loc="upper right",
               bbox_to_anchor=(0.995, 1.045), ncol=len(years), frameon=False)
    fig.suptitle("补充案例: 区域销售额 / 利润 / 利润率对比", y=1.02, x=0.02, ha="left")
    add_source(fig, "数据来源: 第1章 数据分析思维/sale.csv (GBK 编码, 共 12 条记录)", y=0.005)
    save_fig(fig, "ch01_11_区域销售与利润")
    return df


# ============================================================================ #
# 5. 主入口
# ============================================================================ #

def main() -> int:
    font = setup_style()
    print("=" * 78)
    print("《商业数据分析》第1章 数据分析思维 —— Power BI / Excel 内容的 Python 迁移")
    print("=" * 78)
    print(f"Python    : {sys.version.split()[0]}")
    print(f"解释器    : {sys.executable}")
    print(f"项目目录  : {ROOT}")
    print(f"中文字体  : {font}")
    print(f"图片输出  : {FIG}  (PNG, 300dpi)")

    for p in (RAW_REL, RAW_XLSX, RAW_SALE):
        if not p.exists():
            print(f"\n[错误] 找不到教材数据: {p}")
            return 2

    r1 = demo_relational()
    r2 = demo_dimensional()
    draw_sale_csv()

    rule("全章结论")
    fact = r2["fact"]
    t2018 = fact[fact["年"] == 2018]["销售额"].sum()
    t2019 = fact[fact["年"] == 2019]["销售额"].sum()
    print(f"""
演示一 · 关系模型 (银行客户与交易数据)
  1. 关系结构本身就是分析口径的一部分
     - 8 张表中 disp 是客户与账户多对多关系的桥梁, {r1['n_shared']} 个账户为 2 人共管
     - 同一个"贷款金额"经不同关联路径得出
       {r1['right']['分摊金额'].sum():,.0f} 元 或 {r1['wrong'].amount.sum():,.0f} 元,
       差 {r1['wrong'].amount.sum() / r1['right']['分摊金额'].sum() - 1:+.1%}
  2. Power BI 的两类默认行为会造成口径错误
     - 默认全连接 -> 生成"(空白)"组, 占 {r1['blank_cnt'] / r1['n_clients']:.0%},
       掩盖 {r1['card_holders']:,} 名持卡客户的真实分布
     - 一对多关系上的度量值被自动传播 -> 金额重复累计
  3. Python 的优势
     - 关联方式显式可控 (how="inner"/"left"/"outer"), 口径写在代码里, 可复核
     - 1,056,320 行交易表全量载入 + 多表关联, 无需抽样
  4. 正确做法不是"不用关系模型", 而是明确粒度与基数, 先聚合后关联

演示二 · 维度模型 (商场销售数据表)
  1. 事实表 {len(fact):,} 行, 粒度 = 订单日期 × 商铺城市 × 产品
     5 张维度表: 日期({len(r2['dims']['日期表'])}), 产品({len(r2['dims']['产品信息表'])}),
     城市({len(r2['dims']['商铺城市'])}), 品牌({len(r2['dims']['品牌'])}),
     类别({len(r2['dims']['类别'])})
  2. 键的一致性决定关联成败
     - 事实表 a001 vs 维表 A001: 直接关联匹配 {r2['raw_match']} 行, 全部静默丢失
     - 规范化键后匹配 {r2['norm_match']:,} 行, 覆盖率 100%
  3. 宽表与星型模型口径一致 (差异 0 元), 但星型模型更易维护一致性
  4. 日期维度是时间智能的前提
     - 2019 年销售额 {t2019:,.0f} 元, 较 2018 年 ({t2018:,.0f} 元)
       增长 {t2019 / t2018 - 1:.1%}
     - 同比增幅最大的月份: {int(r2['yoy'].idxmax())} 月 ({r2['yoy'].max():+.1%})
  5. 关键驱动因素: 销售数量与销售额相关系数 {r2['corr']['销售数量']:.2f}
""")
    print("图片已输出到:", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
