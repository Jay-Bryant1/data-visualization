# -*- coding: utf-8 -*-
r"""
第7章 指标体系构建方法 —— Excel 模板 -> Python
运行: python -X utf8 ch07_指标体系构建.py
输出: figures/*.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd

HERE = Path(__file__).resolve().parent      # 本章交付目录
ROOT = HERE.parent                          # 项目根目录 (教材原文所在位置)
SRC = ROOT / "第7章 指标体系构建方法" / "1、指标体系设计模板.xlsx"
FIG = HERE                        ; FIG.mkdir(parents=True, exist_ok=True)

C = {"primary": "#4E79A7", "accent": "#F28E2B", "positive": "#59A14F",
     "negative": "#E15759", "grid": "#E4E7EB", "text": "#2F3B45",
     "soft": "#7A8794", "bg": "#F7F9FB"}
PAL = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#76B7B2"]


def setup():
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "font.family": "sans-serif", "axes.unicode_minus": False,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.18,
        "figure.facecolor": "white", "font.size": 11,
        "axes.titlesize": 15, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 14,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": C["grid"], "axes.labelcolor": C["text"],
        "xtick.color": C["soft"], "ytick.color": C["soft"],
        "xtick.major.size": 0, "ytick.major.size": 0,
        "grid.color": C["grid"], "grid.linewidth": 0.8,
        "axes.grid": True, "axes.axisbelow": True,
        "legend.frameon": False,
    })


def save(fig, name):
    p = FIG / f"{name}.png"; fig.savefig(p); plt.close(fig); return p


def clean(ax):
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["grid"])


def main():
    setup()
    xl = pd.ExcelFile(SRC)
    step1 = xl.parse("Step1")
    step2 = xl.parse("Step2")
    print("Step1 字段:", list(step1.columns))
    print("Step2 行数:", len(step2))

    # ---------- 图1: 指标体系构建流程 ----------
    steps = ["业务场景描述", "核心目标", "制定策略", "指标拆分", "明确口径", "明确分析维度"]
    desc = ["说清业务问题\n与分析目的", "用一句话定义\n要达成的结果",
            "拆解达成目标的\n业务动作", "动作 -> 可量化\n的指标",
            "定义分子/分母\n时间与维度", "确定下钻与\n对比的维度"]
    fig, ax = plt.subplots(figsize=(17, 6.2))
    ax.set_xlim(0, len(steps) * 2.2)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    for i, (s, d) in enumerate(zip(steps, desc)):
        x = i * 2.2 + 0.15
        ax.add_patch(FancyBboxPatch((x, 1.55), 1.85, 0.95,
                                    boxstyle="round,pad=0.03,rounding_size=0.12",
                                    fc=PAL[i], ec="none", alpha=0.92))
        ax.text(x + 0.925, 2.02, s, ha="center", va="center", color="white",
                fontsize=13, fontweight="bold")
        ax.text(x + 0.925, 1.15, d, ha="center", va="top", fontsize=10,
                color=C["text"], linespacing=1.6)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.86, 2.02), (x + 2.19, 2.02),
                                         arrowstyle="-|>", mutation_scale=18,
                                         color=C["soft"], lw=1.6))
    ax.text(0.15, 2.85, "指标体系构建的 6 个步骤", fontsize=16,
            fontweight="bold", color=C["text"])
    ax.text(0.15, 0.45,
            "核心原则: 指标必须从业务目标倒推, 先有策略再有指标; "
            "口径与分析维度必须在建指标时就写清楚, 否则后期无法复用。",
            fontsize=11, color=C["soft"])
    save(fig, "ch07_01_指标体系构建流程")

    # ---------- 图2: 指标字典示例 ----------
    df = step2.dropna(how="all")
    if len(df):
        df = df.iloc[:, :6]
        df.columns = ["指标名称(KPI)", "计算规则", "值", "口径", "事件", "维度"][:df.shape[1]]
        fig, ax = plt.subplots(figsize=(16, 0.62 * len(df) + 1.8))
        ax.axis("off")
        tbl = ax.table(cellText=df.astype(str).values, colLabels=df.columns,
                       loc="center", cellLoc="left")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.9)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor(C["grid"])
            if r == 0:
                cell.set_facecolor(C["primary"]); cell.set_text_props(color="white", weight="bold")
            elif r % 2 == 0:
                cell.set_facecolor(C["bg"])
        ax.set_title("指标字典 (Step2 模板): 一个指标 = 名称 + 计算规则 + 口径 + 维度",
                     pad=18)
        save(fig, "ch07_02_指标字典示例")
        print(df.to_string(index=False))
    print("图片输出:", FIG)


if __name__ == "__main__":
    main()
