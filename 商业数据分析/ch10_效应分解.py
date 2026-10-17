# -*- coding: utf-8 -*-
r"""
第10章 使用效应分解方法做预报 —— 时间序列分解
运行: python -X utf8 ch10_效应分解.py
输出: figures/*.png

教材的两组案例:
  1. AirPassengers 月度客流: 趋势 + 季节 + 残差 (加法分解)
  2. 信用卡消费额: 含节日 vs 去除节日, 用一个"节日效应"把两者拆开
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

HERE = Path(__file__).resolve().parent      # 本章交付目录
ROOT = HERE.parent                          # 项目根目录 (教材原文所在位置)
D = ROOT / "第10章 使用效应分解方法做预报"
FIG = HERE                        ; FIG.mkdir(parents=True, exist_ok=True)
C = {"primary": "#4E79A7", "accent": "#F28E2B", "positive": "#59A14F",
     "negative": "#E15759", "grid": "#E4E7EB", "text": "#2F3B45", "soft": "#7A8794"}
PAL = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]


def setup():
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "font.family": "sans-serif", "axes.unicode_minus": False,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.18,
        "figure.facecolor": "white", "font.size": 11,
        "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": C["grid"], "axes.labelcolor": C["text"],
        "xtick.color": C["soft"], "ytick.color": C["soft"],
        "xtick.major.size": 0, "ytick.major.size": 0,
        "grid.color": C["grid"], "grid.linewidth": 0.8,
        "axes.grid": True, "axes.axisbelow": True, "legend.frameon": False,
    })


def save(fig, n):
    p = FIG / f"{n}.png"; fig.savefig(p); plt.close(fig); return p


def clean(ax):
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["grid"])


def main():
    setup()
    # ---------- 1. AirPassengers 加法分解 ----------
    df = pd.read_excel(D / "AirPassengers.xlsx", sheet_name="AirPassengers")
    df = df.iloc[:, :2]
    df.columns = ["DATE", "AIR"]
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    s = df.dropna().set_index("DATE")["AIR"].astype(float).asfreq("MS")
    s = s.interpolate()
    print(f"AirPassengers: {len(s)} 个月 ({s.index.min():%Y-%m} ~ {s.index.max():%Y-%m})")

    res = seasonal_decompose(s, model="additive", period=12)
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True,
                             gridspec_kw={"hspace": 0.30})
    for ax, series, title, color in [
            (axes[0], s, "① 原始序列 Observed", C["primary"]),
            (axes[1], res.trend, "② 趋势项 Trend", C["accent"]),
            (axes[2], res.seasonal, "③ 季节项 Seasonal", C["positive"]),
            (axes[3], res.resid, "④ 残差项 Residual", C["negative"])]:
        ax.plot(series.index, series.values, color=color, lw=1.8)
        if title.endswith("Residual"):
            ax.axhline(0, color=C["soft"], lw=0.8, ls=":")
        ax.set_title(title)
        clean(ax)
    axes[0].set_ylabel("客流量")
    axes[3].set_xlabel("年份")
    fig.suptitle("AirPassengers 加法分解: 趋势 × 季节 × 残差", y=0.995, x=0.02, ha="left")
    fig.text(0.005, 0.005,
             "数据来源: AirPassengers.xlsx  |  加法模型 观测值 = 趋势 + 季节 + 残差, 周期 12 个月",
             fontsize=9, color=C["soft"], ha="left", va="top")
    save(fig, "ch10_01_AirPassengers分解")

    # ---------- 2. 信用卡消费额: 含节日 vs 去节日 ----------
    a = pd.read_excel(D / "信用卡消费额_含节日.xlsx", sheet_name="pos")
    a.columns = [str(c).strip() for c in a.columns]
    a["date"] = pd.to_datetime(a["date"])
    a = a.set_index("date")["amount"].astype(float)

    b = pd.read_excel(D / "信用卡消费额_去除节日.xlsx", sheet_name="Sheet1")
    b.columns = [str(c).strip() for c in b.columns]
    b["日期"] = pd.to_datetime(b["日期"])
    # 去节日表以"万元"计量, 含节日表以"元"计量, 统一换算为元后再比较
    b = b.set_index("日期").iloc[:, 0].astype(float) * 1e4

    idx = a.index.intersection(b.index)
    a, b = a.reindex(idx).ffill(), b.reindex(idx).ffill()
    eff = a - b
    print(f"信用卡消费额: {len(idx)} 个日频点 ({idx.min():%Y-%m-%d} ~ {idx.max():%Y-%m-%d})")
    print(f"含节日均值 {a.mean():,.0f} 元, 去节日均值 {b.mean():,.0f} 元, "
          f"节日效应均值 {eff.mean():,.0f} 元 ({eff.mean()/b.mean():+.2%})")

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.4), height_ratios=[1.2, 0.8],
                             gridspec_kw={"hspace": 0.28})
    ax = axes[0]
    ax.plot(idx, b.values, color=C["primary"], lw=1.5, label="去除节日")
    ax.plot(idx, a.values, color=C["accent"], lw=1.2, alpha=0.85, label="含节日")
    ax.set_title("① 信用卡消费额: 含节日 vs 去除节日")
    ax.set_ylabel("消费额")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e4:,.0f}万"))
    ax.legend(loc="upper left", ncol=2)
    clean(ax)

    ax = axes[1]
    ax.fill_between(idx, eff.values, color=C["negative"], alpha=0.25)
    ax.plot(idx, eff.values, color=C["negative"], lw=1.3)
    ax.axhline(0, color=C["soft"], lw=0.8, ls=":")
    ax.set_title("② 节日效应 = 含节日 − 去除节日")
    ax.set_ylabel("差额")
    ax.set_xlabel("日期")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e4:,.1f}万"))
    clean(ax)
    fig.suptitle("效应分解: 把节日带来的增量单独剥离出来", y=0.995, x=0.02, ha="left")
    fig.text(0.005, 0.005,
             "数据来源: 信用卡消费额_含节日.xlsx / _去除节日.xlsx  |  "
             "原表单位不同 (元 / 万元), 已统一为元; 本例节日效应为负", fontsize=9, color=C["soft"], ha="left", va="top")
    save(fig, "ch10_02_信用卡节日效应")
    print("图片输出:", FIG)


if __name__ == "__main__":
    main()
