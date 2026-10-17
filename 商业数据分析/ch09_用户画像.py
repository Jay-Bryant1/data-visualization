# -*- coding: utf-8 -*-
r"""
第9章 用户标签体系与用户画像 —— 信用卡客户画像
运行: python -X utf8 ch09_用户画像.py
输出: figures/*.png

原教材用 MySQL + SQL 从 kdd99 银行库生成客户画像, 本脚本用 pandas 复现同一套口径:
  人口属性 (性别 / 发卡时年龄) + 产品属性 (卡类型) + 行为属性 (R / F / M)
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent      # 本章交付目录
ROOT = HERE.parent                          # 项目根目录 (教材原文所在位置)
DATA = ROOT / "第1章 数据分析思维" / "关系模型演示-不建议用poweBI操作关系型数据模型"
FIG = HERE                        ; FIG.mkdir(parents=True, exist_ok=True)
C = {"primary": "#4E79A7", "accent": "#F28E2B", "positive": "#59A14F",
     "negative": "#E15759", "grid": "#E4E7EB", "text": "#2F3B45", "soft": "#7A8794"}
PAL = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1"]


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
        "axes.grid": True, "axes.axisbelow": True, "legend.frameon": False,
    })


def save(fig, n):
    p = FIG / f"{n}.png"; fig.savefig(p); plt.close(fig); return p


def clean(ax):
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["grid"])


def load():
    def r(n):
        return pd.read_csv(DATA / f"{n}.csv", encoding="gbk")
    card, disp, clients = r("card"), r("disp"), r("clients")
    tr = r("trans")
    tr["amount"] = pd.to_numeric(tr["amount"].astype(str).str.replace(",", "", regex=False),
                                 errors="coerce")
    tr["date"] = pd.to_datetime(tr["date"])
    return card, disp, clients, tr


def main():
    setup()
    card, disp, clients, tr = load()
    # SQL: card -> disp -> clients (LEFT JOIN), 发卡时年龄 = datediff(issued, birth)/365
    c = (card.merge(disp[["disp_id", "client_id", "account_id"]], on="disp_id", how="left")
             .merge(clients[["client_id", "sex", "birth_date"]], on="client_id", how="left"))
    c["issued"] = pd.to_datetime(c["issued"])
    c["birth_date"] = pd.to_datetime(c["birth_date"])
    c["年龄"] = ((c["issued"] - c["birth_date"]).dt.days / 365).round(0)
    c["年龄段"] = pd.cut(c["年龄"], [0, 30, 40, 50, 60, 120],
                         labels=["18-30", "31-40", "41-50", "51-60", "60+"])
    print(f"客户画像样本: {len(c):,} 张卡")

    # 图1: 卡类型 x 性别 + 年龄段
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.0), width_ratios=[1, 1.15],
                             gridspec_kw={"wspace": 0.28})
    ax = axes[0]
    ct = pd.crosstab(c["type"], c["sex"])
    x = np.arange(len(ct)); w = 0.36
    for i, s in enumerate(ct.columns):
        b = ax.bar(x + (i - 0.5) * w, ct[s].values, width=w, label=s,
                   color=[C["primary"], "#B07AA1"][i % 2])
        for bb, v in zip(b, ct[s].values):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 6, f"{v}", ha="center",
                    fontsize=10, fontweight="bold", color=C["soft"])
    ax.set_xticks(x); ax.set_xticklabels(ct.index)
    ax.set_ylim(0, ct.values.max() * 1.20)
    ax.set_title("① 卡类型 × 性别")
    ax.set_ylabel("客户数"); ax.legend(loc="upper right", ncol=2)
    clean(ax)

    ax = axes[1]
    age = pd.crosstab(c["年龄段"], c["type"])
    bottom = np.zeros(len(age))
    xa = np.arange(len(age))
    for i, col in enumerate(age.columns):
        ax.bar(xa, age[col].values, bottom=bottom, width=0.6, label=col, color=PAL[i])
        bottom += age[col].values
    ax.set_xticks(xa); ax.set_xticklabels(age.index)
    ax.set_title("② 发卡时年龄段 × 卡类型")
    ax.set_ylabel("客户数"); ax.set_xlabel("发卡时年龄")
    ax.legend(loc="upper right", ncol=1, fontsize=9)
    clean(ax)
    fig.suptitle("信用卡客户画像: 人口属性 × 产品属性", y=1.02, x=0.02, ha="left")
    fig.text(0.005, 0.005, "数据来源: kdd99 card / disp / clients  |  "
                           "对应原 SQL 第 1 步 (发卡时年龄 = datediff(issued, birth_date)/365)",
             fontsize=9, color=C["soft"], ha="left", va="top")
    save(fig, "ch09_01_人口与产品画像")

    # 图2: 行为画像 R/F/M by 卡类型 (近 180 天)
    snap = tr["date"].max() + pd.Timedelta(days=1)
    last = tr.groupby("account_id")["date"].max()
    stat = tr.groupby("account_id").agg(F=("trans_id", "count"),
                                         M=("amount", "sum"))
    stat["R"] = (snap - stat.index.map(last)).days
    stat = stat.reset_index().merge(
        c[["account_id", "type"]].drop_duplicates("account_id"), on="account_id", how="inner")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw={"wspace": 0.30})
    for ax, col, title in zip(axes, ["R", "F", "M"],
                              ["① 最近交易间隔 R (天)", "② 交易频次 F (笔)",
                               "③ 交易金额 M (元)"]):
        data = [stat.loc[stat["type"] == t, col].dropna().values
                for t in stat["type"].dropna().unique()]
        labels = list(stat["type"].dropna().unique())
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55,
                        medianprops=dict(color=C["accent"], lw=2),
                        flierprops=dict(marker="o", markersize=3, alpha=0.35,
                                        markerfacecolor=C["negative"], markeredgecolor="none"))
        for patch, col_ in zip(bp["boxes"], PAL):
            patch.set_facecolor(col_); patch.set_alpha(0.55)
            patch.set_edgecolor(col_)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        clean(ax)
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e4:,.0f}万"))
    fig.suptitle("行为画像: 不同卡类型的 R / F / M 分布", y=1.02, x=0.02, ha="left")
    fig.text(0.005, 0.005, "数据来源: kdd99 trans 按账户聚合  |  "
                           "对应原 SQL 第 2-3 步 (最后交易间隔、半年交易笔数与金额)",
             fontsize=9, color=C["soft"], ha="left", va="top")
    save(fig, "ch09_02_行为画像_RFM")

    # 标签体系: 人群规模
    seg = c.groupby(["type", "sex"]).size().unstack(fill_value=0)
    print(seg.to_string())
    print("图片输出:", FIG)


if __name__ == "__main__":
    main()
