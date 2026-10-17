# -*- coding: utf-8 -*-
r"""
第8章 SQL 语言基础与 MySQL 入门 —— SQL 逻辑的 pandas 等价实现
运行: python -X utf8 ch08_SQL基础.py
输出: figures/*.png

教材的 SQL 示例都在 kdd99 银行库上。这里用 pandas 复现同样的五类操作:
  SELECT / WHERE / ORDER BY / DISTINCT / GROUP BY-HAVING / JOIN / 窗口函数
并在每张图上标注对应的 SQL 子句, 方便对照学习。
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
        "axes.grid": True, "axes.axisbelow": True, "legend.frameon": False,
    })


def save(fig, n):
    p = FIG / f"{n}.png"; fig.savefig(p); plt.close(fig); return p


def clean(ax):
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["grid"])


def load():
    def r(n, **kw):
        return pd.read_csv(DATA / f"{n}.csv", encoding="gbk", **kw)
    trans = r("trans", usecols=["trans_id", "account_id", "date", "type",
                                "operation", "amount", "balance"])
    trans["amount"] = pd.to_numeric(trans["amount"].astype(str).str.replace(",", "", regex=False),
                                    errors="coerce")
    trans["date"] = pd.to_datetime(trans["date"])
    return {"trans": trans, "accounts": r("accounts"), "disp": r("disp"),
            "clients": r("clients"), "card": r("card")}


def main():
    setup()
    T = load()
    tr = T["trans"]
    print(f"trans {len(tr):,} 行; 账户 {tr.account_id.nunique():,}")

    # ---------- 1. WHERE + ORDER BY ----------
    q1 = tr[(tr["date"] >= "1998-12-01") & (tr["date"] <= "1998-12-31")
            & (tr["operation"] == "房屋贷款")].copy()
    q1["日"] = q1["date"].dt.day
    by_day = q1.groupby("日").agg(笔数=("trans_id", "count"), 金额=("amount", "sum"))
    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.bar(by_day.index, by_day["金额"], color=C["primary"], width=0.66)
    ax.set_title("① 单表查询: 1998 年 12 月的「房屋贷款」交易")
    ax.set_xlabel("日期 (12 月)")
    ax.set_ylabel("交易金额")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e4:,.0f}万"))
    ax.set_xticks(by_day.index)
    ax.text(0.02, 0.95,
            "SELECT * FROM trans\nWHERE date BETWEEN '1998-12-01' AND '1998-12-31'\n"
            "  AND k_symbol = '房屋贷款'\nORDER BY date;",
            transform=ax.transAxes, va="top", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.45", fc="#F7F9FB", ec=C["grid"]))
    clean(ax)
    save(fig, "ch08_01_单表查询_WHERE")

    # ---------- 2. GROUP BY + HAVING ----------
    g = (tr.groupby("operation").agg(笔数=("trans_id", "count"),
                                     金额=("amount", "sum"))
           .query("笔数 >= 1000").sort_values("金额"))
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.barh(g.index, g["金额"], color=PALETTE_SAFE(len(g)), height=0.6)
    for i, (n, r) in enumerate(g.iterrows()):
        ax.text(r["金额"] * 1.01, i, f"{r['金额']/1e8:.1f}亿 / {r['笔数']:,} 笔",
                va="center", fontsize=9.5, color=C["text"])
    ax.set_xlim(0, g["金额"].max() * 1.32)
    ax.set_title("② 分组聚合: 按交易类型统计 (HAVING 笔数 >= 1000)")
    ax.set_xlabel("交易金额")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e8:,.0f}亿"))
    ax.grid(axis="y", visible=False)
    ax.text(0.98, 0.05,
            "SELECT operation, COUNT(*) , SUM(amount)\nFROM trans\n"
            "GROUP BY operation\nHAVING COUNT(*) >= 1000\nORDER BY 3;",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5,
            family="DejaVu Sans",
            bbox=dict(boxstyle="round,pad=0.45", fc="#F7F9FB", ec=C["grid"]))
    clean(ax)
    save(fig, "ch08_02_分组聚合_GROUPBY")
    print(g.to_string())

    # ---------- 3. JOIN + 窗口函数 ----------
    j = (tr.merge(T["accounts"][["account_id", "district_id"]], on="account_id", how="left")
           .merge(T["disp"][["account_id", "client_id"]], on="account_id", how="left")
           .merge(T["clients"][["client_id", "sex"]], on="client_id", how="left"))
    # 共管账户在 JOIN 后会出现两条记录, 先按账户聚合再取第一位持有人
    acct = j.groupby("account_id", as_index=False).agg(
        金额=("amount", "sum"), sex=("sex", "first"), district_id=("district_id", "first"))
    acct = acct.rename(columns={"金额": "amount"})
    acct["排名"] = acct["amount"].rank(ascending=False, method="min").astype(int)
    top = acct.nsmallest(10, "排名").sort_values("amount")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.0), width_ratios=[1.1, 1.0],
                             gridspec_kw={"wspace": 0.30})
    ax = axes[0]
    ax.barh([f"账户 {int(r.account_id)}" for r in top.itertuples()],
            top["amount"] / 1e4, height=0.62,
            color=[C["primary"] if s == "男" else "#B07AA1" for s in top["sex"]])
    for i, (v, s) in enumerate(zip(top["amount"] / 1e4, top["sex"])):
        ax.text(v * 1.01, i, f"{v:,.0f}万 ({s})", va="center", fontsize=9,
                color=C["text"])
    ax.set_xlim(0, top["amount"].max() / 1e4 * 1.30)
    ax.set_title("③ 多表 JOIN + 窗口函数: 交易额 Top 10 账户")
    ax.set_xlabel("累计交易金额 (万元)")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    clean(ax)

    ax = axes[1]
    ax.axis("off")
    ax.text(0.02, 0.95,
            "SELECT a.account_id, c.sex, SUM(t.amount) AS amt,\n"
            "       RANK() OVER (ORDER BY SUM(t.amount) DESC) AS rk\n"
            "FROM trans t\n"
            "JOIN disp d     ON t.account_id = d.account_id\n"
            "JOIN clients c  ON d.client_id  = c.client_id\n"
            "JOIN accounts a ON t.account_id = a.account_id\n"
            "GROUP BY a.account_id, c.sex\n"
            "ORDER BY rk\nLIMIT 10;",
            family="DejaVu Sans", fontsize=10.5, va="top", color=C["text"],
            linespacing=1.9,
            bbox=dict(boxstyle="round,pad=0.6", fc="#F7F9FB", ec=C["grid"]))
    save(fig, "ch08_03_JOIN与窗口函数")

    print("\nTop10:\n", top.to_string(index=False))
    print("图片输出:", FIG)


def PALETTE_SAFE(n):
    return [PAL[i % len(PAL)] for i in range(n)]


if __name__ == "__main__":
    main()
