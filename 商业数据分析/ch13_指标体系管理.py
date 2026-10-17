# -*- coding: utf-8 -*-
r"""第13章 指标体系管理 —— 指标标准与维度树
运行: python -X utf8 ch13_指标体系管理.py   输出: figures/*.png
依据: 1、基础数据标准编制模板.xlsx / 2、指标数据标准编制模板.xlsx
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent      # 本章交付目录
ROOT = HERE.parent                          # 项目根目录 (教材原文所在位置)
SRC = ROOT / "第13章 指标体系管理" / "2、指标数据标准编制模板.xlsx"
FIG = HERE                        ; FIG.mkdir(parents=True, exist_ok=True)
C = {"p": "#4E79A7", "a": "#F28E2B", "g": "#59A14F", "r": "#E15759",
     "t": "#2F3B45", "s": "#7A8794", "bg": "#F7F9FB"}
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "font.family": "sans-serif", "axes.unicode_minus": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.18})


def box(ax, x, y, w, h, text, fc, fs=10.5, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.018",
                                fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, linespacing=1.6, fontweight="bold")


# ---------- 图1: 指标标准的三类属性 ----------
fig, ax = plt.subplots(figsize=(15, 8))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.02, 0.95, "指标数据标准：一个指标要说清三件事", fontsize=19,
        fontweight="bold", color=C["t"])

box(ax, 0.30, 0.74, 0.40, 0.10, "指标：5 月贷款申请量", C["p"], 13)
groups = [
    (0.05, "业务属性", C["g"],
     "业务定义 · 业务口径\n指标类型 · 分析维度\n所属主题域 · 负责人"),
    (0.36, "技术属性", C["a"],
     "数据源表 · 字段映射\n计算规则 (SQL/公式)\n数据类型 · 更新频率"),
    (0.67, "管理属性", C["r"],
     "指标编号 · 版本\n安全等级 · 共享范围\n审批记录 · 变更历史"),
]
for x, title, color, desc in groups:
    box(ax, x, 0.46, 0.28, 0.10, title, color, 12.5)
    ax.text(x + 0.14, 0.40, desc, ha="center", va="top", fontsize=10.5,
            color=C["t"], linespacing=1.9,
            bbox=dict(boxstyle="round,pad=0.45", fc=C["bg"], ec="#E4E7EB"))
    ax.plot([0.5, x + 0.14], [0.735, 0.565], color="#B9C4CF", lw=1.3, zorder=0)

ax.text(0.02, 0.10,
        "为什么必须写清楚：同一句“贷款申请量”，业务口径（按申请时间还是审批时间）、\n"
        "技术口径（去重主键、状态过滤）、管理口径（谁负责、谁能用）任一不同，\n"
        "两个部门就会得出两个数字。指标标准就是把这三类属性固化成可执行的字典。",
        fontsize=10.5, color=C["s"], linespacing=1.9)
fig.savefig(FIG / "ch13_01_指标标准三属性.png"); plt.close(fig)

# ---------- 图2: 维度树 ----------
try:
    d = pd.read_excel(SRC, sheet_name="维度树-账户维度")
    d = d.dropna(how="all").dropna(axis=1, how="all")
    print("维度树列:", list(d.columns)[:8])
    print(d.head(8).to_string(index=False))
except Exception as e:
    d = pd.DataFrame()
    print("维度树读取失败:", e)

fig, ax = plt.subplots(figsize=(15, 7.6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.02, 0.94, "维度树：把“账户”拆成可下钻的层级", fontsize=19,
        fontweight="bold", color=C["t"])

levels = [("账户维度", 0.02, C["p"]),
          ("一级：账户类型", 0.24, "#5B7FA6"),
          ("二级：产品大类", 0.46, C["a"]),
          ("三级：产品小类", 0.68, C["g"])]
for title, x, color in levels:
    ax.add_patch(FancyBboxPatch((x, 0.60), 0.18, 0.12,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=color, ec="none"))
    ax.text(x + 0.09, 0.71, title, ha="center", va="center", fontsize=10.5,
            color="white", fontweight="bold", wrap=True)
children = {0: ["储蓄账户", "信用账户"], 1: ["借记卡", "信用卡", "贷款账户"],
            2: ["个人", "对公"], 3: ["活期", "定期"]}
for lv in range(4):
    x = 0.02 + lv * 0.22
    for i, ch in enumerate(children[lv]):
        y = 0.44 - i * 0.10
        ax.add_patch(FancyBboxPatch((x + 0.02, y), 0.16, 0.07,
                                    boxstyle="round,pad=0.008,rounding_size=0.015",
                                    fc=C["bg"], ec="#C9D3DD", lw=1.0))
        ax.text(x + 0.10, y + 0.035, ch, ha="center", va="center", fontsize=9.5,
                color=C["t"])
        ax.plot([x + 0.09, x + 0.10], [0.595, y + 0.072], color="#C9D3DD", lw=0.9)

ax.text(0.02, 0.06,
        "建模含义：维度树决定了报表能下钻到多细。\n"
        "维度值必须带编码（如 A01 / A0101），否则跨系统关联时只能靠中文名称硬匹配，\n"
        "一旦改名，历史数据就对不上——这正是第 3 章“键的一致性”问题的根源。",
        fontsize=10.5, color=C["s"], linespacing=1.9)
fig.savefig(FIG / "ch13_02_维度树.png"); plt.close(fig)
print("图片输出:", FIG)
