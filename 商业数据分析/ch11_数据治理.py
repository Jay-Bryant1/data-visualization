# -*- coding: utf-8 -*-
r"""第11章 数据治理介绍 —— 治理框架图
运行: python -X utf8 ch11_数据治理.py   输出: figures/*.png
依据: GBT 36073-2018 DCMM / GBT 34960.5-2018 数据治理规范 / IBM 数据治理统一流程
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent      # 本章交付目录
ROOT = HERE.parent                          # 项目根目录 (教材原文所在位置)
FIG = HERE                        ; FIG.mkdir(parents=True, exist_ok=True)
C = {"p": "#4E79A7", "a": "#F28E2B", "g": "#59A14F", "r": "#E15759",
     "t": "#2F3B45", "s": "#7A8794", "bg": "#F7F9FB"}
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "font.family": "sans-serif", "axes.unicode_minus": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.18})


def box(ax, x, y, w, h, text, fc, fs=11, tc="white", bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, linespacing=1.6, fontweight="bold" if bold else "normal")


fig, ax = plt.subplots(figsize=(15, 8.6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

ax.text(0.02, 0.95, "数据治理框架", fontsize=19, fontweight="bold", color=C["t"])
ax.text(0.02, 0.905, "组织保障 + 制度流程 + 治理活动, 三者缺一不可", fontsize=11, color=C["s"])

box(ax, 0.02, 0.78, 0.96, 0.085, "治理保障层：数据治理组织  ·  数据治理制度  ·  数据治理流程  ·  考核与问责",
    C["p"], 12.5)

ax.add_patch(FancyBboxPatch((0.02, 0.60), 0.96, 0.115,
                            boxstyle="round,pad=0.012,rounding_size=0.02",
                            fc=C["bg"], ec="none"))
ax.text(0.035, 0.727, "治理活动层（DCMM 八大能力域）", fontsize=10.5,
        color=C["s"], fontweight="bold")
doms = ["数据战略", "数据治理", "数据架构", "数据应用", "数据安全", "数据质量", "数据标准", "数据生存周期"]
for i, d in enumerate(doms):
    x = 0.035 + i * 0.118
    box(ax, x, 0.628, 0.105, 0.055, d, [C["p"], C["a"], C["g"], C["r"]][i % 4], 9.5)

box(ax, 0.02, 0.40, 0.30, 0.13, "数据标准\n业务术语 / 数据元 / 编码", C["g"], 10.5)
box(ax, 0.35, 0.40, 0.30, 0.13, "数据质量\n完整性 / 准确性 / 一致性", C["a"], 10.5)
box(ax, 0.68, 0.40, 0.30, 0.13, "数据安全\n分级分类 / 权限 / 脱敏", C["r"], 10.5)

box(ax, 0.02, 0.16, 0.96, 0.16,
    "支撑能力：元数据管理  ·  主数据管理  ·  数据仓库 / 数据中台  ·  数据资产目录  ·  数据服务",
    C["bg"], 11.5, C["t"])

ax.text(0.02, 0.10,
        "治理 ≠ 技术项目：它是一套“谁负责、按什么标准做、做完怎么考核”的机制。\n"
        "标准定义“什么是正确的”，质量监控“是否一直正确”，安全约束“谁可以看/改”，\n"
        "生命周期管理“数据从产生到归档的每一环”。",
        fontsize=10.5, color=C["s"], linespacing=1.9)

for y in (0.778, 0.598):
    ax.add_patch(FancyArrowPatch((0.5, y + 0.010), (0.5, y - 0.001),
                                 arrowstyle="-|>", mutation_scale=14, color=C["s"], lw=1.2))
fig.savefig(FIG / "ch11_01_数据治理框架.png"); plt.close(fig)
print("图片输出:", FIG)
