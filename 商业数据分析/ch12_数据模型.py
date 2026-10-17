# -*- coding: utf-8 -*-
r"""第12章 数据模型与数据建模 —— 数据中台分层模型
运行: python -X utf8 ch12_数据模型.py   输出: figures/*.png
依据: 阿里云数据中台零售数据模型白皮书
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


def layer(ax, y, h, title, desc, color, fs=12):
    ax.add_patch(FancyBboxPatch((0.06, y), 0.68, h,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc=color, ec="none", alpha=0.92))
    ax.text(0.40, y + h * 0.62, title, ha="center", va="center", fontsize=fs,
            color="white", fontweight="bold")
    ax.text(0.40, y + h * 0.26, desc, ha="center", va="center", fontsize=10,
            color="white", alpha=0.95)


fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.04, 0.955, "数据中台分层模型", fontsize=19, fontweight="bold", color=C["t"])
ax.text(0.04, 0.912, "贴源 → 明细 → 汇总 → 应用，逐层收敛口径", fontsize=11, color=C["s"])

layer(ax, 0.74, 0.11, "ADS 应用数据层", "面向报表/看板/接口的宽表与指标结果", C["r"])
layer(ax, 0.60, 0.11, "DWS 汇总数据层", "按主题聚合的轻度汇总 (日/月/客户/商品)", C["a"])
layer(ax, 0.46, 0.11, "DWD 明细数据层", "清洗、规范化、维度退化后的业务明细", C["p"])
layer(ax, 0.32, 0.11, "ODS 贴源数据层", "按原始结构落地的业务库/日志/文件", "#5B7FA6")

ax.add_patch(FancyBboxPatch((0.79, 0.32), 0.17, 0.53,
                            boxstyle="round,pad=0.01,rounding_size=0.02",
                            fc=C["bg"], ec=C["p"], lw=1.6))
ax.text(0.875, 0.80, "公共维度", ha="center", fontsize=11.5,
        fontweight="bold", color=C["p"])
for i, t in enumerate(["日期", "客户", "商品", "门店", "渠道", "组织"]):
    ax.text(0.875, 0.735 - i * 0.065, t, ha="center", fontsize=10, color=C["t"])
ax.text(0.875, 0.36, "一致性维度", ha="center", fontsize=9, color=C["s"])

for y in (0.735, 0.595, 0.455):
    ax.add_patch(FancyArrowPatch((0.40, y), (0.40, y - 0.028), arrowstyle="-|>",
                                 mutation_scale=16, color="white", lw=1.5))

ax.add_patch(FancyBboxPatch((0.06, 0.12), 0.90, 0.13,
                            boxstyle="round,pad=0.01,rounding_size=0.02",
                            fc=C["bg"], ec=C["s"], lw=1.2))
ax.text(0.51, 0.205, "指标 / 标签 / 数据服务", ha="center", fontsize=12,
        fontweight="bold", color=C["t"])
ax.text(0.51, 0.155, "指标体系统一口径  ·  标签体系沉淀人群  ·  API 与数据集市对外输出",
        ha="center", fontsize=10, color=C["s"])

ax.text(0.06, 0.045,
        "建模要点：① 先定粒度（一行代表什么）；② 事实表存度量、维度表存属性；\n"
        "③ 公共维度必须全局唯一，否则跨域分析会对不上；④ 汇总层只做聚合，不引入新口径。",
        fontsize=10.5, color=C["s"], linespacing=1.9)
fig.savefig(FIG / "ch12_01_数据中台分层模型.png"); plt.close(fig)
print("图片输出:", FIG)
