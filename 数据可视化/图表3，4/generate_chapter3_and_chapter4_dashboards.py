#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第三章动态图表重绘 + 第四章销售/人力资源数据看板生成器。

输出：
  第三章_7图数据展板.png
  第三章_动态图表_7张.zip
  第四章_销售数据可视化看板.png
  第四章_人力资源数据可视化看板.png

依赖与 generate_charts_and_dashboard.py 相同：
  numpy>=1.26,<2
  matplotlib>=3.8,<3.10
  openpyxl>=3.1
  pillow>=10
"""

from __future__ import annotations

import math
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from openpyxl import load_workbook
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from generate_charts_and_dashboard import (  # noqa: E402
    BG, BG_DARK, TEXT, MUTED, GRID, BLUE, BLUE_2, BLUE_DARK,
    LIGHT_BLUE, PINK, RED, DARK_RED, YELLOW, ORANGE, PURPLE, TEAL,
    frame, clean_axes, rounded_bar, value, col_values, find_font,
)

CH3_DEFAULT = Path(r"C:\Users\dell\Desktop\第三章 动态图表.xlsm")
SALES_DEFAULT = Path(r"C:\Users\dell\Desktop\第四章 销售看板参考.xlsx")
HR_DEFAULT = Path(r"C:\Users\dell\Desktop\第四章 人力资源可视化看板.xlsx")


def panel(fig, rect, title=""):
    box = patches.FancyBboxPatch(
        (rect[0], rect[1]), rect[2], rect[3],
        boxstyle="round,pad=0.004,rounding_size=0.008",
        transform=fig.transFigure, facecolor="#232C60",
        edgecolor="#3A4E7B", linewidth=1.1, zorder=-2
    )
    fig.add_artist(box)
    ax = fig.add_axes(rect, facecolor="#232C60")
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.text(0.5, 0.98, title, transform=ax.transAxes, ha="center",
                va="top", fontsize=11.5, color=TEXT, fontweight="bold")
    return ax


def donut(ax, values, colors, center_text="", small_text="",
          start=90, width=0.28, text_size=23):
    vals = np.asarray(values, dtype=float)
    total = vals.sum()
    start_angle = start
    for v, c in zip(vals, colors):
        extent = 360 * v / total
        ax.add_patch(patches.Wedge((0, 0), 1.0, start_angle, start_angle + extent,
                                   width=width, facecolor=c, edgecolor="#232C60",
                                   linewidth=1.0))
        start_angle += extent
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25); ax.axis("off")
    if center_text:
        ax.text(0, 0.06, center_text, ha="center", va="center", color=TEXT,
                fontsize=text_size, fontweight="bold")
    if small_text:
        ax.text(0, -0.25, small_text, ha="center", va="center", color=MUTED,
                fontsize=8.5)


def dashboard_header(fig, title, subtitle, right_text=""):
    fig.text(0.035, 0.955, title, ha="left", va="top", fontsize=26,
             color="#E9EFFF", fontweight="bold")
    fig.text(0.037, 0.905, subtitle, ha="left", va="top", fontsize=11,
             color="#9FB1D6")
    if right_text:
        fig.text(0.965, 0.94, right_text, ha="right", va="top",
                 fontsize=12, color="#DCE7FF",
                 bbox=dict(boxstyle="round,pad=0.32", fc="#0D69B0", ec="#70C9F2"))
    fig.add_artist(patches.Rectangle((0.03, 0.875), 0.94, 0.002,
                                     transform=fig.transFigure, color="#465B8D"))


# ---------------- 第三章：动态图表重绘 ----------------

def render_ch3_bar(ws, out):
    source = "*注：数据来源于公司销售系统"
    idx = int(value(ws, "J3", 5))
    row = 3 + idx
    cats = [value(ws, f"{c}3") for c in "CDEFGH"]
    vals = [value(ws, f"{c}{row}") for c in "CDEFGH"]
    fig, ax = frame("2022年上半年各区域销售情况", "", source,
                    ax_rect=(0.09, 0.15, 0.85, 0.61), title_size=20)
    x = np.arange(len(cats))
    for xi, yi in zip(x, vals):
        rounded_bar(ax, xi, 0.42, yi, "#1CA7DE", "#1A94D1", radius=0.018)
    for xi, yi in zip(x, vals):
        ax.text(xi, yi + 15, f"{yi:.0f}", color=TEXT, fontsize=7.5,
                ha="center", va="bottom")
    ax.set_xticks(x, cats)
    ax.set_ylim(0, 880)
    ax.set_yticks(np.arange(0, 801, 200))
    ax.yaxis.set_visible(True)
    ax.tick_params(axis="y", labelsize=7)
    clean_axes(ax, grid_axis="y", hide_y=False)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def _race_ring(ax, radius, width, extent, color):
    ax.add_patch(patches.Wedge((0, 0), radius, 90, 90 - extent,
                               width=width, facecolor=color, edgecolor=BG,
                               linewidth=0.7, alpha=0.98))


def render_ch3_runway(ws, out):
    month = value(ws, "Q5", "4月")
    months = [value(ws, f"{c}3") for c in "CDEF"]
    mcol = "CDEF"[months.index(month)]
    cats = col_values(ws, "B", 4, 9)
    vals = [value(ws, f"{mcol}{r}") for r in range(4, 10)]
    total = sum(vals)
    source = "*注：数据来源公司人力资源系统"
    fig, ax = frame(f"{month}公司总人数为{total}", "", source,
                    ax_rect=(0.12, 0.10, 0.76, 0.72), title_align="center",
                    title_y=0.91, title_size=17)
    ax.set_aspect("equal"); ax.set_xlim(-1.28, 2.05); ax.set_ylim(-1.15, 1.15); ax.axis("off")
    colors = [PURPLE, BLUE_2, TEAL, YELLOW, ORANGE, PINK]
    for i, (cat, val, color) in enumerate(zip(cats, vals, colors)):
        r = 1.00 - i * 0.13
        extent = (val / max(vals)) * 270
        _race_ring(ax, r, 0.10, extent, color)
        ang = math.radians(90 - extent / 2)
        lx, ly = math.cos(ang) * r, math.sin(ang) * r
        ax.plot([lx, 1.20], [ly, ly], color="#B4C2D9", linewidth=0.5)
        ax.text(1.27, ly, f"{cat}\n{val}人", color=TEXT, fontsize=6.8,
                ha="left", va="center", linespacing=1.0)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_ch3_nightingale(ws, out):
    idx = int(value(ws, "I3", 1))
    row = 3 + idx
    cats = [value(ws, f"{c}3") for c in "CDEFG"]
    vals = [value(ws, f"{c}{row}") for c in "CDEFG"]
    colors = [PINK, YELLOW, TEAL, BLUE_2, PURPLE]
    source = "*注：数据来源公司网站"
    fig, ax = frame("2022年6月30日流量来源分布", "", source,
                    ax_rect=(0.06, 0.10, 0.88, 0.68), title_size=19)
    ax.set_aspect("equal"); ax.set_xlim(-1.35, 2.15); ax.set_ylim(-1.15, 1.15); ax.axis("off")
    start = 90
    for cat, val, color in zip(cats, vals, colors):
        extent = -(val * 360)
        ax.add_patch(patches.Wedge((0, 0), 1.0, start, start + extent,
                                   width=0.35, facecolor=color, edgecolor=BG,
                                   linewidth=1.0))
        mid = math.radians(start + extent / 2)
        lx, ly = math.cos(mid) * 1.13, math.sin(mid) * 1.13
        ax.plot([math.cos(mid) * 0.99, lx], [math.sin(mid) * 0.99, ly],
                color="#AEBED7", linewidth=0.55)
        ax.text(lx + (0.05 if lx >= 0 else -0.05), ly,
                f"{cat}，{val * 100:.0f}%", color=MUTED, fontsize=7.2,
                ha="left" if lx >= 0 else "right", va="center")
        start += extent
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_ch3_combo(ws, out):
    cats = [value(ws, f"{c}3") for c in "CDEF"]
    sales = [value(ws, f"{c}4") for c in "CDEF"]
    profit = [value(ws, f"{c}5") for c in "CDEF"]
    rates = [profit[i] / sales[i] for i in range(len(cats))]
    source = "*注：数据来源公司销售系统"
    fig, ax = frame("2022年化妆品销售情况", "", source,
                    ax_rect=(0.09, 0.14, 0.85, 0.61), title_size=20)
    x = np.arange(len(cats)); w = 0.29
    for xi, a, b in zip(x, sales, profit):
        rounded_bar(ax, xi - w / 2, w, a, BLUE, BLUE_2, radius=0.015)
        rounded_bar(ax, xi + w / 2, w, b, PINK, "#FF6A87", radius=0.015)
        ax.text(xi - w / 2, a + 45, f"{a:.0f}", color=TEXT, fontsize=6.5, ha="center")
        ax.text(xi + w / 2, b + 45, f"{b:.0f}", color=TEXT, fontsize=6.5, ha="center")
    yline = [3000 + r * 900 for r in rates]
    ax.plot(x, yline, color=YELLOW, linewidth=1.3, marker="o", markersize=3)
    for xi, r in zip(x, rates):
        ax.text(xi, 3000 + r * 900 + 80, f"{r * 100:.0f}%", color=TEXT,
                fontsize=6.8, ha="center")
    ax.set_xticks(x, cats)
    ax.set_ylim(0, 3900)
    clean_axes(ax, grid_axis=None, hide_y=True)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_ch3_wage(ws, out):
    cats = col_values(ws, "B", 4, 8)
    vals = col_values(ws, "C", 4, 8)
    source = "*注：数据来源于人力资源管理系统，统计日期截至2022.03.31"
    fig, ax = frame("各学历平均工资情况", "", source,
                    ax_rect=(0.11, 0.16, 0.82, 0.60), title_size=20)
    x = np.arange(len(cats))
    for xi, yi in zip(x, vals):
        rounded_bar(ax, xi, 0.43, yi, "#35A9DA", "#52BEE7", radius=0.018)
        ax.text(xi, yi + 20, f"{yi:,.2f}", color=TEXT, fontsize=6.7,
                ha="center", va="bottom")
    ax.set_xticks(x, cats)
    ax.set_ylim(0, 8000)
    ax.set_yticks([])
    clean_axes(ax, grid_axis=None, hide_y=True)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_ch3_jade(ws, out):
    period = value(ws, "H3", "近30天")
    row = 4 if period == "近7天" else 5
    cats = [value(ws, f"{c}3") for c in "CDEF"]
    vals = [value(ws, f"{c}{row}") for c in "CDEF"]
    colors = [PINK, YELLOW, TEAL, BLUE_2]
    source = "*注：数据来源公司网站"
    fig, ax = frame("流量来源分布", "", source,
                    ax_rect=(0.12, 0.10, 0.76, 0.70), title_align="center",
                    title_y=0.92, title_size=18)
    ax.set_aspect("equal"); ax.set_xlim(-1.28, 1.85); ax.set_ylim(-1.15, 1.15); ax.axis("off")
    for i, (cat, val, color) in enumerate(zip(cats, vals, colors)):
        r = 1.0 - i * 0.17
        extent = val * 360
        ax.add_patch(patches.Wedge((0, 0), r, 90, 90 - extent,
                                   width=0.13, facecolor=color, edgecolor=BG,
                                   linewidth=0.8))
        mid = math.radians(90 - extent / 2)
        lx, ly = math.cos(mid) * (r + 0.03), math.sin(mid) * (r + 0.03)
        ax.plot([lx, 1.05], [ly, ly], color="#B8C4D8", linewidth=0.5)
        ax.text(1.12, ly, f"{cat} {val * 100:.0f}%", color=TEXT,
                fontsize=6.6, ha="left", va="center")
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_ch3_slide(ws, out):
    month_idx = int(value(ws, "K2", 1))
    mcol = "CDEFGH"[month_idx - 1]
    month = value(ws, f"{mcol}2", "1月")
    cats = col_values(ws, "B", 3, 8)
    vals = [value(ws, f"{mcol}{r}") for r in range(3, 9)]
    source = "*注：数据来源于公司销售系统，日期截至2022.06.30"
    fig, ax = frame("2022年上半年区域销量目标达成率情况", "", source,
                    ax_rect=(0.12, 0.16, 0.82, 0.60), title_size=18)
    y = np.arange(len(cats))
    for yi, p in zip(y, vals):
        ax.barh(yi, 1, height=0.24, color="#8997AD", alpha=0.85)
        ax.barh(yi, p, height=0.24, color=BLUE)
        ax.scatter([p], [yi], s=130, color=BLUE_2, edgecolor="#D8EEF9",
                   linewidth=0.7, zorder=4)
        ax.text(p + 0.03, yi, f"{p * 100:.0f}%", color=TEXT, fontsize=7,
                ha="left", va="center")
    ax.set_yticks(y, cats)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(len(cats) - 0.5, -0.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xticks([])
    for s in ax.spines.values(): s.set_visible(False)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


# ---------------- 第四章：销售数据看板 ----------------

def read_sales_data(path):
    wb = load_workbook(path, data_only=False)
    sales_ws = wb["销售明细"]
    cost_ws = wb["成本明细"]
    stat = wb["统计数据"]
    month_no = int(stat["C2"].value or 9)
    month = f"{month_no}月"

    sales_rows = []
    for r in range(2, sales_ws.max_row + 1):
        vals = [sales_ws.cell(r, c).value for c in range(1, 11)]
        if vals[0] is not None:
            sales_rows.append(vals)
    selected = [r for r in sales_rows if r[9] == month]

    monthly = np.array([sum(r[4] for r in sales_rows if r[9] == f"{m}月") / 10000
                        for m in range(1, 13)], dtype=float)
    region_counts = Counter(r[2] for r in selected)
    courier_counts = Counter(r[3] for r in selected)
    category_counts = Counter(r[7] for r in selected)
    product = {}
    for r in selected:
        name = r[8]
        product.setdefault(name, [0.0, 0])
        product[name][0] += r[4]
        product[name][1] += 1
    top3 = sorted(((name, amount / 10000, qty) for name, (amount, qty) in product.items()),
                  key=lambda x: x[1], reverse=True)[:3]

    costs = {}
    for r in range(2, cost_ws.max_row + 1):
        if cost_ws.cell(r, 1).value == month:
            cat = cost_ws.cell(r, 2).value
            costs[cat] = costs.get(cat, 0.0) + float(cost_ws.cell(r, 3).value or 0)
    total_sales = sum(r[4] for r in selected)
    total_profit = sum(r[6] for r in selected)
    rate = total_profit / total_sales if total_sales else 0
    prev_month = f"{month_no - 1}月" if month_no > 1 else "12月"
    prev_count = sum(1 for r in sales_rows if r[9] == prev_month)
    cur_count = len(selected)
    yoy = cur_count / prev_count - 1 if prev_count else 0

    return {
        "month": month, "month_no": month_no, "total_sales": total_sales,
        "total_profit": total_profit, "rate": rate, "orders": len(selected),
        "monthly": monthly, "region": region_counts, "courier": courier_counts,
        "category": category_counts, "top3": top3, "costs": costs,
        "prev_count": prev_count, "yoy": yoy,
    }


def sales_dashboard(data, out):
    fig = plt.figure(figsize=(18, 10), facecolor="#121A45")
    dashboard_header(fig, "公司销售数据可视化看板",
                     "数据口径：2021 年订单明细 | 月度动态切片", f"{data['month']} ▼")

    # 产品类别销量
    ax = panel(fig, (0.035, 0.485, 0.295, 0.35), "产品类别销量")
    cats = ["办公用品", "家具产品", "数码电子"]
    vals = [data["category"].get(c, 0) for c in cats]
    colors = [PINK, BLUE, ORANGE]
    donut(ax, vals, colors, center_text=f"{data['orders']}", small_text="总销量", text_size=26)
    for i, (c, v, col) in enumerate(zip(cats, vals, colors)):
        ax.text(1.22, 0.42 - i * 0.32, "■ " + c, color=col, fontsize=8.5,
                ha="left", va="center")
        pct = v / sum(vals) * 100 if sum(vals) else 0
        ax.text(-0.98, 0.70 - i * 0.55, f"{pct:.0f}%", color=TEXT,
                fontsize=7.5, ha="center")

    # 成本费用四张卡
    ax = panel(fig, (0.035, 0.075, 0.295, 0.36), "成本费用")
    ax.axis("off")
    cost_order = ["推广", "人工", "产品", "其他"]
    positions = [(0.07, 0.53), (0.55, 0.53), (0.07, 0.05), (0.55, 0.05)]
    for cat, (px, py) in zip(cost_order, positions):
        val = data["costs"].get(cat, 0)
        share = val / sum(data["costs"].values()) if data["costs"] else 0
        card = patches.FancyBboxPatch((px, py), 0.38, 0.39,
                                      boxstyle="round,pad=0.015,rounding_size=0.03",
                                      transform=ax.transAxes, facecolor="#263268",
                                      edgecolor="#5A89BF", linewidth=0.8)
        ax.add_patch(card)
        ax.text(px + 0.05, py + 0.30, cat, color=TEXT, fontsize=10,
                transform=ax.transAxes)
        ax.text(px + 0.33, py + 0.23, f"{cat}费用\n合计", color=MUTED,
                fontsize=6.8, ha="center", va="center", transform=ax.transAxes)
        ax.text(px + 0.34, py + 0.08, f"{val:.2f}", color=TEXT, fontsize=8,
                ha="center", transform=ax.transAxes)
        cx, cy, r = px + 0.10, py + 0.14, 0.075
        ax.add_patch(patches.Circle((cx, cy), r, transform=ax.transAxes,
                                    facecolor="none", edgecolor="#4B5F91", linewidth=2.5,
                                    clip_on=False))
        arc = patches.Arc((cx, cy), 2*r, 2*r, theta1=270, theta2=270 + share * 360,
                          transform=ax.transAxes, color="#7AD5F4", linewidth=3)
        ax.add_patch(arc)
        ax.text(cx, cy, f"{share * 100:.2f}%", color=TEXT, fontsize=6.2,
                ha="center", va="center", transform=ax.transAxes)

    # 销量环比
    ax = panel(fig, (0.355, 0.77, 0.285, 0.13), "销量环比")
    ax.axis("off")
    prev_month = f"{data['month_no']-1}月" if data['month_no'] > 1 else "12月"
    items = [(prev_month, data["prev_count"], "#345CA9"),
             (data["month"], data["orders"], "#66C6DE")]
    maxv = max(v for _, v, _ in items) * 1.10
    for i, (label, v, c) in enumerate(items):
        yy = 0.68 - i * 0.38
        ax.text(0.02, yy, label, color=TEXT, fontsize=9, va="center")
        ax.barh(yy, v / maxv * 0.54, left=0.16, height=0.20, color=c)
        ax.text(0.16 + v / maxv * 0.54 + 0.02, yy, f"{v}", color=TEXT,
                fontsize=8, va="center")
    ax.text(0.83, 0.58, f"{data['yoy']*100:.2f}%", color="#F45B78",
            fontsize=14, fontweight="bold", ha="center")
    ax.text(0.83, 0.30, "同比", color=TEXT, fontsize=8, ha="center")

    # Top3 表格
    ax = panel(fig, (0.355, 0.50, 0.285, 0.24), "商品销售额Top3")
    ax.axis("off")
    ax.text(0.17, 0.82, "产品", color=MUTED, fontsize=8, ha="center")
    ax.text(0.56, 0.82, "销售额(万)", color=MUTED, fontsize=8, ha="center")
    ax.text(0.84, 0.82, "销量", color=MUTED, fontsize=8, ha="center")
    for i, (name, amount, qty) in enumerate(data["top3"]):
        yy = 0.62 - i * 0.25
        ax.text(0.05, yy, ["1", "2", "3"][i], color=YELLOW, fontsize=12,
                fontweight="bold", ha="center", va="center")
        ax.text(0.18, yy, name, color=TEXT, fontsize=7.2, ha="left", va="center")
        ax.add_patch(patches.FancyBboxPatch((0.46, yy - 0.07), 0.20, 0.14,
                                            boxstyle="square,pad=0.005",
                                            transform=ax.transAxes, facecolor="#3F5BA6",
                                            edgecolor="#7288C8", linewidth=0.7))
        ax.text(0.56, yy, f"{amount:.2f}", color=TEXT, fontsize=7.5,
                ha="center", va="center")
        ax.text(0.84, yy, f"{qty}", color=TEXT, fontsize=7.5,
                ha="center", va="center")


    # 利润额占比销售
    ax = panel(fig, (0.355, 0.075, 0.285, 0.38), "利润额占比销售额")
    donut(ax, [data["rate"], 1 - data["rate"]], [PINK, "#58B7D6"],
          center_text=f"{data['rate']*100:.2f}%", small_text="利润率", text_size=21)
    ax.text(1.25, 0.28, "利润额(万)", color=MUTED, fontsize=8, ha="left")
    ax.text(1.25, 0.04, f"{data['total_profit']/10000:.2f}", color=PINK,
            fontsize=12, fontweight="bold", ha="left")
    ax.text(1.25, -0.30, "销售额(万)", color=MUTED, fontsize=8, ha="left")
    ax.text(1.25, -0.54, f"{data['total_sales']/10000:.2f}", color=PINK,
            fontsize=12, fontweight="bold", ha="left")

    # 月度销售额
    ax = panel(fig, (0.675, 0.72, 0.29, 0.22), "各月销售额(万)")
    x = np.arange(12)
    ax.fill_between(x, data["monthly"], color="#4768C7", alpha=0.90)
    ax.plot(x, data["monthly"], color="#5B7BE0", linewidth=0.9)
    ax.set_xlim(-0.2, 11.2); ax.set_ylim(0, max(data["monthly"])*1.28)
    ax.set_xticks(x, [f"{i}月" for i in range(1, 13)])
    ax.tick_params(labelsize=6.8, length=0, colors=TEXT)
    ax.grid(axis="y", color=GRID, alpha=0.18, linestyle=(0,(4,4)))
    for s in ax.spines.values(): s.set_visible(False)

    # 区域销量
    ax = panel(fig, (0.675, 0.405, 0.29, 0.26), "区域销量")
    reg_order = ["华北", "华南", "东北", "西北", "西南", "华东"]
    reg_vals = [data["region"].get(c, 0) for c in reg_order]
    x = np.arange(6)
    ax.bar(x, reg_vals, color="#6DC7DE", width=0.44)
    for xi, yi in zip(x, reg_vals):
        ax.text(xi, yi + 5, f"{yi}", color=TEXT, fontsize=7, ha="center")
    ax.set_xticks(x, reg_order)
    ax.set_ylim(0, max(reg_vals)*1.25)
    ax.set_yticks([])
    clean_axes(ax, grid_axis=None, hide_y=True)

    # 快递公司销量
    ax = panel(fig, (0.675, 0.075, 0.29, 0.27), "快递公司销量")
    exp_order = ["顺丰", "韵达", "中通", "申通", "圆通", "EMS"]
    exp_vals = [data["courier"].get(c, 0) for c in exp_order]
    x = np.arange(6)
    bar_colors = [PINK if c in ("顺丰", "圆通") else "#FF708B" for c in exp_order]
    ax.bar(x, exp_vals, color=bar_colors, width=0.40)
    for xi, yi in zip(x, exp_vals):
        ax.text(xi, yi + 5, f"{yi}", color=TEXT, fontsize=7, ha="center")
    ax.set_xticks(x, exp_order)
    ax.set_ylim(0, max(exp_vals)*1.25)
    ax.set_yticks([])
    clean_axes(ax, grid_axis=None, hide_y=True)

    fig.savefig(out, dpi=100, facecolor="#121A45")
    plt.close(fig)


# ---------------- 第四章：人力资源数据看板 ----------------

def read_hr_data(path):
    wb = load_workbook(path, data_only=False)
    ws = wb.worksheets[0]
    rows = []
    for r in range(2, ws.max_row + 1):
        if ws.cell(r, 1).value is not None:
            rows.append([ws.cell(r, c).value for c in range(1, 9)])
    age_values = [int(r[1]) for r in rows if isinstance(r[1], (int, float))]
    age_order = ["18-24", "25-29", "30-34", "35-39", "40=<"]
    age_counts = Counter(r[7] for r in rows)
    gender_counts = Counter(r[2] for r in rows)
    edu_counts = Counter(r[3] for r in rows)
    marital_counts = Counter(r[4] for r in rows)
    dept_counts = Counter(r[5] for r in rows)
    status_counts = Counter(r[6] for r in rows if r[6])
    bachelor_plus = sum(v for k, v in edu_counts.items()
                        if k in ("本科", "硕士研究生", "博士研究生"))
    return {
        "rows": rows, "total": len(rows), "avg_age": float(np.mean(age_values)),
        "age": {k: age_counts.get(k, 0) for k in age_order},
        "gender": gender_counts, "education": edu_counts,
        "marital": marital_counts, "dept": dept_counts,
        "status": status_counts, "bachelor_plus": bachelor_plus,
    }


def kpi_card(fig, rect, label, value, sub=""):
    box = patches.FancyBboxPatch(
        (rect[0], rect[1]), rect[2], rect[3],
        boxstyle="round,pad=0.004,rounding_size=0.01",
        transform=fig.transFigure, facecolor="#263268",
        edgecolor="#4A679D", linewidth=1.0
    )
    fig.add_artist(box)
    fig.text(rect[0] + 0.015, rect[1] + rect[3] - 0.02, label,
             color=MUTED, fontsize=9, ha="left", va="top")
    fig.text(rect[0] + 0.015, rect[1] + 0.018, str(value),
             color=TEXT, fontsize=23, fontweight="bold", ha="left", va="bottom")
    if sub:
        fig.text(rect[0] + rect[2] - 0.015, rect[1] + 0.018, sub,
                 color="#72D4F4", fontsize=8.5, ha="right", va="bottom")


def hr_dashboard(data, out):
    fig = plt.figure(figsize=(18, 10), facecolor="#121A45")
    dashboard_header(fig, "公司人员结构看板",
                     "2022年3月人员基础信息 | 总人数、年龄、学历、部门与流动情况",
                     "1470 人")
    total = data["total"]
    female = data["gender"].get("女", 0)
    kpi_card(fig, (0.035, 0.785, 0.21, 0.085), "总人数", total, "COUNTA")
    kpi_card(fig, (0.265, 0.785, 0.21, 0.085), "平均年龄", f"{data['avg_age']:.1f}岁", "AVERAGE")
    kpi_card(fig, (0.495, 0.785, 0.21, 0.085), "女性占比", f"{female / total * 100:.1f}%", "性别结构")
    kpi_card(fig, (0.725, 0.785, 0.24, 0.085), "本科及以上占比",
             f"{data['bachelor_plus'] / total * 100:.1f}%", "学历结构")

    # 年龄
    ax = panel(fig, (0.035, 0.43, 0.295, 0.30), "年龄")
    cats = list(data["age"].keys()); vals = list(data["age"].values())
    colors = [BLUE, ORANGE, BLUE_2, YELLOW, PINK]
    donut(ax, vals, colors, center_text=f"{total}", small_text="总人数", text_size=20)
    for i, (c, v, col) in enumerate(zip(cats, vals, colors)):
        ax.text(1.18, 0.58 - i * 0.28, f"■ {c}   {v} 人",
                color=col, fontsize=7.2, ha="left", va="center")

    # 性别
    ax = panel(fig, (0.355, 0.43, 0.285, 0.30), "性别")
    cats = ["男", "女"]; vals = [data["gender"].get(c, 0) for c in cats]
    colors = [BLUE, PINK]
    donut(ax, vals, colors, center_text=f"{vals[0]/total*100:.0f}%", small_text="男性占比", text_size=20)
    for i, (c, v, col) in enumerate(zip(cats, vals, colors)):
        ax.text(1.18, 0.34 - i * 0.34, f"■ {c}   {v} 人",
                color=col, fontsize=8, ha="left", va="center")

    # 婚姻状况
    ax = panel(fig, (0.675, 0.43, 0.29, 0.30), "婚姻状况")
    cats = ["已婚", "单身", "离异"]; vals = [data["marital"].get(c, 0) for c in cats]
    colors = [BLUE, PINK, ORANGE]
    donut(ax, vals, colors, center_text=f"{max(vals)}", small_text="最大群体已婚", text_size=20)
    for i, (c, v, col) in enumerate(zip(cats, vals, colors)):
        ax.text(1.18, 0.40 - i * 0.33, f"■ {c}   {v} 人",
                color=col, fontsize=8, ha="left", va="center")


    # 学历
    ax = panel(fig, (0.035, 0.085, 0.295, 0.31), "学历")
    cats = ["博士研究生", "硕士研究生", "本科", "专科", "专科以下"]
    vals = [data["education"].get(c, 0) for c in cats]
    colors = [PINK, BLUE_2, ORANGE, YELLOW, BLUE]
    y = np.arange(len(cats))
    maxv = max(vals)
    for yi, v, col in zip(y, vals, colors):
        ax.barh(yi, v, height=0.42, color=col)
        ax.text(v + maxv * 0.02, yi, f"{v}", color=TEXT, fontsize=7.5,
                va="center")
    ax.set_yticks(y, cats); ax.invert_yaxis()
    ax.set_xlim(0, maxv * 1.18); ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=7.5)
    clean_axes(ax, grid_axis=None, hide_y=False)

    # 部门
    ax = panel(fig, (0.355, 0.085, 0.285, 0.31), "部门")
    cats = ["研发部", "销售部", "信息技术部", "人力资源部", "财务部", "行政部"]
    vals = [data["dept"].get(c, 0) for c in cats]
    colors = [PINK, BLUE, ORANGE, YELLOW, BLUE_2, TEAL]
    y = np.arange(len(cats)); maxv = max(vals)
    for yi, v, col in zip(y, vals, colors):
        ax.barh(yi, v, height=0.46, color=col)
        ax.text(v + maxv * 0.02, yi, f"{v}", color=TEXT, fontsize=7,
                va="center")
    ax.set_yticks(y, cats); ax.invert_yaxis()
    ax.set_xlim(0, maxv * 1.20); ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=7.2)
    clean_axes(ax, grid_axis=None, hide_y=False)

    # 本月入转调
    ax = panel(fig, (0.675, 0.085, 0.29, 0.31), "本月入转调")
    cats = ["入职", "转入", "转出"]
    vals = [data["status"].get(c, 0) for c in cats]
    colors = [BLUE, BLUE_2, PINK]
    x = np.arange(len(cats))
    bars = ax.bar(x, vals, color=colors, width=0.48)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.8, f"{v}", color=TEXT, fontsize=9,
                ha="center", va="bottom")
    ax.set_xticks(x, cats)
    ax.set_ylim(0, max(vals) * 1.28)
    ax.set_yticks([])
    clean_axes(ax, grid_axis=None, hide_y=True)

    fig.text(0.035, 0.035,
             "注：按 1470 条员工基础信息实时汇总；年龄、性别、学历、婚姻、部门、入转调均按明细重新计数。",
             color="#8D9BC0", fontsize=8.5)
    fig.savefig(out, dpi=100, facecolor="#121A45")
    plt.close(fig)


def create_ch3_board(image_paths, out_png):
    cols, rows = 4, 2
    cell_w, cell_h, gap = 640, 480, 22
    header_h, footer_h = 150, 60
    width = cols * cell_w + (cols + 1) * gap
    height = header_h + rows * cell_h + (rows + 1) * gap + footer_h
    canvas = Image.new("RGB", (width, height), "#0B1231")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap + 8, 26), "第三章 · 动态图表 Python 重绘",
              font=find_font(38, bold=True), fill="#F6F8FF")
    draw.text((gap + 10, 88),
              "7 个动态图表按工作簿顺序集中呈现：柱形图、跑道图、南丁格尔圆环、组合图、透视切片、玉玦图、滑珠图",
              font=find_font(18), fill="#BECBE8")
    draw.line([(gap, header_h - 12), (width - gap, header_h - 12)],
              fill="#35456F", width=2)
    for i, p in enumerate(image_paths):
        r, c = divmod(i, cols)
        x = gap + c * (cell_w + gap)
        y = header_h + gap + r * (cell_h + gap)
        img = Image.open(p).convert("RGB")
        canvas.paste(img, (x, y))
        draw.rectangle([x - 1, y - 1, x + cell_w, y + cell_h],
                       outline="#3A4974", width=2)
        draw.rounded_rectangle([x + 10, y + 10, x + 54, y + 38],
                               radius=12, fill="#0B69AE", outline="#69C7F2", width=1)
        draw.text((x + 20, y + 14), f"{i + 1:02d}", font=find_font(16, bold=True), fill="white")
    draw.text((gap + 8, height - footer_h + 16),
              "注：数据源为《第三章 动态图表.xlsm》；当前控件状态按工作簿默认选择值重绘。",
              font=find_font(16), fill="#AEBBD8")
    canvas.save(out_png, quality=96)
    return width, height


def main():
    import argparse
    parser = argparse.ArgumentParser(description="第三章动态图表与第四章双数据看板生成器")
    parser.add_argument("--ch3", type=Path, default=CH3_DEFAULT)
    parser.add_argument("--sales", type=Path, default=SALES_DEFAULT)
    parser.add_argument("--hr", type=Path, default=HR_DEFAULT)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR)
    args = parser.parse_args()

    out_dir = args.output.resolve()
    ch3_dir = out_dir / "第三章动态图表_7张"
    ch3_dir.mkdir(parents=True, exist_ok=True)

    wb3 = load_workbook(args.ch3, data_only=False, keep_vba=True)
    ch3_specs = [
        (0, render_ch3_bar, "01_动态柱形图.png"),
        (1, render_ch3_runway, "02_动态跑道图.png"),
        (2, render_ch3_nightingale, "03_动态南丁格尔圆环图.png"),
        (3, render_ch3_combo, "04_动态组合图.png"),
        (5, render_ch3_wage, "05_透视表切片器.png"),
        (6, render_ch3_jade, "06_VBA动态玉玦图.png"),
        (7, render_ch3_slide, "07_动态滑珠图.png"),
    ]
    image_paths = []
    for idx, func, name in ch3_specs:
        p = ch3_dir / name
        func(wb3.worksheets[idx], p)
        image_paths.append(p)
        print(name)

    board = out_dir / "第三章_7图数据展板.png"
    w, h = create_ch3_board(image_paths, board)
    zip_path = out_dir / "第三章_动态图表_7张.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in image_paths:
            z.write(p, p.name)

    sales_data = read_sales_data(args.sales)
    sales_out = out_dir / "第四章_销售数据可视化看板.png"
    sales_dashboard(sales_data, sales_out)

    hr_data = read_hr_data(args.hr)
    hr_out = out_dir / "第四章_人力资源数据可视化看板.png"
    hr_dashboard(hr_data, hr_out)

    print(f"第三章展板：{board} ({w}x{h})")
    print(f"销售看板：{sales_out}")
    print(f"人力资源看板：{hr_out}")


if __name__ == "__main__":
    main()




