#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Excel 数据可视化第二章：30 张商业图表与数据展板生成器
数据源:
  第二章 图表(前15).xlsx
  第二章 图表(后15).xlsx

输出:
  outputs/30_charts/01_*.png ... 30_*.png
  outputs/30图数据展板.png
  outputs/30图数据展板.jpg

依赖:
  python>=3.10
  numpy>=1.26,<2
  matplotlib>=3.8,<3.10
  openpyxl>=3.1
  pillow>=10
"""

from __future__ import annotations

import argparse
import math
import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
from openpyxl import load_workbook
from PIL import Image, ImageDraw, ImageFont

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS"],
    "axes.unicode_minus": False,
    "figure.dpi": 100,
    "savefig.dpi": 100,
    "savefig.facecolor": "#202957",
    "axes.facecolor": "#202957",
    "text.color": "#F4F7FF",
    "axes.labelcolor": "#F4F7FF",
    "xtick.color": "#F4F7FF",
    "ytick.color": "#F4F7FF",
})

BG = "#202957"
BG_DARK = "#151E4A"
GRID = "#D8E2F0"
TEXT = "#F4F7FF"
MUTED = "#DDE5F4"
BLUE = "#007BC7"
BLUE_2 = "#10A0E0"
BLUE_DARK = "#0B4C93"
LIGHT_BLUE = "#78BCE8"
PINK = "#FF4D70"
RED = "#D74A4A"
DARK_RED = "#9E3C47"
YELLOW = "#F4D245"
ORANGE = "#F07A31"
PURPLE = "#6247C9"
TEAL = "#00A8A8"
GRAY_BAR = "#8997B3"

FRONT_DEFAULT = Path(r"C:\Users\dell\Desktop\第二章 图表(前15).xlsx")
BACK_DEFAULT = Path(r"C:\Users\dell\Desktop\第二章 图表(后15).xlsx")


def value(ws, coord, default=None):
    v = ws[coord].value
    return default if v is None else v


def col_values(ws, col, start, end, transform=None):
    out = []
    for r in range(start, end + 1):
        v = ws[f"{col}{r}"].value
        if transform is not None:
            v = transform(v)
        out.append(v)
    return out


def source_text(fig, source):
    if source:
        fig.text(0.075, 0.038, source, ha="left", va="bottom",
                 fontsize=7.6, color=MUTED)


def frame(title, subtitle="", source="", ax_rect=(0.085, 0.16, 0.87, 0.62),
          title_size=21, subtitle_size=12.2, title_y=0.94,
          subtitle_y=0.845, title_align="left"):
    fig = plt.figure(figsize=(6.4, 4.8), facecolor=BG)
    x = 0.5 if title_align == "center" else 0.075
    fig.text(x, title_y, title, ha=title_align, va="top", fontsize=title_size,
             fontweight="bold", color=TEXT, linespacing=1.08)
    if subtitle:
        fig.text(x, subtitle_y, subtitle, ha=title_align, va="top",
                 fontsize=subtitle_size, color=TEXT, linespacing=1.2)
    ax = fig.add_axes(ax_rect)
    ax.set_facecolor(BG)
    source_text(fig, source)
    return fig, ax


def clean_axes(ax, grid_axis="y", hide_y=True, hide_x=False):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", length=0, labelsize=9, pad=6, colors=TEXT)
    ax.tick_params(axis="y", length=0, labelsize=8, pad=4, colors=MUTED)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, alpha=0.20, linestyle=(0, (5, 5)), linewidth=0.8)
        ax.set_axisbelow(True)
    if hide_y:
        ax.yaxis.set_visible(False)
    if hide_x:
        ax.xaxis.set_visible(False)


def rounded_bar(ax, x, width, height, color1, color2=None, radius=None,
                zorder=3, alpha=1.0):
    if height <= 0:
        return None
    if radius is None:
        radius = width * 0.16
    p = patches.FancyBboxPatch(
        (x - width / 2, 0), width, height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0, facecolor=color1, zorder=zorder, alpha=alpha
    )
    ax.add_patch(p)
    if color2 and color2 != color1:
        grad = np.linspace(0, 1, 180).reshape(-1, 1)
        im = ax.imshow(grad, extent=(x - width / 2, x + width / 2, 0, height),
                       origin="lower", aspect="auto",
                       cmap=LinearSegmentedColormap.from_list("bar", [color1, color2]),
                       zorder=zorder + 0.1, alpha=alpha)
        im.set_clip_path(p)
    return p


def value_labels(ax, xs, ys, fmt=lambda v: f"{v:,.0f}", color=TEXT,
                 fontsize=8, dy=None):
    span = max(ys) - min(ys) if len(ys) else 1
    off = dy if dy is not None else span * 0.025
    for x, y in zip(xs, ys):
        ax.text(x, y + off, fmt(y), ha="center", va="bottom",
                fontsize=fontsize, color=color, zorder=6)


def make_line_smooth(x, y, samples=220):
    """Catmull-Rom spline implemented with numpy; avoids SciPy dependency."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out_x, out_y = [], []
    for i in range(len(x) - 1):
        p0 = y[max(i - 1, 0)]
        p1, p2 = y[i], y[i + 1]
        p3 = y[min(i + 2, len(y) - 1)]
        t = np.linspace(0, 1, samples, endpoint=False)
        yy = 0.5 * ((2 * p1) + (-p0 + p2) * t +
                    (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 +
                    (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)
        xx = x[i] + (x[i + 1] - x[i]) * t
        out_x.extend(xx)
        out_y.extend(yy)
    out_x.append(x[-1]); out_y.append(y[-1])
    return np.asarray(out_x), np.asarray(out_y)


def render_01(ws, out):
    cats = col_values(ws, "B", 3, 8)
    vals = col_values(ws, "C", 3, 8)
    source = "*注：数据来源于公司销售系统，统计日期截至2022.03.31"
    fig, ax = frame("3月各区域销量分布",
                    "东北销量最多占总销量的22%，华南销量最低",
                    source, ax_rect=(0.085, 0.16, 0.86, 0.58))
    x = np.arange(len(cats))
    for xi, yi in zip(x, vals):
        rounded_bar(ax, xi, 0.43, yi, BLUE, BLUE_2)
    value_labels(ax, x, vals, fontsize=8)
    ax.set_xlim(-0.6, len(cats) - 0.4)
    ax.set_ylim(0, max(vals) * 1.16)
    ax.set_xticks(x, cats)
    ax.yaxis.set_visible(True)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#CBD5E5")
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_yticks(np.arange(0, 5000, 1000))
    clean_axes(ax, grid_axis="y", hide_y=False)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_02(ws, out):
    cats = col_values(ws, "B", 3, 8)
    vals = col_values(ws, "C", 3, 8)
    avg = float(np.mean(vals))
    source = "*注：数据来源于公司销售系统，统计日期截至2022.03.31"
    fig, ax = frame("3月各区域销量分布",
                    "东北销量最多占总销量的22%，华南销量最低",
                    source, ax_rect=(0.085, 0.16, 0.86, 0.58))
    x = np.arange(len(cats))
    for xi, yi in zip(x, vals):
        rounded_bar(ax, xi, 0.43, yi, BLUE, BLUE_2)
    value_labels(ax, x, vals, fontsize=8)
    ax.axhline(avg, color=YELLOW, linewidth=1.35, zorder=5)
    ax.text(5.45, avg - max(vals) * 0.035,
            f"平均值：{avg:,.0f}", color=YELLOW, fontsize=8.5,
            ha="right", va="top")
    ax.set_xlim(-0.6, len(cats) - 0.4)
    ax.set_ylim(0, max(vals) * 1.16)
    ax.set_xticks(x, cats)
    clean_axes(ax, grid_axis=None, hide_y=True)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_03(ws, out):
    cats = col_values(ws, "B", 3, 8)
    vals = col_values(ws, "C", 3, 8)
    source = "*注：数据来源于公司销售系统，统计日期截至2022.03.31"
    fig, ax = frame("3月商品销量对比",
                    "防晒销量最多，3月销量856；面膜最少，3月销量523",
                    source, ax_rect=(0.09, 0.16, 0.85, 0.58))
    x = np.arange(len(cats))
    for xi, yi in zip(x, vals):
        rounded_bar(ax, xi, 0.34, yi, BLUE_DARK, BLUE_2, radius=0.045)
    value_labels(ax, x, vals, fontsize=8)
    ax.set_xlim(-0.6, len(cats) - 0.4)
    ax.set_ylim(0, 1000)
    ax.set_xticks(x, cats)
    ax.yaxis.set_visible(True)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_yticks(np.arange(0, 1001, 200))
    clean_axes(ax, grid_axis="y", hide_y=False)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_04(ws, out):
    cats = col_values(ws, "B", 3, 10)
    vals = col_values(ws, "C", 3, 10)
    colors = [BLUE, "#7FA6B1", "#D6633C", "#EE9F36", "#F5CB50",
              "#2870B7", "#39598C", "#624A9C"]
    source = "*注：数据来源于公司销售系统，统计日期截至2022.06.30"
    fig, ax = frame("2021年商品销量情况",
                    "口红销量最好达9221，是眼影最低值2645近3.5倍",
                    source, ax_rect=(0.08, 0.15, 0.87, 0.60))
    x = np.arange(len(cats))
    for xi, yi, c in zip(x, vals, colors):
        rounded_bar(ax, xi, 0.58, yi, c, radius=0.02)
    value_labels(ax, x, vals, fontsize=7.3, dy=max(vals) * 0.018)
    ax.set_xlim(-0.6, len(cats) - 0.4)
    ax.set_ylim(0, max(vals) * 1.12)
    ax.set_xticks(x, cats)
    clean_axes(ax, grid_axis=None, hide_y=True)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_05(ws, out):
    cats = col_values(ws, "B", 3, 8)
    sales = col_values(ws, "C", 3, 8)
    profit = col_values(ws, "D", 3, 8)
    source = "*注：数据来源于公司销售系统，统计日期截至2022.06.30"
    fig, ax = frame("2021年至今季度销售额(万)和利润额\n(万)",
                    "2022年第二季度销售额首次出现下降，降幅达到15%",
                    source, ax_rect=(0.085, 0.15, 0.86, 0.57),
                    title_size=19.5, subtitle_size=11.6)
    x = np.arange(len(cats))
    w = 0.29
    for xi, a, b in zip(x, sales, profit):
        rounded_bar(ax, xi - w / 2, w, a, BLUE, BLUE_2, radius=0.02)
        rounded_bar(ax, xi + w / 2, w, b, PINK, "#FF6886", radius=0.02)
    value_labels(ax, x - w / 2, sales, fontsize=6.8, dy=max(sales) * 0.015)
    value_labels(ax, x + w / 2, profit, fontsize=6.8, dy=max(sales) * 0.015)
    ax.text(5.02, 1700, "销售\n额", color=TEXT, fontsize=6.5, ha="center",
            bbox=dict(boxstyle="square,pad=0.18", fc=BLUE_DARK, ec="#2B77BC", lw=0.7))
    ax.text(5.02, 1030, "利润\n额", color=PINK, fontsize=6.5, ha="center",
            bbox=dict(boxstyle="square,pad=0.18", fc=BG_DARK, ec=PINK, lw=0.7))
    ax.set_xlim(-0.6, len(cats) - 0.25)
    ax.set_ylim(0, 5400)
    ax.set_xticks(x, cats)
    clean_axes(ax, grid_axis=None, hide_y=True)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def _butterfly(ax, cats, left_vals, right_vals, left_color, right_color,
               max_val, fmt=lambda v: f"{v:,.0f}"):
    y = np.arange(len(cats))
    ax.barh(y, [-v for v in left_vals], height=0.40, color=left_color, zorder=3)
    ax.barh(y, right_vals, height=0.40, color=right_color, zorder=3)
    for yi, v in zip(y, left_vals):
        ax.text(-v / 2, yi, fmt(v), color="#DFF1FF", ha="center", va="center",
                fontsize=7.2, zorder=5)
    for yi, v in zip(y, right_vals):
        ax.text(v / 2, yi, fmt(v), color="#F7E5E5", ha="center", va="center",
                fontsize=7.2, zorder=5)
    for yi, c in zip(y, cats):
        ax.text(0, yi, c, ha="center", va="center", color=TEXT,
                fontsize=9, zorder=6,
                bbox=dict(boxstyle="square,pad=0.35", fc=BG, ec="none"))
    ax.text(-max_val * 0.46, -0.72, "2022", color="#55A8E8", ha="center",
            fontsize=10)
    ax.text(max_val * 0.46, -0.72, "2021", color=PINK, ha="center", fontsize=10)
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(len(cats) - 0.5, -1.0)
    ax.axis("off")


def render_06(ws, out):
    cats = col_values(ws, "B", 3, 7)
    v22 = [2238, 1531, 1426, 1321, 1215]
    v21 = [2066, 1436, 1531, 1265, 1003]
    source = "*注：数据来源于公司销售系统，统计日期截至2022.06.30"
    fig, ax = frame("2022年上半年各区域对比去年销量",
                    "2022年整体销量高于2021年，只有东北区域较2021有所下降",
                    source, ax_rect=(0.075, 0.17, 0.88, 0.59))
    _butterfly(ax, cats, v22, v21, BLUE, PINK, 2600)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_07(ws, out):
    cats = col_values(ws, "B", 3, 7)
    v22 = col_values(ws, "C", 3, 7)
    v21 = col_values(ws, "D", 3, 7)
    source = "*注：数据来源于公司销售系统，统计日期截至2022.03.31"
    fig, ax = frame("2022年第一季度销售目标完成情况",
                    "华东区域完成率最高达到36%，但是相比去年的42%有所下降",
                    source, ax_rect=(0.075, 0.17, 0.88, 0.59))
    _butterfly(ax, cats, v22, v21, BLUE, PINK, 0.52,
               fmt=lambda v: f"{v * 100:.0f}%")
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_08(ws, out):
    cats = ["华东", "西南", "西北", "东北", "华南", "华北"]
    vals = [2109, 1369, 1872, 1536, 1946, 4321]
    growth = [-0.058, -0.179, -0.159, -0.093, -0.208, -0.136]
    source = "*注：数据来源于公司销售系统，统计日期截至2022.01.01"
    fig, ax = frame("2021年各区域销量及同比情况",
                    "各区域商品销量同比去年均有下降，其中华南下降最多，同比下降20.8%",
                    source, ax_rect=(0.075, 0.18, 0.89, 0.56),
                    subtitle_size=10.5)
    y = np.arange(len(cats))
    base = 4321
    scale = 5200
    for yi, v in zip(y, vals):
        ax.barh(yi, v, height=0.42, color=BLUE_DARK, zorder=3)
        ax.barh(yi, base - v, left=v, height=0.42, color="#96C3E4", zorder=2)
        ax.barh(yi, scale - base, left=base, height=0.42, color="#A94A54", zorder=2)
        ax.text(v / 2, yi, f"{v:,}", color="#E7F3FF", fontsize=7.2,
                ha="center", va="center")
    for yi, g in zip(y, growth):
        ax.text((base + scale) / 2, yi, f"{g * 100:.1f}%", color="#F7E6E6",
                fontsize=7.5, ha="center", va="center")
    ax.set_yticks(y, cats)
    ax.set_xlim(0, scale)
    ax.set_ylim(len(cats) - 0.5, -0.5)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_09(ws, out):
    cats = col_values(ws, "B", 3, 7)
    v21 = col_values(ws, "C", 3, 7)
    v22 = col_values(ws, "D", 3, 7)
    source = "*注：数据来源于公司销售系统，统计日期截至2022.01.01"
    fig, ax = frame("2022年商品对比去年销售情况",
                    "商品整体比去年销量有所下降，其中隔离下降最多，降幅33%",
                    source, ax_rect=(0.085, 0.15, 0.86, 0.59))
    x = np.arange(len(cats))
    w = 0.27
    for xi, a, b in zip(x, v21, v22):
        rounded_bar(ax, xi - w / 2, w, a, "#36A3D9", "#66C2E7", radius=0.015)
        rounded_bar(ax, xi + w / 2, w, b, PINK, "#FF6A89", radius=0.015)
    value_labels(ax, x - w / 2, v21, fontsize=6.8, dy=max(v21) * 0.015)
    value_labels(ax, x + w / 2, v22, fontsize=6.8, dy=max(v21) * 0.015)
    ax.text(0.18, 5050, "■ 2021年", color="#63BCEB", fontsize=8)
    ax.text(1.12, 5050, "■ 2022年", color=PINK, fontsize=8)
    ax.set_xlim(-0.6, len(cats) - 0.4)
    ax.set_ylim(0, 5600)
    ax.set_xticks(x, cats)
    clean_axes(ax, grid_axis=None, hide_y=True)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_10(ws, out):
    cats = col_values(ws, "B", 4, 10)
    starts = [datetime.fromisoformat(str(v)) for v in col_values(ws, "C", 4, 10)]
    dur = col_values(ws, "D", 4, 10)
    pct = col_values(ws, "E", 4, 10)
    fig, ax = frame("2022年化妆品类目采购项目进度", "", "",
                    ax_rect=(0.16, 0.14, 0.79, 0.66), title_align="center",
                    title_y=0.94, title_size=17.5)
    base = min(starts)
    y = np.arange(len(cats))
    for yi, st, du, p in zip(y, starts, dur, pct):
        left = (st - base).days
        ax.barh(yi, du, left=left, height=0.20, color=BLUE_2, zorder=4)
        ax.text(left + du / 2, yi, f"{p * 100:.0f}%", color="#E4F3FF",
                fontsize=7, ha="center", va="center", zorder=6)
    tick_days = [0, 15, 30, 45, 60, 75, 90, 105]
    tick_dates = [(base + timedelta(days=d)).strftime("%Y/%#m/%#d")
                  for d in tick_days]
    ax.set_xticks(tick_days, tick_dates)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False,
                   labelbottom=False, labelsize=7, pad=5)
    ax.set_yticks(y, cats)
    ax.tick_params(axis="y", labelsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-1, 113)
    ax.grid(axis="x", color=GRID, alpha=0.38, linestyle=(0, (5, 6)), linewidth=0.8)
    ax.spines["bottom"].set_visible(False)
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_11(ws, out):
    cats = col_values(ws, "C", 3, 13)
    vals = col_values(ws, "D", 3, 13)
    source = "*注：数据来源于公司销售系统，统计日期截至2022.03.31"
    fig, ax = frame("化妆品品类月度销量走势",
                    "2022年销量迅速增加，1月最高，销量达到3782",
                    source, ax_rect=(0.09, 0.20, 0.85, 0.55))
    x = np.arange(len(cats))
    xs, ys = make_line_smooth(x, vals)
    ax.plot(xs, ys, color="#EF6A43", linewidth=1.15)
    peak = int(np.argmax(vals))
    ax.axvline(peak, color="#E86A52", linewidth=0.8, linestyle=(0, (4, 4)), alpha=0.8)
    ax.text(peak, vals[peak] + 100, f"{vals[peak]:.0f}", color=TEXT,
            fontsize=7.5, ha="center")
    ax.set_xticks(x, cats)
    ax.set_ylim(0, 4400)
    ax.set_yticks(np.arange(0, 4001, 1000))
    ax.yaxis.set_visible(True)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines["bottom"].set_color("#D8E2F0")
    ax.spines["bottom"].set_linewidth(0.7)
    clean_axes(ax, grid_axis="y", hide_y=False)
    fig.add_artist(patches.Rectangle((0.09, 0.115), 0.49, 0.045,
                                     transform=fig.transFigure, facecolor="#55BCDA"))
    fig.add_artist(patches.Rectangle((0.58, 0.115), 0.36, 0.045,
                                     transform=fig.transFigure, facecolor="#F2CC3F"))
    fig.text(0.335, 0.137, "2021", color="white", fontsize=9, ha="center", va="center")
    fig.text(0.76, 0.137, "2022", color="white", fontsize=9, ha="center", va="center")
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_12(ws, out):
    cats = col_values(ws, "B", 3, 10)
    vals = col_values(ws, "C", 3, 10)
    source = "*注：数据来源于公司销售系统，统计日期截至2022.08.31"
    fig, ax = frame("2022年1-8月公司计划完成率",
                    "公司整体完成率55%，4月和8月超过70%，2月和6月较低未过半",
                    source, ax_rect=(0.08, 0.16, 0.87, 0.58),
                    subtitle_size=10.5)
    x = np.arange(len(cats))
    ax.vlines(x, 0, vals, colors="#D65A61", linewidth=0.75)
    ax.scatter(x, vals, marker="D", s=17, facecolor=BG, edgecolor="#D65A61",
               linewidth=1.0, zorder=5)
    for xi, yi in zip(x, vals):
        ax.text(xi, yi + 0.018, f"{yi * 100:.2f}%", color=MUTED,
                fontsize=7.2, ha="center", va="bottom")
    ax.set_xticks(x, cats)
    ax.set_ylim(0, 0.84)
    ax.set_yticks([])
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#CBD5E5")
    ax.spines["bottom"].set_linewidth(0.8)
    clean_axes(ax, grid_axis=None, hide_y=True)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_13(ws, out):
    cats = col_values(ws, "B", 3, 8)
    v21 = col_values(ws, "C", 3, 8)
    v22 = col_values(ws, "D", 3, 8)
    source = "*注：数据来源于公司销售系统，统计日期截至2022.06.30"
    fig, ax = frame("2022年上半年各月同比去年销量",
                    "上半年同比去年增长明显，5月份同比增长最多，增长近40%",
                    source, ax_rect=(0.09, 0.17, 0.85, 0.56))
    x = np.arange(len(cats))
    ax.plot(x, v21, marker="s", color=PINK, linewidth=1.3, markersize=3.2, label="2021年")
    ax.plot(x, v22, marker="s", color="#2EA5DF", linewidth=1.3, markersize=3.2, label="2022年")
    for xi, a, b in zip(x, v21, v22):
        ax.text(xi, a - 145, f"{a}", color=TEXT, fontsize=6.7,
                ha="center", va="top")
        ax.text(xi, b + 100, f"{b}", color=TEXT, fontsize=6.7,
                ha="center", va="bottom")
    ax.set_xticks(x, cats)
    ax.set_ylim(0, 3200)
    ax.set_yticks(np.arange(0, 3001, 500))
    ax.yaxis.set_visible(True)
    ax.tick_params(axis="y", labelsize=7)
    clean_axes(ax, grid_axis="y", hide_y=False)
    ax.legend(loc="upper right", bbox_to_anchor=(0.97, 1.16), frameon=False,
              fontsize=8, labelcolor=TEXT, handlelength=2.4)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_14(ws, out):
    pct = float(value(ws, "B3"))
    source = "*注：数据来源于公司销售系统"
    fig, ax = frame("2022年上半年目标完成率",
                    "截至6月30日销售目标总体完成率达到85%",
                    source, ax_rect=(0.16, 0.11, 0.68, 0.67))
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.axis("off")
    start, extent = -60, 306
    ax.add_patch(patches.Wedge((0, 0), 0.95, start, start + extent,
                               width=0.12, facecolor="#A958DF", alpha=0.50))
    ax.add_patch(patches.Wedge((0, 0), 0.84, start, start + extent * pct,
                               width=0.08, facecolor=PINK, alpha=1.0))
    ax.add_patch(patches.Wedge((0, 0), 0.76, start, start + extent * pct,
                               width=0.025, facecolor="#FF8B9D", alpha=0.9))
    ax.text(0, 0.04, f"{pct * 100:.0f}%", ha="center", va="center",
            fontsize=29, fontweight="bold", color=TEXT)
    ax.text(0, -0.32, "目标完成率", ha="center", va="center",
            fontsize=9, color=MUTED)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def _waterball(ws, out, title, subtitle, source, wave=False):
    raw = value(ws, "B3")
    if not isinstance(raw, (int, float)):
        raw = value(ws, "B4")
    pct = float(raw)
    fig, ax = frame(title, subtitle, source,
                    ax_rect=(0.25, 0.09, 0.50, 0.72), title_size=17,
                    subtitle_size=10.5, title_align="center",
                    title_y=0.94, subtitle_y=0.84)
    ax.set_aspect("equal")
    ax.set_xlim(-1.18, 1.18); ax.set_ylim(-1.18, 1.18); ax.axis("off")
    outer = patches.Circle((0, 0), 1.0, facecolor="#1B2A5D",
                           edgecolor="#2A8FC8", linewidth=0.8)
    inner = patches.Circle((0, 0), 0.91, facecolor="#004B96",
                           edgecolor="#2587C3", linewidth=1.2)
    ax.add_patch(outer); ax.add_patch(inner)
    clip = patches.Circle((0, 0), 0.90, transform=ax.transData)
    level = -0.90 + 1.80 * pct
    xx = np.linspace(-0.95, 0.95, 220)
    yy = level + (0.04 * np.sin(4.5 * xx) if wave else 0)
    poly = ax.fill_between(xx, -1.0, yy, color="#087CC0", alpha=0.96, zorder=3)
    poly.set_clip_path(clip)
    ax.add_patch(patches.Circle((0, 0), 1.08, fill=False,
                                edgecolor="#207CC0", linewidth=0.9, alpha=0.8))
    ax.text(0, 0.02, f"{pct * 100:.0f}%", ha="center", va="center",
            fontsize=31, color=TEXT, zorder=8)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_15(ws, out):
    _waterball(ws, out, "2022年上半年目标完成率",
               "截至6月30日销售目标总体完成率达到65%",
               "*注：数据来源于公司销售系统", wave=False)


def render_16(ws, out):
    _waterball(ws, out, "本科及以上学历员工占比",
               "6月30日最新统计数据本科及以上员工占比65%",
               "*注：数据来源于公司人力资源系统", wave=True)


def _ring_segment(ax, radius, width, start, extent, color, alpha=1.0):
    ax.add_patch(patches.Wedge((0, 0), radius, start, start + extent,
                               width=width, facecolor=color, alpha=alpha,
                               edgecolor=BG, linewidth=0.5, zorder=4))


def render_17(ws, out):
    cats = col_values(ws, "B", 3, 6)
    vals = col_values(ws, "C", 3, 6)
    colors = [BLUE_2, TEAL, YELLOW, PINK]
    source = "*注：数据来源于公司人力资源系统，统计日期截至2022.06.30"
    fig, ax = frame("2022年上半年年龄分布",
                    "公司平均年龄32.5，23-30员工比例最高",
                    source, ax_rect=(0.08, 0.14, 0.84, 0.64))
    ax.set_aspect("equal"); ax.set_xlim(-1.3, 2.55); ax.set_ylim(-1.15, 1.15); ax.axis("off")
    for i, (cat, val, color) in enumerate(zip(cats, vals, colors)):
        r = 0.94 - i * 0.17
        extent = -(val * 360)
        _ring_segment(ax, r, 0.12, 90, extent, color)
        angle = math.radians(90 + extent / 2)
        x, y = math.cos(angle) * r, math.sin(angle) * r
        ax.plot([x, 1.20], [y, y], color="#AAB5CB", linewidth=0.55)
        ax.text(1.29, y, f"{val * 100:.1f}%", color=TEXT, fontsize=7.5,
                ha="left", va="center")
    for i, cat in enumerate(cats):
        ax.text(-1.22, 0.70 - i * 0.30, cat, color=MUTED, fontsize=8.5,
                ha="left", va="center")
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_18(ws, out):
    cats = col_values(ws, "B", 3, 8)
    vals = col_values(ws, "C", 3, 8)
    total = sum(vals)
    colors = [BLUE_2, "#1A9BC1", TEAL, YELLOW, "#EF9F32", PINK]
    source = "*注：数据来源于公司人力资源系统，统计日期截至2022.06.30"
    fig, ax = frame("2022年上半年各部门人数",
                    f"公司总人数{total}，销售部人数最多451，占比27%",
                    source, ax_rect=(0.10, 0.12, 0.80, 0.66))
    ax.set_aspect("equal"); ax.set_xlim(-1.25, 2.40); ax.set_ylim(-1.15, 1.15); ax.axis("off")
    for i, (cat, val, color) in enumerate(zip(cats, vals, colors)):
        r = 0.98 - i * 0.13
        extent = -(val / max(vals)) * 240
        _ring_segment(ax, r, 0.09, 90, extent, color)
        angle = math.radians(90 + extent / 2)
        x, y = math.cos(angle) * r, math.sin(angle) * r
        ax.plot([x, 1.16], [y, y], color="#B6C0D2", linewidth=0.45)
        ax.text(1.23, y, f"{val}\n{val / total * 100:.0f}%", color=TEXT,
                fontsize=6.8, ha="left", va="center", linespacing=0.95)
    ax.text(-1.22, 0.98, "部门", color=MUTED, fontsize=8, ha="left", va="top")
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_19(ws, out):
    cats = col_values(ws, "B", 3, 8)
    vals = col_values(ws, "C", 3, 8)
    colors = ["#E2576E", "#F5B849", "#38B6B0", "#4A94D1", "#7656B8", "#E15A8D"]
    source = "*注：数据来源于公司人力资源系统，统计日期截至2022.01.01"
    fig, ax = frame("2021年各部门人数分布",
                    "公司总人数1664，销售部人数最多451，占比29.2%",
                    source, ax_rect=(0.08, 0.13, 0.84, 0.65))
    ax.set_aspect("equal"); ax.set_xlim(-1.35, 2.2); ax.set_ylim(-1.15, 1.15); ax.axis("off")
    start = 90
    for cat, val, color in zip(cats, vals, colors):
        extent = -(val * 360)
        r = 0.62 + 0.32 * val / max(vals)
        ax.add_patch(patches.Wedge((0, 0), r, start, start + extent,
                                   facecolor=color, edgecolor=BG, linewidth=0.8))
        mid = math.radians(start + extent / 2)
        lx, ly = math.cos(mid) * (r + 0.06), math.sin(mid) * (r + 0.06)
        ax.plot([lx, lx + 0.30], [ly, ly], color="#AEB9CE", linewidth=0.45)
        ax.text(lx + 0.34, ly, f"{val * 100:.1f}%", color=TEXT, fontsize=6.7,
                ha="left", va="center")
        start += extent
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_20(ws, out):
    cats = col_values(ws, "B", 3, 6)
    vals = col_values(ws, "C", 3, 6)
    colors = [BLUE_2, YELLOW, PINK, PURPLE]
    source = "*注：数据来源于公司人力资源系统，统计日期截至2022.06.30"
    fig, ax = frame("2022年上半年各年龄段人数分布",
                    "公司平均年龄32.5，20-30员工比例最高占比37.5%",
                    source, ax_rect=(0.09, 0.13, 0.83, 0.64))
    ax.set_aspect("equal"); ax.set_xlim(-1.3, 2.0); ax.set_ylim(-1.12, 1.12); ax.axis("off")
    for i, (cat, val, color) in enumerate(zip(cats, vals, colors)):
        r = 1.0 - i * 0.16
        extent = -(val * 360)
        _ring_segment(ax, r, 0.14, 90, extent, color)
        mid = math.radians(90 + extent / 2)
        lx, ly = math.cos(mid) * (r + 0.03), math.sin(mid) * (r + 0.03)
        ax.plot([lx, 1.10], [ly, ly], color="#B3BDD0", linewidth=0.5)
        ax.text(1.18, ly, f"{val * 100:.1f}%", color=TEXT, fontsize=7,
                ha="left", va="center")
    for i, cat in enumerate(cats):
        ax.text(-1.25, 0.65 - i * 0.31, cat, color=MUTED, fontsize=8,
                ha="left", va="center")
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_21(ws, out):
    cats = col_values(ws, "B", 3, 8)
    vals = col_values(ws, "C", 3, 8)
    colors = ["#E35B78", "#F0B647", "#25AFA8", "#3D92CF", "#7152B6", "#D75883"]
    source = "*注：数据来源于公司人力资源系统，统计日期截至2022.01.01"
    fig, ax = frame("2021年各部门人数分布",
                    "公司总人数1664，销售部人数最多451，占比29.2%",
                    source, ax_rect=(0.10, 0.14, 0.80, 0.63))
    ax.set_aspect("equal"); ax.set_xlim(-1.3, 2.0); ax.set_ylim(-1.15, 1.15); ax.axis("off")
    start = 270
    for i, (cat, val, color) in enumerate(zip(cats, vals, colors)):
        extent = val * 360
        r = 0.42 + i * 0.095
        ax.add_patch(patches.Wedge((0, 0), r, start, start + extent,
                                   width=0.055, facecolor=color, edgecolor=BG,
                                   linewidth=0.7, alpha=0.95))
        mid = math.radians(start + extent / 2)
        lx, ly = math.cos(mid) * (r + 0.03), math.sin(mid) * (r + 0.03)
        ax.plot([lx, 1.03], [ly, ly], color="#BCC5D7", linewidth=0.45)
        ax.text(1.10, ly, f"{cat} {val * 100:.1f}%", color=TEXT,
                fontsize=6.6, ha="left", va="center")
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_22(ws, out):
    score = float(value(ws, "H3"))
    fig, ax = frame("2022年6月29日公司整体运营指数良好", "", "",
                    ax_rect=(0.16, 0.10, 0.68, 0.72), title_align="center",
                    title_y=0.92, title_size=16.8)
    ax.set_aspect("equal"); ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.15, 1.15); ax.axis("off")
    start, sweep = 210, 300
    bands = [(0.00, 0.30, "#EF4B62"), (0.30, 0.65, "#F1D84A"),
             (0.65, 1.00, "#21D6A5")]
    for a, b, c in bands:
        ax.add_patch(patches.Wedge((0, 0), 1.0, start + sweep * a,
                                   start + sweep * b, width=0.16,
                                   facecolor=c, edgecolor=BG, linewidth=0.5))
    ax.add_patch(patches.Wedge((0, 0), 0.73, start, start + sweep,
                               width=0.03, facecolor="#5AB5DF", edgecolor="none"))
    pct = max(0, min(score - 50, 100)) / 100
    angle = math.radians(start + sweep * pct)
    ax.plot([0, math.cos(angle) * 0.74], [0, math.sin(angle) * 0.74],
            color="#E8F1FB", linewidth=1.8)
    ax.add_patch(patches.Circle((0, 0), 0.09, facecolor="#E9F2FC", edgecolor=BG))
    for tick in range(0, 11):
        a = math.radians(start + sweep * tick / 10)
        r1, r2 = 0.79, 0.69
        ax.plot([math.cos(a) * r1, math.cos(a) * r2],
                [math.sin(a) * r1, math.sin(a) * r2],
                color=MUTED, linewidth=0.7)
        tv = 50 + tick * 10
        tx, ty = math.cos(a) * 0.60, math.sin(a) * 0.60
        ax.text(tx, ty, f"{tv}", color=TEXT, fontsize=6.5,
                ha="center", va="center")
    ax.text(0, -0.34, f"{score:.0f}", color=TEXT, fontsize=24,
            fontweight="bold", ha="center", va="center")
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_23(ws, out):
    cats = col_values(ws, "B", 3, 8)
    vals = col_values(ws, "C", 3, 8)
    growth = [0.27, 0.31, 0.14, 0.36, 0.14, 0.05]
    source = "*注：数据来源于公司销售系统"
    fig, ax = frame("近六年销售量及增长率",
                    "平台销量近6年持续增长，但近两年增长率有所放缓",
                    source, ax_rect=(0.085, 0.15, 0.86, 0.59))
    x = np.arange(len(cats))
    for xi, yi in zip(x, vals):
        rounded_bar(ax, xi, 0.31, yi, BLUE, "#0FA1DF", radius=0.035)
    value_labels(ax, x, vals, fontsize=7.2, dy=max(vals) * 0.018)
    yline = [g * 10000 for g in growth]
    ax.plot(x, yline, color=PINK, marker="o", markersize=3.3,
            linewidth=1.25, zorder=6)
    for xi, g in zip(x, growth):
        ax.text(xi, g * 10000 + 120, f"{g * 100:.0f}%", color=TEXT,
                fontsize=7.3, ha="center", va="bottom")
    ax.set_xticks(x, cats)
    ax.set_ylim(0, 5000)
    clean_axes(ax, grid_axis=None, hide_y=True)
    ax.text(0.60, 4450, "■ 销售量", color="#229BD8", fontsize=8)
    ax.text(1.65, 4450, "●— 同比", color=PINK, fontsize=8)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_24(ws, out):
    cats = col_values(ws, "B", 3, 8)
    actual = col_values(ws, "C", 3, 8)
    target = col_values(ws, "D", 3, 8)
    source = "*注：数据来源于公司销售系统，统计日期截至2022.06.30"
    fig, ax = frame("2022年上半年各商品销量完成情况",
                    "防晒整体销量最好，达到856，面霜远超目标，超额完成30%",
                    source, ax_rect=(0.085, 0.16, 0.86, 0.58), subtitle_size=11.3)
    x = np.arange(len(cats))
    w = 0.34
    for xi, a, t in zip(x, actual, target):
        tb = patches.Rectangle((xi - w / 2, 0), w, t, facecolor="#184A79",
                               edgecolor="#94D7F6", linewidth=0.8, zorder=2)
        ax.add_patch(tb)
        rounded_bar(ax, xi, w * 0.72, a, BLUE, "#14A3E1", radius=0.018, zorder=4)
    value_labels(ax, x, actual, fontsize=7.2, dy=max(target) * 0.02)
    ax.text(-0.35, 1010, "□ 目标销量", color="#BFD7E9", fontsize=7.2)
    ax.text(0.96, 1010, "■ 实际销量", color="#1B9AD8", fontsize=7.2)
    ax.set_xticks(x, cats)
    ax.set_xlim(-0.6, len(cats) - 0.4)
    ax.set_ylim(0, 1120)
    clean_axes(ax, grid_axis=None, hide_y=True)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_25(ws, out):
    cats = col_values(ws, "B", 4, 9)
    actual = col_values(ws, "C", 4, 9)
    target = col_values(ws, "D", 4, 9)
    source = "*注：数据来源于公司销售系统，统计日期截至2022.06.30"
    fig, ax = frame("2022年上半年各商品销量完成情况",
                    "防晒整体销量最好，达到856，面霜远超目标，超额完成30%",
                    source, ax_rect=(0.085, 0.16, 0.86, 0.58), subtitle_size=11.3)
    y = np.arange(len(cats))
    for yi, a, t in zip(y, actual, target):
        ax.barh(yi, 600, height=0.38, color="#8995A9", alpha=0.75, zorder=1)
        ax.barh(yi, 200, left=600, height=0.38, color=BLUE, alpha=0.85, zorder=1)
        ax.barh(yi, 200, left=800, height=0.38, color=TEAL, alpha=0.9, zorder=1)
        ax.barh(yi, a, height=0.20, color="#087CC2", zorder=4)
        ax.plot([t, t], [yi - 0.18, yi + 0.18], color=YELLOW,
                linewidth=1.7, zorder=5)
    ax.text(0.02, 5.75, "■ 及格", color="#9AA7BA", fontsize=7.2)
    ax.text(0.24, 5.75, "■ 良好", color=BLUE, fontsize=7.2)
    ax.text(0.46, 5.75, "■ 优秀", color=TEAL, fontsize=7.2)
    ax.text(0.68, 5.75, "■ 实际", color="#087CC2", fontsize=7.2)
    ax.text(0.90, 5.75, "— 目标", color=YELLOW, fontsize=7.2)
    ax.set_yticks(y, cats)
    ax.set_xlim(0, 1020)
    ax.set_ylim(-0.6, 6.2)
    ax.set_xticks(np.arange(0, 1001, 200))
    ax.tick_params(axis="y", labelsize=8)
    clean_axes(ax, grid_axis="x", hide_y=False)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_26(ws, out):
    cats = col_values(ws, "B", 3, 8)
    vals = col_values(ws, "C", 3, 8)
    growth = col_values(ws, "E", 3, 8)
    source = "*注：数据来源于公司销售系统，统计日期截至2022.06.30"
    fig, ax = frame("各区域上半年销量以同比",
                    "东北区域销量持续保持第一，华东和华南同比去年增长最多",
                    source, ax_rect=(0.085, 0.16, 0.86, 0.58), subtitle_size=11.2)
    x = np.arange(len(cats))
    for xi, yi, g in zip(x, vals, growth):
        rounded_bar(ax, xi, 0.34, yi, PINK, "#FF6683", radius=0.025)
        ax.text(xi, yi + 90, f"{yi}", color=TEXT, fontsize=7.3,
                ha="center", va="bottom")
        cy = max(vals) * 1.03 + g * 1900
        ax.scatter([xi], [cy], s=180, color=BLUE, zorder=6)
        ax.text(xi, cy, f"{g * 100:.0f}%", color=TEXT, fontsize=6.7,
                ha="center", va="center", zorder=7)
    ax.set_xticks(x, cats)
    ax.set_ylim(0, max(vals) * 1.40)
    clean_axes(ax, grid_axis=None, hide_y=True)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_27(ws, out):
    cats = col_values(ws, "B", 3, 8)
    v22 = col_values(ws, "C", 3, 8)
    v21 = col_values(ws, "D", 3, 8)
    growth = col_values(ws, "E", 3, 8)
    source = "*注：数据来源于公司销售系统，统计日期截至2022.06.30"
    fig, ax = frame("上半年各月商品销量同比去年情况",
                    "2022年相比于2021年销量都有提升，半年整体提升17%",
                    source, ax_rect=(0.085, 0.16, 0.86, 0.58), subtitle_size=10.8)
    x = np.arange(len(cats))
    w = 0.28
    for xi, a, b in zip(x, v22, v21):
        rounded_bar(ax, xi - w / 2, w, a, BLUE, "#0FA2DF", radius=0.015)
        rounded_bar(ax, xi + w / 2, w, b, PINK, "#FF6683", radius=0.015)
        ax.text(xi - w / 2, a / 2, f"{a}", color=TEXT, fontsize=6.7,
                ha="center", va="center", rotation=90)
        ax.text(xi + w / 2, b / 2, f"{b}", color=TEXT, fontsize=6.7,
                ha="center", va="center", rotation=90)
    yline = [2600 + g * 10000 for g in growth]
    ax.plot(x, yline, color=YELLOW, linewidth=1.25, marker="o", markersize=2.8)
    for xi, g in zip(x, growth):
        ax.text(xi, 2600 + g * 10000 + 110, f"{g * 100:.0f}%", color=TEXT,
                fontsize=6.7, ha="center", va="bottom")
    ax.set_xticks(x, cats)
    ax.set_ylim(0, 4100)
    clean_axes(ax, grid_axis=None, hide_y=True)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_28(ws, out):
    cats = col_values(ws, "B", 3, 17)
    vals = col_values(ws, "C", 3, 17)
    pairs = [(c, v) for c, v in zip(cats, vals) if c not in (None, "") and v is not None]
    cats = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    source = "*注：数据来源于公司销售系统，统计日期截至2021.12.31"
    fig, ax = frame("2021年各月化妆品销量走势",
                    "2021年第二季度销量最多9673，9月单月销量最大3621",
                    source, ax_rect=(0.08, 0.16, 0.87, 0.58))
    months, xvals, colors = [], [], []
    group_colors = ["#353B86", "#6C3436", "#5D4D34", "#53315E"]
    for gi in range(4):
        s = gi * 3
        months.extend(cats[s:s + 3])
        xvals.extend(vals[s:s + 3])
        colors.extend([["#4E70D8", "#3B5EC7", "#3C71D0"],
                       ["#F26A26", "#F7832B", "#F49735"],
                       ["#D5A425", "#E5B32C", "#F4C126"],
                       ["#A64A91", "#B9569D", "#C765A8"]][gi])
        gv = vals[s:s + 3]
        ax.add_patch(patches.Rectangle((s - 0.42, 0), 2.84, sum(gv),
                                       facecolor=group_colors[gi], alpha=0.78,
                                       zorder=1))
        ax.text(s + 1, sum(gv) + 80, f"{sum(gv)}", color=TEXT, fontsize=7.5,
                ha="center", va="bottom", zorder=8)
    x = np.arange(12)
    for xi, yi, c in zip(x, xvals, colors):
        ax.bar(xi, yi, width=0.54, color=c, zorder=4)
        ax.text(xi, yi + 70, f"{yi}", color=TEXT, fontsize=6.6,
                ha="center", va="bottom", zorder=8)
    ax.set_xticks(x, months)
    ax.set_ylim(0, 10800)
    clean_axes(ax, grid_axis=None, hide_y=True)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_29(ws, out):
    cats = col_values(ws, "B", 3, 7)
    vals = col_values(ws, "C", 3, 7)
    source = "*注：数据来源于公司销售系统，统计日期截至2022.06.30"
    fig, ax = frame("2022年上半年产品销量目标达成率情况",
                    "华南完成率最高达到86%，华东最低35%",
                    source, ax_rect=(0.12, 0.16, 0.83, 0.58))
    y = np.arange(len(cats))
    for yi, p in zip(y, vals):
        ax.barh(yi, 1.0, height=0.21, color="#8998AF", alpha=0.78, zorder=1)
        ax.barh(yi, p, height=0.21, color=BLUE, zorder=2)
        ax.scatter([p], [yi], s=170, color="#0E86C8", edgecolor="#D9EEF7",
                   linewidth=0.8, zorder=5)
        ax.text(p + 0.035, yi, f"{p * 100:.0f}%", color=TEXT, fontsize=7.2,
                ha="left", va="center")
    ax.set_yticks(y, cats)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(len(cats) - 0.5, -0.5)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def render_30(ws, out):
    cats = col_values(ws, "B", 4, 8)
    p22 = col_values(ws, "C", 4, 8)
    p21 = col_values(ws, "D", 4, 8)
    source = "*注：数据来源于公司销售系统，统计日期截至2022.06.30"
    fig, ax = frame("2022年上半年销量目标达成率同比去年\n情况",
                    "华南完成率最高达到86%，华东最低35%，其中华南和华东不及2021年",
                    source, ax_rect=(0.12, 0.16, 0.83, 0.58),
                    title_size=18, subtitle_size=9.8)
    y = np.arange(len(cats))
    for yi, a, b in zip(y, p22, p21):
        ax.barh(yi, 1.0, height=0.17, color="#8998AF", alpha=0.78, zorder=1)
        ax.barh(yi, a, height=0.17, color=BLUE, zorder=2)
        ax.plot([a, b], [yi, yi], color="#9BA8B8", linewidth=2.0, zorder=3)
        ax.scatter([b], [yi], s=95, color="#8E9AAE", edgecolor="#D3DAE4",
                   linewidth=0.7, zorder=5)
        ax.scatter([a], [yi], s=95, color="#1487C7", edgecolor="#E0F1F8",
                   linewidth=0.7, zorder=6)
        ax.text(max(a, b) + 0.028, yi - 0.24, f"{a * 100:.0f}%",
                color=TEXT, fontsize=6.8, ha="left", va="center")
    ax.scatter([], [], s=40, color="#1487C7", label="2022完成率")
    ax.scatter([], [], s=40, color="#8E9AAE", label="2021完成率")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.14), frameon=False,
              fontsize=7, labelcolor=TEXT, ncol=2, handletextpad=0.3)
    ax.set_yticks(y, cats)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(len(cats) - 0.5, -0.5)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


RENDERERS = [
    render_01, render_02, render_03, render_04, render_05,
    render_06, render_07, render_08, render_09, render_10,
    render_11, render_12, render_13, render_14, render_15,
    render_16, render_17, render_18, render_19, render_20,
    render_21, render_22, render_23, render_24, render_25,
    render_26, render_27, render_28, render_29, render_30,
]


def sanitize_filename(s: str) -> str:
    s = "".join(ch if ch not in r'\/:*?"<>|' else "_" for ch in s)
    return s.strip().replace(" ", "_")


def find_font(size, bold=False):
    names = ["msyhbd.ttc", "msyh.ttc"] if bold else ["msyh.ttc", "simhei.ttf"]
    for name in names:
        p = Path(r"C:\Windows\Fonts") / name
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def create_dashboard(image_paths, output_png, output_jpg):
    cols, rows = 5, 6
    cell_w, cell_h = 640, 480
    gap_x, gap_y = 24, 24
    header_h, footer_h = 165, 68
    width = cols * cell_w + (cols + 1) * gap_x
    height = header_h + rows * cell_h + (rows + 1) * gap_y + footer_h
    canvas = Image.new("RGB", (width, height), "#0B1231")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap_x + 6, 28), "Excel 数据可视化 · 第二章 30 图数据展板",
              font=find_font(42, bold=True), fill="#F6F8FF")
    draw.text((gap_x + 8, 96),
              "以数据叙事为核心：清晰标题、重点标注、一致配色、弱化非数据元素，按 1—30 顺序集中呈现",
              font=find_font(21), fill="#CBD6ED")
    draw.line([(gap_x, header_h - 12), (width - gap_x, header_h - 12)],
              fill="#34436E", width=2)
    for i, p in enumerate(image_paths):
        r, c = divmod(i, cols)
        x = gap_x + c * (cell_w + gap_x)
        y = header_h + gap_y + r * (cell_h + gap_y)
        img = Image.open(p).convert("RGB")
        if img.size != (cell_w, cell_h):
            img = img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        canvas.paste(img, (x, y))
        draw.rectangle([x - 1, y - 1, x + cell_w, y + cell_h],
                       outline="#3A4974", width=2)
        badge = f"{i + 1:02d}"
        draw.rounded_rectangle([x + 10, y + 10, x + 55, y + 38],
                               radius=13, fill="#0B69AE", outline="#69C7F2", width=1)
        draw.text((x + 20, y + 14), badge, font=find_font(16, bold=True), fill="white")
    footer_y = height - footer_h + 14
    draw.text((gap_x + 6, footer_y),
              "数据来源：两个工作簿内 30 张工作表及公司销售/人力资源系统  |  Python + Matplotlib 重绘",
              font=find_font(17), fill="#AEBBD8")
    draw.text((width - gap_x - 215, footer_y), "生成日期：2026-09-20",
              font=find_font(17), fill="#AEBBD8")
    canvas.save(output_png, quality=96)
    canvas.save(output_jpg, quality=94, subsampling=0)
    return width, height


def main():
    parser = argparse.ArgumentParser(description="重绘第二章 30 张 Excel 图表并生成数据展板")
    parser.add_argument("--front", type=Path, default=FRONT_DEFAULT,
                        help="前15个图表工作簿路径")
    parser.add_argument("--back", type=Path, default=BACK_DEFAULT,
                        help="后15个图表工作簿路径")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent,
                        help="输出目录")
    args = parser.parse_args()

    front = load_workbook(args.front, data_only=False)
    back = load_workbook(args.back, data_only=False)
    if len(front.worksheets) < 15 or len(back.worksheets) < 15:
        raise ValueError("两个工作簿都必须至少包含15个工作表")

    out_dir = args.output.resolve()
    chart_dir = out_dir / "30_charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for i, renderer in enumerate(RENDERERS, start=1):
        ws = front.worksheets[i - 1] if i <= 15 else back.worksheets[i - 16]
        name = sanitize_filename(ws.title)
        out = chart_dir / f"{i:02d}_{name}.png"
        renderer(ws, out)
        image_paths.append(out)
        print(f"[{i:02d}/30] {out.name}")

    board_png = out_dir / "30图数据展板.png"
    board_jpg = out_dir / "30图数据展板.jpg"
    w, h = create_dashboard(image_paths, board_png, board_jpg)

    zip_path = out_dir / "30张图表_Python重绘.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in image_paths:
            z.write(p, p.name)
    print(f"展板：{board_png} ({w}x{h})")
    print(f"单独图片压缩包：{zip_path}")


if __name__ == "__main__":
    main()


