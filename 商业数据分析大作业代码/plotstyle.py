# -*- coding: utf-8 -*-
"""plotstyle.py —— 统一绘图风格、中文字体、数据装载与保存helper。"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cfg

PALETTE = ["#1F4E79", "#2E8B8B", "#C0504D", "#F79646", "#4472C4",
           "#ED7D31", "#7F7F7F", "#A5A5A5", "#FFC000", "#5B9BD5"]
ACCENT = {"good": "#2E8B8B", "warn": "#F79646", "bad": "#C0504D", "navy": "#1F4E79"}

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "figure.dpi": 150, "savefig.dpi": 300,          # 终版：印刷分辨率提升至 300dpi（要求 ≥150dpi）
    "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
    "axes.grid": True, "grid.alpha": 0.28, "grid.linestyle": "--",
    "axes.edgecolor": "#666666", "axes.linewidth": 0.9,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
    "legend.frameon": False, "figure.facecolor": "white",
})


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    A = pd.read_csv(cfg.DATA / "model_table.csv")
    L = pd.read_csv(cfg.DATA / "asset_ledger.csv")
    B = pd.read_csv(cfg.DATA / "features_block.csv")
    import json
    R = json.loads((cfg.EDA / "model_results.json").read_text(encoding="utf-8"))
    return A, L, B, R


def tbl(name: str) -> pd.DataFrame:
    return pd.read_csv(cfg.TABLE_DIR / name)


def finish(fig, fname: str, note: str = "") -> None:
    """统一保存：文件名编号 + 中文主题，便于图表索引。"""
    path = cfg.FIG / fname
    fig.savefig(path)
    plt.close(fig)
    print(f"  [fig] {fname}  {path.stat().st_size // 1024} KB  {note}")


def style_ax(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def footer(fig, text: str):
    """底部说明行：置于坐标区之外并自动折行，避免与旋转的 x 轴标签重叠。"""
    import textwrap
    lines = textwrap.wrap(text, 76) or [text]
    fig.text(0.5, -0.02, "\n".join(lines), ha="center", va="top",
             fontsize=7.2, color="#888888", linespacing=1.5)


def cbar_format(v, unit=""):
    return f"{v:,.1f}{unit}"

