# -*- coding: utf-8 -*-
"""一键复现：原始数据 -> 特征 -> EDA -> 模型 -> 图表 -> 大屏。"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent if ((CODE.parent / "raw").exists() or (CODE.parent / "data").exists()) else CODE
SCRIPTS = [
    "s01_load.py",
    "s02_features.py",
    "s03_eda.py",
    "s04_models.py",
    "s05_figures.py",
    "s06_figures.py",
    "s07_dashboard.py",
]
RAW_FILES = ["HS.zip", "LS.zip", "Mix.zip", "HS.csv", "LS.csv", "Mix.csv"]


def find_browser() -> Path | None:
    names = ["msedge", "chrome", "chromium", "google-chrome"]
    for name in names:
        hit = shutil.which(name)
        if hit:
            return Path(hit)
    for candidate in (
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    ):
        if candidate.exists():
            return candidate
    return None


def capture_dashboard(env: dict[str, str]) -> None:
    browser = find_browser()
    htmls = sorted((ROOT / "dashboard").glob("*.html"))
    if browser is None or not htmls:
        print("[skip] 未找到 Edge/Chrome 或大屏 HTML，保留仓库内已有截图")
        return
    base = htmls[0].resolve().as_uri()
    jobs = [
        (ROOT / "figures" / "fig28_交互大屏_资产经营总览.png", ""),
        (ROOT / "figures" / "fig29_交互大屏_队列下钻联动.png", "?cohort=mixed"),
    ]
    for out, query in jobs:
        profile = Path(tempfile.mkdtemp(prefix="dashboard-capture-"))
        try:
            cmd = [
                str(browser), "--headless", "--disable-gpu", "--hide-scrollbars",
                "--no-first-run", f"--user-data-dir={profile}", "--window-size=1920,1160",
                "--run-all-compositor-stages-before-draw", "--virtual-time-budget=5000",
                f"--screenshot={out}", base + query,
            ]
            cp = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
            if cp.returncode != 0 or not out.exists():
                print(f"[warn] 大屏截图失败：{out.name}")
                if cp.stderr:
                    print(cp.stderr[-600:])
            else:
                print(f"[fig] {out.name}")
        finally:
            shutil.rmtree(profile, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="HLS-CMDS 商业数据分析项目一键复现")
    parser.add_argument("--raw", type=Path, help="原始 HS/LS/Mix 文件所在目录")
    args = parser.parse_args()

    raw = args.raw.resolve() if args.raw else None
    if raw is None:
        candidate = ROOT / "raw"
        if candidate.exists():
            raw = candidate
        elif all((ROOT.parent / f).exists() for f in RAW_FILES):
            raw = ROOT.parent
    if raw is None or not all((raw / f).exists() for f in RAW_FILES):
        print("原始数据不完整。请将 HS/LS/Mix 三组 zip+csv 放入 raw/，或使用 --raw 指定目录。")
        return 2

    env = os.environ.copy()
    env["HLS_RAW"] = str(raw)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"[raw] {raw}")
    print(f"[python] {sys.executable}")

    for script in SCRIPTS:
        print(f"\n==> {script}")
        cp = subprocess.run([sys.executable, str(CODE / script)], cwd=ROOT, env=env)
        if cp.returncode != 0:
            print(f"执行失败：{script}（退出码 {cp.returncode}）")
            return cp.returncode

    capture_dashboard(env)
    print("\n完成：数据、图表与大屏已重建。最终 Word/PDF 报告见 docs/。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())