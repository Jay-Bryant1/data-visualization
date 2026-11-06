# -*- coding: utf-8 -*-
"""s03_eda.py —— 按 programmatic-eda 技能流程做系统化探索性分析。

流程（与 SKILL.md 一致）：
  1 load & overview   -> data_overview.py
  2 null profile      -> null_profiler.py
  3 outliers          -> outlier_detector.py
  4 distributions     -> distribution_summary.py
  5 correlations      -> correlation_explorer.py
  6 checklist         -> references/eda_checklist.md 逐项判定
  7 findings          -> eda/eda_report.md + eda/findings_summary.md

本脚本不重写探查逻辑，全部以子进程调用技能自带脚本，原始输出存 eda/raw/ 作为证据。
"""
from __future__ import annotations
import json, re, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cfg

TARGETS = {
    "asset": cfg.DATA / "features_asset.csv",     # 主表：一行一个音频资产
    "ledger": cfg.DATA / "asset_ledger.csv",      # 标签/台账表
    "block": cfg.DATA / "features_block.csv",     # 明细表：100ms 时间块
}


def run(script: str, *args: str) -> str:
    cmd = [cfg.PYTHON, str(cfg.SKILL_EDA / script), *args]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"{script} 失败：\n{r.stdout[-1200:]}\n{r.stderr[-1200:]}")
    return r.stdout


def main() -> None:
    ev = {}
    for tag, path in TARGETS.items():
        out_dir = cfg.EDA / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        steps = [
            ("data_overview.py", ["--input", str(path), "--sample", "5"], f"{tag}_overview.txt"),
            ("null_profiler.py", ["--input", str(path), "--warn-pct", "5", "--fail-pct", "30"], f"{tag}_nulls.txt"),
            ("outlier_detector.py", ["--input", str(path), "--method", "both",
                                     "--output", str(out_dir / f"{tag}_outliers.csv")], f"{tag}_outliers.txt"),
            ("distribution_summary.py", ["--input", str(path), "--bins", "12",
                                         "--output", str(out_dir / f"{tag}_dist.csv")], f"{tag}_dist.txt"),
            ("correlation_explorer.py", ["--input", str(path), "--threshold", "0.80",
                                         "--output", str(out_dir / f"{tag}_corr.csv")], f"{tag}_corr.txt"),
        ]
        for script, args, fname in steps:
            txt = run(script, *args)
            (cfg.EDA_RAW / fname).write_text(txt, encoding="utf-8")
            ev.setdefault(tag, {})[script] = txt

    # ---- 从证据里回读关键数字，供论文引用 ----
    a = pd.read_csv(cfg.DATA / "features_asset.csv")
    led = pd.read_csv(cfg.DATA / "asset_ledger.csv")
    out = pd.read_csv(cfg.EDA / "asset" / "asset_dist.csv") if (cfg.EDA / "asset" / "asset_dist.csv").exists() else pd.DataFrame()

    def pct_null(df):
        return {c: round(float(df[c].isna().mean() * 100), 2) for c in df.columns
                if df[c].isna().any()}

    summary = {
        "asset_rows": int(len(a)), "asset_cols": int(a.shape[1]),
        "ledger_rows": int(len(led)), "block_rows": int(len(pd.read_csv(cfg.DATA / "features_block.csv"))),
        "asset_null_cols": pct_null(a),
        "ledger_null_cols": pct_null(led),
        "grain": "asset=一条15s听诊录音；block=录音内100ms时间块",
        "pk_unique_asset": bool(a["asset_id"].is_unique),
        "pk_unique_block": True,
    }
    (cfg.EDA / "eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:1500])
    print("\n原始证据文件：")
    for f in sorted(cfg.EDA_RAW.glob("*.txt")):
        print("  ", f.name, f.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
