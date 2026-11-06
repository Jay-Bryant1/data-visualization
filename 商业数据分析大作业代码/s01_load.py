# -*- coding: utf-8 -*-
"""s01_load.py —— 资产台账构建（合规获取 + 多表对齐 + 去重 + 标签缺陷登记）。

只读 RAW 下的 zip/CSV，产物写入 redo/data/：
  asset_ledger.csv   一行一条音频资产（535 行），主键 asset_id
  label_defects.csv  标签层缺陷清单（EDA 证据）
  audit_load.json    行数对账、sha256 去重对账
"""
from __future__ import annotations
import csv, hashlib, io, json, sys, zipfile
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cfg

DEFECTS: list[dict] = []


def log_defect(kind: str, where: str, detail: str, action: str) -> None:
    DEFECTS.append({"defect_type": kind, "location": where, "detail": detail, "action": action})


def read_labels(pkg: str) -> list[dict]:
    """读标签 CSV：utf-8-sig 解码 + 全字段 strip（登记原始空白缺陷）。"""
    path = cfg.RAW / cfg.LABEL_FILES[pkg]
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    clean = []
    for i, r in enumerate(rows, start=2):  # 第 1 行是表头
        c = {}
        for k, v in r.items():
            ks, vs = (k or "").strip(), (v or "")
            if vs != vs.strip():
                log_defect("空白字符", f"{cfg.LABEL_FILES[pkg]}:行{i}:{ks}",
                           f"原值 {vs!r} 含首尾空白", "strip 后入库，保留原始值列")
            c[ks] = vs.strip()
            c[ks + "__raw"] = vs
        clean.append(c)
    return clean


def wav_entries(pkg: str) -> list[tuple[str, bytes]]:
    """按 zip 目录顺序读取 wav 字节（顺序即“入库流水”代理）。"""
    out = []
    with zipfile.ZipFile(cfg.RAW / cfg.ZIP_FILES[{"HS": 0, "LS": 1, "Mix": 2}[pkg]]) as z:
        for info in z.infolist():
            n = info.filename
            if info.is_dir() or "__MACOSX" in n or not n.lower().endswith(".wav"):
                continue
            if n.startswith("._") or Path(n).name.startswith("._"):
                continue
            out.append((Path(n).name, z.read(info)))
    return out


def split_stem(stem: str) -> tuple[str, str, str]:
    """F_N_RC -> (F, N, RC)；M_LDM_LUSB -> (M, LDM, LUSB)。"""
    parts = stem.split("_")
    if len(parts) != 3:
        return ("", "", "")
    g, code, loc = parts
    return (g, code, cfg.LOCATION_CODE_FIX.get(loc, loc))


def main() -> None:
    labels = {pkg: read_labels(pkg) for pkg in ("HS", "LS", "Mix")}
    rows, seq = [], 0

    # ---------------- HS：50 条单心音 ----------------
    idx = {r["Heart Sound ID"]: r for r in labels["HS"]}
    for name, blob in wav_entries("HS"):
        stem = Path(name).stem
        lab = idx.get(stem)
        if lab is None:
            log_defect("关联缺失", f"HS.zip/{name}", "标签表无对应 Heart Sound ID", "剔除出建模队列")
            continue
        g, code, loc = split_stem(stem)
        t_csv = lab["Heart Sound Type"]
        t_code = cfg.HEART_CODE.get(code, "")
        if t_code != t_csv:
            log_defect("标签不一致", f"HS.zip/{name}", f"文件名代码 {code}->{t_code} 与 CSV {t_csv} 不符", "以 CSV 为准并登记")
        if g != lab["Gender"]:
            log_defect("标签不一致", f"HS.zip/{name}", f"文件名性别 {g} 与 CSV {lab['Gender']} 不符", "以 CSV 为准")
        if loc != lab["Location"]:
            log_defect("标签不一致", f"HS.zip/{name}", f"文件名点位 {loc} 与 CSV {lab['Location']} 不符", "以 CSV 为准")
        seq += 1
        rows.append(dict(asset_seq=seq, file_name=name, stem=stem, source_package="HS.zip",
                         modality="heart", gender=lab["Gender"], heart_type=t_csv, lung_type="",
                         location=lab["Location"], label_id=stem, label_id_raw=lab["Heart Sound ID__raw"],
                         mixed_id="", n_bytes=len(blob),
                         sha256=hashlib.sha256(blob).hexdigest()))

    # ---------------- LS：50 条单肺音 ----------------
    # LS 关联缺陷：CSV 的 ID 编码 (C/G) 与 wav 文件名编码 (FC/CC) 不一致，先做别名归一
    def ls_join_key(raw_id: str) -> str:
        parts = raw_id.split("_")
        if len(parts) == 3 and parts[1] in cfg.LUNG_CSV_CODE_ALIAS:
            parts[1] = cfg.LUNG_CSV_CODE_ALIAS[parts[1]]
        return "_".join(parts)

    idx = {}
    n_alias = 0
    for r in labels["LS"]:
        k = ls_join_key(r["Lung Sound ID"])
        if k != r["Lung Sound ID"]:
            n_alias += 1
            log_defect("编码体系不一致", f"LS.csv:Lung Sound ID={r['Lung Sound ID']}",
                       f"标签 ID 用代码 {r['Lung Sound ID'].split('_')[1]}，对应 wav 文件名用 {k.split('_')[1]}"
                       f"（{cfg.LUNG_CODE.get(k.split('_')[1], '')}）",
                       "建立别名映射 cfg.LUNG_CSV_CODE_ALIAS 后关联，品类以 CSV 为准")
        idx[k] = r
    print(f"[LS] 别名对齐 {n_alias} 条 / 共 {len(labels['LS'])} 条")
    for name, blob in wav_entries("LS"):
        stem = Path(name).stem
        lab = idx.get(stem)
        if lab is None:
            log_defect("关联缺失", f"LS.zip/{name}", "标签表无对应 Lung Sound ID", "剔除出建模队列")
            continue
        g, code, loc = split_stem(stem)
        t_csv = lab["Lung Sound Type"]
        t_code = cfg.LUNG_CODE.get(code, "")
        if t_code != t_csv:
            log_defect("命名不规范", f"LS.zip/{name}",
                       f"文件名代码 {code} 对应 {t_code}，与 CSV {t_csv} 表述冲突（C=细湿啰音、G=粗湿啰音属反直觉编码）",
                       "以 CSV 品类为准，代码->品类映射固化在 cfg.LUNG_CODE")
        if loc != lab["Location"]:
            log_defect("标签不一致", f"LS.zip/{name}", f"文件名点位 {loc} 与 CSV {lab['Location']} 不符", "以 CSV 为准")
        seq += 1
        rows.append(dict(asset_seq=seq, file_name=name, stem=stem, source_package="LS.zip",
                         modality="lung", gender=lab["Gender"], heart_type="", lung_type=t_csv,
                         location=lab["Location"], label_id=stem, label_id_raw=lab["Lung Sound ID__raw"],
                         mixed_id="", n_bytes=len(blob),
                         sha256=hashlib.sha256(blob).hexdigest()))

    # ---------------- Mix：145×3 = 435 条 ----------------
    by_h = {r["Heart Sound ID"]: r for r in labels["Mix"]}
    by_l = {r["Lung Sound ID"]: r for r in labels["Mix"]}
    by_m = {r["Mixed Sound ID"]: r for r in labels["Mix"]}
    for name, blob in wav_entries("Mix"):
        stem = Path(name).stem
        pre = stem[0].upper()
        if pre == "H":
            lab, modality, lid = by_h.get(stem), "heart_source", stem
        elif pre == "L":
            lab, modality, lid = by_l.get(stem), "lung_source", stem
        elif pre == "M":
            lab, modality, lid = by_m.get(stem), "mixed", stem
        else:
            log_defect("命名不规范", f"Mix.zip/{name}", "文件名前缀非 H/L/M", "跳过")
            continue
        if lab is None:
            log_defect("关联缺失", f"Mix.zip/{name}", "Mix.csv 无对应 ID", "剔除")
            continue
        seq += 1
        rows.append(dict(asset_seq=seq, file_name=name, stem=stem, source_package="Mix.zip",
                         modality=modality, gender=lab["Gender"], heart_type=lab["Heart Sound Type"],
                         lung_type=lab["Lung Sound Type"], location=lab["Location"],
                         label_id=lab["Heart Sound ID"], label_id_raw=lab["Heart Sound ID"],
                         mixed_id=lab["Mixed Sound ID"], n_bytes=len(blob),
                         sha256=hashlib.sha256(blob).hexdigest()))

    df = pd.DataFrame(rows)

    # ---------------- 主键与冲突检查 ----------------
    df["asset_id"] = ["A%04d" % (i + 1) for i in range(len(df))]
    dup_names = [n for n, c in Counter(df["file_name"]).items() if c > 1]
    if dup_names:
        log_defect("主键冲突", "file_name", f"{len(dup_names)} 个文件名跨包重复：{dup_names[:5]}",
                   "改用 asset_id 作为主键，file_name 仅作展示")
    df["path_key"] = df["source_package"] + "/" + df["file_name"]
    assert df["asset_id"].is_unique and df["path_key"].is_unique

    # ---------------- sha256 指纹去重 ----------------
    first_seen: dict[str, str] = {}
    gid: dict[str, str] = {}
    g_counter = 0
    groups: dict[str, list[str]] = defaultdict(list)
    for _, r in df.sort_values("asset_seq").iterrows():
        h = r["sha256"]
        groups[h].append(r["asset_id"])
        if h not in first_seen:
            g_counter += 1
            first_seen[h] = "G%04d" % g_counter
        gid[h] = first_seen[h]
    df["group_id"] = df["sha256"].map(gid)
    df["group_size"] = df["sha256"].map({h: len(v) for h, v in groups.items()})
    df["primary_asset_id"] = df["sha256"].map({h: v[0] for h, v in groups.items()})
    df["is_primary_copy"] = (df["asset_id"] == df["primary_asset_id"]).astype(int)
    n_dup = int((df["group_size"] > 1).sum())
    n_unique = int(df["group_id"].nunique())
    log_defect("重复值", "sha256",
               f"{len(df)} 条资产按内容指纹归并为 {n_unique} 条独立录音，冗余 {len(df) - n_unique} 条（涉及 {n_dup} 行）",
               "按 asset_seq 保留组内首条为 is_primary_copy=1，其余标 0；去重不按文件名")

    # ---------------- 派生标签与队列口径 ----------------
    df["gender_zh"] = df["gender"].map(cfg.GENDER_ZH)
    df["modality_zh"] = df["modality"].map(cfg.MODALITY_ZH)
    df["package_zh"] = df["source_package"].map(cfg.PACKAGE_ZH)
    df["location_zh"] = df["location"].map(cfg.LOCATION_ZH).fillna(df["location"])
    df["heart_type_zh"] = df["heart_type"].map(cfg.HEART_ZH).fillna("")
    df["lung_type_zh"] = df["lung_type"].map(cfg.LUNG_ZH).fillna("")
    df["side"] = df["location"].str[0].map({"L": "左", "R": "右"}).fillna("中")
    df["auscultation_system"] = df["location"].map(
        lambda x: "心脏听诊区" if x in ("Apex", "LC", "RC", "LLSB", "LUSB", "RUSB") else "肺野听诊区")
    df["batch_no"] = ((df["asset_seq"] - 1) // cfg.BATCH_SIZE + 1).astype(int)
    df["batch_label"] = "B" + df["batch_no"].astype(str).str.zfill(2)

    # ---------------- 口径定义（重做版核心改进）----------------
    # 旧版把 mixed 同时塞进心音队列与肺音队列，两队列共享 145 条混合录音，导致
    # 队列规模 195/195 无法与“独立录音 344”对账。重做版按“内容独立的单模态/混合”三分：
    #   heart_pure = HS 单心音 + Mix 包内内容独立的心源轨
    #   lung_pure  = LS 单肺音 + Mix 包内内容独立的肺源轨
    #   mixed      = Mix 混合轨
    #   duplicate_copy = sha256 重复副本（不入模）
    # 三者相加恰好等于 344 条独立录音，可与去重结果逐条对账。
    def _cohort(r) -> str:
        if r["is_primary_copy"] == 0:
            return "duplicate_copy"
        return {"heart": "heart_pure", "heart_source": "heart_pure",
                "lung": "lung_pure", "lung_source": "lung_pure",
                "mixed": "mixed"}[r["modality"]]

    df["cohort"] = df.apply(_cohort, axis=1)
    df["cohort_zh"] = df["cohort"].map({"heart_pure": "纯心音队列", "lung_pure": "纯肺音队列",
                                        "mixed": "混合队列", "duplicate_copy": "重复副本"})
    df["in_model"] = (df["cohort"] != "duplicate_copy").astype(int)
    df["in_heart_cohort"] = (df["cohort"] == "heart_pure").astype(int)
    df["in_lung_cohort"] = (df["cohort"] == "lung_pure").astype(int)
    # 心音/肺音标签适用范围：纯心音队列有心音标签，纯肺音队列有肺音标签，混合队列两者都有
    df["heart_abnormal"] = -1
    mh = df["cohort"].isin(["heart_pure", "mixed"])
    df.loc[mh, "heart_abnormal"] = (df.loc[mh, "heart_type"] != "Normal").astype(int)
    df["lung_abnormal"] = -1
    ml = df["cohort"].isin(["lung_pure", "mixed"])
    df.loc[ml, "lung_abnormal"] = (df.loc[ml, "lung_type"] != "Normal").astype(int)

    # 标签完整度（品类缺失登记）
    miss_h = int(((df["modality"].isin(["heart", "mixed"])) & (df["heart_type"] == "")).sum())
    miss_l = int(((df["modality"].isin(["lung", "mixed"])) & (df["lung_type"] == "")).sum())
    if miss_h or miss_l:
        log_defect("缺失值", "品类标签", f"心音品类缺失 {miss_h} 条、肺音品类缺失 {miss_l} 条", "按队列口径核查")

    df["duration_sec"] = 15.0
    df["sample_rate"] = cfg.SR_EXPECTED
    df["n_channels"] = 1
    df["n_samples"] = 60000
    df["storage_mb"] = df["n_bytes"] / 1024 / 1024

    cols = ["asset_id", "asset_seq", "path_key", "file_name", "stem", "source_package", "package_zh",
            "modality", "modality_zh", "gender", "gender_zh", "heart_type", "heart_type_zh",
            "lung_type", "lung_type_zh", "location", "location_zh", "side", "auscultation_system",
            "label_id", "label_id_raw", "mixed_id", "batch_no", "batch_label", "sha256", "group_id",
            "group_size", "primary_asset_id", "is_primary_copy", "cohort", "cohort_zh",
            "in_model", "in_heart_cohort",
            "in_lung_cohort", "heart_abnormal", "lung_abnormal", "n_bytes", "storage_mb",
            "duration_sec", "sample_rate", "n_channels", "n_samples"]
    df = df[cols]
    df.to_csv(cfg.DATA / "asset_ledger.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(DEFECTS).to_csv(cfg.DATA / "label_defects.csv", index=False, encoding="utf-8-sig")

    # ---------------- 对账 ----------------
    uniq_h = int(df.loc[df["modality"] == "heart", "sha256"].nunique())
    uniq_l = int(df.loc[df["modality"] == "lung", "sha256"].nunique())
    uniq_mix = int(df.loc[df["modality"] == "mixed", "sha256"].nunique())
    audit = {
        "assets_total": int(len(df)),
        "by_package": df["source_package"].value_counts().to_dict(),
        "by_modality": df["modality"].value_counts().to_dict(),
        "unique_recordings_sha256": n_unique,
        "redundant_rows": int(len(df) - n_unique),
        "reconciliation": f"{uniq_h}(HS) + {uniq_l}(LS) + {uniq_mix}(Mix混合轨) = {uniq_h + uniq_l + uniq_mix} 条内容独立录音（含跨包重复前）",
        "unique_after_cross_package_dedup": n_unique,
        "heart_cohort": int(df["in_heart_cohort"].sum()),
        "lung_cohort": int(df["in_lung_cohort"].sum()),
        "mixed_cohort": int((df["cohort"] == "mixed").sum()),
        "duplicate_copy": int((df["cohort"] == "duplicate_copy").sum()),
        "in_model": int(df["in_model"].sum()),
        "cohort_reconciliation": "heart_pure + lung_pure + mixed = in_model = 独立录音数",
        "cohort_check_pass": bool(int(df["in_heart_cohort"].sum()) + int(df["in_lung_cohort"].sum())
                                  + int((df["cohort"] == "mixed").sum()) == int(df["in_model"].sum()) == n_unique),
        "mix_internal_duplicate_pairs": [[str(x) for x in list(v)] for k, v in
                                          df[df["modality"] == "mixed"].groupby("group_id")["file_name"]
                                          .apply(list).items() if len(v) > 1],
        "heart_abnormal_rate": round(float(df.loc[df["in_heart_cohort"] == 1, "heart_abnormal"].mean()), 4),
        "lung_abnormal_rate": round(float(df.loc[df["in_lung_cohort"] == 1, "lung_abnormal"].mean()), 4),
        "label_defects": len(DEFECTS),
        "defect_types": dict(Counter(d["defect_type"] for d in DEFECTS)),
        "batches": int(df["batch_no"].nunique()),
        "seed": cfg.SEED,
    }
    (cfg.DATA / "audit_load.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("rows:", len(df), "cols:", len(df.columns))


if __name__ == "__main__":
    main()
