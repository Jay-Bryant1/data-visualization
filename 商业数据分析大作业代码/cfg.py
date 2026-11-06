# -*- coding: utf-8 -*-
"""cfg.py —— 重做版（redo）全局配置。

设计原则：所有阈值/单价/参数集中声明，报告里出现的每个数字都能追溯到本文件。
项目默认从 raw/ 读取 HS/LS/Mix 三个 zip 与三个标签 CSV；也可用 HLS_RAW
环境变量指定外部原始数据目录。所有中间产物写入当前项目目录。
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------- 路径
CODE = Path(__file__).resolve().parent                 # 全部脚本统一位于 代码/
ROOT = CODE.parent if ((CODE.parent / "raw").exists() or (CODE.parent / "data").exists()) else CODE
if os.environ.get("HLS_RAW"):
    RAW = Path(os.environ["HLS_RAW"])
elif (ROOT / "raw").exists():
    RAW = ROOT / "raw"
else:
    RAW = ROOT.parent                                 # 兼容原始数据位于项目上一级
DATA = ROOT / "data"
EDA = ROOT / "eda"
EDA_RAW = EDA / "raw"
FIG = ROOT / "figures"
TBL = ROOT / "tables"
DASH = ROOT / "dashboard"
REPORT = ROOT / "report"
LOGS = ROOT / "logs"
for _d in (DATA, EDA, EDA_RAW, FIG, TBL, DASH, REPORT, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

ZIP_FILES = ["HS.zip", "LS.zip", "Mix.zip"]
LABEL_FILES = {"HS": "HS.csv", "LS": "LS.csv", "Mix": "Mix.csv"}
SKILL_EDA = Path(os.environ.get(
    "PROGRAMMATIC_EDA_SCRIPTS",
    str(CODE),
))
PYTHON = os.environ.get("PYTHON_EXE", sys.executable)

# ---------------------------------------------------------------- 可复现性
SEED = 20261103          # 与旧版 (42) 区分：重做版独立随机种子
N_FOLDS = 5
N_BOOTSTRAP = 2000
BATCH_SIZE = 50          # 每 50 条资产为一个“入库批次”（zip 目录顺序代理采集顺序）

# ---------------------------------------------------------------- 信号处理
SR_EXPECTED = 4000
FRAME_MS, HOP_MS = 25.0, 10.0
N_FFT = 128              # 频率分辨率 31.25 Hz
N_MELS, N_MFCC = 26, 13
MEL_FMIN, MEL_FMAX = 20.0, 2000.0
BANDS_HZ = [(20, 150), (150, 300), (300, 600), (600, 1200), (1200, 2000)]
BAND_NAMES = ["b1_20_150", "b2_150_300", "b3_300_600", "b4_600_1200", "b5_1200_2000"]
HR_LAG_SEC = (0.5, 1.5)  # 心率自相关搜索窗 -> 40~120 bpm
RR_LAG_SEC = (2.0, 6.0)  # 呼吸率搜索窗 -> 10~30 次/分
BLOCK_MS = 100.0         # 帧级明细表的时间块粒度（15 s -> 150 块）

# 质检阈值（业务依据：临床听诊录音可接受质量下限）
DC_MAX = 0.01            # 直流偏置 |mean| 上限
CLIP_LEVEL = 0.999
CLIP_RATIO_MAX = 0.001   # 削波样本占比上限 0.1%
# 静音判定用“相对阈值”：本批录音整体电平偏低（peak_db 中位数 -29.6 dBFS），
# 若沿用绝对 -40 dBFS 会把 93% 的资产误判为静音超标。改为低于自身 p95 帧能量 25 dB 记为静音帧，
# 同时保留绝对口径 low_level_ratio 供对照。
SILENCE_REL_DB = 25.0    # 相对 p95 帧能量的衰减门限（dB）
SILENCE_DB = -40.0       # 绝对低电平门限（dBFS），仅用于 low_level_ratio
SILENCE_RATIO_MAX = 0.60 # 静音帧占比上限 60%
SNR_FLOOR_DB = 0.0
SNR_CEIL_DB = 30.0

# ---------------------------------------------------------------- 成本模型（单位：元）
# 依据：对象存储标准型公开报价量级 0.12 元/GB/月；医学音频标注按“标注员 + 医师复核”两级人力计。
STORAGE_YUAN_PER_GB_MONTH = 0.12
STORAGE_MONTHS = 36                      # 3 年留存周期
LABEL_MIN_BASE = 2.0                     # 单条基础标注工时（分钟）
LABEL_MIN_ABNORMAL_EXTRA = 1.5           # 异常品类追加（需逐拍/逐呼吸相判读）
LABEL_MIN_MIXED_EXTRA = 2.0              # 混合模态追加（需分离心/肺事件）
LABELER_YUAN_PER_MIN = 0.75              # 标注员 45 元/小时
PHYSICIAN_YUAN_PER_MIN = 2.50            # 医师复核 150 元/小时
QC_AUTO_MIN = 0.30                       # 自动质检工时
QC_SAMPLE_RATE = 0.10                    # 人工抽检比例
QC_MANUAL_MIN = 1.5                      # 单条人工抽检工时
PHYSICIAN_REVIEW_RATE = 0.30             # 需医师复核的资产比例（异常品类全检 + 正常抽检）

# ---------------------------------------------------------------- 价值模型
# 资产价值指数 V = w_q*质量 + w_i*信息量 + w_s*稀缺度 + w_r*复用度（0~100）
W_QUALITY, W_INFO, W_SCARCITY, W_REUSE = 0.35, 0.30, 0.25, 0.10
ABC_CUM = (70.0, 90.0)                   # ABC 累计价值切分（A:0-70%, B:70-90%, C:90-100%）
RFM_LEVELS = 5                           # RFM 五分位打分 1~5
BCG_SHARE_MEDIAN = None                  # None -> 用中位数动态切分
FORECAST_HORIZON = 15                    # Holt-Winters 前瞻 15 个时间块 = 1.5 s
FORECAST_TRAIN_RATIO = 0.80              # 回测：前 80% 训练，后 20% 检验
CI_ALPHA = 0.05                          # 95% 置信区间

# ---------------------------------------------------------------- 标签字典
HEART_CODE = {
    "N": "Normal", "MSM": "Mid Systolic Murmur", "ESM": "Early Systolic Murmur",
    "LSM": "Late Systolic Murmur", "LDM": "Late Diastolic Murmur", "S3": "S3",
    "S4": "S4", "AF": "Atrial Fibrillation", "AVB": "AV Block", "T": "Tachycardia",
}
HEART_ZH = {
    "Normal": "正常心音", "Mid Systolic Murmur": "收缩中期杂音", "Early Systolic Murmur": "收缩早期杂音",
    "Late Systolic Murmur": "收缩晚期杂音", "Late Diastolic Murmur": "舒张晚期杂音", "S3": "第三心音S3",
    "S4": "第四心音S4", "Atrial Fibrillation": "心房颤动", "AV Block": "房室传导阻滞", "Tachycardia": "心动过速",
}
LUNG_ZH = {
    "Normal": "正常肺音", "Pleural Rub": "胸膜摩擦音", "Coarse Crackles": "粗湿啰音",
    "Rhonchi": "鼾音", "Wheezing": "哮鸣音", "Fine Crackles": "细湿啰音",
}
# 文件名代码 -> 品类（数据源用 C 表示 Fine Crackles、G 表示 Coarse Crackles，属命名缺陷，需登记）
LUNG_CODE = {"N": "Normal", "PR": "Pleural Rub", "R": "Rhonchi", "W": "Wheezing",
             "C": "Fine Crackles", "G": "Coarse Crackles",
             "FC": "Fine Crackles", "CC": "Coarse Crackles"}
# LS.csv 的 ID 用 C/G，而 wav 文件名用 FC/CC：两套编码并存，需别名对齐后才能关联
LUNG_CSV_CODE_ALIAS = {"C": "FC", "G": "CC"}
LOCATION_ZH = {
    "Apex": "心尖", "LC": "左锁骨", "LLSB": "胸骨左下缘", "LUSB": "胸骨左上缘",
    "RC": "右锁骨", "RUSB": "胸骨右上缘", "LUA": "左上肺野", "RUA": "右上肺野",
    "LMA": "左中肺野", "RMA": "右中肺野", "LLA": "左下肺野", "RLA": "右下肺野",
}
LOCATION_CODE_FIX = {"A": "Apex"}      # 文件名里的 A -> Apex
GENDER_ZH = {"M": "男", "F": "女"}
MODALITY_ZH = {"heart": "心音", "lung": "肺音", "mixed": "心肺混合",
               "heart_source": "心音源轨", "lung_source": "肺音源轨"}
PACKAGE_ZH = {"HS.zip": "HS 单心音包", "LS.zip": "LS 单肺音包", "Mix.zip": "Mix 混合包"}

# 合规声明（论文/README 引用）
COMPLIANCE = {
    "dataset_name": "HLS-CMDS：Heart and Lung Sounds Dataset Recorded from a Clinical Manikin using Digital Stethoscope",
    "url": "https://archive.ics.uci.edu/dataset/1202",
    "upstream": "UCI Machine Learning Repository 数据集 1202；Torabi、Shirani、Reilly，2025",
    "access_method": "UCI 官方页面一次性下载发布包 HLS-CMDS.zip（白名单方式，未使用任何爬虫）",
    "access_date": "2026-10-22（本地文件创建时间）",
    "license": "CC BY 4.0（Creative Commons Attribution 4.0 International）",
    "local_files": "HS.zip(3.37MB) / LS.zip(3.48MB) / Mix.zip(30.89MB) + HS.csv / LS.csv / Mix.csv",
}

# ---------------------------------------------------------------- 别名（便于下游脚本统一引用）
DATA_DIR, FIG_DIR, TABLE_DIR, REPORT_DIR = DATA, FIG, TBL, REPORT

