"""通用批量出图流程：扫描 年度数据/<分类>/ 下的导出文件，一个文件一张图。

代码按分类分目录（``统计局代码/<分类>/``），与 ``年度数据/<分类>/`` 一一镜像对应；
本模块提供跨分类共用的流程与主题登记表，分类目录下的 ``build_all.py`` 只做一层薄包装。
数据目录由 :func:`stj_core.data_root` 自本文件位置上溯定位，不依赖当前工作目录。

用法（在任意目录执行均可）：
    python 统计局代码/综合/build_all.py                  # 处理 年度数据/综合
    python 统计局代码/综合/build_all.py --check           # 只体检，不出图
    python 统计局代码/综合/build_all.py --scan-all        # 处理所有非空分类
    python 统计局代码/国民经济核算/build_all.py
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

# 共享核心（stj_core / charts）与本模块同处一个目录；显式插入 sys.path[0]，
# 保证优先命中同目录的那份副本，而不是碰巧被 cwd 里的同名模块顶掉
sys.path.insert(0, str(Path(__file__).resolve().parent))

from charts import (  # noqa: E402
    is_chain_index,
    plot_indicator_grid,
    plot_indicator_points,
    plot_single_year_dots,
)
from stj_core import EmptyTable, StjTable, data_root, parse_table  # noqa: E402

CODE_ROOT = Path(__file__).resolve().parent   # 本文件所在目录
ROOT = data_root(CODE_ROOT)                   # 年度数据（自本目录向上找到）

# 官方「下载」可能是 CSV，也可能是 Excel，两种都接
SOURCE_SUFFIXES: tuple[str, ...] = (".csv", ".xlsx", ".xls")

# 官网导出名「年度数据.csv」「年度数据 (12).csv」没有任何信息量；
# 文件名整理成描述性名字后，直接用文件名当图标题，不必再登记一次。
_GENERIC_STEM = re.compile(r"^年度数据(\s*[\(（]\d+[\)）])?$")

# 归档时给文件加的序号前缀（「01-国内生产总值…」）只为在资源管理器里保持官方表序，
# 不该出现在图标题上，取主题名时剥掉。
_ORDER_PREFIX = re.compile(r"^\d{2,3}-")

# 导出文件名本身不含主题信息（「年度数据 (3).csv」），这里人工登记主题。
# 键带上分类名：不同分类里重名的「年度数据 (1).csv」是不同的表，不能串用主题。
# 文件名已整理成描述性名字的分类不需要登记。
THEMES: dict[tuple[str, str], str] = {
    ("综合", "年度数据.csv"): "行政区划数",
    ("综合", "年度数据 (1).csv"): "人均主要农产品占有量",
    ("综合", "年度数据 (2).csv"): "法人单位数（三次产业）",
    ("综合", "年度数据 (3).csv"): "法人单位数（按机构类型）",
    ("综合", "年度数据 (4).csv"): "法人单位数（按行业门类）",
    ("综合", "年度数据 (5).csv"): "法人单位数（按行业门类）",
    ("综合", "年度数据 (6).csv"): "企业法人单位数（按控股类型）",
    ("综合", "年度数据 (7).csv"): "企业法人单位数（按登记注册类型）",
    ("综合", "年度数据 (8).csv"): "企业法人单位构成",
}

# 个别表存在口径断点，图上必须说明，否则变化率会被当成真实趋势读。键为主题名。
DATA_CAVEATS: dict[str, str] = {
    "民族自治地方人口": "2016 年数据与 2017 年起不同口径：总人口 2016 年 21,567 万、"
    "2017 年骤降至 18,943 万后稳定在 1.9 亿，跨 2016 年的增减不可直接比较。",
}

# 资金流量表是 SNA 体系下的部门间交易矩阵，与支出法 GDP 不是同一套口径，
# 且「净金融投资」在总量行上按会计恒等式恒为 0，必须提前说明，否则会被读成数据缺失。
_FUND_FLOW_PREFIX = "资金流量"
_FUND_FLOW_CAVEAT = (
    "资金流量表按 SNA 口径编制，与支出法 GDP 的行业/需求分类不一一对应；"
    "总量行的「净金融投资」是资金运用与来源轧差的结果，全经济总体、国内两行依会计恒等式恒为 0。"
)


def _digest(path: Path) -> str:
    """文件内容哈希，用于识别重复下载的同一份导出。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources(category_dir: Path) -> list[Path]:
    """列出分类目录下的所有导出文件（CSV / Excel，扩展名不区分大小写）。

    跳过 Excel 打开时产生的 ``~$`` 临时锁文件，否则会被当成数据读进来。
    """
    found = [
        p
        for p in category_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SOURCE_SUFFIXES and not p.name.startswith("~$")
    ]
    return sorted(found, key=lambda p: p.name)


# 只有「按定义恒定」的行才该被剔掉：合计行恒为 100%、净金融投资恒为 0。
# 行政区划数这类恒定值不是冗余，是「九年没变过」这个事实本身，删掉等于把数据扔了。
_AGGREGATE_HINTS: tuple[str, ...] = ("合计", "总计", "三次产业", "国内生产总值", "净金融投资")


def _flat_rows(table: StjTable) -> list[str]:
    """找出「按定义恒定」的指标（恒为 0 的轧差行、恒为 100% 的合计行）。

    资金流量表总量行的「净金融投资」按会计恒等式恒为 0，构成表的「国内生产总值」
    合计行按定义为 100%，画出来都是一条压在地面或横贯全图的直线：占掉一个面板、
    零信息量，还容易被误读成「没数据 / 画错了」。

    Args:
        table: 数据表。

    Returns:
        需要剔除的指标名。
    """
    rows: list[str] = []
    for name in table.indicators:
        values = table.wide.loc[name].dropna()
        if len(values) < 3:
            continue
        span = float(values.max() - values.min())
        if span > max(1e-9, 0.002 * abs(float(values.mean()))):
            continue
        # 恒定值也可能是有效信息（自治州数九年都是 30 个），只有恒为 0
        # 或名字本身就是汇总口径的才剔
        if abs(float(values.mean())) < 1e-9 or any(h in name for h in _AGGREGATE_HINTS):
            rows.append(name)
    return rows


def _drop_flat_rows(table: StjTable) -> tuple[StjTable, list[str]]:
    """剔除恒定不变的科目，返回 (新表, 被剔除的科目名)。

    整表都恒定时原样返回——那种情况下继续画总比出一张空图强。

    Args:
        table: 原始表。

    Returns:
        (剔除后的表, 被剔除的指标名列表)。
    """
    flats = _flat_rows(table)
    if not flats or len(flats) == len(table.indicators):
        return table, []
    keep = [n for n in table.indicators if n not in flats]
    trimmed = StjTable(
        path=table.path,
        wide=table.wide.loc[keep],
        units={k: v for k, v in table.units.items() if k in keep},
        notes=table.notes,
    )
    return trimmed, flats


def _drop_empty_years(table: StjTable) -> tuple[StjTable, list[int]]:
    """剔掉整列无值的年份，返回 (新表, 被剔除的年份)。

    统计局的最新一年常常只占位不填数（2025 年尚未发布）。保留该列会在图右侧留下
    一条通栏灰带，看起来像「面板被切掉一块」，实际上什么都没画。
    """
    keep = [y for y in table.years if table.wide[y].notna().any()]
    if len(keep) == len(table.years) or not keep:
        return table, []
    dropped = [y for y in table.years if y not in keep]
    trimmed = StjTable(
        path=table.path,
        wide=table.wide.loc[:, keep],
        units=table.units,
        notes=table.notes,
    )
    return trimmed, dropped


def _subtitle(table: StjTable, dropped: list[str] | None = None,
              empty_years: list[int] | None = None) -> str:
    """按数据形状生成副标题：多年份 vs 单年份。

    Args:
        table: 数据表。
        dropped: 被剔除的恒定科目，用于在副标题里交代「少掉的那几行去哪了」。
        empty_years: 整列无值、未绘入的年份。
    """
    years = table.years
    units = {u for u in table.units.values() if u}
    if len(units) == 1:
        unit_txt = f"单位：{next(iter(units))}"
    elif units:
        unit_txt = "各指标单位不同，见小图标题"
    else:
        unit_txt = "单位未标注"

    drop_txt = ""
    if dropped:
        shown = "、".join(dropped[:2]) + ("…" if len(dropped) > 2 else "")
        drop_txt = f"　|　已略去 {len(dropped)} 个恒定不变的科目（{shown}），非缺失"

    empty_txt = ""
    if empty_years:
        empty_txt = f"　|　{'、'.join(map(str, empty_years))} 年整列无数据，未绘入"

    # 末年份常常只发布了部分指标（2025 年城乡居民消费支出还没有数），
    # 图右侧那一条空白要有个说法，否则会被当成画漏了
    partial_txt = ""
    last = years[-1]
    if bool(table.wide[last].isna().any()) and bool(table.wide[last].notna().any()):
        partial_txt = f"　|　{last} 年部分指标尚未发布，图中空白处即未发布"

    index_txt = ""
    if any(is_chain_index(table.unit_of(n)) for n in table.indicators):
        index_txt = "　|　指数图：虚线 100 表示与上年持平，高于/低于 100 分色"

    if len(years) == 1:
        return f"{years[0]} 年 · {unit_txt}{drop_txt}{empty_txt}{partial_txt}"
    if len(years) <= 4:
        # 只有三四个观测点时年份往往不等距，必须说明横轴是按真实年份画的
        return (
            f"{'、'.join(map(str, years))} 年 · {len(table.indicators)} 个指标 · "
            f"每格独立纵轴 · {unit_txt}{drop_txt}{empty_txt}{partial_txt}{index_txt}"
            "　|　观测年份不等距，横轴按实际年份位置绘制"
        )
    gaps = [y for y in range(min(years), max(years) + 1) if y not in years]
    gap_txt = f"　|　缺 {('、'.join(map(str, gaps)))} 年" if gaps else ""
    return (
        f"{min(years)}–{max(years)} 年 · {len(table.indicators)} 个指标 · "
        f"每格独立纵轴 · {unit_txt}{gap_txt}{drop_txt}{empty_txt}{partial_txt}{index_txt}"
    )


def _footnote(table: StjTable, theme: str = "") -> str:
    """页脚：数据来源 + 原始尾注要点 + 该表的已知口径提示。

    Args:
        table: 数据表。
        theme: 主题名，用于查口径提示表。
    """
    note = next((n for n in table.notes if n.startswith("注")), "")
    note = f"　{note}" if note else ""
    caveats = [DATA_CAVEATS.get(theme, "")]
    if theme.startswith(_FUND_FLOW_PREFIX):
        caveats.append(_FUND_FLOW_CAVEAT)
    caveat = "".join(f"　【注意】{c}" for c in caveats if c)
    return f"数据来源：国家统计局 · 年度数据　|　图表由 stj 脚本生成，数值未做插补。{note}{caveat}"


def _safe_name(text: str) -> str:
    """把主题名整理成合法文件名（去掉 Windows 不允许的字符，限长 60 字）。"""
    cleaned = "".join("_" if ch in '\\/:*?"<>|' else ch for ch in text).strip(" .")
    return (cleaned or "未命名")[:60]


def _theme_of(path: Path, category: str, table: StjTable) -> str:
    """确定图的主题名，按「人工登记 → 有描述性的文件名 → 首个指标名」依次回退。

    官方导出名一律是「年度数据 (n)」，没有信息量；把文件整理成描述性名字之后，
    文件名本身就是最合适的标题（去掉归档序号前缀），不必再登记一遍。
    """
    registered = THEMES.get((category, path.name))
    if registered:
        return registered
    if not _GENERIC_STEM.match(path.stem):
        return _ORDER_PREFIX.sub("", path.stem)
    return table.indicators[0]


def render_one(src_path: Path, category: str, used: set[str]) -> tuple[Path, str]:
    """为一个导出文件生成一张图。

    Args:
        src_path: 源文件（CSV 或 Excel）。
        category: 分类名，用于查主题登记表。
        used: 本分类已占用的图片名（不含扩展名），用于给重名图加序号避免互相覆盖。

    Returns:
        (图片路径, 主题名)。

    Raises:
        ValueError: 文件无法解析。
    """
    table = parse_table(src_path)
    theme = _theme_of(src_path, category, table)
    # 恒定的科目（净金融投资、三次产业合计行）先摘掉，免得占面板又让人怀疑画错了
    table, dropped = _drop_flat_rows(table)
    # 整列无值的年份（尚未发布的 2025）也摘掉，否则图右侧留一条通栏灰带
    table, empty_years = _drop_empty_years(table)
    subtitle = _subtitle(table, dropped, empty_years)

    stem = _safe_name(theme)
    unique, n = stem, 2
    while unique in used:
        unique = f"{stem} ({n})"
        n += 1
    used.add(unique)
    out_path = src_path.with_name(f"{unique}.png")

    if len(table.years) == 1:
        return (
            plot_single_year_dots(
                table,
                out_path,
                title=theme,
                subtitle=subtitle,
                footnote=_footnote(table, theme),
            ),
            theme,
        )

    # 列数取「空格最少」者，同分时偏向 4 列，避免出现 5 个指标排成 4+1 的孤行
    n_ind = len(table.indicators)
    ncols = min((3, 4), key=lambda c: (-(-n_ind // c) * c - n_ind, -c))
    ncols = min(ncols, n_ind)

    # 观测年份太少时面积图会连成一块斜面，改用真时间轴散点（点只落在有数据的年份上）
    if len(table.years) <= 4:
        return (
            plot_indicator_points(
                table,
                out_path,
                title=theme,
                subtitle=subtitle,
                footnote=_footnote(table, theme),
                ncols=ncols,
            ),
            theme,
        )

    return (
        plot_indicator_grid(
            table,
            out_path,
            title=theme,
            subtitle=subtitle,
            footnote=_footnote(table, theme),
            ncols=ncols,
        ),
        theme,
    )


def check_category(category_dir: Path) -> list[str]:
    """只体检、不画图：把每个文件的解析结果摊开，先确认格式能被识别。

    新导出的文件可能换了编码、分隔符或表头写法，先跑这一步能避免「图出来了但数据是错的」。

    Args:
        category_dir: 分类目录。

    Returns:
        每个文件一行的体检结论。
    """
    lines: list[str] = []
    for src in _sources(category_dir):
        try:
            t = parse_table(src)
        except EmptyTable:
            lines.append(f"[—] {src.name} — 空表（只有注释、无数据行），已跳过")
            continue
        except (ValueError, UnicodeDecodeError) as exc:
            lines.append(f"[X] {src.name} — 无法解析：{exc}")
            continue
        years = t.years
        gaps = [y for y in range(min(years), max(years) + 1) if y not in years]
        units = sorted({u for u in t.units.values() if u}) or ["未标注"]
        missing = int(t.wide.isna().sum().sum())
        total = t.wide.size
        head = "、".join(t.indicators[:3]) + ("…" if len(t.indicators) > 3 else "")
        lines.append(
            f"[√] {src.name} — 指标 {len(t.indicators)} 个（{head}），"
            f"年份 {min(years)}–{max(years)}（{len(years)} 列）"
            f"，缺失 {missing}/{total}，单位 {'/'.join(units)[:20]}"
            + (f"，缺 {('、'.join(map(str, gaps)))} 年" if gaps else "")
        )
    return lines


def build_category(category_dir: Path, seen: dict[str, Path]) -> list[str]:
    """处理一个分类目录下的所有导出文件。

    Args:
        category_dir: 分类目录。
        seen: 已处理的 {内容哈希: 路径}，用于跳过重复导出。

    Returns:
        日志行列表。
    """
    logs: list[str] = []
    used: set[str] = set()
    for src in _sources(category_dir):
        digest = _digest(src)
        if digest in seen:
            logs.append(f"跳过（与 {seen[digest].name} 内容完全相同）：{src.name}")
            continue
        seen[digest] = src
        try:
            out, theme = render_one(src, category_dir.name, used)
        except EmptyTable:
            logs.append(f"跳过空表（无数据行）：{src.name}")
            continue
        except (ValueError, UnicodeDecodeError) as exc:
            logs.append(f"失败：{src.name} — {exc}")
            continue
        logs.append(f"生成：{out.name}　← {src.name}（{theme}）")
    return logs


def run_category(default_category: str, argv: list[str] | None = None) -> None:
    """命令行入口：处理指定分类（或全部非空分类）的导出文件。

    Args:
        default_category: 缺省分类名，一般传分类脚本所在目录名。
        argv: 命令行参数，缺省读 sys.argv。
    """
    parser = argparse.ArgumentParser(description="按分类目录批量生成图表")
    parser.add_argument("--category", default=None,
                        help="分类名，如 综合 / 人口 / 财政；缺省取与代码目录同名的分类")
    parser.add_argument("--scan-all", action="store_true", help="扫描所有非空分类目录")
    parser.add_argument("--check", action="store_true", help="只体检 CSV 能否解析，不出图")
    args = parser.parse_args(argv)

    seen: dict[str, Path] = {}
    nonempty = [d for d in sorted(ROOT.iterdir()) if d.is_dir() and _sources(d)]
    if args.scan_all:
        targets = nonempty
    elif args.category:
        targets = [ROOT / args.category]
    else:
        # 代码目录名与数据分类名对应；万一目录被改名，退回第一个有数据的分类
        preferred = ROOT / default_category
        targets = [preferred] if preferred.is_dir() else nonempty[:1]

    for target in targets:
        if not target.is_dir():
            print(f"[缺失] 目录不存在：{target}")
            continue
        if args.check:
            rows = check_category(target)
            if not rows:
                print(f"\n=== {target.name} 体检（0 个文件，该分类还没有数据）")
                continue
            print(f"\n=== {target.name} 体检（{len(rows)} 个文件）")
            for line in rows:
                print("  " + line)
            continue
        logs = build_category(target, seen)
        print(f"\n=== {target.as_posix()} （{len(logs)} 项）")
        for line in logs:
            print("  " + line)
