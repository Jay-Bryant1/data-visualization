"""国家统计局「年度数据」导出 CSV 的解析与通用指标计算。

所有绘图脚本共用本模块，避免解析逻辑在各文件里重复实现。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pandas as pd

_ENCODINGS: Final[tuple[str, ...]] = ("utf-8-sig", "gb18030", "gbk")
_UNIT_PATTERN: Final = re.compile(r"^(?P<name>.*?)\s*[\(（](?P<unit>[^\)）]{1,10})[\)）]\s*$")
_YEAR_PATTERN: Final = re.compile(r"^(?P<year>\d{4})\s*年?$")
_NA_TOKENS: Final[frozenset[str]] = frozenset({"", "-", "—", "…", "...", "nan", "None", "null"})
_NOTE_PREFIXES: Final[tuple[str, ...]] = ("注：", "注:", "数据来源", "说明")
_DATA_DIRNAME: Final[str] = "年度数据"


class EmptyTable(ValueError):
    """导出文件只有表头元数据与尾注，没有任何数据行。

    官网对某些「仅说明、无数据」的条目也会给出下载按钮，导出的就是一个空表；
    这种情况要单独识别出来提示跳过，不能报成「格式无法解析」让人去改文件。
    """


def data_root(start: str | Path | None = None) -> Path:
    """定位项目的「年度数据」目录。

    代码与数据分处两个并列目录（``代码/<分类>/`` 与 ``年度数据/<分类>/``），
    脚本不能靠相对 cwd 找数据，因此从本文件位置逐级上溯，找到含「年度数据」
    子目录的那一层。这样在项目根、在代码目录、在任意其它 cwd 下执行结果一致。

    Args:
        start: 上溯起点，默认为本文件所在目录。

    Returns:
        「年度数据」目录的绝对路径。

    Raises:
        FileNotFoundError: 上溯到磁盘根仍未找到。
    """
    here = Path(start).resolve() if start is not None else Path(__file__).resolve().parent
    if here.is_file():
        here = here.parent
    for parent in (here, *here.parents):
        candidate = parent / _DATA_DIRNAME
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"自 {here} 上溯未找到「{_DATA_DIRNAME}」目录")


@dataclass(slots=True)
class StjTable:
    """一个导出文件的解析结果。

    Attributes:
        path: 源文件路径。
        wide: 宽表，index=指标名（已剥离单位），columns=年份(int，升序)。
        units: 指标 → 单位。
        notes: 尾注与数据来源。
    """

    path: Path
    wide: pd.DataFrame
    units: dict[str, str | None] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def years(self) -> list[int]:
        """升序年份列表。"""
        return [int(y) for y in self.wide.columns]

    @property
    def indicators(self) -> list[str]:
        """指标名列表（保持文件中的原始顺序）。"""
        return list(self.wide.index)

    def unit_of(self, indicator: str) -> str:
        """返回指标单位，缺失时返回空串。"""
        return self.units.get(indicator) or ""


def _decode(raw: bytes, path: Path) -> str:
    """按候选编码依次解码。"""
    for enc in _ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", raw[:32], 0, 1, f"无法识别 {path.name} 的编码")


def _split_row(line: str) -> list[str]:
    """拆行：按实际分隔符切分，并去掉行首尾的空字段。

    导出文件的列分隔符并不稳定：逗号为主，但同一文件里指标名与单位之间又用制表符，
    个别版本整行改成制表符分隔。这里按「制表符数 > 逗号数」自适应，
    避免换一个导出批次就把整行读成一个字段。
    """
    delimiter = "\t" if line.count("\t") > line.count(",") else ","
    cells = [c.strip().strip("\t").replace("\u3000", " ").strip() for c in line.split(delimiter)]
    while cells and cells[-1] == "":
        cells.pop()
    while cells and cells[0] == "":
        cells.pop(0)
    return cells


def _to_float(value: str) -> float | None:
    """把单元格转成 float；空值与各类占位符返回 None（保留为缺失，不填 0）。"""
    text = value.strip().replace(",", "").replace(" ", "")
    if text in _NA_TOKENS:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _looks_numeric(values: list[str]) -> bool:
    """判断字段是否全部可解释为数值或缺失标记（用于区分多行尾注）。"""
    if not values:
        return False
    return all(v.strip() in _NA_TOKENS or _to_float(v) is not None for v in values)


def _read_matrix(path: Path) -> list[list[str]]:
    """把源文件读成二维字符串矩阵，屏蔽 CSV / Excel 的差异。

    Args:
        path: .csv / .xlsx 路径。

    Returns:
        逐行的单元格列表，已去除全空行。

    Raises:
        ValueError: 扩展名不支持，或表格为空。
    """
    suffix = path.suffix.lower()
    rows: list[list[str]] = []

    if suffix in {".csv", ".txt"}:
        text = _decode(path.read_bytes(), path)
        for line in text.splitlines():
            cells = _split_row(line)
            if cells:
                rows.append(cells)
    elif suffix == ".xlsx":
        # 新版站点「下载」默认可能是 Excel；读成字符串避免 2024 年被当成 2024.0
        frame = pd.read_excel(path, header=None, dtype=str, engine="openpyxl")
        for record in frame.itertuples(index=False, name=None):
            cells = [("" if pd.isna(c) else str(c)).strip() for c in record]
            while cells and cells[-1] == "":
                cells.pop()
            if cells:
                rows.append(cells)
    elif suffix == ".xls":
        raise ValueError(f"{path.name}: 不支持旧版 .xls，请在 Excel 里另存为 .xlsx 或 CSV")
    else:
        raise ValueError(f"{path.name}: 不支持的扩展名 {path.suffix}")

    if not rows:
        raise ValueError(f"{path.name}: 文件为空")
    return rows


def _year_of(cell: str) -> int | None:
    """把表头单元格解析成年份（接受 2024 / 2024年 / 2024 年）。"""
    m = _YEAR_PATTERN.match(cell.strip())
    return int(m.group("year")) if m else None


def _locate_header(rows: list[list[str]]) -> tuple[int, dict[int, int]] | None:
    """定位表头行，返回 (行号, {列号: 年份})。

    依次尝试两种排版：
      1. 常规版式——首列为「指标」，其余列是年份；
      2. 转置版式——首列是年份，首行其余列是指标名（新站点个别表会这么导出）。
    """
    for idx, row in enumerate(rows):
        if not row or row[0] != "指标":
            continue
        mapping = {col: y for col, cell in enumerate(row[1:], start=1) if (y := _year_of(cell))}
        if mapping:
            return idx, mapping

    # 回退：首列就是年份的竖排版，表头行取其上一行
    col_years = {r: y for r, row in enumerate(rows) if row and (y := _year_of(row[0]))}
    if len(col_years) >= 2:
        return min(col_years) - 1, {}

    return None


def parse_table(path: str | Path) -> StjTable:
    """解析一个导出文件（CSV 或 Excel）为「指标 × 年份」宽表。

    原始格式并不规范：前两行是元数据、表头与行尾各多一个逗号、指标名与单位用制表符
    分隔、末尾有多行「注：…」尾注。这些差异在这里一次性抹平。

    Args:
        path: 源文件路径。

    Returns:
        StjTable；数值为 float64，缺失保留 NaN。

    Raises:
        ValueError: 找不到表头或年份列，或竖向版式无法还原。
    """
    path = Path(path)
    rows = _read_matrix(path)
    located = _locate_header(rows)
    if located is None:
        # 全表没有任何数值 → 这是官网导出的空表（只有「数据库/时间/注」几行），不是格式坏了
        if not any(_to_float(cell) is not None for row in rows for cell in row):
            raise EmptyTable(f"{path.name}: 空表（只有表头与注释，无数据行）")
        raise ValueError(f"{path.name}: 未找到表头行（既无『指标』列，也无年份列）")

    header_idx, year_cols = located
    records: dict[str, list[float | None]] = {}
    units: dict[str, str | None] = {}
    notes: list[str] = []
    years: list[int] = []

    if year_cols:  # 常规版式：行=指标，列=年份
        years = sorted(year_cols.values())
        ordered_cols = [col for col, _ in sorted(year_cols.items(), key=lambda kv: kv[1])]
        for row in rows[header_idx + 1 :]:
            label = row[0]
            values = [row[c] if c < len(row) else "" for c in ordered_cols]

            # 尾注可能占多行，既无逗号也不含数值，统一收进 notes
            if label.startswith(_NOTE_PREFIXES) or len(row) == 1 or not _looks_numeric(values):
                notes.append(" ".join(row))
                continue

            m = _UNIT_PATTERN.match(label)
            name = m.group("name").strip() if m else label.strip()
            if not name or name in records:
                continue
            units[name] = m.group("unit").strip() if m else None
            records[name] = [_to_float(v) for v in values]
    else:  # 竖向版式：行=年份，列=指标
        if header_idx < 0:
            raise ValueError(f"{path.name}: 竖向版式缺少指标名表头行")
        header = rows[header_idx]
        names = [(col, cell) for col, cell in enumerate(header) if col and cell]
        year_rows = [
            (r, _year_of(row[0])) for r, row in enumerate(rows) if r > header_idx and row and _year_of(row[0])
        ]
        if not names or len(year_rows) < 2:
            raise ValueError(f"{path.name}: 竖向版式无法还原为指标×年份表")
        years = sorted(y for _, y in year_rows)
        for col, raw_name in names:
            m = _UNIT_PATTERN.match(raw_name)
            name = m.group("name").strip() if m else raw_name.strip()
            units[name] = m.group("unit").strip() if m else None
            by_year = {y: (rows[r][col] if col < len(rows[r]) else "") for r, y in year_rows}
            records[name] = [_to_float(by_year[y]) for y in years]

    if not records:
        raise ValueError(f"{path.name}: 未解析到任何数值行")

    wide = pd.DataFrame.from_dict(records, orient="index", columns=years, dtype="float64")
    wide.index.name = "指标"
    return StjTable(path=path, wide=wide, units=units, notes=notes)


def parse_stj_csv(path: str | Path) -> StjTable:
    """兼容旧调用：等价于 ``parse_table``。"""
    return parse_table(path)



def cagr(first: float, last: float, periods: int) -> float:
    """复合年均增长率；用 CAGR 而非首末相减，避免年化失真。

    Raises:
        ValueError: 期初非正或期间非正。
    """
    if first <= 0 or periods <= 0:
        raise ValueError(f"CAGR 无定义：first={first}, periods={periods}")
    return (last / first) ** (1 / periods) - 1


def change_pct(series: pd.Series) -> float | None:
    """首末有效值的变化率（小数）；无法定义时返回 None。

    期初值必须为正。货物和服务净出口的贡献率常年为负，此时 ``末/首-1`` 会给出
    「首 -10、末 +50 → -600%」这种算术上成立、经济上毫无意义的数字，
    与其画一个会被误读的百分比，不如不标。

    Args:
        series: 待计算序列（可含 NaN）。

    Returns:
        变化率；有效点少于 2 个或期初值非正时返回 None。
    """
    valid = series.dropna()
    if len(valid) < 2 or valid.iloc[0] <= 0:
        return None
    return float(valid.iloc[-1] / valid.iloc[0] - 1)


def fmt_number(value: float, unit: str = "") -> str:
    """按量级选择 亿 / 万 / 原值 的紧凑写法，便于放进小图标注。"""
    abs_v = abs(value)
    if abs_v >= 1e8:
        return f"{value / 1e8:,.2f} 亿{unit}"
    if abs_v >= 1e4:
        return f"{value / 1e4:,.0f} 万{unit}"
    return f"{value:,.0f}{unit}"


def segments(series: pd.Series) -> list[list[int]]:
    """把含缺失的序列切成连续有值的片段，返回各片段的年份下标。

    断档年份不能连线，否则等于凭空插值出一条并不存在的趋势。
    """
    idx: list[list[int]] = []
    current: list[int] = []
    for i, v in enumerate(series.to_numpy()):
        if pd.isna(v):
            if current:
                idx.append(current)
                current = []
        else:
            current.append(i)
    if current:
        idx.append(current)
    return idx


def leaf_rows(series: pd.Series, tol: float = 0.0002) -> pd.Series:
    """从含汇总行的序列中剔掉汇总项，只保留末级科目。

    判定规则：自下而上维护一个栈，若某行的数值等于栈顶若干节点之和（相对误差 < tol），
    则这些节点是它的子项，该行是汇总行。统计局的导出表按「总数 → 大类 → 细分」
    自上而下排布，单年构成图若不剔除汇总行，会与明细重复计数（占比之和远超 100%）。

    Args:
        series: index=科目名，值为数值，须按原始表顺序（父在子前）。
        tol: 判定汇总的相对容差；取 0.02% 是为了容下整数求和的舍入，又不至于把巧合同和误判为父子关系。

    Returns:
        仅含末级科目的序列（保持原顺序）。

    Raises:
        ValueError: 序列为空。
    """
    if series.empty:
        raise ValueError("leaf_rows: 输入序列为空")

    stack: list[tuple[str, float, bool]] = []  # (名称, 数值, 是否汇总行)
    order: dict[str, int] = {name: i for i, name in enumerate(series.index)}
    leaves: list[str] = []

    for name, value in reversed(list(series.items())):
        value = float(value)
        acc, take = 0.0, 0
        for depth in range(len(stack), 0, -1):
            acc += stack[depth - 1][1]
            if abs(acc - value) <= abs(value) * tol:
                take = len(stack) - depth + 1
                break
            if acc > value * (1 + tol):
                break

        if take:
            for child_name, _v, is_agg in stack[len(stack) - take :]:
                if not is_agg:  # 只有明细行才是真正的末级；汇总行内部已记录过
                    leaves.append(child_name)
            del stack[len(stack) - take :]
            stack.append((name, value, True))
        else:
            stack.append((name, value, False))

    leaves.extend(n for n, _v, is_agg in stack if not is_agg)

    # 全部被判为汇总（表里没有明细）时退回原序列，避免画出空图
    if not leaves:
        return series
    return series.reindex(sorted(leaves, key=order.__getitem__))
