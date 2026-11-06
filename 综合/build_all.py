"""综合分类的出图入口。

通用批量流程（扫描、去重、选图版、写脚注）在**同目录**的 ``render.py``；
本文件只固定「分类名 = 本目录名」，保证 ``统计局代码/<分类>/`` 与 ``年度数据/<分类>/`` 一一对应。

用法：
    python 统计局代码/综合/build_all.py              # 处理 年度数据/综合
    python 统计局代码/综合/build_all.py --check       # 只体检，不出图
    python 统计局代码/综合/build_all.py --scan-all    # 处理所有非空分类
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render import run_category  # noqa: E402

if __name__ == "__main__":
    run_category(Path(__file__).resolve().parent.name)
