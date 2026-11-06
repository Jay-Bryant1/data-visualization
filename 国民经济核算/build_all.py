"""国民经济核算分类的出图入口。

通用批量流程见同目录的 ``render.py``；本文件只固定「分类名 = 本目录名」。
本分类的文件已按官方表序整理成描述性文件名（``01-…``、``02-…``），
主题名直接取文件名并剥掉序号前缀，不需要主题登记表。

用法：
    python 统计局代码/国民经济核算/build_all.py            # 逐表出图
    python 统计局代码/国民经济核算/build_all.py --check     # 只体检，不出图
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render import run_category  # noqa: E402

if __name__ == "__main__":
    run_category(Path(__file__).resolve().parent.name)
