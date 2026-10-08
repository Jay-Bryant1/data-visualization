# -*- coding: utf-8 -*-
"""
嘉兴大学 · 专业设置三层级旭日图（组织架构风格 Sunburst Chart）
================================================================================
层级结构 :  学校（中心圆） → 学院（第一圈） → 专业（第二圈）
扇区面积 :  由「该节点所包含的专业数量」决定（源数据没有学生人数，故完全不涉及人数）
标签显示 :  只显示名称，不显示数值、不显示百分比
交互悬停 :  鼠标移到任意扇区，显示「学院 → 专业」的完整路径

依赖    :  仅需 plotly（本机 plotly 5.24.1 可直接运行，无需 pandas / matplotlib）
           pip install plotly
运行    :  python jiaxing_university_sunburst.py
输出    :  sunburst_嘉兴大学.html  —— 可交互网页：悬停看路径、点击下钻、双击返回
           sunburst_嘉兴大学.png   —— 仅当本机额外安装了 kaleido 时才会一起导出静态图
"""

from __future__ import annotations

import os

import plotly.graph_objects as go

# =================================================================================
# 1. 数据准备（全部直接内嵌，方便一键运行 / 修改）
# =================================================================================
# 结构：{学院: [专业, ...]}；程序会自动统计专业数量，用它决定每个扇区的面积。
SCHOOL_NAME = "嘉兴大学"

COLLEGE_MAJORS = {
    "经济学院": ["经济学", "金融学", "跨境电子商务", "数字经济"],
    "商学院": ["会计学", "人力资源管理", "财务管理", "市场营销",
               "工商管理", "信息管理与信息系统", "物流管理"],
    "机械工程学院": ["机械设计制造及其自动化", "电气工程及其自动化", "车辆工程",
                     "机器人工程", "智能制造工程", "机械设计制造及其自动化（中外合作办学）"],
    "信息科学与工程学院": ["电子信息工程", "通信工程"],
    "人工智能学院": ["计算机科学与技术", "人工智能", "网络工程", "软件工程"],
    "医学院": ["临床医学", "护理学", "药学", "麻醉学"],
    "设计学院": ["视觉传达设计", "服装设计与工程", "工业设计", "数字媒体艺术"],
    "生物与化学工程学院": ["化学工程与工艺", "应用化学", "生物工程", "环境工程", "制药工程"],
    "材料与纺织工程学院": ["纺织工程", "高分子材料与工程", "轻化工程",
                           "新能源材料与器件", "非织造材料与工程"],
    "建筑工程学院": ["土木工程", "工程管理", "建筑环境与能源应用工程", "建筑学"],
    "文法学院": ["汉语言文学", "汉语国际教育", "法学", "知识产权"],
    "外国语学院": ["英语", "日语"],
    "平湖师范学院": ["学前教育", "小学教育", "体育教育"],
    "数据科学学院": ["数学与应用数学", "应用统计学", "金融数学", "数据科学与大数据技术"],
}

# =================================================================================
# 2. 颜色与字体配置
# =================================================================================
FONT_FAMILY = ('"Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", '
               '"Source Han Sans SC", "Hiragino Sans GB", sans-serif')

# 2.1 中心：嘉兴大学「红船红」——整张图最深、最饱和的一抹红，让中心层级最醒目
RED_BOAT_RED = "#A3171D"

# 2.2 第一圈（学院）：同色系渐变（深红 → 暗红 → 砖红 → 朱红 → 赤陶）。
#     下面的锚点会被线性插值成与学院数量相同的过渡色：整体统一，相邻学院又有区分度。
COLLEGE_RAMP_ANCHORS = [
    "#7E1218",   # 深红 / 酒红
    "#931919",   # 暗红
    "#A62320",   # 砖红
    "#B53027",   # 朱红
    "#BF4430",   # 赤朱
    "#BC5B45",   # 赤陶
    "#B86A55",   # 陶土色（收尾，仍属红色系，避免发灰发粉）
]

# 2.3 第二圈（专业）：把所属学院的颜色与白色混合得到浅色，实现「同学院 = 同色系」，
#     并在同一学院内部做极轻微的深浅变化，让层级关系一眼可辨。
MAJOR_TINT_LIGHT = 0.52      # 混入白色的比例（越大越浅）
MAJOR_TINT_DARK = 0.70

# 2.4 分隔线与文字
SEPARATOR_COLOR = "#FFFFFF"  # 用白色缝隙代替描边，形成组织架构图的「块面」感
SEPARATOR_WIDTH = 1.6

DARK_TEXT = "#2E2A2A"
LIGHT_TEXT = "#FFFFFF"

# 5 个字号：中心 / 学院 / 专业（普通）/ 专业（超长名）/ 标题
FONT_CENTER, FONT_COLLEGE, FONT_MAJOR, FONT_MAJOR_MIN, FONT_TITLE = 30, 14, 13, 9, 26
LONG_LABEL_CHARS = 12        # 超过这个字数的专业名会自动缩小字号，防止被隐藏


# =================================================================================
# 3. 颜色小工具
# =================================================================================
def hex_to_rgb(h: str):
    """'#RRGGBB' -> (r, g, b)"""
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb) -> str:
    """(r, g, b) -> '#RRGGBB'"""
    return "#%02X%02X%02X" % tuple(max(0, min(255, int(round(c)))) for c in rgb)


def lerp(c1, c2, t: float):
    """两个颜色线性插值，t∈[0,1]"""
    return tuple(a + (b - a) * t for a, b in zip(c1, c2))


def mix(c1: str, c2: str, t: float) -> str:
    """把颜色 c1 按比例 t 混向 c2（t=0 全为 c1，t=1 全为 c2）"""
    return rgb_to_hex(lerp(hex_to_rgb(c1), hex_to_rgb(c2), t))


def make_ramp(anchors, n: int):
    """在若干锚点颜色之间线性插值，生成 n 个过渡色（用于学院圈的统一渐变）。"""
    rgb = [hex_to_rgb(a) for a in anchors]
    if n <= 1:
        return [rgb_to_hex(rgb[0])]
    segs = len(rgb) - 1
    out = []
    for i in range(n):
        pos = i / (n - 1) * segs           # 把 [0,1] 映射到锚点区间
        k = min(int(pos), segs - 1)
        out.append(rgb_to_hex(lerp(rgb[k], rgb[k + 1], pos - k)))
    return out


def _luminance(hex_color: str) -> float:
    """WCAG 相对亮度"""
    def chan(u):
        u /= 255.0
        return u / 12.92 if u <= 0.03928 else ((u + 0.055) / 1.055) ** 2.4

    r, g, b = hex_to_rgb(hex_color)
    return 0.2126 * chan(r) + 0.7152 * chan(g) + 0.0722 * chan(b)


def text_color_on(hex_color: str) -> str:
    """在白色文字与深色文字之间选对比度更高的一个，保证任何底色上标签都清晰可读。"""
    lum = _luminance(hex_color)
    contrast_white = 1.05 / (lum + 0.05)
    contrast_dark = (lum + 0.05) / (_luminance(DARK_TEXT) + 0.05)
    return LIGHT_TEXT if contrast_white >= contrast_dark else DARK_TEXT


# =================================================================================
# 4. 组装旭日图所需的层级数组
# =================================================================================
# Plotly 旭日图用 ids / parents 描述父子关系：
#   id 写成完整路径（学校/学院/专业）保证唯一，parent 指向上一级的完整 id。
ROOT_ID = "jxu"

ids, labels, texts, parents, values = [], [], [], [], []
colors, text_colors, text_sizes, hover_texts = [], [], [], []

# 4.1 中心：学校（面积 = 全校专业总数）
total_majors = sum(len(v) for v in COLLEGE_MAJORS.values())
ids.append(ROOT_ID)
labels.append(SCHOOL_NAME)
texts.append(SCHOOL_NAME)
parents.append("")
values.append(total_majors)
colors.append(RED_BOAT_RED)
text_colors.append(LIGHT_TEXT)
text_sizes.append(FONT_CENTER)                 # 中心字号单独放大 → 中心最醒目
hover_texts.append(
    f"<b>{SCHOOL_NAME}</b><br>"
    f"<span style='color:#8C8C8C'>{len(COLLEGE_MAJORS)} 个学院 · {total_majors} 个专业</span>"
)

# 4.2 第一圈：学院（同色系渐变，面积 = 该学院专业数量）
college_names = list(COLLEGE_MAJORS.keys())
college_colors = make_ramp(COLLEGE_RAMP_ANCHORS, len(college_names))

for college, color in zip(college_names, college_colors):
    majors = COLLEGE_MAJORS[college]
    cid = f"{ROOT_ID}/{college}"
    ids.append(cid)
    labels.append(college)
    texts.append(college)
    parents.append(ROOT_ID)
    values.append(len(majors))
    colors.append(color)
    text_colors.append(text_color_on(color))
    text_sizes.append(FONT_COLLEGE)
    hover_texts.append(
        f"<b>{college}</b><br>"
        f"<span style='color:#8C8C8C'>共 {len(majors)} 个专业</span>"
    )

    # 4.3 第二圈：专业（学院主色的浅色同色系，面积恒为 1）
    for i, major in enumerate(majors):
        mid = f"{cid}/{major}"
        t = MAJOR_TINT_LIGHT + (MAJOR_TINT_DARK - MAJOR_TINT_LIGHT) * (
            i / max(len(majors) - 1, 1)
        )
        tint = mix(color, "#FFFFFF", t)
        ids.append(mid)
        labels.append(major)
        texts.append(major)
        parents.append(cid)
        values.append(1)
        colors.append(tint)
        text_colors.append(text_color_on(tint))
        # 名称特别长的专业自动缩小字号，尽量让它完整显示在扇区里
        if len(major) > LONG_LABEL_CHARS:
            text_sizes.append(max(FONT_MAJOR_MIN,
                                  round(FONT_MAJOR * LONG_LABEL_CHARS / len(major), 1)))
        else:
            text_sizes.append(FONT_MAJOR)
        # 悬停提示：学院 → 专业 的完整路径
        hover_texts.append(
            f"<b>{college} → {major}</b><br>"
            f"<span style='color:#8C8C8C'>{SCHOOL_NAME} · 本科专业</span>"
        )

# =================================================================================
# 5. 绘制图表
# =================================================================================
fig = go.Figure(
    go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",          # 父级面积 = 子级之和（学院面积即专业数量）
        text=texts,                    # 只用名称作为文本，绝不拼接数值 / 百分比
        textinfo="text",               # 只画文字，不画 value、不画 percent
        textfont=dict(size=text_sizes, color=text_colors, family=FONT_FAMILY),
        insidetextorientation="radial",  # 文字沿半径方向排布，最省空间也最整齐
        marker=dict(
            colors=colors,
            line=dict(color=SEPARATOR_COLOR, width=SEPARATOR_WIDTH),
        ),
        customdata=hover_texts,
        hovertemplate="%{customdata}<extra></extra>",   # <extra> 清空，去掉多余的 trace 名
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#E0D6D6",
            font=dict(family=FONT_FAMILY, size=14, color="#3A2E2E"),
        ),
        sort=False,                    # 保持数据原始顺序，结构稳定、便于对照
        rotation=0,                    # 起始角度：第一个学院从 12 点方向开始
        maxdepth=2,                    # 只展示到第二圈：学校 → 学院 → 专业
        name="",
    )
)

fig.update_layout(
    title=dict(
        text=f"<b>{SCHOOL_NAME} 专业设置</b>",
        x=0.5, xanchor="center", y=0.975,
        font=dict(family=FONT_FAMILY, size=FONT_TITLE, color="#8C1418"),
    ),
    width=1240,
    height=1240,
    margin=dict(t=100, l=16, r=16, b=16),
    paper_bgcolor="#FFFFFF",       # 纯白背景
    plot_bgcolor="#FFFFFF",
    showlegend=False,              # 组织架构图不需要图例
    font=dict(family=FONT_FAMILY, color="#3A2E2E"),
)

# =================================================================================
# 6. 输出
# =================================================================================
out_dir = os.path.dirname(os.path.abspath(__file__))

# 6.1 交互网页（推荐）：悬停显示路径、点击下钻、双击返回上一级
html_path = os.path.join(out_dir, "sunburst_嘉兴大学.html")
fig.write_html(html_path, include_plotlyjs=True, full_html=True)
print(f"[OK] 交互网页已生成：{html_path}")

# 6.2 静态图片（可选）：只有本机装有 kaleido 时才会成功
png_path = os.path.join(out_dir, "sunburst_嘉兴大学.png")
try:
    fig.write_image(png_path, scale=2, width=1240, height=1240)
    print(f"[OK] 静态图片已生成：{png_path}")
except Exception as exc:
    print(f"[跳过] 未导出 PNG（需要 pip install kaleido）：{str(exc)[:80]}")

# 6.3 想在浏览器里直接预览，取消下面这行的注释即可
# fig.show()
