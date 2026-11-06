# -*- coding: utf-8 -*-
"""s07_dashboard.py —— Pyecharts 交互大屏（组件 12 个，双筛选器联动全部组件）。

组件清单：
  KPI 卡片区（7 项指标）｜① 品类资产量条形图 ② 队列结构环形图 ③ 批次质量趋势折线（含缩放）
  ④ 价值-性价比散点（ABC 三系列，含缩放）⑤ ABC 占比分组柱 ⑥ 点位×品类堆积柱
  ⑦ 投产漏斗 ⑧ 平均质量分仪表盘 ｜ 明细表（Top15）｜ 筛选器（队列 × 听诊侧）
联动实现：两个下拉筛选器对同一份数据交叉过滤，16 种组合的完整 ECharts option 预生成后由前端 setOption 切换；
折线与散点内置 dataZoom 缩放，图例可点选隐藏系列，散点支持框选缩放。
输出：dashboard/听诊数据资产经营大屏.html（echarts.min.js 内联，离线可开）
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Bar, Funnel, Gauge, Line, Pie, Scatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cfg

PALETTE = ["#1F6FEB", "#2FBFA0", "#F0883E", "#E5534B", "#8250DF",
           "#DB61A2", "#57606A", "#79B8FF", "#3FB950", "#D29922"]
COHORTS = [("all", "全部队列"), ("heart_pure", "纯心音"), ("lung_pure", "纯肺音"), ("mixed", "混合")]
SIDES = [("all", "全部点位"), ("左", "左侧听诊"), ("右", "右侧听诊"), ("中", "心前区")]
CHART_W, CHART_H = "470px", "300px"


def load() -> pd.DataFrame:
    L = pd.read_csv(cfg.DATA / "asset_ledger.csv")
    F = pd.read_csv(cfg.DATA / "features_asset.csv")
    M = pd.read_csv(cfg.DATA / "model_table.csv")
    keep = ["asset_id", "quality_score", "info_score", "scarcity_score", "value_index",
            "total_cost", "value_cost_ratio", "qc_pass", "group_size", "category"]
    D = L.merge(F[keep], on="asset_id", how="left")
    D = D.merge(M[["asset_id", "abc", "cluster"]], on="asset_id", how="left")
    D["abc"] = D["abc"].fillna("-")
    D["side"] = D["side"].fillna("中")
    return D


def grid(opt, top="20%", bottom="18%", left="12%", right="7%"):
    """ECharts 原生 grid 配置：给坐标轴类图表留出标题与底部图例的空间。"""
    opt["grid"] = [{"top": top, "bottom": bottom, "left": left, "right": right, "containLabel": True}]
    return opt


def legend_bottom():
    return opts.LegendOpts(pos_bottom="1%", pos_left="center", type_="scroll")


def title(t, sub=""):
    return opts.TitleOpts(title=t, subtitle=sub, pos_left="10", pos_top="6", item_gap=6, title_textstyle_opts=opts.TextStyleOpts(color="#f2f7fc", font_size=13), subtitle_textstyle_opts=opts.TextStyleOpts(color="#9fb7cf", font_size=9.5))


def c_bar_cat(d: pd.DataFrame):
    top = d["category"].value_counts().head(12)[::-1]
    c = Bar(init_opts=opts.InitOpts(width=CHART_W, height=CHART_H, bg_color="transparent"))
    c.add_xaxis(list(top.index)).add_yaxis("资产条数", [int(v) for v in top.values],
                                           bar_width="58%",
                                           label_opts=opts.LabelOpts(is_show=True, position="right",
                                                                     font_size=10))
    c.set_colors([PALETTE[0]])
    c.set_global_opts(title_opts=title("① 品类资产量 Top12", "单位：条"),
                      tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="shadow"),
                      legend_opts=opts.LegendOpts(is_show=False),
                      xaxis_opts=opts.AxisOpts(name="品类",
                                               axislabel_opts=opts.LabelOpts(rotate=42, interval=0, font_size=8.5)),
                      yaxis_opts=opts.AxisOpts(name="条数"))
    return c.dump_options()


def c_pie_cohort(d: pd.DataFrame):
    s = d["cohort_zh"].value_counts()
    c = Pie(init_opts=opts.InitOpts(width=CHART_W, height=CHART_H, bg_color="transparent"))
    c.add("队列", [(k, int(v)) for k, v in s.items()],
          radius=["34%", "54%"], center=["50%", "46%"], min_angle=4,
          label_opts=opts.LabelOpts(formatter="{b} {d}%", font_size=9, distance=6, color="#dbeafe"))
    c.set_colors(PALETTE[:len(s)])
    c.set_global_opts(title_opts=title("② 队列结构", "环形｜占比与绝对量"),
                      tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{b}: {c} 条（{d}%）"),
                      legend_opts=legend_bottom())
    return c.dump_options()


def c_line_batch(d: pd.DataFrame):
    g = d[d["in_model"] == 1].groupby("batch_label").agg(
        q=("quality_score", "mean"), v=("value_index", "mean"),
        n=("asset_id", "size"), c=("total_cost", "mean")).reset_index()
    g = g.sort_values("batch_label")
    x = list(g["batch_label"])
    c = Line(init_opts=opts.InitOpts(width=CHART_W, height=CHART_H, bg_color="transparent"))
    c.add_xaxis(x)
    c.add_yaxis("平均质量分", [round(float(v), 2) for v in g["q"]], is_smooth=True,
                symbol_size=7, label_opts=opts.LabelOpts(is_show=False))
    c.add_yaxis("平均价值指数", [round(float(v), 2) for v in g["v"]], is_smooth=True,
                symbol_size=7, label_opts=opts.LabelOpts(is_show=False))
    c.add_yaxis("平均成本×5", [round(float(v) * 5, 2) for v in g["c"]], is_smooth=True,
                symbol_size=7, linestyle_opts=opts.LineStyleOpts(type_="dashed"),
                label_opts=opts.LabelOpts(is_show=False))
    c.set_colors([PALETTE[0], PALETTE[2], PALETTE[1]])
    c.set_global_opts(title_opts=title("③ 入库批次走势", "折线｜下方滑块可缩放"),
                      tooltip_opts=opts.TooltipOpts(trigger="axis"),
                      legend_opts=opts.LegendOpts(pos_top="5%", pos_right="2%", item_gap=8,
                                                   textstyle_opts=opts.TextStyleOpts(font_size=9)),
                      xaxis_opts=opts.AxisOpts(name="批次"),
                      yaxis_opts=opts.AxisOpts(name="分 / 元×5"),
                      datazoom_opts=[opts.DataZoomOpts(type_="slider", height=16, pos_bottom=6),
                                     opts.DataZoomOpts(type_="inside")])
    return c.dump_options()


def c_scatter(d: pd.DataFrame):
    c = Scatter(init_opts=opts.InitOpts(width=CHART_W, height=CHART_H, bg_color="transparent"))
    c.add_xaxis([])
    for i, (abc, sub) in enumerate(d[d["abc"] != "-"].groupby("abc")):
        items = [opts.ScatterItem(value=[round(float(r.value_cost_ratio), 3), round(float(r.value_index), 2)],
                                  name=r.asset_id)
                 for r in sub.itertuples() if pd.notna(r.value_cost_ratio)]
        c.add_yaxis(f"{abc} 类（n={len(sub)}）", items, symbol_size=9,
                    label_opts=opts.LabelOpts(is_show=False))
    c.set_colors([PALETTE[3], PALETTE[2], PALETTE[0], PALETTE[4]])
    c.set_global_opts(title_opts=title("④ 价值 × 性价比散点", "图例可点选，支持框选缩放"),
                      tooltip_opts=opts.TooltipOpts(trigger="item"),
                      legend_opts=legend_bottom(),
                      xaxis_opts=opts.AxisOpts(type_="value", name="价值成本比", is_scale=True,
                                               name_location="middle", name_gap=26),
                      yaxis_opts=opts.AxisOpts(type_="value", name="价值指数", is_scale=True, name_location="middle",
                                               name_gap=38),
                      datazoom_opts=[opts.DataZoomOpts(type_="inside")])
    return c.dump_options()


def c_bar_abc(d: pd.DataFrame):
    m = d[d["in_model"] == 1]
    tot_v, tot_c = float(m["value_index"].sum()), float(m["total_cost"].sum())
    g = m.groupby("abc").agg(n=("asset_id", "size"), v=("value_index", "sum"),
                             c=("total_cost", "sum")).reindex(["A", "B", "C"]).fillna(0)
    x = list(g.index)
    c = Bar(init_opts=opts.InitOpts(width=CHART_W, height=CHART_H, bg_color="transparent"))
    c.add_xaxis(x)
    c.add_yaxis("数量占比%", [round(100 * v / max(float(g["n"].sum()), 1), 2) for v in g["n"]],
                label_opts=opts.LabelOpts(position="top", font_size=10))
    c.add_yaxis("价值占比%", [round(100 * v / max(tot_v, 1), 2) for v in g["v"]],
                label_opts=opts.LabelOpts(position="top", font_size=10))
    c.add_yaxis("成本占比%", [round(100 * v / max(tot_c, 1), 2) for v in g["c"]],
                label_opts=opts.LabelOpts(position="top", font_size=10))
    c.set_colors([PALETTE[0], PALETTE[1], PALETTE[2]])
    c.set_global_opts(title_opts=title("⑤ ABC 分层占比对比", "帕累托偏离度=价值占比−数量占比"),
                      tooltip_opts=opts.TooltipOpts(trigger="axis"),
                      legend_opts=legend_bottom(),
                      xaxis_opts=opts.AxisOpts(name="分层"), yaxis_opts=opts.AxisOpts(name="%"))
    return c.dump_options()


def c_bar_location(d: pd.DataFrame):
    cats = d[d["in_model"] == 1]["category"].value_counts().head(6).index.tolist()
    ct = pd.crosstab(d[d["in_model"] == 1]["location_zh"],
                     d[d["in_model"] == 1]["category"])
    ct = ct[[c for c in cats if c in ct.columns]]
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
    c = Bar(init_opts=opts.InitOpts(width=CHART_W, height=CHART_H, bg_color="transparent"))
    c.add_xaxis(list(ct.index.astype(str)))
    for col in ct.columns:
        c.add_yaxis(col, [int(v) for v in ct[col]], stack="总量",
                    label_opts=opts.LabelOpts(is_show=False))
    c.set_colors(PALETTE)
    c.set_global_opts(title_opts=title("⑥ 听诊点位 × 品类", "堆积柱｜图例可筛选品类"),
                      tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="shadow"),
                      legend_opts=legend_bottom(),
                      xaxis_opts=opts.AxisOpts(name="点位",
                                               axislabel_opts=opts.LabelOpts(rotate=42, interval=0, font_size=8.5)),
                      yaxis_opts=opts.AxisOpts(name="条数"))
    return c.dump_options()


def c_funnel(d: pd.DataFrame):
    n_all = len(d)
    uniq = int(d["is_primary_copy"].sum())
    ok = int(((d["is_primary_copy"] == 1) & (d["qc_pass"] == 1)).sum())
    ab = int((d["abc"].isin(["A", "B"])).sum())
    a = int((d["abc"] == "A").sum())
    c = Funnel(init_opts=opts.InitOpts(width=CHART_W, height=CHART_H, bg_color="transparent"))
    c.add("投产漏斗", [("筛选资产", n_all), ("内容唯一", uniq), ("质检达标", ok),
                      ("A+B 类", ab), ("A 类", a)],
          label_opts=opts.LabelOpts(position="inside", formatter="{b}: {c}", font_size=11, color="#fff"),
          gap=3, sort_="descending")
    c.set_colors(PALETTE[:5])
    c.set_global_opts(title_opts=title("⑦ 投产漏斗", "入库→唯一→达标→高价值"),
                      tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{b}: {c} 条"),
                      legend_opts=opts.LegendOpts(is_show=False))
    return c.dump_options()


def c_gauge(d: pd.DataFrame):
    m = d[d["in_model"] == 1]
    v = round(float(m["quality_score"].mean()), 1)
    c = Gauge(init_opts=opts.InitOpts(width=CHART_W, height=CHART_H, bg_color="transparent"))
    c.add("质量", [("平均质量分", v)], min_=0, max_=100, split_number=10, radius="76%",
          start_angle=210, end_angle=-30,
          detail_label_opts=opts.LabelOpts(formatter="{value} 分", font_size=13),
          title_label_opts=opts.LabelOpts(font_size=11))
    c.set_global_opts(title_opts=title("⑧ 平均质量分", "仪表盘｜0–100 分"))
    return c.dump_options()


def kpi(d: pd.DataFrame) -> dict:
    m = d[d["in_model"] == 1]
    return {
        "资产条数": f"{len(d):,}",
        "独立录音": f"{int(d['is_primary_copy'].sum()):,}",
        "A 类占比": f"{100 * float((m['abc'] == 'A').mean()) if len(m) else 0:.1f}%",
        "平均质量分": f"{float(m['quality_score'].mean()) if len(m) else 0:.1f}",
        "平均价值指数": f"{float(m['value_index'].mean()) if len(m) else 0:.1f}",
        "全成本合计": f"¥{float(m['total_cost'].sum()):,.0f}",
        "质检通过率": f"{100 * float(d['qc_pass'].mean()):.0f}%",
    }


def detail(d: pd.DataFrame):
    cols = ["asset_id", "file_name", "cohort_zh", "category", "location_zh", "gender_zh",
            "quality_score", "value_index", "total_cost", "abc"]
    t = d[d["in_model"] == 1].nlargest(15, "value_index")[cols]
    zh = {"asset_id": "资产号", "file_name": "文件名", "cohort_zh": "队列", "category": "品类组合",
          "location_zh": "听诊点位", "gender_zh": "性别", "quality_score": "质量分",
          "value_index": "价值指数", "total_cost": "全成本(元)", "abc": "ABC"}
    return {"head": [zh[c] for c in cols], "rows": [[("" if pd.isna(v) else (round(float(v), 2) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v))
                                   for v in row] for row in t.itertuples(index=False)]}


def build(D: pd.DataFrame) -> dict:
    data = {}
    for ck, cl in COHORTS:
        dc = D if ck == "all" else D[D["cohort"] == ck]
        for sk, sl in SIDES:
            ds = dc if sk == "all" else dc[dc["side"] == sk]
            if len(ds) == 0:
                continue
            data[f"{ck}|{sk}"] = {
                "label": f"{cl} · {sl}｜{len(ds)} 条",
                "kpi": kpi(ds), "tbl": detail(ds),
                "c1": grid(json.loads(c_bar_cat(ds)), top="20%", bottom="26%"),
                "c2": json.loads(c_pie_cohort(ds)),
                "c3": grid(json.loads(c_line_batch(ds)), top="26%", bottom="24%"),
                "c4": grid(json.loads(c_scatter(ds)), top="22%", bottom="24%", right="10%"),
                "c5": grid(json.loads(c_bar_abc(ds)), top="22%", bottom="26%"),
                "c6": grid(json.loads(c_bar_location(ds)), top="22%", bottom="30%"),
                "c7": json.loads(c_funnel(ds)), "c8": json.loads(c_gauge(ds)),
            }
    return data


HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>心肺听诊数据资产经营大屏</title>
<script>__ECHARTS__</script>
<style>
:root{--bg:#03070d;--card:#0b1524;--ink:#e8eef7;--sub:#8ea8c4;--line:#1f344c;--accent:#60d2ff;}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% 0%,#0a1930 0,#03070d 52%,#020407 100%);font-family:"Microsoft YaHei","PingFang SC",sans-serif;color:var(--ink)}
header{display:flex;align-items:center;gap:18px;padding:14px 22px;background:linear-gradient(90deg,#050b14,#0b1e35);border-bottom:2px solid #1f5a8f;box-shadow:0 8px 26px rgba(0,0,0,.38)}
header h1{font-size:20px;margin:0;letter-spacing:2px;color:#eaf2fb}
header .meta{font-size:12px;color:#9dbbd8;margin-left:6px}
.filters{margin-left:auto;display:flex;gap:10px}
select{height:30px;border:1px solid #315f8f;background:#071321;color:#dcecff;border-radius:5px;padding:0 10px;font-size:13px;color-scheme:dark}
select:focus{outline:2px solid #57a5e6}
.kpis{display:grid;grid-template-columns:repeat(7,1fr);gap:12px;padding:14px 22px 4px}
.kpi{background:linear-gradient(180deg,#0b1b2e,#071321);border:1px solid #1d3c5b;border-radius:8px;padding:10px 12px;box-shadow:0 8px 22px rgba(0,0,0,.24)}
.kpi .v{font-size:20px;font-weight:700;color:#6fdcff;line-height:1.2;text-shadow:0 0 18px rgba(80,190,255,.2)}
.kpi .k{font-size:11px;color:#9dbbd8;margin-top:3px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;padding:12px 22px}
.card{background:linear-gradient(180deg,rgba(12,25,43,.98),rgba(7,17,29,.98));border:1px solid #1b3652;border-radius:8px;padding:6px 4px 2px;box-shadow:0 10px 26px rgba(0,0,0,.32)}
.chart{width:100%;height:300px}
.hint{font-size:10px;color:#8aa4c0;text-align:right;padding:0 10px 2px}
.tablecard{grid-column:1/-1;background:linear-gradient(180deg,rgba(12,25,43,.98),rgba(7,17,29,.98));border:1px solid #1b3652;border-radius:8px;padding:10px 14px;margin:0 22px 18px;box-shadow:0 10px 26px rgba(0,0,0,.32)}
.tablecard h3{margin:2px 0 8px;font-size:13px;color:var(--ink)}
table{border-collapse:collapse;width:100%;font-size:11.5px}
th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:center;white-space:nowrap}
th{background:#10243a;color:#c9ddf0;position:sticky;top:0}
tbody tr:hover{background:#102942}
.scroll{max-height:230px;overflow:auto}
footer{color:#8fb3d4;font-size:11px;padding:0 22px 16px;line-height:1.7}
</style></head><body>
<header><h1>心肺听诊数据资产经营大屏</h1>
<span class="meta" id="ctx"></span>
<div class="filters"><select id="fCohort"></select><select id="fSide"></select></div></header>
<div class="kpis" id="kpis"></div>
<div class="grid">
<div class="card"><div class="chart" id="c1"></div></div>
<div class="card"><div class="chart" id="c2"></div><div class="hint">点击扇区可下钻到该队列</div></div>
<div class="card"><div class="chart" id="c3"></div></div>
<div class="card"><div class="chart" id="c4"></div></div>
<div class="card"><div class="chart" id="c5"></div></div>
<div class="card"><div class="chart" id="c6"></div></div>
<div class="card"><div class="chart" id="c7"></div></div>
<div class="card"><div class="chart" id="c8"></div></div>
</div>
<div class="tablecard"><h3>⑨ 资产明细表｜当前筛选下价值指数 Top15（点击表头所在行可横向滚动查看全部字段）</h3>
<div class="scroll"><table id="tbl"></table></div></div>
<footer>联动与下钻：右上两个下拉框交叉过滤全部 12 类组件；点击 ② 队列结构扇区可下钻到该队列；
③ 折线与 ④ 散点内置 dataZoom 缩放/框选；各图图例可点选隐藏系列。
数据来源：HS.zip / LS.zip / Mix.zip 官方一次性下载文件（未使用爬虫）；口径与阈值见 代码/cfg.py，
随机种子 __SEED__；全部组件由 Python（Pandas + Pyecharts/ECharts）生成，未使用 Power BI / Tableau / Fine BI。</footer>
<script>
const DATA = __DATA__;
const COHORTS = __COHORTS__, SIDES = __SIDES__;
const charts = {};
function fill(sel, arr){sel.innerHTML = arr.map(([k,l])=>`<option value="${k}">${l}</option>`).join("");}
function ensure(id){if(!charts[id]) charts[id] = echarts.init(document.getElementById(id), "dark");return charts[id];}
function renderTable(t){
  const head = `<thead><tr>${t.head.map(h=>`<th>${h}</th>`).join("")}</tr></thead>`;
  const body = `<tbody>${t.rows.map(r=>`<tr>${r.map(v=>`<td>${v}</td>`).join("")}</tr>`).join("")}</tbody>`;
  document.getElementById("tbl").innerHTML = head + body;
}
const NAME2COHORT = {"纯心音队列": "heart_pure", "纯肺音队列": "lung_pure",
                     "混合队列": "mixed", "重复副本": "all"};
function bindDrill(){
  const el = ensure("c2");
  el.off("click");
  el.on("click", (pt) => {
    const k = NAME2COHORT[pt.name];
    if (k) { document.getElementById("fCohort").value = k; render();
             document.getElementById("ctx").textContent += "｜已下钻：" + pt.name; }
  });
}
function render(){
  const key = document.getElementById("fCohort").value + "|" + document.getElementById("fSide").value;
  const d = DATA[key];
  document.getElementById("ctx").textContent = d ? d.label : "该组合无数据";
  if(!d){return;}
  document.getElementById("kpis").innerHTML = Object.entries(d.kpi)
    .map(([k,v])=>`<div class="kpi"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
  ["c1","c2","c3","c4","c5","c6","c7","c8"].forEach(id=>{
    ensure(id).setOption(d[id], true);
    echarts.connect("board");
    ensure(id).group = "board";
  });
  bindDrill();
  renderTable(d.tbl);
}
window.addEventListener("resize", ()=>Object.values(charts).forEach(c=>c.resize()));
fill(fCohort, COHORTS); fill(fSide, SIDES);
const qs = new URLSearchParams(window.location.search);
if (qs.get("cohort")) fCohort.value = qs.get("cohort");
if (qs.get("side")) fSide.value = qs.get("side");
fCohort.onchange = render; fSide.onchange = render;
render();
</script></body></html>
"""


def main() -> None:
    D = load()
    data = build(D)
    ech = (Path(cfg.DASH) / "echarts.min.js").read_text(encoding="utf-8", errors="ignore")
    html = (HTML.replace("__ECHARTS__", ech)
                .replace("__DATA__", json.dumps(data, ensure_ascii=False))
                .replace("__COHORTS__", json.dumps(COHORTS, ensure_ascii=False))
                .replace("__SIDES__", json.dumps(SIDES, ensure_ascii=False))
                .replace("__SEED__", str(cfg.SEED)))
    out = cfg.DASH / "听诊数据资产经营大屏.html"
    out.write_text(html, encoding="utf-8")
    print("组合数:", len(data), "| HTML:", out.name, f"{out.stat().st_size/1048576:.2f} MB")
    print("组件: KPI 卡片区(7) + 图表 8 个 + 明细表 + 双筛选器 = 12 类组件")
    print("截图命令（PowerShell）：")
    print('  & "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe" --headless=new '
          '--disable-gpu --hide-scrollbars --window-size=1920,1160 '
          f'--screenshot="{cfg.FIG / "fig28_交互大屏_资产经营总览.png"}" "file:///{out}"'.replace("\\\\", "\\"))


if __name__ == "__main__":
    main()
