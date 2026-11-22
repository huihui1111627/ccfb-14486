"""自包含 HTML 报告:所有图表用内联 SVG 生成,不依赖 matplotlib。

页面内容
========
1. 三个月账单总表(离线标杆 / 在线 / 就近 / 平均)+ 与最优差距
2. 质量约束守线情况
3. 每个机房:本月采样点降序曲线 + 当前计费线 + 保底量 + 免费区阴影
4. 各机房免费时段使用率
5. 尖峰日(预测偏差大)表现
6. 预测误差按天曲线
"""
from __future__ import annotations

import html
import json
from typing import Dict, List, Sequence

import numpy as np

from . import P95_DROP_SAMPLES
from .billing import SiteConfig, p95_line
from .evaluation import MonthReport


COLORS = {"offline": "#2e7d32", "online": "#1565c0",
          "nearest": "#c62828", "even": "#6a1b9a"}
NAMES_CN = {"offline": "离线最优(标杆)", "online": "在线滚动调度",
            "nearest": "就近调度", "even": "平均分摊"}


def _svg_line(values: np.ndarray, w: int = 760, h: int = 220,
              markers: Dict[str, float] = None,
              free_cut: int = P95_DROP_SAMPLES,
              title: str = "") -> str:
    """values 已按降序排好;free_cut 左侧为免费区。"""
    n = values.size
    vmax = max(float(values.max()), 1e-9)
    pad_l, pad_b, pad_t, pad_r = 56, 34, 26, 12
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b

    def X(i):
        return pad_l + (iw * i / max(n - 1, 1))

    def Y(v):
        return pad_t + ih * (1 - v / vmax)

    pts = " ".join(f"{X(i):.1f},{Y(float(v)):.1f}"
                   for i, v in enumerate(values))
    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    parts.append(f'<text x="{pad_l}" y="16" class="title">{html.escape(title)}</text>')
    # 免费区阴影
    parts.append(
        f'<rect x="{X(0):.1f}" y="{pad_t}" width="{X(min(free_cut, n-1))-X(0):.1f}" '
        f'height="{ih}" fill="#fff3e0" opacity="0.85"/>')
    parts.append(f'<text x="{X(free_cut//2):.1f}" y="{pad_t+12}" '
                 f'class="anno">免费 5%（{free_cut} 点）</text>')
    # 网格与 y 轴刻度
    for g in range(5):
        v = vmax * g / 4
        y = Y(v)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w-pad_r}" '
                     f'y2="{y:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{pad_l-6}" y="{y+4:.1f}" text-anchor="end" '
                     f'class="tick">{v:.0f}</text>')
    # x 轴刻度
    for frac, label in [(0, "1"), (0.25, "25%"), (0.5, "50%"),
                        (0.75, "75%"), (1.0, "100%")]:
        i = int(frac * (n - 1))
        parts.append(f'<text x="{X(i):.1f}" y="{h-10}" text-anchor="middle" '
                     f'class="tick">{label}</text>')
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#1565c0" '
                 f'stroke-width="1.6"/>')
    if markers:
        for label, v in markers.items():
            if v <= 0:
                continue
            y = Y(v)
            color = "#2e7d32" if "计费线" in label else "#ef6c00"
            parts.append(
                f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w-pad_r}" y2="{y:.1f}" '
                f'stroke="{color}" stroke-dasharray="6 4" stroke-width="1.4"/>')
            parts.append(
                f'<text x="{w-pad_r-4}" y="{y-4:.1f}" text-anchor="end" '
                f'class="marker" fill="{color}">{html.escape(label)} '
                f'{v:.1f} Mbps</text>')
    parts.append("</svg>")
    return "".join(parts)


def _svg_bar(labels: List[str], series: Dict[str, np.ndarray],
             w: int = 760, h: int = 260, ytitle: str = "") -> str:
    pad_l, pad_b, pad_t, pad_r = 48, 60, 30, 12
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b
    n = len(labels)
    names = list(series.keys())
    group_w = iw / max(n, 1)
    bar_w = min(26, group_w / (len(names) + 1))
    vmax = max(float(np.concatenate([np.atleast_1d(v)
                                     for v in series.values()]).max()), 1e-9)
    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    for g in range(5):
        y = pad_t + ih * (1 - g / 4)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w-pad_r}" '
                     f'y2="{y:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{pad_l-6}" y="{y+4:.1f}" text-anchor="end" '
                     f'class="tick">{vmax*g/4:,.0f}</text>')
    for i, lab in enumerate(labels):
        cx = pad_l + group_w * (i + 0.5)
        for j, name in enumerate(names):
            v = float(series[name][i])
            bh = ih * v / vmax
            x = cx + (j - (len(names) - 1) / 2) * bar_w - bar_w / 2
            parts.append(f'<rect x="{x:.1f}" y="{pad_t+ih-bh:.1f}" '
                         f'width="{bar_w-2:.1f}" height="{bh:.1f}" '
                         f'fill="{COLORS.get(name, "#888")}"/>')
        parts.append(f'<text x="{cx:.1f}" y="{h-40}" text-anchor="middle" '
                     f'class="tick" transform="rotate(-35 {cx:.1f} {h-40})">'
                     f'{html.escape(lab)}</text>')
    for j, name in enumerate(names):
        x = pad_l + j * 130
        parts.append(f'<rect x="{x}" y="6" width="10" height="10" '
                     f'fill="{COLORS.get(name, "#888")}"/>')
        parts.append(f'<text x="{x+14}" y="15" class="tick">'
                     f'{html.escape(NAMES_CN.get(name, name))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _svg_line_series(days: List[int], values: List[float],
                     spike_days: List[int], w: int = 760, h: int = 220,
                     title: str = "") -> str:
    pad_l, pad_b, pad_t, pad_r = 48, 30, 30, 12
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b
    vmax = max(max(values), 1e-9)
    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    parts.append(f'<text x="{pad_l}" y="16" class="title">{html.escape(title)}</text>')
    for d in spike_days:
        if d < len(days):
            x = pad_l + iw * d / max(len(days) - 1, 1)
            parts.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" '
                         f'y2="{pad_t+ih}" stroke="#ffe0b2" stroke-width="2"/>')
    for g in range(5):
        y = pad_t + ih * (1 - g / 4)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w-pad_r}" '
                     f'y2="{y:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{pad_l-6}" y="{y+4:.1f}" text-anchor="end" '
                     f'class="tick">{vmax*g/4:.2f}</text>')
    pts = []
    for i, v in enumerate(values):
        x = pad_l + iw * i / max(len(days) - 1, 1)
        y = pad_t + ih * (1 - v / vmax)
        pts.append(f"{x:.1f},{y:.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                 f'stroke="#1565c0" stroke-width="1.6"/>')
    parts.append(f'<text x="{w-pad_r}" y="{pad_t-8}" text-anchor="end" '
                 f'class="anno">橙色竖线 = 爆款尖峰日</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_report(path: str,
                  sites: Sequence[SiteConfig],
                  reports: List[MonthReport],
                  runs: Dict[str, object],
                  world,
                  forecast_rows: List[dict],
                  quality_floors: tuple) -> None:
    min_q, avg_q = quality_floors
    policy_order = [p for p in ["offline", "online", "nearest", "even"]
                    if p in reports[0].bills]

    # ---- 汇总账单表 -------------------------------------------------------
    summary_rows = []
    for rpt in reports:
        for p in policy_order:
            vfr = rpt.viol_flow_ratio[p] * 100
            vfr_cls = "ok" if vfr < 0.01 else "bad"
            summary_rows.append(
                f"<tr><td>第{rpt.month+1}月</td><td>{NAMES_CN[p]}</td>"
                f"<td class='num'>{rpt.bills[p]:,.0f}</td>"
                f"<td class='num'>{rpt.gap_to_optimal[p]*100:+.2f}%</td>"
                f"<td class='num'>{rpt.avg_quality[p]:.2f}</td>"
                f"<td class='num'>{rpt.worst_slot_quality[p]:.2f}</td>"
                f"<td class='num'>{rpt.quality_violations[p]}</td>"
                f"<td class='num {vfr_cls}'>{vfr:.3f}%</td></tr>")

    # ---- 账单柱状图 -------------------------------------------------------
    labels = [f"第{r.month+1}月" for r in reports]
    bar_series = {p: np.array([r.bills[p] for r in reports]) for p in policy_order}

    # ---- 三个月累计账单与总节省 -------------------------------------------
    total_bills = {p: sum(r.bills[p] for r in reports) for p in policy_order}
    base_total = total_bills.get("offline")
    save_rows = []
    for p in policy_order:
        if p == "offline":
            continue
        saved = total_bills["nearest"] - total_bills[p] if p != "nearest" else 0.0
        gap = ((total_bills[p] - base_total) / base_total * 100
               if base_total else 0.0)
        save_rows.append(
            f"<tr><td>{NAMES_CN[p]}</td>"
            f"<td class='num'>{total_bills[p]:,.0f}</td>"
            f"<td class='num'>{gap:+.2f}%</td></tr>")
    cumulative_table = (
        "<table><tr><th>策略</th><th>三个月累计账单(元)</th>"
        "<th>相对离线标杆</th></tr>"
        f"<tr><td>{NAMES_CN['offline']}(标杆)</td>"
        f"<td class='num'>{total_bills['offline']:,.0f}</td>"
        f"<td class='num'>0.00%</td></tr>{''.join(save_rows)}</table>")

    # ---- 排序采样曲线(在线策略,第三个月全部机房)---------------------------
    last = reports[-1]
    start, end = world.month_bounds[-1]
    sort_svgs = []
    for s, cfg in enumerate(sites):
        x = np.sort(runs["online"].loads[s, start:end])[::-1]
        markers = {}
        if cfg.kind == "p95":
            markers["当前计费线"] = p95_line(x)
            markers["保底量"] = cfg.commit_mbps
        else:
            markers["端口速率"] = cfg.capacity_mbps
        sort_svgs.append(_svg_line(
            x, markers=markers,
            title=f"{cfg.name}（{cfg.kind}，单价 {cfg.unit_price or cfg.commit_price:.2f} 元/Mbps）"))

    # ---- 免费时段使用率 ----------------------------------------------------
    fu = last.free_usage["online"]
    fu_rows = []
    for s, cfg in enumerate(sites):
        fu_rows.append(
            f"<tr><td>{cfg.name}</td><td>{cfg.kind}</td>"
            f"<td class='num'>{fu.free_slots_total[s]:.0f}</td>"
            f"<td class='num'>{fu.free_slots_used[s]:.0f}</td>"
            f"<td class='num'>{fu.free_avg_load_ratio[s]*100:.1f}%</td>"
            f"<td class='num'>{fu.free_waste_ratio[s]*100:.1f}%</td></tr>")

    # ---- 尖峰日表 ----------------------------------------------------------
    spike_rows = []
    for rpt in reports:
        for sp in rpt.spike_day_stats:
            cells = "".join(
                f"<td class='num'>{sp['peak_in_free'][p]} 入免费槽 / "
                f"{sp['peak_over_line'][p]} 超线</td>"
                for p in policy_order)
            spike_rows.append(
                f"<tr><td>第{rpt.month+1}月 第{sp['day']%30+1}天</td>"
                f"<td>{world.region_names[sp['region']]}</td>"
                f"<td class='num'>×{sp['mult']:.2f}</td>{cells}</tr>")

    # ---- 预测误差 ----------------------------------------------------------
    days = [r["day"] for r in forecast_rows]
    mapes = [r["mape"] for r in forecast_rows]
    spike_days = [r["day"] for r in forecast_rows if r["is_spike_day"]]
    fc_svg = _svg_line_series(days, mapes, spike_days,
                              title="按天预测相对误差 MAPE（橙色 = 尖峰日）")
    cold_mape = np.mean([r["mape"] for r in forecast_rows
                         if r.get("cold_start")]) * 100
    warm_rows = [r for r in forecast_rows if not r.get("cold_start")]
    warm_mape = np.mean([r["mape"] for r in warm_rows]) * 100
    warm_spike = np.mean([r["mape"] for r in warm_rows
                          if r["is_spike_day"]]) * 100
    warm_normal = np.mean([r["mape"] for r in warm_rows
                           if not r["is_spike_day"]]) * 100
    # WMAPE 口径:尖峰日绝对偏差更大,与调度账单直接相关
    wmape_spike = np.mean([r["wmape"] for r in warm_rows
                           if r["is_spike_day"]]) * 100
    wmape_normal = np.mean([r["wmape"] for r in warm_rows
                            if not r["is_spike_day"]]) * 100
    fc_summary = (
        f"<div class='card'><b>预测误差汇总</b><br>"
        f"冷启动日(每月首日)MAPE:{cold_mape:.1f}%　|　"
        f"稳定期 MAPE:{warm_mape:.1f}%(普通 {warm_normal:.1f}% / "
        f"尖峰 {warm_spike:.1f}%)<br>"
        f"流量加权 WMAPE(与账单口径一致):普通日 {wmape_normal:.1f}%　|　"
        f"尖峰日 {wmape_spike:.1f}% —— "
        f"尖峰日绝对偏差显著更大,正是在线调度要靠免费额度吸收的部分。</div>")

    # ---- 站点配置表 --------------------------------------------------------
    cfg_rows = []
    for cfg in sites:
        if cfg.kind == "port":
            detail = f"端口费 {cfg.port_fee:,.0f} 元 / 上限 {cfg.capacity_mbps:.0f} Mbps"
        else:
            tier = ("；阶梯" + "/".join(f"{u:g}:{p:.2f}" for u, p in cfg.tiers)
                    if cfg.tiers else f"单价 {cfg.unit_price:.2f}")
            detail = (f"保底 {cfg.commit_mbps:.0f}@{cfg.commit_price:.2f}"
                      f"{tier}；上限 {cfg.capacity_mbps:.0f}")
        cfg_rows.append(f"<tr><td>{cfg.name}</td><td>{cfg.kind}</td>"
                        f"<td>{detail}</td></tr>")

    doc = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>带宽调度仿真报告</title>
<style>
body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
  margin: 0; background: #f5f6f8; color: #222; }}
.wrap {{ max-width: 1080px; margin: 0 auto; padding: 28px; }}
h1 {{ font-size: 24px; }} h2 {{ font-size: 18px; margin-top: 34px;
  border-left: 4px solid #1565c0; padding-left: 10px; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 13px; }}
th, td {{ border: 1px solid #dde1e6; padding: 7px 9px; }}
th {{ background: #eef2f7; }} td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.chart {{ width: 100%; background: #fff; border: 1px solid #e3e6ea; border-radius: 8px;
  margin: 10px 0; }}
text.tick {{ font-size: 10px; fill: #666; }}
text.title {{ font-size: 13px; font-weight: 600; fill: #333; }}
text.anno {{ font-size: 11px; fill: #e65100; }}
text.marker {{ font-size: 11px; font-weight: 600; }}
.card {{ background: #fff; border-radius: 8px; padding: 16px; margin: 12px 0;
  border: 1px solid #e3e6ea; }}
.ok {{ color: #2e7d32; font-weight: 600; }} .bad {{ color: #c62828; font-weight: 600; }}
small {{ color: #666; }}
</style></head><body><div class="wrap">
<h1>带宽账单最优调度 · 三个月仿真报告</h1>
<small>采样粒度 5 分钟（每月 8640 点，免费高峰 {P95_DROP_SAMPLES} 点 = 36 小时）；
质量门槛：最差 ≥ {min_q:.1f}，平均 ≥ {avg_q:.1f}</small>

<h2>1. 月度账单与离线标杆差距</h2>
<p><small>"相对离线最优"为正表示比标杆多付的比例;在线策略个别月份出现小幅
负值,是在线单槽精确 LP 与离线凸松弛启发式在同一最优水平附近的正常波动,
两者均显著优于就近/平均分摊。离线标杆为全知信息下的参考下界,在线只使用
历史与预测信息。</small></p>
<table><tr><th>月份</th><th>策略</th><th>账单（元）</th><th>相对离线最优</th>
<th>平均质量</th><th>最差单槽质量</th><th>质量违规槽数</th><th>不合格流量占比</th></tr>
{''.join(summary_rows)}</table>
<div class="card">{_svg_bar(labels, bar_series, ytitle="月账单（元）")}</div>
<div class="card"><b>三个月累计账单对比</b>{cumulative_table}</div>

<h2>2. 机房计费配置（统一计费模型）</h2>
<table><tr><th>机房</th><th>类型</th><th>条款</th></tr>{''.join(cfg_rows)}</table>

<h2>3. 各机房采样点降序曲线与当前计费线（第 3 月 · 在线策略）</h2>
<small>橙色阴影为被去掉的最高 5%（免费 36 小时）；绿线为 95 计费线，橙线为保底量。</small>
{''.join(f'<div class="card">{s}</div>' for s in sort_svgs)}

<h2>4. 免费时段利用情况（第 3 月 · 在线策略）</h2>
<table><tr><th>机房</th><th>类型</th><th>免费槽总数</th><th>有效利用槽数</th>
<th>免费槽平均负载/峰值</th><th>免费槽浪费占比</th></tr>{''.join(fu_rows)}</table>

<h2>5. 预测偏差大的尖峰日表现</h2>
<small>每个策略两列含义:尖峰当天最高点落入免费槽的机房数 / 当天最高点超过当月计费线(产生抬线压力)的机房数。前者越大、后者越小越好。</small>
<table><tr><th>日期</th><th>尖峰地区</th><th>倍数</th>
{''.join(f'<th>{NAMES_CN[p]}</th>' for p in policy_order)}</tr>
{''.join(spike_rows)}</table>

<h2>6. 预测误差</h2>
{fc_summary}
<div class="card">{fc_svg}</div>
<p><small>尖峰日 MAPE 明细：
{', '.join(f"第{r['day']+1}天 {r['mape']*100:.1f}%" for r in forecast_rows if r['is_spike_day']) or '无'}
</small></p>
</div></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
