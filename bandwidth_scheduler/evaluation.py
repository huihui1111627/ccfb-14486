"""验收指标计算。

输出内容对应需求:
- 每月账单,以及与离线最优的差距(节省金额 / 节省比例)
- 与就近调度、平均分摊的账单对比
- 平均质量 / 最差质量是否守住下限
- 每个机房免费时段(最高 5% 采样点)的实际利用情况
- 预测偏差大的尖峰日当天的调度表现
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import P95_DROP_SAMPLES, SLOTS_PER_DAY
from .billing import (SiteConfig, free_slot_mask, p95_line,
                      site_monthly_cost)
from .simulation import World


@dataclass
class MonthReport:
    month: int
    bills: Dict[str, float]                    # 策略 -> 总账单
    site_bills: Dict[str, np.ndarray]          # 策略 -> (S,) 各机房账单
    gap_to_optimal: Dict[str, float]           # 策略 -> 相对离线超支比例
    avg_quality: Dict[str, float]
    worst_slot_quality: Dict[str, float]      # 全月最差单槽质量
    min_quality_floor: float
    avg_quality_floor: float
    quality_violations: Dict[str, int]         # 低于最差门槛的槽数
    viol_flow_ratio: Dict[str, float]          # 不合格流量占比(严格口径)
    free_usage: Dict[str, "FreeUsage"]
    spike_day_stats: List[dict]


@dataclass
class FreeUsage:
    site_names: List[str]
    free_slots_used: np.ndarray      # (S,) 免费槽中负载 >= 80% 计费线的点数
    free_slots_total: np.ndarray     # (S,) 免费槽总数(=432,端口机房除外)
    free_avg_load_ratio: np.ndarray  # (S,) 免费槽平均负载 / 当月峰值
    free_waste_ratio: np.ndarray     # (S,) 免费槽中 < 50% 计费线的占比(浪费)


def _free_usage(loads: np.ndarray, sites: Sequence[SiteConfig]) -> FreeUsage:
    n_s = loads.shape[0]
    used = np.zeros(n_s)
    total = np.zeros(n_s)
    ratio = np.zeros(n_s)
    waste = np.zeros(n_s)
    for s, cfg in enumerate(sites):
        if cfg.kind == "port":
            continue
        x = loads[s]
        mask = free_slot_mask(x)
        line = p95_line(x)
        peak = max(x.max(), 1e-9)
        total[s] = mask.sum()
        if total[s] > 0:
            used[s] = np.count_nonzero(x[mask] >= 0.8 * max(line, 1e-9))
            ratio[s] = x[mask].mean() / peak
            waste[s] = np.count_nonzero(x[mask] < 0.5 * max(line, 1e-9)) / total[s]
    return FreeUsage([c.name for c in sites], used, total, ratio, waste)


def evaluate_month(month: int,
                   world: World,
                   sites: Sequence[SiteConfig],
                   runs: Dict[str, "object"],
                   min_quality: float,
                   avg_quality_target: float) -> MonthReport:
    start, end = world.month_bounds[month]
    bills, site_bills, gaps = {}, {}, {}
    avg_q, worst_q, violations = {}, {}, {}
    viol_flow_ratio: Dict[str, float] = {}
    free: Dict[str, FreeUsage] = {}

    for name, run in runs.items():
        month_loads = run.loads[:, start:end]
        per_site = np.array([
            site_monthly_cost(month_loads[s], sites[s]) for s in range(len(sites))
        ])
        bills[name] = float(per_site.sum())
        site_bills[name] = per_site
        q = run.per_slot_quality[start:end]
        avg_q[name] = float(np.mean(q))
        worst_q[name] = float(np.min(q))
        violations[name] = int(np.count_nonzero(q < min_quality - 1e-9))
        # 严格口径:直接从 (地区,站点) 分配统计不合格流量占比,
        # 槽级平均会掩盖单地区被调到远机房
        month_alloc = run.alloc[:, :, start:end]
        eligible = world.quality >= min_quality
        bad_flow = float(sum(
            month_alloc[r, s].sum() for r in range(world.cfg.n_regions)
            for s in range(len(sites)) if not eligible[r, s]))
        total_flow = float(month_alloc.sum())
        viol_flow_ratio[name] = bad_flow / max(total_flow, 1e-9)
        free[name] = _free_usage(month_loads, sites)

    opt = bills.get("offline")
    for name, bill in bills.items():
        gaps[name] = 0.0 if opt is None or opt <= 0 else (bill - opt) / opt

    # ---- 尖峰日(预测偏差大的日子)表现 -------------------------------------
    # 95 计费以全月为口径,单日无法独立算账单;这里衡量尖峰当天的两个关键
    # 能力:峰值流量是否被引导进免费槽、当天各站点最高点相对当月计费线的压力
    spike_stats: List[dict] = []
    month_spikes = [s for s in world.spike_log
                    if start <= s["start"] < end]
    for sp in month_spikes:
        d0 = (sp["start"] // SLOTS_PER_DAY) * SLOTS_PER_DAY
        d1 = min(d0 + SLOTS_PER_DAY, end)
        row = {"day": sp["day"], "region": sp["region"],
               "mult": sp["mult"], "peak_in_free": {},
               "peak_over_line": {}}
        for name, run in runs.items():
            day_loads = run.loads[:, d0:d1]
            month_loads = run.loads[:, start:end]
            peak_slot = int(day_loads.sum(axis=0).argmax())
            inside_free, over_line = 0, 0
            for k, cfg in enumerate(sites):
                if cfg.kind == "port":
                    continue
                mask = free_slot_mask(month_loads[k])
                local = peak_slot - (d0 - start)
                if 0 <= local < mask.size and mask[local]:
                    inside_free += 1
                # 当天该站点最高点是否超过当月计费线(即抬线压力)
                line = p95_line(month_loads[k])
                if day_loads[k].max() > line + 1e-9:
                    over_line += 1
            row["peak_in_free"][name] = inside_free
            row["peak_over_line"][name] = over_line
        spike_stats.append(row)

    return MonthReport(
        month=month, bills=bills, site_bills=site_bills, gap_to_optimal=gaps,
        avg_quality=avg_q, worst_slot_quality=worst_q,
        min_quality_floor=min_quality, avg_quality_floor=avg_quality_target,
        quality_violations=violations, viol_flow_ratio=viol_flow_ratio,
        free_usage=free, spike_day_stats=spike_stats)


def forecast_accuracy(world: World) -> List[dict]:
    """用与在线预测相同的 EWMA 方法做事后预测误差分析(按天汇总,标准 MAPE)。

    每月用独立预测器并在首观测 seed,避免跨月状态串扰与冷启动 level=1。
    """
    from .forecast import DemandForecaster
    rows = []
    for m, (start, end) in enumerate(world.month_bounds):
        fc = DemandForecaster(world.cfg.n_regions,
                              slots_per_month=end - start)
        for d in range((end - start) // SLOTS_PER_DAY):
            errs = []
            abs_err = 0.0
            abs_tot = 0.0
            for tod in range(SLOTS_PER_DAY):
                t = start + d * SLOTS_PER_DAY + tod
                slot_in_month = t - start
                pred = fc.forecast_remaining(
                    slot_in_month).today[:, 0]
                actual = world.demand[:, t]
                if slot_in_month == 0 and tod == 0:
                    fc.seed_level(actual)
                errs.append(np.mean(
                    np.abs(actual - pred)
                    / np.maximum(np.abs(actual), 1e-6)))
                abs_err += float(np.abs(actual - pred).sum())
                abs_tot += float(actual.sum())
                fc.update(slot_in_month, actual, pred)
            # 日终更新 level,与在线调度器的 observe 路径保持一致
            day_obs = world.demand[:, start + d * SLOTS_PER_DAY:
                                  start + (d + 1) * SLOTS_PER_DAY]
            fc.end_of_day_update(day_obs)
            global_day = m * (world.cfg.slots_per_month // SLOTS_PER_DAY) + d
            rows.append({
                "day": global_day,
                "mape": float(np.mean(errs)),
                # 流量加权绝对百分比误差:尖峰日流量大,WMAPE 能如实反映
                # 尖峰带来的绝对偏差(MAPE 因分母大反而显小)
                "wmape": abs_err / max(abs_tot, 1e-9),
                "is_spike_day": any(s["day"] == global_day
                                    for s in world.spike_log),
                # 每月首日为冷启动(日内模板尚未建立),单独标记
                "cold_start": d == 0})
    return rows
