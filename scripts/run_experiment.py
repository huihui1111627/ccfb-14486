"""三个月仿真主实验入口。

用法:
    python3 -m scripts.run_experiment                # 完整三个月
    python3 -m scripts.run_experiment --smoke        # 快速冒烟(缩小视野)
    python3 -m scripts.run_experiment --smoke --days 6

产出:
    outputs/report.html              可视化报告
    outputs/metrics.json             全部数值指标
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bandwidth_scheduler import SLOTS_PER_DAY, SLOTS_PER_MONTH
from bandwidth_scheduler.billing import SiteConfig, total_monthly_cost
from bandwidth_scheduler.evaluation import evaluate_month, forecast_accuracy
from bandwidth_scheduler.offline import offline_optimal
from bandwidth_scheduler.policies import run_policy
from bandwidth_scheduler.report import render_report
from bandwidth_scheduler.simulation import WorldConfig, build_world

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "outputs")

MIN_QUALITY = 72.0          # 最差质量硬门槛
AVG_QUALITY_TARGET = 88.0   # 平均质量下限


def build_sites() -> list:
    """12 个机房:普通 95、带保底、阶梯价、包端口四类混合。

    容量按峰值需求约 70% 利用率设计,留出免费冲高与故障转移余量。
    """
    sites = []
    # 4 个普通 95 峰值,单价不同
    for i, (cap, price) in enumerate([
            (4000, 22.0), (3700, 24.5), (3400, 26.0), (3100, 28.0)]):
        sites.append(SiteConfig(f"IDC-{i:02d}", kind="p95",
                                capacity_mbps=cap, unit_price=price))
    # 4 个带保底(两个保底用不满、两个接近用满,由容量差异体现)
    sites.append(SiteConfig("IDC-04", kind="p95", capacity_mbps=4200,
                            unit_price=17.0, commit_mbps=2000, commit_price=15.0))
    sites.append(SiteConfig("IDC-05", kind="p95", capacity_mbps=4000,
                            unit_price=18.0, commit_mbps=1400, commit_price=16.0))
    sites.append(SiteConfig("IDC-06", kind="p95", capacity_mbps=2600,
                            unit_price=16.0, commit_mbps=1800, commit_price=14.5))
    sites.append(SiteConfig("IDC-07", kind="p95", capacity_mbps=2300,
                            unit_price=19.0, commit_mbps=900, commit_price=17.0))
    # 2 个阶梯价(量大单价降,整单价跳档 + 分段边际各一)
    sites.append(SiteConfig(
        "IDC-08", kind="p95", capacity_mbps=4500,
        commit_mbps=1100, commit_price=19.0,
        tiers=[(1700, 21.0), (2600, 18.5), (np.inf, 16.0)],
        tier_mode="whole"))
    sites.append(SiteConfig(
        "IDC-09", kind="p95", capacity_mbps=4200,
        commit_mbps=1100, commit_price=20.0,
        tiers=[(1100, 22.0), (2300, 19.0), (np.inf, 16.5)],
        tier_mode="marginal"))
    # 2 个包端口:固定费 + 物理上限,边际成本为 0(应优先吃满)
    sites.append(SiteConfig("IDC-10", kind="port", capacity_mbps=3700,
                            commit_mbps=3700, commit_price=14.0,
                            port_fee=3700 * 14.0))
    sites.append(SiteConfig("IDC-11", kind="port", capacity_mbps=2600,
                            commit_mbps=2600, commit_price=15.5,
                            port_fee=2600 * 15.5))
    return sites


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="小规模冒烟测试")
    ap.add_argument("--days", type=int, default=6, help="冒烟模式每月天数")
    ap.add_argument("--seed", type=int, default=20261122)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    if args.smoke:
        spm = args.days * SLOTS_PER_DAY
        cfg = WorldConfig(months=2, slots_per_month=spm, seed=args.seed)
    else:
        cfg = WorldConfig(months=3, seed=args.seed)
        spm = SLOTS_PER_MONTH
    # 免费点数 = 当月槽数的 5%(整月即 432 个点 = 36 小时)
    n_free = max(2, int(round(spm * 0.05)))

    world = build_world(cfg)
    sites = build_sites()
    print(f"仿真: {world.cfg.n_regions} 地区 / {len(sites)} 机房 / "
          f"{world.n_slots} 槽 ({world.n_slots*5/60:.0f} 小时)")
    print(f"总需求均值 {world.demand.sum(axis=0).mean():,.0f} Mbps, "
          f"峰值 {world.demand.sum(axis=0).max():,.0f} Mbps")
    total_cap = sum(c.capacity_mbps for c in sites)
    print(f"物理总容量 {total_cap:,.0f} Mbps，故障 {len(world.failures)} 次，"
          f"尖峰 {len(world.spike_log)} 次")

    # ------------------------------------------------------------------ #
    # 1) 三种在线策略跑全程(DNS 滞后 + 执行误差 + 故障)
    # ------------------------------------------------------------------ #
    runs = {}
    for policy in ("nearest", "even", "online"):
        ts = time.time()
        runs[policy] = run_policy(world, sites, policy,
                                  MIN_QUALITY, AVG_QUALITY_TARGET,
                                  n_free=n_free)
        print(f"  策略 {policy:8s} 完成，用时 {time.time()-ts:,.1f}s，"
              f"未承接 {runs[policy].unmet_total:,.1f} Mbps·槽")

    # ------------------------------------------------------------------ #
    # 2) 离线标杆:每月单独解,全知信息(故障槽已知、容量置零)
    # ------------------------------------------------------------------ #
    offline_loads = np.zeros_like(runs["online"].loads)
    offline_alloc = np.zeros_like(runs["online"].alloc)
    for m, (start, end) in enumerate(world.month_bounds):
        ts = time.time()
        res = offline_optimal(
            world.demand[:, start:end], sites, world.quality,
            MIN_QUALITY, AVG_QUALITY_TARGET,
            failure_mask=world.failure_mask[:, start:end],
            n_free=n_free, iters=25, verbose=False)
        offline_loads[:, start:end] = res.loads
        offline_alloc[:, :, start:end] = res.alloc
        bill = total_monthly_cost(res.loads, sites)
        print(f"  离线标杆 第{m+1}月 账单 {bill:,.0f} 元，"
              f"起点账单 {[f'{c:,.0f}' for c in res.start_costs]}，"
              f"用时 {time.time()-ts:,.1f}s")

    class OfflineRun:
        pass
    off = OfflineRun()
    off.loads = offline_loads
    off.alloc = offline_alloc
    total_demand = np.maximum(world.demand.sum(axis=0), 1e-9)
    off.per_slot_quality = np.einsum(
        "rst,rs->t", offline_alloc, world.quality) / total_demand
    runs["offline"] = off

    # ------------------------------------------------------------------ #
    # 3) 验收指标
    # ------------------------------------------------------------------ #
    reports = [evaluate_month(m, world, sites, runs,
                              MIN_QUALITY, AVG_QUALITY_TARGET)
               for m in range(cfg.months)]
    fc_rows = forecast_accuracy(world)

    metrics = {
        "config": {"min_quality": MIN_QUALITY,
                   "avg_quality_target": AVG_QUALITY_TARGET,
                   "slots_per_month": spm, "n_free": n_free,
                   "n_failures": len(world.failures),
                   "n_spikes": len(world.spike_log),
                   "elapsed_sec": time.time() - t0},
        "months": [],
    }
    for rpt in reports:
        m = {
            "month": rpt.month,
            "bills": rpt.bills,
            "gap_to_optimal": rpt.gap_to_optimal,
            "avg_quality": rpt.avg_quality,
            "worst_slot_quality": rpt.worst_slot_quality,
            "quality_violations": rpt.quality_violations,
            "viol_flow_ratio": rpt.viol_flow_ratio,
            "site_bills": {k: v.tolist() for k, v in rpt.site_bills.items()},
            "free_usage": {
                k: {"used": v.free_slots_used.tolist(),
                    "total": v.free_slots_total.tolist(),
                    "load_ratio": v.free_avg_load_ratio.tolist(),
                    "waste": v.free_waste_ratio.tolist()}
                for k, v in rpt.free_usage.items()},
            "spike_days": rpt.spike_day_stats,
        }
        metrics["months"].append(m)
    metrics["forecast"] = fc_rows
    metrics["failures"] = [f._asdict() for f in world.failures]

    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)

    render_report(os.path.join(OUT_DIR, "report.html"),
                  sites, reports, runs, world, fc_rows,
                  (MIN_QUALITY, AVG_QUALITY_TARGET))

    # ------------------------------------------------------------------ #
    # 4) 控制台摘要
    # ------------------------------------------------------------------ #
    print("\n==================== 验收摘要 ====================")
    header = f"{'月份':<6}{'策略':<12}{'账单(元)':>14}{'vs离线':>9}{'平均质量':>9}{'最差质量':>9}"
    print(header)
    for rpt in reports:
        for p in ("offline", "online", "nearest", "even"):
            print(f"第{rpt.month+1}月  {p:<12}{rpt.bills[p]:>14,.0f}"
                  f"{rpt.gap_to_optimal[p]*100:>8.2f}%"
                  f"{rpt.avg_quality[p]:>9.2f}{rpt.worst_slot_quality[p]:>9.2f}")
    print(f"\n质量门槛: 最差≥{MIN_QUALITY} 平均≥{AVG_QUALITY_TARGET}；"
          f"违规槽数(各策略): "
          + ", ".join(f"{p}={reports[-1].quality_violations[p]}"
                      for p in ("online", "nearest", "even")))
    print("不合格流量占比(严格口径): "
          + ", ".join(f"{p}={reports[-1].viol_flow_ratio[p]*100:.3f}%"
                      for p in ("offline", "online", "nearest", "even")))
    # 免费时段使用(在线策略,末月)
    fu = reports[-1].free_usage["online"]
    used = fu.free_slots_used.sum()
    total = fu.free_slots_total.sum()
    print(f"末月在线免费时段有效利用: {used:.0f}/{total:.0f} 槽, "
          f"平均浪费占比 {fu.free_waste_ratio.mean()*100:.1f}%")
    print(f"报告: {os.path.join(OUT_DIR, 'report.html')}")
    print(f"指标: {os.path.join(OUT_DIR, 'metrics.json')}")
    print(f"总用时 {time.time()-t0:,.1f}s")


if __name__ == "__main__":
    main()
