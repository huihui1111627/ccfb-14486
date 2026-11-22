"""故障触发立即重排与流量转移的端到端测试。"""
import numpy as np

from bandwidth_scheduler.simulation import (
    WorldConfig, build_world, DNSWeightController)
from bandwidth_scheduler.online import OnlineScheduler
from scripts.run_experiment import build_sites, MIN_QUALITY, AVG_QUALITY_TARGET


def test_failure_triggers_immediate_replan_and_reroutes():
    cfg = WorldConfig(months=1, slots_per_month=2 * 288, seed=7)
    world = build_world(cfg)
    sites = build_sites()
    n_free = max(2, int(round(2 * 288 * 0.05)))
    # 人为在槽 300 制造一次单站点故障
    fail_site = 3
    world.failure_mask[fail_site, 300:320] = True

    dns = DNSWeightController(len(sites), tau_slots=6)
    sch = OnlineScheduler(sites, world.quality, MIN_QUALITY,
                          AVG_QUALITY_TARGET, dns_k=dns.k, n_free=n_free)
    replan_slots = []
    original_replan = sch._replan

    def spy(t, forecast, fail):
        replan_slots.append(t)
        return original_replan(t, forecast, fail)

    sch._replan = spy
    for t in range(2 * 288):
        d = world.demand[:, t]
        avail = ~world.failure_mask[:, t]
        fm = world.failure_mask[:, :2 * 288]
        target = sch.act(t, d, avail, fm, dns.share)
        # 故障期间目标权重必须为 0
        if 300 <= t < 320:
            assert target[fail_site] == 0.0
        share = dns.step(target, avail)
        load = share * d.sum()
        sch.observe(t, d, load, None)

    # 故障起点槽 300 应触发一次重排
    assert 300 in replan_slots
    # 每日重排 + 故障重排都有记录
    assert len(replan_slots) >= 2
