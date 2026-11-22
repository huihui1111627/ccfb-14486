"""计费模型正确性测试。"""
import numpy as np

from bandwidth_scheduler import P95_DROP_SAMPLES, SLOTS_PER_MONTH
from bandwidth_scheduler.billing import (SiteConfig, billed_quantity,
                                         current_line_and_free_budget,
                                         free_slot_mask,
                                         marginal_tier_cost, p95_line,
                                         site_monthly_cost)


def test_p95_drops_top_5_percent():
    x = np.arange(1, SLOTS_PER_MONTH + 1, dtype=float)  # 1..8640 升序
    # 降序去掉 432 个,剩下最大 = 8640 - 432 = 8208
    assert abs(p95_line(x) - (SLOTS_PER_MONTH - P95_DROP_SAMPLES)) < 1e-9
    assert P95_DROP_SAMPLES == 432
    assert P95_DROP_SAMPLES * 5 / 60 == 36.0  # 正好 36 小时


def test_p95_flat_then_spike_is_free():
    x = np.full(SLOTS_PER_MONTH, 100.0)
    x[:100] = 500.0                     # 100 个尖峰,远少于 432
    assert abs(p95_line(x) - 100.0) < 1e-9
    mask = free_slot_mask(x)
    assert mask.sum() == 100            # 尖峰全部落在免费区


def test_commit_floor_applies():
    cfg = SiteConfig("a", commit_mbps=400, commit_price=10, unit_price=20)
    x = np.full(SLOTS_PER_MONTH, 100.0)
    assert billed_quantity(x, cfg) == 400
    # 用不满也要付保底费
    assert abs(site_monthly_cost(x, cfg) - 400 * 10) < 1e-6


def test_commit_plus_overflow():
    cfg = SiteConfig("a", commit_mbps=400, commit_price=10, unit_price=20)
    x = np.full(SLOTS_PER_MONTH, 600.0)
    # p95=600 > 保底 400:保底 4000 + 超出 200*20 = 8000
    assert abs(site_monthly_cost(x, cfg) - (4000 + 4000)) < 1e-6


def test_marginal_tiers():
    tiers = [(400, 22.0), (800, 19.0), (np.inf, 16.5)]
    assert marginal_tier_cost(0, tiers) == 0
    assert abs(marginal_tier_cost(500, tiers) - (400 * 22 + 100 * 19)) < 1e-6
    assert abs(marginal_tier_cost(1000, tiers)
               - (400 * 22 + 400 * 19 + 200 * 16.5)) < 1e-6


def test_whole_tier_and_port():
    cfg = SiteConfig("t", commit_mbps=400, commit_price=19,
                     tiers=[(600, 21.0), (900, 18.5), (np.inf, 16.0)],
                     tier_mode="whole")
    x = np.full(SLOTS_PER_MONTH, 1000.0)
    # 整单价:1000 落第三档 16;总费用 16000,保底已付 7600,补 8400
    assert abs(site_monthly_cost(x, cfg) - 16000) < 1e-6

    port = SiteConfig("p", kind="port", capacity_mbps=900,
                      port_fee=13950)
    y = np.full(SLOTS_PER_MONTH, 880.0)
    assert site_monthly_cost(y, port) == 13950
    assert billed_quantity(y, port) == 0.0


def test_online_free_budget_tracking():
    samples = list(np.full(1000, 100.0)) + list(np.full(50, 400.0))
    line, left = current_line_and_free_budget(np.array(samples))
    assert line == 100.0
    assert left == 432 - 50
