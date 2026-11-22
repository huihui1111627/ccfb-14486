"""故障 + 紧容量下的兜底与质量门槛测试。"""
import numpy as np

from bandwidth_scheduler.allocator import allocate, loads_from_alloc
from bandwidth_scheduler.billing import SiteConfig


def test_failure_fallback_minimizes_quality_violation():
    """大部分站点故障时,不丢流量,违约流量只走质量最高的非合格站点。"""
    r_n, s_n, t_n = 3, 4, 5
    demand = np.full((r_n, t_n), 100.0)
    quality = np.array([
        [95, 90, 70, 60],
        [92, 88, 68, 55],
        [85, 80, 75, 65]], dtype=float)
    # 前两个高质量站点容量很小,后两个大;门槛 72 把后两个判为非合格
    caps = np.array([
        [60, 60, 60, 60, 60],
        [60, 60, 60, 60, 60],
        [400, 400, 400, 400, 400],
        [400, 400, 400, 400, 400]], dtype=float)
    sites = [SiteConfig(f"S{s}", capacity_mbps=400, unit_price=20)
             for s in range(s_n)]
    res = allocate(demand, caps, quality,
                   min_quality=72.0, avg_quality_target=85.0,
                   site_cost=np.full(s_n, 20.0))
    # 所有需求都被承接
    np.testing.assert_allclose(res.alloc.sum(axis=1), demand, atol=1e-7)
    # 违约流量存在但被最小化:R0 的非合格站点中,应优先质量 70 的站点 2
    assert res.quality_violation > 0
    # R0 落到站点 2(质量 70)的量应不少于站点 3(质量 60)
    assert res.alloc[0, 2].sum() >= res.alloc[0, 3].sum() - 1e-7
    # 未承接为零
    assert res.unmet == 0.0


def test_no_violation_when_capacity_adequate():
    """合格容量充足时不应产生任何质量违约。"""
    r_n, s_n, t_n = 2, 4, 10
    rng = np.random.default_rng(5)
    demand = rng.uniform(50, 100, (r_n, t_n))
    quality = np.array([[95, 90, 60, 55], [92, 88, 58, 50]], dtype=float)
    caps = np.full((s_n, t_n), 500.0)
    res = allocate(demand, caps, quality, 72.0, 90.0,
                   site_cost=np.full(s_n, 20.0))
    assert res.quality_violation == 0.0
    # 低质量站点完全不承接
    assert res.alloc[:, 2:].sum() == 0.0
    loads = loads_from_alloc(res.alloc)
    assert (loads <= caps + 1e-9).all()
