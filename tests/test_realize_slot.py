"""单槽运输 LP 承接环节的质量门槛与容量约束测试。"""
import numpy as np

from bandwidth_scheduler.policies import _realize_slot


def test_realize_respects_quality_floor_and_capacity():
    # R1 的合格站点只有 0、1,且容量偏紧,验证不违约也不超容量
    quality = np.array([
        [95, 90, 70, 60],
        [92, 88, 68, 55]], dtype=float)
    demand = np.array([500.0, 400.0])
    phys = np.array([500.0, 500.0, 800.0, 800.0])
    available = np.ones(4, dtype=bool)
    share = np.array([0.4, 0.3, 0.2, 0.1])
    alloc = _realize_slot(demand, share, available, phys, quality, 72.0)
    # 需求全部承接
    np.testing.assert_allclose(alloc.sum(axis=1), demand, atol=1e-6)
    # 物理容量不超
    assert (alloc.sum(axis=0) <= phys + 1e-7).all()
    # 不合格对(站点 2、3 对两个地区都 <72)零分配
    assert alloc[:, 2:].sum() == 0.0


def test_realize_fallback_when_eligible_capacity_exhausted():
    # 合格站点容量不足以承接全部需求,验证不丢流量、违约走最高质量站点
    quality = np.array([[90.0, 60.0, 55.0]])
    demand = np.array([1000.0])
    phys = np.array([300.0, 800.0, 800.0])
    available = np.ones(3, dtype=bool)
    share = np.array([0.5, 0.3, 0.2])
    alloc = _realize_slot(demand, share, available, phys, quality, 72.0)
    # 全部承接
    np.testing.assert_allclose(alloc.sum(), 1000.0, atol=1e-6)
    # 合格站点先打满
    assert abs(alloc[0, 0] - 300.0) < 1e-6
    # 残余 700 走非合格站点中质量更高的站点 1
    assert alloc[0, 1] >= alloc[0, 2]


def test_realize_failure_zeroes_unavailable():
    quality = np.full((2, 3), 90.0)
    demand = np.array([200.0, 200.0])
    phys = np.array([500.0, 500.0, 500.0])
    available = np.array([False, True, True])
    share = np.array([0.5, 0.3, 0.2])
    alloc = _realize_slot(demand, share, available, phys, quality, 72.0)
    assert alloc[:, 0].sum() == 0.0
    np.testing.assert_allclose(alloc.sum(axis=1), demand, atol=1e-6)
