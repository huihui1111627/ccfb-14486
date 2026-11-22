"""分配核心测试:与 scipy linprog 精确 LP 解对拍 + 质量约束验证。"""
import numpy as np
from scipy.optimize import linprog

from bandwidth_scheduler.allocator import allocate, loads_from_alloc


def _brute_lp(demand, caps, quality, min_q, avg_q, site_cost):
    """全局 LP 基准:需求逐槽等式、容量逐槽不等式、质量为全局加权平均约束。"""
    r_n, t_n = demand.shape
    s_n = quality.shape[1]
    n = r_n * s_n * t_n

    def idx(r, s, t):
        return (r * s_n + s) * t_n + t

    c = np.zeros(n)
    for r in range(r_n):
        for s in range(s_n):
            for t in range(t_n):
                c[idx(r, s, t)] = site_cost[s]
    A_eq, b_eq, A_ub, b_ub = [], [], [], []
    for r in range(r_n):
        for t in range(t_n):
            row = np.zeros(n)
            for s in range(s_n):
                row[idx(r, s, t)] = 1
            A_eq.append(row)
            b_eq.append(demand[r, t])
    for s in range(s_n):
        for t in range(t_n):
            row = np.zeros(n)
            for r in range(r_n):
                row[idx(r, s, t)] = 1
            A_ub.append(row)
            b_ub.append(caps[s, t])
    row_q = np.zeros(n)
    for r in range(r_n):
        for s in range(s_n):
            for t in range(t_n):
                row_q[idx(r, s, t)] = avg_q - quality[r, s]
    A_ub.append(row_q)
    b_ub.append(0.0)
    bounds = [(0, None)] * n
    for r in range(r_n):
        for s in range(s_n):
            if quality[r, s] < min_q:
                for t in range(t_n):
                    bounds[idx(r, s, t)] = (0, 0)
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=bounds, method="highs")
    assert res.success, res.message
    alloc = np.zeros((r_n, s_n, t_n))
    for r in range(r_n):
        for s in range(s_n):
            for t in range(t_n):
                alloc[r, s, t] = res.x[idx(r, s, t)]
    return alloc


def test_allocator_matches_lp():
    rng = np.random.default_rng(0)
    r_n, s_n, t_n = 4, 5, 7
    demand = rng.uniform(40, 120, size=(r_n, t_n))
    quality = rng.uniform(60, 100, size=(r_n, s_n))
    caps = rng.uniform(120, 260, size=(s_n, t_n))
    cost = rng.uniform(10, 30, size=s_n)
    min_q, avg_q = 62.0, 80.0

    lp = _brute_lp(demand, caps, quality, min_q, avg_q, cost)
    res = allocate(demand, caps, quality, min_q, avg_q, site_cost=cost)

    # 全局贪心相对精确 LP 的差距在 0.5% 以内(换取约两个数量级加速);
    # 固定 μ 时两者目标值差距为浮点舍入量级,差距主要来自质量约束顶点跳变。
    lp_cost = (lp * cost[None, :, None]).sum()
    my_cost = (res.alloc * cost[None, :, None]).sum()
    assert abs(lp_cost - my_cost) / max(lp_cost, 1) < 5e-3
    # 需求全部承接
    np.testing.assert_allclose(res.alloc.sum(axis=1), demand, rtol=1e-8, atol=1e-8)
    # 容量约束
    loads = loads_from_alloc(res.alloc)
    assert (loads <= caps + 1e-7).all()
    # 最差质量门槛:没有任何流量分到不合格站点
    for r in range(r_n):
        for s in range(s_n):
            if quality[r, s] < min_q:
                assert res.alloc[r, s].sum() == 0.0
    # 平均质量达标
    total = demand.sum()
    achieved = (res.alloc * quality[:, :, None]).sum() / total
    assert achieved >= avg_q - 1e-7


def test_global_greedy_matches_lp_at_fixed_mu():
    """固定 μ 下全局贪心与逐槽运输 LP 的目标值应在浮点舍入量级一致。"""
    rng = np.random.default_rng(1)
    r_n, s_n, t_n = 8, 12, 300
    demand = rng.uniform(40, 120, (r_n, t_n))
    caps = rng.uniform(300, 600, (s_n, t_n))
    quality = rng.uniform(60, 100, (r_n, s_n))
    cost = rng.uniform(10, 30, s_n)
    mu = 0.05
    eligible = quality >= 62
    from bandwidth_scheduler.allocator import _global_greedy
    adj = np.where(eligible, cost[None, :] - mu * quality, np.inf)
    alloc, unmet, _viol = _global_greedy(demand, caps, adj, eligible, quality)
    assert unmet == 0.0
    gr_obj = (alloc * (cost[None, :, None]
                       - mu * quality[:, :, None])).sum()

    lp_obj = 0.0
    for t in range(t_n):
        n = r_n * s_n
        c = np.where(eligible, cost[None, :] - mu * quality, 0.0).reshape(-1)
        Aeq, beq, Aub, bub = [], [], [], []
        for r in range(r_n):
            row = np.zeros(n); row[r * s_n:(r + 1) * s_n] = 1
            Aeq.append(row); beq.append(demand[r, t])
        for s in range(s_n):
            row = np.zeros(n)
            for r in range(r_n):
                row[r * s_n + s] = 1
            Aub.append(row); bub.append(caps[s, t])
        bd = [(0, None)] * n
        for r in range(r_n):
            for s in range(s_n):
                if not eligible[r, s]:
                    bd[r * s_n + s] = (0, 0)
        r = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub),
                    A_eq=np.array(Aeq), b_eq=np.array(beq),
                    bounds=bd, method="highs")
        a = r.x.reshape(r_n, s_n)
        lp_obj += (a * (cost[None, :] - mu * quality)).sum()
    assert abs(gr_obj - lp_obj) / abs(lp_obj) < 1e-4


def test_allocator_infeasible_reports_unmet():
    demand = np.ones((2, 3)) * 100
    caps = np.ones((4, 3)) * 10          # 总容量远小于需求
    quality = np.full((2, 4), 90.0)
    res = allocate(demand, caps, quality, 60.0, 80.0)
    assert res.unmet > 0
