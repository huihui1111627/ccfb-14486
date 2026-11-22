"""离线启发式与 MILP 精确解对拍(小规模实例)。

用 scipy.optimize.milp 把 95 峰值直接建成混合整数规划:
每个站点选 n_free 个免费槽(二进制 z),非免费槽负载 ≤ 计费线 L,
最小化真实账单。规模刻意压到 2~3 站点、几十槽,用于标定离线启发式的
次优差距(预期 <2%)。这是"离线标杆本身有多接近最优"的质量门。
"""
import numpy as np
import pytest
from scipy.optimize import milp, LinearConstraint, Bounds

from bandwidth_scheduler.billing import SiteConfig, total_monthly_cost
from bandwidth_scheduler.offline import offline_optimal

scipy_milp = pytest.importorskip("scipy.optimize").milp


def _exact_milp(demand, sites, quality, min_q, avg_q, n_free):
    """小规模精确 MILP。变量: a[r,s,t] 连续 + z[s,t] 二进制 + L[s] 连续。"""
    r_n, t_n = demand.shape
    s_n = len(sites)
    n_a = r_n * s_n * t_n
    n_z = s_n * t_n
    n_l = s_n
    n = n_a + n_z + n_l

    def ia(r, s, t):
        return (r * s_n + s) * t_n + t

    def iz(s, t):
        return n_a + s * t_n + t

    def il(s):
        return n_a + n_z + s

    phys = np.array([c.capacity_mbps for c in sites])
    cvec = np.zeros(n)
    # 目标:用线性近似表达 p95 账单——对无阶梯站点 L[s]*unit_price + 保底;
    # 这里只取简单单价站点做对拍,规避阶梯非凸
    for s, cfg in enumerate(sites):
        cvec[il(s)] = cfg.unit_price

    cons = []
    # 需求等式
    for r in range(r_n):
        for t in range(t_n):
            row = np.zeros(n)
            for s in range(s_n):
                row[ia(r, s, t)] = 1
            cons.append(LinearConstraint(row, demand[r, t], demand[r, t]))
    # 站点负载 = 地区汇聚
    for s in range(s_n):
        for t in range(t_n):
            row = np.zeros(n)
            for r in range(r_n):
                row[ia(r, s, t)] = -1
            row[iz(s, t)] = 0  # placeholder
            # load[s,t] <= L[s] + phys*z[s,t]
            row2 = np.zeros(n)
            for r in range(r_n):
                row2[ia(r, s, t)] = 1
            row2[il(s)] = -1
            row2[iz(s, t)] = -phys[s]
            cons.append(LinearConstraint(row2, -np.inf, 0))
    # 每站点免费槽数量 = n_free
    for s in range(s_n):
        row = np.zeros(n)
        for t in range(t_n):
            row[iz(s, t)] = 1
        cons.append(LinearConstraint(row, n_free, n_free))
    # 物理上限
    for s in range(s_n):
        for t in range(t_n):
            row = np.zeros(n)
            for r in range(r_n):
                row[ia(r, s, t)] = 1
            cons.append(LinearConstraint(row, 0, phys[s]))
    # 平均质量
    row_q = np.zeros(n)
    total = demand.sum()
    for r in range(r_n):
        for s in range(s_n):
            for t in range(t_n):
                row_q[ia(r, s, t)] = (avg_q - quality[r, s]) / total * total
    cons.append(LinearConstraint(row_q, -np.inf, 0))

    integrality = np.zeros(n)
    integrality[n_a:n_a + n_z] = 1
    lb = np.zeros(n)
    ub = np.full(n, np.inf)
    for r in range(r_n):
        for s in range(s_n):
            if quality[r, s] < min_q:
                for t in range(t_n):
                    ub[ia(r, s, t)] = 0
    res = milp(c=cvec, constraints=cons,
               integrality=integrality, bounds=Bounds(lb, ub))
    assert res.success, res.message
    loads = np.zeros((s_n, t_n))
    for s in range(s_n):
        for t in range(t_n):
            loads[s, t] = sum(res.x[ia(r, s, t)] for r in range(r_n))
    return total_monthly_cost(loads, sites)


def test_offline_heuristic_close_to_milp():
    rng = np.random.default_rng(3)
    r_n, s_n, t_n = 3, 2, 60
    demand = rng.uniform(80, 200, (r_n, t_n))
    quality = np.array([[95, 70], [88, 75], [72, 92]], dtype=float)
    sites = [
        SiteConfig("A", capacity_mbps=450, unit_price=22.0),
        SiteConfig("B", capacity_mbps=450, unit_price=16.0,
                   commit_mbps=120, commit_price=15.0),
    ]
    n_free = max(1, int(round(t_n * 0.05)))
    heur = offline_optimal(demand, sites, quality,
                           min_quality=68.0, avg_quality_target=82.0,
                           n_free=n_free, iters=20)
    exact = _exact_milp(demand, sites, quality, 68.0, 82.0, n_free)
    # 启发式账单不应低于 MILP(它是下界方向),且差距控制在 5% 内
    assert heur.cost >= exact * 0.999
    assert (heur.cost - exact) / max(exact, 1) < 0.05
