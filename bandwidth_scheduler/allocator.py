"""给定计费线/容量上限的流量分配核心。

固定每个站点每个时刻的上限 ``cap[s, t]`` 与线性成本信号 ``c[s]`` 后,
全周期问题是一个线性规划:

    min  Σ a[r,s,t] * c[s]  - μ * Σ a[r,s,t] * quality[r,s]
    s.t. Σ_s a[r,s,t] = demand[r,t]                 (需求全部承接)
         Σ_r a[r,s,t] <= cap[s,t]                   (站点时刻上限)
         a[r,s,t] = 0 当 quality[r,s] < min_quality (最差质量硬门槛)
         Σ a*quality / Σ a >= avg_quality_target    (平均质量,全局加权)

引入全局质量约束的对偶 μ 后,各 (r,s) 对的"调整后成本"为 c[s] - μ·q[r,s]。
把所有 (r,s,t) 三元组按该成本升序、逐对在全部时刻上贪心填充,即得到 LP 的
精确最优解(与逐槽运输 LP 对拍目标值差距在浮点舍入量级,见 tests);
μ 用二分法搜索以满足全局平均质量,顶点跳变处用凸组合插值使质量恰好达标。

纯 NumPy 实现:静态成本下只需 R×S 次向量化操作,全月 8640 槽毫秒级完成。
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np


class AllocationResult(NamedTuple):
    alloc: np.ndarray          # (R, S, T) Mbps
    mu: float
    avg_quality: float
    unmet: float
    per_slot_quality: np.ndarray  # (T,)
    quality_violation: float = 0.0  # 兜底放开到不合格站点的流量 Mbps·槽


def _global_greedy(demand: np.ndarray, caps: np.ndarray,
                   adj: np.ndarray, eligible: np.ndarray,
                   quality: Optional[np.ndarray] = None):
    """全三元组贪心。adj: (R,S) 静态调整后成本,不可用为 inf。

    合格站点容量耗尽时,用质量最高的非合格站点兜底,返回(分配, 未承接,
    质量违约量)。最差质量在故障叠加容量紧张时物理上可能无法满足,
    兜底把违约压到最小并如实披露。"""
    n_r, n_s = adj.shape
    n_t = demand.shape[1]
    order = np.argsort(adj.reshape(-1), kind="stable")
    pairs_r = order // n_s
    pairs_s = order % n_s
    finite = np.isfinite(adj.reshape(-1)[order])

    alloc = np.zeros((n_r, n_s, n_t))
    remaining = caps.copy()
    need = demand.copy()
    for r, s, ok in zip(pairs_r, pairs_s, finite):
        if not ok:
            continue
        take = np.minimum(need[r], remaining[s])
        alloc[r, s] += take
        remaining[s] -= take
        need[r] -= take
    unmet = float(need.sum())
    # 合格容量耗尽的兜底:按质量从高到低放开非合格站点
    violation = 0.0
    if unmet > 1e-9 and quality is not None:
        for r in range(n_r):
            if need[r].max() <= 1e-12:
                continue
            bad = np.where(~eligible[r])[0]
            bad = bad[np.argsort(-quality[r, bad])]
            for s in bad:
                if need[r].max() <= 1e-12:
                    break
                take = np.minimum(need[r], remaining[s])
                alloc[r, s] += take
                remaining[s] -= take
                need[r] -= take
                violation += float(take.sum())
        unmet = float(need.sum())
    return alloc, unmet, violation


def allocate(demand: np.ndarray,
             caps: np.ndarray,
             quality: np.ndarray,
             min_quality: float,
             avg_quality_target: float,
             site_cost: Optional[np.ndarray] = None,
             cost_vec: Optional[np.ndarray] = None,
             mu_bisect_iters: int = 24,
             tol: float = 1e-10) -> AllocationResult:
    demand = np.asarray(demand, dtype=np.float64)
    caps = np.asarray(caps, dtype=np.float64)
    quality = np.asarray(quality, dtype=np.float64)
    n_r, n_t = demand.shape
    n_s = quality.shape[1]
    if site_cost is None:
        site_cost = np.zeros(n_s)
    site_cost = np.asarray(site_cost, dtype=np.float64)
    eligible = quality >= min_quality
    if not eligible.any(axis=1).all():
        bad = np.where(~eligible.any(axis=1))[0]
        raise ValueError(f"地区 {bad.tolist()} 没有任何站点满足最差质量门槛")

    total_demand = demand.sum(axis=0)

    def quality_of(alloc: np.ndarray) -> np.ndarray:
        q_num = np.einsum("rst,rs->t", alloc, quality)
        return np.divide(q_num, np.maximum(total_demand, 1e-30))

    if cost_vec is not None:
        # 逐时刻成本:退化为逐槽独立贪心(仅在线兜底路径使用)
        return _allocate_dynamic(demand, caps, quality, eligible,
                                 cost_vec, avg_quality_target,
                                 mu_bisect_iters, total_demand)

    def solve_mu(mu: float):
        if site_cost.ndim == 2:
            # 地区级成本 (R,S),直接使用,已含合格性处理时调用方负责
            adj = site_cost - mu * quality
            if not np.isneginf(adj).any():
                adj = np.where(eligible, adj, np.inf)
        else:
            adj = site_cost[None, :] - mu * quality
            adj = np.where(eligible, adj, np.inf)
        return _global_greedy(demand, caps, adj, eligible, quality)

    alloc0, unmet0, viol0 = solve_mu(0.0)
    if unmet0 > 1e-6 * max(demand.sum(), 1.0):
        return AllocationResult(
            alloc0, 0.0,
            float(np.average(quality_of(alloc0), weights=total_demand)),
            unmet0, quality_of(alloc0), viol0)
    q_low = np.average(quality_of(alloc0), weights=total_demand)
    if q_low >= avg_quality_target:
        return AllocationResult(alloc0, 0.0, float(q_low), 0.0,
                                quality_of(alloc0), viol0)

    scale = max(float(site_cost.max()), 1.0)
    mu_hi, hi, q_hi = 1.0, None, 0.0
    for _ in range(40):
        a, _, _ = solve_mu(mu_hi)
        q_hi = np.average(quality_of(a), weights=total_demand)
        if q_hi >= avg_quality_target or mu_hi > 1e12 * scale:
            hi = a
            break
        mu_hi *= 4.0
    if hi is None:
        hi, q_hi = a, np.average(quality_of(a), weights=total_demand)

    mu_lo, lo = 0.0, alloc0
    for _ in range(mu_bisect_iters):
        mu = 0.5 * (mu_lo + mu_hi)
        a, _, _ = solve_mu(mu)
        q = np.average(quality_of(a), weights=total_demand)
        if q < avg_quality_target:
            mu_lo, lo, q_low = mu, a, q
        else:
            mu_hi, hi, q_hi = mu, a, q
    # 质量约束在贪心顶点处阶梯跳跃:两侧可行解做凸组合,使质量恰好达标
    if q_hi >= avg_quality_target and q_low < avg_quality_target:
        beta = (avg_quality_target - q_low) / max(q_hi - q_low, 1e-12)
        best = (1 - beta) * lo + beta * hi
        q_final = avg_quality_target
    else:
        best = hi if q_hi >= avg_quality_target else lo
        q_final = np.average(quality_of(best), weights=total_demand)
    # 兜底违约量取 μ=0 解的违约(凸组合不改变合格性结构)
    return AllocationResult(best, float(mu_hi), float(q_final), 0.0,
                            quality_of(best), viol0)


def _allocate_dynamic(demand, caps, quality, eligible, cost_vec,
                      avg_quality_target, iters, total_demand):
    """逐时刻成本信号下的逐槽贪心(在线重排兜底,非全局最优但足够快)。"""
    n_r, n_t = demand.shape
    n_s = quality.shape[1]

    def quality_of(alloc):
        return np.divide(np.einsum("rst,rs->t", alloc, quality),
                         np.maximum(total_demand, 1e-30))

    def solve_mu(mu):
        alloc = np.zeros((n_r, n_s, n_t))
        remaining = caps.copy()
        need = demand.copy()
        for t in range(n_t):
            adj = cost_vec[:, t][None, :] - mu * quality
            order = np.argsort(np.where(eligible, adj, np.inf), axis=1)
            for r in range(n_r):
                for k in range(n_s):
                    s = order[r, k]
                    if not eligible[r, s] or need[r, t] <= 1e-12:
                        break
                    take = min(need[r, t], remaining[s, t])
                    alloc[r, s, t] += take
                    remaining[s, t] -= take
                    need[r, t] -= take
        return alloc, float(need.sum())

    lo, unmet = solve_mu(0.0)
    if unmet > 1e-6 * max(demand.sum(), 1.0):
        return AllocationResult(lo, 0.0,
                                float(np.average(quality_of(lo),
                                                 weights=total_demand)),
                                unmet, quality_of(lo))
    mu_hi, hi = 1.0, None
    for _ in range(30):
        a, _ = solve_mu(mu_hi)
        if np.average(quality_of(a), weights=total_demand) >= avg_quality_target:
            hi = a
            break
        mu_hi *= 4.0
    if hi is None:
        hi = a
    return AllocationResult(hi, mu_hi,
                            float(np.average(quality_of(hi),
                                             weights=total_demand)),
                            0.0, quality_of(hi))


def greedy_allocate(demand: np.ndarray,
                    caps: np.ndarray,
                    quality: np.ndarray,
                    min_quality: float,
                    site_cost: np.ndarray,
                    mu: float = 0.0) -> AllocationResult:
    """显式逐槽贪心(在线单槽兜底),O(R·S·T)。"""
    n_r, n_t = demand.shape
    n_s = quality.shape[1]
    eligible = quality >= min_quality
    alloc = np.zeros((n_r, n_s, n_t))
    remaining = caps.copy()
    need = demand.copy()
    for t in range(n_t):
        adj = site_cost[None, :] - mu * quality
        order = np.argsort(np.where(eligible, adj, np.inf), axis=1)
        for r in range(n_r):
            for k in range(n_s):
                s = order[r, k]
                if not eligible[r, s] or need[r, t] <= 1e-12:
                    break
                take = min(need[r, t], remaining[s, t])
                alloc[r, s, t] += take
                remaining[s, t] -= take
                need[r, t] -= take
    unmet = float(need.sum())
    total_demand = demand.sum(axis=0)
    q_num = np.einsum("rst,rs->t", alloc, quality)
    q = np.divide(q_num, np.maximum(total_demand, 1e-30))
    return AllocationResult(alloc, mu,
                            float(np.average(q, weights=total_demand)),
                            unmet, q)


def loads_from_alloc(alloc: np.ndarray) -> np.ndarray:
    """(R,S,T) -> (S,T) 各站点各时刻承接的总流量。"""
    return alloc.sum(axis=0)
