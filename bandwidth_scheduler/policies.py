"""对照策略与统一仿真运行器。

策略统一接口:给定本槽各地区需求、可用站点、质量矩阵,返回 DNS 目标权重。
- nearest:就近调度(每地区只给质量最高的可用机房,模拟现状)
- even   :平均分摊(所有可用机房权重相等)
- online :见 online.OnlineScheduler
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from . import SLOTS_PER_DAY, SLOTS_PER_MONTH
from .billing import SiteConfig
from .online import OnlineScheduler
from .simulation import DNSWeightController, World


# --------------------------------------------------------------------------- #
# 基线策略:输出 (S,) 目标份额
# --------------------------------------------------------------------------- #
def nearest_weights(demand_vec: np.ndarray, quality: np.ndarray,
                    available: np.ndarray) -> np.ndarray:
    n_r, n_s = quality.shape
    weights = np.zeros(n_s)
    for r in range(n_r):
        q = np.where(available, quality[r], -np.inf)
        weights[int(np.argmax(q))] += demand_vec[r]
    total = weights.sum()
    return weights / max(total, 1e-9)


def even_weights(demand_vec: np.ndarray, quality: np.ndarray,
                 available: np.ndarray) -> np.ndarray:
    w = available.astype(float)
    return w / max(w.sum(), 1e-9)


# --------------------------------------------------------------------------- #
# 单槽承接:运输 LP(合格性硬约束 + 物理容量 + DNS 份额偏好)
# --------------------------------------------------------------------------- #
_MATRIX_CACHE: dict = {}


def _static_constraints(n_r: int, n_s: int):
    """预构建并缓存不随槽变化的等式/不等式约束矩阵。"""
    key = (n_r, n_s)
    if key in _MATRIX_CACHE:
        return _MATRIX_CACHE[key]
    n = n_r * n_s
    A_eq, A_ub = [], []
    for r in range(n_r):
        row = np.zeros(n)
        row[r * n_s:(r + 1) * n_s] = 1.0
        A_eq.append(row)
    for s in range(n_s):
        row = np.zeros(n)
        for r in range(n_r):
            row[r * n_s + s] = 1.0
        A_ub.append(row)
    matrices = (np.array(A_eq), np.array(A_ub))
    _MATRIX_CACHE[key] = matrices
    return matrices


def _realize_slot(demand_vec: np.ndarray, share: np.ndarray,
                  available: np.ndarray, phys: np.ndarray,
                  quality: np.ndarray, min_quality: float) -> np.ndarray:
    """求一个槽的 (R,S) 承接量。

    目标:在合格站点内、物理容量约束下,最小化与 DNS 目标份额的偏离
    (成本取负份额,同份额偏好高质量);合格容量物理不足时加松弛变量,
    把违约流量压到最小并导向质量最高的非合格站点。
    """
    from scipy.optimize import linprog
    n_r, n_s = quality.shape
    eligible = (quality >= min_quality) & available[None, :]
    n = n_r * n_s
    A_eq, A_ub = _static_constraints(n_r, n_s)
    # 目标:负的份额偏好(高份额→低成本)+ 极小的质量项做同份额时的次序
    cost = np.zeros(n)
    for r in range(n_r):
        cost[r * n_s:(r + 1) * n_s] = (
            -share - 1e-4 * quality[r])
    b_eq = demand_vec
    b_ub = np.where(available, phys, 0.0)
    bounds = [(0.0, None)] * n
    for r in range(n_r):
        for s in range(n_s):
            if not eligible[r, s]:
                bounds[r * n_s + s] = (0.0, 0.0)
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if res.success:
        return res.x.reshape(n_r, n_s)
    # 合格容量物理不足:放开非合格站点参与,但给很高的目标惩罚,
    # 让求解器先用合格站点、残余才进非合格站点,并在其中偏好高质量。
    cost2 = cost.copy()
    bounds2 = list(bounds)
    max_abs = max(abs(cost).max(), 1.0)
    for r in range(n_r):
        for s in range(n_s):
            if not eligible[r, s] and available[s]:
                # 非合格对:高惩罚,且质量越高惩罚略低 → 优先高质量兜底
                cost2[r * n_s + s] = (max_abs * 1e3
                                      - 0.01 * quality[r, s])
                bounds2[r * n_s + s] = (0.0, None)
    res2 = linprog(cost2, A_ub=A_ub, b_ub=b_ub,
                   A_eq=A_eq, b_eq=b_eq,
                   bounds=bounds2, method="highs")
    if res2.success:
        return res2.x.reshape(n_r, n_s)
    # 最后兜底:按质量从高到低填充(理论上不会走到)
    alloc = np.zeros((n_r, n_s))
    used = np.zeros(n_s)
    for r in range(n_r):
        need = demand_vec[r]
        for s in np.argsort(-np.where(available, quality[r], -np.inf)):
            if need <= 1e-9:
                break
            take = min(need, max(0.0, phys[s] - used[s]))
            alloc[r, s] += take
            used[s] += take
            need -= take
    return alloc


# --------------------------------------------------------------------------- #
# 仿真运行
# --------------------------------------------------------------------------- #
@dataclass
class RunArtifacts:
    name: str
    loads: np.ndarray            # (S, T)
    alloc: np.ndarray            # (R, S, T)
    weights: np.ndarray          # (S, T) 实际承接份额
    target_weights: np.ndarray   # (S, T) 下发权重
    per_slot_quality: np.ndarray # (T,)
    unmet_total: float
    replan_events: int
    failure_events_seen: int


def _physical_capped_share(share: np.ndarray, total_demand: float,
                           caps: np.ndarray) -> np.ndarray:
    """份额对应流量超过物理上限时,把超出部分重分给还有余量的站点。"""
    load = share * total_demand
    over = load > caps
    excess = np.where(over, load - caps, 0.0).sum()
    load = np.minimum(load, caps)
    if excess > 1e-9:
        room = caps - load
        if room.sum() > 1e-9:
            load += excess * room / room.sum()
    return load / max(total_demand, 1e-9)


def run_policy(world: World,
               sites: Sequence[SiteConfig],
               policy: str,
               min_quality: float,
               avg_quality_target: float,
               n_free: int = 432,
               dns_tau_slots: float = 6.0,
               dns_noise: float = 0.02,
               online_cfg=None) -> RunArtifacts:
    n_r, t_total = world.demand.shape
    n_s = len(sites)
    phys = np.array([c.capacity_mbps for c in sites])

    dns = DNSWeightController(n_s, tau_slots=dns_tau_slots,
                              noise_sigma=dns_noise, seed=31)
    scheduler: Optional[OnlineScheduler] = None
    if policy == "online":
        dns_k = dns.k
        scheduler = OnlineScheduler(sites, world.quality, min_quality,
                                    avg_quality_target, dns_k=dns_k,
                                    n_free=n_free, cfg=online_cfg)

    loads = np.zeros((n_s, t_total))
    alloc = np.zeros((n_r, n_s, t_total))
    weights = np.zeros((n_s, t_total))
    targets = np.zeros((n_s, t_total))
    slot_q = np.zeros(t_total)
    unmet_total = 0.0
    replan_events = 0
    failures_seen = 0
    prev_available = np.ones(n_s, bool)

    month = -1
    month_start = 0

    for t in range(t_total):
        m = t // world.cfg.slots_per_month
        if m != month:
            month = m
            month_start = t
            if scheduler is not None:
                scheduler.month_boundary(month, world.cfg.slots_per_month)
            dns.reset()

        slot_in_month = t - month_start
        demand_vec = world.demand[:, t]
        available = ~world.failure_mask[:, t]
        if not np.array_equal(available, prev_available):
            failures_seen += 1
        total_demand = float(demand_vec.sum())

        if policy == "nearest":
            target = nearest_weights(demand_vec, world.quality, available)
        elif policy == "even":
            target = even_weights(demand_vec, world.quality, available)
        elif policy == "online":
            assert scheduler is not None
            fail_month = world.failure_mask[:, month_start:
                                            month_start + world.cfg.slots_per_month]
            target = scheduler.act(
                slot_in_month, demand_vec, available,
                fail_month, dns.share)
            replan_events = len(scheduler.replan_log)
        else:
            raise ValueError(policy)

        share = dns.step(target, available)
        # 单槽运输 LP:合格性硬约束 + 物理容量 + DNS 份额偏好,
        # 合格容量物理不足时最小化违约,保证不丢流量且正常工况零违约
        slot_alloc = _realize_slot(
            demand_vec, share, available, phys,
            world.quality, min_quality)
        load = slot_alloc.sum(axis=0)
        actual_share = load / max(total_demand, 1e-9)

        loads[:, t] = load
        alloc[:, :, t] = slot_alloc
        weights[:, t] = actual_share
        targets[:, t] = target
        q_num = (slot_alloc * world.quality).sum()
        slot_q[t] = q_num / max(total_demand, 1e-9)
        unmet = max(0.0, total_demand - load.sum())
        unmet_total += unmet

        if policy == "online" and scheduler is not None:
            # 预测误差统计:用缓存的当日预测当前列,避免逐槽全月重算
            base = (slot_in_month // SLOTS_PER_DAY) * SLOTS_PER_DAY
            if (scheduler._fc_cache is not None
                    and 0 <= slot_in_month - scheduler._fc_base
                    < scheduler._fc_cache.shape[1]):
                pred = scheduler._fc_cache[:, slot_in_month - scheduler._fc_base]
            else:
                pred = None
            scheduler.observe(slot_in_month, demand_vec, load, pred)
        prev_available = available.copy()

    return RunArtifacts(policy, loads, alloc, weights, targets, slot_q,
                        unmet_total, replan_events, failures_seen)
