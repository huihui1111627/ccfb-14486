"""离线最优标杆:全知信息下的月度账单最小化。

直接把 95 峰值写成混合整数规划,一个月有 S×8640 个二进制变量,通用求解器
规模上解不动。这里利用问题结构做凸松弛迭代:

* 固定每个站点的"免费槽集合"(允许冲破计费线的 432 个时刻),其余时刻上限压在
  计费线上,问题退化为线性规划,由 ``allocator.allocate``(LP 等价贪心)秒解;
* 用解出来的负载重新确定计费线与免费槽(最高的 432 个点),交替迭代到稳定;
* 多起点(均匀猜测 / 保底填充 / 总需求高峰优先)避免局部最优,每次迭代都用
  真实账单函数(含非凸阶梯跳档)评估,保留历史最好解。

小规模实例上与 scipy MILP 精确解对拍(见 tests/test_offline_vs_milp.py),
用于标定该启发式与真实最优的差距。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import SLOTS_PER_MONTH, P95_DROP_SAMPLES
from .allocator import allocate, loads_from_alloc
from .billing import SiteConfig, marginal_price_at, p95_line, total_monthly_cost


@dataclass
class OfflineResult:
    alloc: np.ndarray                  # (R, S, T)
    loads: np.ndarray                  # (S, T)
    cost: float
    lines: np.ndarray                  # (S,) 最终计费线
    start_costs: List[float]           # 各起点的最终账单
    cost_curve: List[float]            # 最优起点的迭代账单轨迹
    unmet: float
    quality_violation: float = 0.0     # 兜底放开到不合格站点的流量


def _free_slots(loads: np.ndarray, n_free: int) -> np.ndarray:
    """每个站点最高的 n_free 个采样点标记为免费槽。"""
    s, t = loads.shape
    mask = np.zeros_like(loads, dtype=bool)
    k = min(n_free, t)
    if k == 0:
        return mask
    idx = np.argpartition(-loads, kth=min(k - 1, t - 1), axis=1)[:, :k]
    mask[np.arange(s)[:, None], idx] = True
    return mask


def _build_caps(sites: Sequence[SiteConfig], lines: np.ndarray,
                free_mask: np.ndarray, failure_mask: Optional[np.ndarray],
                t_count: int) -> np.ndarray:
    """非免费槽上限 = max(计费线, 保底);免费槽放开到物理容量。"""
    caps = np.empty((len(sites), t_count))
    for i, cfg in enumerate(sites):
        if cfg.kind == "port":
            caps[i] = cfg.capacity_mbps
        else:
            floor = max(lines[i], cfg.commit_mbps)
            caps[i] = np.where(free_mask[i], cfg.capacity_mbps, floor)
    if failure_mask is not None:
        caps = np.where(failure_mask, 0.0, caps)
    return caps


def _assign_free_slots(loads: np.ndarray, n_free: int,
                       failure_mask: Optional[np.ndarray]) -> np.ndarray:
    """给每个 p95 站点强制分配恰好 n_free 个免费槽(按负载高低挑),
    可用槽不足时按实际数量。端口站点不需要免费槽。"""
    n_s, t_count = loads.shape
    mask = np.zeros_like(loads, dtype=bool)
    for s in range(n_s):
        available = np.ones(t_count, dtype=bool)
        if failure_mask is not None:
            available = ~failure_mask[s]
        idx_avail = np.where(available)[0]
        if idx_avail.size == 0:
            continue
        k = min(n_free, idx_avail.size)
        order = idx_avail[np.argsort(-loads[s, idx_avail])[:k]]
        mask[s, order] = True
    return mask


def _site_cost_signals(sites: Sequence[SiteConfig], lines: np.ndarray) -> np.ndarray:
    return np.array([marginal_price_at(max(lines[i], sites[i].commit_mbps), cfg)
                     for i, cfg in enumerate(sites)])


def _region_aware_cost(base_cost: np.ndarray, quality: np.ndarray,
                       min_quality: float) -> np.ndarray:
    """在站点级成本上叠加地区维度,返回 (R,S) 调整后成本。

    - 受限地区(合格站点少)的合格对给成本加成,优先保住其稀缺合格容量;
    - 非合格对加大硬惩罚,存在全合格可行解时不违约。
    base_cost: (S,) 站点成本。
    """
    n_r, n_s = quality.shape
    eligible = quality >= min_quality
    n_elig = eligible.sum(axis=1)
    max_elig = max(n_elig.max(), 1)
    flex_bonus = (max_elig - n_elig)[:, None] * (
        abs(base_cost).max() + 1.0)
    cost = np.broadcast_to(base_cost[None, :], (n_r, n_s)).astype(float).copy()
    cost = cost - np.where(eligible, flex_bonus, 0.0)
    hard = (abs(cost).max() + 1e6)
    cost = np.where(eligible, cost, cost + hard)
    return cost


def cfg_commit(cfg: SiteConfig) -> float:
    return cfg.commit_mbps if cfg.kind == "p95" else 0.0


def offline_optimal(demand: np.ndarray,
                    sites: Sequence[SiteConfig],
                    quality: np.ndarray,
                    min_quality: float,
                    avg_quality_target: float,
                    failure_mask: Optional[np.ndarray] = None,
                    n_free: int = P95_DROP_SAMPLES,
                    iters: int = 30,
                    extra_starts: Optional[List[np.ndarray]] = None,
                    verbose: bool = False) -> OfflineResult:
    """全知离线求解。

    demand       : (R, T)
    failure_mask : (S, T) True 表示该时刻站点不可用
    n_free       : 免费点数(整月 432;缩短实验时按比例传)
    """
    demand = np.asarray(demand, dtype=np.float64)
    n_r, t_count = demand.shape
    n_s = len(sites)
    total_per_slot = demand.sum(axis=0)
    mean_load = total_per_slot.mean() / max(1, n_s)
    physical_cap = np.array([c.capacity_mbps for c in sites])

    # ---- 可行性引导:物理容量下先求一个可行分配,用其负载初始化免费槽 --------
    phys_caps = np.tile(physical_cap[:, None], (1, t_count))
    if failure_mask is not None:
        phys_caps = np.where(failure_mask, 0.0, phys_caps)
    base = allocate(
        demand, phys_caps, quality, min_quality,
        avg_quality_target,
        site_cost=_region_aware_cost(
            np.array([c.unit_price or c.commit_price or 15.0 for c in sites]),
            quality, min_quality))
    # 故障叠加质量门槛可能导致合格容量物理不足。分配器已用"质量最高的非合格
    # 站点"兜底,把未承接与最差质量违约都压到最小;这里只在仍有真实未承接时报错
    if base.unmet > 1e-3 * max(demand.sum(), 1.0):
        raise RuntimeError(
            f"物理容量都无法承接需求,unmet={base.unmet:.1f};请扩容或下调质量门槛")
    quality_violation = base.quality_violation
    base_loads = loads_from_alloc(base.alloc)
    base_free = _assign_free_slots(base_loads, n_free, failure_mask)
    base_lines = np.array([
        p95_line(base_loads[i]) if sites[i].kind == "p95" else 0.0
        for i in range(n_s)])

    # 全局需求最高峰的 n_free 个槽,作为所有 p95 站点共享的免费槽:
    # 保证受限地区在最紧张的时刻所有合格站点都放开到物理容量
    peak_idx = np.argpartition(-total_per_slot,
                               min(n_free - 1, t_count - 1))[:n_free]
    global_peak_free = np.zeros((n_s, t_count), dtype=bool)
    for s, cfg in enumerate(sites):
        if cfg.kind == "p95":
            avail = np.ones(t_count, dtype=bool)
            if failure_mask is not None:
                avail = ~failure_mask[s]
            global_peak_free[s, peak_idx[avail[peak_idx]]] = True
            # 若故障占用了部分高峰槽,用其余最高槽补足到 n_free
            if global_peak_free[s].sum() < min(n_free, avail.sum()):
                have = set(np.where(global_peak_free[s])[0])
                cand = np.argsort(-total_per_slot)
                for t in cand:
                    if avail[t] and t not in have:
                        global_peak_free[s, t] = True
                        have.add(t)
                    if global_peak_free[s].sum() >= min(n_free, avail.sum()):
                        break
    # 削峰线初值:站点均值的 2.5 倍(经验可行水平),保底托底
    shave_lines = np.array([
        max(mean_load * 2.5, cfg_commit(c)) if c.kind == "p95" else 0.0
        for c in sites])

    # ---- 多起点:计费线初值 + 首版免费槽 -----------------------------------
    starts: List[Tuple[str, np.ndarray, np.ndarray]] = [
        ("globalpeak", shave_lines, global_peak_free),
        ("feasible", base_lines, base_free),
        ("uniform", np.full(n_s, mean_load), global_peak_free.copy()),
        ("commit",
         np.array([c.commit_mbps if c.kind == "p95" else 0.0 for c in sites]),
         global_peak_free.copy()),
    ]
    if extra_starts is not None:
        for k, ls in enumerate(extra_starts):
            starts.append((f"warm{k}", np.asarray(ls, float),
                           global_peak_free.copy()))

    best = None
    best_cost = np.inf
    best_curve: List[float] = []
    start_costs: List[float] = []

    for name, init_lines, init_free in starts:
        lines = init_lines.copy()
        free_mask = init_free.copy()
        curve: List[float] = []
        local_best_cost = np.inf
        local_best = None
        for it in range(iters):
            caps = _build_caps(sites, lines, free_mask, failure_mask, t_count)
            # 站点边际单价(免费/非免费槽的差异已由 caps 上限体现),
            # 再叠加地区维度的受限优先与合格硬惩罚
            site_sig = _site_cost_signals(sites, lines)
            region_cost = _region_aware_cost(site_sig, quality, min_quality)
            res = allocate(demand, caps, quality, min_quality,
                           avg_quality_target, site_cost=region_cost)
            if res.unmet > 1e-5 * max(demand.sum(), 1.0):
                # 信任域自适应:非免费槽上限压得过低。把出现缺口的站点
                # 计费线试探性抬升 25%(向物理容量),在可行与省钱间平衡
                gap_loads = loads_from_alloc(res.alloc)
                served_per_slot = gap_loads.sum(axis=0)
                tight_slots = served_per_slot < demand.sum(axis=0) - 1e-6
                for s in range(n_s):
                    if sites[s].kind == "port":
                        continue
                    # 该站点在紧张槽参与承接 → 抬线
                    if tight_slots.any() and (gap_loads[s, tight_slots] > 1).any():
                        lines[s] = min(sites[s].capacity_mbps,
                                       max(lines[s], cfg_commit(sites[s])) * 1.25)
                free_mask = _assign_free_slots(gap_loads, n_free, failure_mask)
                caps = _build_caps(sites, lines, free_mask,
                                   failure_mask, t_count)
                region_cost = _region_aware_cost(
                    _site_cost_signals(sites, lines), quality, min_quality)
                res = allocate(demand, caps, quality, min_quality,
                               avg_quality_target, site_cost=region_cost)
                if res.unmet > 1e-5 * max(demand.sum(), 1.0):
                    if verbose:
                        print(f"[{name}] iter{it} 抬线后仍不可行,物理兜底")
                    res = allocate(
                        demand, phys_caps, quality, min_quality,
                        avg_quality_target,
                        site_cost=_region_aware_cost(
                            _site_cost_signals(sites, lines),
                            quality, min_quality))
                    if res.unmet > 1e-5 * max(demand.sum(), 1.0):
                        break
            loads = loads_from_alloc(res.alloc)
            cost = total_monthly_cost(loads, sites)
            curve.append(cost)
            if cost < local_best_cost - 1e-8:
                local_best_cost = cost
                local_best = (res.alloc.copy(), loads.copy())
            # 重新估计计费线;免费槽按配额强制分配(保证每站点都有免费额度可用)
            new_lines = np.array([
                p95_line(loads[i]) if sites[i].kind == "p95" else 0.0
                for i in range(n_s)])
            new_free = _assign_free_slots(loads, n_free, failure_mask)
            moved = np.abs(new_lines - lines).max()
            lines = 0.5 * lines + 0.5 * new_lines
            free_mask = new_free
            if verbose:
                print(f"[{name}] iter{it} bill={cost:,.0f} line_move={moved:.2f}")
            if moved < 1e-6 and it > 2:
                break
        start_costs.append(local_best_cost)
        if local_best is not None and local_best_cost < best_cost:
            best_cost = local_best_cost
            best = local_best
            best_curve = curve

    # 所有迭代起点都失败时,用物理容量可行解兜底(保证返回可行方案)
    if best is None:
        best = (base.alloc.copy(), base_loads.copy())
        best_cost = total_monthly_cost(base_loads, sites)
        best_curve = [best_cost]
        start_costs.append(best_cost)
    alloc, loads = best
    # 最终解的真实质量违约:落在不合格站点的流量
    eligible = quality >= min_quality
    viol = float(sum(
        alloc[r, s].sum() for r in range(n_r) for s in range(n_s)
        if not eligible[r, s]))
    # ---- 质量净化:逐槽用精确运输 LP,在合格性硬约束下做成本最优重分配 ----
    # 全局贪心在紧约束下可能把少量流量放到非合格站点。逐槽 LP 以站点边际
    # 成本的倒数为偏好(低成本=高偏好),既强制合格性又保持成本最优,
    # 而不是机械维持原负载比例(那样会无谓抬高账单)。
    from .policies import _realize_slot
    phys = np.array([c.capacity_mbps for c in sites])
    marginal = np.array([
        0.0 if sites[i].kind == "port"
        else marginal_price_at(sites[i].commit_mbps, sites[i])
        for i in range(n_s)])
    # 用各站点单位成本构造份额偏好:成本越低偏好越高,加小常数避免除零
    cost_pref = 1.0 / (marginal + 1.0)
    clean = np.zeros_like(alloc)
    for t in range(t_count):
        avail = (np.ones(n_s, bool) if failure_mask is None
                 else ~failure_mask[:, t])
        clean[:, :, t] = _realize_slot(
            demand[:, t], cost_pref, avail, phys, quality, min_quality)
    clean_loads = clean.sum(axis=0)
    clean_cost = total_monthly_cost(clean_loads, sites)
    # 质量硬约束优先:只要净化能降低违约就采用零违约的分配,账单按净化后的
    # 真实负载如实计算(标杆仍合法,只是为满足质量门槛付出了成本)。
    clean_viol = float(sum(
        clean[r, s].sum() for r in range(n_r) for s in range(n_s)
        if not eligible[r, s]))
    if clean_viol < viol - 1e-9:
        alloc, loads, best_cost, viol = clean, clean_loads, clean_cost, clean_viol

    final_lines = np.array([p95_line(loads[i]) if sites[i].kind == "p95" else 0.0
                            for i in range(n_s)])
    return OfflineResult(
        alloc=alloc, loads=loads, cost=best_cost, lines=final_lines,
        start_costs=start_costs, cost_curve=best_curve,
        unmet=0.0, quality_violation=viol,
    )
