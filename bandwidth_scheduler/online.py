"""在线滚动调度器。

核心决策逻辑
============
1. 免费额度节奏(432 个点 = 36 小时不能早用光也不能浪费)
   - 按本月进度计算"理论上此刻还应剩多少免费点":budget_allowance(t);
   - 剩余额度高于 allowance → 本月偏保守,允许在预测高峰时冲高;
   - 低于 allowance → 进入止血,非免费槽严格压在当前计费线以下。
2. 槽位分级
   - free 槽:预测属于剩余窗口内最高的一批时刻,可冲到物理容量;
   - normal 槽:上限 = 当前计费线 × soft 系数(月初宽松,月末收紧);
   - 保底:任何时刻 cap 不低于 commit_mbps,鼓励先吃满保底。
3. 滚动重排:每隔 replan_every 个槽,用"真实历史 + 预测未来"拼一个缩短视野的
   离线优化(offline.offline_optimal),免费点数按剩余槽数等比例折算;
   故障发生时立即重排,重排免费槽集合即"重新安排免费额度计划"。
4. DNS 滞后:下发权重时按一阶惯性做开环补偿
   w_target = x_desired + (1/k)·(x_desired - x_prev),让实际份额更快到位;
   实现误差由仿真侧噪声体现,控制侧不假设精确执行。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import P95_DROP_SAMPLES, SLOTS_PER_DAY, SLOTS_PER_MONTH
from .allocator import allocate
from .billing import (SiteConfig, current_line_and_free_budget,
                      marginal_price_at, p95_line)
from .forecast import DemandForecaster
from .offline import offline_optimal


@dataclass
class OnlineConfig:
    replan_every: int = SLOTS_PER_DAY               # 每天重排一次
    plan_horizon_days: int = 3                   # 缩短视野,控制求解规模
    soft_line_early: float = 1.15                # 月初计费线上浮 15%
    soft_line_late: float = 1.02                 # 月末只留 2% 余量
    dns_gain: float = 1.0                        # 滞后补偿强度
    forecast_peak_quantile: float = 0.95
    free_budget_cushion: float = 1.2             # 剩余额度 > 节奏线 × 该值才放开冲
    plan_blend: float = 0.5                      # 合格站点内的计划倾斜权重


class OnlineScheduler:
    def __init__(self,
                 sites: Sequence[SiteConfig],
                 quality: np.ndarray,
                 min_quality: float,
                 avg_quality_target: float,
                 dns_k: float,
                 cfg: Optional[OnlineConfig] = None,
                 n_free: int = P95_DROP_SAMPLES,
                 seed: int = 11):
        self.sites = sites
        self.n_s = len(sites)
        self.quality = np.asarray(quality, float)
        self.min_q = min_quality
        self.avg_q = avg_quality_target
        self.dns_k = dns_k
        self.cfg = cfg or OnlineConfig()
        self.rng = np.random.default_rng(seed)
        self.n_r = quality.shape[0]
        self.n_free_total = n_free
        self.forecaster = DemandForecaster(self.n_r)
        self._reset_month(0, SLOTS_PER_MONTH)

    # ------------------------------------------------------------------ #
    def _reset_month(self, month: int, spm: int) -> None:
        self.month = month
        self.spm = spm
        self.samples = [[] for _ in range(self.n_s)]
        self.lines = np.zeros(self.n_s)
        self.free_left = np.full(self.n_s, self.n_free_total)
        self.plan_table: Optional[np.ndarray] = None     # (S, horizon) 目标份额
        self.plan_base_slot = -1
        self._fc_cache: Optional[np.ndarray] = None
        self._fc_base = -1
        self.last_replan = -10 ** 9
        self.actual_share = np.full(self.n_s, 1.0 / self.n_s)
        self.replan_log: List[dict] = []
        self._day_obs: List[np.ndarray] = []

    # ------------------------------------------------------------------ #
    def month_boundary(self, month: int, spm: int) -> None:
        self._reset_month(month, spm)

    # ------------------------------------------------------------------ #
    def _pacing_allowance(self, t: int) -> float:
        """按进度线性释放免费额度,月末允许留 5% 机动(避免浪费)。"""
        frac_left = max(0.0, 1.0 - t / self.spm)
        return self.n_free_total * frac_left * 0.95

    def _soft_factor(self, t: int) -> float:
        frac = t / self.spm
        return self.cfg.soft_line_early + (
            self.cfg.soft_line_late - self.cfg.soft_line_early) * frac

    # ------------------------------------------------------------------ #
    def _build_caps(self, t: int, horizon: int,
                    forecast_demand: np.ndarray,
                    failure_mask_future: np.ndarray) -> np.ndarray:
        """根据计费线/免费节奏构造未来 horizon 槽的 (S, H) 上限。"""
        caps = np.empty((self.n_s, horizon))
        soft = self._soft_factor(t)
        allowance = self._pacing_allowance(t)
        total_fc = forecast_demand.sum(axis=0)                 # (H,)
        # 预测高峰槽:每个站点各自挑 top-k,数量 = 该站点剩余免费额度
        k_global = min(horizon, max(1, int(
            np.ceil(self.n_free_total * horizon / self.spm))))
        peak_global = np.zeros(horizon, bool)
        if k_global < horizon:
            idx = np.argpartition(-total_fc, k_global - 1)[:k_global]
            peak_global[idx] = True
        else:
            peak_global[:] = True

        for s, cfg in enumerate(self.sites):
            if cfg.kind == "port":
                caps[s] = cfg.capacity_mbps
                continue
            # 冷启动(line 尚未建立)时,用"物理容量的公平份额"托底,
            # 避免 line≈0 把正常槽误判成透支而压死
            fair_share = cfg.capacity_mbps / max(1, self.n_s)
            line_floor = max(self.lines[s], cfg.commit_mbps,
                             0.5 * fair_share)
            normal_cap = line_floor * soft
            cap_row = np.full(horizon, normal_cap)
            if self.free_left[s] >= allowance * self.cfg.free_budget_cushion:
                # 额度充裕:预测高峰放开到物理容量
                cap_row = np.where(peak_global, cfg.capacity_mbps, cap_row)
            elif self.free_left[s] < allowance and self.lines[s] > 1.0:
                # 额度透支且计费线已建立:压到计费线以下,保底仍保留
                cap_row = np.maximum(self.lines[s] * 0.995,
                                     cfg.commit_mbps)
            caps[s] = cap_row
        caps = np.where(failure_mask_future, 0.0, caps)
        return caps

    # ------------------------------------------------------------------ #
    def _replan(self, t: int,
                forecast_demand: np.ndarray,
                future_fail: np.ndarray) -> np.ndarray:
        """只在未来预测窗口上做缩短视野离线优化。

        历史不重新求解(否则每轮成本随月内进度平方增长),其影响通过当前
        计费线 self.lines 与剩余免费额度 self.free_left 注入:用 warm 起点
        把当前线作为初始线,免费额度按剩余窗口等比例折算。
        """
        horizon = forecast_demand.shape[1]
        n_free = min(self.n_free_total,
                     max(1, int(round(
                         self.n_free_total * horizon / self.spm))))
        # 用当前已建立的计费线做 warm 起点,避免冷启动把线压到 0
        warm_lines = self.lines.copy()
        try:
            res = offline_optimal(
                forecast_demand, self.sites, self.quality,
                self.min_q, self.avg_q,
                failure_mask=future_fail,
                n_free=n_free, iters=6,
                extra_starts=[warm_lines],
            )
            future_loads = res.loads
        except RuntimeError as e:
            # 极端容量紧张:退回启发式上限,由逐槽贪心兜底
            future_loads = np.tile(
                np.maximum(self.lines,
                           [c.commit_mbps for c in self.sites])[:, None],
                (1, horizon))
            self.replan_log.append({"slot": t, "status": f"fallback:{e}"})
        future_total = forecast_demand.sum(axis=0)
        shares = np.divide(future_loads,
                           np.maximum(future_total[None, :], 1e-6))
        # 物理份额裁剪后重归一化交给 DNS
        shares = np.clip(shares, 0.0, 1.0)
        self.plan_table = shares
        self.plan_base_slot = t
        self.last_replan = t
        self.replan_log.append({"slot": t, "status": "ok",
                                "horizon": horizon})
        return shares

    # ------------------------------------------------------------------ #
    def _greedy_weights(self, demand_vec: np.ndarray, caps_vec: np.ndarray
                        ) -> np.ndarray:
        """单槽贪心:按地区逐个把需求分给调整后成本最低且满足质量门槛的站点。"""
        alloc = np.zeros((self.n_r, self.n_s))
        remaining = caps_vec.copy()
        for r in range(self.n_r):
            need = demand_vec[r]
            elig = self.quality[r] >= self.min_q
            # 成本信号 + 质量软偏好(μ 用小常数,硬门槛已保证最差质量)
            scores = np.array([
                marginal_price_at(max(self.lines[s], self.sites[s].commit_mbps),
                                  self.sites[s])
                for s in range(self.n_s)]) - 0.02 * self.quality[r]
            scores = np.where(elig, scores, np.inf)
            order = np.argsort(scores)
            for s in order:
                if not np.isfinite(scores[s]) or need <= 1e-9:
                    break
                take = min(need, remaining[s])
                alloc[r, s] += take
                remaining[s] -= take
                need -= take
            if need > 1e-6:
                # 容量不可行:放开到物理容量,按质量兜底(故障场景的最后防线)
                phys = np.array([c.capacity_mbps for c in self.sites])
                order2 = np.argsort(-np.where(elig, self.quality[r], -np.inf))
                for s in order2:
                    if not elig[s] or need <= 1e-9:
                        continue
                    room = max(0.0, phys[s] - alloc[:, s].sum())
                    take = min(need, room)
                    alloc[r, s] += take
                    need -= take
        total = alloc.sum()
        return alloc.sum(axis=0) / max(total, 1e-9)

    # ------------------------------------------------------------------ #
    def act(self, slot_in_month: int,
            demand_vec: np.ndarray,
            available: np.ndarray,
            failure_mask_full: np.ndarray,
            dns_share: np.ndarray) -> np.ndarray:
        """返回本槽下发给 DNS 的目标权重(和为 1)。

        failure_mask_full : (S, spm) 本月故障掩码;在线侧只用当前槽可用性,
                            未来故障不可见,重排按全部可用处理。
        dns_share         : 上一槽实际承接份额
        """
        t = slot_in_month
        self.actual_share = np.asarray(dns_share, float)

        # 月初冷启动:用一个全天预测先排一次
        force_replan = (t - self.last_replan) >= self.cfg.replan_every
        # 故障新发:可用集合与上一槽不同 → 立即重排
        if t > 0:
            prev_avail = ~failure_mask_full[:, t - 1]
            if not np.array_equal(prev_avail, available):
                force_replan = True

        horizon = min(self.cfg.plan_horizon_days * SLOTS_PER_DAY,
                      self.spm - t)
        # 当日预测缓存:只在日界/重排时重建,逐槽只取一列
        day_start = (t // SLOTS_PER_DAY) * SLOTS_PER_DAY
        if self.plan_base_slot != day_start or self._fc_cache is None:
            fc_full = self.forecaster.forecast_remaining(day_start)
            self._fc_cache = fc_full.month_remaining
            self._fc_base = day_start
        fc_month = self._fc_cache[:, t - self._fc_base:]
        forecast_demand = fc_month[:, :horizon]

        if force_replan or self.plan_table is None:
            # 重排只看未来窗口:在线视角未来故障不可见(无通告),按全部可用;
            # 当前槽可用性已由 available 在权重生成环节强制
            fail_future = np.zeros((self.n_s, horizon), dtype=bool)
            shares_plan = self._replan(t, forecast_demand, fail_future)
            plan_col0 = shares_plan[:, 0]
        else:
            offset = t - self.plan_base_slot
            if offset < self.plan_table.shape[1]:
                plan_col0 = self.plan_table[:, offset]
            else:
                plan_col0 = self.actual_share

        # 逐槽安全上限 + 贪心(本身已强制每地区只走合格站点,质量安全)
        caps_now = self._build_caps(t, 1, demand_vec[:, None],
                                    (~available)[:, None])[:, 0]
        greedy = self._greedy_weights(demand_vec, caps_now)
        # plan 仅在"对所有活跃地区都合格"的站点集合内做成本倾斜,
        # 基准始终是质量安全的贪心,保证混合后不破坏最差质量门槛
        active_regions = demand_vec > 1e-9
        demand_eligible = np.array([
            np.all(self.quality[active_regions, s] >= self.min_q)
            for s in range(self.n_s)])
        safe_plan = np.where(demand_eligible & available, plan_col0, 0.0)
        if safe_plan.sum() > 1e-12:
            safe_plan /= safe_plan.sum()
            b = self.cfg.plan_blend
            desired = b * safe_plan + (1 - b) * greedy
        else:
            desired = greedy
        desired = np.where(available, desired, 0.0)
        if desired.sum() <= 0:
            desired = np.where(available, greedy, 0.0)
        if desired.sum() <= 0:
            desired = available.astype(float)
        desired /= desired.sum()

        # DNS 滞后开环补偿: w = x* + (1/k)(x* - x_prev)
        compensated = desired + self.cfg.dns_gain * (
            desired - self.actual_share) / max(self.dns_k, 1e-3)
        compensated = np.where(available, np.maximum(compensated, 0.0), 0.0)
        if compensated.sum() <= 0:
            compensated = desired
        return compensated / compensated.sum()

    # ------------------------------------------------------------------ #
    def observe(self, slot_in_month: int, demand_vec: np.ndarray,
                realized_loads: np.ndarray, pred_vec: Optional[np.ndarray]
                ) -> None:
        """收到本槽真实承接结果后更新状态。"""
        for s in range(self.n_s):
            self.samples[s].append(float(realized_loads[s]))
            line, budget = current_line_and_free_budget(
                np.asarray(self.samples[s]), self.spm,
                n_free_total=self.n_free_total)
            self.lines[s] = line
            self.free_left[s] = budget
        self.forecaster.update(slot_in_month, demand_vec, pred_vec)
        self._day_obs.append(demand_vec.copy())
        if (slot_in_month + 1) % SLOTS_PER_DAY == 0 and self._day_obs:
            self.forecaster.end_of_day_update(np.column_stack(self._day_obs))
            self._day_obs = []
