"""仿真环境:地区流量、服务质量、故障、DNS 权重滞后。

流量模型(短剧分发的典型形状)
------------------------------
demand[r, t] = 基础日周期 × 周内效应 × 月增长 × 地区系数
               + 爆款尖峰(稀疏、短促、幅度大,用来考验"免费 36 小时"的用法)
               + 噪声
所有尖峰在生成时记录到 ``spike_log``,便于后面分析"预测偏差大的那几天"。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from . import SLOTS_PER_DAY, SLOTS_PER_MONTH


class FailureEvent(NamedTuple):
    site_idx: int
    start_slot: int       # 全局槽序号(含月份偏移)
    end_slot: int         # 开区间
    reason: str = ""


@dataclass
class WorldConfig:
    n_regions: int = 8
    n_sites: int = 12
    months: int = 3
    slots_per_month: int = SLOTS_PER_MONTH
    seed: int = 20261122
    base_demand_mbps: float = 2800.0
    # 每天几个尖峰、尖峰相对当日均值的倍数
    spike_rate_per_day: float = 0.35
    spike_mult_range: Tuple[float, float] = (1.4, 1.9)
    spike_len_slots: Tuple[int, int] = (3, 14)     # 15 分钟 ~ 70 分钟
    # 每月 1~2 次机房故障,持续 20 分钟 ~ 6 小时
    failure_rate_per_month: float = 1.4
    failure_len_slots: Tuple[int, int] = (4, 72)
    # 质量矩阵:就近质量高,远距离衰减
    geo_decay: float = 0.55
    geo_noise: float = 8.0


@dataclass
class World:
    cfg: WorldConfig
    demand: np.ndarray                  # (R, T_total)
    quality: np.ndarray                 # (R, S)
    failures: List[FailureEvent]
    failure_mask: np.ndarray            # (S, T_total)
    spike_log: List[dict]
    region_names: List[str]
    site_names: List[str]
    month_bounds: List[Tuple[int, int]]

    @property
    def n_slots(self) -> int:
        return self.demand.shape[1]


# --------------------------------------------------------------------------- #
# 流量生成
# --------------------------------------------------------------------------- #
def _diurnal(t: np.ndarray) -> np.ndarray:
    """短剧流量:午间小峰 + 晚间 21 点大峰,凌晨低谷。"""
    hour = (t % SLOTS_PER_DAY) / SLOTS_PER_DAY * 24.0
    evening = 0.95 * np.exp(-0.5 * ((hour - 21.0) / 2.6) ** 2)
    noon = 0.38 * np.exp(-0.5 * ((hour - 12.5) / 1.8) ** 2)
    base = 0.22
    return base + evening + noon


def build_world(cfg: Optional[WorldConfig] = None) -> World:
    cfg = cfg or WorldConfig()
    rng = np.random.default_rng(cfg.seed)
    n_r, n_s = cfg.n_regions, cfg.n_sites
    t_total = cfg.months * cfg.slots_per_month
    t = np.arange(t_total)

    # ---- 需求 ---------------------------------------------------------------
    region_scale = rng.lognormal(mean=0.0, sigma=0.45, size=n_r)
    region_scale /= region_scale.mean()
    shape = _diurnal(t)
    # 周内效应:周末高 12%
    dow = (t // SLOTS_PER_DAY) % 7
    weekly = np.where((dow == 5) | (dow == 6), 1.12, 1.0)
    # 月增长:每月 6% 的自然增长
    month_idx = t // cfg.slots_per_month
    growth = 1.0 + 0.06 * month_idx
    noise = rng.lognormal(mean=0.0, sigma=0.10, size=(n_r, t_total))
    demand = (cfg.base_demand_mbps * region_scale[:, None]
              * shape[None, :] * weekly[None, :] * growth[None, :]
              * noise)

    # ---- 爆款尖峰 ------------------------------------------------------------
    spike_log: List[dict] = []
    n_days = t_total // SLOTS_PER_DAY
    for day in range(n_days):
        if rng.random() < cfg.spike_rate_per_day:
            r = rng.integers(0, n_r)
            length = int(rng.integers(*cfg.spike_len_slots))
            start = day * SLOTS_PER_DAY + int(rng.integers(24, SLOTS_PER_DAY - 30))
            mult = rng.uniform(*cfg.spike_mult_range)
            end = min(start + length, t_total)
            env = np.sin(np.linspace(0, np.pi, end - start))
            extra = cfg.base_demand_mbps * region_scale[r] * (mult - 1.0) * env
            demand[r, start:end] += extra
            spike_log.append({"region": int(r), "start": start, "end": end,
                              "day": day, "mult": float(mult),
                              "peak_mbps": float(demand[r, start:end].max())})

    # ---- 质量矩阵(地理就近)--------------------------------------------------
    # 把地区和机房都放在一个一维"地理轴"上,距离越远质量越低
    region_pos = np.linspace(0, 1, n_r) + rng.normal(0, 0.03, n_r)
    site_pos = np.linspace(0, 1, n_s) + rng.normal(0, 0.05, n_s)
    dist = np.abs(region_pos[:, None] - site_pos[None, :])
    quality = 100.0 - cfg.geo_decay * 100.0 * dist ** 0.7 \
        + rng.normal(0, cfg.geo_noise, size=(n_r, n_s))
    quality = np.clip(quality, 35.0, 100.0)

    # ---- 故障 ----------------------------------------------------------------
    failures: List[FailureEvent] = []
    failure_mask = np.zeros((n_s, t_total), dtype=bool)
    for m in range(cfg.months):
        n_fail = rng.poisson(cfg.failure_rate_per_month)
        for _ in range(n_fail):
            s = int(rng.integers(0, n_s))
            length = int(rng.integers(*cfg.failure_len_slots))
            # 故障起点需给故障长度留出空间,短月时收缩可选范围
            hi = max(SLOTS_PER_DAY + 1,
                     cfg.slots_per_month - SLOTS_PER_DAY - length)
            start = m * cfg.slots_per_month + int(rng.integers(
                min(SLOTS_PER_DAY, hi), hi))
            end = min(start + length, t_total)
            failures.append(FailureEvent(s, start, end,
                                         reason=rng.choice(
                                             ["链路中断", "设备宕机", "拥塞"])))
            failure_mask[s, start:end] = True

    region_names = [f"R{i}" for i in range(n_r)]
    site_names = [f"IDC-{i:02d}" for i in range(n_s)]
    month_bounds = [(m * cfg.slots_per_month,
                     (m + 1) * cfg.slots_per_month) for m in range(cfg.months)]
    return World(cfg, demand, quality, failures, failure_mask, spike_log,
                 region_names, site_names, month_bounds)


# --------------------------------------------------------------------------- #
# DNS 权重滞后模型
# --------------------------------------------------------------------------- #
class DNSWeightController:
    """改权重后流量按一阶惯性慢慢切过去,且实际比例带实现误差。

    实际承接比例 x(t) = x(t-1) + k * (w(t) - x(t-1)) + 噪声,
    k 由时间常数 tau_slots 决定:k = 1 - exp(-1/tau)。
    权重 w 是策略下发的目标(按站点)。
    """

    def __init__(self, n_sites: int, tau_slots: float = 6.0,
                 noise_sigma: float = 0.02, seed: int = 7):
        self.n = n_sites
        self.k = 1.0 - np.exp(-1.0 / tau_slots)
        self.noise_sigma = noise_sigma
        self.rng = np.random.default_rng(seed)
        self.share = np.full(n_sites, 1.0 / n_sites)

    def reset(self) -> None:
        self.share = np.full(self.n, 1.0 / self.n)

    def step(self, target_share: np.ndarray,
             available: Optional[np.ndarray] = None) -> np.ndarray:
        """推进一个时刻,返回实际各站点承接份额(和为 1,不可用站点为 0)。"""
        w = np.asarray(target_share, float).copy()
        if available is not None:
            w = np.where(available, w, 0.0)
        s = w.sum()
        if s <= 0:
            raise RuntimeError("没有任何可用站点承接流量")
        w /= s
        new_share = self.share + self.k * (w - self.share)
        new_share += self.rng.normal(0, self.noise_sigma, self.n)
        new_share = np.maximum(new_share, 0.0)
        if available is not None:
            new_share = np.where(available, new_share, 0.0)
        total = new_share.sum()
        if total <= 0:
            new_share = w.copy()
        else:
            new_share /= total
        self.share = new_share
        return new_share
