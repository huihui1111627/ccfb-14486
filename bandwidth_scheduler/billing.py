"""统一计费模型。

一个机房(站点)的月费拆成三个部分,各种商务条款都能落到这张表上:

1. 固定费 ``fixed_fee``
   - 包端口站点:端口月租,流量不再按用量收费,只有物理上限 ``capacity_mbps``。
   - 带保底的 95 站点:保底费 = ``commit_price * commit_mbps``,用不到也要付。
2. 95 峰值计费量 ``billed``:当月采样点降序,去掉最高 5%(432 个点 = 36 小时),
   第 433 个点的值即计费线;保底只抵扣不抬高,故
   ``billed = max(p95_line, commit_mbps)``。
3. 可变费:超过保底的部分按单价或阶梯价结算。

阶梯价支持两种行业常见口径:
- ``marginal``(分段边际价):超过保底的部分按档积分;
- ``whole``(整单价):计费量落到哪一档,全部量按该档单价(量大降价的跳档模型)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import P95_BILLABLE_RANK, P95_DROP_SAMPLES, SLOTS_PER_MONTH


# --------------------------------------------------------------------------- #
# 站点配置
# --------------------------------------------------------------------------- #
@dataclass
class SiteConfig:
    name: str
    kind: str = "p95"                       # "p95" 或 "port"
    capacity_mbps: float = 1000.0           # 物理上限(端口速率 / 开通带宽)
    unit_price: float = 0.0                 # 元 / Mbps / 月(超保底单价)
    commit_mbps: float = 0.0                # 保底量 Mbps
    commit_price: float = 0.0               # 保底单价 元 / Mbps / 月
    # 阶梯:[(该档上限 Mbps, 该档单价), ...],上限用 inf 表示最后一档
    tiers: Optional[List[Tuple[float, float]]] = None
    tier_mode: str = "marginal"             # "marginal" | "whole"
    port_fee: float = 0.0                   # 包端口月租(元)

    def __post_init__(self) -> None:
        if self.kind not in ("p95", "port"):
            raise ValueError(f"未知站点类型 {self.kind!r}")
        if self.kind == "port":
            self.port_fee = self.port_fee or self.commit_price * self.commit_mbps
            self.commit_mbps = 0.0
            self.tiers = None


# --------------------------------------------------------------------------- #
# 95 峰值
# --------------------------------------------------------------------------- #
def p95_line(samples: np.ndarray) -> float:
    """降序去掉最高 5% 后,剩余点的最大值(即第 433 高的点)。

    样本不足一个月时按实际点数等比例剔除,保证在线过程中也能算"当前计费线"。
    """
    x = np.asarray(samples, dtype=float)
    if x.size == 0:
        return 0.0
    drop = int(round(x.size * 0.05))
    order = np.argpartition(-x, min(drop, x.size - 1)) if drop < x.size else np.arange(x.size)
    if drop == 0:
        return float(x.max())
    kept = x[order[drop:]] if drop < x.size else x[order[-1:]]
    return float(kept.max()) if kept.size else 0.0


def free_slot_mask(samples: np.ndarray) -> np.ndarray:
    """哪些采样点属于"白送的 36 小时":数值严格高于计费线的点。

    降序前 432 个点免费;与计费线持平的点不计入免费(持平不抬价,本来就免费)。
    """
    x = np.asarray(samples, dtype=float)
    if x.size == 0:
        return np.zeros(0, dtype=bool)
    line = p95_line(x)
    return x > line + 1e-9


def billed_quantity(samples: np.ndarray, cfg: SiteConfig) -> float:
    """月计费量 = max(95 计费线, 保底量);包端口站点没有计费量概念。"""
    if cfg.kind == "port":
        return 0.0
    return max(p95_line(samples), cfg.commit_mbps)


# --------------------------------------------------------------------------- #
# 阶梯价
# --------------------------------------------------------------------------- #
def _whole_tier_price(q: float, tiers: Sequence[Tuple[float, float]]) -> float:
    """整单价:q 落到哪一档就用哪档的单价。"""
    price = tiers[0][1]
    for upper, p in tiers:
        if q <= upper + 1e-9:
            return p
        price = p
    return price


def marginal_tier_cost(q: float, tiers: Sequence[Tuple[float, float]]) -> float:
    """分段边际价:对 [0, q] 按档积分。"""
    if q <= 0:
        return 0.0
    cost = 0.0
    prev = 0.0
    for upper, price in tiers:
        seg = max(0.0, min(q, upper) - prev)
        cost += seg * price
        prev = upper
        if q <= upper:
            break
    return cost


def marginal_price_at(q: float, cfg: SiteConfig) -> float:
    """用量为 q 时,再增加 1 Mbps 的边际单价(供分配核心做价格信号)。"""
    if cfg.kind == "port":
        return 0.0
    excess = max(0.0, q - cfg.commit_mbps)
    if cfg.tiers is not None:
        if cfg.tier_mode == "whole":
            # 跳档模型不是凸的,用当前档单价做信号,跳档瞬间由外层迭代修正
            return _whole_tier_price(max(q, cfg.commit_mbps), cfg.tiers)
        prev = 0.0
        for upper, price in cfg.tiers:
            if excess <= upper + 1e-9:
                return price
            prev = upper
        return cfg.tiers[-1][1]
    return cfg.unit_price


# --------------------------------------------------------------------------- #
# 月费
# --------------------------------------------------------------------------- #
def fixed_fee(cfg: SiteConfig) -> float:
    if cfg.kind == "port":
        return cfg.port_fee
    return cfg.commit_price * cfg.commit_mbps


def variable_cost(billed: float, cfg: SiteConfig) -> float:
    """超过保底部分的可变费用。"""
    if cfg.kind == "port" or billed <= cfg.commit_mbps + 1e-9:
        return 0.0
    excess = billed - cfg.commit_mbps
    if cfg.tiers is not None:
        if cfg.tier_mode == "whole":
            # 整体落档价 × 全部计费量,保底费已在 fixed_fee 中收取,这里只补差额
            full = _whole_tier_price(billed, cfg.tiers) * billed
            return max(0.0, full - cfg.commit_price * cfg.commit_mbps)
        return marginal_tier_cost(excess, cfg.tiers)
    return cfg.unit_price * excess


def site_monthly_cost(samples: np.ndarray, cfg: SiteConfig) -> float:
    """单个站点一个月的总费用(元)。"""
    if cfg.kind == "port":
        return cfg.port_fee
    billed = billed_quantity(samples, cfg)
    return fixed_fee(cfg) + variable_cost(billed, cfg)


def total_monthly_cost(all_samples: np.ndarray, sites: Sequence[SiteConfig]) -> float:
    """all_samples: (S, T) 各站点全月采样;返回总账单。"""
    return float(sum(
        site_monthly_cost(all_samples[i], sites[i]) for i in range(len(sites))
    ))


# --------------------------------------------------------------------------- #
# 在线辅助:给定已有采样,预估"再放一个高峰点"会不会抬升计费线
# --------------------------------------------------------------------------- #
def current_line_and_free_budget(samples: np.ndarray,
                                 month_slots: int = SLOTS_PER_MONTH,
                                 n_free_total: int = P95_DROP_SAMPLES
                                 ) -> Tuple[float, int]:
    """返回(当前计费线, 本月剩余可免费冲高的点数)。

    全月有 n_free_total 个免费额度(整月 432)。当前已经严格高于计费线的点
    用掉了一部分;剩余点数 = 总额度 - 已用(粗略口径,在线控制器再结合节奏系数)。
    """
    x = np.asarray(samples, dtype=float)
    if x.size == 0:
        return 0.0, n_free_total
    line = p95_line(x)
    used = int(np.count_nonzero(x > line + 1e-9))
    budget_left = n_free_total - used
    return line, max(0, budget_left)
