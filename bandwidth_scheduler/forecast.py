"""在线流量预测:只使用已经发生的采样数据。

思路(纯 NumPy,可解释、可热替换成更强的模型)
---------------------------------------------
1. 日内形状:按"星期几 × 日内 288 槽"维护 EWMA 形状表,新观测逐槽更新;
2. 水平:最近几天同时段均值做水平修正,捕捉月增长;
3. 月剩余形状:用星期几模板 + 日历位置外推整月;
4. 不确定性:用近期相对误差的标准差给出预测区间,供在线节奏控制使用。

爆款尖峰不可预测,是本模块误差的主要来源——评测时专门挑尖峰日看表现。
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import numpy as np

from . import SLOTS_PER_DAY, SLOTS_PER_MONTH


class Forecast(NamedTuple):
    today: np.ndarray            # (R, 剩余槽数) 今天剩余时刻的预测
    month_remaining: np.ndarray  # (R, 剩余槽数) 本月剩余全部时刻的预测
    rel_sigma: float             # 近期相对预测误差标准差
    today_peak: float            # 今日预测峰值(汇总地区)


class DemandForecaster:
    def __init__(self, n_regions: int, slots_per_month: int = SLOTS_PER_MONTH,
                 ewma_alpha: float = 0.6, warmup_days: int = 1):
        self.n_r = n_regions
        self.spm = slots_per_month
        self.alpha = ewma_alpha
        self.warmup = warmup_days * SLOTS_PER_DAY
        # 星期几 × 日内槽 的形状模板(用水平归一化后的形状)
        self.templates = np.full((7, SLOTS_PER_DAY, n_regions), np.nan)
        self.level = np.ones(n_regions)          # 近期日均水平
        self._seeded = False
        self._day_seeded = False
        self.history: list = []                  # [相对误差] 用于误差统计
        self._last_weekday_obs = [[] for _ in range(7)]

    def seed_level(self, first_obs: np.ndarray) -> None:
        """用第一个真实观测初始化水平,避免冷启动 level=1 导致的爆炸误差。"""
        if not self._seeded:
            self.level = np.maximum(np.asarray(first_obs, float), 1e-6)
            self._seeded = True

    # ------------------------------------------------------------------ #
    def update(self, slot_in_month: int, actual_vec: np.ndarray,
               pred_vec: Optional[np.ndarray] = None) -> None:
        """每个时刻用真实观测更新模板。"""
        dow = (slot_in_month // SLOTS_PER_DAY) % 7
        tod = slot_in_month % SLOTS_PER_DAY
        if pred_vec is not None:
            # 标准 MAPE:以真实值为分母,避免预测值趋零时误差爆炸
            denom = np.maximum(np.abs(actual_vec), 1e-6)
            self.history.append(float(np.mean(
                np.abs(actual_vec - pred_vec) / denom)))
            if len(self.history) > SLOTS_PER_DAY * 7:
                self.history = self.history[-(SLOTS_PER_DAY * 7):]
        if not self._seeded:
            self.seed_level(actual_vec)
        cur = self.templates[dow, tod]
        if np.isnan(cur).any():
            self.templates[dow, tod] = actual_vec
        else:
            self.templates[dow, tod] = (1 - self.alpha) * cur + self.alpha * actual_vec
    def end_of_day_update(self, day_obs: np.ndarray) -> None:
        """每天结束用当天总量更新水平。day_obs: (R, 288)。"""
        day_mean = day_obs.mean(axis=1)
        # 第一个完整日直接采用真实日均值,避免从初始 level=1 缓慢爬升
        if not self._day_seeded:
            self.level = day_mean
            self._day_seeded = True
        else:
            self.level = 0.8 * self.level + 0.2 * day_mean

    # ------------------------------------------------------------------ #
    def _slots_matrix(self, slots: np.ndarray) -> np.ndarray:
        """给定一组月内槽序号,一次性向量化返回 (R, len(slots)) 预测。"""
        if slots.size == 0:
            return np.empty((self.n_r, 0))
        dow = (slots // SLOTS_PER_DAY) % 7
        tod = slots % SLOTS_PER_DAY
        templ = self.templates[dow, tod]                    # (H, R)
        nan_rows = np.isnan(templ).any(axis=1)
        if nan_rows.any():
            tod_lookup = tod[nan_rows]
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fill = np.nanmean(self.templates[:, tod_lookup, :], axis=0)
                global_mean = np.nanmean(
                    self.templates.reshape(-1, self.n_r), axis=0)
            if np.isnan(global_mean).any():
                global_mean = np.broadcast_to(
                    self.level, (self.n_r,)).copy()
            fill = np.where(np.isnan(fill), global_mean, fill)
            templ[nan_rows] = fill
        # 模板存的是各 (星期几, 日内槽) 的 EWMA 绝对量级,直接用其日内形状。
        # trend 必须同口径比较:近期日均 level / 该星期几模板的"全天"均值
        # (而非单槽值,否则峰值槽会被错误压向日均)。逐地区、逐 dow 计算。
        import warnings
        out = np.empty_like(templ)                              # (H, R)
        for d in np.unique(dow):
            idx = np.where(dow == d)[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                day_mean = np.nanmean(self.templates[d], axis=0)   # (R,)
            day_mean = np.where(np.isnan(day_mean), self.level, day_mean)
            trend = np.clip(self.level / np.maximum(day_mean, 1e-9),
                            0.5, 2.0)
            out[idx] = templ[idx] * trend[None, :]
        return out.T

    def forecast_remaining(self, current_slot: int) -> Forecast:
        """返回今天剩余与本月剩余的预测(含当前槽)。"""
        start_today = (current_slot // SLOTS_PER_DAY) * SLOTS_PER_DAY
        horizon_today = start_today + SLOTS_PER_DAY - current_slot
        horizon_month = self.spm - current_slot
        today = self._slots_matrix(
            np.arange(current_slot, current_slot + horizon_today))
        month = self._slots_matrix(
            np.arange(current_slot, current_slot + horizon_month))
        rel_sigma = float(np.std(self.history[-SLOTS_PER_DAY:])) if self.history else 0.15
        return Forecast(today, month, max(rel_sigma, 0.05),
                        float(today.sum(axis=0).max()))

    def rel_error_sigma(self) -> float:
        if not self.history:
            return 0.15
        return max(float(np.std(self.history[-SLOTS_PER_DAY:])), 0.05)
