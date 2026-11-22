"""带宽成本最优的流量调度系统。

模块说明
--------
billing     : 统一计费模型(95 峰值 / 保底 / 阶梯价 / 包端口)与计费线
simulation  : 三个月流量仿真、DNS 权重滞后、故障事件
allocator   : 给定"计费线"的单周期可行分配核心(LP 等价贪心,NumPy 向量化)
offline     : 离线最优标杆(全知信息的迭代凸松弛 + 多起点)
forecast    : 日内/月剩余流量形状预测
online      : 在线滚动调度(免费额度节奏 + 预测 + 重排)
policies    : 就近调度 / 平均分摊两个对照策略
evaluation  : 账单、质量、免费时段使用率等指标
report      : 纯 SVG 的 HTML 报告(排序采样曲线 + 当前计费线)
"""

SLOTS_PER_HOUR = 12          # 每 5 分钟一个采样点
SLOTS_PER_DAY = 288
SLOTS_PER_MONTH = 8640
P95_DROP_FRACTION = 0.05     # 去掉最高的 5%
P95_DROP_SAMPLES = int(round(SLOTS_PER_MONTH * P95_DROP_FRACTION))  # 432 = 36 小时
P95_BILLABLE_RANK = P95_DROP_SAMPLES + 1                           # 降序第 433 个
