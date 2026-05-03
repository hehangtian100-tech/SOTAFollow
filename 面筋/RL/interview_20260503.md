## 日期：20260503

### RL 领域

**通用思考题（当日无新论文）**

1. [基础概念题] GRPO（Group Relative Policy Optimization）和 PPO 在策略更新的采样方式上有什么本质区别？为什么 GRPO 能够在不引入 critic 网络的情况下实现稳定的策略更新？
   - **答案/思考提示**：PPO 需要从当前策略采样 trajectory，然后用 value function（critic）估计 advantage；GRPO 则是对同一个 state 采样多个 action，计算这些 action 相对于 group 均值的相对优势，省去了独立的 critic 网络。具体来说：GRPO 先对每个 state 采样 G 个 action，计算每个 action 的reward，再将每个action的reward归一化（减去均值除以标准差）得到 advantage。核心洞察是：当 group size 足够大时，group 内的相对排序足够稳定，可以近似替代 baseline 的作用。这实际上是一种 "self-critic" 思想——用采样得到的 group 统计量代替 learnable baseline。

2. [深度思考题] 在 RLHF 流程中，为什么通常先用 SFT 模型初始化策略，而不是从随机策略开始训练？SFT 阶段和 RLHF 阶段各自解决什么问题？为什么不能只靠 RLHF 从头训练？
   - **答案/思考提示**：SFT（Supervised Fine-Tuning）阶段的核心作用是提供良好的初始化。具体：①将预训练语言模型的能力（知识、语法、常识）引导到任务特定的行为模式；②让策略学会「如何生成合理的响应」，而不是从随机探索开始学习语言本身。RLHF 解决的问题是「根据人类偏好调整策略」——让模型生成更符合人类期望的内容。两者分工：SFT 解决「说什么」，RLHF 解决「怎么说更好」。如果直接从随机策略开始 RLHF，策略需要同时学习语言能力（几乎不可能）和偏好对齐，样本复杂度会爆炸。

3. [深度思考题] 在线策略（on-policy）算法（如 PPO）和离线策略（off-policy）算法（如 DQN、SAC）在探索-利用（exploration-exploitation）权衡上的策略有何不同？为什么说 offline RL 的核心挑战是分布偏移（distribution shift）？
   - **答案/思考提示**：On-policy 算法如 PPO 只能用当前策略采样的数据进行更新，每次策略变化后都需要重新采样，这自然地限制了策略偏离数据分布的程度。Off-policy 算法如 SAC 可以复用历史数据，但需要用 importance sampling 修正分布偏移。Offline RL（完全用固定数据集训练，不与环境交互）的核心问题是：数据收集策略 π_beta 和学习策略 π_θ 不同，当 π_θ 偏离 π_beta 时，用 OOD（out-of-distribution）动作的 Q 值估计会严重高估。解决方案包括：①约束 π_θ 不要偏离 π_beta 太远（cQL、TD3+BC）；②学习悲观 Q 函数（CQL）；③Implicit 保守正则化。

4. [实践应用题] 你在开发一个机械臂的精确抓取控制器，已知抓取任务的成功率与末端执行器的位置精度、物体识别精度、力控精度都相关。请设计一个奖励函数，使得 RL 策略能够在以下约束下优化：
   - 抓取成功率 > 95%
   - 平均执行时间 < 2秒
   - 碰撞率 < 1%
   
   请说明奖励塑形（reward shaping）时如何平衡各子目标，以及稀疏奖励（sparse reward）vs 稠密奖励（dense reward）的选择考量。
   - **答案/思考提示**：
     - **奖励函数设计**：可分层次设计——①最终奖励：成功抓取 +1，失败 -0.5，碰撞 -1；②中间奖励（稠密奖励）：末端执行器接近目标物体的距离奖励（-d/d_max），力控稳定性奖励（力传感器方差低），执行时间惩罚（-t/T_max）；③辅助奖励：碰撞预测奖励（接近障碍物时减速）。
     - **平衡策略**：用加权求和 r = r_success + λ1*r_precision + λ2*r_time + λ3*r_collision，可通过调整权重来平衡各目标；或使用多目标 RL（Pareto 最优）而非单一标量奖励。
     - **稀疏 vs 稠密**：稀疏奖励更简单、不易引入误奖励，但对复杂任务探索难度大；稠密奖励能加速收敛，但需要精心设计避免 reward hacking。对于高精度要求的任务，建议先用稠密奖励快速学习，再用稀疏奖励微调最终行为。

5. [优缺点对比题] 对比 GRPO、DPO、PPO、TRPO 等策略优化算法，在以下维度上各自表现如何：①超参数敏感性；②样本效率；③收敛稳定性；④对动作空间大小的适用性；⑤实现复杂度。请分析为什么当前 LLM alignment 领域 DPO/GRPO 比 PPO 更流行。
   - **答案/思考提示**：
     | 算法 | 超参敏感性 | 样本效率 | 收敛稳定性 | 动作空间 | 实现复杂度 |
     |------|-----------|---------|-----------|---------|-----------|
     | PPO  | 中（clip ratio ε）| 中（需critic）| 较好 | 大（连续/离散）| 中 |
     | TRPO  | 低（trust region）| 中 | 较好 | 大 | 高（需共轭梯度）|
     | DPO  | 低 | 高（off-policy）| 对数据分布敏感 | 离散文本token | 低 |
     | GRPO | 低 | 高（group baseline）| 好 | 离散token | 低 |
     
     **DPO/GRPO 流行的原因**：①LLM alignment 本质上是离散 token 序列的优化，动作空间巨大但结构化（token 序列）；②PPO 训练需要同时维护 actor、critic、reward model 三个网络，复杂度高；③DPO/GRPO 通过对比损失绕过 reward model，直接用偏好数据优化；④LLM 场景下数据获取成本低（可以大量标注偏好），off-policy 的样本效率优势不明显；⑤PPO 的 clipped objective 在 token-level 优化时效果不如 sequence-level 的对比损失直接。
