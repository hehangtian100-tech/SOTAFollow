## 日期：20260507

### RL 领域（通用思考题）

**基于近期 RL 前沿进展的综合思考题——策略优化与价值估计专题**

1. [基础概念题] REINFORCE 和 PPO 都属于策略梯度算法，但 PPO 通过什么机制避免了 REINFORCE 的高方差问题？TRPO 和 PPO 的信任域约束在实现上有什么核心差异？
   - **答案/思考提示**：REINFORCE 使用 on-policy 采样通过蒙特卡洛估计 return，高方差来自轨迹级别的随机性。PPO 通过 clipped surrogate objective 限制策略更新幅度，同时使用 GAE（Generalized Advantage Estimation）降低方差——GAE 通过 TD 残差的指数加权综合了 low-variance TD 估计和 low-bias MC 估计。TRPO 使用 KL 散度约束（硬约束）通过共轭梯度法求解，PPO 使用一阶优化+clip，TRPO 计算开销大但理论基础更严格，PPO 实际中更易用。

2. [深度思考题] Soft Actor-Critic (SAC) 的最大熵框架和标准 RL 的最大期望回报框架在数学形式上是什么关系？为什么说最大熵目标既是 exploration bonus 又是 implicit regularization？
   - **答案/思考提示**：标准 RL 优化 `J(π) = E[∑r]`；SAC 优化 `J(π) = E[∑r + αH(π)]`。数学上，最大熵目标等价于在原奖励上添加熵项，这可以被解读为：① exploration——熵项鼓励策略保持随机，覆盖更多状态；② regularization——防止策略过早收敛到确定性解，缓解 overestimation 和 reward hacking。温度系数 α 控制 exploration-exploitation trade-off，论文中使用自动温度调整（entropy constraint）而非固定 α。

3. [实践应用题] 如果你要在稀疏奖励的机器人任务中训练 RL 策略，而环境无法提供 shaped reward，你会采用哪些技术来缓解稀疏奖励问题？至少列举 3 种并说明原理。
   - **答案/思考提示**：① Hindsight Experience Replay (HER)——将失败轨迹中实际达到的目标重新标记为目标，使每个失败都变成"成功"，解决 sparse reward 下的 credit assignment；② Intrinsic Motivation（如 ICM、RND）——添加 curiosity-driven bonus，鼓励探索新状态；③ Goal-conditioned RL——将目标作为策略输入，用 goal relabeling 批量生成正样本；④ Reward Shaping（如果可做）——用 potential-based shaping 保证最优策略不变；⑤ Hierarchy RL——用高层的 subgoal 提供密集的伪奖励。HER 最常用且几乎不增加计算开销。

4. [优缺点对比题] Model-Based RL（如 Dreamer、PlaNet）和 Model-Free RL（如 PPO、SAC）在 sample efficiency 和 asymptotic performance 上各有什么优劣？在机器人控制场景中应该如何选择？
   - **答案/思考提示**：Model-Based RL 样本效率高（学到一个 world model 后可以无限 rollout），但 world model 的 compounding error 会限制 asymptotic performance；Model-Free 样本效率低但最终性能更强（不受模型误差累积影响）。在机器人控制中：如果真实机器人交互成本高（真实机器人），选 Model-Based（如 Dreamer）省样本；如果仿真到真实迁移（sim-to-real），可用 Model-Free 追求极限性能；实际中 hybrid 方法（如 AlphaZero 的 model + search）往往最有效。Being-H0.7、Fast-WAM 等 world model 工作正在改变这个权衡。

5. [深度思考题] 在 RLHF（基于人类反馈的强化学习）中，reward model 的训练数据通常来自人类比较标注。为什么用比较数据而不是绝对评分训练 reward model？reward model 的 quality 对最终 policy 有什么决定性影响？
   - **答案/思考提示**：人类比较比绝对评分更 reliable——人类对相对偏好判断更一致，避免绝对量化的主观偏差（"这个回答 7 分还是 8 分？"）。Bradley-Terry 模型将比较数据转化为 reward difference 的概率建模，是标准做法。Reward model quality 决定最终 policy 上限：如果 reward model 对某些输入覆盖不足，policy 会找到 reward hacking 的漏洞；如果 reward model 过度平滑，会限制 policy 的精细度。这导致 RLHF 训练需要 iterative red teaming 和 reward model ensembling 来提高覆盖度。
