## 日期：20260502

### RL 领域

**通用思考题（当日无新论文）**

1. [基础概念题] PPO（Proximal Policy Optimization）算法中，Clipped Surrogate Objective 的设计是为了解决什么问题？请解释 KL 散度约束与 Clip 机制的区别与联系。
   - **答案/思考提示**：PPO 的 Clip 机制旨在解决策略梯度方法中步长过大的问题——如果一次更新让策略变化太大，训练会不稳定甚至崩溃。具体通过将策略比率 r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t) 限制在 [1-ε, 1+ε] 范围内，防止极端策略更新。KL 散度约束通过直接限制新旧策略的 KL 散度上界来控制步长，PPO-Penalty 用拉格朗日项加入目标函数，PPO-Clip 则通过 Clip 操作隐式地实现了类似效果。相比之下，Clip 更简单高效，不需要显式计算 KL 散度。

2. [深度思考题] 在 RLHF（Reinforcement Learning from Human Feedback）中，Reward Model 的质量直接决定了最终策略的效果。为什么用 Reward Model 而非直接用人类偏好训练策略？Reward Model 存在哪些潜在问题，以及如何缓解？
   - **答案/思考提示**：Reward Model 本质上是一个偏好分布的压缩表示，允许在任意状态下查询任意动作的相对偏好，而无需每次都请人标注。具体优势包括：可复用（一次训练多次使用）、可微（支持梯度反传）、可泛化（学习到偏好背后的隐含规律）。潜在问题包括：①Reward hacking（策略学会「骗过」RM 而非真正完成目标）；②RM 分布外泛化差（对未见过的状态-动作对预测不准）；③RM 本身可能存在偏见。缓解方法包括：Diverse reward modeling、KL 约束限制策略偏离 SFT 模型、对抗性训练等。

3. [实践应用题] 你正在开发一个四足机器人的 locomotion 控制器，已有一个基于 PPO 训练的策略，但发现机器人在不平整地面上容易摔倒。请列出可能的原因以及对应的算法改进方向。
   - **答案/思考提示**：可能原因包括：①训练分布不够多样（只用了平整地面）；②奖励函数中对平衡/稳定性的激励不足；③观测信息缺少触觉或地面不平整度等关键信号。改进方向：①Domain Randomization：在训练时随机引入地面高度变化、摩擦系数变化等；②奖励塑形：增加对躯干姿态稳定性、支撑多边形面积、摆动腿协调性等的奖励；③课程学习：从平整地面逐步过渡到复杂地形；④加入特权信息（如地形先验）作为 critic 的额外输入。

4. [优缺点对比题] 比较 Model-based RL（如 Dreamer、World Models）和 Model-free RL（如 PPO、SAC）在 sample efficiency 和 asymptotic performance 上的 trade-off。为什么机器人控制任务通常更关注 sample efficiency？
   - **答案/思考提示**：Model-based RL 通过学习世界模型来模拟环境交互，通常 sample efficiency 更高（因为可以在模型内部进行大量想象 rollout），但 asymptotic performance 受限于模型误差累积；Model-free RL 直接从真实交互学习，asymptotic performance 上限更高，但需要大量真实样本。机器人控制任务通常 physical sample efficiency 极低（真实机器人交互成本高、磨损大），且任务边界相对清晰（模型误差更容易控制），因此 model-based 方法更受青睐。但当任务过于复杂、状态空间巨大或奖励函数设计困难时，model-free 方法的灵活性和最终性能上限可能更有优势。
