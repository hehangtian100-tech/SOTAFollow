## 日期：20260515

### RL 领域
**通用领域思考题**

1. [基础概念题] 在 RL 中，Value Function 和 Q Function 的区别是什么？为什么在连续动作空间中直接学习 Q Function 往往不可行？
   - 答案/思考提示：V(s) 衡量在给定状态下的期望累计回报，Q(s,a) 衡量在给定状态执行特定动作的期望累计回报。连续动作空间中 Q(s,a) 的动作维度是连续的，无法用表格存储；此外直接对每个动作求 argmax 需要遍历无限动作空间。解决方案包括：① 使用确定性策略直接输出动作（如 DDPG）；② 使用随机策略配合均值/方差参数化；③ 使用基于路径积分的方法（如 PI^2）。

2. [深度思考题] Model-Based RL（如 Dreamer、World Models）相比 Model-Free RL 在样本效率上有显著优势，但其预测模型本身存在模型误差（model error）问题。这个误差会如何影响长期规划和策略学习？当前有哪些方法用于缓解这个问题？
   - 答案/思考提示：模型误差在时序展开中会累积，导致 imagination horizon 拉长时预测偏离真实环境。缓解方法：① latent space 预测（而非像素级）如 Dreamer/JEPA；② ensembles 或 dropouts 增加模型多样性；③ 保守预测（conservative planning）限制想象步数或置信区间；④ 重补贴（revisit）机制，当误差增大时回到真实环境；⑤ 世界模型蒸温（DREAMERpro 等）。

3. [实践应用题] 假设你需要训练一个机械臂抓取策略，但仅有 1000 条人类演示轨迹，且奖励函数难以设计。你会采用什么方法组合来实现高效学习？请给出具体的技术路线。
   - 答案/思考提示：① 首先用 GAIL/IRL 从演示中推断奖励函数或直接用行为克隆（BC）初始化策略；② 结合 HER 将失败轨迹的目标重标记为成功；③ 辅助动作捕捉（.Action chunks）降低动作维度和序列建模难度；④ 在模拟环境中用 Domain Randomization 提升泛化；⑤ 如果有稀疏奖励，用好奇心驱动（intrinsic motivation）探索。推荐组合：BC + HER + 领域随机化。

4. [优缺点对比题] HPG（Hierarchical Policy Gradient）和 Option-Critic 框架都涉及层级强化学习，但它们在"如何定义高层动作"上有本质差异。这两种方法各自的优缺点是什么？在什么任务中层级方法收益最大？
   - 答案/思考提示：HPG 通常有预定义的高层动作/技能，通过 extrinsic reward 或 intrinsic motivation 学习切换；Option-Critic 则通过内部 critic 端到端学习 option 的终止条件和策略。HPG 优势：可解释性强、技能可复用；劣势：需要手工设计或独立发现技能。Option-Critic 优势：端到端可微、可以学习何时切换；劣势：训练不稳定、option 数量敏感。收益最大的任务：长期稀疏奖励任务（导航+操作组合）、多尺度时间抽象、部分可观测环境。

5. [基础概念题] 策略梯度方法（如 REINFORCE、PPO）和 Q-Learning 方法（如 DQN、SAC）的根本区别是什么？它们各自在什么场景下表现更好？
   - 答案/思考提示：策略梯度直接参数化策略并沿梯度方向优化预期回报，适合连续动作空间和高方差场景，但 sample inefficiency（每次更新需要新样本）；Q-Learning 隐式学习最优动作选择，适合离散动作空间，可以 off-policy 利用历史数据。策略梯度更适合连续控制、稀疏奖励、需要随机策略的任务；Q-Learning 更适合样本可快速获取的离散控制任务。
