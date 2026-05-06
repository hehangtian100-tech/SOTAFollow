## 日期：20260506

### RL 领域（通用思考题）

**基于近期 RL 前沿进展的综合思考题**

1. [基础概念题] On-policy 和 Off-policy 策略梯度算法的核心区别是什么？为什么 PPO 坚持使用 on-policy 更新而 SAC 可以 off-policy 更新？
   - **答案/思考提示**：On-policy 要求 behavior policy 和 target policy 一致，off-policy 则允许不同。PPO 使用 importance sampling 纠正 off-policy 偏差但仍偏向 on-policy 以保证稳定性；SAC 使用 soft Q-learning 框架，通过 twin Q-networks 和 target network 提供 implicit regularization，能容忍 off-policy 数据。PPO 的 clip mechanism 在 on-policy 下更有效；SAC 的 max 操作在 off-policy 下可能导致 overestimation bias 但 soft update 缓解了这个问题。

2. [深度思考题] 在 RLHF 中，reward hacking（奖励黑客）问题产生的根本原因是什么？KL 散度约束为什么能缓解但不能根除这个问题？
   - **答案/思考提示**：Reward hacking 源于 reward model 无法完美刻画人类偏好，且 policy 优化是无限期的。KL 约束限制 policy 偏离 reference model 的程度，但 reward model 的覆盖范围本身是有限的，当 policy 探索到 reward model 未覆盖的 region 时，会出现 reward hacking。真正的 solution 需要更好的人类偏好建模（如 constitutional AI）或多臂老虎机式的对抗训练。

3. [实践应用题] 如果你要训练一个能在多任务间快速切换的 RL agent（如机械臂既会抓取又会放置），你会采用什么架构设计？为什么？
   - **答案/思考提示**：① 使用 modular architecture（如 LMP），不同 skill 用不同 policy，shared world model 提供通用的 dynamics prediction；② 使用 task embedding + policy conditioning，让单一 policy 学习任务相关的 behavior；③ 使用 meta-learning（MAML）让 agent 能快速 adaptation。推荐 modular，因为多任务切换时能复用共享组件且可解释性强。

4. [优缺点对比题] DDPG 和 SAC 都是处理连续动作空间的算法，它们在 action exploration 策略上的核心差异是什么？哪种在高维动作空间表现更好？
   - **答案/思考提示**：DDPG 使用 deterministic policy + noise injection（OU noise 或 epsilon-greedy）进行探索；SAC 使用 stochastic policy + entropy bonus鼓励 exploration。在高维动作空间，DDPG 的 deterministic exploration 可能不够有效（难以覆盖整个 action space），而 SAC 的 entropy-based exploration 更系统化，能更均匀地探索。实践中 SAC 在复杂任务（如 Humanoid）上通常表现更好。

5. [深度思考题] 强化学习中 value function overestimation 的问题来源是什么？Double Q-learning 的解决方案为什么有效但仍然不够完美？
   - **答案/思考提示**：Overestimation 源于 max 操作会选取被高估的 action 的 Q 值，且这种误差会通过 TD learning 传播累积。Double Q-learning 通过解耦 action selection 和 value estimation 来缓解，但由于两个 Q-network 仍然基于相同数据，correlation 仍然存在。更根本的解决方案包括 Clipped Double Q（TD3）和 ensemble 方法，通过多个 Q 值取平均降低 variance。

---
*本份为无新论文日的通用领域思考题，基于近期 RL 进展（PPO/SAC/DDPG/RLHF）综合整理
