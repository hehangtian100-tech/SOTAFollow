## 日期：20260508

### RL 领域

**通用领域思考题（无当日新论文）**

1. [基础概念题] 请解释 PPO 算法中 Clipped Surrogate Objective 的作用，并说明为什么要对策略更新幅度进行限制？
   - 答案/思考提示：PPO 通过 clip(r_t(θ), 1-ε, 1+ε) 限制新旧策略概率比 r_t 的变化范围，避免单步更新过大导致策略崩溃。这解决了 TRPO 需要复杂二阶优化的核心痛点。

2. [深度思考题] 为什么 GRPO/DeepSeek-R1 等 LLM RL 方法采用 group relative advantage，而不像传统 RL 那样使用 GAEs(λ) 计算优势函数？
   - 答案/思考提示：语言任务中单个 prompt 可生成多个 answer，难以准确估计 value function。Group relative advantage 只需同 group 内比较，不需要 learnable critic，降低了方差和训练复杂度。

3. [实践应用题] 如果你要将 PPO 应用于机器人控制任务（低维连续动作空间），有哪些超参数需要特别调整？
   - 答案/思考提示：① learning rate 需要更小（1e-4~1e-5）；② clip epsilon 可适当增大到 0.2；③ value function 的重要性提升，需单独调 value loss weight；④ entropy bonus 有助于探索；⑤ observation normalization 更关键。

4. [优缺点对比题] 对比 on-policy（PPO）和 off-policy（Q-learning/DQN）方法在 sample efficiency 和训练稳定性上的差异。
   - 答案/思考提示：Off-policy 更高 sample efficiency 但可能不稳定（experience replay 引入 bias）；On-policy 更稳定但 sample efficiency 低。PPO 通过 importance sampling 和 clip 尝试兼顾两者。

5. [深度思考题] 在 RLHF 中，reward model 往往会遇到 reward hacking 问题（模型找到获得高奖励但实际质量差的 trick）。有哪些缓解措施？
   - 答案/思考提示：① KL 散度约束策略接近 reference model；② 对抗训练（reward model ensemble）；③ 更精细的 reward shaping；④ 人类反馈多样化；⑤ 后训练评估结合多个维度（Helpful/Harmless/Honest）。
