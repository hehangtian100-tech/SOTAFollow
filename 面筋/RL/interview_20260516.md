## 日期：20260516

### RL 领域

**论文：** 无新论文，基于 RL 领域核心论文综合整理

**基础概念题：**

1. **PPO 中的 Clipped Surrogate Objective 解决了什么问题？为什么剪裁能够防止策略越界？**
   - 答案/思考提示：PPO 通过限制策略更新比例避免过大的策略跳跃。关键洞察是 TRPO 使用 KL 散度约束而 PPO 使用一阶优化+剪裁，剪裁使得即便 KL 约束未被显式满足，策略也不会偏离太远。

2. **GRPO 与 PPO 的核心区别是什么？GRPO 为什么在大模型对齐中更受欢迎？**
   - 答案/思考提示：GRPO 取消 value network，用 group relative baseline 估计 advantage，大幅减少约 50% 参数。适合大模型场景因为无需训练 critic，且 group sampling 可利用模型自身生成的多样性。

3. **近端策略优化的"近端"二字体现在哪里？与 Trust Region Methods 有何联系？**
   - 答案/思考提示：近端体现在用 clippd objective 限制策略更新步长，本质上是 Trust Region 的近似但通过一阶优化实现。TRPO 是二阶优化且计算量大，PPO 是一阶但通过裁剪隐式约束信任域。

4. **RAD（Rugged Aesthetic Design）增强学习为什么强调 Random Dilation 作为数据增强的核心？**
   - 答案/思考提示：视觉 RL 中图像 augment 是关键，Random Dilation 在不改变语义的前提下增强鲁棒性，比 rotation/crop 更适合机械臂控制任务，因为末端执行器姿态变化有限。

5. **FlowGRPO 将 flow matching 引入 RL 的动机是什么？与 score-based RL 相比有何优势？**
   - 答案/思考提示：Flow matching 提供确定性轨迹建模，比 stochastic diffusion 更适合决策问题。可直接预测动作方向而非去噪，训练更稳定、推理更快。

**深度思考题：**

6. **为什么说 PPO 是 on-policy 算法但实际训练中经常使用 off-policy 技巧（如 replay buffer）？如何理解 PPO 的 on/off-policy 本质？**
   - 答案/思考提示：PPO 要求数据由当前策略生成，但重要性采样允许一定程度的 off-policy。关键是 clip 限制了 off-policy 的影响，使得即便使用历史数据也能训练。核心是约束新旧策略的比率不要太大。

7. **在 RLHF 中，reward model 的训练质量直接决定最终效果。如何设计 reward model 来避免 reward hacking 和 reward collapse？**
   - 答案/思考提示：reward hacking 发生在 agent 找到 reward 函数的漏洞；reward collapse 是 reward 都趋向于某个值。解决方案包括：ensemble of reward models、rank-based reward、process-based reward vs outcome-based reward。关键是 reward model 要比 policy 慢更新。

8. **从信息论角度解释，为什么 KL 散度可以作为策略更新的约束？KL 约束与最大熵 RL 之间有什么联系？**
   - 答案/思考提示：KL 散度衡量两个分布的差异，用作约束是限制信息增益。最大熵 RL 的目标是 maximize expected return + entropy，和 KL 约束都旨在平衡 exploitation 和 exploration。Soft Q-Learning 可以看作 KL-constrained RL 的特例。

**实践应用题：**

9. **假设你要在机器人操作任务中用 RL，让机械臂将不同形状的物体放入对应凹槽中。请设计你的 reward shaping 方案，并说明如何处理物体形状/颜色变化的泛化问题。**
   - 答案/思考提示：sparse reward + curriculum learning。先用 shaped reward 训练（距离、接触），逐渐过渡到 sparse reward。可考虑用 goal-conditioned RL、domain randomization、或是用 vision-based reward 让 agent 学会识别目标。

10. **如果 PPO 在你的任务中 sample efficiency 太低，有哪些改进方向？请至少提出 3 种方案并分析优劣。**
    - 答案/思考提示：① 改用 off-policy 算法（PPO + importance sampling 调参）；② 增加 parallel environments 数量；③ 引入 reward shaping 或 HRL；④ 使用 imitation learning 预训练；⑤ 改用decision transformer 等 sequence model 范式。

**优缺点对比题：**

11. **对比 PPO、SAC、TD3 三种算法在连续控制任务中的适用场景。**
    - 答案/思考提示：
      - PPO：稳定、鲁棒，适合所有场景，是 baseline 首选
      - SAC：最大熵框架，探索性强，适合需要多样性的任务
      - TD3：Twin critic + delayed policy update，适合需要避免 overestimate 的任务
    - SAC 比 PPO sample efficiency 高但调参更敏感；TD3 比 DDPG 稳定很多。

12. **对比 on-policy（PPO）与 off-policy（DDPG/SAC）算法在深度 RL 中的根本性差异与各自适用场景。**
    - 答案/思考提示：
      - On-policy：数据必须由当前策略生成，sample efficiency 低但训练稳定
      - Off-policy：可以用任意数据，sample efficiency 高但训练可能不稳定（尤其 deep RL 中）
      - 根本原因是 Q-function 的 learning target 依赖于当前 policy，off-policy 下会导致 correlation 问题。
      - 大模型 alignment 中 GRPO 受欢迎正是因为不需要单独的 critic，reduces 50% 参数。

---

**参考文献**：
- PPO: https://arxiv.org/abs/1707.06347
- GRPO: DeepSeekMath
- SAC: https://arxiv.org/abs/1801.01290
- RAD: https://arxiv.org/abs/2010.13628
