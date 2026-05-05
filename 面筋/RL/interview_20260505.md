## 日期：20260505

### RL 领域（通用思考题）

**基于近期 RL 前沿进展的综合思考题**

1. [基础概念题] PPO 和 GRPO 的核心区别是什么？为什么 GRPO 在某些场景下比 PPO 更高效？
   - **答案/思考提示**：PPO 使用完整 critic 网络估计价值函数，依赖额外的 value network；GRPO 则通过 group relative ranking 生成 baseline，仅需要 policy network。在reward signal稀疏的场景下，GRPO 避免了 value network 估计偏差的问题，但当 action space 很大或 reward 变化剧烈时，group ranking 可能不够稳定。

2. [深度思考题] 在 RLHF 中，为什么reward model的泛化能力往往比policy model更难保证？这对长文本生成任务有什么启示？
   - **答案/思考提示**：Reward model 通常在有限的人类偏好数据上训练，分布外泛化能力有限。当 policy model 探索到 reward model 未覆盖的 region 时，reward model 的 signal 会变得不可靠。对于长文本生成，这意味着 early-stage 的 KL penalty 需要更保守，防止 policy 偏离 reward model 学到的偏好太远。

3. [实践应用题] 如果你要将 GRPO 应用到机器人控制任务（连续动作空间、高维观测），需要做哪些修改？
   - **答案/思考提示**：① 将离散的 logprob ranking 改为连续的 advantage estimation；② 考虑使用 GAE(λ) 来平衡 bias-variance tradeoff；③ 连续动作空间需要 Gaussian policy 替代 categorical；④ 可能需要 layer normalization 或 spectral normalization 来稳定 training dynamics。

4. [优缺点对比题] FlowGRPO 与传统 REINFORCE 类算法的核心差异是什么？什么时候选择 FlowGRPO 更合适？
   - **答案/思考提示**：FlowGRPO 将 action generation 建模为 Flow 模型，通过 pathwise gradient 估计梯度，而非 likelihood ratio。适合 action space 连续且需要 expressiveness 的场景（如扩散策略）。当已有强大的扩散生成模型时，FlowGRPO 可以复用其 representation；但如果 action space 简单，REINFORCE 计算效率更高。

5. [深度思考题] RL 中 "deadly triad"（function approximation, off-policy learning, bootstrapping）为什么会导致训练不稳定？你认为当前哪些算法设计有效缓解了这一问题？
   - **答案/思考提示**：TD learning 在 off-policy 下会引入 distribution mismatch，导致值函数估计偏差累积。PPO 通过 clipping 限制 policy 变化幅度缓解；SAC 通过 soft update 和 twin Q-network 提供 implicit regularization；Dreamer 系列通过 latent world model 提供 on-policy learning。理论上证明了收敛性但实践中仍需调参。

---
*本份为无新论文日的通用领域思考题，基于近期 RL 进展（PPO/GRPO/FlowGRPO）综合整理
