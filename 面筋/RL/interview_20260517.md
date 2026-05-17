## 日期：20260517

### RL 领域

**主题：Policy Optimization 系列算法演进与对比**

1. **[基础概念题]** PPO 使用 CLIP 机制限制策略更新幅度的动机是什么？相比 TRPO 的 Trust Region 方法，CLIP 机制在工程实现上有何优势？
   - **答案/思考提示**：CLIP 通过直接裁剪概率比来防止策略更新过大，避免了 TRPO 需要二阶优化计算 Hessian 矩阵的昂贵代价。核心思想是用一种简单的代理目标来近似 KL 约束，在样本效率和计算效率之间取得平衡。

2. **[深度思考题]** GRPO 用 Group-relative sampling 替代 Value Model，大幅降低训练成本。但 value model 的优势在于能提供相对稳定的 baseline 来控制方差。请分析：在什么场景下，GRPO 的方差会成为瓶颈？如何在不引入完整 value model 的情况下缓解这一问题？
   - **答案/思考提示**：当任务的奖励信号稀疏或延迟较长时，方差问题更显著。可以考虑引入部分 value baseline（如 last-visit baseline）、返回值分解（Return Decomposition）、或使用更宽的采样组来平滑方差。

3. **[实践应用题]** 假设你在训练一个机械臂抓取任务的 RL 策略，发现 PPO 在初期探索时容易陷入局部最优。请设计一个结合 GRPO 和 PPO 思想的混合算法，既能保持探索多样性，又能享受 PPO 的稳定性。
   - **答案/思考提示**：可以采用 GRPO 的 group-relative advantage 估计来引导探索，同时用 PPO 的 CLIP 机制来约束策略更新幅度。或者设计课程学习策略：早期用 GRPO 快速探索，后期切回 PPO 精调。

4. **[优缺点对比题]** 比较 DAPO（Decoupled Clip + Dynamic Sampling）与标准 PPO/GRPO 的核心差异。Dynamic Sampling 解决了什么问题？这种"选择性忽略低优势样本"的策略是否可能带来引入偏差的风险？
   - **答案/思考提示**：DAPO 解决的是 over-exploration 问题——模型在低价值区域浪费了太多更新。动态采样会跳过低优势样本，但也可能错过有价值的新发现。需要验证采样策略是否在理论上满足 importance sampling 的无偏性条件。

5. **[综合思考题]** 从 PPO → GRPO → DAPO 的演进路径，思考 Policy Optimization 算法的设计哲学是否正在从"约束优化"向"选择性优化"转变？这种转变背后的驱动力是什么？
   - **答案/思考提示**：约束优化（PPO/TRPO）强调"不做什么"，选择性优化（GRPO/DAPO）强调"专注做什么"。背后驱动力包括：大规模模型对计算效率的极致追求、合成数据场景下分布偏移问题相对可控、简单性优先（Occam's Razor）的工程哲学回归。
