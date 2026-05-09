## 日期：20260509

### RL 领域

**论文：《Near-Future Policy Optimization (NPO)》**（arXiv:2604.20733）

1. **[基础概念题]** NPO 提出的核心问题是什么？为什么纯 RLVR（Reinforcement Learning with Verifiable Rewards）在训练早期和后期分别会遇到什么瓶颈？
   - **答案/思考提示**：早期：稀疏正确轨迹，大多数 rollout 都是错的，梯度信号几乎为零。后期：rollout 分布坍缩，策略趋于固化，pass@1 提升主要来自对已有解空间的重新分配。NPO 的解决思路是用"未来的自己来教现在的自己"——利用同一训练过程中更靠后的 checkpoint 作为近未来策略提供辅助轨迹。

2. **[深度思考题]** NPO 论文定义了有效学习信号 ℱ(Δ) = Q(Δ)/V(Δ)，其中 Q 和 V 分别代表什么？为什么这个信号会随 Δ 呈现"先升后降"的凹型特征？
   - **答案/思考提示**：Q(Δ) 是信号质量——近未来 checkpoint 在当前策略失败 prompt 上的 pass rate，Δ 越大 checkpoint 越强，Q 越高。V(Δ) 是方差代价——由于 importance weighting 引入的梯度方差，Δ 越大参数漂移越多，V 呈指数增长。凹型是因为 Q 会饱和而 V 指数增长，存在唯一内点最优 Δ*。

3. **[实践应用题]** 如果你要在自己的 RL 训练流程中引入 NPO 机制，请描述具体的实现步骤，包括如何选择 Δ、如何处理 importance weight、以及何时触发 Early Bootstrap 或 Late Plateau Breakthrough？
   - **答案/思考提示**：① 对每个 prompt，若当前 group 准确率低（p̂ ≤ τgate），则将 rollout group 的第 n 个 slot 替换为近未来 checkpoint 产生的已验证正确轨迹；② 由于 π(t+Δ) 是近策略，πθ/π(t+Δ) ≈ 1，IS 修正几乎不需要；③ Early Bootstrap：冷启动时先跑 scout run 取最终 checkpoint 重启训练；④ Late Plateau Breakthrough：监控到 EMA reward 停滞 + entropy 下降时，穿越 plateau 取更强 checkpoint 回滚重放。

4. **[优缺点对比题]** NPO 与 LUFFY（外部教师）、ExGRPO（历史回放）、RLEP（远未来回放）相比，在信号质量和方差代价上各有什么优劣？为什么 NPO 可以省略 importance sampling 修正而 LUFFY 不行？
   - **答案/思考提示**：外部教师 Q 高但 V 高（分布差距大）；历史回放 Q/V 中等但被早期 checkpoint 上限束缚；远未来回放 Q 高但 V 极高（差距爆炸）；NPO 的 Δ 足够小（20-70步），π(t+Δ) 与 πθ 足够近，IS 权重比 ≈ 1，几乎不需要修正。LUFFY 的外部 teacher 分布太远，IS 修正是其训练稳定性的关键。

5. **[深度思考题]** AutoNPO 实现了 NPO 的全自动化，请分析其触发条件（EMA reward 停滞 + entropy 下降）和 Δ* 搜索策略（最大化经验 ℱ(Δ) = Q̂(Δ)/V̂(Δ)）的设计合理性。如果在某些任务中 entropy 下降并不明显，AutoNPO 还能有效工作吗？
   - **答案/思考提示**：触发条件基于"训练曲线走平 + 探索坍缩"这两个 RL 收敛的典型信号。若 entropy 下降不明显，可能说明策略仍在有效探索，此时不需要干预——这正是 AutoNPO 的保守性设计。但如果任务本身 entropy 本身就低（例如答案非常确定的任务），AutoNPO 可能会错过最佳干预时机，需要结合其他信号（如 loss plateau）一起判断。
