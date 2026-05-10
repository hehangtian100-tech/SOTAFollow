## 日期：20260510

### RL 领域

**论文：《Flow-GRPO: Training Flow Matching Models via Online RL》**（arXiv:2505.05470）

1. **[基础概念题]** Flow Matching（如 Rectified Flow）的概率流 ODE 与 DDPM 的随机微分方程有什么本质区别？为什么这种区别导致 Flow Matching 无法直接应用 GRPO 等在线 RL 算法？
   - **答案/思考提示**：DDPM 使用随机 SDE，每步都有随机噪声注入，天然具有采样多样性；Flow Matching 使用确定性 ODE，从同一噪声出发只能产生唯一轨迹，无法计算 importance sampling ratio。GRPO 依赖概率比 $\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 计算优势函数，这要求能计算每个动作的概率——确定性采样无法满足。

2. **[深度思考题]** FlowGRPO 提出的 ODE-to-SDE 转换中，marginal-preserving reverse-time SDE 的三项（确定性漂移、Itô 修正项、Wiener 扩散项）各自承担什么作用？为什么必须保留 Itô 修正项而不能只加随机噪声？
   - **答案/思考提示**：① 确定性漂移 $\mathbf{v}_t(\mathbf{x}_t)dt$ 沿原 ODE 推进；② Itô 修正项 $-\frac{\sigma_t^2}{2}\nabla\log p_t(\mathbf{x}_t)dt$ 保证 reverse-time SDE 的 marginal distribution 与原 ODE 一致（不变）；③ Wiener 扩散项引入随机探索。只加噪声会改变生成分布，加上 Itô 修正项才能在保持 marginal 不变的前提下引入可控随机性。

3. **[实践应用题]** FlowGRPO 的 Denoising Reduction 策略允许用 10 步采样训练、40 步推理，这对实际部署有什么意义？请分析为什么减少训练步数不会显著损害模型性能？
   - **答案/思考提示**：意义：大幅降低 RL 数据收集的计算成本（10步 vs 40步，采样速度提升 4 倍）。不减损性能的原因：① 训练目标是学习速度场 $\mathbf{v}_\theta(\mathbf{x}_t, t)$，10 步已经能覆盖从噪声到数据的主要分布迁移；② 推理时用完整 40 步是生成质量保障，训练只需学到正确的向量场方向，不需要在推理分布上完全对齐。

4. **[优缺点对比题]** 与 DDPO（用额外网络估计 score function）相比，FlowGRPO 的 score function 闭式推导有什么优势？这种方式是否存在局限性？
   - **答案/思考提示**：优势：无需额外网络，利用 Rectified Flow 的线性结构 $\mathbf{x}_t=(1-t)\mathbf{x}_0+t\mathbf{x}_1$ 直接推导 score function 解析形式，计算效率高、无估计误差。局限性：闭式推导依赖 Rectified Flow 的线性插值假设，其他 Flow Matching 变体（如余弦调度）可能无法直接应用，需要重新推导。

5. **[深度思考题]** FlowGRPO 在文生图任务上取得了显著提升（GenEval: 63%→95%），但几乎无 reward hacking。请分析为什么 Flow Matching 模型比 Diffusion 模型更不容易出现 reward hacking？这与生成模型的特性有什么关系？
   - **答案/思考提示**：Reward hacking 发生在模型过度优化奖励信号而忽略其他质量维度。Flow Matching 的确定性采样+短步数推理使其更难\"找到奖励函数的漏洞\"——轨迹空间更受约束。Diffusion 模型的高多样性采样反而给 reward hacking 更多可乘之机。另外 FlowGRPO 的 KL 散度闭式形式提供了 PPO-style 的 clipped objective，对策略更新幅度有更精细的控制。
