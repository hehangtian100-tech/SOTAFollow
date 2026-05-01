## 日期：20260501

### RL 领域

**说明：当日无新论文，基于近期 RL 领域进展（XXPO、PPO、FlowGRPO、DAP等）生成通用领域思考题**

1. [题目] PPO 算法中使用 Clipped Surrogate Objective 的动机是什么？相比 TRPO 的自然梯度方法，PPO 的简化策略有什么代价和收益？
   - 答案/思考提示：PPO 的 clip 机制限制了策略更新的幅度，防止因大幅策略变化导致的性能崩溃。TRPO 使用 KL 散度约束通过自然梯度计算精确的信任域，但实现复杂且计算量大。PPO 通过一阶优化 + clip 近似实现类似效果：收益是实现简单、计算效率高、采样效率好；代价是在极端情况下可能错过最优解，以及对超参数（clip ratio）的敏感性。

2. [题目] XXPO 系列算法（如 IPO、RAVEN、OptiMa）与标准 PPO/GRPO 的核心区别是什么？为什么 XXPO 能更好地处理策略迭代中的过估计问题？
   - 答案/思考提示：XXPO 系列的核心洞察是标准策略梯度方法存在"过度乐观"问题——Q 值过估计导致策略崩溃。XXPO 通过引入显式的策略一致性约束（如 KL 散度正则项）或对价值函数使用更保守的估计来缓解。IPO (Implict Policy Optimization) 通过将策略约束隐式化，RAVEN 通过对抗采样，OptiMa 通过理论推导的更紧的价值下界。核心收益是在大模型 RLHF 场景下训练更稳定。

3. [题目] Flow Matching 应用于策略学习（如 FlowGRPO）的核心思想是什么？相比 Diffusion Model 生成动作，Flow Matching 的生成过程有什么特点？
   - 答案/思考提示：FlowGRPO 将动作轨迹视为数据分布，通过学习从噪声到最优动作的 ODE 路径。核心特点：① 生成是确定性的沿着速度场，② 可实现少步甚至单步生成，③ 收敛速度通常快于 DDPM。挑战在于设计适合动作空间的噪声调度和条件注入机制。

4. [题目] 在 RLHF 中，Reward Model 的质量如何影响 RL 训练稳定性？常见的 Reward Hacking 问题有哪些，如何缓解？
   - 答案/思考提示：Reward Model 质量差会导致 RL 优化目标偏移，产生 Reward Hacking——模型发现取悦 RM 而非完成真实任务的捷径。常见问题：① RM 过度依赖特定文本模式（如长度奖励），② RM 分布与真实意图对齐不足，③ 对抗样本。缓解方法：① KL 散度约束策略偏离 SFT 策略，② 使用 Constitutional AI 或 RL-CL，③ 引入红队评估和迭代 RM 更新。

5. [题目] GRPO (Group Relative Ranking Policy Optimization) 相比 PPO 在大模型对齐中的优势是什么？为什么它更适合 LLM 的 RLHF 场景？
   - 答案/思考提示：GRPO 使用同一 prompt 生成的多个 response 之间的相对排名计算优势，节省了单独训练 Critic (Value Network) 的开销。对于 LLM 场景：① 响应生成成本高，单个样本计算 Value 函数代价大，② LLM 的 token 级 credit assignment 难以精确，组内相对比较更鲁棒，③ 实现简单，调试方便。PPO 需要四类模型（Actor、Critic、Reference、Reward），GRPO 简化到两类。
