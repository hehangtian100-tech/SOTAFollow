## 日期：20260511

### WM 领域

**论文：（通用领域思考题 - 无新增论文）**

1. [基础概念题] 世界模型 (World Model) 的核心目标是学习 $p(s_{t+1}|s_t, a_t)$ 还是 $p(o_{t+1}|s_t, a_t)$？这两种建模方式有什么区别，各适用于什么场景？
   - **答案/思考提示**：学习 $p(s_{t+1}|s_t, a_t)$ 是表征世界模型的隐式动态，学 $p(o_{t+1}|s_t, a_t)$ 是 observation reconstruction-based 方法。前者（如 JEPA）只建模抽象表征空间中的动态，效率高但需要额外 decoder 来可视化；后者（如 Dreamer）重建像素级观察，可解释性强但计算量大。隐式动态更适合高层规划，像素重建适合需要生成逼真视频的场景。

2. [深度思考题] Dreamer 系列算法和基于 RL 的世界模型（如 World Action Model）在表征学习和策略学习目标上的分歧点是什么？如何在训练目标上统一两者？
   - **答案/思考提示**：Dreamer 通过世界模型重建 observation，同时学习 action prediction；WAM 等方法关注在隐空间中预测未来状态以支持 planning。两者的核心分歧是 objective：reconstruction vs. latent dynamics prediction。统一方式：使用 latent dynamics prediction 作为 auxiliary objective 辅助 reconstruction；在 latent space 中同时做 policy learning 和 world model learning（如 DreamerV3）。

3. [实践应用题] 如果你要设计一个用于自动驾驶的世界模型，要求能够在给定当前观察下生成多样化的未来轨迹（包括正常驾驶和事故场景），你会如何设计模型架构和训练策略？
   - **答案/思考提示**：① 架构：使用 stochastic latent dynamics（VAE/Gaussian），而非 deterministic，以建模多样化未来；② 训练：contrastive learning 或 VAE prior matching 确保 latent space 捕获多模态性；③ conditioning：当前观察作为 condition，不同的 latent sample 产生不同未来；④ loss 设计：reconstruction loss + KL divergence to prior + optional discriminator；⑤ 极端场景：需要特意在训练数据中加入 corner cases，或使用 classifier-guided generation 来增加事故场景概率。

4. [优缺点对比题] 基于 Diffusion 的世界模型（如 IMAGINE、Genie-2）与基于 RNN/Transformer 的隐式世界模型（如 Dreamer、LeWorldModel）在生成质量和计算效率上的权衡是什么？
   - **答案/思考提示**：Diffusion 模型在建模复杂多模态分布时更强（可以表示多种合理未来），生成质量高但推理需要多步迭代（10-50步），延迟高。RNN/Transformer 隐式模型单步 forward 即可预测未来，效率高但在多模态建模上较弱。混合方法：用 diffusion 建模 latent prior，RNN 建模 dynamics，可在质量和效率间取得平衡。

5. [深度思考题] 世界模型训练中常见的 "model non-delusion" 问题是什么？即世界模型可能学习到"捷径"而不是真实物理规律，导致在 imagined trajectories 中策略过度乐观。
   - **答案/思考提示**：Model non-delusion 要求世界模型在 rollover（想象轨迹展开）中的行为与真实环境中一致。如果模型在单步预测时准确，但多步展开时 error accumulation 导致偏离真实物理（如物体穿墙、违反重力），策略会学会利用这些"幻觉"。解决方案：① 集成多个世界模型，取 consensus；② 对 rollover 过程中的 uncertainty 建模，主动降低 confidence；③ 对抗训练让世界模型对 rollout 分布更鲁棒；④ 定期用真实环境 rollouts 校正。
