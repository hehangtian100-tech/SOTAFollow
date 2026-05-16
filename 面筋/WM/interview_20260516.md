## 日期：20260516

### WM 领域

**论文：** 无新论文，基于 WorldModel 领域核心论文综合整理

**基础概念题：**

1. **World Model 与传统强化学习中的环境模型有何本质区别？为什么 World Model 近年来成为研究热点？**
   - 答案/思考提示：传统环境模型（dynamics model）预测 s' = f(s,a)，World Model 建模更广义的"世界表征"——学习压缩的潜在空间表示，能够进行 imagination-based planning。热门原因：① 生成式 AI 进展（diffusion、transformer）；② sample efficiency 在现实机器人中至关重要；③ JEPA 架构突破。

2. **Dreamer 系列的核心思想是什么？它如何利用 world model 进行 planning？与 Dyna 架构有何关系？**
   - 答案/思考提示：Dreamer 学习 world model + policy，在 latent space 中 imagination rollout 而非真实环境交互。类似 Dyna 架构（learned model + RL）但在 latent space 中进行大大提升 sample efficiency。核心是 RSSM (Recurrent State Space Model)。

3. **什么是 JEPA（Joint Embedding Predictive Architecture）？它与 masked autoencoder 有什么关系？为什么 JEPA 比直接重建像素更适合视觉表示学习？**
   - 答案/思考提示：JEPA 学习两个视图的表示并预测一个视图的表示，而非重建像素。LeWorldModel 将 JEPA 用于世界建模。JEPA 避免学习 trivial identity mapping，比重建更高效因为表示空间比像素空间更紧凑和语义化。

4. **Fast-WAM 和 Latent-WAM 的核心区别是什么？为什么需要 Latent World Action Model？**
   - 答案/思考提示：Fast-WAM 侧重推理加速（用 world model 做 planning 来加速 action generation），Latent-WAM 在 latent space 中建模。Latent WAM 对于高维观测（如图像）更高效，因为直接在像素空间预测动作维度高、计算密集。

5. **World Model 的训练 loss 通常包含哪几个部分？如何平衡 reconstruction loss 和 prediction loss？**
   - 答案/思考提示：通常包含：① 重建 loss（观测）；② 动作预测 loss；③ KL 散度（latent prior vs posterior）。平衡方式：加权求和 + 调度（如 early stage 重重建，后期重预测）。关键洞察：World Model 不仅要重建观测，更要能预测未来。

**深度思考题：**

6. **从信息瓶颈的角度分析：为什么 World Model 需要学习压缩的潜在表示？过于 detailed 或过于 abstract 的表示会带来什么问题？**
   - 答案/思考提示：过于 detailed → 存储和计算成本高，且容易过拟合到无关细节；过于 abstract → 丢失关键信息，无法支持决策。需要找到 minimal sufficient statistic——足以预测未来又不包含冗余信息的表示。这与 representation learning 的核心问题一致。

7. **Being-H0.7 提出 hierarchical world model 用于 robotics，为什么 hierarchical 在长时序任务中比 flat model 更有效？**
   - 答案/思考提示：hierarchical 允许不同时间尺度的学习——高层学习抽象的 task-level 规划，低层学习 skill-level 执行。这与人类认知的 dual-process theory 一致。Flat model 需要同时处理所有时间尺度，参数效率低且难以学习长期依赖。

8. **World Model 在视频生成和机器人控制中都需要建模未来，但两者对"真实"的要求不同。如何理解这个 distinction？**
   - 答案/思考提示：视频生成追求 perceptual realism（看起来真实）；机器人控制追求 dynamics realism（物理上合理）。一个视觉上逼真但物理上不可能的视频对机器人无用。所以 World Model for control 需要 physics-informed inductive bias，不能仅靠 generative model。

**实践应用题：**

9. **假设你要训练一个机器狗的 World Model，用于在未知地形中行走。请设计你的 training pipeline，包括数据收集、表示学习、planning 几个阶段。**
   - 答案/思考提示：
     - 数据：用强化学习 + sim-to-real 收集 diverse gaits
     - 表示学习：用 JEPA 或 RSSM 学习 low-dim latent state
     - Planning：用 model predictive control (MPC) 在 latent space 中 planning
     - 关键：加入 proprioception（本体感觉）和 exteroception（视觉/触觉）的多模态融合

10. **在 World Model 部署到机器人时，sim-to-real gap 是主要挑战。如何用 World Model 本身来弥合这个 gap？**
    - 答案/思考提示：可以用 World Model 做 imagination-based RL——在 sim 中训练 policy，然后 World Model 对真实观测做 online adaptation。另一个方向是 DreamerPro 提出的 prototype-based world model，在 latent space 中对齐 sim 和 real。

**优缺点对比题：**

11. **对比 Dreamer 系列和 Model-based RL（如 PETS、ME-TRPO）的 world modeling 方式。**
    - 答案/思考提示：
      - Dreamer：latent space planning，learned representation，计算高效
      - PETS：particle-based uncertainty quantification，explicit model uncertainty，但计算量大
      - ME-TRPO：ensemble of models 处理 uncertainty
      - Dreamer 在大规模场景更实用；PETS 在小规模精确控制更鲁棒

12. **对比 World Model 作为 generator（生成未来观测）vs 作为 model（预测未来状态）的适用场景。**
    - 答案/思考提示：
      - Generator（视频扩散）：适合长时序想象、视觉规划，但计算密集
      - Model（RSSM/JEPA）：适合 real-time control、policy learning，轻量
      - 最新趋势：两者结合——用 generator 做 long-horizon planning，用 model 做 fast reactive control（如 Uni-World VLA 的交错式架构）

---

**参考文献**：
- DreamerAD: arXiv 2026
- LeWorldModel: arXiv 2026
- Fast-WAM: arXiv 2026
- Being-H0.7: https://research.beingbeyond.com
- Uni-World VLA: ECCV 2026
