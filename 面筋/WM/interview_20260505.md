## 日期：20260505

### WM 领域（通用思考题）

**基于近期 World Model 前沿进展的综合思考题**

1. [基础概念题] JEPA（Joint Embedding Predictive Architecture）和传统 GAN/VAE 在世界模型中的核心区别是什么？为什么 JEPA 更适合视觉为主的决策任务？
   - **答案/思考提示**：JEPA 学习 observation space 的 latent 表征，并预测未来 latent state 而非 pixel-level reconstruction。这种设计避免了生成式模型（如 GAN）的不稳定训练和 pixel-level 重建的歧义性。对于视觉输入，JEPA 可以过滤掉无关的视觉细节（如光照变化），聚焦于语义相关的 state 信息。

2. [深度思考题] Being-H0.7 和 Fast-WAM 都声称是 "action-free world model"，它们和传统需要 action 输入的世界模型有什么本质区别？这种设计的优势和局限是什么？
   - **答案/思考提示**：Action-free 世界模型假设世界 dynamics 是 action-agnostic 的，或者 action information 可以通过 state transition 隐式学习。优势是不需要收集 action-labeled trajectories，降低数据收集成本；局限是当 action 影响重要 state 变化时（如开关门），模型可能无法准确预测。适合场景：视频预测、计划（给定目标状态反推 action）。

3. [实践应用题] 如果你要用世界模型做MPC（Model Predictive Control）规划，但模型推理速度慢于实时要求，你会怎么加速？
   - **答案/思考提示**：① 使用 distill/quantize 压缩模型；② 减少规划 horizon（短视规划）；③ 使用 CPL（Contrastive Predictive Coding）等方法加速 latent dynamics 的预测；④ 采用 EMA 而不是 full rollout 更新 action sequence；⑤ 层次化规划——粗粒度用世界模型，细粒度用简单控制器。

4. [优缺点对比题] Dreamer 系列（DreamerV2/V3）和 LMP（Latent Modular Policy）在处理多任务世界模型时的设计哲学有何不同？
   - **答案/思考提示**：Dreamer 通过统一的 world model 和 policy gradient 解决所有任务，优点是数据效率高、端到端优化；LMP 通过模块化组合不同 skill 模型，优点是可解释、可热插拔。Dreamer 适合任务相似且共享 dynamics 的场景；LMP 适合任务差异大、需要快速适应新 skill 的场景。

5. [深度思考题] 世界模型训练中，reconstruction loss 和 contrastive/predictive loss 的权衡是什么？是否有一种永远优于另一种？
   - **答案/思考提示**：Reconstruction loss 保证 pixel-level 忠实但容易被无关细节主导；Predictive loss 保证语义正确但可能丢失细节。在视觉信息丰富的任务中（如视频生成），reconstruction 仍然重要；在决策为主的任务中，predictive loss 更有效。实际上很多 SOTA 方法（如 DreamerV3）两者结合，用不同权重平衡。

---
*本份为无新论文日的通用领域思考题，基于近期 WorldModel 进展（LeWorldModel/Dreamer/Fast-WAM）综合整理
