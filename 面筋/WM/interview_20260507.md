## 日期：20260507

### WM 领域（通用思考题）

**基于近期 World Model 前沿进展的综合思考题——世界模型架构与想象力规划专题**

1. [基础概念题] JEPA（Joint Embedding Predictive Architecture）和传统自编码器（AE/VAE）在学习世界表征的核心目标上有什么本质区别？为什么 JEPA 更适合用于世界模型的视觉编码器？
   - **答案/思考提示**：AE/VAE 重建像素级信息，迫使编码器保留所有低层细节（如纹理、噪声），但这对决策有用的只是高层抽象；JEPA 在表征空间做预测，只预测感兴趣区域的表征，编码器可以丢弃与任务无关的细节。LeWorldModel 等工作用 IEBN（Invariant Energy-Based Network）实现 JEPA，通过 energy-based regularization 让模型只关注 motion-relevant features。JEPA 比 AE 更适合世界模型：因为它学到的是对 action 敏感的表征（能预测 action 效果），而非对所有像素变化都敏感的表征。

2. [深度思考题] Dreamer 系列（DreamerV2/V3）和 DreamerAD 在世界模型训练目标上有什么差异？为什么 DreamerAD 需要对物理建模（如碰撞、自由度）而 DreamerV3 可以在纯像素空间工作？
   - **答案/思考提示**：DreamerV3 训练 RSSM（Recurrent State Space Model）在像素空间做 image+reward 重建，所有 physics reasoning 是隐式学到的；DreamerAD 需要显式建模物理结构（如碰撞响应、自由度约束），因为自动驾驶需要在开环 imagining 中对 long-horizon consequences 准确建模，纯像素重建在长序列想象时会累积误差。更根本地，自动驾驶场景的 long-horizon planning 需要精确的物理预测能力，而视频生成的 pixel-level accuracy 不能满足这个需求。Fast-WAM 和 Latent-WAM 的核心创新都在于如何设计既能快速推理又能保持物理一致性的世界模型架构。

3. [实践应用题] 世界模型在机器人控制中的典型使用范式是什么？Imagine Rollout（想象 rollout）具体是怎么操作的？它和 Model Predictive Control (MPC) 有什么关系？
   - **答案/思考提示**：典型范式：① 学到一个世界模型 p(s'|s,a)；② 用世界模型想象多条未来轨迹；③ 选取最优动作。Imagine Rollout 操作：给定当前状态 s_t，用世界模型 roll out 想象未来 N 步 (s_{t+1}, a_{t+1}, ...)，评估每条 roll-out 的 expected return，选择 return 最高的第一个动作执行。这本质上是 Model Predictive Control 的一个实例——MPC 在每步用模型预测未来并选取动作，执行后再重新规划。关键区别在于 world model 通常用 policy gradient 或 EM 在 latent space 做 imagination，而 MPC 通常在原始状态空间做短时域优化。

4. [优缺点对比题] 基于 Diffusion 的世界模型（如 MCVD、FDM）和基于 RNN/Transformer 隐状态的世界模型（如 Dreamer、World Transformer）在生成未来视频方面各有什么优缺点？
   - **答案/思考提示**：Diffusion world models 优点：能生成高质量、多模态的像素级视频，适合生成多样化场景和人类的视觉体验；缺点：每步需要 denoising 迭代，imagination rollout 计算成本高，且难以精确控制长时序物理一致性。RNN/Transformer latent world models 优点：latent space 推理快（无需像素级生成），适合需要快速规划的场景；缺点：表征空间可能过于压缩，丢失长时序细节。Being-H0.7 尝试在 diffusion 世界模型中引入 action 分辨率控制；Fast-WAM 用 action-conditioned 采样来加速推理。

5. [深度思考题] 端到端的世界模型（直接用视频预测训练）和组合式世界模型（视觉 tokenizer + 动作预测 + 视频生成）各是如何保证物理一致性的？哪种方式更有可能scale到真实机器人场景？
   - **答案/思考提示**：端到端（如 DayDreamer）：所有物理一致性是隐式学到的——模型从数据中自己发现物理规律；组合式（Being-H0.7、Fast-WAM）：将物理建模组件化（geometry model、action decoder 分离），物理一致性可以显式约束（如 3D geometry loss）。端到端在数据足够时可能学到更通用的物理先验，但在稀缺数据下容易学到 spurious correlations；组合式在数据效率上有优势，因为 geometry/物理结构是人为设计的inductive bias。Scale 到真实机器人的挑战在于：真实世界的物理复杂度远超市售数据集，组合式更能利用人类的物理知识注入，而端到端需要海量数据才能竞争。
