## 日期：20260506

### WM 领域（通用思考题）

**基于近期 World Model 前沿进展的综合思考题**

1. [基础概念题] 世界模型中 "imagination" 和 "planning" 的区别是什么？为什么 JEPA 架构更适合 imagination-based planning？
   - **答案/思考提示**：Imagination 是指在世界模型内部模拟未来状态序列；Planning 是利用 imagination 的结果来选择最优 action。世界模型先想象（predict future states），再基于想象结果规划（select actions that lead to desired states）。JEPA 通过在 latent space 做 predictive modeling，避免了 pixel-level imagination 的模糊性和计算开销，更适合 long-horizon planning，因为 latent 表征更 compact 且语义丰富。

2. [深度思考题] Dreamer 系列使用 RSSM（Recurrent State Space Model）来建模世界模型，RSSM 的 latent variables 是 deterministic 还是 stochastic 的？为什么这样设计？
   - **答案/思考提示**：RSSM 的 latent 包含 deterministic（stochastic RNN 的 hidden state）和 stochastic（posterior/prior）两部分。Deterministic 部分提供记忆能力，捕捉时间序列的 deterministic dynamics；stochastic 部分建模 stochastic transitions（如物体随机出现）。这种 hybrid 设计让世界模型既能保持 gradient-based learning（through deterministic path）又能建模 stochastic environment。

3. [实践应用题] 如果你要用世界模型做 autonomous driving 的 simulation（数据闭环），世界模型需要满足哪些关键特性？当前方法的 gap 在哪里？
   - **答案/思考提示**：关键特性：① 长时间一致性（不能 drift）；② 交互性（agent 的 action 能正确影响未来 state）；③ 物理真实性（车辆 dynamics 正确）；④ 高效性（能生成 100Hz+ 的 simulation）。Current gaps：① 长时间 simulation 仍会 drift；② 多数世界模型是 action-free 的，不支持闭环 simulation；③ 计算效率不足（难以 real-time）。Being-H0.7/DreamerAD 在这些方向有所突破但仍有改进空间。

4. [优缺点对比题] Video Prediction 模型（如 DVD-GAN）和 World Model 的核心区别是什么？为什么 World Model 更适合 RL 训练？
   - **答案/思考提示**：Video Prediction 优化的是 pixel-level 重建质量，不一定有 action conditioning；World Model 在 latent space 操作，explicitly modeling P(s_{t+1}|s_t, a_t)，是 Markov decision process 的 explicit capture。World Model 更适合 RL 因为：① Latent space 更 compact，planning 效率更高；② Action-conditioning 使得 imagination + planning 成为可能；③ 世界模型可以直接用作 RL 的 gradient source（imagination rollouts），而 video prediction 的 RL 训练需要额外的 policy gradient。

5. [深度思考题] 世界模型的 "representation learning" 和 "dynamics learning" 是独立的还是耦合的？解耦它们有什么好处？当前有哪些方法实现了这种解耦？
   - **答案/思考提示**：传统方法中两者是耦合的——representation 和 dynamics 共同优化可能产生 misaligned 表征（学到的表征可能对 dynamics 无用）。解耦的好处：① 表征学习可以复用（无监督 pretraining on large dataset）；② Dynamics learning 更稳定（不会因为 representation 变化而变化）；③ 更容易进行 modular design。JEPA 架构通过 predictive objective 解耦了表征和 dynamics；IbaPO 类似的 contrastive 方法也能实现解耦。

---
*本份为无新论文日的通用领域思考题，基于近期 WorldModel 进展（Dreamer/JEPA/Being-H0.7/Fast-WAM）综合整理
