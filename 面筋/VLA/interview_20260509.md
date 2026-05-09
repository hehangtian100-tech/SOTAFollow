## 日期：20260509

### VLA 领域

**论文1：《Vega: Learning to Drive with Natural Language Instructions》**（arXiv:2603.25741）

1. **[基础概念题]** Vega 提出的"Imitation Driving → Instructional Driving"转变是什么意思？传统 VLA 模型的"稀疏动作监督"问题是什么？Vega 如何解决？
   - **答案/思考提示**：Imitation Driving：模型只能模仿训练数据中专家的平均策略；Instructional Driving：能根据用户自然语言指令生成多样化的、符合指令的轨迹。稀疏动作监督：高维视觉-语言输入 → 低维动作输出 的映射缺乏足够监督信号。Vega 解决方式：① 将自然语言指令引入决策过程；② 利用未来帧图像生成作为密集监督信号（同时优化动作规划任务和世界建模任务）。

2. **[深度思考题]** Vega 采用 AR（自回归）+ Diffusion 混合架构，其中 MoT（Mixture-of-Transformers）与 MoE（Mixture of Experts）有什么本质区别？为什么 Action Expert 用更小的 hidden size (256)？
   - **答案/思考提示**：MoE 仅 FFN 分离，MoT 对所有可学习参数（attention + FFN）都为每个模组复制一套，实现更深度的模态专用化。Action Expert 用更小的 hidden size 是因为动作空间维度低，降低计算开销，同时不显著损伤性能——这是一个工程效率和效果的权衡。

3. **[实践应用题]** 假设你要将 Vega 的指令跟随能力扩展到机器人操作任务（非驾驶），请设计数据标注 pipeline（仿照 InstructScene），并说明需要解决哪些挑战？
   - **答案/思考提示**：① Stage 1: 用 VLM 描述场景和操作行为（输入多视角图像序列）；② Stage 2: 将描述组合后用 VLM 生成自然语言指令；③ 挑战：机器人操作指令比驾驶指令更多样化（"轻柔放置"、"快速抓取"等），需要更细粒度的动作语义标注；驾驶是单 agent，机器人操作涉及多步骤和接触动力学。

4. **[优缺点对比题]** 与 DriveVLA-W0 相比，Vega 的核心优势是什么？在 EPDMS 指标上 Vega (86.9) vs Vega† (89.4)，这个 Best-of-N (N=6) 策略说明了什么？
   - **答案/思考提示**：Vega 在 NC (No at-fault Collision) 和 EP (Ego Progress) 上领先，EPDMS 达到 86.9（vs DriveVLA-W0 的 86.1）。Vega† 的 Best-of-N 策略说明：Vega 生成多样化轨迹的能力是真实的——可以从 6 条候选中选最优，证明多模态生成确实在捕捉不同的策略意图，而不是结构上有多个模态但训练后收缩到主模态附近。

**论文2：《Actuate 2025 | Sergey Levine：第二代 VLA》**

5. **[深度思考题]** Sergey Levine 强调"The key is not the architecture, but the data"，这句话在 π₀ 系列设计中如何体现？互联网视频预训练和 cross-embodiment 数据策略分别解决了什么问题？
   - **答案/思考提示**：π₀ 采用扩散模型 + VLA 架构 + RL 后训练，而非特殊架构创新——相同的架构设计可以用在不同数据上。互联网视频解决"物理常识缺失"问题（柔软物体要轻拿轻放等）；cross-embodiment（跨形态）数据通过动作抽象层（抓住/放置/推动等任务级别语义）实现跨机器人泛化。

6. **[基础概念题]** 第一代 VLA（RT-1/RT-2）和第二代 VLA 的核心区别是什么？为什么第二代 VLA 需要强化学习后训练（RL Post-training）？
   - **答案/思考提示**：第一代：端到端模仿学习，数据瓶颈导致泛化不足。第二代：大规模预训练 + RL 后训练，弥补模仿学习在长尾任务上的失败。"Imitation learning can only replicate what it has seen. Reinforcement learning can discover what has never been demonstrated."——RL 能发现从未被示教过的行为。
