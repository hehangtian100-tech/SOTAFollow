## 日期：20260517

### WM 领域（世界模型）

**主题：World Action Model 与世界模型训练**

1. **[基础概念题]** 在 AIM 论文中，Spatial Value Map 的作用是什么？它解决了传统 WAM（World Action Model）中"从密集的 RGB future latent 反推稀疏控制意图"的什么问题？
   - **答案/思考提示**：Value-map 将高维 RGB 特征压缩为低维的空间价值图，每个像素对应场景中的物理位置及其价值。这样动作头可以直接从"高价值空间位置"回归动作，而不需要从外观丰富的 RGB latent 做隐式逆动力学，显著降低了动作预测的难度。

2. **[深度思考题]** 世界模型的"observation prediction"和"action prediction"两个目标在训练时通常如何平衡？为什么联合训练可能导致表征学习的退化？
   - **答案/思考提示**：多任务学习存在表征干扰——动作预测 loss 可能迫使视觉编码器偏向动作相关特征，损害纯观察任务的重建质量。解决方案：解耦视觉编码器（观察分支 vs 动作分支）、预训练观察分支再冻接训练动作头、或使用 gradient surgery 来分离梯度。

3. **[实践应用题]** 如果你需要为"仿人机器人双足行走"任务训练一个世界模型，请设计 reward shaping 策略：如何从稀疏的二元成功信号（走得稳/摔倒）设计密集的 intermediate reward？
   - **答案/思考提示**：(1) 躯干高度惩罚（保持一定高度）；(2) 质心投影位于支撑多边形内；(3) 步频奖励（鼓励自然步态）；(4) 关节角平滑度惩罚（避免抖动）；(5) 足地接触时序奖励（先跟后跟）。通过这些 intermediate reward 将稀疏的最终成功信号分解为每步可学习的 dense reward。

4. **[优缺点对比题]** Dreamer 系列（DreamerV2/V3）和 TD-MPC（World Model）代表了两种不同的世界模型范式：一种基于 Recurrent State Model + ELBO 重建，另一种基于 MPC 框架下的即时动作优化。比较两者在样本效率、计算效率、长时序规划能力上的差异。
   - **答案/思考提示**：Dreamer 样本效率高（重放buffer+截断BPTT），但计算效率中等（需要重建 loss）；TD-MPC 计算效率高（无重建，pure MPC），但样本效率较低（需要在线交互）。长时序规划：Dreamer 通过 latent imagination 做长期；TD-MPC 通过 model-predictive control 即时规划。两者在实时性要求高的场景各有权衡。

5. **[综合思考题]** 世界模型的核心价值在于"在 imagination space 中规划"而非"在真实环境中试错"。但 imagination space 和真实物理空间之间存在"domain gap"。有哪些方法可以最小化这个 gap？
   - **答案/思考提示**：主要方法：(1) 域随机化（Domain Randomization）在仿真中增加多样性；(2) .sys_id 和 meta-learning 让模型快速适应新物理；(3) 基于保守 Q 函数的约束规划（CQL、ICRL）；(4) 用 real-world data 微调 world model 而非从头训练；(5) 组合式规划：用多个 specialized world model 组合覆盖不同 regime。
