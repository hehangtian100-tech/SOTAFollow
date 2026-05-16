## 日期：20260516

### VLA 领域

**论文：** 无新论文，基于 VLA 领域核心论文综合整理

**基础概念题：**

1. **VLA (Vision-Language-Action) 模型与传统 VLMs 的核心区别是什么？为什么 VLA 需要将动作空间离散化？**
   - 答案/思考提示：VLA 在 VLMs 基础上输出动作 token，需要将连续动作空间 tokenize 为离散 token 以适配自回归生成范式。Pi-0.7 使用 Action Tokenizer，Vega 使用 VLA-specific token。

2. **RT-1 提出的 Bootstrap 训练范式解决了什么问题？与 RT-2 相比有何本质不同？**
   - 答案/思考提示：RT-1 用 transformer 处理「图像+语言指令→动作」，RT-2 用 VLMs 基础上 fine-tune 直接输出动作。RT-2 的 VLA 输出虽然泛化好但计算成本高；RT-1 更轻量但泛化能力弱。

3. **OpenVLA 和 π0.7 在架构上的核心区别是什么？为什么 π0.7 选择 diffusion 而非 autoregressive？**
   - 答案/思考提示：OpenVLA 是 causal transformer（autoregressive），π0.7 是 diffusion model。Diffusion 在多模态动作分布上更 smooth，避免 autoregressive 的 error accumulation，且能处理多峰分布（同一指令对应多个合理动作）。

4. **什么是 mRoPE（Multi-Dimensional Rotary Position Embedding）？它如何解决 VLA 中视频/多帧输入的位置编码问题？**
   - 答案/思考提示：mRoPE 将位置编码分为 temporal、spatial height、spatial width 三个维度，分别编码时间步和空间位置。这对于视频理解很重要，因为视频帧之间的时间关系与图像内空间关系本质不同。

5. **MemoryVLA 中的 Cognitive Memory Bank 机制解决了什么问题？为什么记忆机制对长时序任务至关重要？**
   - 答案/思考提示：机械臂任务需要短时记忆（当前任务状态）和长时记忆（相似任务经验）。Memory Bank 通过检索-融合-整合三阶段让模型"记住"历史经验，避免重复犯错。

**深度思考题：**

6. **从 System 1 vs System 2 的角度分析：为什么 VLA 需要同时具备 fast reactive 和 slow strategic 两种能力？如何在一 个模型中实现这两种能力的平衡？**
   - 答案/思考提示：System 1（快）用于实时反应如 obstacle avoidance，System 2（慢）用于复杂任务规划。π0.7 通过 hierarchical diffusion 实现——high-level planner 生成子目标，low-level controller 执行。MemoryVLA 的 cognition 模块类似 system 2。

7. **VLA 在 Sim-to-Real 迁移上面临哪些核心挑战？为什么视觉 domain randomization 对泛化如此重要？**
   - 答案/思考提示：视觉差异（texture、lighting、camera angle）是 sim-to-real gap 的主要来源。Domain randomization 在仿真中随机化视觉参数，让 policy 在看到任何视觉输入时都能应对。本质是让模型学会忽略不相关信息而关注语义相关特征。

8. **DVGT-2 提出 Vision-Geometry-Action 端到端范式，相比传统 geometric perception pipeline 有何优势？为什么端到端在自动驾驶中更受青睐？**
   - 答案/思考提示：传统 pipeline（感知→地图→规划→控制）error accumulation 且难以优化；端到端 DVGT-2 直接从视觉和几何输入预测动作。优势是联合优化、信息无损传递；但可解释性差是主要 trade-off。

**实践应用题：**

9. **假设你要开发一个家庭服务机器人，能够理解"把桌子上的杯子拿给我"这样的指令。请设计你的 VLA 系统 pipeline，并说明你会选择 diffusion 还是 autoregressive 范式以及理由。**
   - 答案/思考提示：pipeline = 视觉编码器 + 语言理解 + 动作生成。Diffusion 更适合因为：家庭场景动作空间连续且多峰（同一目标可用不同路径到达），diffusion 可以建模多模态动作分布；推理速度可通过 distilled model 改善。

10. **在 VLA 训练中，如果只有 1000 条人类演示数据，你会如何设计 data efficiency 策略？至少提出 3 种方法。**
    - 答案/思考提示：① 使用 pretrained VLMs + behavior cloning 初始化；② 用 RL 或 RHLN 在仿真中 self-improving；③ 使用 domain randomization + data augmentation；④ 用 skill library 分解任务；⑤ 用 video prediction model 生成 synthetic data。

**优缺点对比题：**

11. **对比 autoregressive VLA（如 OpenVLA、Vega）和 diffusion-based VLA（如 π0.7）的优劣。**
    - 答案/思考提示：
      - Autoregressive：推理快（sequential decode），但有 error accumulation，多峰分布建模差
      - Diffusion：多峰分布建模好，训练稳定，但推理慢（需要多步 denoising）
      - π0.7 在长-horizon 任务上优于 autoregressive；但在 real-time 要求的场景 autoregressive 仍有优势
      - 最新趋势：hierarchical + diffusion（如 π0.7 的两层 diffusion）结合两者优点

12. **对比闭环 VLA（closed-loop，如 Uni-World VLA）和开环 VLA（open-loop）的适用场景。**
    - 答案/思考提示：
      - 闭环：长时序任务、环境有干扰、需要在线修正（如复杂 manipulation）
      - 开环：短时序、确定环境、高速控制（如无人机快速机动）
      - Uni-World VLA 的交错式架构结合两者：高层开环（task plan）低层闭环（reactive control）

---

**参考文献**：
- π0.7: Physical Intelligence
- OpenVLA: https://arxiv.org/abs/2407.04569
- MemoryVLA: https://arxiv.org/abs/2508.19236
- Vega: arXiv 2026
- DVGT-2: https://arxiv.org/abs/2512.16919
