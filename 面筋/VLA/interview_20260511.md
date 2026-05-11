## 日期：20260511

### VLA 领域

**论文：（通用领域思考题 - 无新增论文）**

1. [基础概念题] Vision-Language-Action (VLA) 模型与传统的 Vision-Language Model (VLM) 在输出层面有什么本质区别？为什么 VLA 需要 action tokens？
   - **答案/思考提示**：VLM 输出文本 token，VLA 输出文本 token + action tokens（如机器人控制指令的离散或连续动作）。Action tokens 需要与视觉、语言 token 在同一 embedding space 中对齐，通过多模态 autoregressive training 实现。关键挑战是让 action prediction 与环境交互保持一致性和实时性。

2. [深度思考题] 为什么大多数 VLA 模型（如 π0.7、Vega）采用 diffusion-based action decoder 而非直接 autoregressive 生成动作序列？这两种方法在处理多模态动作分布时各有何优劣？
   - **答案/思考提示**：Diffusion-based 方法能更好地建模多模态动作分布（同一状态可能有多种合理动作），通过 denoising 过程逐步细化动作预测。Autoregressive 方法在生成效率上更优（single forward pass），但在多模态分布上容易出现 mode collapse。Diffusion 的缺点是推理速度慢（需要多步迭代），不适合 real-time control。

3. [实践应用题] 如果你需要在低算力设备上部署 VLA 模型进行机器人控制，且模型需要保持较好的泛化能力，你会采用哪些策略？
   - **答案/思考提示**：① 模型压缩：使用 LoRA/QLoRA 进行 parameter-efficient fine-tuning；② 知识蒸馏：从大模型蒸馏到小模型；③ 动作空间量化：将连续动作离散化或使用低精度表示；④ 异步推理：预计算+缓存视觉特征；⑤ 混合架构：视觉编码器定期更新，动作decoder轻量化；⑥ 迁移学习：在大模型上预训练，在边缘设备上只微调adapter。

4. [优缺点对比题] 端到端 VLA 方法（如 Vega、Actuate）与模块化方法（如 ThinkBot with VLA-JEPA）在系统设计和泛化能力上有何本质差异？
   - **答案/思考提示**：端到端方法将感知、规划、执行统一到单个模型中，联合优化所有模块，系统更简洁但在出现 failure 时难以诊断，且对训练数据要求高。模块化方法通过分离感知（world model）、规划（VLM）和执行（control policy）提高可解释性和容错性，可独立替换各模块。模块化在 domain shift 时可能各模块间的接口假设会失效。

5. [深度思考题] VLA 模型在具身场景中如何处理 "perception-action loop" 的延迟问题？即视觉输入到动作执行之间存在的时间差可能导致观察过时。
   - **答案/思考提示**：① Action chunking：一次生成多步动作序列，减少频率；② 异步感知-动作：使用历史观察队列，输入端主动延迟或等待；③ Predictive coding：用 world model 预测未来状态，提前做出反应；④ Memory-augmented architecture（如 MemoryVLA）：维护短期/长期记忆，融合历史信息；⑤ Temporal abstraction：分层决策，高层规划频率低，底层执行频率高。
