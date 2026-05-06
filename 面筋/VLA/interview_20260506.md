## 日期：20260506

### VLA 领域（通用思考题）

**基于近期 VLA 前沿进展的综合思考题**

1. [基础概念题] VLA 模型中 "action tokenization" 和语言模型的 tokenization 有什么本质区别？为什么动作空间需要特殊的离散化处理？
   - **答案/思考提示**：语言 tokenization 通常基于 subword BPE/BBPE，语义信息丰富且 vocabulary size 可控；动作 tokenization 需要将连续动作空间（关节角度/末端执行器 pose）离散化，常见方法包括：① vector quantization（如 VQVAE）；② uniform binning；③ learned discretization。动作空间通常是低维但连续的，离散化后需要保持物理意义（如相邻 token 应对应相似动作），否则会引入物理不一致性。

2. [深度思考题] RT-2/π0 等 VLA 模型在泛化到未见过的任务时，依赖的是什么类型的"泛化"？这种泛化和 LLM 的 semantic 泛化有什么本质区别？
   - **答案/思考提示**：RT-2/π0 的泛化主要是 visuomotor skill composition——模型学会的是 "observation → action" 的 mapping pattern，而非 high-level semantic reasoning。当遇到新任务时，模型通过组合已学会的 primitive skills 来解决。LLM 的泛化是 semantic 和 syntactic 的，能处理抽象推理。VLA 泛化依赖于 demonstration 的覆盖范围，LLM 泛化可以处理 zero-shot 抽象推理。

3. [实践应用题] 如果你要让 VLA 模型支持新的机械臂（DOOF 与训练时的机械臂不同），你需要做哪些适配工作？最少需要多少条数据？
   - **答案/思考提示**：① Action space normalization/adapter：调整不同机械臂的 joint limits 和 action 分布；② Camera extrinsic calibration：统一视觉坐标系；③ 视觉 Domain adaptation：统一视觉输入风格。Minimum data：理论上 10-50 条同任务 demonstrations 即可通过 LoRA 或 head adapter 适配。关键在于新机械臂的 action space 是否能通过线性变换映射到旧机械臂的 action space。

4. [优缺点对比题] Diffusion policy 和 autoregressive policy 在生成动作序列时各有什么优缺点？什么场景下 diffusion 更适合？
   - **答案/思考提示**：Diffusion policy 通过 denoising 生成动作序列，能更好地建模 multi-modal action distribution，适合 tasks with multiple valid solutions；AR policy 生成效率高，但难以处理 multi-modal。Diffusion 更适合：① 动作有 natural noise（如人类示范）；② 需要 long-horizon planning；③ action space 多模态（如同一个 goal 可以用不同方式达成）。AR policy 在需要 low-latency 的 real-time control 场景更有优势。

5. [深度思考题] VLA 模型的「物理常识」是从哪里来的？是来自视觉预训练还是交互数据？为什么很多 VLA 模型仍然缺乏基本的物理直觉（如不知道杯子会掉落破碎）？
   - **答案/思考提示**：VLA 的物理常识主要来自视觉预训练（如 CLIP/SAM）学到的物体 permanence、support relationship 等，但不来自交互因此缺乏 contact dynamics 和 physics of breakage。杯子掉落破碎需要 contact dynamics 和 material properties，这种知识在互联网视频中很少被标注，只有少量 interaction data 覆盖。解决方案：① 合成物理数据增强；② 具身数据收集；③ 显式物理建模。

---
*本份为无新论文日的通用领域思考题，基于近期 VLA 进展（RT-2/π0/Vega/DiffusionPolicy）综合整理
