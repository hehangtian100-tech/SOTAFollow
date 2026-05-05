## 日期：20260505

### VLA 领域（通用思考题）

**基于近期 VLA 前沿进展的综合思考题**

1. [基础概念题] Vega 和 π0.7 在架构设计上的核心区别是什么？各自的优势场景是什么？
   - **答案/思考提示**：Vega 采用 unified vision-language-world-action 架构，强调跨模态对齐和通用性；π0.7 是面向机器人控制的 action foundation model，更注重 action prediction 的精度。Vega 适合开放世界的 VLA 任务，π0.7 在精细控制任务（如手术机器人）上更有优势。

2. [深度思考题] 为什么大多数 VLA 模型需要 large-scale action data（如 RT-X）才能泛化？数据量与泛化能力之间是否存在临界点？
   - **答案/思考提示**：VLA 本质是在学习 "perception → action" 的映射，这种映射高度任务相关且依赖于物理世界的先验。大规模数据帮助模型学到 action space 的 manifold structure 和与视觉特征的对应关系。临界点取决于任务多样性，当数据覆盖了主要场景的 manifold 时，增益递减。

3. [实践应用题] 如果你要在低算力设备上部署 VLA 模型（如自动驾驶的嵌入式平台），你会考虑哪些优化策略？
   - **答案/思考提示**：① 量化（INT8/FP16）压缩模型；② 蒸馏小模型；③ 异步多级推理（fast path 用小模型，slow path 用大模型 refinement）；④ 结构化 pruning 去除不重要的 attention heads；⑤ 缓存重复的视觉特征（如静态背景）。

4. [优缺点对比题] MINT 的 multi-view inference 和传统单目 VLA 方法相比，有什么本质区别？什么场景下多视图是必要的？
   - **答案/思考提示**：MINT 通过多视角一致性约束来增强几何推理能力，适用于需要 depth/structure awareness 的任务（如操作要被抓取的物体）。单目 VLA 在纹理丰富但几何模糊的场景下容易出错。在手术机器人、精密装配等对几何精度要求高的场景，多视图是必要的。

5. [深度思考题] MemoryVLA 的记忆机制和传统 RNN/LSTM 的 hidden state 机制有什么本质区别？为什么记忆机制对长时任务更有效？
   - **答案/思考提示**：RNN hidden state 是固定维度的压缩表征，容量有限且难以选择性遗忘；MemoryVLA 的 external memory bank 可以存储更多历史信息，且通过 attention-based reading 选择性检索。记忆机制允许模型在长时任务中保留关键中间结果（如已堆叠的积木位置），而不被无关信息稀释。

---
*本份为无新论文日的通用领域思考题，基于近期 VLA 进展（Vega/π0.7/MemoryVLA/MINT）综合整理
