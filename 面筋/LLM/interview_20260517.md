## 日期：20260517

### LLM 领域（FM基础知识）

**主题：Transformer 架构优化与注意力机制演进**

1. **[基础概念题]** 标准残差连接（PreNorm/PostNorm）中，"信息稀释"问题在长文本场景下为何更严重？Kimi Attention Residuals 提出的跨层选择机制是如何缓解这一问题的？
   - **答案/思考提示**：随着层数增加，信息需要经历更多 transformation，早期层的重要信号可能被覆盖。Attention Residuals 让模型能自适应地选择保留哪些层的哪些信息，通过可学习的跨层 attention 路由，使得关键信息可以"跳过"无关的中间层，直接传递到需要的深层。

2. **[深度思考题]** FlashAttention 通过 IO-Awareness 优化矩阵运算，减少 HBM 访问次数。但 FlashAttention-2 和 FlashAttention-3 在算法层面有哪些关键改进？这些改进分别解决了什么瓶颈？
   - **答案/思考提示**：FA2 主要改进是减少非矩阵乘法（non-matmul）运算的 overhead，增加 tiling 效率；FA3 引入异步执行和 warp specialization，在 A100/H100 等新硬件上进一步提升occupancy。FA3 的关键是将 attention 计算流水线和数据加载流水线重叠，减少 idle warp。

3. **[实践应用题]** 你需要在 LLM 推理服务中部署一个 70B 参数的模型，但只有 4 张 80GB A100（总计 320GB）。请设计一个完整的部署方案，包括：量化策略、并行方案、显存优化、推理优化。
   - **答案/思考提示**：(1) 量化：AWQ 或 GPTQ 4-bit 量化，70B → ~40GB；(2) 张量并行：TP=4 将模型分片到 4 卡，每卡 ~10GB；(3) 流水线并行：PP=2 或 3 减少 pipeline bubble；(4) KV-cache 优化：FlashAttention + 动态 batch；(5) 推理引擎：vLLM 或 TensorRT-LLM，支持 continuous batching 和 paged KV cache。

4. **[优缺点对比题]** Attention 的 O(n²) 复杂度是长文本的主要瓶颈。对比 Linear Attention、State Space Model（Mamba）、以及 Hierarchical Attention（如 Hawk/H仙女座）等高效变体，在不同场景（短文本、长文本、序列生成）下的适用性。
   - **答案/思考提示**：Linear Attention 适合短到中等长度（< 4K），但丢失了 softmax attention 的非线性归纳偏置；SSM（Mamba）适合超长序列（> 10K）的选择性复制任务，但在需要精确 token matching 的任务上弱于 attention；Hierarchical Attention 在长文本上效果好但需要修改架构。实际系统往往是混合架构。

5. **[综合思考题]** 从 PETR → PETR V2 的演进路径，思考"纯视觉 3D 感知"vs"多传感器融合"的技术取舍。如果让你设计一个面向自动驾驶的下一代感知系统，你会如何平衡精度、延迟、计算成本？
   - **答案/思考提示**：PETR 系列证明了"不需要显式深度估计"的纯视觉路线可行性。技术取舍：LiDAR 提供精确深度但成本高、雨雪天气退化；视觉成本低但深度估计不准确。下一代系统建议：多模态融合 + 轻量级 LiDAR（4-8 线）作为补充 + 视觉为主的感知栈 + 时序融合（BEVFormer style）+ 传感器外参在线标定。延迟瓶颈在 BEV 特征融合，建议用 Sparse BEV 而非 Dense BEV。
