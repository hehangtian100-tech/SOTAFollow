## 日期：20260506

### LLM 领域（通用思考题 - FM基础知识）

**基于近期 Foundation Model 基础知识的综合思考题**

1. [基础概念题] Transformer 的 self-attention 计算复杂度是 O(N²)，但在实践中影响其效率的瓶颈是什么？FlashAttention 为什么能接近理论计算复杂度？
   - **答案/思考提示**：实际瓶颈是 memory bandwidth——attention matrix 的 HBM 读写是主要延迟来源，而非算术计算本身。FlashAttention 通过分块计算（tiling）将 partial attention 保留在 SRAM 中，只将最终结果写回 HBM，将 IO complexity 从 O(N²) 降到 O(N²d/M)。FlashAttention-2 进一步优化了 parallelism 和 work balancing，接近硬件的理论上限。

2. [深度思考题] 为什么大多数 LLM 预训练使用 next-token prediction loss 而不是其他 loss？这种 loss 的归纳偏置是什么？它的局限性在哪里？
   - **答案/思考提示**：Next-token prediction 是 language modeling 的 natural objective，能从互联网上大量获取训练数据。Inductive bias：语言是序列依赖的，上下文决定下一个 token。但局限：① 单一 token 粒度的预测可能不足以捕捉 complex semantic dependencies；② 训练信号稀疏（只有一个 token 的监督）；③ 容易被 surface-level patterns 主导（如 n-gram statistics）。MMLU 等 benchmark 揭示了这种 loss 在 reasoning 上的不足。

3. [实践应用题] 如果你要用 8B 参数的 LLM 部署一个 RAG 系统，在 24GB 显存的推理卡上运行，你会如何配置量化参数（bits、method）和 context length？
   - **答案/思考提示**：① 量化：GGUF Q4_K_M（4-bit，balanced quality-speed）或 Q5_K_M（5-bit，更高质量但更慢）；② Context length：考虑 vLLM 的 paged attention，可以利用 streaming prefix caching 节省显存；③ 如果用 16GB 显存，Q4_K_M + 32k context 可以运行；24GB 显存可以用 Q5_K_M + 64k context；④ 还可以使用 LoRA 进行 task-specific 压缩。

4. [优缺点对比题] RoPE 位置编码和 Sinusoidal 位置编码的核心实现差异是什么？为什么 RoPE 能更好地外推到训练长度之外的上下文？
   - **答案/思考提示**：Sinusoidal 是 absolute position encoding，每个位置有独特的 sinusoidal pattern；RoPE 是相对位置编码，通过旋转 Q/K vectors 实现相对位置信息的注入。RoPE 的 extrapolation 能力来自：① 相对位置关系直接编码在 attention score 中，不受 absolute position 限制；② 通过 rope scaling（linear/YaRN）可以在 finetune 后处理更长 context。理论上是 rope 频率特性与相对位置的数学对应使得 extrapolation 成为可能。

5. [深度思考题] MoE（Mixture of Experts）架构中，为什么只用 few experts 的激活而不是 all experts？这种设计的理论依据是什么？有什么 trade-offs？
   - **答案/思考提示**：MoE 的 sparse activation 假设不同 token 需要不同的 "expertises"，而非所有参数都参与每个 token。Trade-off：efficiency vs specialization。使用 few experts（如 top-2）能保持模型容量大（many parameters）但推理成本低（few active）；但如果 tasks 需要 global reasoning（每个 token 需要全部知识），sparse activation 可能不如 dense model。理论依据是 task modularity——不同输入确实需要不同处理流程。Expert collapse 是主要训练挑战。

---
*本份为无新论文日的通用领域思考题，基于近期 FM 基础知识进展（FlashAttention/Transformer/RoPE/MoE）综合整理
