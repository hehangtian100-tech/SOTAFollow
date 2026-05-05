## 日期：20260505

### LLM 领域（通用思考题 - FM基础知识）

**基于近期 Foundation Model 基础知识的综合思考题**

1. [基础概念题] FlashAttention 的核心优化是什么？为什么 IO-aware 的 attention 实现能显著加速且不损失精度？
   - **答案/思考提示**：FlashAttention 通过 tiling 技术将 attention matrix 分块计算，避免了 HBM（High Bandwidth Memory）的 O(N²) 读写。核心是利用 GPU SRAM 的高速带宽，在 SRAM 上完成 partial attention 计算，然后写回 HBM。这种 design 使得 attention 计算的 IO complexity 从 O(N²) 降到 O(N²d/M)，其中 d 是 head dimension，M 是 SRAM 大小。

2. [深度思考题] LoRA 的低秩假设在理论上为什么能 work？什么时候 LoRA 可能失效？
   - **答案/思考提示**：LoRA 假设 fine-tuning 过程中 weight 更新是低秩的，核心是 pretrained model 已在足够丰富的任务上学习了有用的 representation，fine-tuning 只需在 task-specific direction 上调整。失效场景：① 新任务需要 pretrained knowledge 彻底重塑（如完全不同的 domain）；② 模型 size 很小，full fine-tuning 和 LoRA 差异不大；③ 任务需要精细控制某些 layer 的行为，LoRA 的 rank decomposition 可能过于 global。

3. [实践应用题] 如果你要用 LoRA 训练一个 7B 模型进行 instruction tuning，但显存受限（单卡 80GB），你会如何配置 LoRA rank、target modules 和 learning rate？
   - **答案/思考提示**：① Rank 一般选 8-16，太大失去效率优势；② Target modules 应包含 attention 的 Q/K/V/O 四个projection；③ LoRA 层的 lr 可以比 base model 高 10-100x（因为参数量小）；④ 使用 gradient checkpointing 节省显存；⑤ 可以用 QLoRA（4-bit 量化 + LoRA）进一步降低显存。

4. [优缺点对比题] RoPE（Rotary Position Embedding）和 ALiBi（Attention with Linear Biases）在处理长上下文方面的核心差异是什么？各有什么优缺点？
   - **答案/思考提示**：RoPE 通过旋转 encoding 实现相对位置编码，天然支持 extrapolation（理论上可以处理训练长度以外的 context），但 2D 形式直接应用到高维（3D vision 等）需要 mRoPE 等扩展；ALiBi 通过线性偏差实现，相对位置关系直接注入 attention score，实现简单且在长 context 上有一定泛化能力，但破坏了 pretrained attention pattern。实际应用中 RoPE 是主流。

5. [深度思考题] VQVAE 的 codebook 机制和 Transformer 的 tokenization 本质上有什么共同点？为什么离散表征在某些视觉生成任务中比连续表征更有效？
   - **答案/思考提示**：两者都是将高维连续信号映射到离散 token 空间的学习量化过程。离散表征的优势：① 消除 decoder 对 fine-grained 细节的 reliance，逼迫 encoder 学习 semantic meaning；② 离散 codebook 提供 natural compression，适用于 world model 的想象推理；③ 离散的 token space 可以用 autoregressive model 自然建模。但离散表征也有信息损失，在需要精细纹理的场景（如超分辨率）表现可能不如连续表征。

---
*本份为无新论文日的通用领域思考题，基于近期 FM 基础知识进展（FlashAttention/LoRA/RoPE/VQVAE）综合整理
