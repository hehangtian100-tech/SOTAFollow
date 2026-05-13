## 日期：20260513

### LLM 领域（FM基础知识）
**通用领域思考题**

1. [基础概念题] Transformer 中的 Self-Attention 的计算复杂度是 O(n²d)，其中 n 是序列长度，d 是 hidden dimension。请解释这个复杂度的来源，并说明 FlashAttention 是如何将其降低的。
   - 答案/思考提示：O(n²d) 来自 Attention Score 矩阵的计算（QK^T，n×n）和加权求和（n×n × n×d）。FlashAttention 通过分块计算（tiling）+ 核融合（kernel fusion）减少 HBM 访问次数：它将 Q、K、V 分块读入 SRAM，在 SRAM 内计算局部 attention，再通过 online softmax 技巧将结果合并，避免了将整个 n×n 矩阵写回 HBM。

2. [深度思考题] LoRA (Low-Rank Adaptation) 假设大模型的权重更新具有低秩特性。请从数学角度解释这个假设的合理性，以及它为什么能在少量参数的情况下有效微调大模型。
   - 答案/思考提示：神经网络在任务适配过程中，不需要大幅度改变预训练学到的知识，只需要调整很少的方向。数学上，ΔW 的奇异值分解往往显示只有少数奇异值较大，说明 ΔW 确实近似低秩。LoRA 将 ΔW = BA（其中 B 是 d×r，A 是 r×k），通过优化 B 和 A 而冻结原始权重，参数从 d×k 降到 r(d+k)，但效果接近全量微调。

3. [实践应用题] 在训练大模型时，如果遇到显存不足的问题，除了简单地减少 batch size 之外，还有哪些技术手段可以缓解？它们各自适用于什么场景？
   - 答案/思考提示：① 梯度累积（Gradient Accumulation）：虚拟增大 batch size，适合显存不足但可以用时间换空间；② 混合精度训练（FP16/BF16）：减少存储格式开销，适合有硬件支持（Tensor Core）的场景；③ ZeRO（分片优化）：将 optimizer state/gradient/parameter 分片到不同 GPU，适合多卡场景；④ 梯度检查点（Gradient Checkpointing）：用计算换显存，适合单卡且显存极度受限；⑤ 量化（QLoRA）：用 INT4 等低精度表示，适合指令微调等场景。

4. [优缺点对比题] RoPE（Rotary Position Embedding）和 ALiBi（Attention with Linear Biases）都是处理序列位置信息的方法，它们各自的核心思想是什么？为什么近年来更多新模型倾向于使用 RoPE？
   - 答案/思考提示：RoPE 通过旋转矩阵将位置信息编码到 Q/K 向量，使得 attention score 只与相对位置相关，具有更好的长度外推潜力；ALiBi 通过在 attention score 上添加线性偏置来编码相对位置，实现简单但通常性能略差。RoPE 近年来更流行是因为：① 更好的长度外推能力；② 可以自然地处理任意长度的位置编码，无需固定最大长度；③ 与 flash attention 等优化兼容性好。

5. [基础概念题] 自回归语言模型在生成时面临的主要挑战是什么？为什么会出现"复读"（repetition）问题？有哪些缓解方法？
   - 答案/思考提示：自回归生成时，模型倾向于最大化当前 token 的似然，可能导致生成内容单调重复。复读问题的根源包括：① 训练数据中重复模式的存在；② 交叉熵 loss 对高概率 token 的过度强化；③ 解码策略（如 greedy）在概率分布尖锐时选择单一 token。缓解方法包括：① temperature 采样增加多样性；② top-k/top-p 采样截断低概率 token；③ n-gram/phrase blocking 强制避免重复；④ 引入对比学习或 diversity penalty 在训练时鼓励多样性。
