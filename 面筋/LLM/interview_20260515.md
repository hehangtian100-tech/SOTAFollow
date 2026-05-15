## 日期：20260515

### LLM 领域
**通用领域思考题**

1. [基础概念题] FlashAttention 通过 tiling 和 flash 技术大幅提升了 Attention 的计算效率。请从 GPU 内存层级结构的角度解释：为什么标准 Attention 实现会成为内存和计算瓶颈？FlashAttention 的核心改进是什么？
   - 答案/思考提示：标准 Attention 需要计算完整的 S = QK^T 矩阵（N×N），对于长序列（N=100k+）这需要 O(N^2) 的显存存储 S 矩阵。GPU HBM（High Bandwidth Memory）带宽有限，频繁读写 N×N 矩阵成为性能瓶颈。FlashAttention 的核心：① 分块计算（tiling）将 Q/K/V 切分为 blocks，每次只将一个 block 调入 SRAM；② 不存储完整 S 矩阵，而是在线计算 softmax 利用分块归一化统计量。实际效果：显存从 O(N^2) 降到 O(N)，同时利用 SRAM 的高带宽加速。

2. [深度思考题] 在训练大模型时，ZeRO（Zero Redundancy Optimizer）的三种 Stage（Stage 1/2/3）分别优化了什么？为什么 ZeRO-3 可以支持超大模型训练但 ZeRO-2 通常对大模型训练更高效？
   - 答案/思考提示：ZeRO-1：分片优化器状态（optimizer states）；ZeRO-2：分片优化器状态 + 梯度；ZeRO-3：分片优化器状态 + 梯度 + 模型参数。ZeRO-3 需要全量参数在每个设备间 all-gather/reduce-scatter，通信开销大；ZeRO-2 通信量小（只需梯度），且激活内存节省不如 ZeRO-3 但通信效率更高。选择策略：小模型（可单卡装下）用 ZeRO-1；中模型（多卡但每卡有足够内存）用 ZeRO-2；超大模型（单卡装不下）用 ZeRO-3 + 卸载（CPU offload）。

3. [实践应用题] 你需要将一个 7B 大模型微调到特定领域（医疗问答），但只有 2 张 A100 40GB GPU。请从技术选型角度给出完整方案，包括：量化方法、PEFT 方法、梯度累积策略、以及如何防止灾难性遗忘。
   - 答案/思考提示：量化：QLoRA（4-bit NF Quantization + LoRA）；PEFT：LoRA 或 QLoRA（rank=16~64，目标模块：q_proj, v_proj, k_proj, o_proj）；梯度累积：micro_batch_size=1, gradient_accumulation_steps=16 解决显存不足；防止灾难性遗忘：① 使用猫流（catastrophic forgetting）相关正则化；② 混合原始预训练数据（通常 1:1 比例）；③ 使用较大学习率进行微调但配合权重衰减；④ RLHF/DPO 进一步对齐。推荐配置：QLoRA + 4-bit NF4 + lora_alpha=32 + target_modules=[q_proj,v_proj]。

4. [优缺点对比题] RoPE（Rotary Position Embedding）和 ALiBi（Attention with Linear Biases）是两种主流的位置编码方案，它们在处理超长上下文时的表现有何差异？各自的理论优势是什么？
   - 答案/思考提示：RoPE 通过旋转矩阵将绝对位置编码融入注意力分数，具备良好的相对位置建模能力，且可以通过 YaRN 等技术扩展到超长上下文；ALiBi 通过线性偏置建模相对距离，实现简单、理论上有更好的外推能力。RoPE 优势：可学习、有成熟的外推技术（RoPE scaling）、与 FlashAttention 兼容性好；ALiBi 优势：无需位置编码参数、外推性好（理论无界外推）、实现简单。长上下文场景推荐 RoPE + scaling；短上下文 + 强外推需求推荐 ALiBi。

5. [基础概念题] VQVAE 和 VQ-GAN 在视觉 Tokenizer 领域各有代表工作（如 ViT-VQGAN、VAR）。请问：为什么视觉生成需要 Tokenizer？它与语言模型中的 BPE/WordPiece Tokenizer 有什么本质区别？视觉 Tokenizer 的设计核心考量是什么？
   - 答案/思考提示：视觉 Tokenizer 将高维图像压缩为低维离散 token（类似 word piece），使得自回归框架可以建模视觉生成；同时大幅降低生成计算量。语言 tokenizer 基于语义切分（word/subword），视觉 tokenizer 基于像素块/特征压缩；语言 token 有明确语义（word），视觉 token 是高度压缩的连续表征的离散化。设计核心考量：① 压缩率（codebook size × spatial resolution）；② 表征质量（重建质量 vs 语义质量权衡）；③ 下游任务适配性（detection/segmentation 需要保留空间结构）；④ GAN loss vs. MSE loss 对 codebook 学习的影响；⑤ 码本利用率和崩溃问题。
