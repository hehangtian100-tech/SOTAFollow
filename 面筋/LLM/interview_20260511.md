## 日期：20260511

### LLM 领域（FM基础知识）

**论文：（通用领域思考题 - 无新增论文）**

1. [基础概念题] Transformer 中的 Self-Attention 的计算复杂度是 $O(n^2 \cdot d)$，其中 $n$ 是序列长度，$d$ 是隐藏维度。FlashAttention 是如何将复杂度降低的？其核心思想是什么？
   - **答案/思考提示**：FlashAttention 通过 IO-aware  tiling 技术，将 attention 计算分块处理，全部在 SRAM 中完成，避免 HBM（High Bandwidth Memory）的多次读写。核心思想是利用 GPU memory hierarchy：HBM 带宽远低于 SRAM，将 $O(n^2)$ 的 HBM access 减少到 $O(n^2/d)$ 级别（$d$ 为 block size），同时通过 online softmax 技巧在分块计算时保持数值准确性。

2. [深度思考题] LoRA 的核心insight是什么？它为什么能在只更新少量参数的情况下达到接近全参数微调的效果？这种方法的理论依据是什么？
   - **答案/思考提示**：LoRA 的核心假设是预训练模型的权重更新矩阵 $\Delta W$ 是低秩的。形式上 $W_0 + \Delta W = W_0 + BA$，其中 $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$，$r \ll \min(d,k)$。理论依据：① 大模型的 intrinsic dimension 较低（Aghajanyan et al., 2020），即过参数化模型的梯度空间可以用低维参数化描述；② LoRA 的低秩更新等价于在特定子空间中进行梯度下降；③ 注意力机制的秩远低于隐藏维度（投影到低维仍保留足够信息）。

3. [实践应用题] 如果你需要在有限 GPU 显存（24GB）下训练一个 7B 参数的模型，并且需要支持 full fine-tuning 和 LoRA 两种模式，你会如何配置 ZeRO 优化和内存管理策略？
   - **答案/思考提示**：7B 模型 fp16 下约 14GB。ZeRO Stage 1 将 optimizer states 分片（可节省 12GB），Stage 2 加上梯度分片，Stage 3 加上参数分片。对于 24GB 显存：① 使用 ZeRO-2 + gradient checkpointing 可以 fit；② LoRA 模式下，冻结原模型参数，只训练 adapter（约 0.1% 参数），无需 ZeRO 也能运行；③ 如果要 full fine-tune，需要 ZeRO-3 + 混合精度 + activation checkpointing；④ 可考虑 QLoRA（nf4 量化 + LoRA）进一步降低显存。

4. [优缺点对比题] RoPE (Rotary Position Embedding) 和 ALiBi (Attention with Linear Biases) 是两种流行的位置编码方案，它们在处理长上下文外推时的行为有何本质差异？
   - **答案/思考提示**：RoPE 通过旋转操作将位置信息编码到 Q/K 向量中，具有相对位置衰减特性，长距离 token 之间的 attention 自然降低，适合外推。ALiBi 通过线性偏置直接加到 attention score 上，偏置随距离线性增长，外推时超出训练长度的位置偏置不可知，容易崩溃。RoPE 的外推能力来自其相对位置编码的性质，但需要配合 extended context window（如 YaRN）或 careful warm-up；ALiBi 在训练长度内表现稳定，但外推能力弱。

5. [深度思考题] VQVAE 和 VQ-GAN 等 tokenizer 方法在视频/图像生成中的作用是什么？它们与 Diffusion Model 的关系是什么？为什么很多现代 VLA 模型（如 ViViT、VideoGPT）仍使用 discrete tokenizer？
   - **答案/思考提示**：Discrete tokenizer 将像素级数据压缩到低维 latent code 空间（通常 1-2% 的压缩率），使得生成任务变成在 codebook 上的 autoregressive 建模，大幅降低计算复杂度。关系：① VQVAE/VQ-GAN 是 prior learning，Diffusion 可以作为 prior 或 decoder；② Diffusion 在 latent space 中操作比 pixel-level 更高效（LDM）；③ Discrete tokenizer + AR prior 适合需要精确控制（如语言模型架构统一）的场景。VLA 使用 discrete tokenizer 是因为可以复用 LLM 的训练范式和 infrastructure，AR 生成在高维动作空间中更高效。
