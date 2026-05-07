## 日期：20260507

### LLM 领域（通用思考题）

**基于近期 Foundation Model 前沿进展的综合思考题——注意力机制、Tokenizer 与训练稳定性专题**

1. [基础概念题] FlashAttention 是如何通过 IO-Awareness（IO感知）设计来加速注意力计算的？它和标准 Attention 在计算复杂度上没有本质差异，但为什么实际运行快了几个数量级？
   - **答案/思考提示**：标准 Attention 需要将完整 attention matrix（ N×N ）存入 HBM，然后逐 block 计算再合并，需要 O(N²) HBM IO 操作。FlashAttention 通过 tiling 和 softmax 分解，将计算分成小块在 SRAM 中完成，减少了 HBM 读写次数。关键洞察是：HBM 带宽远低于算力，IO 是瓶颈而非计算量。FlashAttention 的加速来自：① 减少 HBM 访问量（从 O(N²) 次读写降到 O(N²d) 伴随大矩阵分块）；② 避免存储大的中间结果（N×N matrix）。它的 recomputation 策略用额外计算换内存，也显著降低了峰值显存。

2. [深度思考题] 旋转位置编码（RoPE）和相对位置编码（如 T5 的 bias、RPE）在建模序列位置关系上有什么本质差异？为什么 LLaMA 等主流模型选择 RoPE 而非相对位置编码？
   - **答案**：RoPE 通过将位置信息编码为旋转矩阵，使 token 之间的注意力分数只依赖于它们的相对位置（通过内积的旋转不变性），这天然支持无限上下文长度（因为旋转角度可以无限叠加）。T5 的相对位置编码通过可学习的标量偏置添加到注意力分数中，表达能力更强但需要预先设定最大长度，且在处理超长序列时需要位置插值。RoPE 的优势在于：① 无需额外参数；② 推理时位置信息是隐式的，支持 length extrapolation（GPT-4 早期版本通过这种方式处理超长上下文）；③ 实现简单，和标准 Attention 兼容。mRoPE（多维 RoPE）进一步扩展到多维位置（如图像、视频）。

3. [实践应用题] LoRA（Low-Rank Adaptation）的核心数学假设是什么？为什么它假设大模型的权重更新矩阵是低秩的？这个假设在实际大模型训练中是否总是成立？
   - **答案/思考提示**：LoRA 假设 `ΔW = W_post - W_pre` 是低秩的，即 `ΔW = BA, B ∈ R^{d×r}, A ∈ r^{r×k}, rank << min(d,k)`。这个假设的motivation 是：模型去适应新任务时，不需要 full-rank update，有效自由度很低（intrinsic dimensionality hypothesis）。但这个假设不总是成立——对于需要大幅改变模型行为的任务（如完全切换语言 domain），低秩假设可能限制适应能力。实践中 LoRA 的 r 通常选 8~64，Q/V attention matrices 加 RoPE 后效果不错。A 和 B 用高斯初始化，训练时固定 W，梯度更新 A/B。最新研究表明 combining multiple adaptation methods（如 LoRA + prefix tuning）有时比单独使用更好。

4. [优缺点对比题] VQVAE 和 Diffusion Model 在视觉 Tokenizer 设计上各有什么优缺点？为什么世界模型（如 Fast-WAM）和 VLA 模型（如 π0）选择了不同的视觉 tokenization 策略？
   - **答案/思考提示**：VQVAE 优点：codebook 索引使 latent space 离散，推理快，易于和 autoregressive language model 集成；缺点：codebook collapse、hard assignment 丢失细节、重建质量受 codebook size 影响。Diffusion tokenizer（用 VAE + diffusion prior）优点：重建质量高，能建模 multi-modal 分布；缺点：latent space 连续，需要额外的 diffusion process 来生成，推理更慢。Fast-WAM 等 world model 需要快速 imagination rollout，选择 VQVAE 是因为推理速度关键；π0 等 VLA 需要高保真动作生成，选择 continuous diffusion 是因为动作精度要求更高。关键权衡是：discrete token 利于 sequence modeling，continuous latent 利于保真度。

5. [深度思考题] 为什么大模型训练中经常使用 gradient checkpointing（梯度检查点）技术？它是如何在计算时间和显存之间做 trade-off 的？对于一个 70B 参数的模型，启用 gradient checkpointing 后实际能节省多少显存？
   - **答案/思考提示**：Gradient checkpointing 的核心思想是：不存储 forward pass 中所有激活值，只存储部分 checkpoints，backward 时重新计算被丢弃的激活值。Trade-off：用额外 forward 计算换取显存（通常是线性节省）。70B 模型 full activation 需要约 ~6TB（每参数 72 bytes for bf16 + optimizer states + gradients），checkpointing 后可降至 ~1-2TB。具体节省取决于模型结构和 batch size——transformer 的激活主要来自 attention matrices (O(N²) per layer)。常用策略是 only checkpoint input of each transformer block，让每个 block 内部不做 checkpointing。PyTorch 实现中 `torch.utils.checkpoint.checkpoint()` 可以直接使用。对于 70B+ 模型，checkpointing 是能在单卡或多卡上训练的关键技术，否则激活值显存会直接 OOM。
