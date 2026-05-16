## 日期：20260516

### LLM 领域（FM基础知识）

**论文：** 无新论文，基于 FM 基础知识领域核心概念综合整理

**基础概念题：**

1. **Transformer 中 Self-Attention 的计算复杂度是多少？为什么它成为大模型的瓶颈？有哪些优化方法？**
   - 答案/思考提示：Self-Attention 复杂度 O(n²d)，n 是序列长度，d 是 hidden dimension。长序列时二次方增长成为瓶颈。优化方法：① FlashAttention（IO-aware exact attention）；② Sparse Attention（局部+全局）；③ Linear Attention（kernel trick）；④ Low-rank Attention（Linformer/Performer）。

2. **FlashAttention 如何通过 IO-Awareness 提升计算效率？它与传统 Attention 的数学等价性如何保证？**
   - 答案/思考提示：FlashAttention 利用 GPU memory hierarchy，将 attention 计算 tiling 分块，避免 HBM 读写。数学上完全等价，只是实现方式不同。关键技巧是 online softmax 和分块计算 attention 的融合。

3. **LoRA 的核心思想是什么？为什么它能够高效微调大模型？它与 Full Fine-tuning 相比有哪些 trade-offs？**
   - 答案/思考提示：LoRA 在预训练权重旁添加低秩 adapters，用 A、B 两个矩阵替代 full fine-tuning。Trade-offs：① 参数量大大减少（万倍级）；② 但表达能力受限——对于需要大幅改变模型行为的任务可能不如 full fine-tuning；③ 推理时可合并为单矩阵，无额外 latency。

4. **什么是 VQVAE（Vector Quantized VAE）？它与 VQ-VAE-2 有何改进？为什么 VQ 是视觉 Tokenizer 的主流方法？**
   - 答案/思考提示：VQVAE 用 codebook 将 latent vectors 量化到离散空间，codebook 也称 latent vocabulary。VQ-VAE-2 引入 hierarchical latent 提高重建质量。VQ 之所以流行：①离散表示便于与语言模型结合（统一 token space）；② codebook 可学习且 efficient；③ TiTok 等工作证明其有效性。

5. **什么是 RoPE（Rotary Position Embedding）？它如何解决 Transformer 中的位置编码外推问题？mRoPE 为何对 VLA 视频理解重要？**
   - 答案/思考提示：RoPE 用旋转矩阵编码位置信息，具有良好的相对位置建模能力。外推问题指训练时序列短但推理时长，RoPE 通过调整旋转角度的 base 参数部分缓解。mRoPE 将位置分为 temporal、height、width 多个维度，使模型能区分同一帧内不同空间位置和不同帧的时间顺序——这对视频理解至关重要。

**深度思考题：**

6. **从信息论角度分析：为什么 Attention 需要 O(n²) 复杂度？能否设计一个理论上 O(n) 的 Attention 变体且保持等价表达能力？**
   - 答案/思考提示：O(n²) 源于 query-key-value 的全连接交互。O(n) attention 如 Linear Attention 通过 kernel trick 近似，但表达能力受限（只能建模 query-key 的某种特定 interaction）。理论上 sparse attention 在某些任务上可接近 full attention，但完美等价且 O(n) 的设计仍是 open problem。

7. **LoRA 中 rank r 的选择有何 trade-offs？为什么通常 r 在 4-64 之间而非更大？是否存在"最优 rank"的概念？**
   - 答案/思考提示：r 越大表达能力越强但参数量增加；r 越小越高效但可能欠拟合。实践中 r 和下游任务难度相关——简单任务 r=4-8 足够，复杂任务需要 r=16-64。值得注意的是 LoRA 的 effective rank 通常比预设 r 小，实际 information capacity 取决于数据。论文 "LoRA+" 指出不同学习率对 A、B 矩阵可能更好。

8. **ZeRO（Zero Redundancy Optimizer）的 Stage 1/2/3 分别优化了什么？为什么 ZeRO-3 对大模型训练至关重要但也会引入通信开销？**
   - 答案/思考提示：ZeRO-1 分 optimizer states，ZeRO-2 分 gradients + optimizer states，ZeRO-3 分 parameters + gradients + optimizer states。Stage 3 让每 GPU 只存储 1/N 的参数，但 ALLREDUCE 通信随 stage 增加，N 小时可能成为 bottleneck。FSDP 是 ZeRO-3 的分布式实现。

**实践应用题：**

9. **假设你要训练一个 7B 参数的视觉语言模型，需要处理 1024x1024 图像并生成文本描述。请设计你的 training pipeline，包括视觉 tokenizer、LLM backbone、训练策略等关键组件。**
   - 答案/思考提示：
     - 视觉 tokenizer：可用VQVAE/TiTok/SigLIP encoder
     - LLM backbone：LLaMA/Qwen 等
     - Training：两阶段——① frozen LLM + train vision encoder（matching）；② full fine-tuning 或 LoRA
     - 关键设计：vision-language alignment loss、是否使用 cross-attention vs full LLM fine-tuning
     - 训练策略：curriculum learning from low-res to high-res

10. **在部署大模型时，如何决定使用 quantization、pruning 还是 distillation？三者有何本质区别和适用场景？**
    - 答案/思考提示：
      - Quantization：降低权重精度（FP16→INT8→INT4），几乎不改变架构，loss 相对可控
      - Pruning：移除不重要的权重/attention head，改变架构，需要 retraining
      - Distillation：训练小模型模仿大模型输出，训练成本高但可得到专门的小模型
      - 通常组合使用：quantize 后 fine-tune 恢复质量

**优缺点对比题：**

11. **对比 FlashAttention v1 vs v2 vs v3 的演进路径。每一代主要解决了什么问题？**
    - 答案/思考提示：
      - FA1：提出 IO-aware attention tiling，基本思想
      - FA2：重构 kernel 实现，warp specialization，parallel work reduction，2-4x speedup
      - FA3：适用不同 GPU architectures，更好的 occupancy，对于不同 sequence length 更鲁棒
      - 核心演进是工程优化 + 理论改进的结合

12. **对比 diffusion-based image generation 和 autoregressive image generation 的优劣势。**
    - 答案/思考提示：
      - Autoregressive：统一框架（next token prediction），与 LLM 架构一致；但采样慢（需要 N steps），难以建模多模态分布
      - Diffusion：采样质量高，多步去噪稳定；但需要大量 steps 生成，latent space 训练复杂
      - 趋势：MAR（Masked Autoregressive）和 MaskGIT 结合两者优点，兼具 AR 的统一性和 diffusion 的高质量

---

**参考文献**：
- FlashAttention: https://arxiv.org/abs/2205.14135
- LoRA: https://arxiv.org/abs/2106.09685
- VQVAE: https://arxiv.org/abs/1712.09663
- TiTok: https://arxiv.org/abs/2406.07550
- RoPE: https://arxiv.org/abs/2104.09864
- ZeRO: https://arxiv.org/abs/1910.02054
