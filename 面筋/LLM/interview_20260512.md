## 日期：20260512

### LLM 领域（FM基础知识）

**论文：（基于 TiTok & VQVAE & RoPE 深度思考）**

1. [基础概念题] TiTok 和 VQVAE 都属于 discrete VAE (dVAE) 家族，但 TiTok 在 tokenization 策略上与标准 VQVAE 有何本质区别？这种区别如何影响它们在视频生成任务中的压缩效率和重建质量？
   - **答案/思考提示**：标准 VQVAE 是per-token quantization：每个空间位置独立量化，codebook size 通常 8K-32K，产生 H×W 个 tokens。TiTok 采用了 1D discrete tokenization：将 H×W×C 的 3D tensor flatten 成 1D sequence 后量化，codebook 可以扩展到更大的规模（如 16K-32K），同时产生更少的 tokens（如 16×16 图像 → 256 tokens vs VQVAE 可能需要 256×1=256 但每个 token 独立）。这使得 TiTok 在压缩率上更高（4096×4096×3 → 64 tokens，压缩率 ~98.5%），更适合 video generation（低带宽传输）。代价是空间结构被破坏，需要更强大的 decoder 恢复空间信息。

2. [深度思考题] RoPE 中通过旋转矩阵将位置编码融入 Q/K 向量，使得 attention score $q_m^T k_n$ 自然依赖于相对位置 $m-n$。请从线性代数角度解释为什么这种"旋转"操作能够编码位置信息？为什么它比其他位置编码（如 Sinusoidal、Learnable）更适合长上下文外推？
   - **答案/思考提示**：RoPE 对 q/k 向量做旋转：$R_m^q Q_m$ 和 $R_n^k K_n$，其中 $R_m = \text{diag}(\cos(m\theta), \sin(m\theta))$。内积 $q_m^T k_n = (R_m^q Q_m)^T (R_n^k K_n) = Q_m^T R_{m-n} K_n$，即相对位置 $m-n$ 通过旋转矩阵 $R_{m-n}$ 直接调制 attention，不依赖绝对位置编码。优势：① 相对位置编码的外推性天然优于绝对位置（超出训练长度时，绝对位置编码完全未知，但相对位置编码仍可通过 $m-n$ 的旋转外推）；② RoPE 的旋转是连续操作，warm-up 后可以处理更长的 context；③ 计算上高效，不需要额外的位置偏置项。ALiBi 的线性偏置是离散的，超出范围后完全失效，不如 RoPE 的旋转平滑。

3. [优缺点对比题] FlashAttention 和 FlashAttention-2 在算法层面有哪些关键改进？这些改进如何影响了 BF16 训练中的内存占用和计算吞吐量的权衡？
   - **答案/思考提示**：FA1 → FA2 的关键改进：① 减少了 non-matrix multiply (non-MatMul) FLOPs 的比例（从 ~50% 降至 ~30%），显著提升了 GPU utilization；② 更好的 thread block tiling：FA1 的 tiling 对某些序列长度不够优化，FA2 改进了 grid 和 block 分配，对不同 sequence length 都能高效；③ 序列长度不再是平方关系限制，FA2 的 tiling 使得长序列时 memory 仍为 $O(N)$ 而非 $O(N^2)$。在 BF16 训练中，attention 是 memory-bound 操作而非 compute-bound，FA2 通过减少 HBM 访问次数（从 $O(N^2)$ 次降至 $O(N^2/d)$）使得内存带宽不再是瓶颈，throughput 提升约 2-3x。

4. [实践应用题] 假设你在训练一个 34B 参数的 MoE (Mixture of Experts) 模型，EP (Expert Parallelism) = 8，每个 token 只会激活 2 个 experts。请从显存、通信开销和负载均衡三个角度分析：(1) EP=8 是否合理？(2) 如何设计 expert capacity 和 capacity boosting 策略来避免 expert imbalance？
   - **答案/思考提示**：EP=8 的合理性：需要看硬件拓扑，8 个 EP 意味着 8 个 GPU 之间的 all-to-all 通信，如果 GPU 在同一 node 内（NVLink）通信效率高。显存：每个 GPU 只需持有 $\frac{34B}{8} \times 2/6$（假设 6 个 experts 被分散）而非全量参数，但需要存储所有 experts 的 embedding 和路由器的完整副本。负载均衡问题：某些 expert 可能收到过多 tokens（hot experts），其他 expert 空转。解决方案：① Expert capacity：每个 expert 有固定 capacity，超出部分需要 drop 或 overflow 到其他 GPU；② Capacity boosting：定期统计 expert utilization，对利用率低的 expert 增加 capacity 上限；③ Auxiliary loss：添加 expert selection diversity loss，鼓励更均匀的分布；④ Token dropping：当某个 batch 中某些 expert 超载时随机 drop 部分 tokens。

5. [深度思考题] mRoPE（多维旋转位置编码）将位置编码从 1D 扩展到多维，请分析：mRoPE 是如何在视觉Transformer（如 PETR、VGGT）中处理 2D/3D 空间位置的？为什么在视觉任务中需要比 NLP 更复杂的位置编码方案？这与 SwinTransformer 的相对位置编码有何本质区别？
   - **答案/思考提示**：mRoPE 将位置分解为多个维度：1D RoPE 中每个 token 位置 $m$ 对应旋转角度 $m\theta$；mRoPE 在 2D 图像中对 (x, y) 坐标分别使用不同频率的旋转矩阵 $R_x^{(d)}$ 和 $R_y^{(d)}$，组合起来编码完整的 2D 位置。在视频中甚至加入时间维 $R_t^{(d)}$。本质原因：图像/视频的坐标是有物理意义的——x 和 y 方向的距离关系不同（透视畸变），时间维与其他两维的性质也不同（时间单向性 vs 空间各向同性）。SwinTransformer 的相对位置编码使用可学习的偏置 $B_{ij}$ 加到 attention score 上，本质上是 2D 卷积式的归纳偏置，只能处理固定大小的空间关系。mRoPE 是更具表达力的连续位置编码，通过不同频率的旋转，自然地建模了任意相对位置关系，且具备外推能力。
