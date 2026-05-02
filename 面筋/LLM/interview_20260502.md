## 日期：20260502

### LLM 领域

**通用思考题（当日无新论文）**

1. [基础概念题] 请解释 LLM 训练中的 SFT（Supervised Fine-Tuning）、RLHF 和 DPO（Direct Preference Optimization）三种训练范式的核心区别和各自优缺点。
   - **答案/思考提示**：
     - **SFT**：在指令-响应对上做监督学习，最小化语言模型的交叉熵损失。优点是简单高效、训练稳定；缺点是只学习正例，无法利用对比信息，且容易出现「暖迪性（sycophancy）」——过度迎合用户而非坚持事实。
     - **RLHF**：先用 Reward Model 学习人类偏好，再用 PPO 等强化学习算法优化策略。优点是能利用对比信息，学习隐式的偏好函数；缺点是训练流程复杂、RRHF/Mixtral 等方法存在 reward hacking 问题，PPO 训练不稳定。
     - **DPO**：将 RLHF 的 reward maximization 转化为直接的对比损失（contrastive loss），绕过 reward model 直接用偏好数据优化策略。优点是训练更稳定、流程更简单；缺点是对数据质量要求更高，偏好数据的分布直接影响最终效果，且理论上是 RLHF 的上近似，在某些情况下效果可能略逊于完整 RLHF。

2. [深度思考题] FlashAttention 和 FlashAttention-2 在算法层面做了哪些关键优化，使得 Transformer 的训练速度大幅提升？这些优化为什么对长上下文场景尤为重要？
   - **答案/思考提示**：FlashAttention 的核心优化是 **IO-aware 切片算法**，通过tiling技术将注意力矩阵分成小块处理，使得每次只需要的中间结果（softmax归一化系数）常驻 SRAM，避免 HBM（High Bandwidth Memory）的大规模读写。具体包括：①分块计算：避免实例化完整的 N×N 注意力矩阵；②在线 softmax：分块计算 softmax 的分子分母，避免同时存储所有行；③重计算：在反向传播时重新计算注意力矩阵，而非存储它。FlashAttention-2 进一步优化了thread block的任务分配（减少idle warp）和序列并行的效率。对长上下文场景尤为重要是因为：标准注意力的内存复杂度是 O(N²)（N=序列长度），长文本会导致内存爆炸；FlashAttention 将内存降到 O(N)，同时保持计算吞吐量接近理论上限。

3. [实践应用题] 你正在用 LoRA 方法微调一个 7B 参数的 LLM，要求在单卡 A100（80GB）上完成训练。请说明 LoRA 的核心原理、超参数选择策略（如 rank、alpha、target modules），以及如何判断 LoRA 微调是否收敛良好。
   - **答案/思考提示**：
     - **核心原理**：LoRA 在预训练权重的旁边增加低秩分解矩阵 A 和 B，冻结原始权重 W_0，训练时只更新 ΔW = BA。由于 rank=r << d_model，参数量从 O(d_model²) 降到 O(2·r·d_model)。推理时将 W_0 + ΔW 合并，不引入额外推理延迟。
     - **超参数选择**：rank 通常选 4/8/16/32，较大的 rank 表达能力更强但参数量更大；alpha/rank 的比例（scale factor）通常设为 2；target modules 通常选 q_proj, v_proj（Attention 的 Query 和 Value），有时加上 k_proj, o_proj；一般建议先在小 rank（如 8）上快速验证，再逐步增大。
     - **收敛判断**：①训练 loss 下降且趋于平稳；②验证集上的困惑度（perplexity）不再下降；③实际任务评测（如 HumanEval、MMLU）有提升；④检查 LoRA 权重矩阵的奇异值分布，确保不是过于低秩。

4. [优缺点对比题] 对比 VQVAE/TiTok 等基于离散码本的视觉 Tokenizer 与基于 Diffusion 的视觉 Tokenizer（如 VAR、SVD），它们在视频生成任务中的各自优劣是什么？
   - **答案/思考提示**：
     - **VQVAE/TiTok（离散码本）**：将图像压缩为离散 token 序列，token 有固定码本大小。优点是：①训练稳定，VAE 架构成熟；②推理高效，离散 token 可用斋戒自回归生成；③与 LLM 架构天然兼容（文本 token + 视觉 token 统一表征）。缺点：①码本利用率问题（码本崩溃、某些码本从未被使用）；②离散化引入量化误差；③重建质量受码本大小限制。
     - **Diffusion/SVD（连续表示）**：直接在连续空间生成，不做离散化。优点：①无量化误差，重建质量更高；②能更好地保留细节信息。缺点：①推理延迟高（需要多步去噪）；②与 LLM 架构整合更困难；③难以直接复用大语言模型预训练知识。
     - **视频生成场景**：对于需要高帧率、高运动保真度的任务（如 Sora、Runway），连续表示的 Diffusion 优势更大；对于需要与 LLM 深度整合、追求推理效率的视觉-语言任务（如视频问答、视频字幕），离散 token 更有优势。
