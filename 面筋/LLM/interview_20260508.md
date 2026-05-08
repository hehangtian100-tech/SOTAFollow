## 日期：20260508

### LLM 领域

**通用领域思考题（无当日新论文）**

1. [基础概念题] LoRA 的核心思想是什么？为什么它能大幅降低微调大模型的计算成本？
   - 答案/思考提示：LoRA 冻结预训练权重 W_0，只训练低秩分解 ΔW = BA（r × d × r × d，r << min(d,k)）。更新参数量从 O(d×k) 降至 O(2×r×(d+k))。推理时可合并 W_0 + ΔW，无额外延迟。

2. [深度思考题] FlashAttention 为什么能显著加速 Attention 计算且不占用 O(N²) 显存？其核心算法创新是什么？
   - 答案/思考提示：核心创新是 IO-aware tiling + SRAM 级别的矩阵分块计算。利用 register/SRAM 的高带宽，将 N×N attention matrix 的 softmax 分解为块级别计算（block-wise softmax），通过 Scandinavian algorithm 累积正确归一化因子，避免 materialization 完整 attention matrix。

3. [实践应用题] 在实际部署中，如果需要将 70B 规模的模型部署到单卡 80GB 显存，有哪些关键技术可用？它们的组合策略是什么？
   - 答案/思考提示：① FP16/BF16 量化（2-4x 压缩）；② QLoRA（4-bit NF4 + gradient checkpointing）；③ Tensor parallelism（张量并行）；④ KV cache 量化；⑤ 组合：QLoRA 量化 + TP4 可在 80GB 部署 70B；70B 原版需要约 140GB。

4. [优缺点对比题] RoPE（Rotary Position Embedding）和 ALiBi（Attention with Linear Biases）在处理长上下文方面各有什么优劣？
   - 答案/思考提示：RoPE：相对位置通过旋转矩阵编码，可外推但外推能力有限（需 extended context training）。ALiBi：线性偏置，无额外参数，外推性好但 inductive bias 较强。RoPE 已成为主流（如 LLaMA、Qwen），ALiBi 在一些模型中使用（MISTRAL）。

5. [深度思考题] 自回归语言模型在生成时面临「exposure bias」问题（训练时用 ground truth，推理时用预测）。有哪些方法可以缓解这个问题？
   - 答案/思考提示：① Scheduled sampling（逐步增加使用模型预测的比例）；② DPO/PPO 等序列级 RL 训练；③ Professor forcing（对抗正则化）；④ 免费午餐：CLM 本身通过大规模数据已部分缓解；⑤ 在微调阶段使用 human feedback 校准。
