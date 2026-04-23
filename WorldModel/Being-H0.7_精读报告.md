# Being-H0.7: A Latent World-Action Model from Egocentric Videos

**论文**：Being-H0.7: A Latent World-Action Model from Egocentric Videos

**arXiv**：待查证（论文尚未发布于 arXiv）

**作者**：BeingBeyond Team

**机构**：BeingBeyond

**项目主页**：https://research.beingbeyond.com/being-h07

**发布日期**：2026-04-14

---

## 1 Motivation（问题背景）

### 1.1 VLA 的困境

视觉-语言-动作模型（VLA）在机器人操作任务上取得了显著进展，但在实际部署中存在根本性问题：

- **动作监督稀疏且有限**：任务规格和动作数据稀疏且多样性不足
- **行为崩溃**：VLA 往往将各种情况坍缩为少量重复行为，依靠虚假相关性而非学习物理可落地和组合性的动作表示
- **因果混淆**：缺乏对未来世界的显式建模

### 1.2 WAM 的探索与瓶颈

视频生成世界动作模型（WAM）尝试通过未来预测增强动作学习，但存在以下问题：

- **像素级计算成本高**：Cosmos Policy 需要超过 3000 H100 GPU 小时训练，而 Being-H0.5 仅需约 50 小时（60 倍效率差距）
- **推理延迟大**：需要迭代视频生成模拟未来 rollout
- **像素预测误差**：预测未来像素中的错误会传播到动作解码

### 1.3 核心问题

能否以一种高效的方式统一指令理解、世界建模和动作生成？

Being-H0.7 的解决思路：
- 在感知和动作之间引入紧凑的**潜在推理空间**
- 无需在推理时进行像素级 rollout
- 通过双分支设计对齐未来信息

---

## 2 一句话总结

Being-H0.7 通过**潜在世界动作模型**设计，在感知和动作之间引入紧凑潜在推理空间，结合先验-后验双分支联合对齐，在不进行像素级 rollout 的情况下实现高效的未来感知动作生成，在 6 个仿真基准上达到 SOTA 性能。

---

## 3 核心贡献

1. **潜在世界动作模型**：首次在 VLA 和 WAM 之间建立桥梁，通过潜在推理空间连接当前观测和未来预测
2. **双分支设计（Prior-Posterior）**：先验分支用于部署，后验分支（训练时）提供未来感知信号，通过隐藏状态对齐联合优化
3. **MoT 高效实现**：将双分支打包为单一 MoT 序列，通过双分支注意力掩码隔离先验/后验同时共享上下文
4. **Flow Matching + 联合对齐**：结合动作 flow matching 目标与未来感知对齐损失
5. **大规模预训练**：在 200,000 小时自我中心视频上预训练，15 倍于先前规模

---

## 4 方法详述

### 4.1 整体架构

![Being-H0.7 架构图](./images/being_h07_architecture.png)

```
┌─────────────────────────────────────────────────────────────┐
│                     Being-H0.7 Architecture                 │
├─────────────────────────────────────────────────────────────┤
│  Input: [Instruction x; Observation o₋ₕ:₀; State s]         │
│                              ↓                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Shared Context (ViT + Encoder)               │   │
│  └──────────────────────────────────────────────────────┘   │
│                              ↓                              │
│  ┌─────────────────────┬────────────────────────────────┐  │
│  │   Prior Branch      │     Posterior Branch            │  │
│  │   (Train+Inference) │     (Train-only)               │  │
│  │                     │                                │  │
│  │  [Latent Queries Q] │  [Future Embeddings zₚₒₛₜ]   │  │
│  │        ↓            │           ↓                   │  │
│  │  Noised Actions     │    Noised Actions              │  │
│  │  + Shared Context   │    + Shared Context           │  │
│  │        ↓            │           ↓                   │  │
│  │  Action Prediction   │    Action Prediction          │  │
│  └─────────────────────┴────────────────────────────────┘  │
│           ↓                        ↓                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Joint Alignment Loss (Lₐₗᵢₐₙ)               │   │
│  │  ℒ = ℒFₘ + wₐₗᵢₐₙℒₐₗᵢₐₙ + ℒᵣₑₐ                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 潜在推理空间（Latent Reasoning Space）

#### 4.2.1 问题定义

给定指令 $x$、观测上下文 $o_{-H:0}$（历史 horizon $H$）、状态 $s$，以及动作块 $a_{0:T}$。

#### 4.2.2 潜在查询插入

在动作块之前插入一组可学习的潜在查询 $Q \in \mathbb{R}^{K \times d}$：

$$
S = [x; o_{-H:0}; s; Q; a_{0:T}]
$$

其中：
- $K$ 是潜在查询数量
- $d$ 是隐藏维度
- $T$ 是动作块长度

#### 4.2.3 层间交互

潜在查询与指令、观测、状态、动作一起参与逐层 Transformer 传播：
- 整合多模态上下文的任务相关信息
- 组织成面向动作的潜在状态
- 引导下游动作生成

### 4.3 双分支设计（Dual-Branch Design）

#### 4.3.1 先验分支（Prior Branch）

**训练和推理都使用**：
- 条件：当前指令、观测上下文、状态、可学习的潜在查询
- 输出：动作预测

#### 4.3.2 后验分支（Posterior Branch）

**仅训练时使用**：
- 接收未来观测 $\tilde{o}_{0:T}$
- 将潜在查询替换为未来嵌入 $z_{\text{post}} = E(\tilde{o}_{0:T}) \in \mathbb{R}^{K \times d}$
- 揭示对未来动作决策真正有用的未来信息

#### 4.3.3 未来嵌入提取

$$
z_{\text{post}} = E(\tilde{o}_{0:T})
$$

其中 $E$ 是由冻结预训练 ViT 和 Perceiver 重采样器组成的时间视觉编码器。

### 4.4 联合对齐（Joint Alignment）

#### 4.4.1 对齐损失函数

令 $h_{\text{prior}}^\ell$ 和 $h_{\text{post}}^\ell$ 分别表示先验和后验分支在第 $\ell$ 层的对齐隐藏状态：

$$
\mathcal{L}_{\text{align}} = \frac{1}{L} \sum_{\ell=1}^{L} \| h_{\text{prior}}^\ell - h_{\text{post}}^\ell \|_2^2
$$

通过未来感知联合对齐，潜在推理空间不再仅仅是动作解码的中间载体，而是被显式塑造成编码未来相关、面向动作的结构。

### 4.5 MoT 高效实现（Mixture of Transformers）

#### 4.5.1 打包策略

将先验和后验分支打包为单一 MoT 序列：
- 两个分支共享相同的当前上下文 tokens
- 分支特定 tokens 占据不同的潜在推理位置
- 先验分支使用动作前的可学习潜在查询
- 后验分支使用相同形状的未来嵌入

#### 4.5.2 双分支注意力掩码

```
┌─────────────────────────────────────────────────────────────┐
│              Dual-Branch Attention Mask                      │
├─────────────────────────────────────────────────────────────┤
│  Shared Context: 可见于所有 tokens（先验和后验）            │
│  Prior Tokens: 彼此隔离，仅通过对齐的潜在推理状态通信        │
│  Posterior Tokens: 彼此隔离，仅通过对齐的潜在推理状态通信    │
│  位置 ID：两分支在对应 token 位置保持相同                    │
└─────────────────────────────────────────────────────────────┘
```

#### 4.5.3 计算效率

这种设计允许模型在单次 backbone 前向传播中实现双分支潜在世界动作模型 formulation。

### 4.6 损失函数详细设计

#### 4.6.1 Flow Matching 目标

设 $a$ 为动作标签，$t \in [0, 1]$ 为 flow time，$\epsilon \sim \mathcal{N}(0, I)$ 为采样噪声。

构造插值动作：$\tilde{a}_t = t \cdot a + (1 - t) \cdot \epsilon$，目标速度 $u_t = a - \tilde{a}_t$。

先验分支预测速度：$v_{\text{prior}}^\theta(a_t, c)$

后验分支预测速度：$v_{\text{post}}^\theta(a_t, c, z_{\text{post}})$

**Flow Matching 损失**：

$$
\mathcal{L}_{\text{prior}}^{\text{FM}} = \| v_{\text{prior}}^\theta(a_t, c) - u_t \|_2^2
$$

$$
\mathcal{L}_{\text{post}}^{\text{FM}} = \| v_{\text{post}}^\theta(a_t, c, z_{\text{post}}) - u_t \|_2^2
$$

$$
\mathcal{L}_{\text{FM}} = \mathcal{L}_{\text{prior}}^{\text{FM}} + \mathcal{L}_{\text{post}}^{\text{FM}}
$$

#### 4.6.2 正则化项

**范数正则化**（防止对齐状态坍缩到消失幅度）：

$$
R_{\text{norm}}(h) = [\text{ReLU}(\tau - \|h\|_2)]^2
$$

**秩正则化**（防止潜在推理空间的方向坍缩）：

设 $H \in \mathbb{R}^{B \times n}$ 为一批潜在隐藏状态在随机 $n$ 维子空间上的投影。归一化每行为单位范数得到 $\hat{H}$，计算 Gram 矩阵 $G = \hat{H} \hat{H}^\top$。

令 $\{\lambda_i\}_{i=1}^B$ 为 $G$ 的特征值，归一化谱 $p_i = \lambda_i / \sum_j \lambda_j$：

$$
R_{\text{rank}}(H) = -\sum_{i=1}^{B} p_i \log p_i
$$

**综合正则化**：

$$
\mathcal{L}_{\text{reg}} = w_{\text{norm}} R_{\text{norm}} + w_{\text{rank}} R_{\text{rank}}
$$

#### 4.6.3 总损失

$$
\mathcal{L} = \mathcal{L}_{\text{FM}} + w_{\text{align}} \mathcal{L}_{\text{align}} + \mathcal{L}_{\text{reg}}
$$

### 4.7 训练配置

| 参数 | 值 |
|------|-----|
| 观测 horizon $H$ | 4 |
| 动作块长度 $T$ | 20 |
| 潜在查询数量 $K$ | 16 |
| 对齐权重 $w_{\text{align}}$ | $10^{-3}$ |
| 正则化权重 $w_{\text{norm}}, w_{\text{rank}}$ | $10^{-4}$ |

### 4.8 一二阶段训练

#### Stage 1: 统一格式预训练

- 在混合人类和机器人操作数据上预训练
- 遵循 UniHand 2.0 的统一格式
- 提供跨具身的动作学习共享序列结构
- 允许以统一方式在异构操作轨迹上训练

#### Stage 2: 领域微调（可选）

在特定任务（如 LIBERO-plus）上微调。

---

## 5 训练与推理伪代码

```python
# Being-H0.7 训练伪代码

class BeingH07(nn.Module):
    def __init__(self, vision_encoder, backbone, action_head,
                 latent_query_num=16, hidden_dim=768):
        # 先验分支
        self.latent_queries = nn.Parameter(torch.randn(latent_query_num, hidden_dim))
        # 后验分支（仅训练）
        self.future_encoder = FutureEncoder()  # ViT + Perceiver Resampler

        # 其他组件
        self.vision_encoder = vision_encoder
        self.backbone = backbone
        self.action_head = action_head

    def forward_prior(self, x, obs, state, actions_noised, t):
        """
        先验分支前向传播（训练+推理）
        """
        # 1. 编码当前上下文
        ctx = self.vision_encoder(obs)  # [B, seq_len, d]
        ctx = torch.cat([x, ctx, state], dim=1)  # [B, total_len, d]

        # 2. 构建序列：[ctx; latent_queries; noised_actions]
        latent_q = self.latent_queries.unsqueeze(0).expand(bs, -1, -1)
        seq = torch.cat([ctx, latent_q, actions_noised], dim=1)

        # 3. Transformer 前向传播
        hidden = self.backbone(seq)

        # 4. 分离潜在推理位置和动作位置
        latent_hidden = hidden[:, -T-action_chunk_len:-action_chunk_len]
        action_hidden = hidden[:, -action_chunk_len:]

        # 5. 预测速度
        velocity = self.action_head(action_hidden, t)
        return velocity, latent_hidden

    def forward_posterior(self, x, obs, state, future_obs, actions_noised, t):
        """
        后验分支前向传播（仅训练）
        """
        # 1. 编码当前上下文
        ctx = self.vision_encoder(obs)
        ctx = torch.cat([x, ctx, state], dim=1)

        # 2. 编码未来观测为嵌入
        future_emb = self.future_encoder(future_obs)  # [B, K, d]

        # 3. 构建序列：[ctx; future_emb; noised_actions]
        seq = torch.cat([ctx, future_emb, actions_noised], dim=1)

        # 4. Transformer 前向传播
        hidden = self.backbone(seq)

        # 5. 分离位置
        latent_hidden = hidden[:, -T-action_chunk_len:-action_chunk_len]
        action_hidden = hidden[:, -action_chunk_len:]

        # 6. 预测速度
        velocity = self.action_head(action_hidden, t)
        return velocity, latent_hidden

    def compute_loss(self, batch):
        x, obs, state, future_obs, actions, t = batch

        # 采样噪声和构建插值
        eps = torch.randn_like(actions)
        actions_noised = t * actions + (1 - t) * eps
        target_velocity = actions - actions_noised

        # 先验分支
        v_prior, h_prior = self.forward_prior(x, obs, state, actions_noised, t)
        L_prior = F.mse_loss(v_prior, target_velocity)

        # 后验分支
        v_post, h_post = self.forward_posterior(x, obs, state, future_obs, actions_noised, t)
        L_post = F.mse_loss(v_post, target_velocity)

        # 对齐损失
        L_align = F.mse_loss(h_prior, h_post)

        # 正则化损失
        L_reg = self.compute_regularization(h_prior, h_post)

        # 总损失
        L_total = L_prior + L_post + 1e-3 * L_align + L_reg
        return L_total

    def compute_regularization(self, h_prior, h_post):
        # 范数正则化
        R_norm_prior = torch.relu(self.tau - h_prior.norm(p=2, dim=-1)).pow(2).mean()
        R_norm_post = torch.relu(self.tau - h_post.norm(p=2, dim=-1)).pow(2).mean()

        # 秩正则化
        R_rank_prior = self.entropy_of_spectrum(h_prior)
        R_rank_post = self.entropy_of_spectrum(h_post)

        L_reg = self.w_norm * (R_norm_prior + R_norm_post) + \
                self.w_rank * (R_rank_prior + R_rank_post)
        return L_reg


# ============ 推理（仅先验分支）============

def inference(model, instruction, observations, states):
    model.eval()
    noised_action = torch.randn(batch, T, action_dim).to(device)

    for step in range(num_diffusion_steps):
        t = schedule[step]  # e.g., linear from 1 to 0
        velocity = model.forward_prior(instruction, observations, states,
                                     noised_action, t)
        noised_action = noised_action + velocity * dt  # Euler integration

    return noised_action
```

---

## 6 实验结论

### 6.1 仿真基准

| Model | Size | LIBERO | LIBERO-plus | LIBERO-plus* | RoboCasa-50 | GR1 | CALVIN | CALVIN* | Robotwin2 |
|-------|------|--------|-------------|--------------|--------------|-----|--------|----------|-----------|
| **VLA** |
| π0 | 3B | 94.4 | 53.6 | - | 42.4 | - | 3.92 | 65.9/58.4 | - |
| π0-FAST | 3B | 85.5 | 61.6 | - | - | - | - | - | - |
| X-VLA | 0.9B | - | - | - | - | 4.43 | - | 72.9/72.8 | - |
| UniVLA | 8B | 95.5 | - | - | - | 4.63 | 4.41 | - | - |
| gr00t-N1.6 | 3B | 93.9 | - | - | 36.0 | 47.6 | 4.60 | 4.24 | - |
| π0.5 | 3B | 96.9 | 77.4 | - | 41.4 | - | 4.06 | 4.13 | 82.7/76.8 |
| starVLA | 4B | 96.5 | 77.0 | - | - | 48.8 | - | 88.2/88.3 | - |
| ABot-M0 | 4B | 98.6 | 80.5 | - | 58.3 | - | - | 81.2/80.4 | - |
| Being-H0.5 | 2B | 98.9 | 78.5 | 83.1 | 53.5 | - | 4.63 | 4.48 | - |
| **WAM** |
| UWM | - | - | 79.0 | - | - | 48.2 | - | - | - |
| UVA | - | - | - | - | 50.0 | - | - | - | - |
| VPP | 1.5B | - | - | - | - | - | 4.33 | - | - |
| DreamVLA | - | 92.6 | - | - | - | - | 4.44 | - | - |
| JEPA-VLA | - | 96.4 | 25.6 | - | - | - | - | 73.5/- | - |
| Cosmos-Policy | 7B | 98.5 | - | - | 67.1 | - | - | - | - |
| Fast-WAM | 6B | 97.6 | - | - | - | - | - | 91.9/91.8 | - |
| **Being-H0.7** | 3B | **99.2** | **82.1** | **84.8** | **62.1** | **49.2** | **4.67** | **4.48** | **90.2/89.6** |

### 6.2 真实机器人评估

| Suite | Being-H0.7 | Being-H0.5 | π0.5 | Fast-WAM |
|-------|------------|------------|-------|----------|
| Dynamic Scene | 70.0 | 59.4 | 46.7 | 45.8 |
| Physical Reasoning | 66.9 | 58.3 | 57.5 | 56.1 |
| Motion Reasoning | 67.5 | 60.6 | 49.2 | 58.3 |
| Long Horizon | 66.7 | 57.5 | 45.0 | 46.9 |
| Generalization | 67.5 | 55.0 | 49.2 | 58.3 |

### 6.3 推理效率

| Method | Latency (ms/step) | Memory (GB) |
|--------|------------------|-------------|
| Being-H0.7-UAC | 3.45 | 5.6 |
| Being-H0.7 | 3.61 | 5.6 |
| Being-H0.5-UAC | 12.45 | 6.8 |
| Being-H0.5 | 17.12 | 6.8 |
| π0.5-RTC | 28.81 | 15.6 |
| Fast-WAM | 41.26 | 9.3 |

---

## 7 KnowHow（核心洞察）

1. **VLA + WAM 的互补性**：VLA 擅长直接映射当前观测到动作，WAM 擅长未来演化建模。Being-H0.7 通过潜在推理空间将两者连接，而非简单选择其一。

2. **双分支对齐的物理意义**：先验分支学习"从当前上下文推理"，后验分支学习"从未来信息推理"。对齐损失强迫两者在潜在空间中一致，从而让先验分支隐式包含未来预测能力。

3. **潜在查询的灵活性**：与 JALA 将未来信息对齐到动作侧隐藏状态不同，Being-H0.7 使用显式的潜在查询 slots，提供更灵活的推理空间。

4. **MoT 打包的工程意义**：单次前向传播同时完成双分支训练，无需分离的 forward pass，大幅降低训练计算成本。

5. **范数 + 秩正则化的必要性**：对齐损失可能导致潜在状态坍缩到低维子空间。范数正则化防止幅度坍缩，秩正则化防止方向坍缩。

6. **未来嵌入提取策略**：冻结 ViT + Perceiver Resampler 比端到端训练更稳定，且无需额外的梯度流。

7. **动作 chunk 的设计**：T=20 的动作 chunk 提供了足够的上下文，同时保持推理效率。

8. **UAC 加速**：Unified Action Chunking 可将延迟从 3.61ms 降至 3.45ms，同时保持相同 GPU 内存占用。

---

## 8 arXiv Appendix 关键点总结

由于论文尚未发布于 arXiv，以下列出从 PDF 内容推断的补充内容：

- **A**: 更多真实机器人任务细节（12 个任务的详细描述）
- **B**: 不同具身平台（PND Adam-U、Unitree G1、Franka FR3）的具体配置
- **C**: 更多消融实验（MoT 组件贡献、对齐权重敏感性）
- **D**: 潜在查询数量 K 的敏感性分析
- **E**: 未来观测 horizon 的影响
- **F**: 更多定性结果（注意力可视化）
- **G**: 与其他世界模型（Cosmos Policy、Fast-WAM）的详细对比

---

## 9 总结

### 三大核心贡献

1. **潜在世界动作模型范式**：通过在感知和动作之间引入紧凑潜在推理空间，连接 VLA 和 WAM 的优势
2. **先验-后验双分支联合对齐**：训练时利用未来信息塑造潜在空间，推理时保持高效单分支部署
3. **MoT 高效实现**：单次前向传播完成双分支训练，结合 flow matching 和对齐损失

### 最重要洞察

**潜在空间是连接现在和未来的桥梁**。不同于直接在像素空间预测未来（计算昂贵且有误差），Being-H0.7 将未来信息压缩到潜在推理空间，通过对齐损失强迫先验分支隐式学习未来预测能力。这种"推理时无需 rollout"的设计是效率关键。

### 与相关工作对比

| 方法 | 潜在空间 | 推理时 Rollout | 效率 |
|------|---------|---------------|------|
| VLA | 无 | 否 | 高 |
| WAM (像素) | 无 | 是 | 低 |
| JALA | 动作侧隐藏状态 | 否 | 中 |
| **Being-H0.7** | **显式潜在查询** | **否** | **高** |

---

**参考链接**：

| 资源 | 链接 |
|------|------|
| **论文 PDF** | [being-h07.pdf](https://research.beingbeyond.com/projects/being-h07/being-h07.pdf) |
| **项目主页** | [Being-H0.7](https://research.beingbeyond.com/being-h07) |
| **BeingBeyond** | [beingbeyond.com](https://www.beingbeyond.com) |
