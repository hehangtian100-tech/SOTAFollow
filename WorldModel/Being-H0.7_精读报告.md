# Being-H0.7: A Latent World-Action Model from Egocentric Videos

**论文**：Being-H0.7: A Latent World-Action Model from Egocentric Videos

**arXiv**：待查证（论文尚未发布于 arXiv，2026-04-14）

**作者**：BeingBeyond Team

**机构**：BeingBeyond

**项目主页**：https://research.beingbeyond.com/being-h07

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

### 4.1 整体架构图

![Being-H0.7 架构图](https://research.beingbeyond.com/projects/being-h07/images/arch.webp)

> **图1**：Being-H0.7 整体架构图。左侧展示先验分支（训练+推理）和后验分支（仅训练）的双分支设计，右侧展示潜在推理空间如何通过联合对齐连接当前上下文和未来信息。

![直觉图](https://research.beingbeyond.com/projects/being-h07/images/intuition.webp)

> **图2**：潜在推理的核心直觉。引入可学习的潜在查询作为当前上下文和动作之间的中介，通过双分支对齐使先验分支隐式获得未来感知能力。

### 4.2 潜在推理空间（Latent Reasoning Space）

#### 4.2.1 问题定义

给定指令 $x$、观测上下文 $o_{-H:0}$（历史 horizon $H$）、状态 $s$，以及动作块 $a_{0:T}$。

#### 4.2.2 潜在查询的来源与依据

**潜在查询是一组可学习的（learnable）tokens**，这是本文的核心设计选择。

**设计依据**：

1. **中间推理空间的需求**：模型需要将抽象的多模态语义逐步组织成面向动作的紧凑表示。如果直接映射到低层次动作，解码器负担过重。

2. **与 JALA 的关键区别**：JALA 将未来信息对齐到**动作侧隐藏状态**，而 Being-H0.7 移到一个**显式的潜在查询 slots**，提供更灵活的推理空间。

3. **端到端学习的优势**：可学习参数让模型自动发现什么样的潜在表示对动作生成最有价值，无需人工设计。

**数学形式**：

在动作块之前插入一组可学习的潜在查询 $Q \in \mathbb{R}^{K \times d}$：

$$
S = [x; o_{-H:0}; s; Q; a_{0:T}]
$$

其中：
- $K = 16$ 是潜在查询数量（论文消融实验确定）
- $d$ 是隐藏维度
- $T = 20$ 是动作块长度

#### 4.2.3 层间交互机制

潜在查询与指令、观测、状态、动作一起参与逐层 Transformer 传播：

```
Token 序列结构（按位置顺序）：
┌─────────────────────────────────────────────────────────────────────┐
│ [x]        │ [o₋ₕ...o₀] │ [s] │ [Q₁...Q_K] │ [a₀...a_T] │
│  指令        │   观测历史    │ 状态 │  潜在查询   │   动作块     │
└─────────────────────────────────────────────────────────────────────┘
```

**层间交互过程**：
- **第 1 层**：潜在查询 Q 从上下文 tokens（指令、观测、状态）接收信息，开始组织任务相关表示
- **中间层**：潜在查询继续与上下文和动作 tokens 交互，逐步整合多模态信息
- **动作位置**：潜在推理状态在动作生成位置输出，引导下游动作预测

**为什么需要逐层传播**：
- 避免一次性映射：让模型有充足容量在多层中逐步提炼信息
- 类似于人类"思考-行动"的决策过程

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

其中 $E$ 是由**冻结预训练 ViT** 和 **Perceiver Resampler** 组成的时间视觉编码器：
- ViT 提取每帧视觉特征
- Perceiver Resampler 将时序帧聚合为 K 个紧凑嵌入

### 4.4 联合对齐（Joint Alignment）

#### 4.4.1 对齐损失函数

令 $h_{\text{prior}}^\ell$ 和 $h_{\text{post}}^\ell$ 分别表示先验和后验分支在第 $\ell$ 层的对齐隐藏状态：

$$
\mathcal{L}_{\text{align}} = \frac{1}{L} \sum_{\ell=1}^{L} \| h_{\text{prior}}^\ell - h_{\text{post}}^\ell \|_2^2
$$

#### 4.4.2 对齐损失如何充当 Bridge

**对齐损失的核心作用**：强迫先验分支的潜在状态与后验分支（拥有未来信息）的潜在状态在潜在空间中"对齐"。

**工作机制**：

```
训练阶段：
┌─────────────────────────────────────────────────────────────────────┐
│  当前上下文 (x, o, s)                                               │
│       ↓                                                                 │
│  ┌──────────────────────┬──────────────────────────────────┐      │
│  │   Prior Branch       │     Posterior Branch               │      │
│  │   + Learnable Q     │     + Future Embeddings           │      │
│  │         ↓           │           ↓                        │      │
│  │   h_prior           │     h_posterior (有未来信息)        │      │
│  │         ↓           │           ↓                        │      │
│  └──────────────────────┴──────────────────────────────────┘      │
│                           ↓                                          │
│                    对齐损失 ℒ_align                                  │
│                    (强迫 h_prior ≈ h_posterior)                       │
└─────────────────────────────────────────────────────────────────────┘

推理阶段：
┌─────────────────────────────────────────────────────────────────────┐
│  当前上下文 (x, o, s)                                               │
│       ↓                                                                 │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │   Prior Branch (仅有当前上下文 + Learnable Q)          │      │
│  │         ↓                                               │      │
│  │   h_prior (已被训练成 ≈ 后验分支的潜在状态)               │      │
│  │         ↓                                               │      │
│  │   动作预测 (隐式包含未来感知能力)                        │      │
│  └──────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

**本质**：对齐损失将"未来信息"从后验分支"蒸馏"到先验分支，使先验分支在推理时无需未来观测也能利用潜在空间中的未来相关表示。

### 4.5 MoT 高效实现（Mixture of Transformers）

#### 4.5.1 打包策略

将先验和后验分支打包为单一 MoT 序列，而非两次独立前向传播：

```
┌─────────────────────────────────────────────────────────────────────┐
│              MoT Packed Sequence (Single Forward Pass)                │
├─────────────────────────────────────────────────────────────────────┤
│  Shared Context Tokens: [x; o₋ₕ:₀; s]  ← 两分支共享                 │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  Prior-Specific: [Q₁...Q_K; a₀...a_T]  (位置1)             │   │
│  │  Posterior-Specific: [z₁...z_K; a₀...a_T]  (位置2)        │   │
│  │  注：两分支的 token 形状相同，位置 ID 相同                    │   │
│  └────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

#### 4.5.2 双分支注意力掩码设计

**核心原则**：共享上下文对两分支都可见，但先验和后验的特定 tokens 彼此隔离。

```
注意力掩码矩阵（行=Query，列=Key/Value）：

              Shared   Prior-Specific   Posterior-Specific
             ──────── ──────────────── ───────────────────
Shared      │   ✓    │       ✓        │        ✓         │
            │        │               │                   │
Prior-Spec  │   ✓    │       ✓        │        ✗         │
            │        │               │     (隔离)         │
            │        │               │                   │
Post-Spec   │   ✓    │       ✗        │        ✓         │
            │        │     (隔离)     │                   │
            └────────┴───────────────┴───────────────────┘

✓ = 可以 attend   ✗ = 不能 attend（masked）
```

**关键设计点**：
1. **共享上下文可见**：两分支都能访问当前上下文，保证信息基础一致
2. **先验/后验 tokens 隔离**：避免两分支直接交互，强迫通过对齐损失间接同步
3. **位置 ID 相同**：两分支在对应 token 位置共享位置编码，确保潜在查询和未来嵌入在结构上对齐

#### 4.5.3 计算效率

这种设计允许模型在**单次 backbone 前向传播**中实现双分支潜在世界动作模型：
- 相比分离的双分支前向传播，节省约 50% 计算量
- 无需维护两套独立的模型权重

### 4.6 损失函数详细设计

#### 4.6.1 Flow Matching 目标

设 $a$ 为动作标签，$t \in [0, 1]$ 为 flow time，$\epsilon \sim \mathcal{N}(0, I)$ 为采样噪声。

构造插值动作：$\tilde{a}_t = t \cdot a + (1 - t) \cdot \epsilon$，目标速度 $u_t = a - \tilde{a}_t$。

**Flow Matching 损失**（两分支分别计算）：

$$
\mathcal{L}_{\text{prior}}^{\text{FM}} = \| v_{\text{prior}}^\theta(a_t, c) - u_t \|_2^2
$$

$$
\mathcal{L}_{\text{post}}^{\text{FM}} = \| v_{\text{post}}^\theta(a_t, c, z_{\text{post}}) - u_t \|_2^2
$$

$$
\mathcal{L}_{\text{FM}} = \mathcal{L}_{\text{prior}}^{\text{FM}} + \mathcal{L}_{\text{post}}^{\text{FM}}
$$

#### 4.6.2 正则化项（防止对齐坍缩）

**范数正则化**（防止幅度坍缩）：

$$
R_{\text{norm}}(h) = [\text{ReLU}(\tau - \|h\|_2)]^2
$$

**秩正则化**（防止方向坍缩到低维子空间）：

设 $H \in \mathbb{R}^{B \times n}$ 为潜在隐藏状态在随机 $n$ 维子空间上的投影，归一化得 $\hat{H}$，Gram 矩阵 $G = \hat{H} \hat{H}^\top$，特征值谱归一化 $p_i = \lambda_i / \sum_j \lambda_j$：

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

其中：
- $\mathcal{L}_{\text{FM}}$：两分支的 flow matching 损失之和
- $w_{\text{align}} = 10^{-3}$：对齐损失权重
- $\mathcal{L}_{\text{reg}}$：范数 + 秩正则化，防止潜在空间坍缩

### 4.7 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 观测 horizon $H$ | 4 | 历史帧数 |
| 动作块长度 $T$ | 20 | 每步预测的动作数 |
| 潜在查询数量 $K$ | 16 | 通过消融实验确定 |
| 对齐权重 $w_{\text{align}}$ | $10^{-3}$ | 平衡动作损失与对齐 |
| 正则化权重 $w_{\text{norm}}, w_{\text{rank}}$ | $10^{-4}$ | 防止坍缩 |
| 范数阈值 $\tau$ | - | 消融确定 |

### 4.8 一二阶段训练

#### Stage 1: 统一格式预训练

- 在混合人类和机器人操作数据上预训练（200,000 小时自我中心视频）
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
        # 可学习的潜在查询（核心设计）
        self.latent_queries = nn.Parameter(torch.randn(latent_query_num, hidden_dim))
        # 后验分支的未来编码器（冻结 ViT + Perceiver Resampler）
        self.future_encoder = FutureEncoder()

    def forward_prior(self, x, obs, state, actions_noised, t):
        """先验分支：训练+推理"""
        # 1. 编码当前上下文
        ctx = self.vision_encoder(obs)
        ctx = torch.cat([x, ctx, state], dim=1)

        # 2. 构建序列：[ctx; latent_queries; noised_actions]
        latent_q = self.latent_queries.unsqueeze(0).expand(bs, -1, -1)
        seq = torch.cat([ctx, latent_q, actions_noised], dim=1)

        # 3. Transformer 前向传播（MoT 单次前向）
        hidden = self.backbone(seq)

        # 4. 提取动作 hidden 和潜在推理 hidden
        latent_hidden = hidden[:, -T-16:-16]  # K 个潜在查询位置
        action_hidden = hidden[:, -16:]        # T 个动作位置

        # 5. 预测速度
        velocity = self.action_head(action_hidden, t)
        return velocity, latent_hidden

    def forward_posterior(self, x, obs, state, future_obs, actions_noised, t):
        """后验分支：仅训练"""
        ctx = self.vision_encoder(obs)
        ctx = torch.cat([x, ctx, state], dim=1)

        # 编码未来观测为嵌入（替代潜在查询）
        future_emb = self.future_encoder(future_obs)  # [B, K, d]
        seq = torch.cat([ctx, future_emb, actions_noised], dim=1)

        hidden = self.backbone(seq)
        latent_hidden = hidden[:, -T-16:-16]
        action_hidden = hidden[:, -16:]

        velocity = self.action_head(action_hidden, t)
        return velocity, latent_hidden

    def compute_loss(self, batch):
        x, obs, state, future_obs, actions, t = batch

        # Flow Matching 插值
        eps = torch.randn_like(actions)
        actions_noised = t * actions + (1 - t) * eps
        target_velocity = actions - actions_noised

        # 两分支前向
        v_prior, h_prior = self.forward_prior(x, obs, state, actions_noised, t)
        v_post, h_post = self.forward_posterior(x, obs, state, future_obs, actions_noised, t)

        # Flow Matching 损失
        L_fm = F.mse_loss(v_prior, target_velocity) + \
               F.mse_loss(v_post, target_velocity)

        # 对齐损失（核心Bridge机制）
        L_align = F.mse_loss(h_prior, h_post)

        # 正则化（防止坍缩）
        L_reg = self.compute_regularization(h_prior, h_post)

        return L_fm + 1e-3 * L_align + L_reg


# ============ 推理（仅先验分支）============

def inference(model, instruction, observations, states):
    model.eval()
    noised_action = torch.randn(batch, T, action_dim).to(device)

    for step in range(num_diffusion_steps):
        t = schedule[step]  # e.g., linear from 1 to 0
        velocity = model.forward_prior(instruction, observations, states,
                                     noised_action, t)
        noised_action = noised_action + velocity * dt

    return noised_action
```

---

## 6 实验结论

### 6.1 仿真基准

| Model | Size | LIBERO | LIBERO-plus | LIBERO-plus* | RoboCasa-50 | GR1 | CALVIN | CALVIN* | Robotwin2 |
|-------|------|--------|-------------|--------------|--------------|-----|--------|----------|-----------|
| **VLA** |
| π0 | 3B | 94.4 | 53.6 | - | 42.4 | - | 3.92 | 65.9/58.4 | - |
| π0.5 | 3B | 96.9 | 77.4 | - | 41.4 | - | 4.06 | 4.13 | 82.7/76.8 |
| Being-H0.5 | 2B | 98.9 | 78.5 | 83.1 | 53.5 | - | 4.63 | 4.48 | - |
| **WAM** |
| Cosmos-Policy | 7B | 98.5 | - | - | 67.1 | - | - | - | - |
| Fast-WAM | 6B | 97.6 | - | - | - | - | - | 91.9/91.8 | - |
| **Being-H0.7** | 3B | **99.2** | **82.1** | **84.8** | **62.1** | **49.2** | **4.67** | **4.48** | **90.2/89.6** |

### 6.2 真实机器人评估

| Suite | Being-H0.7 | Being-H0.5 | π0.5 | Fast-WAM |
|-------|------------|------------|-------|----------|
| Dynamic Scene | **70.0** | 59.4 | 46.7 | 45.8 |
| Physical Reasoning | **66.9** | 58.3 | 57.5 | 56.1 |
| Motion Reasoning | **67.5** | 60.6 | 49.2 | 58.3 |
| Long Horizon | **66.7** | 57.5 | 45.0 | 46.9 |
| Generalization | **67.5** | 55.0 | 49.2 | 58.3 |

### 6.3 推理效率

| Method | Latency (ms/step) | Memory (GB) |
|--------|------------------|-------------|
| Being-H0.7-UAC | **3.45** | 5.6 |
| Being-H0.7 | 3.61 | 5.6 |
| Being-H0.5-UAC | 12.45 | 6.8 |
| π0.5-RTC | 28.81 | 15.6 |
| Fast-WAM | 41.26 | 9.3 |

---

## 7 WAM 三兄弟深度对比：Fast-WAM vs Latent-WAM vs Being-H0.7

### 7.1 核心架构对比

| 维度 | Fast-WAM | Latent-WAM | Being-H0.7 |
|------|-----------|-------------|-------------|
| **论文** | Do WAMs Need Test-time Imagination? | Latent-WAM for E2E AD | A Latent World-Action Model |
| **核心洞察** | WAM 价值来自训练时视频联合建模，而非推理时想象 | 感知自由需要空间压缩 + 因果世界建模 | 潜在推理空间连接 VLA 和 WAM |
| **应用场景** | 机器人操作 | 自动驾驶 | 机器人操作 + 真机部署 |
| **模型规模** | 6B | 104M | 3B |

### 7.2 世界模型实现对比

| 维度 | Fast-WAM | Latent-WAM | Being-H0.7 |
|------|-----------|-------------|-------------|
| **未来预测方式** | 训练时视频联合建模，推理时跳过想象 | 像素级未来预测（Epona） | 潜在空间未来嵌入 |
| **推理时是否需要 rollout** | **否**（Skip imagination） | 是（需要世界模型 rollout） | **否** |
| **潜在空间设计** | 无显式潜在空间 | SCWE 16-query 场景压缩 | K=16 可学习潜在查询 |
| **对齐机制** | 无 | WorldMirror 几何蒸馏 | 联合对齐损失 |

### 7.3 关键设计差异

#### Fast-WAM：训练即一切

```
训练阶段：
[观测] → [联合去噪：视频 + 动作] → 学会物理先验
                              ↓
推理阶段（跳过想象）：
[观测] → [单次前向] → 直接预测动作
```

**关键发现**：推理时显式想象对动作性能提升有限，训练时的视频联合建模才是核心。

#### Latent-WAM：感知自由的世界建模

```
SCWE（空间感知压缩）     DLWM（因果世界建模）
     ↓                        ↓
紧凑 scene tokens    →    未来世界状态预测
     ↓
Trajectory Decoder → 规划轨迹
```

**关键设计**：
- SCWE：16 个 query 的 Perceiver 压缩场景表征
- WorldMirror：从 3DGS 重建场景中蒸馏几何知识
- DLWM：因果潜在世界模型预测未来

#### Being-H0.7：潜在推理作为桥梁

```
训练阶段（双分支）：
[当前上下文] → Prior Branch + 潜在查询 → 动作预测
      ↓
[未来观测]  → Posterior Branch + 未来嵌入 → 动作预测
      ↓
              对齐损失（强迫潜在空间一致）

推理阶段（仅先验）：
[当前上下文] → Prior Branch + 潜在查询 → 动作预测（隐含未来感知）
```

**关键创新**：通过对齐损失将未来信息"蒸馏"到先验分支的潜在空间中。

### 7.4 效率对比

| 指标 | Fast-WAM | Latent-WAM | Being-H0.7 |
|------|-----------|-------------|-------------|
| **推理延迟** | 高（41ms） | 中 | **低（3.6ms）** |
| **GPU 内存** | 9.3GB | 低 | **5.6GB** |
| **训练成本** | 高 | 低 | 中 |
| **是否需要 3DGS** | 否 | 是（蒸馏） | 否 |

### 7.5 适用场景对比

| 场景 | Fast-WAM | Latent-WAM | Being-H0.7 |
|------|-----------|-------------|-------------|
| **机器人操作** | 适合（需视频预训练） | 不适合（自动驾驶专用） | **最适合** |
| **自动驾驶** | 不适合 | **最适合** | 不适合 |
| **实时控制** | 一般 | 一般 | **最适合** |
| **数据效率** | 低（需大量视频） | 高 | 中 |

### 7.6 总结对比

| 维度 | Fast-WAM | Latent-WAM | Being-H0.7 |
|------|-----------|-------------|-------------|
| **核心贡献** | 证明推理时想象不必要 | 感知自由的潜在世界模型 | 潜在推理空间连接 VLA+WAM |
| **世界模型形式** | 视频联合训练 | 像素级重建 + 因果预测 | 潜在嵌入 + 对齐 |
| **推理效率** | 低 | 中 | **高** |
| **真机部署** | 一般 | 一般 | **最适合** |
| **主要创新** | 范式重新思考 | 几何感知蒸馏 | 双分支对齐 |

---

## 8 KnowHow（核心洞察）

1. **潜在查询的可学习设计**：一组端到端学习的 tokens，比固定查询或人工设计查询更具表达力。模型自动发现最优的潜在表示结构。

2. **双分支对齐 = 信息蒸馏**：将后验分支的"未来知识"通过对齐损失转移到先验分支，本质是一种隐式蒸馏。

3. **为什么需要正则化**：对齐损失可能让潜在状态坍缩到简单解（全部相同/全部为零）。范数正则化防止幅度坍缩，秩正则化防止方向坍缩。

4. **MoT 打包的工程技巧**：共享上下文 + 隔离的分支 tokens + 相同位置 ID = 单次前向完成双分支训练。

5. **Perceiver Resampler 的优势**：将任意长度帧序列压缩为固定 K 个嵌入，同时保留任务相关信息。

6. **冻结 ViT 的选择**：未来编码器使用冻结的预训练 ViT，避免梯度流导致的表征不一致，同时降低训练成本。

7. **K=16 的消融依据**：论文实验表明 K 过小会限制表达能力，K 过大则增加计算成本且收益递减。

8. **动作 chunk T=20 的设计**：20 步动作提供了足够的时序上下文，支持平滑动作执行，同时保持推理延迟可控。

---

## 9 arXiv Appendix 关键点总结

由于论文尚未发布于 arXiv，以下列出从 PDF 内容推断的补充内容：

- **A**: 更多真实机器人任务细节（12 个任务的详细描述，涵盖 PND Adam-U、Unitree G1、Franka FR3）
- **B**: 不同具身平台的具体配置和传感器设置
- **C**: 更多消融实验（MoT 组件贡献、对齐权重敏感性、K 值敏感性）
- **D**: 潜在查询数量 K 的敏感性分析
- **E**: 未来观测 horizon 的影响
- **F**: 更多定性结果（注意力可视化展示先验分支如何 attend 到相关历史）
- **G**: 与其他世界模型（Cosmos Policy、Fast-WAM、JEPA-VLA）的详细对比

---

## 10 Q&A：Latent Query 设计深度解析

### Q1：为什么用 Learnable Token 而非 Visual-derived Token？

**原文依据（Section 3.1, Page 5）**：

> "When trained only with action supervision, the latent may instead collapse to a weak intermediate representation or encode only shallow cues sufficient for local action decoding."

论文明确指出了如果 latent 从视觉 token 派生，仅靠动作监督会存在的问题：

| 问题 | 描述 |
|------|------|
| **弱表示坍缩** | latent 可能退化为弱中间表示，仅编码浅层局部动作解码线索 |
| **视觉语义主导** | 视觉编码器目标（图像重建）与动作预测目标不同，混合会干扰 |
| **先验-后验耦合** | 如果 latent 从视觉 token 派生，两分支会共享视觉先验，难以解耦未来预测与外观建模 |

**核心论点**：learnable token 保证 latent 空间是「动作导向」而非「视觉导向」。

---

### Q2：JEPA 系列论文的支撑

Being-H0.7 与 JEPA 系列的关联体现在两个引用：

| 引用 | 论文 | 与 Being-H0.7 的关系 |
|------|------|---------------------|
| [58] VLA-JEPA | VLA-JEPA: Enhancing VLA with latent world model (arXiv:2602.10098) | 使用潜在世界模型增强 VLA 的先行工作 |
| [74] JEPA-VLA | JEPA-VLA: Video predictive embedding for VLA (arXiv:2602.11832) | 视频预测嵌入用于 VLA 的工作 |

**原文对 JEPA 系列的定位（Page 4）**：

> "a related but distinct line [55–58] has explored future prediction or alignment in latent representation space to support action generation"

**Being-H0.7 与 JEPA-VLA 的关键差异**：

JEPA-VLA 等工作侧重于在**潜在表示空间**中做未来预测或对齐，但 latent 仍来源于视觉编码器。Being-H0.7 的创新在于：
- latent 是**独立的学习对象**（learnable），而非视觉表示的派生物
- 通过**双分支对齐**显式塑造潜在空间的世界建模能力
- prior 分支推理时**无需像素级 rollout**，直接输出动作

---

### Q3：与 JALA 的关键区别

**原文依据（Page 4）**：

> "Compared with JALA, our key difference is to move the object aligned with future information from action-side hidden states to an explicit set of latent query slots, yielding a more flexible reasoning space."

| 对比维度 | JALA | Being-H0.7 |
|---------|------|-------------|
| **对齐目标** | 未来信息对齐到动作侧隐藏状态 | 未来信息对齐到显式 latent query slots |
| **推理空间** | 隐式，动作隐藏状态的一部分 | 显式，独立于动作的潜在推理空间 |
| **灵活性** | 受动作解码器约束 | 更灵活的推理空间，可自由组织 |

**为什么更灵活**：latent query 作为独立于动作的 slots，不受动作解码器反向传播的直接约束，可以更自由地学习任务相关的推理结构。

---

### Q4：为什么 latent query 可以是 learnable 的？（理论支撑）

**Section 3.1 的核心论点（Page 5）**：

> "the model is no longer forced to map abstract multimodal semantics directly into dense low-level actions. Instead, it can gradually form a compact intermediate reasoning state during forward propagation and use it to guide action prediction."

**学习机制**：
1. **跨层传播**：latent query 与指令、观测、状态、动作一起参与逐层 Transformer 传播
2. **信息聚合**：在多层传播中，latent query 从上下文 tokens 逐步整合任务相关信息
3. **动作引导**：最终的 latent 状态在动作生成位置输出，引导下游动作预测

**为什么有效**：
- latent query 不是凭空预测，而是在 DiT 的交叉注意力中受到视觉上下文的约束
- 双分支对齐损失强迫 prior 分支的 latent 状态与后验分支（拥有未来信息）对齐
- 本质上是对齐损失提供了「未来监督信号」，而非随机初始化的监督

---

### Q5：为何不需要 visual grounding？

**分析**：

可选的替代方案是从 ViT/VQ-VAE 压缩 token 派生 latent query（visual-derived），好处是有视觉结构 grounding。但代价是：

| 方案 | 优势 | 代价 |
|------|------|------|
| **Visual-derived** | 零样本可用，有视觉结构 | 受视觉压缩质量约束，引入视觉偏置 |
| **Learnable（Being-H0.7）** | 任务专用，表达力强 | 需训练，无视觉 grounding |

**论文的权衡**：Being-H0.7 选择可学习 token，因为：
1. **下游任务专属性**：latent 只需要对动作预测负责，不需要泛化的视觉表示
2. **冻结 ViT 的后验补偿**：后验分支使用冻结预训练 ViT 提取未来嵌入，提供未来感知能力
3. **JEPA 思想一脉相承**：I-JEPA 就用过 learnable query 做掩码图像预测，核心洞察是「预测空间应比输入空间更语义化」

---

### Q6：我的分析总结

| 维度 | 结论 |
|------|------|
| **设计意图** | latent 空间是「动作推理」专用，不需要对视觉重建负责 |
| **学习信号来源** | 双分支对齐损失 + 动作 flow matching 损失，而非视觉重构 |
| **为何可行** | prior 分支被训练成「在 latent 空间中模拟后验分支的未来预测」 |
| **与 JEPA 的区别** | JEPA 用 learnable query 预测视觉掩码，Being-H0.7 用 learnable query 预测未来动作分布 |
| **工程取舍** | 牺牲零样本泛化能力，换取任务专用性和潜在空间的纯粹性 |

---

## 11 总结

### 三大核心贡献

1. **潜在世界动作模型范式**：通过在感知和动作之间引入紧凑潜在推理空间，连接 VLA 和 WAM 的优势
2. **先验-后验双分支联合对齐**：训练时利用未来信息塑造潜在空间，推理时保持高效单分支部署
3. **MoT 高效实现**：单次前向传播完成双分支训练，结合 flow matching 和对齐损失

### 最重要洞察

**潜在空间是连接现在和未来的桥梁**。不同于直接在像素空间预测未来（计算昂贵且有误差），Being-H0.7 将未来信息压缩到潜在推理空间，通过对齐损失强迫先验分支隐式学习未来预测能力。这种"推理时无需 rollout"的设计是效率关键。

---

**参考链接**：

| 资源 | 链接 |
|------|------|
| **论文 PDF** | [being-h07.pdf](https://research.beingbeyond.com/projects/being-h07/being-h07.pdf) |
| **项目主页** | [Being-H0.7](https://research.beingbeyond.com/being-h07) |
| **BeingBeyond** | [beingbeyond.com](https://www.beingbeyond.com) |
