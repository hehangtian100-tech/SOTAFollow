# MoE 做 RL：训练-推理不一致问题

> 本文档结合 Routing Replay 机制与 GSPO 算法解析

---

## 1. 核心问题：训练时和推理时 MoE 行为不一致

### 1.1 现象描述

MoE（Mixture of Experts）模型中，训练阶段和推理阶段的 token-expert 分配路径不同：

| 阶段 | Expert 路由行为 |
|------|----------------|
| **推理** | 稀疏激活，每个 token 仅激活 Top-K 个 expert（如 Top-2/8） |
| **训练（PPO/GRPO）** | 需要对所有 expert 计算 value function、优势函数，反向传播需全 expert 参与计算 |
| **核心矛盾** | RL 训练时的 expert 利用分布 ≠ 推理时的 expert 利用分布 |

### 1.2 根因：Policy Gradient 中的 Importance Sampling

PPO/GRPO 类算法依赖旧策略 $\pi_{\text{old}}$ 采样：

$$
\frac{\nabla \mathcal{L}_{\text{RL}}}{\nabla \theta} \approx \mathbb{E}_{x \sim \pi_{\text{old}}} \left[ \frac{\pi_\theta(x)}{\pi_{\text{old}}(x)} \nabla \log \pi_\theta(x) \right]
$$

问题在于：$\pi_{\text{old}}$ 中某个 token 分配给 Expert A，但在 $\pi_\theta$ 中可能分配给 Expert B，导致 **distribution shift**（分布偏移）。

---

## 2. 解决方案一：Routing Replay（路由回放）

### 2.1 核心思想

训练时**记录每个 token 在推理阶段的 routing 决策**（即 expert 分配结果），在后续训练中使用这些记录来保持一致性。

```python
# 伪代码：Routing Replay 机制
class RoutingReplayBuffer:
    def __init__(self):
        self.routing_history = {}  # token_id -> expert_assignments
    
    def record(self, token_ids, expert_assignments):
        """记录推理时的 routing 决策"""
        for token_id, expert_list in zip(token_ids, expert_assignments):
            self.routing_history[token_id] = expert_list
    
    def replay(self, token_ids):
        """回放历史 routing 决策，约束训练时的分配"""
        return [self.routing_history.get(tid, None) for tid in token_ids]
```

### 2.2 训练流程

1. **推理阶段**（rollout）：记录每个 token 的 expert 分配
2. **训练阶段**：强制要求训练时的 expert 分配与记录的分配保持一致
3. **KL 约束**：对 deviation 施加惩罚

$$
\mathcal{L}_{\text{RL}} = \mathcal{L}_{\text{RL}}^{\text{original}} + \lambda \cdot D_{\text{KL}}(\pi_{\theta}^{\text{routing}} \| \pi_{\text{replay}}^{\text{routing}})
$$

### 2.3 优势与局限

| 优点 | 局限 |
|------|------|
| 训练与推理 routing 一致性提升 | 需要额外存储 routing 历史 |
| 减少因 distribution shift 导致的梯度方差 | 记录频率和存储开销 |
| 实现相对简单 | 长期训练中历史 routing 可能过时 |

---

## 3. 解决方案二：GSPO（Group Sampling Policy Optimization）

### 3.1 背景：GRPO 的问题

DeepSeek GRPO 核心思想是对同一 prompt 采样多个回复，用 group 内 baseline 计算优势：

$$
A_i = \frac{r_i - \mu_{\text{group}}}{\sigma_{\text{group}}}
$$

**MoE 场景下的问题**：GRPO 的优势估计基于 final reward，但 MoE 的 routing 决策是在 token-level 做出的——单个 token 的 routing 质量无法从 episode-level reward 直接监督。

### 3.2 GSPO 核心思想

GSPO = **Group Sampling + Token-level Routing Alignment**

**关键改进**：

1. **Group 内共享 Routing Pattern**：同一 prompt 的多个样本共享 expert 分配策略，减少 variance
2. **Token-level Advantage Estimation**：不仅估计 sequence-level reward，还估计每个 token 对应的 advantage
3. **Adaptive Routing Regularization**：在 advantage 估计中引入 routing 稳定性项

数学表述：

$$
\mathcal{L}_{\text{GSPO}} = \mathbb{E}_{g \sim \mathcal{G}} \left[ \sum_{i \in g} \hat{A}_i^{\text{token}} \cdot \log \pi_\theta(a_i | s_i) \right] - \beta \cdot \text{RoutingReg}(\pi_\theta)
$$

其中 $\hat{A}_i^{\text{token}}$ 是 token-level 优势函数，RoutingReg 约束训练时的 routing 分布接近推理分布：

$$
\text{RoutingReg} = D_{\text{KL}}\left( \text{RoutingDist}_{\text{train}} \| \text{RoutingDist}_{\text{inference}} \right)
$$

### 3.3 GSPO vs GRPO 对比

| 维度 | GRPO | GSPO |
|------|------|------|
| 优势估计粒度 | Sequence-level（整个回复） | Token-level + Sequence-level |
| Expert 利用 | 无约束，可能偏斜 | 通过 Group Sampling 约束 routing |
| 训练稳定性 | 较高 variance | 通过 Group 内共享降低 variance |
| MoE 适配性 | 一般 | **专门针对 MoE 设计** |

---

## 4. 综合方案：Routing Replay + GSPO 联合使用

### 4.1 推荐实践流程

```
┌─────────────────────────────────────────────────────────┐
│                    Rollout 阶段                          │
│  1. 推理采样，记录每个 token 的 expert 分配              │
│  2. 计算 episode reward                                 │
│  3. Routing Replay Buffer 记录 routing 历史             │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                    GSPO 训练阶段                         │
│  1. Group Sampling：对同一 prompt 采样多条轨迹           │
│  2. Token-level Advantage Estimation                   │
│  3. Routing Regularization：KL(训练分布 ‖ 推理分布)    │
│  4. Backpropagation 更新 policy + routing network      │
└─────────────────────────────────────────────────────────┘
```

### 4.2 其他辅助手段

| 方法 | 描述 |
|------|------|
| **Expert Dropout** | 训练时随机 drop 部分 expert，模拟推理时的稀疏激活 |
| **容量限制（Capacity Capping）** | 限制每个 token 最多分配的 expert 数量 |
| **辅助负载均衡损失** | 防止少数 expert 主导，造成 overfitting |
| **软路由蒸馏** | 用软概率分布替代硬 TopK，减少训练-推理 gap |

---

## 5. 面试一句话回答

> **MoE 做 RL 训推不一致的核心矛盾是：推理时稀疏激活 Top-K expert，但 RL 训练需要对所有 expert 计算 advantage 和梯度，导致 routing 分布不同。解决方案一是用 Routing Replay 记录推理时的 expert 分配并在训练中约束一致性；方案二是用 GSPO（Group Sampling Policy Optimization），通过 Group 内共享 routing pattern 和 token-level advantage 估计，配合 KL 正则项使训练分布逼近推理分布，从根本上降低 variance 并提升 MoE 训练稳定性。**

---

## 6. 追问储备

| 问题 | 回答要点 |
|------|----------|
| GSPO 的 Group Sampling 是什么？ | 对同一 prompt 采样多条回复构成 group，用 group 内 baseline 计算 advantage，减少 reward variance |
| 为什么 GRPO 在 MoE 上不够用？ | GRPO 只做 sequence-level advantage，没有约束 token-level routing 分布，MoE expert 容易偏斜 |
| Expert Dropout 的 dropout rate 通常多少？ | 0.1~0.2，训练初期可以更高让 expert 更均匀，后期降低保留性能 |

