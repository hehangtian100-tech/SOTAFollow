# XXPO 系列算法精读报告

> 本文档持续记录 RL 领域中各类 XXPO（Policy Optimization）算法的核心原理、改进点与实验结论。

---

## Contents（目录索引）

| 算法 | 全称 | 发布时间 | 机构 | 会议/期刊 | 一句话总结 | 章节 |
|------|------|---------|------|----------|-----------|------|
| [PPO](#1-ppo-proximal-policy-optimization) | Proximal Policy Optimization | 2017 | OpenAI | arXiv | 策略优化奠基之作，CLIP 机制控制策略更新幅度 | [§1](#1-ppo-proximal-policy-optimization) |
| [GRPO](#2-grpo-group-relative-policy-optimization) | Group Relative Policy Optimization | [待验证] | [待验证] | — | Group-relative sampling 替代 Value Model，大幅降低训练成本 | [§2](#2-grpo-group-relative-policy-optimization) |
| [GSPO](#3-gspo-group-sequence-policy-optimization) | Group Sequence Policy Optimization | [待验证] | 阿里 | — | [待验证] | [§3](#3-gspo-group-sequence-policy-optimization) |
| [DAPO](#4-dapo) | [待验证] | [待验证] | [待验证] | — | [待验证] | [§4](#4-dapo) |
| [GMPO](#5-gmpo-geometric-mean-policy-optimization) | Geometric-Mean Policy Optimization | [待验证] | [待验证] | — | [待验证] | [§5](#5-gmpo-geometric-mean-policy-optimization) |

*最后更新：2026-04-22*

---

## 1. PPO (Proximal Policy Optimization)

### 参考链接

| 资源 | 链接 | 状态 |
|------|------|------|
| **论文** | [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) | ✅ 已验证 |
| **代码** | [openai/baselines](https://github.com/openai/baselines) | ✅ 已验证 |

### 基本信息

| 字段 | 内容 | 状态 |
|------|------|------|
| **论文标题** | Proximal Policy Optimization Algorithms | ✅ |
| **发布时间** | 2017.08 | ✅ |
| **机构** | OpenAI | ✅ |
| **作者** | John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov | ✅ |

### 1.1 Motivation（问题背景）

PPO 提出的背景是：标准策略梯度方法（如 REINFORCE）面临两个核心问题：

1. **策略更新幅度难以控制**：大步伐更新导致策略崩溃，小步伐更新学习太慢
2. **数据效率低**：每个样本只能使用一次（on-policy）

Trust Region Policy Optimization (TRPO) 通过 KL 散度约束解决了更新幅度问题，但计算复杂（需要二阶导数）。PPO 提出用一阶优化近似 TRPO，同时保证策略单调提升。

### 1.2 一句话总结

**PPO 通过 CLIP 机制限制策略更新比例，在保证训练稳定性的同时实现数据高效利用，成为强化学习最广泛使用的基线算法。**

### 1.3 核心贡献

1. **Clipped Surrogate Objective**：限制策略更新比例 $\in [1-\epsilon, 1+\epsilon]$
2. **Adaptive KL Penalty**：可选的 KL 散度惩罚项，自适应调整系数
3. **Multiple Epochs**：允许对同一批数据多次更新，提高数据效率
4. **GAE**：Generalized Advantage Estimation，提供方差-偏差权衡的控制

### 1.4 方法详述

#### 1.4.1 PPO-Clip 目标函数

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是概率比。

```python
def ppo_clip_loss(theta, theta_old, states, actions, advantages, epsilon=0.2):
    """
    PPO Clip Loss

    Args:
        theta: 当前策略参数
        theta_old: 旧策略参数
        states: 状态序列
        actions: 动作序列
        advantages: 优势函数估计
        epsilon: 裁剪参数（默认0.2）

    Returns:
        clip_loss: PPO裁剪损失
    """
    # 计算概率比 r(theta)
    pi_theta = policy_prob(theta, states, actions)
    pi_old = policy_prob(theta_old, states, actions)
    ratio = pi_theta / (pi_old + 1e-8)

    # Clipped loss
    surr1 = ratio * advantages  # 标准策略梯度
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages  # 裁剪后

    # 取较小值
    clip_loss = -torch.min(surr1, surr2).mean()

    return clip_loss
```

### 1.5 实验结论

PPO 在多个 benchmark 上与 TRPO 性能相当，但计算效率提升 3 倍。

---

## 2. GRPO (Group Relative Policy Optimization)

### 参考链接

| 资源 | 链接 | 状态 |
|------|------|------|
| **论文** | [待补充 arXiv ID] | ⚠️ 待验证 |
| **代码** | [待补充] | ⚠️ 待验证 |

### 基本信息

| 字段 | 内容 | 状态 |
|------|------|------|
| **论文标题** | [待验证] | ⚠️ 待验证 |
| **发布时间** | [待验证] | ⚠️ 待验证 |
| **机构** | [待验证] | ⚠️ 待验证 |
| **作者** | [待验证] | ⚠️ 待验证 |

### 2.1 Motivation（问题背景）

[待补充]

### 2.2 一句话总结

**GRPO 通过 Group-relative Sampling 机制，利用同组样本的相对表现估计 Advantage，消除了训练 Value Model 的高昂成本。**

### 2.3 核心贡献

[待补充]

### 2.4 方法详述

[待补充]

---

## 3. GSPO (Group Sequence Policy Optimization)

### 参考链接

| 资源 | 链接 | 状态 |
|------|------|------|
| **论文** | [待补充 arXiv ID] | ⚠️ 待验证 |
| **代码** | [待补充] | ⚠️ 待验证 |

### 基本信息

| 字段 | 内容 | 状态 |
|------|------|------|
| **论文标题** | Group Sequence Policy Optimization | ⚠️ 待验证 |
| **发布时间** | [待验证] | ⚠️ 待验证 |
| **机构** | 阿里巴巴 | ⚠️ 待验证 |
| **作者** | [待验证] | ⚠️ 待验证 |

### 3.1 Motivation（问题背景）

[待补充]

### 3.2 一句话总结

**[待验证]**

### 3.3 核心贡献

[待补充]

### 3.4 方法详述

[待补充]

---

## 4. DAPO

### 参考链接

| 资源 | 链接 | 状态 |
|------|------|------|
| **论文** | [待补充] | ⚠️ 待验证 |
| **代码** | [待补充] | ⚠️ 待验证 |

### 基本信息

| 字段 | 内容 | 状态 |
|------|------|------|
| **论文标题** | [待验证] | ⚠️ 待验证 |
| **发布时间** | [待验证] | ⚠️ 待验证 |
| **机构** | [待验证] | ⚠️ 待验证 |

### 4.1 Motivation（问题背景）

[待补充]

### 4.2 一句话总结

**[待补充]**

### 4.3 核心贡献

[待补充]

### 4.4 方法详述

[待补充]

---

## 5. GMPO (Geometric-Mean Policy Optimization)

### 参考链接

| 资源 | 链接 | 状态 |
|------|------|------|
| **论文** | [待补充 arXiv ID] | ⚠️ 待验证 |
| **代码** | [待补充] | ⚠️ 待验证 |

### 基本信息

| 字段 | 内容 | 状态 |
|------|------|------|
| **论文标题** | Geometric-Mean Policy Optimization | ⚠️ 待验证 |
| **发布时间** | [待验证] | ⚠️ 待验证 |
| **机构** | [待验证] | ⚠️ 待验证 |

### 5.1 Motivation（问题背景）

[待补充]

### 5.2 一句话总结

**[待补充]**

### 5.3 核心贡献

[待补充]

### 5.4 方法详述

[待补充]

---

## 6. 附录：算法对比总结

### 6.1 算法演化路径（待完善）

```
PPO (2017)
    │
    ▼
GRPO (2025) ── [待验证]
    │
    ├──→ GSPO ── [待验证]
    │
    └──→ DAPO ── [待验证]
    │
    ▼
GMPO ── [待验证]
```

### 6.2 核心机制对比

| 算法 | Advantage 估计 | Clip 机制 | 状态 |
|------|--------------|----------|------|
| PPO | GAE + Value Model | $r_t \in [1-\epsilon, 1+\epsilon]$ | ✅ |
| GRPO | Group-relative | Clipped | ⚠️ |
| GSPO | [待验证] | [待验证] | ⚠️ |
| DAPO | [待验证] | [待验证] | ⚠️ |
| GMPO | [待验证] | [待验证] | ⚠️ |

---

## ⚠️ 待完成工作

以下信息需要验证后补充：

1. [ ] GRPO：arXiv ID、论文标题、机构、完整方法
2. [ ] GSPO：arXiv ID、论文标题、完整方法
3. [ ] DAPO：全称、arXiv ID、机构、完整方法
4. [ ] GMPO：arXiv ID、完整方法

---

*整理 by 优酱 🍃 | 2026-04-22*
*精读标准参考 CLAUDE.md § 论文精读格式标准*
