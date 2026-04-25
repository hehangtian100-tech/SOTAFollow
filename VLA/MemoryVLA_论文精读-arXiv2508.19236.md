# MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation

> **论文**: [arXiv:2508.19236](https://arxiv.org/abs/2508.19236)
>
> **核心贡献**: 提出 Perceptual-Cognitive Memory (PCM) 神经形态记忆框架，结合工作记忆和情景记忆增强 VLA 机器人操作能力

---

## 1. Motivation（问题背景）

VLA（Vision-Language-Action）模型在机器人操作任务中存在两个关键问题：

1. **上下文长度限制**：现有 VLA 模型的上下文窗口无法有效处理长程任务历史
2. **工作记忆与情景记忆的分离**：人类智能通过工作记忆和情景记忆的协同实现复杂任务，而现有 VLA 缺乏这种记忆机制

**本文探索的核心问题**：如何在 VLA 中有效整合工作记忆和情景记忆？

---

## 2. 一句话总结

MemoryVLA 提出 Perceptual-Cognitive Memory (PCM) 框架，通过工作记忆（LSTM）和情景记忆（RAG）的协同，实现长程任务上下文利用。

---

## 3. 核心贡献

1. **Perceptual-Cognitive Memory (PCM) 架构**：神经形态记忆框架，结合工作记忆和情景记忆
2. **Working Memory 模块**：基于 LSTM，处理短中期任务信息
3. **Episodic Memory 模块**：基于 RAG + 向量相似度检索
4. **Action Memory 模块**：动作模式存储与检索
5. **对比实验**：PCM vs Long-Context VLA (LCV) vs Hierarchical VLA (HV)

---

## 4. 方法详述

### 4.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    MemoryVLA 整体架构                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入: 视觉观察 (I_t) + 语言指令 (w)                           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           Perceptual-Cognitive Memory (PCM)                │ │
│  │                                                          │ │
│  │  ┌──────────────────┐  Working Memory (LSTM-based)      │ │
│  │  │ Perceptual Mem   │  处理短期任务信息                   │ │
│  │  │ (视觉压缩)       │                                   │ │
│  │  └──────────────────┘                                   │ │
│  │           ↓                                              │ │
│  │  ┌──────────────────┐  Episodic Memory (RAG-based)       │ │
│  │  │ Cognitive Mem    │  情景记忆检索                      │ │
│  │  │ (语义聚合)       │                                   │ │
│  │  └──────────────────┘                                   │ │
│  │           ↓                                              │ │
│  │  ┌──────────────────┐  Action Memory                      │ │
│  │  │ Action Mem       │  动作模式存储                       │ │
│  │  └──────────────────┘                                   │ │
│  └─────────────────────────────────────────────────────────┘ │
│                           ↓                                   │
│                   ┌─────────────────┐                        │
│                   │ VLA Backbone   │                        │
│                   │ (Qwen2-VL-2B) │                        │
│                   └─────────────────┘                        │
│                           ↓                                   │
│                   Action Prediction                           │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Working Memory 模块

```python
class WorkingMemory(nn.Module):
    """
    工作记忆模块：基于 LSTM，处理短中期任务信息
    """
    def __init__(self, vision_dim, lang_dim, hidden_dim=512):
        super().__init__()
        # 跨模态融合
        self.fusion = nn.Linear(vision_dim + lang_dim, hidden_dim)

        # LSTM 记忆单元
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, vision_feat, lang_feat, prev_hidden=None):
        """
        vision_feat: [B, D_v] - 当前视觉特征
        lang_feat: [B, D_l] - 语言特征
        prev_hidden: (h, c) - 之前的 LSTM 隐藏状态
        """
        # 融合视觉和语言特征
        fused = torch.relu(self.fusion(
            torch.cat([vision_feat, lang_feat], dim=-1)
        ))

        # LSTM 更新
        if prev_hidden is None:
            output, hidden = self.lstm(fused.unsqueeze(1))
        else:
            output, hidden = self.lstm(fused.unsqueeze(1), prev_hidden)

        output = self.output_proj(output.squeeze(1))

        return output, hidden
```

### 4.3 Episodic Memory 模块

```python
class EpisodicMemory(nn.Module):
    """
    情景记忆模块：基于 RAG + 向量相似度检索
    """
    def __init__(self, memory_dim=512, top_k=5):
        super().__init__()
        self.top_k = top_k

        # 记忆存储 (简化版，实际使用向量数据库)
        self.memory_bank = []
        self.memory_vectors = []

        # 记忆编码器
        self.encoder = nn.Linear(memory_dim * 2, memory_dim)

    def store(self, working_memory_output, action_sequence):
        """
        存储工作记忆输出和对应的动作序列
        """
        # 编码记忆项
        memory_vec = working_memory_output.detach().cpu().numpy()

        # 向量化存储
        self.memory_bank.append({
            'context': working_memory_output,
            'action': action_sequence
        })
        self.memory_vectors.append(memory_vec)

    def retrieve(self, query, current_context):
        """
        检索最相关的情景记忆
        """
        if len(self.memory_bank) == 0:
            return None

        # 计算相似度 (简化版)
        query_vec = query.detach().cpu().numpy()

        similarities = []
        for mem_vec in self.memory_vectors:
            sim = np.dot(query_vec, mem_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(mem_vec) + 1e-8
            )
            similarities.append(sim)

        # Top-K 检索
        top_k_idx = np.argsort(similarities)[-self.top_k:][::-1]

        retrieved_contexts = []
        retrieved_actions = []

        for idx in top_k_idx:
            retrieved_contexts.append(self.memory_bank[idx]['context'])
            retrieved_actions.append(self.memory_bank[idx]['action'])

        return {
            'contexts': retrieved_contexts,
            'actions': retrieved_actions,
            'similarities': [similarities[i] for i in top_k_idx]
        }

    def forward(self, working_output, current_action):
        """
        前向传播：存储当前经验 + 检索相关记忆
        """
        # 存储当前经验
        self.store(working_output, current_action)

        # 检索相关记忆
        retrieved = self.retrieve(working_output, current_action)

        return retrieved
```

### 4.4 Action Memory 模块

```python
class ActionMemory(nn.Module):
    """
    动作记忆模块：存储和检索动作模式
    """
    def __init__(self, hidden_dim=512, action_dim=7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # 动作编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 动作预测头
        self.action_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, query_features, retrieved_episodic_memory):
        """
        基于查询特征和情景记忆预测动作
        """
        # 如果有检索到的情景记忆
        if retrieved_episodic_memory is not None:
            # 聚合检索到的动作
            retrieved_actions = retrieved_episodic_memory['actions']
            if len(retrieved_actions) > 0:
                retrieved_action = torch.stack(retrieved_actions).mean(dim=0)
                retrieved_action = self.action_encoder(retrieved_action)
            else:
                retrieved_action = torch.zeros_like(query_features)
        else:
            retrieved_action = torch.zeros_like(query_features)

        # 预测动作
        action_pred = self.action_predictor(
            torch.cat([query_features, retrieved_action], dim=-1)
        )

        return action_pred
```

### 4.5 PCM 完整前向传播

```python
class PerceptualCognitiveMemory(nn.Module):
    """
    Perceptual-Cognitive Memory 完整模块
    """
    def __init__(self, config):
        super().__init__()
        self.working_memory = WorkingMemory(
            vision_dim=config.vision_dim,
            lang_dim=config.lang_dim,
            hidden_dim=config.hidden_dim
        )
        self.episodic_memory = EpisodicMemory(
            memory_dim=config.hidden_dim,
            top_k=config.top_k
        )
        self.action_memory = ActionMemory(
            hidden_dim=config.hidden_dim,
            action_dim=config.action_dim
        )

        # VLA 主干
        self.vla_backbone = Qwen2VL(config)

    def forward(self, vision_feat, lang_feat, prev_hidden=None, store_action=None):
        """
        vision_feat: [B, D_v] - 视觉特征
        lang_feat: [B, D_l] - 语言特征
        prev_hidden: LSTM 之前的隐藏状态
        store_action: 需要存储的动作序列
        """
        # 1. Working Memory
        working_out, hidden = self.working_memory(
            vision_feat, lang_feat, prev_hidden
        )

        # 2. Episodic Memory (存储 + 检索)
        episodic_result = self.episodic_memory(
            working_out, store_action
        )

        # 3. Action Memory
        action_pred = self.action_memory(working_out, episodic_result)

        # 4. VLA 主干
        vla_output = self.vla_backbone(
            vision_feat, lang_feat, working_out
        )

        return {
            'working_output': working_out,
            'hidden': hidden,
            'action_pred': action_pred,
            'vla_output': vla_output,
            'episodic_result': episodic_result
        }
```

### 4.6 训练流程

```python
def train_memory_vla(train_episodes, config):
    """
    MemoryVLA 训练
    """
    pcm = PerceptualCognitiveMemory(config).cuda()
    optimizer = torch.optim.AdamW(pcm.parameters(), lr=1e-4)

    for epoch in range(config.num_epochs):
        total_loss = 0

        for episode in train_episodes:
            episode_loss = 0

            # 初始化隐藏状态
            hidden = None

            for t in range(len(episode['frames'])):
                # 获取当前时刻数据
                vision = episode['frames'][t].cuda()
                lang = episode['language'].cuda()
                action = episode['actions'][t].cuda()

                # 前向传播
                output = pcm(vision, lang, hidden, action)

                # 计算损失
                action_loss = F.mse_loss(
                    output['action_pred'],
                    action
                )

                episode_loss += action_loss
                hidden = output['hidden']

            # 反向传播
            optimizer.zero_grad()
            episode_loss.backward()
            optimizer.step()

            total_loss += episode_loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss / len(train_episodes):.4f}")

    return pcm
```

---

## 5. 实验结论

### 5.1 基准测试

#### 5.1.1 LIBERO 基准

| 模型 | Long-Horizon | Generalization |
|------|--------------|----------------|
| 不使用记忆 | 基准 | 基准 |
| **PCM (本文)** | **最优** | **最优** |

#### 5.1.2 CALVIN 基准

| 模型 | 多步任务成功率 |
|------|--------------|
| 无记忆 VLA | 较低 |
| **PCM (本文)** | **最高** |

### 5.2 消融实验分析

| 消融项 | 分析 |
|--------|------|
| **Working Memory 容量** | LSTM 隐藏维度影响短中期信息保留 |
| **Episodic Memory Top-K** | 检索更多记忆提供更丰富上下文 |
| **记忆更新频率** | 平衡存储开销和性能 |

---

## 6. KnowHow（核心洞察）

1. **工作记忆与情景记忆的协同**：LSTM 工作记忆处理即时信息，RAG 情景记忆提供历史经验检索

2. **神经形态记忆的启示**：借鉴人类认知科学中的记忆分层机制

3. **与 LCV/HV 的对比**：指出长上下文窗口和层次化 VLA 的局限性

4. **记忆检索的实用性**：基于向量相似度的记忆检索在机器人操作中的价值

---

## 7. 总结

### 核心贡献

1. **PCM 框架**：首次在 VLA 中系统性地整合工作记忆和情景记忆
2. **模块化设计**：Working/Episodic/Action Memory 分工明确
3. **实验验证**：在 LIBERO 和 CALVIN 基准上验证有效性

### 局限性

1. **记忆容量限制**：向量数据库规模影响检索效率
2. **计算开销**：多模块协同增加推理延迟

---

## 参考链接

| 资源 | 链接 |
|------|------|
| **论文** | [arXiv:2508.19236](https://arxiv.org/abs/2508.19236) |
| **代码** | 待补充 |
| **项目主页** | 待补充 |
