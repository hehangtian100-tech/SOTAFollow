## 日期：20260509

### WM 领域

**论文：《AIM: Intent-Aware Unified World Action Modeling with Spatial Value Maps》**（arXiv:2604.11135）

1. **[基础概念题]** AIM 论文指出的"WAM（World Action Model）的关键瓶颈"是什么？为什么 action head 不能直接从未来 RGB latent 反推出控制意图？
   - **答案/思考提示**：未来 RGB 回答的是"世界会长什么样"，但动作控制真正需要的是"在哪里交互、为什么在那里交互"。在机器人操作里，真正决定下一步动作的信号往往很稀疏（夹爪要接触的瓶身区域、物体要放置的支撑面）。从 dense RGB future 恢复控制意图是一个隐式逆动力学问题，信号太弱且容易受外观干扰。

2. **[深度思考题]** ASVM（Action-based Spatial Value Map）和 intent-causal self-attention 的设计意图是什么？它们如何保证"动作分支不能偷看未来 RGB"这个约束？
   - **答案/思考提示**：ASVM 是与未来 RGB 对齐的空间价值图，起到"控制相关空间结构"的显式化作用。Intent-causal attention 规定了信息路径：Language/history → future RGB dynamics → future ASVM → future action。动作 token 只能看到当前观察、历史动作、future ASVM，不能看 future RGB——这相当于在结构上强制动作分支必须经由 ASVM 这个信息瓶颈，而非直接从 RGB 特征"猜"控制意图。

3. **[实践应用题]** AIM 采用了两阶段训练（Stage I 监督训练 + Stage II GRPO 后训练），请分析 Stage II 中"冻结 video/value 分支，只更新 action head"的设计是否合理？如果同时更新所有分支会有什么风险？
   - **答案/思考提示**：Stage I 联合训练已经建立了 video/value 与 action 的对齐关系；Stage II 冻结 video/value 是为了保护预训练的视频生成先验和空间价值图表示，只让 action head 适应具体任务。如果同时更新所有分支，video/value 可能为了迁就 action 的错误而遗忘预训练先验，导致世界建模能力退化——这是一个保持已学到知识的选择性更新策略。

4. **[优缺点对比题]** 与 LingBot-VA、Motus、Fast-WAM 等其他 WAM 相比，AIM 的核心创新是什么？为什么说"在 dense RGB future 和 action 之间加入 ASVM 接口"是一个必要的设计而非过度工程化？
   - **答案/思考提示**：AIM 的核心创新是提出 ASVM 作为 WAM 的动作接口，以及 intent-causal attention 强制动作分支只能经由 value map 读取未来信息。这不是过度工程化，因为：如果动作可以直接从 RGB future 解码，模型会退回到"从密集外观表示隐式反推控制"的老路，ASVM 起到信息瓶颈+显式化的双重作用——既压缩语义又把控制所需的空间意图显式化。
