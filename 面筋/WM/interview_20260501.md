## 日期：20260501

### WM 领域

**论文：《AIM: Intent-Aware Unified world action Modeling with Spatial Value Maps》**

1. [题目] AIM 论文中提出 ASVM（Action-based Spatial Value Map）作为 WAM 的动作接口，请解释为什么不能直接从未来 RGB latent 解码动作，而需要引入空间价值图作为中间层？
   - 答案/思考提示：未来 RGB 图像信息密度高、偏外观重建，而机器人控制真正需要的是"在哪里交互、为什么在那里交互"的稀疏空间信号。直接从 RGB latent 反推控制意图等于把空间定位问题变成隐式逆动力学问题，ASVM 把控制所需的空间交互意图显式化。

2. [题目] AIM 的 intent-causal self-attention 机制是如何规定信息流动的？为什么动作 token 不能直接看 future RGB token？
   - 答案/思考提示：信息路径为 Language/history → future RGB → future ASVM → future action。动作分支被禁止直接看未来 RGB 是为了防止模型退回"从密集外观表示隐式反推控制"的老路，ASVM 作为信息瓶颈把控制意图显式化。

3. [题目] Stage II 的 self-distillation RL 后训练中，reward 是如何计算的？为什么只更新 action head 而冻结 video/value 分支？
   - 答案/思考提示：奖励由稀疏任务成功信号 R_task 和稠密空间奖励 R_value 组成，R_value 来自 action 落点投影到 ASVM 上的响应值。这是一种自蒸馏机制，value head 作为在线教师监督 action head。只更新 action head 是为了避免 RL 破坏预训练视频先验和已学习的 value map 稳定性。

4. [题目] 在 Stage I 监督训练中，AIM 的三个 loss 分别是什么？Video branch 和 Action head 的梯度更新有何关系？
   - 答案/思考提示：三个 loss 是 L_rgb_flow（未来 RGB flow-matching）、L_value_flow（未来 ASVM flow-matching）和 L_action（连续动作逆动力学预测）。Video branch 和 action head 共享每层 self-attention 子层但各自保留 FFN，通过 intent-causal attention 让动作 token 和视觉 token 紧耦合，同时保持预训练视频主干结构稳定。

5. [题目] AIM 在 Hanging Mug 和 Blocks Ranking Size 等任务上表现较差，分析 2D ASVM 的局限性是什么？未来可能的改进方向？
   - 答案/思考提示：2D ASVM 无法充分表达 3D 姿态约束、深度信息和空间排序关系。Hanging Mug 需要悬挂姿态的物理约束，Blocks Ranking Size 需要物体尺寸的语义/几何比较。改进方向可以是引入 3D 空间价值图、体素化表示或结合几何感知模块来处理这些任务。
