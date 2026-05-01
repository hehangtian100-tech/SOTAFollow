## 日期：20260501

### LLM / FM 基础知识 领域

**论文：《LDA-1B: VLA 机器人操作模型》**

1. [题目] Flow Matching 用于动作生成的核心原理是什么？相比传统 Diffusion Model，它在机器人控制场景有哪些优势？
   - 答案/思考提示：Flow Matching 通过常微分方程（ODE）在数据分布和噪声分布之间进行插值，生成过程是确定性的从噪声到数据的路径。相比DDPM需要多步迭代，Flow Matching 可实现单步或少量步生成，大幅降低推理延迟。在动作生成中，Flow Matching 能高效生成连续动作序列，且收敛速度更快，适合实时机器人控制。

2. [题目] 在 LDA-1B 的 VLA 架构中，VLM（Qwen2.5-VL-3B）如何与 DiT Action Head 进行跨模态信息传递？Future Tokens Embedding 的作用是什么？
   - 答案/思考提示：VLM 提取视觉-语言联合表征后，通过 Cross-Attention 机制将 hidden_states 传递给 DiT Transformer。Future Tokens 是 32 个可学习的查询 token，作为"查询向量"从 VLM 表征中检索相关动作信息。作用：① 提供动作预测的查询接口② 融合多模态信息（视觉、文本、动作历史）③ 隐式学习动作与视觉语义的对应关系。

3. [题目] q99 归一化相比 min-max 和 mean-std 归一化有何优势？为什么适合机器人动作数据？
   - 答案/思考提示：q99 使用 1% 和 99% 分位数而非 min/max，对异常值鲁棒。输出范围固定在 [-1, 1]，适合神经网络训练。可逆，支持反归一化恢复原始值。机器人动作数据常包含噪声和异常值，q99 比 min-max 更鲁棒；相比 mean-std 的零均值假设，q99 的固定范围更利于梯度稳定。

4. [题目] 旋转表示从 axis-angle 转换为 rotation-6d 的动机是什么？6D 旋转表示相比 3D 或 4D (quaternion) 表示的优缺点？
   - 答案/思考提示：axis-angle 有奇异性（单位元附近欧拉角表示不连续），不适合神经网络学习。rotation-6d 通过取旋转矩阵前两列的 6 个元素表示 3D 旋转，连续可微无奇异性，适合学习。相比 quaternion（4D，约束 ||q||=1），6D 更直观；相比 3D 欧拉角，无奇异性问题。在机器人操作中，末端执行器姿态控制需要稳定连续的旋转表示。

5. [题目] 多 embodiment 联合训练时，如何处理不同机器人形态的动作空间差异？
   - 答案/思考提示：LDA-1B 使用 CategorySpecificLinear——不同 embodiment 有独立的动作头 MLP 层。State encoder 接受不同形态的状态（位置、旋转、夹爪），通过归一化和投影到统一隐空间。关键设计：① 共享 VLM backbone（学习通用视觉-语言表征）② 独立 action head（适配各形态动作空间）③Embodiment tag 作为条件输入。这种设计实现跨形态泛化，同时保留各形态特有的控制细节。
