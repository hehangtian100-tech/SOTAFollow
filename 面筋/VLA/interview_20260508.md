## 日期：20260508

### VLA 领域

**通用领域思考题（无当日新论文）**

1. [基础概念题] 解释 VLA（Vision-Language-Action）模型中"动作空间离散化"的设计考量。为什么很多方法选择将连续动作映射到离散 token 序列？
   - 答案/思考提示：① 复用 LLM/VLM 的预训练权重和 next-token prediction 训练范式；② 动作序列可自然地与语言 token 对齐；③ 离散化后可用 teacher forcing 训练，简化推理；④ 可利用大规模互联网文本进行预训练。

2. [深度思考题] π0.7、Fast-WAM 等端到端模型和模块化方案（如 ThinkBot、Uni-World VLA）各有什么优缺点？为什么说端到端方案在 scaling law 上更有潜力？
   - 答案/思考提示：端到端优势：① 梯度可从动作直接传回感知模块；② 参数量大时表现更好（scaling law）；③ 无需人工设计模块接口。模块化优势：① 可解释性强；② 各模块可独立优化/替换；③ 对数据量需求低。

3. [实践应用题] 如果要在只有 10K 条演示数据的场景下训练 VLA 机器人策略，如何利用已有的大规模预训练模型避免过拟合？
   - 答案/思考提示：① 使用 pre-trained vision encoder（如 CLIP、SigLIP）和 LLM；② 冻结或轻量微调这些组件；③ 使用 LoRA/QLoRA 微调 action head；④ 数据增强（图像增强、动作噪声）；⑤ 模仿学习 + 少量 RL fine-tuning。

4. [优缺点对比题] 对比 DVGT-2 的 Geometry-Aware Attention 和传统 cross-attention 在处理 3D 场景理解时的差异。
   - 答案/思考提示：传统 cross-attention 平等对待所有 spatial queries；Geometry-Aware Attention 将 3D 几何先验（深度、相机参数）融入 attention bias，使模型能区分不同相机的观测和空间位置，提升泛化能力。

5. [深度思考题] 在 VLA 领域，memory/history 机制对于长程任务执行非常重要。MemoryVLA 的 Memory Bank 设计和传统的 LSTM/GRU hidden state 方案相比，有何本质区别？
   - 答案/思考提示：Memory Bank 是 external memory，可选择性读写（content-based retrieval），容量不受限；LSTM hidden state 是 internal recurrent state，容量有限且信息压缩在固定维度。External memory 使得跨 episode 的历史信息可被精确检索。
