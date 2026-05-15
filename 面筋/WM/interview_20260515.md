## 日期：20260515

### WM 领域
**通用领域思考题**

1. [基础概念题] 世界模型（World Model）的核心目标是什么？它与传统 MDP 模型的强化学习方法（如 Dyna-Q）有何本质区别？请以 Dreamer 系列为例说明。
   - 答案/思考提示：世界模型学习环境的状态转移分布 p(s'|s,a)，可以在潜在空间中进行无限想象和规划，无需真实环境交互。Dreamer 通过 RSSM（Recurrent State Space Model）在 latent space 中学习世界模型：encoder 将图像压缩为 stochastic latent z 和 deterministic h，dynamics model 预测下一时刻的 z，reward model 预测奖励。训练时直接在 imagined latent trajectory 上反向传播 actor-critic 梯度，实现样本效率的大幅提升。Dyna-Q 则是在真实转移上学习，且通常在离散状态空间。

2. [深度思考题] Latent World Model（如 JEPA、LeWorldModel）相比 Pixel-level World Model（如 Dreamer）有哪些优势？为什么说在 latent space 中规划是更有效的设计？
   - 答案/思考提示：Pixel-level 模型需要重建高维图像，浪费大量建模容量在无关纹理细节上；Latent space 模型只建模语义相关的低维表征，过滤噪声。Latent space 规划的优势：① 计算效率高（低维空间）；② 语义更一致（不受像素级无关变化干扰）；③ 更容易学习到物体持久性（object permanence）；④ 对视觉变化（光照、遮挡）更鲁棒。JEPA 的 IBRAIN 通过 encoder-dynamics-predictor 架构预测 latent 表征而非像素。

3. [实践应用题] 假设你要为自动驾驶场景构建一个世界模型，用于高速规划避障任务。你会如何设计世界模型的观察空间、动作空间和奖励函数？请给出架构设计的核心考量。
   - 答案/思考提示：观察空间：多视角相机（前方+后方+侧方）+ 雷达/激光雷达点云 + 车辆状态（速度、加速度、航向）；动作空间：连续控制（油门、刹车、转向）或离散（换道/保持/减速）；奖励函数：安全奖励（与前车距离、与车道线偏离）、效率奖励（速度接近目标速度）、舒适奖励（加速度变化率）。架构考量：① 使用 Bird-eye view latent 表征便于规划和控制；② 集成高精地图信息作为 context；③ 预测模块需考虑多智能体交互（其他车辆意图）；④ 不确定性建模对安全关键场景至关重要。

4. [优缺点对比题] 基于扩散模型的世界模型（如 Galileo、Sora）和基于自回归模型的世界模型（如 World Models、Dreamer）在长程预测上有何本质差异？各自的长处和局限性是什么？
   - 答案/思考提示：扩散模型在像素空间逐步去噪，长程预测时可以保持细节一致性，但计算成本高；自回归模型在 latent space 逐步预测，长程预测有误差累积问题但计算效率高。长处：扩散模型适合高保真视频生成、可以建模多模态未来；自回归模型适合快速规划推理。局限性：扩散模型需要数百步采样不适合 real-time 交互；自回归模型误差累积限制想象 horizon。当前趋势：latent diffusion + AR dynamics 结合（如 EMMA）。

5. [基础概念题] 在 MV-VDP（Multi-View Visual Distinction Prediction）中，区分性表征（Distinction Representation）和生成性表征（Generative Representation）的区别是什么？为什么对于视觉运动规划任务，区分性表征更有优势？
   - 答案/思考提示：生成性表征学习"如何重建观测"，需要建模所有观测细节包括无关背景和纹理；区分性表征学习"什么导致变化"，只建模与任务相关的语义变化（如物体运动、交互结果）。对于运动规划，我们关心的是"这个动作会导致什么结果"，而非"图像看起来是什么样的"。区分性表征的优势：① 对视觉干扰更鲁棒；② 表征更紧凑高效；③ 更直接服务于决策任务；④ 避免浪费建模容量在无关细节上。
