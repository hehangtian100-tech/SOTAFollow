## 日期：20260512

### WM 领域

**论文：（基于 LeWorldModel & DreamerAD 深度思考）**

1. [基础概念题] LeWorldModel 中提出的 SIGReg 正则器与 Dreamer 系列中使用的 EMA (Exponential Moving Average) 编码器在防表示崩溃（Representation Collapse）上的机制有何本质不同？为什么说 SIGReg 是"从统计学角度"解决崩溃问题？
   - **答案/思考提示**：EMA 编码器防崩溃的机制是：通过缓慢更新的指数加权平均，使编码器在短时间窗口内保持相对稳定，提供"软"的正则化。但这本质上是工程技巧，没有从数学上保证不崩溃。SIGReg 的核心是 Cramer-Wold 定理——任意分布可以通过其协方差矩阵完整描述。SIGReg 强制 latent embedding 的分布匹配高斯分布（全分布匹配而非逐点匹配），用数学约束从根本上消除常数解（常数向量的协方差为 0，天然被正则化惩罚）。这相当于对 latent space 施加了信息论约束：模型必须保留输入的变化信息才能满足高斯分布假设。

2. [深度思考题] DreamerAD 提出了"Analysis-by-Synthesis"世界模型与 DreamerV3 系列的纯隐式世界模型在表示学习和规划范式上的核心分歧是什么？Pixel-level reconstruction 是否真的比 latent dynamics prediction 提供了更丰富的监督信号？
   - **答案/思考提示**：DreamerAD 的分析-综合范式：世界模型需要重建未来观察（reconstruction），隐式认为"能重建好图像 = 学到了物理规律"。这提供了丰富的多模态监督（颜色、纹理、几何），但计算代价高且语义层次低。DreamerV3 的隐式动态建模只预测隐空间中的 future state，效率高但监督信号更抽象，可能丢失物理细节。两者分歧本质是"reconstruction is the objective" vs "reconstruction is only for learning representations"。Pixel-level reconstruction 确实提供了更丰富的监督（像素级细节），但这些细节对 planning 是否必要仍有争议——很多物理规律（碰撞、速度）可以在粗糙的隐空间中捕获，因此 LLM-style 的离散表示+预测可能足够。

3. [优缺点对比题] 基于 Diffusion 的世界模型（如 Genie-2、IMAGINE）和基于 RNN/Transformer 隐式世界模型（如 LeWorldModel、DreamerV3）在处理多模态未来预测时的表示能力差异是什么？为什么说"多模态"是世界模型能力的试金石？
   - **答案/思考提示**：Diffusion 模型通过 denoising 过程在每步对条件分布建模，自然捕获多模态（如"左转"或"右转"两种可能未来）。隐式世界模型需要在单步预测中同时捕获所有模式，输出确定性向量，容易出现 mode averaging（预测"平均方向"而非多模态）。多模态是试金石的原因：真实世界固有多模态——同一状态可能有多种合理未来。只能预测单一未来的世界模型在分布外场景（corner cases）会失效，而多模态模型能生成多种应对方案供策略选择，这是通往鲁棒决策的关键。

4. [实践应用题] 假设你需要为一个仓储物流机器人设计世界模型，该场景需要：(1) 实时规划（<50ms），(2) 对突发障碍物的快速重规划，(3) 能够生成长期（>100步）规划轨迹。请从世界模型架构选择、latent space 设计和训练策略三个角度给出具体方案。
   - **答案/思考提示**：架构选择：使用 LeWorldModel 风格的轻量 JEPA（~15M 参数），而非计算密集的 Diffusion world model，保证单步 forward < 10ms；或使用 Fast-WAM 的 action model 预测未来动作序列。Latent space：使用 stochastic latent（Gaussian）而非 deterministic，以便建模多种可能未来；latent dim 不宜过大（128-256），过大增加计算延迟。训练策略：① 在正常操作数据上预训练世界模型；② 用对抗性数据（突发障碍场景）微调，提高在低概率区域的建模能力；③ 使用 rollouts-based 自适应：如果实时检测到规划失败（reward 突变），触发世界模型在线更新。实时重规划：使用 MPC + 世界模型，每步重新预测前 10 步，允许快速中断和重规划。

5. [深度思考题] 世界模型训练中常出现的"rollout divergence"问题（想象轨迹越走越偏离真实）本质上是什么原因造成的？LeWorldModel 声称在 1000+ 步 rollout 中保持稳定，这是否意味着 rollout divergence 被彻底解决了？还有什么潜在的失效模式？
   - **答案/思考提示**：Rollout divergence 的根源是 error accumulation 和 extrapolation：每一步的预测误差在多步展开中被放大，模型对自身错误越来越自信（confident but wrong）。这与 RL 中的 deadly triad（off-policy + bootstrap + function approximation）有类似本质。LeWorldModel 的 SIGReg 通过强制 latent 服从高斯分布，某种程度上约束了预测的尺度，避免极端预测导致快速崩溃；同时 stochastic latent space 在每步注入 noise，提供了类似 ensemble 的效果，减缓了错误累积。但潜在的失效模式：① 如果领域偏移太大（如新物体、新光照），latent space 的分布假设失效；② 多模态场景下，stochastic 采样可能选到低概率模式导致后续完全错误；③ 1000 步的稳定性不代表在其他任务上稳定，可能只是在这类任务上误差累积慢。
