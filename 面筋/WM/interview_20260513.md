## 日期：20260513

### WM 领域
**通用领域思考题**

1. [基础概念题] 世界模型（World Model）的核心思想是什么？它与传统的 model-based RL 有什么联系和区别？
   - 答案/思考提示：世界模型的核心是学习环境的状态转移函数和奖励函数的压缩表示，可以想象（生成）未来状态而不需要实际与环境交互。与传统 MBRL 的区别在于：传统 MBRL 通常关注有限时域的 planning（如 MPC），而世界模型强调通用可泛化的环境表征，支持长时域想象、梦醒（dream）等能力。

2. [深度思考题] Dreamer 系列算法采用了 Reccurrent State Space Model (RSSM) 来分离世界的 deterministic 和 stochastic 成分。这种设计背后的洞察是什么？
   - 答案/思考提示：世界的发展既有可预测的 deterministic 成分（如物理定律），也有不可预测的 stochastic 成分（如随机干扰）。RSSM 通过 deterministic 路径传递确定性的序列信息，通过 stochastic path 建模随机性，这使得模型既能记忆长期依赖（通过 h），又能表达环境固有不确定性（通过 s）。

3. [实践应用题] 如果你想用世界模型来提升机械臂的 sample efficiency，但发现想象 rollout 的质量不够好（误差累积严重），你会如何改进？
   - 答案/思考提示：可以从以下角度改进：① 使用 truncated horizon planning，不要做太长序列的想象；② 引入 ensemble 或 dropout 来建模 uncertainty，避免单模型过度自信；③ 混合真实 rollouts 和 imagined rollouts，用 real data 校准 imagined trajectory；④ 使用价值函数重塑（value rescaling）来 early terminate 低质量想象。

4. [优缺点对比题] 基于 VAE/VQ-VAE 的 discrete world model（如 VQ-VAE tokenizer）与基于 JEPA 的 world model（如 LeWorldModel）在建模世界方面有什么本质差异？
   - 答案/思考提示：Discrete world model 将世界状态离散化到 token 空间，可以用语言模型方式自回归预测，但存在信息瓶颈和量化误差；JEPA-based 方法学习连续状态的 predictive representation，不经过离散化，能更好地保留视觉细节。JEPA 通过"世界预测世界"的 objective 避免了重建像素级别的琐碎细节，专注于高层次的语义预测。

5. [基础概念题] 为什么视频扩散模型（VDM）用于世界模型时，长视频生成会出现质量下降的问题？有哪些技术可以缓解这个问题？
   - 答案/思考提示：长视频生成质量下降是因为误差累积 + 推理成本平方增长。缓解方法包括：① temporal compression（时序压缩）减少生成帧数；② 层次化生成（hierarchical generation）：先生成关键帧，再插值；③ 自回归方式生成 video chunks 而非一次性生成全部；④ 引入缓存机制减少重复计算。
