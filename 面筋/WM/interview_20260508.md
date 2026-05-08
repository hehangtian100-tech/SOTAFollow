## 日期：20260508

### WM 领域

**通用领域思考题（无当日新论文）**

1. [基础概念题] 世界模型的核心价值是什么？请从「预测未来」和「规划」两个角度解释。
   - 答案/思考提示：① 预测未来：学习环境 dynamics p(s_{t+1}|s_t, a_t)，可在 imagination 中评估动作后果，无需真实交互；② 规划：在 imagined rollout 上反向传播优化动作序列（MPC），或用 gradient-based 方法改进 policy。

2. [深度思考题] Dreamer 系列和 World Models（Ha et al.）都学习世界模型，但采用不同的表示学习方式。Dreamer 的 EPISOL 相比 VAE-based 世界模型有何优势？
   - 答案/思考提示：EPISOL（Episodic State Object Layers）通过分层潜在变量建模，将 episodic memory 和 semantic representation 解耦。这解决了传统 VAE posterior collapse 和表示模糊的问题，使 latent 空间更具语义可解释性。

3. [实践应用题] 如果你需要为一个高频交易环境设计世界模型（状态空间高维、奖励信号稀疏），会面临哪些独特挑战？如何解决？
   - 答案/思考提示：挑战：① 状态空间复杂（订单簿、新闻、宏观数据）；② 奖励稀疏导致 representation learning 困难；③ 市场 non-stationarity。解决：① 混合表示（CNN 处理订单簿 + Transformer 处理序列）；② 逆强化学习或 contrastive learning 辅助表示；③ online adaptation 机制。

4. [优缺点对比题] 对比 JEPA（Joint Embedding Predictive Architecture）和 GAN-based 世界模型的表示学习策略。
   - 答案/思考提示：JEPA：predict in latent space，避免生成细节，sampling-efficient，依赖负样本防止 collapse。GAN：predict in pixel space，生成更逼真但训练不稳定（minimax）。JEPA 更适合抽象空间预测，GAN 适合图像级生成。

5. [深度思考题] 在 Being-H0.7 和 Fast-WAM 等自动驾驶世界模型中，如何处理传感器融合（camera、LiDAR、radar）带来的异构表示问题？
   - 答案/思考提示：① 独立 encoder 各自处理不同传感器；② Cross-modal attention 融合；③ 统一到 BEV 或 3D 空间表示；④ 对于 LiDAR/radar，可用 point cloud encoder 或直接转换到图像格式。关键是保持时间一致性（sequential consistency）。
