## 日期：20260511

### RL 领域

**论文：（通用领域思考题 - 无新增论文）**

1. [基础概念题] PPO 算法中的 clipped surrogate objective 的数学形式是什么？它是如何防止策略更新的过大变化的？
   - **答案/思考提示**：PPO 的目标函数为 $L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$，其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是概率比。当 $A>0$ 时，$r_t$ 被 clip 到 $1+\epsilon$ 之上；当 $A<0$ 时，$r_t$ 被 clip 到 $1-\epsilon$ 之下，从而限制策略变化幅度。

2. [深度思考题] 为什么 GRPO 和 FlowGRPO 等 on-policy 算法在长 horizon 任务中往往比 off-policy 方法更稳定，但在 sample efficiency 上却更差？
   - **答案/思考提示**：on-policy 方法每次更新都使用当前策略采集的样本，策略与数据分布一致，避免了 off-policy 的 distribution shift 问题。但这也意味着每次策略更新后旧样本失效，需要重新采集。长 horizon 任务的 credit assignment 困难，off-policy 的 extrapolation bias 在长序列上会被放大，导致训练不稳定。

3. [实践应用题] 如果你要将 PPO 应用到稀疏奖励的机器人操控任务中，但原始 reward signal 几乎全部为 0，你会如何设计 reward shaping 和 auxiliary rewards 来帮助学习？
   - **答案/思考提示**：① 添加 shaping rewards（如与目标的距离、末端执行器速度、关节力矩平滑度）；② 使用 Hindsight Experience Replay (HER) 将失败轨迹转化为成功轨迹；③ 设计 intermediate milestone rewards；④ 使用 intrinsic motivation（ curiosity-driven exploration）；⑤ 考虑用 inverse RL 从 demonstrations 中学习 reward function。

4. [优缺点对比题] 比较 GRPO 和 PPO 在处理 value function estimation 方面的异同，以及各自的优势场景。
   - **答案/思考提示**：PPO 使用广义优势估计 (GAE) 进行多步 bootstrap，平衡 bias 和 variance。GRPO 通过 group-relative advantage 避免使用 value function，减少了对 value estimation 的依赖，但也失去了 value function 提供的 baseline。GRPO 在 value function 难以准确估计的长序列任务中更有优势；PPO 在需要高效 bootstrap 和 value-guided exploration 的任务中表现更好。

5. [深度思考题] 在 on-policy RL 中，importance sampling 比率 $r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\theta_{old}}(a_t|s_t)$ 过大或过小会导致什么问题？如何通过自适应方法缓解？
   - **答案/思考提示**：比率过大会导致策略变化剧烈，可能引发训练崩溃（catastrophic policy shift）；比率过小则学习停滞。自适应方法包括：PPO 的 clip 机制直接限制比率变化范围；Adaptive KL penalty 动态调整 KL 目标；TRPO 通过 KL 约束自然地限制策略变化；一些方法如 MPO 使用 soft constraint 形式而非 hard clip。
