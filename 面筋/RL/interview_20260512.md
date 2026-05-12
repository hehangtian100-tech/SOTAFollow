## 日期：20260512

### RL 领域

**论文：（基于 XXPO系列 & Near-Future Policy Optimization 深度思考）**

1. [基础概念题] GRPO 中 "Group-relative Advantage" 的计算方式与 PPO 中基于 value function 的 GAE 优势估计有何本质区别？Group-relative 采样为什么能替代 value function？
   - **答案/思考提示**：GRPO 对同一个 state 采样 G 个 action，计算每个 action 的平均 reward 作为 baseline：$A_i = \frac{1}{|G_i|}\sum_{j \in G_i} r_j - \frac{1}{|G|}\sum_{k \in G} r_k$。这本质上是 within-group normalization，避开了对全局 value function 的依赖。Value function 估计的是绝对 Q 值（需要 bootstrap），而 group-relative 只关心同组内的相对排序，不依赖时序 bootstrap，因此天然避免了 bootstrap 带来的 extrapolation bias 和 value overestimation 的问题。但代价是无法利用跨状态的全局信息进行 baseline 校正。

2. [深度思考题] XXPO 系列算法（GRPO → GSPO → DAPO → GMPO）的演进路径中，每个算法解决了前一代的什么问题？这种"打补丁"的演进模式是否说明了 policy optimization 理论框架的不完整性？
   - **答案/思考提示**：GRPO 解决了 value function 高成本问题，但 group-relative 仍存在估计方差大、对 reward noise 敏感的问题。GSPO 将 group-relative 扩展到 sequence level，解决了单步 reward 信号噪声问题。DAPO 发现 group-relative 反而导致 over-exploration（多样本平均压低了优势），提出 decoupled clip 和 dynamic sampling。GMPO 使用几何平均替代算术平均，增强了对极端 reward 的鲁棒性。这种演进说明：policy optimization 本质上是在 variance reduction（稳定训练）和 bias introduction（限制探索）之间的权衡，目前缺乏统一理论框架来同时解决所有问题。

3. [优缺点对比题] On-policy (PPO) 和 off-policy (Q-learning/DQN) 方法在 credit assignment 上的差异是如何影响它们在 sparse reward 任务中的表现的？理论上如何弥补 off-policy 在 long-horizon credit assignment 上的劣势？
   - **答案/思考提示**：Off-policy 方法如 DQN 使用 replay buffer 打破时序依赖，可以复用任意旧策略的样本，但 bootstrap 从 max Q 估计中引入 overestimation bias，在长序列中这个 bias 会被多次累积放大（deadly triad）。On-policy 方法虽然避免了 extrapolation bias，但 sample efficiency 极低。理论弥补方向：① 使用 n-step Q-learning 而非 1-step bootstrap（减少 bias 但增加 variance）；② Retrace 算法用 IS ratio 修正 off-policy 误差；③ 使用 multi-step returns 配合 retrace；④ 对于 sparse reward，可结合 reward shaping 或 HER 来提供 dense intermediate signals。

4. [实践应用题] 如果你需要在一个 1000+ 步的长周期机器人任务（如家庭服务机器人执行"整理房间"这样的高层任务）中应用 RL，原始 reward 只有任务成功/失败两种，你会如何设计课程学习和 reward shaping 策略？
   - **答案/思考提示**：① 课程学习：将"整理房间"拆解为"抓取物体→移动到目标区域→放置"的子任务序列，每个子任务单独训练到一定水平后再串联；② Hindsight Relabeling：用 HER 将失败轨迹中的状态重新标记为"成功"（将最终状态设为 goal），保证每个 episode 都有有效学习信号；③ Multi-faceted reward shaping：添加 shaping rewards 包括 progress（当前状态与目标状态的差距递减）、efficiency（时间步惩罚）、subgoal achievement（到达子目标区域的 bonus）；④ Intrinsic motivation：添加 curiosity bonus 鼓励探索未访问的状态空间；⑤ 用 PLR（Prioritized Level Replay）优先重放那些具有适当难度（not too easy, not too hard）的轨迹。

5. [深度思考题] 在 PPO 中，GAE(λ) 的 λ 参数本质上控制了 bias-variance tradeoff：λ=0 时等价于 TD(0)，λ=1 时等价于蒙特卡洛。请分析：在实际训练中，为什么通常 λ 不会设置为极端值（如 0 或 1）？如果任务 horizon 很长，λ 的最优选择会如何变化？
   - **答案/思考提示**：λ=0 (TD(0)) 的优势估计方差最小但 bias 最大（完全依赖 bootstrap），如果 value function 初始不准，TD(0) 会持续累积误差；λ=1 (Monte Carlo) bias 最小但方差最大，在长序列中回报的方差随 horizon 指数增长，估计极不稳定。GAE 通过指数加权 $\lambda$ 在两者间插值，实际中 λ 通常在 0.9-0.99 之间。任务 horizon 越长，单步 TD error 的偏差在多步累积后越大，因此应该增大 λ 来更多依赖真实 return，减少对 bootstrap 的依赖——但 λ 太大又会导致 variance 爆炸，实际需要通过实验调参。深层任务（如 1000+ 步）建议从 λ=0.95 开始，配合 value function 的 periodic reset 或 retraining。
