## 日期：20260507

### VLA 领域（通用思考题）

**基于近期 VLA 前沿进展的综合思考题——视觉-语言-动作模型架构与训练专题**

1. [基础概念题] VLA 模型的"动作空间离散化"是实现语言-动作统一建模的关键步骤。目前主流的离散化方法有哪些？它们各有什么优缺点？为什么 Diffusion Policy 不是离散化的？
   - **答案/思考提示**：主流方法：① VQVAE/VQ-GAN tokenizer 将连续动作映射到离散 codebook（如 π0）；② Uniform Binning 简单但破坏动作连续性；③ FiLM-conditioned continuous 输出。Diffusion Policy 不是离散化——它用 denoising score matching 在连续空间建模动作分布，本质上仍是连续生成。离散化的优势是能用 language model 的 next-token prediction 统一建模，坏处是引入 quantization error 且难以精确恢复原始动作的物理意义。OpenVLA 等采用离散动作 token，π0.7 采用 continuous diffusion。

2. [深度思考题] 具身智能中，视觉编码器的选择对 VLA 性能有什么决定性影响？LLAVA 等 VLM 和专用视觉编码器（如 SigLIP、DINOv2）在 VLA 场景中的表现差异可能来自哪里？
   - **答案/思考提示**：视觉编码器负责提取环境状态——这是 VLA 的"眼睛"。差异来源：① 预训练任务不同（DINOv2 自监督学 geometry/structure，LLAVA 用 language supervision 学 semantic）；② 专用视觉编码器在大数据集上学到的手眼协调特征更容易迁移；③ 3D 感知能力（DINOv2 有 depth estimation 能力，LLAVA 弱）。VLA 场景需要的不只是语义理解，还有精确的空间关系判断（物体在哪、距离多远），这正是 DINOv2 等自监督视觉编码器的优势。Vega、DVGT-2 等最新工作都采用专用 vision encoder 而非 LLM vision adapter。

3. [实践应用题] 如果你要在一个全新的机器人平台上部署 VLA 模型，但该平台只有非常少的 demonstration 数据（如少于 100 条），你将如何进行 domain adaptation？需要用到哪些技术？
   - **答案/思考提示**：① 使用视觉动作适配器（visual encoder adapter + action head adapter），冻结语言模型部分，只微调少量参数；② 使用 RT-X 等 cross-robot 数据做 pre-training 再 few-shot adaptation；③ 使用 sim-to-real transfer：用仿真环境生成大量数据再迁移；④ Domain randomization 增强视觉和物理参数的鲁棒性。100 条数据下，LoRA (rank=8~16) + 视觉 adapter 是合理起点。关键是新机器人的 action space 是否和训练数据中的 robot 类型足够接近——如果关节结构和自由度差异太大，纯 adapter 可能不够，需要收集更多同构机器人的数据。

4. [优缺点对比题] 端到端 VLA（如 RT-2/OpenVLA）和模块化 VLA（如 VLA-JEPA、Uni-World VLA）在设计哲学上有什么本质区别？各适合什么场景？
   - **答案/思考提示**：端到端追求 simplicity 和 end-to-end differentiation——所有模块联合优化，理论上能学出最优的感知-动作映射；模块化追求 interpretability 和 sample efficiency——世界模型提供 intermediate representation，允许更好的 planning 和 composability。端到端适合：数据丰富、任务相对简单且不需要复杂 planning 的场景（如单步抓取）；模块化适合：需要长时序规划、多任务共享 world model、需要在训练数据稀缺时利用 offline world model rollout 的场景。VLA-JEPA 用 JEPA 学世界模型再 policy conditioning，是模块化的典型代表。

5. [深度思考题] 为什么当前的 VLA 模型在 zero-shot 泛化到全新视觉场景（如完全不同的房间布局、光照条件）时仍然很脆弱？这种脆弱性是否可以通过更大的视觉预训练解决，还是结构性地需要具身交互数据？
   - **答案/思考提示**：VLA 模型的 zero-shot 脆弱性来自：① 视觉预训练的 distribution gap——CLIP/SigLIP 学到的是 internet image-text alignment，robot 操作需要精确的 pixel-level geometry 和物理属性，这些在 internet 图像中稀缺；② 动作输出需要精确的 3D 空间理解，2D 视觉预训练天然缺乏 depth reasoning；③ 物理常识（接触力、物体刚度等）必须通过交互数据获得，无法从被动视觉中学习。更强的视觉预训练有帮助但不能解决根本问题——需要具身交互数据来补全"物理直觉"。解决方案包括：大规模 robot play 数据（Automa 等）、合成物理数据、physics-informed visual representation。
