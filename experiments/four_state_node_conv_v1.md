# 四状态基座的节点序列卷积：预注册 v1

2026-10-22用户授权：动态边stage3、末端融合stage6依次完成/复核/报告后，检验图二式多尺度Conv1D。此文件冻结stage7，不代表代码已实现或训练已提交。仅pod做计算、验证、训练和绘图，当前在跑的pod源码冻结。

## 历史和新问题

历史多尺度卷积已完成，不得称首次：770ff82b6fa5bf977c9c5e02b18834efcf5e3566，seq2_batch/20261017_172512，台账§15.19–15.20。当时AA embedding64→Conv3/5/7各32通道→96→256，残差加到NodeEncoder后；binary头、折叠训练和当时折叠内部验证。F1−0.0448个百分点，AUC−0.000453，q-Brier+0.000491，q_seq均值+0.002355但仅2/5改善，未通过门槛。本轮是**完整256维残基编码上的更宽分支＋四状态监督/完整原谱验证**，并新增等参数单尺度对照；不能把旧新差异单独归因某一改动。

## 三臂

| 臂/node_conv_mode | 图前残基序列分支 |
|---|---|
| none | 原模型，无卷积 |
| single_k5 | Conv1d(256,192,k5,padding2,bias=True)→SiLU→Linear192→256 |
| multiscale_357 | 并行Conv1d(256,64,k3/pad1)、k5/pad2、k7/pad3，各SiLU；拼接192→Linear192→256 |

stride=dilation=groups=1，不加额外norm/dropout/自由缩放系数。候选卷积权重数相同：256×192×5=256×64×(3+5+7)，bias与投影维度也相同；实际总参数必须pod独立核验。multiscale−none检验新增序列分支整体收益，single−none检验单尺度分支收益，multiscale−single在相同参数预算下检验当前多尺度设计的额外收益。最大感受野仍为7对5，不能把结果单独归因于多分支本身，不扩增k7等额外搜索。

接入GraphTransformer.forward中NodeEncoder输出[B,L,256]之后、加入/打包global节点及FiLM之前。沿序列长度L卷积，计算布局[B,256,L]；每个输出残基保持256维，不卷积特征维或把不同肽链packed为一条长序列。global不是普通残基，完全不进入卷积；后续FiLM、GCN/GAT保持原顺序。三个核分别观察3/5/7个位置，是序列邻域，不是三维接触。

H_new=H_original+Proj(Conv(H_masked))。末端投影weight/bias零初始化，Conv正常非零初始化；新模块在原初始化/四状态替换之后用独立fork_rng创建。初始候选与none函数相同，第一步投影有梯度、卷积任务梯度为0，第二步卷积可收到非零梯度。不开可调λ、不引入无依据0.2等系数。输入先masked_fill或where清除无效位置（不能用NaN×0），卷积/投影输出再按真实长度清零，残差不污染padding。短链L1/L2使用零填充；混合L0修正为0；新分支独立E/L0测试不要求旧GAT支持全空边图。

## 固定基座与执行

所有臂H256、4头、3GCN2GAT、四状态α1/Brier0、原concat MLP融合、theory late、static edges、FiLM/global、原位置规则和nocharge。旧sequence_adapter_type保持none，避免把新256卷积误当旧AA64分支。新增node_conv_mode与marker/保存配置核验，不复用历史名称加载错架构。**不叠加动态边、stage6获胜头、早融合或128维。**

五组原重复序列级80/20划分，各一次从头训练，共15次。训练base42/内部splitbase42、同历史train/val/sourceSHA、group_uniform折叠训练、完整原谱内部验证、strict max内部有效键F1选权重、固定.5与去padding；原预算、优化器、早停均不变。同轮none重新训练，不能用历史最好值作对照。

计划模块models/node_context_conv.py；脚本scripts/four_state_node_conv_ops.py、test_four_state_node_conv.py；配置pre_synthesis_four_state_node_conv_{none,single_k5,multiscale_357}.yaml。stage6 completed/独立复核/报告/ack后才能pod pull与smoke。预检需原none参数/RNG/CPU函数/梯度及checkpoint兼容、候选初始等价、两候选参数匹配、独立Conv逐位置参考与精确感受野、单样本/混合批不串样本、padding poison和延长padding不改有效输出、global不进入卷积、短链/branch空序列、非零分支两步梯度、真实AMP/候选最大train128与val512显存、五组历史SHA、跨模式配置/marker拒绝及严格恢复。全通过才launch。

ops /mnt/pvc/graphtrans/four_state_node_conv_ops，batch前缀checkpoints/graph_transform/four_state_node_conv_batch，launch/terminal独立目录、source/config/目标SHA绑定。arm串行、现有每卡划分调度；仅空闲GPU，当前无空闲则下次单次尝试，不抢占。失败或来源不匹配halt报告，不自行重跑；提交后源码冻结，只检查完成/失败事件。

## 完整结果报告

全报三臂、三比较和原8指标：F1、bondacc、AUC、谱图平均P/R及其调和F1、q-Brier、q_seq；均值SD、配对差SD、每划分胜平负、参数/耗时/显存。相对none候选采用门槛及multiscale−single机制比较均列ΔF1≥.002、F1至少4/5改善、ΔAUC≥0、ΔBrier≤0。多尺度仅胜none而未胜single时，不能归因多尺度优越；未提升也不证明所有Conv1D无用。重叠划分、单初始化、多轮探索边界明确。

pod生成结构变化图、HTML/SVG指标与逐划分波动图，内部验证固定前16行可记录投影残差范数、各卷积分支激活作为实现诊断，不据此宣称特征重要性/物理机理。严格恢复15份权重，检查来源、实际种子/α/模式；保存源码commit/tag、配置及权重SHA、checkpoint_registry、独立heartbeat_review.json及报告。旧否定实验与当前结论并列，冻结论文主表不改。

准备中文提醒后ack准确终态key、stage7 reviewed，auto_advance=false/allowlist清空，授权队列停止；学习图继续deferred_by_user。所有候选及原成功4da48ee等可恢复版本保留。

终态事件处理与接续是两个可恢复步骤：若前序已reviewed/ack、本项在allowlist且未提交，即使本次scan无pending，也应从索引恢复实现/部署/预检/提交；已完成且来源绑定的smoke不得重复运行，提交成功须立即登记root/gitrev。无新终态且无已reviewed前序待接续时安静结束。
