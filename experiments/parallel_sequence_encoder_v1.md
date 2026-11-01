# 原图网络与独立序列 Transformer 并行：预先冻结的最小验证

日期：2026-11-01（北京时间）。用户授权：制定合理实验方案并在已授权 Pod 训练、查看效果。本轮只验证一个候选，不开展分支数、融合器、头数、损失或卷积核搜索。

## 1. 核心问题与模型

原 GCN/GAT 作为完整图路线，与直接读取初始残基编码的一层序列 Transformer 并行，二者是否能提供互补信息，在保留肽键预测表现的同时改善序列排序？

共同输入 S 为原 NodeEncoder 输出 [B,L,256]，取自 k5 卷积之前。该表示已包含原位置编码、残基性质及允许使用的 charge/NCE 等条件；不额外重复增加位置编码。

- 原图路线保持：S → 原单尺度 k5 → 原单全局节点/FiLM → 3GCN → 原两层4头加性 GAT → Hgraph。
- 独立序列路线：S → 一层 TransformerEncoderLayer，H256、8头（每头从完整256维投影32维）、FFN512、GELU、pre-LayerNorm、dropout=0。只在同一肽链的真实残基间进行全连接注意力，不使用图边限制，不新增虚拟节点，不引入三维接触假设。
- 融合：Hfused_i = Hgraph_i + Wo T(S)_i，Wo 为256→256线性投影，权重与偏置零初始化。按真实残基位置映射回原 packed 图，原全局节点的输出不加序列分支残差。
- 融合位于原两层 GAT 之后、原肽键特征构建之前。原边/差/乘积/late15维理论特征和 concat 四状态头 alpha1/Brier0 保持。
- 新模块构造保护历史随机流，序列分支不使用 dropout，原模型已有 dropout 保持不变。零初始化仅保证训练起点一致，不保证训练后的性能。
- 这不是先前读取 H3 的邻域 QKV 分支，也不是读取 H5 的后置串联 QKV。Transformer 的输入是 S；原图网络本身也参与正常训练。

## 2. 对照、预算与来源

原单全局节点+k5、H256、四状态头为匹配对照。只训练候选 seed42 的五个固定序列划分：1222/2252/3514/6072/9075。实际训练/内部划分种子分别为949822381/2775888976/1365222152/612878555/1785808201；训练base42，内部划分base42。

优先严格复用 Stage11 single 的五份权重：ops=/mnt/pvc/graphtrans/four_state_parallel_global_replication_ops，源码111c23313da9ae11a7ee33f8357be0a1876086ec；result SHA=683e19aa5358f9870b4b6ee298d45d3de4493ddb26c67d1f7fdf5d9c0472409d，registry SHA=4f4b7d476d94fe25126d4ca8d4cf85ab2e1285c5d910cff624e98fd62bb61254。复用前须核对来源/数据/目标/配置/种子、strictCPU恢复及关闭新分支时原任务一步等价。若复用不通过，暂停并说明差异，不擅自扩大训练预算。

原训练采样 group_uniform、batch128、val512、300 epoch上限、AdamW、cosine/warmup、早停与完整原谱内部验证保持。只按内部验证有效肽键 F1 strict-max 选权重，外部 p>0.5固定、去padding。预合成有效mask必须state=[true,true,false]、env=[true,false]，不能误用原YAML的占位mask。

最多使用五张真正空闲GPU，一卡一个划分；资源不足只按空闲资源调度，不抢占其他任务。一次成功预检后唯一launch。训练用完成/失败事件，不反复读epoch进度。

## 3. 本轮预先指定的判断规则

本轮主问题是保留F1并提升整体排序，不要求所有指标上涨。以下联合条件全部满足，才记为达到本轮工程目标：

1. Bond-F1 均值差 >= 0。
2. q_seq 均值差 >= 0.01，且至少4/5个划分改善。
3. 谱图平均P/R调和F1的均值差 >= -0.002（最多下降0.20个百分点）。
4. AUC 均值差 >= -0.001。
5. q-Brier 均值变差 <= 0.001。

Top10富集比、Precision、Recall和Bond-Acc为次要描述性指标，完整报告其方向及不确定性，不要求全部同步改善。这是新实验的预先规则；历史替换/并行/串联实验保留各自原门槛及结论，不追溯修改。

全报九项指标：Bond-F1/Bond-Acc/AUC、谱图平均P/R及其调和F1、q-Brier/q_seq/Top10，五组均值±样本SD，候选减对照的配对差±样本SD、逐划分胜平负。记录参数量、图边数、实际训练耗时和可验证的显存口径。

## 4. 预检及独立审计

全部训练、数值验证、数据处理、指标复算及绘图只在Pod；本机只编辑代码配置、同步和查看小报告。

预检包括原关闭模式state/init/RNG/真实任务loss-梯度-更新等价，候选共享原参数与零融合初始函数一致，strict=False错误模式加载在任何部分copy前拒绝；独立MHA/pre-LN/FFN公式及梯度、真实长度mask、NaN padding和跨肽链隔离；S输入确为NodeEncoder输出而非post-k5/H3/H5；原各GCN/GAT仍执行一次、真实残基打包/全局不加残差及原预测头左右端点；真实任务两步分支梯度、最长train128/val512和CPU strict恢复。

完成后持久后台独立审计一次：新旧10份权重strictCPU恢复、来源/配置/数据/目标/实际种子、同次float32九位原始预测独立去padding复算；五份候选权重固定内部验证前16行重算完整Transformer公式并验证路线/融合。来源或数值不通过如实halt，保留权重，不重训。仅元数据定位故障可留档后外置最小修复，不放宽数值与来源断言。

独立ops=/mnt/pvc/graphtrans/dbond_parallel_sequence_encoder_20261101_ops；本地小报告F:/codexout/dbond_parallel_sequence_encoder_results_20261101/。只处理精确ops/root匹配的新终态，不ack其他会话事件。报告准备完毕后准确ack并停止提醒。

五组外测划分互相重叠且已经用于多轮模型开发，本轮是探索性比较，不是确认性显著性结论。新增容量与新机制共同变化；不据单轮结果把收益完全归因于Transformer，不自动更换论文主模型、不追加种子复验或其他组合。
