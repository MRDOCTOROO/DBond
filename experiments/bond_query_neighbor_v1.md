# 肽键Query读取附近残基：单候选最小验证

冻结日期：2026-11-01（北京时间）。用户授权继续验证消息机制互补性，优先进行肽键Query邻域交叉注意力。本轮只实施一个候选、五次固定划分训练；不重跑已完成的串联/并行节点QKV或独立全序列Transformer，不预先启动其他P1/P2结构、头数或宽度搜索。

## 核心问题

由待预测肽键而非单个残基生成Query，直接读取附近残基的上下文，能否在保持肽键F1的同时改善序列排序？这与已完成的四全局槽cross、末端特征槽cross、残基中心QKV均不同。

## 唯一候选

保留原单节点+k5四状态模型：NodeEncoder H256、单尺度k5、单全局节点、FiLM、3GCN、两层原4头加性GAT、static/nocharge、late15维理论特征、原6H concat预测头、四状态alpha1/Brier0。原主干继续从头参与训练，结构不替换、不叠加前轮新分支。

设H5为原两层GAT的最终packed节点表示，待预测键为(i,i+1)，e_ij是原始256维肽键边表示：

1. Q = Wq[H5_i || H5_(i+1) || e_ij]，768→256。
2. K_j = Wk H5_j、V_j = Wv H5_j，各256→256。
3. 固定8头，每头32维，投影读取完整输入向量。
4. 键值集合为原有向图两端入邻居N(i)∪N(i+1)，并显式包含两端残基；同一残基去重。排除全局节点、padding及其他肽链。保留原distance图/max_distance=6，不搜索额外半径。
5. 每个肽键、每个头仅在该集合中进行缩放点积softmax，再对V加权求和；8头拼接为256维。
6. 不增加新边bias/value gate、LayerNorm、dropout、FFN或自适应门控；e_ij已经进入Query。
7. 新输出投影Wo为256→256，weight/bias零初始化。令原肽键头hidden=ReLU(W_original concat6H)，仅将Wo(context)加到该hidden，再复用原dropout和最终4状态线性层。

原H5节点表示和原6H肽键特征不被更新。融合发生在肽键预测头内部，不能描述为串联节点QKV。KV来自近邻残基的H5；这些表示本身已经经过原图主干及全局节点，因此不宣称它们是纯局部物理信息或真实空间接触。

构造新模块时保护原初始化及随机流。none模式不增加checkpoint state；候选mode/schema/heads/query/neighborhood/fusion等marker绑定具体语义，strict=False也必须在任何部分copy前拒绝错误模式。

## 对照与预算

仅新增seed42五个候选训练，固定序列划分1222/2252/3514/6072/9075。训练base42、内部划分base42，实际种子依次949822381/2775888976/1365222152/612878555/1785808201。

严格复用Stage11 single seed42五份基线权重：ops=/mnt/pvc/graphtrans/four_state_parallel_global_replication_ops，源码111c23313da9ae11a7ee33f8357be0a1876086ec；result SHA=683e19aa5358f9870b4b6ee298d45d3de4493ddb26c67d1f7fdf5d9c0472409d；registry SHA=4f4b7d476d94fe25126d4ca8d4cf85ab2e1285c5d910cff624e98fd62bb61254。须再次严格核验来源、数据/目标/配置/实际种子、权重strictCPU及关闭新增模块的真实任务一步等价。复用不通过则停下解释，不自动补基线或扩大预算。

训练采样group_uniform、batch128、val512、300 epoch上限、AdamW/cosine/warmup/早停保持。用完整原谱内部验证的有效肽键F1 strict-max选checkpoint；外测固定p>0.5、全部去padding。预合成有效mask保持state=[true,true,false]、env=[true,false]。

最多五张真正空闲GPU，一卡一划分；资源不足仅用空闲GPU排队，不抢占其他任务。一次完整预检成功后唯一launch，冻结源码不再修改/pull。训练仅监听终态，不反复读取epoch。

## 本轮预设目标及报告

全部满足以下六项才记为达到本轮工程目标：

- Bond-F1均值差>=0。
- q_seq均值差>=0.01，且至少4/5划分改善。
- 谱图平均P/R调和F1均值差>=-0.002。
- AUC均值差>=-0.001。
- q-Brier均值变差<=0.001。

Top10富集比、Precision、Recall和Bond-Acc为次要描述，完整报告，不要求各项同步上涨。不追溯修改历史实验门槛。

九项指标全部报告两臂均值±样本SD、逐划分配对差±样本SD和胜平负；主表来自两臂同口径samepass原始预测独立复算。保留原metric CSV及复算误差；主门槛指标的原值/复算方向或过关冲突则标临界，不宣称稳健过关。记录参数、原图边数、新的bond-neighbor关系条数、实际训练时间与明确口径的显存。

## 预检和终态独立审计

所有训练、数据处理、指标复算、数值验证和绘图只在Pod；本机仅编辑代码配置、同步和查看小报告。CPU验证从开始固定单线程，保留严格断言，正式GPU训练配置不变。

预检包括：legacy none模式state/init/RNG/真实loss-梯度-更新一步等价，候选共享参数与zeroWo初始完整logits/四状态/RNG一致；错误模式strict=False加载无部分copy；独立Python邻居集合与逐键逐头QKV完整公式及全部参数/输入梯度；端点包含、去重、边排列/重复边不变、bond排列等变、global NaN不泄漏及图隔离；nonzeroWo独立fixture验证Q的端点/e_ij和K/V的H5来源、原3GCN2GAT各一次、H5不改、原hidden+delta进入原dropout/final4；真实任务两步Q/K/V/Wo与原GAT梯度；最长train128/val512、CPU strict恢复与五基线完整来源。

completed后只启动一次持久后台独立审计：新旧10份权重strictCPU、samepass float32九位预测/有效掩码/实际种子/source-config-data-target SHA、全部指标复算；每份新权重固定内部验证前16行独立重建邻居集合、完整QKV公式及头内融合路由。三回执绑定后同步小报告、准确ack并停止提醒。

失败保留权重、不自动重训/跳过。仅元数据定位或底层训练已completed的登记故障可留档后外置最小修复，不放宽数值和来源断言。

新增注意力同时增加容量，本轮是整个候选的探索性验证，不能据一轮结果单独归因肽键Query。五组外测互相重叠且已经多轮开发使用；不作显著性或确认性最优宣称。参数匹配控制、独立种子复验及其他分支需后续明确方案，不自动追加，不自动更换论文主模型。

独立ops=/mnt/pvc/graphtrans/dbond_bond_query_neighbor_20261101_ops；本地结果F:/codexout/dbond_bond_query_neighbor_results_20261101/。
