# 原GAT之后串联8头邻域Q/K/V：唯一候选（2026-10-28）

用户授权：按照“原GAT之后、肽键特征拼接之前”的建议执行实验并查看结果。

## 固定结构与唯一改变

保留NodeEncoder/H256、原单尺度k5、单全局节点、FiLM、3层GCN、2层原4头加性GAT、静态原图/nocharge、late 15维理论特征、原concat四状态头α=1/Brier项=0。

```text
NodeEncoder → k5 → 单全局打包/FiLM → 3GCN → H3 → 原GAT4 → 原GAT4 → H5
                                                                      ├→ 原H5 ─────────────┐
                                                                      └→ QKV8(H5) → Wo ────┤相加
                                                                                            ↓
                                                                           原肽键特征拼接/四状态头
```

H5=GAT2(GAT1(H3))。候选Hout=H5+Wo A8(H5)。已完成的并行候选为H5+Wo A8(H3)，本轮只改变新增注意力输入位置，不重训该并行候选。

中心目标i从其完整H5_i生成Q，原图入邻居j从完整H5_j生成K/V。8头各从完整256维投影到32维，按中心入邻域softmax，再对边门控后的V加权求和；拼接8头后用Wo投影回256维。保留同一边bias/value gate和原单全局双向边。不存在新的边、虚拟节点、padding节点、跨肽链边或真实3D结构假设。

沿用同一个ParallelNeighborQKV8计算模块、参数名和初始化方式；内部配置parallel_neighbor_attention_mode=serial_qkv8，仅为新路由标识。新旧并行/串联learned参数和参数数目必须相同。四个persistent marker沿用同名，串联mode/schema/heads/fusion=[2,1,8,1]，旧并行=[1,1,8,1]。三模式none/parallel/serial任何错误模式权重都必须在strict=False下也拒绝，且拒绝前不拷贝共享参数。

Wo的weight/bias零初始化；所有新增模块在历史原模型初始化之后fork_rng构造，保留原主干初始参数和后续随机流。初始完整预测/四状态logits/RNG必须与原模型精确一致。保留原两层GAT且它们继续参与训练；零初始化不保证训练后F1不下降。新增模块不另加LN、dropout、FFN、门控结构或缩放超参。

## 对照、预算与冻结协议

只新增该候选seed42五个固定序列划分1222/2252/3514/6072/9075，从头训练、不加载旧权重。内部splitbase42固定；实际种子按原_stable_fold_seed记录。最多5张确实空闲GPU，一卡一划分；不足时按实际空闲资源调度，不抢占其他任务。

主对照复用Stage11 single seed42五份权重，ops=/mnt/pvc/graphtrans/four_state_parallel_global_replication_ops，源码111c23313da9ae11a7ee33f8357be0a1876086ec。pod必须再次严格核验生产源码、关闭分支schema/初始化/RNG/真实任务一步等价、五组数据/训练/内部验证/外测/目标SHA、实际种子和5权重strictCPU。resultSHA=683e19aa5358f9870b4b6ee298d45d3de4493ddb26c67d1f7fdf5d9c0472409d；registrySHA=4f4b7d476d94fe25126d4ca8d4cf85ab2e1285c5d910cff624e98fd62bb61254。

基线严格复用通过才唯一提交候选5次训练；不通过先报告，不擅自扩大预算补训。原300 epoch上限、batch128/val512、AdamW/cosine/warmup/早停、group_uniform折叠训练、完整原谱内部验证、strict-max有效键F1选checkpoint全部不变。外测固定p>0.5，所有指标去padding。不按外测调整epoch/阈值/种子或结构。不重跑已完成的替换版/并行版，不自动独立种子复验或组合，不修改论文主模型与主表。

## 预设联合工程目标

所有条件同时满足才称达到“保住F1并提升排序”的本轮目标，沿用上一轮规则：

1. 平均肽键F1差≥0；平均谱图平均P/R调和F1差≥0。
2. 平均q_seq差≥0.01，且至少4/5划分q_seq改善。
3. 平均Top10%序列富集比差≥0.01。
4. 平均q-Brier变差≤0.001；平均AUC下降≤0.001。

全报九项指标：有效键F1、BondAcc、AUC、谱图平均P/R及其调和F1、q-Brier、q_seq、Top10。报告均值样本SD、配对差样本SD、逐划分胜平负和资源。不能事后更换目标、把小幅F1下降写成“保住”或只报有利指标。

## Pod预检与终态审计

全部数值验证、数据处理、训练、指标复算、绘图只在pod；本机仅编辑代码配置、同步和查看小报告。

预检：独立逐中心逐头QKV前向和全部梯度、边排列/断连隔离/空入邻域、原主干全部state与RNG、原任务一步等价、串联与并行完全同learned参数和初始化、三模式六方向拒绝且不发生部分加载、零分支初始完整输出等价、非零fixture验证branch_input==H5且不等于H3、原两GAT各执行一次、H5+residual到原有效肽键两端、真实四状态两步QKV梯度和原GAT梯度、最长train128/val512及strictCPU roundtrip。

终态：新旧10份权重strictCPU，实际mask/seed/marker/schema/来源/config/data/target SHA，同次float32九位原始预测独立复算；每份候选权重固定内部验证前16行，独立QKV公式/hook核验读取H5及融合路由。所有完成报告包含全部负结果。注意力描述量不代表化学因果重要性。

独立ops=/mnt/pvc/graphtrans/dbond_serial_neighbor_qkv8_20261028_ops；本地小报告F:/codexout/dbond_serial_neighbor_qkv8_results_20261028/，不覆盖历史队列。训练用完成/失败事件，不反复读取epoch日志或运行进度。只有精确ops和launch.root匹配的新终态才触发持久后台审计；无变化安静。准备好结果并同步小报告后ack精确key、队列audited，停止该提醒。失败保留权重、如实报告，不自动重训/跳过；仅底层训练completed的登记定位故障可独立恢复登记。

五组划分重叠且仅一次训练初始化，是探索性证据；新增容量与注意力机制共同变化。并行与串联等参数只能控制这两种新路由之间的容量差异，不能排除相对原基线的容量作用。
