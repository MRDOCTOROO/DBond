# DBond-GT 注意力与多分支建议的源码核验

核验日期：2026-10-29。本文只读核对已冻结源码、小协议和登记记录；下列串联指标引用本轮已通过的独立审计，不另行复算或启动实验。

## 1. 原GAT与8头Q/K/V确实不同

设有向边为邻居 $j\to i$，$i$ 为中心节点，$a$ 为注意力头，$e_{ji}$ 为已有边表示。原4头GAT先以同一头的共享投影 $z_i^a=W^a h_i$ 得到64维表示，打分为：

\[
s_{ji}^a=\operatorname{LeakyReLU}\!\left((a_s^a)^\top z_j^a+(a_t^a)^\top z_i^a+b_a(e_{ji})\right),\quad
\alpha_{ji}^a=\operatorname{softmax}_{j\in\mathcal N(i)}s_{ji}^a.
\]

边偏置位于 **LeakyReLU内部**；投影后的邻居表示经边门控和注意力加权，再聚合、拼接、残差、LayerNorm及dropout。它属于加性图注意力，不是独立Q/K/V缩放点积。[原GAT源码](F:/codexout/dbond_serial_neighbor_qkv8_dev/graph_transform/models/attention_layers.py:165)

新增8头模块每头均从**完整256维向量**生成32维Q/K/V：

\[
q_i^a=W_Q^a h_i,\quad k_j^a=W_K^a h_j,\quad v_j^a=W_V^a h_j,
\]
\[
s_{ji}^a=\frac{(q_i^a)^\top k_j^a}{\sqrt{32}}+b_a(e_{ji}),\quad
c_i^a=\sum_j\operatorname{softmax}_{j\in\mathcal N(i)}(s_{ji}^a)
\left[\sigma(G_a(e_{ji}))\odot v_j^a\right].
\]

此处没有打分LeakyReLU，边偏置直接加在缩放点积之后。8头拼接再经输出投影 $W_O$。八套参数可学习不同关注模式，但仍是同一种机制，不能预先解释成八类化学作用。[QKV源码](F:/codexout/dbond_serial_neighbor_qkv8_dev/graph_transform/models/parallel_neighbor_attention.py:71)

## 2. 已做过什么，哪些仍不是同一个实验

| 设计 | 已核实的实际含义 |
|---|---|
| 两层GAT替换为QKV8 | 两层均替换；同时改变打分、头数和参数量。不是只改4→8头，也不是只换一层。 |
| 保留原GAT＋并行QKV8 | 原两层GAT串行路径以H3为输入；新增单层QKV也读取H3，输出与原GAT结果H5相加。 |
| 当前后置串联QKV8 | 保留原两层GAT，新单层QKV读取H5，$H_{\mathrm{out}}=H_5+W_O\operatorname{Concat}_a(c_i^a)$。 |
| 原加性GAT仅4→8头 | 所核对的登记记录中未见独立完成的同机制头数对照。 |
| 只将一层GAT换成QKV8 | 所核对的登记记录中未见该精确候选的独立完成记录。 |
| 肽键Query读4个全局槽 | stage10/11已做；K/V为四个全局节点最终表示，非附近残基集合。 |
| 旧末端局部cross | 左右端点作两个Query；边、差、乘积、理论投影作四个K/V，非残基邻域。 |
| 肽键Query读附近残基K/V | 与上述两种cross不同；不能据旧实验宣称该精确方案已训练。 |

用户贴出的原型为 $\operatorname{LayerNorm}(h+message)$。当前串联分支**不新增LayerNorm、dropout或FFN**，输出投影权重/偏置零初始化，初始完整函数与原模型一致；原GAT内已有的LN/dropout保持。这两种实现不能称为完全相同。[串联协议](F:/codexout/dbond_serial_neighbor_qkv8_results_20261028/protocol.md:19)

原型示例显式使用了边bias，但其消息式为attention×V；本轮候选还保留原有边特征value gate，因此也不能把示例代码逐字当作本轮实现。

旧stage6门控是六个已有肽键特征槽分别缩放后拼接；S12则将“左右残基及其差/乘积、边、理论”三组投影后加权求和。两者均不等于“局部Conv、QKV、独立残基对MPNN、全局”四路节点消息门控。GCN/GAT本来属于广义消息传递，但不等于已经验证了独立 $M(h_i,h_j,e_{ij})$ 残基对分支与其他三路的组合。

来源：[替换协议](F:/codexout/dbond_neighbor_qkv8_results_20261027/protocol.md:9)、[逐键读四全局槽](F:/codexout/dbond_hierarchical_nodes/experiments/parallel_global_cross_v1.md:13)、[末端局部cross及六槽门控](F:/codexout/dbond_hierarchical_nodes/experiments/four_state_head_fusion_v1.md:16)、[S12三组来源](F:/codexout/dbond_stage12_dev/experiments/stage12_gated_fusion_v1.md:15)。源码存在某个可选模块不等于该模块已有完成训练记录。

## 3. 当前串联结果与证据范围

| 指标 | 原模型 | 后置串联8头 | 配对变化及改善划分 |
|---|---:|---:|---:|
| 肽键F1 | 80.5206% | 80.5728% | +0.0522个百分点，4/5 |
| 谱图平均P/R调和F1 | 76.2674% | 76.2747% | +0.0073个百分点，4/5 |
| q_seq | 0.6494086 | 0.6701686 | +0.0207600，4/5 |
| Top10富集比 | 1.2581367 | 1.2552305 | −0.0029062，1/5 |

AUC配对差+0.0010606（4/5），q-Brier配对差−0.0000106（3/5）。**本轮保住了两项F1均值且提升q_seq，但Top10下降，因此预设联合目标未通过；不能称全部无效，也不能称排序全面改善。** 这仍是五组重叠划分、每划分仅一次训练初始化的探索结果，不能证明八头本身的独立贡献、MPNN的因果机制或真实空间接触。新增参数与新增注意力共同变化，尚未自动替换论文主模型。

## 4. 唯一后续建议（尚未授权、未启动）

若后续要补一项最清楚的机制对照，建议只比较**原加性GAT的4头与8头**：保留原两层GAT公式、边调制、H256、k5、图结构、四状态头及全部训练评估规则，不新增QKV层或多分支门控。这样补足“仅头数变化”的记录缺口，避免继续把头数、点积机制与插入位置混为同一因素。H256固定时每头宽度会由64变32，相关参数变化仍需报告；该对照也不能单凭一次结果证明八头的因果优越性。本建议不构成新实验、独立种子复验或四分支组合的执行授权。
