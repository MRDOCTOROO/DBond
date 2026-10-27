# 单全局节点 k5 上的 8 头邻域 Q/K/V 注意力（2026-10-27）

用户明确授权中心节点生成 Q、邻接节点生成 K/V，以加权 V 和聚合到中心节点，8 个头，先做一次实验。

## 唯一候选

从已冻结的单模型 seed42 基础源码 111c23313da9ae11a7ee33f8357be0a1876086ec 派生隔离快照。
保留 NodeEncoder H256、单尺度 k5、单全局节点、FiLM、3 GCN、15维理论特征、原 concat 四状态头 alpha1/Brier0。
只把原两层 4 头加性 GAT 替换成两层 8 头目标节点 Q / 入邻居 K,V 的缩放点积注意力。
每头由完整 256 维表示各自投影到 32 维。不是先将原向量切成八段。

对有向边 j→i、每个头 a：

q_i^a=W_Q^a h_i；k_j^a=W_K^a h_j；v_j^a=W_V^a h_j。
s_ji^a=(q_i^a·k_j^a)/sqrt(32)+b_a(e_ji)。
alpha_ji^a 在同一中心 i 的所有入邻居上 softmax。
m_i^a=sum_j alpha_ji^a [sigmoid(G_a(e_ji)) ⊙ v_j^a]。
h'_i=Dropout(LayerNorm(h_i+W_O concat_a m_i^a))。

保留已有边偏置和边消息门控，避免同时删去边信息。邻域仍是原静态序列距离图及单全局节点双向边；无新边、无层级虚拟节点、无跨肽链边、无额外全链稠密注意力。
空入邻域的注意力更新为0，仍走原残差/LN约定。仅有效节点打包后参与消息传递；padding不参与注意力或指标。
不额外增加 Transformer FFN、注意力dropout、归一化层或损失。

## 预算及比较

仅候选五个固定序列划分 1222/2252/3514/6072/9075，从头训练；训练 base42，内部划分 base42。
实际逐组训练及划分种子使用原 _stable_fold_seed 规则，记录在 smoke/launch/checkpoint 中。
最多使用5张确实空闲GPU，一卡一划分；资源不足仅用空闲GPU顺序调度，不能占用他人任务。
历史对照是 Stage11 single，单全局+k5、原4头GAT、seed42；先严格审计数据/目标/训练/评估/权重选择及基线代码关闭新模式的一步等价。
严格复用通过则只新增5次训练；不通过不得把不匹配的基线冒充匹配对照，先报告具体问题。

300 epoch上限、原早停/AdamW/cosine/warmup、batch128/val512、group_uniform折叠训练和完整原谱内部验证均沿用。
仅内部有效键F1的原strict max规则选checkpoint。外测固定p>0.5，全部指标去padding。不得按外测选择epoch/阈值/seed。

## 预检和审计

所有数值/数据处理及训练仅pod。关闭新模式必须保持历史state schema、初始化及后续RNG、真实任务一步更新精确等价。
候选在历史全部初始化完成后隔离RNG构造，未修改模块初始权重和后续RNG保留；替换模块并非与旧GAT初始等函数。
新增模式/schema/头数marker，禁止跨模式或不匹配头数checkpoint加载。
独立逐中心逐头公式及全部梯度、边排列不变、断连图隔离、空图/孤立节点，真实四状态梯度、最长train128/val512显存预检。
每个正式权重须strict CPU恢复，核验训练/划分实际种子、有效掩码、模式和配置，原始同次预测9位float32、全部来源SHA。
终态后独立逐谱图去padding复算 BondF1/BondAcc/AUC、谱图平均P/R及其调和F1、qBrier/q_seq/Top10，全部逐划分和负结果保存。
报告均值±样本SD、候选减对照的配对差±SD、胜/平/负、参数/边数/时间/显存。

预设结构工程门槛：平均BondF1差≥0.002，BondF1至少4/5改善，平均AUC差≥0，平均qBrier差≤0。
排序指标全报，不事后更换主门槛。本轮无自动seed43复验或组合；结果报告后再决定。
这些是重叠序列划分上的探索性比较，不能当五个独立数据集的确认性检验。
本对照同时改变打分机制、头数及参数量，若有收益只说明本完整候选有效，不能单独归因于8头、交叉注意力或容量。
注意力权重仅作描述，不能解释为真实空间接触或因果化学作用。既有论文主模型/主表保持不变。

## 完成事件与恢复

独立ops=/mnt/pvc/graphtrans/dbond_neighbor_qkv8_20261027_ops；独立结果目录F:/codexout/dbond_neighbor_qkv8_results_20261027。
单次成功smoke、独占launch.claim与launch.json、唯一worker。训练中不pull/改源，不重复launch，不读取epoch进度。
完成/失败通过dbond_completion_events.py scan；只处理本队列ops和launch.root精确匹配的新事件。
训练完成事件不等于审计通过；先审计并生成报告/看板，再ack精确key。失败不自动重训。
