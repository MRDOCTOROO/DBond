# 单尺度 k=5 的独立训练随机种子复验 v1

2026-10-23 用户明确授权：复验单尺度 k=5，对照 256 维四状态 α=1 无卷积模型；保持原数据划分及去 padding 指标，在 pod 训练，终态后提醒并分析收益能否复现。此为 stage8，仅两臂，不恢复其他探索。

## 固定问题与两臂

发现批次为 four_state_node_conv_batch/20261023_055934，源码 fba57b5412ba12a96a4bd0588fe688a5ff11d856，已独立复核并 ACK。原 seed42 的 k5 相对 none 肽键 F1 +0.582625 个百分点、五组全部改善；本轮检验换训练随机种子后同轮差值能否保持，不推算绝对指标，不保证可复现。

| 顺序 | node_conv_mode | 隐藏维度/头 | 训练 base seed | 内部验证 split base seed |
|---|---|---|---:|---:|
| 1 同轮对照 | none | 256 / four_state_mlp，α=1 | 43 | 42 |
| 2 候选 | single_k5 | 256 / four_state_mlp，α=1 | 43 | 42 |

五组原重复序列级 80/20 划分（1222、2252、3514、6072、9075），各臂各组一次从头训练，共 10 次。实际训练种子为 _stable_fold_seed(43, fold)，实际内部划分种子仍为 _stable_fold_seed(42, fold)，两者逐组记录和核验。seed43 对本轮卷积模型是未测试的新训练随机性；它曾用于不同宽度复验，不据旧宽度结果选择本轮训练配置。不续训任何旧权重，不加入多尺度或额外种子搜索。

single_k5 沿用完整 NodeEncoder 输出 [B,L,256] 上的 Conv1d(256,192,k=5,pad=2)→SiLU→Linear(192,256) 残差，接在 global 创建/打包及 FiLM 前；投影 weight/bias 零初始化，新模块初始化 fork_rng，显式清除 padding，无任意缩放系数。模型实现与发现批次完全相同，none 1,706,924 参数、k5 2,002,284 参数，新增容量仍是解释边界。

固定 3GCN+2GAT、4头、原 concat 融合、15维理论 late、static edges、FiLM/global、nocharge、原位置规则、Brier loss=0。折叠训练、group_uniform、完整原谱内部验证、strict max 有效键验证 F1 选 best_model、固定阈值0.5、去 padding 指标保持不变。优化器、学习率、warmup、dropout、batch128、val512、epochs300、patience25等继承原配方。只允许训练 seed 与独立输出/缓存身份改变，内部 split 显式保持42。随机训练不仅改变初始化，也可能改变 dropout、采样和优化轨迹；不宣称两臂所有随机流逐元素一致。

## 来源与执行

前置要求 stage7 索引 reviewed，准确 launch/terminal root 和 fba57b5 源码一致，自动 review_receipt 与独立 heartbeat_review 均通过，result/report/registry SHA 匹配，准确完成事件已 ACK。新 ops 为 /mnt/pvc/graphtrans/four_state_node_conv_replication_ops，batch 前缀 checkpoints/graph_transform/four_state_node_conv_replication_batch，独立 launch/terminal，不改旧证据。

本机仅编辑代码、配置与 git，同一 origin 同步后在 pod fast-forward pull；所有预检、数据处理、训练、数值复核、绘图在 pod。仅用空闲 GPU；两臂串行，沿用每 GPU 划分调度。无资源则下次单次尝试，不抢占。每次预检或 launch 需防重复；已完成且来源绑定的 smoke 不重复，未 launch 时核对后续提交。训练开始后源码冻结，不 pull 回执或后续代码，只等终态。来源/预检/执行失败报告并 halt，不自动重跑或跳过。

预检重点：模型源文件与 fba57b5 一致；新旧同臂配置规范化后除声明 seed/身份外完全一致；两臂除卷积分支身份外一致；实际五组 train/val/source SHA 对齐发现批次、最早成功和历史宽度凭据；训练 seed43 与旧42不同但内部划分不变；种子解耦不改变旧默认行为；真实四态 loss/union-q、两步任务梯度、padding/短链/样本隔离及 global/FiLM 路由；strict checkpoint 与模式拒绝；最长真实 train128/val512 的 AMP/显存。复用既有可信分支检查，不重复旧批次 smoke 或旧训练。保存源码/配置/目标/历史回执 SHA 与实际种子。

## 结果判断与完整报告

全报两臂去 padding 的肽键 micro-F1、bondacc、ROC-AUC、谱图平均P与R及其调和F1、q-Brier、q_seq；均值和样本SD、k5−none配对差及SD、逐划分胜平负、参数/耗时/显存。完成时独立核验10份权重 strict CPU 恢复、actual training_rng_protocol、node_conv marker、保存配置、训练/验证/目标来源，原始 CSV 与指标来源绑定；内部验证前16行独立公式核对卷积残差和路由，诊断不当作因果重要性。

沿用预设综合门槛：ΔF1≥0.002（0.2个百分点）、F1至少4/5严格改善、ΔAUC≥0、ΔBrier≤0。另逐项回答 seed42 中 F1/AUC/Brier/谱图F1/排序的改善方向及跨划分一致性是否在 seed43 重现；未过完整门槛也如实呈现局部收益。不得按外测更换 epoch、阈值、seed，或挑结果好的划分。

并列 seed42 发现批次与 seed43 复验的各自同轮 k5−none 配对差，旧数据再次绑定原始结果和权重来源；不合并成十个独立样本，不跨划分混权重，两个训练种子也不能证明总体最优或排除新增容量作用。生成完整中文分析、HTML/SVG均值与逐划分波动图，保存独立 heartbeat_review、源码/tag、配置SHA、全部权重路径/SHA、候选恢复记录。保留原成功4da48ee、fba57b5和所有候选，不改冻结论文主表。

准备结果提醒后 ACK 精确事件、stage8 reviewed、auto_advance=false、allowlist清空，停止自动接续；学习图继续暂缓，任何新实验需明确授权。
