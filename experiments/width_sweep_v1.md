# 四状态模型隐藏宽度对照 v1

本轮只改变 `model.hidden_dim`，所有模型从头初始化。原始输入信息、3层GCN、2层GAT、4个注意力头、位置拼接、全局节点、FiLM及15维理论信息保持不变。隐藏宽度也影响边编码、全局投影和预测MLP，所以这是整套隐藏表示容量的比较，不能单独解释为GCN宽度的收益。

| 配置 | 隐藏维度 D | GAT头数 | 每头维度 | 角色 |
|---|---:|---:|---:|---|
| pre_synthesis_four_state_width_256.yaml | 256 | 4 | 64 | 同轮对照 |
| pre_synthesis_four_state_width_128.yaml | 128 | 4 | 32 | 小容量候选 |
| pre_synthesis_four_state_width_384.yaml | 384 | 4 | 96 | 中容量候选 |
| pre_synthesis_four_state_width_512.yaml | 512 | 4 | 128 | 大容量候选 |

## 可恢复的参考版本

- 最早通过预设提升门槛的四状态实现：`4da48eeab50decd0a31fbf64f2f179478a6624a7`。
- 固定恢复标签：`dbond-four-state-reference-20261020`，指向上述提交，不随未来结果移动。
- 原成功配置：`graph_transform/config/pre_synthesis_four_state_joint.yaml`。
- 原成功权重批次：`/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/four_state_batch/20261019_124645`。
- 原成功报告：`/mnt/pvc/graphtrans/four_state_ops/report.md`。
- 宽度实验源码以本轮 `four_state_width_ops/launch.json` 中的完整 `gitrev` 为准；四档共享此代码，各自配置及SHA、权重路径分别绑定。代码恢复与训练权重恢复是两个步骤，不能只靠git恢复未纳入版本控制的权重。

## 固定协议

四状态soft CE，条件项α=1、union Brier λ=0，关闭新增电荷组成分支。不使用预训练或跨宽度迁移。同一五组重叠的序列级80/20外部划分：1222、2252、3514、6072、9075；这些是重复随机划分，不是五个互不相交的测试折。内部序列划分和数据SHA与原成功批次一致。

折叠条件组训练、group_uniform有效键加权、完整原始谱图内部验证；strict max有效键micro-F1选择checkpoint。AdamW，学习率1e-4、weight decay1e-4、batch128、dropout0.1、warmup40、cosine、最多300 epochs、patience25，固定0.5阈值。四档统一训练配方；不为512单独减batch或调学习率。允许早停产生不同实际epoch。相同seed不等于跨宽度逐元素同初始化或相同batch顺序：模型构造会消耗不同数量的随机数；不据此声称严格共享随机流。

共4×5=20次训练，按256、128、384、512的预先指定顺序逐臂运行，各臂划分任务分配到提交时空闲GPU。只使用空闲设备，不终止其他负载。每臂记录总参数量、可训练参数量和完成时长；时长受早停、设备调度和并发影响，不视为等步速度测试。

## 判定与保存

全部候选分别相对本轮256维报告有效键F1、accuracy、ROC-AUC、谱图平均P/R的调和F1、q-Brier、q_seq；同时提供样本标准差、配对差值、逐划分胜率。padding键全部排除。

预设工程门槛：平均ΔF1≥0.002（0.2个百分点），F1严格改善至少4/5，平均ΔAUC≥0，平均Δq-Brier≤0。全部满足才列入待复验候选。所有宽度如实报告，不根据外部测试分数挑出一个版本作为已确认最优。单种子、重叠划分、多轮探索不能证明统计显著；不替换冻结论文表。通过门槛的模型保留源码提交、配置SHA、每划分checkpoint路径和SHA，后续独立复验再决定推荐版本。

## 完成事件

Pod目录：`/mnt/pvc/graphtrans/four_state_width_ops/`。`launch.json`绑定提交、配置、目标文件、源代码及运行根目录；成功预检记录在`smoke.json`。全部完成后核验原始指标CSV与来源，生成`result.json`、`report.md`、`review_receipt.json`和权重恢复索引，然后写`terminal.json`；失败只写失败终态，不自动重跑。

当前会话提醒只处理该宽度批次的pending终态事件。pending为空时保持安静；完成后小量核验并报告，再ack。旧会话的其他实验和提醒不在本次处理范围。数据处理、数值验证与训练只在pod执行，本机仅编辑、git同步和查看小报告。
