# 四状态宽度的独立训练随机种子复验 v1

用户于2026-10-22明确选择：等待当前理论早/晚融合完成并复核报告，再复验128/256维四状态模型；动态边更新和学习图继续暂缓。此文件在查看融合终态结果前固定下一轮配方，不依据融合结果挑选或叠加新路由。

## 已有证据与本轮问题

原宽度批次`four_state_width_batch/20261021_133620`（源码`08a8cbc771090814d791b60336053bc94f65cc52`）中，256、128、384、512均为四状态头、α=1、union-Brier λ=0。128并非二分类头；其分类F1近乎持平，概率质量和序列排序改善，是本轮要复验的已有组合。首次256维二分类/四状态同轮对照为`four_state_batch/20261019_124645`（`4da48ee`）。未找到已完成核验的128维原二分类头记录，所以当前不是完整同轮2×2。本轮不补二分类格子。

本轮问题：相同数据、验证与训练配方下，换一组训练随机种子后，128相对256的F1、校准和排序变化能否复现？原种子42批次为发现结果，新种子43为首次独立训练随机性复验；不声称只换初始化后训练随机流完全相同，也不承诺单个新种子足以确认稳健性。

## 预注册两臂与固定数据

| 顺序 | 宽度 | 预测头 | 条件权重α | union-Brierλ | 训练base seed | 内部分割base seed |
|---|---:|---|---:|---:|---:|---:|
| 1 对照 | 256 | four_state_mlp | 1 | 0 | 43 | 42 |
| 2 候选 | 128 | four_state_mlp | 1 | 0 | 43 | 42 |

仅两臂×五组重复序列级80/20划分，共10次从头训练，不增加384/512、α搜索、二分类臂或融合候选。既有标签`dbond-four-state-reference-20261020`、`dbond-four-state-width-sweep-v1-20261021`及全部旧权重保留。

五组外部分割ID保持1222/2252/3514/6072/9075；每组实际训练seed继续使用`_stable_fold_seed(43, fold_id)`，实际内部验证split seed固定为`_stable_fold_seed(42, fold_id)`。必须在pod预检实际核对新/旧每组训练seed不同，并确认训练/验证CSV SHA与原成功批次、旧宽度批次逐一相同。不能只改experiment.seed，让验证集也跟着变化。

为此只增加可选的显式内部验证split seed配置；未指定时维持旧行为与随机流。新的base-seed字段由make_fold_config按fold解析为实际split seed，训练初始化仍用新seed。split选择使用局部随机生成器，不消耗模型全局训练RNG。非法seed类型明确拒绝，不因0的真假值而回退。

配置以原对应128/256宽度配置为参考，只允许训练seed、显式固定旧内部分割seed和输出/缓存身份改变。固定3GCN+2GAT、4头拼接、原位置编码、global、FiLM、15维晚融合、nocharge、四状态soft CE。保持输入预合成mask、group_uniform折叠训练、完整原始谱图内部验证、strict max内部有效键micro-F1选checkpoint、固定0.5阈值。

优化器、学习率、warmup、cosine、weight decay、dropout、batch128、max300和patience25全部继承原配置，不因候选表现另行调整。两宽度参数数目不同，模型构造会消耗不同随机数，不能声称逐元素共享初始化或完全相同batch顺序；实际早停训练量也可能不同。

## 执行、预检和来源绑定

前置条件：理论融合对应launch/terminal根目录和源码一致、terminal completed、review passed、结果已报告并ack，且队列该阶段reviewed。失败/来源核验不通过则停止，不自动跳过或重跑。一次仅一轮预检或训练；当前理论融合未终态时，不在pod pull、更改其源码或开始复验预检。

本机可先编辑代码配置和git push；融合复核报告完成后才在pod fast-forward pull、检查空闲GPU并执行新复验smoke。数值验证、数据处理、绘图、训练全部只在pod。没有空闲资源时等待后续一次检查，不抢占其他任务。禁止恢复任何旧权重作为起点。

pod预检至少包含：旧默认split行为保留、显式旧split seed在新训练seed下保持五组历史train/val SHA、两臂train/val匹配、未读取外部测试用于验证或选参、配置规范化只出现允许差异、每臂真实小批AMP更新、有限梯度和loss、逐层形状/global/理论路由/padding投毒不变、严格checkpoint恢复与模式/维度防混用。对新seed解耦使用人工小序列参考验证划分和模型RNG，避免仅复述实现；所有验证只在pod执行。

完成目录`/mnt/pvc/graphtrans/four_state_width_replication_ops/`，批次前缀`checkpoints/graph_transform/four_state_width_replication_batch/`。launch绑定实际源码commit、两配置SHA、旧宽度发现批次、融合前序回执、train/val/target来源及smoke SHA；终态使用独立launch.json/terminal.json。只有全部10次训练和自动原始指标/来源核验通过，才写completed。

## 报告与解释

全报两臂去padding有效键micro-F1/accuracy/AUC、谱图平均P和R及其调和F1、q-Brier和q_seq；列均值/样本SD、128−256配对差均值/SD、逐划分胜平负、参数、训练时间及设备。q-Brier下降为改善。以实际runtime config/seed记录核查，而非仅依赖模板声明。全部best_model.pt严格载入核对维度、模式、参数数目、有限值，保存权重路径/SHA、源码和配置恢复索引。

沿用分类综合提升门槛：平均ΔF1≥0.002、F1至少4/5严格改善、ΔAUC≥0、Δq-Brier≤0；全部通过仅列待进一步复验候选。另完整报告AUC/Brier/q_seq方向是否复现，不因F1门槛失败就掩盖校准排序收益，也不因排序改善就宣称分类F1改善。不基于外部测试调epoch、阈值、seed或选唯一最佳模型。

新seed43结果单独报告，再与旧seed42的同轮128−256差方向并列；不将两次重叠划分结果当10个独立数据样本，不做跨分割权重混合，也不据均值变化推断统计显著。两次初始化仍不足以彻底区分优化随机性、数据分布及模型容量作用。生成Markdown与HTML/SVG，pod独立复核后提醒并ack。

本轮完成后停止自动接续；动态边和学习图仍deferred_by_user，待后续明确授权。保留最早成功模型及新旧候选全部恢复记录，不替换冻结论文主表。
