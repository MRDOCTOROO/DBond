# 固定拓扑上的动态边状态实验 v1

2026-10-22用户重新授权：以256维四状态头、α=1为基准，恢复此前暂缓的动态边更新，与原静态边做同轮对照，在pod训练并通过终态事件报告。本次只恢复stage3；stage4学习图继续暂缓，完成后不自动接续其他实验。当前没有该实验的效果结果。

## 固定基准与三臂

沿用原成功参考4da48ee及其永久标签；本轮所有权重均从头训练，重新训练同轮静态对照，不以历史最高测试分数替代对照。静态实现对固定源码967bd43做CPU函数、参数/RNG和checkpoint兼容预检。

| 臂 | edge_state_mode | 每层传播前的边状态 |
|---|---|---|
| S0原静态对照 | static | 五层均使用原EdgeEncoder输出e0 |
| S1固定上下文容量对照 | fixed_context | 共享更新器递归更新e，但端点节点始终取h0 |
| D动态上下文候选 | dynamic_context | 同一更新器递归更新e，端点节点取当前层输入h_l |

三臂分别五组重复序列级80/20划分，共15次正式训练。宽度256、4头、3GCN+2GAT、FiLM/global、原位置编码、原15维理论晚融合、four_state_mlp、CE4 conditional α=1、union-Brier λ=0、nocharge固定。训练base seed42，显式内部验证split base42，按fold_id稳定派生；原1222/2252/3514/6072/9075实际train/val/source SHA须与原成功和旧宽度凭据一致。原实验max300/patience25、AdamW、LR1e-4、WD1e-4、warmup40、batch128、dropout0.1及其余预算不变。

保持group_uniform折叠训练、完整原始谱图内部验证、strict max内部有效键micro-F1选checkpoint、固定0.5阈值和去padding指标。外部测试不参与epoch/阈值/种子或候选选择。不叠加早融合、128宽度、新特征或学习连接图。

## 更新机制与控制变量

h0定义为FiLM之后、第一层GCN之前的节点状态。所有原有有向边（包括残基—全局节点双向边）均更新，边集合、顺序、距离桶、主链映射和全局节点位置不变。当前历史位置图的等效连接语义保持，不在本轮调整正弦启发式或建图半径。

每个GCN/GAT层前更新一次，共五次，所有层共享一个更新器：

`u = concat(h_src, h_dst, e)`；`delta_e = Linear(H/2→H)(SiLU(Linear(3H→H/2)(LayerNorm(3H)(u))))`；`e_next = e + delta_e`。

H=256，瓶颈H/2=128。末层权重与偏置零初始化，初始传播函数与静态对照一致；新增模块在原参数初始化完成后创建，并隔离初始化RNG。无额外dropout、常数缩放系数或逐层独立更新器。fixed/dynamic的可训练参数、初始化、更新次数和运算形式一致，区别仅是h0还是h_l；不detach h0，保留端点梯度。静态模式无新增参数/持久buffer，保留旧checkpoint schema。

最终逐键预测头三臂始终使用原始相邻边表示e0，不使用最后层e_last；因此本轮检验边更新经节点传播的作用。fixed/dynamic持久保存模式标记，直接load_state_dict和CheckpointManager在strict=False时也须在复制参数前拒绝跨模式、缺更新器、形状不符或保存配置矛盾的checkpoint。

## 预检和提交

所有计算、数值验证、数据处理、绘图及训练只在pod。当前已授权前序宽度、融合、种子复验均completed/reviewed/ack；本轮ops检查种子复验终态及报告绑定、队列stage5 reviewed和stage3授权。先本机静态审阅、git push，再pod fast-forward pull；不覆盖无关ensemble_baseline_preds.py修改。

预检至少核验：static对967bd43参数/RNG/CPU训练forward/gradient及checkpoint兼容；三臂共享参数/RNG一致、零更新器初始函数等价；独立更新公式、第一步fixed/dynamic一致；后层端点扰动仅dynamic改变相应更新；五次更新、拓扑不变、head e0未串位；global、混合L2/L1/L0无跨图、短序列/空键、新更新器空边输入、padding污染不影响有效键（原模型全空边图不在本轮支持范围，不改旧GAT）；每臂实际小批AMP更新及零末层两步梯度；完整train128/val512；去padding指标；三臂实际train/val/source SHA与历史绑定。只使用训练/内部验证样本做预检，不使用外部测试选择设计。

成功后使用独立`/mnt/pvc/graphtrans/four_state_dynamic_edge_ops/launch.json`与`terminal.json`，批次前缀`checkpoints/graph_transform/four_state_dynamic_edge_batch/`。每次仅运行本轮训练或预检。只用提交时空闲GPU，不抢占；资源不足留待下次单次尝试。按现有arm串行与每卡单划分调度，模型与调度因素不同时更改。smoke失败、训练失败或来源核验失败则报告并halt，不自行重跑或跳过。提交后冻结源码，终态前不pull，训练期间仅扫描完成/失败事件。

## 报告与归因

完整报告S1−S0、D−S0、D−S1的有效键F1/accuracy/AUC、谱图平均P/R及其调和F1、q-Brier、q_seq；全报均值/样本SD、逐划分配对差/SD、胜平负、参数量、实际墙钟与预检峰值显存。每份权重严格恢复，核验实际配置、训练/划分seed、模式标记、损失和来源SHA。

沿用工程门槛：平均ΔF1≥0.002（0.2个百分点）、F1至少4/5严格改善、ΔAUC≥0、Δq-Brier≤0。S1和D各自相对S0报告全部四门槛；D−S1也报告同样门槛供机制判断。D−S0衡量整个新增动态边模块的实用收益；D−S1才提供当前节点反馈相对固定h0的额外证据。D胜S0但不胜S1时不能宣称动态反馈得到验证。门槛不代表统计显著，五个划分重叠、单次初始化及多轮探索限制结论，不更新冻结论文主表或自动宣布唯一最优。

完成后额外在每份checkpoint的内部验证前16行做CPU eval诊断，开启默认关闭的edge诊断选项，保存五层每边e_l−e0/单步变化L2均值、边状态范数、GCN/GAT实际sigmoid message gate在≤0.01或≥0.99的比例。阈值仅用于预先固定的饱和描述，不是物理常数；该小样本诊断不是测试集性能、不用于选checkpoint/候选，不代表完整分布。边变化小或gate饱和只提供实现行为线索，不能单独证明机制或真实三维/化学断裂。

保存源码commit与永久tag、三份配置SHA、15份权重路径/SHA、结果报告、HTML/SVG及恢复索引。完成后pod独立复核、提醒当前会话、ack准确终态key并标stage3 reviewed；队列停止，学习图仍deferred_by_user。原成功4da48ee、宽度/融合/种子复验全部候选继续保留。
