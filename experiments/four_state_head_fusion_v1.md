# 四状态预测的末端融合方式：预注册 v1

2026-10-22用户明确授权：当前动态边stage3完成、独立核验和报告后，比较交叉注意力、门控融合、残差深MLP；随后做节点编码多尺度Conv1D。此文件只冻结stage6设计和执行要求，尚未实现、预检或提交本轮训练。数值、数据处理、绘图、训练仅pod。当前动态边训练源码不得pull或修改。

## 现状和历史证据

当前六路有效键输入按顺序为[h_i,h_j,e0_ij,h_i−h_j,h_i*h_j,Linear(t15)]，各256维，拼接1536；原四状态头为Linear1536→256、ReLU、Dropout0.1、Linear256→4。四状态顺序none/b_only/y_only/both，q=1−P(none)。此处没有显式来源注意力或动态来源门控；MLP仍能学习输入相关的非线性交互，上游GAT注意力不等于末端六路融合注意力。

历史交叉头已经完成，不得称首次：84520e67e211877dafe2d528d8110bcf4447beca，cross_fusion_batch/20261019_044452，台账§15.36–15.37，二分类concat→1对二分类cross_attention→1；F1−0.082104个百分点、AUC−0.002566，未通过门槛。彼时内部验证仍使用折叠组，早于完整原谱验证修复。本轮是在四状态监督与完整原谱验证下重新检验，旧负结果和局限须在报告并列说明，不能把新旧差异单独归因于头或验证修复。

## 四臂及固定实现

| 臂 | bond_fusion_mode | 明确计算 |
|---|---|---|
| 原版同轮对照 | concat | 原六路拼接→1536→256→ReLU→Dropout0.1→4 |
| 交叉注意力后接MLP | cross_attention | Q为有序左右端点2×256；K/V为e0、差、乘积、理论投影4×256，加既有来源类型向量；4头×64，注意力Dropout0.1；端点残差+LN→有序展开512→Linear256→ReLU→Dropout0.1→Linear4 |
| 来源门控后接MLP | gated | x=原1536拼接；a=softmax(Linear1536→6(x))；g=6a，各256维来源乘对应标量g，再按原顺序拼接→原四状态MLP |
| 残差深MLP | residual_mlp | z=ReLU(Linear1536→256(x))；z'=z+Linear128→256(ReLU(Linear256→128(LN(z))))；保留原Dropout0.1→Linear256→4 |

只试这三候选，不同时搜索头数、门控粒度、层数或瓶颈。门控Linear权重/偏置零初始化，g初值为1；系数6来自六组均匀权重归一化，不是额外调参常数。门控是六个每键标量、不是256维逐通道门；六路保留有序槽位，不加权求和丢掉左右身份。残差块末层weight/bias为0，无额外dropout或手工残差缩放；初始与原函数一致。一个两层残差块令头部线性层由2层增为4层。cross沿用历史局部2Q/4KV定义，不将1个K/V的退化注意力或6路自注意力误写成交叉注意力。

所有注意力/门控局限于一条有效键内部；不跨键或样本。新模块创建隔离RNG，保证共同骨干初始化与后续RNG不被模块构造扰动；concat完全兼容冻结参考。gated/residual保留原MLP共享权重并验证初始完整函数等价；cross只要求骨干共有参数/RNG一致，不承诺整个头初始函数等价。候选参数量可能不同，预检与报告全列；本轮检验完整设计实用收益，不能独立排除额外容量或参数化的贡献，不自动扩增等参搜索。

## 不可改变的基座和监督

四臂均固定H256、4头、3GCN+2GAT、原late理论、FiLM/global、原位置编码/固定图、edge_state_mode=static、nocharge，原节点编码且无Conv1D。**不得自动叠加stage3动态边赢家、早融合或128维。**

bond_head_type始终four_state_mlp，新增独立bond_fusion_mode；不得直接切换为旧cross_attention_mlp而失去四状态目标、manifest或训练路径。四状态α=1、union-Brier λ=0、状态顺序、union logit变换、折叠group_uniform、完整原谱内部验证、strict max内部有效键F1选checkpoint、固定0.5及去padding保持。

五组重复序列级80/20划分1222/2252/3514/6072/9075，各一次初始化；训练base42、内部splitbase42，实际train/val/source SHA须与原成功/旧宽度凭据一致。原AdamW、LR/WD1e-4、warmup40、batch128、max300、早停patience25/min_delta.001及其余配方固定。每臂从头训练，共20次；新的同轮concat为对照，历史最高分不能替代。

## 实现与预检要求

计划模块models/bond_fusion_heads.py；脚本scripts/four_state_head_fusion_ops.py、test_four_state_head_fusion.py；配置pre_synthesis_four_state_head_fusion_{concat,cross_attention,gated,residual_mlp}.yaml。新mode默认concat无新增状态；旧concat、旧binary cross及四状态checkpoint兼容，新增融合mode/来源顺序/attention头数持久记录并在任何strict=False加载前拒绝错配。检查Point-of-use saved config，而不只检查配置文件名。

前序stage3成功终态、独立复核、报告和ack后，才允许pod部署、预检。实现可本地提前准备，但当前pod不pull。静态审阅后commit/push，再pod fast-forward pull。预检使用真实训练/内部验证数据，禁止外部测试调设计：冻结concat的参数、schema、RNG、CPU前向/梯度回归；共享骨干初始化；四态监督实际启用与union_q一致；六路顺序、每路梯度、局部键隔离、手算注意力/门控参考、gated/residual零初始等价、padding/短链/空有效键；跨mode/来源/输出状态加载拒绝；真实AMP至少两步和全部三候选最大train128/val512显存；全部历史划分SHA匹配。全空图沿用旧GAT限制，不扩张本轮改造范围。

独立ops /mnt/pvc/graphtrans/four_state_head_fusion_ops，batch前缀checkpoints/graph_transform/four_state_head_fusion_batch。保存launch/terminal、源码/配置/目标SHA、完整smoke。只用空闲GPU，arm串行，沿用现有五划分调度；不抢占。预检失败、训练失败、来源失败则halt并报告，不自动重跑；无空闲GPU留到下次单次尝试。提交后源码冻结直到终态。

## 完整报告与后续

全报4臂和各候选相对同轮concat的有效键micro-F1、bondacc、AUC、谱图平均P/R及其调和F1、q-Brier、q_seq，均值/样本SD、配对差/SD、逐划分胜平负、参数/耗时/预检显存。四项工程门槛：ΔF1≥.002、至少4/5F1严格改善、ΔAUC≥0、ΔBrier≤0；全报候选而非只展示赢家。重叠划分和单次初始化不当作独立统计样本，门槛不等于显著性；不按外测选epoch、阈值、配置或seed。

报告在pod生成结构变化示意、HTML看板、SVG指标和逐划分配对波动图；分别显示均值/SD和配对差SD，注明非置信区间。固定内部验证前16行提取cross逐来源注意力、gated六路权重和residual修正范数，仅描述性检查，无标签驱动选择；attention/gate权重不直接等于因果特征重要性。

本轮20份权重逐份strict CPU恢复，核验真实模式/α/种子/来源，保存checkpoint路径SHA、源码commit及永久tag、configSHA、恢复registry。旧成功4da48ee及全部候选不删除。另存独立heartbeat_review.json，准备简短中文报告再ack准确事件并标stage6 reviewed；按新授权仅接续stage7 Conv1D，不据本轮外测赢家更换stage7基座，不更改冻结论文主表。学习图stage4继续暂缓。

终态事件处理与接续是两个可恢复步骤：若前序已reviewed/ack、本项在allowlist且未提交，即使本次scan无pending，也应从索引恢复实现/部署/预检/提交；已完成且来源绑定的smoke不得重复运行，提交成功须立即登记root/gitrev。无新终态且无已reviewed前序待接续时安静结束。
