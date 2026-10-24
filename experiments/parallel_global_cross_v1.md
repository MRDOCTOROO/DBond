# 单尺度 k=5 基座上的四并行全局节点与逐肽键读取：预注册 v1

用户在层级虚拟节点实验结束后授权下一轮尝试。本轮独立检验“单个全局摘要是否容量不足”和“逐肽键能否按需读取多个摘要”；不加入化学关系先验、不改局部卷积或主任务。前轮 stage9 的 15 次训练完成，但预测导出审计失败，终态为 `failed` 且已报告/ack；本轮不以其外测结果选架构，不改写其终态或权重。训练、数据处理、数值核验和绘图只在 pod；本地只编辑代码/配置、同步和看小报告。只使用完成/失败事件，不轮询 epoch。

## 固定基座与三臂

三臂固定 H256、四状态 `none/b_only/y_only/both`、α=1/Brier=0、原六路 concat 四状态头、晚融合15维理论特征、静态序列图、nocharge、NodeEncoder→单尺度残差 Conv1D `k=5`→FiLM→3 GCN＋2 GAT（4头）。仅改虚拟节点及其读取方式：

| 模式 | 图中虚拟节点 | 逐键读取 |
|---|---|---|
| `single` | 原单全局节点，全部有效残基与其双向连接 | 原预测头；与已有 k5 基座同架构 |
| `parallel4` | 四个并行全局槽，每个有效残基与四槽全部双向连接 | 原预测头；四槽仅通过图传播影响残基 |
| `parallel4_cross` | 与 `parallel4` 完全相同的四槽图 | 肽键端点表示形成 Query、四槽最终表示形成 Key/Value；4头注意力上下文经零初始输出投影残差接入原四状态头 |

四槽不分区、不增加上层汇总节点、槽间不连边；虚拟边沿用原 `global` 类型4、距离0。槽0延用原节点初始化，其余槽使用**不同可学习偏置**，避免相同初值＋相同邻居导致对称塌缩。`parallel4_cross` 保持 `bond_fusion_mode=concat` 与四状态监督，不套用旧“六路局部特征交叉注意力头”；其 Query/Key/Value是“肽键→全局槽”独立模块。四槽与交叉注意力增加参数和边数；`parallel4−single` 不能单独归因容量，`parallel4_cross−parallel4` 同时改变读取路径和参数量，均需如实报告。注意力权重、槽间余弦相似度和有效秩只能作为描述性诊断，不是因果解释。

## 同轮训练与模型选择

顺序 `single→parallel4→parallel4_cross`，各五个序列级划分 `1222/2252/3514/6072/9075`，合计15次**从头训练**。训练 base seed **45**，内部 train/val 划分 base seed **42**，使用 `_stable_fold_seed` 派生每组实际值；相同划分的 train/val/source SHA 必须与 stage8 完成批次以及历史来源凭据吻合。折叠训练 `group_uniform`、完整原谱内部验证、按内部有效键 F1 严格最大值选 checkpoint。外测只用于最终报告，固定阈值0.5，全部指标去 padding；不依外测选 epoch、阈值、槽数、种子或叠加其他实验赢家。stage8 seed43 的 k5 结果仅作历史参照，主比较必须使用本轮 seed45 的 `single`。

训练前先在 pod 预检，并且所有检查通过才有独立 `launch.json`：

1. `single` 相对 stage8 k5 的参数键、初始化 RNG、CPU 前向/反向及旧 checkpoint 严格恢复兼容。新两臂专属语义 marker 必须使错模式 checkpoint 在任何公共参数复制前被拒绝，即使 `strict=False`；新增参数初始化不得移动共享基座 RNG。完整四状态 CE/union-q 路由、有效键 mask 和理论晚融合不变。
2. 对 `L=0/1/4/5/6/11/32` 和混长批，独立构建节点数 `L+1` 或 `L+4`、双向虚拟边集合、原相邻键边映射及全局 type4/dist0；验证无槽间边、无跨肽链边、无 padding 节点、缓存按模式隔离。首层 GCN hook 验证 k5 残基编码、四槽不同初值、FiLM 调制及打包顺序。更改/投毒其他肽或无效位置不影响当前肽的有效输出。
3. 用实际四状态损失做小批及最大真实 train128 的前反向、内部 val512 的前向和显存预检。证实四槽偏置收到非零任务梯度，cross 模块 Query/Key/Value/输出投影在两步更新后收到梯度，且改动或切除全局边会影响输出；固定内部验证前16行启用诊断验证四槽表征、逐键注意力形状、有限值和非恒定读取。
4. 检查五组历史 train/val/source/四状态 target SHA、训练45与划分42独立、配置仅改变臂模式及产物路径；新 `export_from_evaluate_cache=true` 使正式测试指标和逐谱图 CSV 复用**同一次前向的原始 logits**，输出明确 sigmoid 且概率精度足以 float32 往返。用旧完成批次一折在 pod 验证同 pass 的 CSV 去 padding 重算能与正式八项指标吻合，杜绝 stage9 `_sigmoid_if_needed` 按单行数值范围误判 logits 的旧导出缺陷。此预检只读旧权重、不重新训练或选模型。

仅占用空闲 GPU，`run_architecture_batch.py` 顺序跑三臂，各臂五划分并行；独立 ops `/mnt/pvc/graphtrans/four_state_parallel_global_ops`，批次前缀 `checkpoints/graph_transform/four_state_parallel_global_batch`。`smoke.json` 绑定源码 Git revision、全部生产 Python SHA、三配置 SHA、目标及旧来源 SHA；`launch.claim/launch.json/terminal.json` 保证单次提交。允许 pod 既有未提交的无关 `ensemble_baseline_preds.py`，但须在 launch 与审计中**显式记录其 live SHA 和 Git blob SHA**，证明它不被训练/评估导入；除该已知文件外生产源码必须与 Git 和 preflight 相符。训练期间不 pod pull/改源文件。

## 报告、工程门槛与恢复

全报三臂八项去 padding 指标：键级 micro-F1、accuracy、AUC、谱图平均 P/R 与其调和 F1、q-Brier、q_seq。分别比较 `parallel4−single`、`parallel4_cross−single`、`parallel4_cross−parallel4`：每臂五组均值和样本 SD、配对差及其 SD、每组胜平负，参数/墙钟/显存。对每一对照列示预定工程门槛：ΔF1≥0.002、F1至少4/5改善、ΔAUC≥0、Δq-Brier≤0。门槛不是显著性检验；五划分重叠且每臂每划分仅一次初始化。

完成后只在 pod 检查 `launch/terminal/status/manifest`、源码/配置/目标与 train-val-test SHA、15份验证选中权重严格 CPU 恢复、实际训练/划分种子和模式 marker。逐行对照同 pass 保存的预测 CSV 与对应外测源，按 `L−1` 显式掩码独立重算八项指标；不能将导出精度差或排序波动用宽泛容差默许。如重算与正式指标不符，终态报审计失败，不宣布性能改进。每份权重在固定内部验证前16行核验四槽打包、全局边集合、FiLM前后和交叉注意力诊断；不得用诊断选模型。保存 `result.json/report.md/review_receipt.json/checkpoint_registry.json`、HTML/SVG及每份权重路径/SHA、源码 commit/tag/配置 SHA，原成功 `4da48ee` 和全部旧权重保留。只提醒本对话新完成/失败事件，报完才 ACK 精确 key；无新 pending 安静结束，失败不自动重跑。stage4 学习图继续暂缓，不自动接下一轮或提升主模型。
