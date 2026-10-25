# Stage 12：三分支门控融合（开发性评估）

本阶段在历史单全局节点、Conv1D k=5、H=256、3 GCN + 2 GAT（4 heads）、FiLM、late theory、static edge、四状态头上独立比较五臂。旧 `bond_fusion_mode=gated` 是六个槽位各自缩放再拼接的历史试验，不代表本阶段三分支加权求和；本阶段只用新增 `stage12_fusion_mode`，默认 `legacy` 不增加权重键或改变旧前向。

| 臂 | 配置值 | 头部可训练参数 | 说明 |
|---|---|---:|---|
| S12-A | `legacy` | 394,500 | 原 6H 拼接 → H → 4 |
| S12-B | `wide_mlp` | 397,582 | 原拼接头隐藏宽度 258 |
| S12-C | `equal` | 395,012 | 三分支投影到 H 后等权平均 |
| S12-D | `softmax_gate` | 397,319 | 三分支投影到 H 后动态 Softmax 加权 |
| S12-E | `softmax_gate_residual` | 403,463 | D 加原始 6H 拼接直通到四状态 logits |

B 比 D 多 263 参数（以 D 为基准 0.0662%）；E 比 D 多 6,144 参数（1.546%，B 比 E 少 1.458%）。因此 D 对 B 的比较较好地控制了头部容量；E 的增益可能包含额外容量和残差作用，不能单独归因门控。总模型参数量由 pod 预检及每份权重恢复再次记录。

三分支固定为：`[left,right,left-right,left*right]`（4H）、原边表示（H）、由序列与 charge/NCE 可在合成前确定的理论离子特征投影（H）。每支 `Linear→ReLU` 到 H；C/D/E 的末层都是 `H→4`，四状态顺序与 `union_logit` 不变。D/E 的 `3H→3` 门控零初始化，初始权重各 1/3。E 的原拼接 `6H→4` 残差零初始化，初始预测与 D 相同。填充键先由原 `valid_bond_mask` 排除；头部的显式三列 source mask 将缺失分支同时从融合、门控输入和 E 残差中排除，且至少保留一个分支。

五组外部序列划分固定为 `1222/2252/3514/6072/9075`。第一轮五臂均从头训练，训练 base seed=42，内部拆分 base seed=42；实际逐折 seed 按仓库 `_stable_fold_seed` 固定。原训练 batch、optimizer、调度、300 epoch 上限、验证/早停、内部 `max_f1` 选权重、0.5 阈值、四状态 alpha=1 与 Brier 权重 0 均相同。所有完成折保存原始逐谱图预测与同轮评估指标；预测须为正式评估同一次前向导出的 float32 九位有效数字，不用外测选权重。历史均值只作背景，同轮 A 是每个候选的唯一配对基线。五组外测被反复探索，本轮明确标为开发性评估。

pod 中先在全新 Stage 12 ops 命名空间执行 `stage12_ops.py preflight --ops <独立ops> --gpu <空闲GPU>`；预检将核对五配置和五组来源/序列拆分，运行真实训练行上的前反向、概率映射、门控梯度/归一、source mask 与 padding 隔离、参数量及 strict checkpoint 恢复。仅当 `smoke.json.status=passed` 且源码快照未变，主调度才可调用 `stage12_ops.py launch --ops <独立ops> --root <全新checkpoint根> --gpus <获准GPU列表> --source-commit <冻结SHA>`。launch 单次占有 claim，worker 顺序跑五臂；每臂五折可在获准空闲 GPU 上并行。主调度只监听独立 `terminal.json` 完成/失败事件，不读取 epoch 进度；不得触碰历史 Stage 11 ops、其他任务进程或结果。一个候选失败时 worker 继续其他独立候选并如实记录失败。

终态后在 pod 调 `stage12_ops.py finalize --ops <独立ops>`。它只读训练产物，逐折核查来源/拆分 SHA、四状态目标 manifest、实际随机种子、内部选权重记录、唯一 checkpoint、严格 CPU 恢复、same-pass 预测与 evaluation_id/checkpoint 绑定；输出 `audit_manifest.json` 与 `checkpoint_registry.json`，均不覆盖既有文件。独立审计程序再从原始预测去 padding 重算 Bond-F1/Acc、ROC-AUC、PR-AUC、MCC、谱图 P/R/F1、q-Brier、ECE、q_seq、Top10、Top20，并生成五折逐项配对结果；审计未通过不得标记 `audited`。

第一轮工程筛选：候选相对同轮 A 的 Bond-F1 均值至少 +0.20 个百分点，至少 4/5 划分改善，ROC-AUC 下降不超过 0.001，q-Brier 恶化不超过 0.001。所有未过门槛和失败均照实呈现。每 Stage 最多一名候选用训练 seed43 与 seed45 复验，两个复验轮都需各自重新训练对应 A，并维持内部拆分 seed42；不能将变化的训练种子称为新外测集。任何组合实验须等独立 Stage 12–14 审计完成后另行设计。

源码快照来自原项目 `3f4ca4d8ed0358bb3aa8b6820cb2e022ad1700dd`，Stage 12 本地独立库另有自己的 commit SHA。部署时将冻结源码快照传至 pod，不要求 pod 对历史仓库 pull；产物的本地同步和总报告由主调度管理。
