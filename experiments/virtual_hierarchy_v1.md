# 单尺度 k=5 基座上的分层虚拟节点：预注册 v1

用户于 2026-10-24 授权在目前证据最强的模型上检验“多个虚拟节点分组聚合，再由上层节点汇总”。本轮只考察固定分组图结构，不更改主任务标签、理论特征、优化器、训练预算或评估口径。所有训练、数据处理、数值核验和绘图都在 pod；本地仅编辑代码、同步与阅读小报告。提交后只等完成或失败事件，不轮询 epoch。

## 对照与三臂

三臂都使用完整残基 NodeEncoder→单尺度 Conv1D 核 `k=5` 残差→FiLM→3 层 GCN＋2 层四头 GAT→末端 15 维理论特征＋六路 concat→四状态预测头。H=256，四状态 α=1、Brier 权重0，原晚融合、静态边、nocharge、group_uniform。这里“分组长度5”和“卷积核 k=5”是两个独立含义。

| 模式 | 有效节点索引 | 双向虚拟连接 |
|---|---|---|
| `single`（同轮对照） | 残基 `0…L−1`，顶部 `L` | 每个残基↔顶部 |
| `grouped_skip` | 残基 `0…L−1`，组 `L…L+G−1`，顶部 `L+G` | 残基↔所属组、各组↔顶部，并保留残基↔顶部直连 |
| `grouped_strict` | 同上 | 残基↔所属组、各组↔顶部；无残基↔顶部直连 |

`G=ceil(L/5)`，第 `i` 个残基分配到 `floor(i/5)` 组，尾组可少于5个。组初始特征为该组有效残基在 k5 卷积之后的**无参数均值**；顶部沿用原单全局节点的初始编码。组和顶部都参加同一 FiLM、GCN/GAT 消息传递；全部虚拟边复用原 `global` 类型编号4、距离0，边类型嵌入不扩容，bond 预测仍只读原相邻残基及原相邻边。三臂可训练参数数目相同，但节点/边数量和通信路径不同。`grouped_skip` 保留单节点捷径，`grouped_strict` 用纯层级路径；3GCN＋2GAT 足以容纳残基→组→顶部→另一组→残基的4跳传播。固定组划分只是序列位置规则，不是假定真实三维接触或折叠。

## 训练与来源

同轮 `single→grouped_skip→grouped_strict`，每臂五组，共15次从头训练。训练 base seed **44** 是本轮预定的新初始化；内部 train/val split base seed 固定 **42**，并保持原五组序列级折叠 80/20 划分 `1222/2252/3514/6072/9075` 的 train/val/source SHA、四状态目标 manifest SHA 和完整原谱内部验证。按内部有效键 F1 的严格最大值选 checkpoint；外部测试绝不用于选 epoch、阈值、组长或 seed。阈值固定0.5，所有指标去 padding。原 seed42/43 的 k5 结果只作历史参照，不和本轮按15个独立划分合并、不替代本轮 `single`。

## pod 预检（全部通过才提交）

1. `single` 与既有 k5 模型的初始化 state、RNG、CPU 前向/反向和旧 checkpoint 严格恢复兼容。三臂可训练参数都必须是 2,002,284；候选仅增加持久模式、schema 和分组长度标记，不增加学习参数。跨模式 checkpoint 加载即使 `strict=False` 也必须在复制旧权重前拒绝。
2. 对 `L=0/1/4/5/6/11` 和混长批，独立构建应有的残基/组/顶部节点与虚拟边集合，核对每个真实有效残基恰好归属一组，组节点均值、顶部位置、type4/dist0、正反双向边、`node_len=L+ceil(L/5)+1`、批次 offset、组间与肽链间不串边。`single` 仍为 `L+1`；所有模式相邻原肽键边与 `bond_edge_map` 保持一致。短链没有有效键时不得让虚拟节点产生伪键。
3. 验证图缓存隔离：同一序列长度的三模式以及组长不同设置不复用旧图；缓存元数据必须记录模式与组长；固定内部验证16行的 cached/uncached 图、相邻边映射及节点数一致。直接对组池化输入的无效位置投毒并延长 padding，不得改变有效组均值；在真实短/长混批中对 NodeEncoder 的无效位置投毒、改变另一谱图条件，也不得改变本样本有效输出。不允许用 padding/外测标签初始化组。
4. hook 首层 GCN 输入，独立重算 NodeEncoder＋k5 后每组均值、顶部编码以及 FiLM 后打包顺序。逐模式做真实四状态前向与反向，检查全部有效输出和共享参数梯度有限，并用移除虚拟边的受控前向检查预测确实使用虚拟边；用现有最长真实 train128/val512 批量做 pod CUDA 显存预检。真实五组 train/val/source SHA、目标 manifest SHA 和 seed44/42 也须通过。

实现代码 `graph_transform/scripts/test_four_state_virtual_hierarchy.py` 仅在 pod 运行，并生成 `smoke.json`；`four_state_virtual_hierarchy_ops.py smoke` 将其绑定源码 Git revision、所有 Python 源文件 SHA、三份配置 SHA、目标 SHA，`launch` 拒绝陈旧或已存在的回执。仅用空闲 GPU，`run_architecture_batch.py` 顺序跑三臂，每臂五组。独立 ops：`/mnt/pvc/graphtrans/four_state_virtual_hierarchy_ops`，批次前缀 `checkpoints/graph_transform/four_state_virtual_hierarchy_batch`。`launch.json`/`terminal.json` 作为完成事件凭据；pod 源码在训练期间冻结。

## 结果与门槛

三臂全报，并分别比较 `grouped_skip−single`、`grouped_strict−single`、`grouped_strict−grouped_skip`。八项去 padding 指标：肽键 micro-F1、accuracy、ROC-AUC、谱图平均 P、谱图平均 R、两者调和 F1、q-Brier、q_seq。给出五组均值和样本 SD、同划分配对差及其 SD、逐划分胜平负、参数/耗时/预检显存。工程门槛对每个比较都列示：ΔF1≥0.002、F1至少4/5改善、ΔAUC≥0、Δq-Brier≤0。这是工程筛选，**不是**统计显著性。`grouped_skip` 胜而 `grouped_strict` 不胜时只能支持“保留原捷径的增强”，不能宣称纯层级更好；三臂节点/边数不同，不能单独归因某一种聚合机制。

完成后只在 pod 复核 `launch/terminal/status/manifest`、源码/配置/目标/来源 SHA、真实训练与 split seed、15份 checkpoint 严格 CPU 恢复及模式标记。对每份权重取本划分固定内部验证前16行，独立核验真实虚拟边集合、组均值和 FiLM 前后打包、有限输出；诊断只作实现说明，不选模型。逐行读15份外测预测 CSV 与对应测试源，核对序列/条件/标签/权重来源/有效键数；由 `L−1` 显式重建去 padding 掩码，独立重算 F1、accuracy、AUC、谱图平均 P/R、q-Brier、q_seq，并记录预测概率六位小数舍入造成的容差和阈值附近歧义键数。该复算逻辑须先在既有完成批次的一折预测上通过 pod smoke。再复算原始 CSV 均值/SD/配对差/胜率，在 pod 保存 `result.json`、`report.md`、`checkpoint_registry.json`、独立 `heartbeat_review.json`、HTML/SVG 图与全部路径/SHA。保留当前 k5 seed42/43、原 4da48ee 和全部候选，不自动更改论文主表或主配置。

完成事件只处理本对话登记的 ops 且 `launch.root` 完全匹配的新 pending；有成功结果才在 pod 审计和提醒，再 ACK 准确 key，标记 stage9 reviewed。失败或来源不通过则报告并 halt，不重跑、不跳过、不启动其他实验。pending 为空立即安静结束，不读取训练进度或 epoch 日志。
