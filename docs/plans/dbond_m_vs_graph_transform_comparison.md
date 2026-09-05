# dbond_m vs graph_transform 对比分析报告

## 一、评估指标对比

### 1.1 dbond_m 的评估指标（arc_dbond/evaluate.dbond_m.py + multi_label_metrics.py）

dbond_m 使用 **多标签分类** 口径，对每个样本（肽段）输出 `max_len-1` 个键的断裂/未断裂标签，阈值 0.5 对 sigmoid 输出二值化：

| 指标类别 | 指标名 | 说明 |
|---------|--------|------|
| **Example-based** | `subset_acc` | 样本子集准确率（所有标签完全匹配才算正确） |
| | `ex_acc` | 样本级 accuracy（Jaccard 系数平均） |
| | `ex_precision` | 样本级精确率 |
| | `ex_recall` | 样本级召回率 |
| | `ex_f1` | 样本级 F1 |
| **Label-based** | `lab_acc_ma` | 标签级 accuracy（macro 平均） |
| | `lab_acc_mi` | 标签级 accuracy（micro 平均） |
| | `lab_precision_ma` | 标签级精确率（macro） |
| | `lab_precision_mi` | 标签级精确率（micro） |
| | `lab_recall_ma` | 标签级召回率（macro） |
| | `lab_recall_mi` | 标签级召回率（micro） |
| | `lab_f1_ma` | 标签级 F1（macro） |
| | `lab_f1_mi` | 标签级 F1（micro） |
| **Loss** | `Loss` | 平均损失 |

### 1.2 graph_transform 的评估指标（graph_transform/evaluation/metrics.py）

graph_transform 的 `BinaryBondMetrics` 同时计算了 **dbond_m 同口径指标** 和 **额外任务指标**：

| 指标类别 | 指标名 | 与 dbond_m 对齐？ |
|---------|--------|-----------------|
| **dbond_m 同口径** | `subset_acc` | ✅ 一致 |
| | `ex_acc` | ✅ 一致 |
| | `ex_precision` | ✅ 一致 |
| | `ex_recall` | ✅ 一致 |
| | `ex_f1` | ✅ 一致 |
| | `lab_acc_ma` | ✅ 一致 |
| | `lab_acc_mi` | ✅ 一致 |
| | `lab_precision_ma` | ✅ 一致 |
| | `lab_precision_mi` | ✅ 一致 |
| | `lab_recall_ma` | ✅ 一致 |
| | `lab_recall_mi` | ✅ 一致 |
| | `lab_f1_ma` | ✅ 一致 |
| | `lab_f1_mi` | ✅ 一致 |
| **额外指标** | `accuracy` | sklearn accuracy（展平后逐标签） |
| | `precision` | sklearn precision（binary） |
| | `recall` | sklearn recall（binary） |
| | `f1` | sklearn f1（binary） |
| | `auc` | ROC AUC |
| | `hamming_loss` | 汉明损失 |
| | `positive_rate` | 正样本比例 |
| | `pred_positive_rate` | 预测正样本比例 |

### 1.3 指标计算差异分析

**结论：dbond_m 同口径指标已完全对齐。** graph_transform 的 `BinaryBondMetrics` 在 `compute()` 方法中：
- 使用 `sample_predictions` / `sample_targets` 按 **样本行** 构建 `pred_matrix` / `target_matrix`，然后计算 example-based 和 label-based 指标
- 使用 `all_valid_predictions` / `all_valid_targets` 展平后计算 sklearn 指标

**细微差异（不影响对齐）：**
1. **epsilon 值**：dbond_m 用 `1e-8`，graph_transform 也用 `EPSILON = 1e-8` → ✅ 一致
2. **数据类型**：dbond_m 的 `_label_quantity` 用 `astype("float")`（即 float64），graph_transform 用 `astype(np.float64)` → ✅ 一致
3. **Loss 计算**：dbond_m 在 evaluate 中用 `loss_sum / len(dataset)` 计算 mean loss；graph_transform 在 evaluator 中也做类似计算 → ✅ 逻辑一致

---

## 二、超参数对比

### 2.1 模型架构超参数

| 超参数 | dbond_m | graph_transform | 差异说明 |
|--------|---------|-----------------|---------|
| **hidden_dim** | 256 | 256 | ✅ 一致 |
| **num_heads / num_attention_heads** | 4 | 8 | ⚠️ **不一致**：GT 用了 2 倍头数 |
| **dropout** | 0.1 | 0.1 | ✅ 一致 |
| **forward_expansion** | 2 | 无此参数 | ⚠️ GT 无 FFN 扩展比概念（GT 用 GCN+GAT 替代 Transformer FFN） |
| **attention_layer_num** | 1 | 无直接对应 | ⚠️ GT 用 `num_gcn_layers=3` + `num_gat_layers=2` 替代 |
| **alphabet** | `#ABCDEFGHIKLMNOPQRSTVWXYZ` | `#ABCDEFGHIKLMNOPQRSTVWXYZ` | ✅ 一致 |
| **pad_char** | `#` | `#` | ✅ 一致 |
| **max_seq_len** | 32 | 100 | ⚠️ **不一致**：GT 用了更长的序列 |
| **aa_embedding_dim** | 隐含在 hidden_dim | 64 | GT 显式配置 |
| **position_embedding_dim** | 隐含（PositionalEncoding1D） | 32 | GT 显式配置 |
| **physicochemical_dim** | 无 | 32 | GT 新增物理化学特征 |
| **edge_dim** | 无 | 32 | GT 新增边特征 |
| **num_classes** | max_len-1 = 31 | 1（二分类） | ⚠️ **架构差异**：dbond_m 输出 max_len-1 个标签，GT 输出每个键的 1 个概率 |

### 2.2 训练超参数

| 超参数 | dbond_m | graph_transform | 差异说明 |
|--------|---------|-----------------|---------|
| **batch_size** | 4096 | 1024 | ⚠️ **不一致**：GT 用了 1/4 的 batch |
| **epoch** | 50 | 60 | ⚠️ 轻微差异 |
| **seed** | 2024 | 42 | ⚠️ 不同种子 |
| **learning_rate** | 1e-4 | 5e-4 | ⚠️ **不一致**：GT 用了 5 倍学习率 |
| **weight_decay** | 1e-4 | 1e-4 | ✅ 一致 |
| **optimizer** | Adam | AdamW | ⚠️ **不一致**：GT 用 AdamW（多了解耦权重衰减） |
| **loss_type** | CE（multilabel_soft_margin_loss） | BCEWithLogitsLoss | ⚠️ **不一致**：但数学上等价（multilabel_soft_margin = BCE sigmoid） |
| **label_smoothing** | 0.1 | 无 | ⚠️ dbond_m 有 label smoothing，GT 没有 |
| **early_stopping patience** | 5 | 10 | ⚠️ GT 更宽容 |
| **early_stopping delta** | 1e-4 | 0.001 | ⚠️ GT 的 delta 更大 |
| **early_stopping metric** | Loss（越小越好） | 未明确指定 | 需确认 GT 的 early stopping 监控指标 |
| **dataloader_workers** | 4 | 32 | ⚠️ GT 用了更多 workers |
| **scheduler** | 无 | cosine（warmup 10 epochs） | ⚠️ GT 有学习率调度 |
| **gradient_clip_norm** | 无 | 1.0 | GT 新增梯度裁剪 |

### 2.3 数据配置

| 超参数 | dbond_m | graph_transform | 差异说明 |
|--------|---------|-----------------|---------|
| **state_var** | charge, pep_mass, intensity | 同 | ✅ 一致 |
| **env_var** | nce, scan_num | nce, rt | ⚠️ **不一致**：dbond_m 用 scan_num，GT 用 rt |
| **label_col** | true_multi | 同（多标签格式） | ✅ 一致 |
| **数据增强** | 无 | 可选（默认关闭） | GT 新增 |
| **图缓存** | 无 | cache_graphs: true | GT 新增 |

---

## 三、关键差异总结

### 3.1 需要对齐的差异（建议修改）

| # | 差异 | 严重程度 | 建议 |
|---|------|---------|------|
| 1 | **env_var 不一致**：dbond_m 用 `scan_num`，GT 用 `rt` | 🔴 高 | 确认数据集中是否同时有 `scan_num` 和 `rt`，GT 应与 dbond_m 保持一致使用 `scan_num`，或确认 `rt` 是有意的改进 |
| 2 | **num_heads 不一致**：dbond_m=4, GT=8 | 🟡 中 | GT 的 8 头是架构升级，但如果要公平对比，应保持一致或记录差异 |
| 3 | **learning_rate 不一致**：dbond_m=1e-4, GT=5e-4 | 🟡 中 | GT 的学习率更大，可能影响收敛行为 |
| 4 | **batch_size 不一致**：dbond_m=4096, GT=1024 | 🟡 中 | 不同 batch size 影响梯度估计噪声和 BN 行为 |
| 5 | **label_smoothing**：dbond_m=0.1, GT 无 | 🟡 中 | 建议在 GT 中也加上 label_smoothing 以保持一致 |
| 6 | **max_seq_len 不一致**：dbond_m=32, GT=100 | 🟡 中 | 取决于数据集实际长度分布，需确认 |

### 3.2 架构差异（预期内的升级，无需修改）

| # | 差异 | 说明 |
|---|------|------|
| 1 | GT 用 GCN+GAT 替代纯 Transformer | 架构升级，引入图结构 |
| 2 | GT 新增边特征、物理化学特征 | 特征增强 |
| 3 | GT 新增全局节点、长程边 | 图结构增强 |
| 4 | GT 用 AdamW 替代 Adam | 优化器升级 |
| 5 | GT 新增 cosine scheduler | 训练策略改进 |
| 6 | GT 新增梯度裁剪 | 训练稳定性改进 |

### 3.3 评估指标对齐状态

✅ **已完全对齐**。graph_transform 的 `BinaryBondMetrics` 已包含 dbond_m 的全部 14 个同口径指标（subset_acc, ex_acc, ex_precision, ex_recall, ex_f1, lab_acc_ma/mi, lab_precision_ma/mi, lab_recall_ma/mi, lab_f1_ma/mi），且计算逻辑与 `arc_dbond/multi_label_metrics.py` 一致。

---

## 四、建议行动项

1. **确认 env_var 差异**：检查数据集中 `scan_num` 和 `rt` 列的可用性，决定 GT 应使用哪个
2. **考虑对齐训练超参数**：如果要公平对比两个模型的性能，建议将 GT 的 `learning_rate`、`batch_size`、`num_heads`、`label_smoothing` 与 dbond_m 对齐
3. **评估指标无需修改**：已完全对齐
