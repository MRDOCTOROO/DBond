# graph_transform/evaluation/metrics.py 审计报告

## 一、模块功能概述

[`metrics.py`](../graph_transform/evaluation/metrics.py) **不仅做了指标排序**，它包含以下完整功能：

| 功能 | 代码位置 | 说明 |
|------|----------|------|
| 指标排序常量与排序函数 | 第 26-86 行 | `DBOND_M_COMPARABLE_METRIC_ORDER` + `TASK_EXTRA_METRIC_ORDER` + `order_binary_bond_metric_dict()` |
| Example-based 指标 | 第 112-137 行 | `_example_subset_accuracy`, `_example_accuracy`, `_example_precision`, `_example_recall`, `_example_f1` |
| Label-based 指标 | 第 140-200 行 | `_label_quantity` + macro/micro 的 accuracy/precision/recall/f1 |
| 核心指标类 `BinaryBondMetrics` | 第 203-342 行 | `update()` → `compute()` 流式计算所有指标 |
| 便捷函数 | 第 345-357 行 | `compute_binary_bond_metrics()` + 向后兼容别名 |

---

## 二、指标定义正确性逐项检查

### 2.1 Example-based 指标（与 [`arc_dbond/multi_label_metrics.py`](../arc_dbond/multi_label_metrics.py) 对比）

| 指标 | graph_transform 定义 | arc_dbond 参考 | 是否一致 |
|------|---------------------|---------------|---------|
| `_example_subset_accuracy` | `np.mean(np.all(np.equal(gt, pred), axis=1))` | `np.mean(np.all(np.equal(gt, predict), axis=1))` | ✅ 一致 |
| `_example_accuracy` | Jaccard 系数均值：`mean(AND/OR)` | 同上 | ✅ 一致 |
| `_example_precision` | `mean(AND / pred_sum)` | 同上 | ✅ 一致 |
| `_example_recall` | `mean(AND / gt_sum)` | 同上 | ✅ 一致 |
| `_example_f1` | 从 precision 和 recall 派生 | 同上 | ✅ 一致 |

### 2.2 Label-based 指标（与 [`arc_dbond/multi_label_metrics.py`](../arc_dbond/multi_label_metrics.py) 对比）

| 指标 | graph_transform 定义 | arc_dbond 参考 | 是否一致 |
|------|---------------------|---------------|---------|
| `_label_quantity` | TP/FP/TN/FN 按 label 维度 stack | 同上 | ✅ 一致 |
| `_label_accuracy_macro` | `mean((TP+TN) / total)` per label | 同上 | ✅ 一致 |
| `_label_accuracy_micro` | `sum(TP+TN) / sum(total)` | 同上 | ✅ 一致 |
| `_label_precision_macro` | `mean(TP / (TP+FP))` per label | 同上 | ✅ 一致 |
| `_label_precision_micro` | `sum(TP) / sum(TP+FP)` | 同上 | ✅ 一致 |
| `_label_recall_macro` | `mean(TP / (TP+FN))` per label | 同上 | ✅ 一致 |
| `_label_recall_micro` | `sum(TP) / sum(TP+FN)` | 同上 | ✅ 一致 |
| `_label_f1_macro` | `mean((1+β²)TP / ((1+β²)TP + β²FN + FP))` per label | 同上 | ✅ 一致 |
| `_label_f1_micro` | `sum((1+β²)TP) / sum((1+β²)TP + β²FN + FP)` | 同上 | ✅ 一致 |

**结论：所有 dbond_m 对齐的指标定义完全正确。**

---

## 三、发现的问题

### 问题 1：`precision`/`recall`/`f1` 使用 `average="binary"` 而非 `"macro"` ⚠️ 语义偏差

**位置**：[`metrics.py`](../graph_transform/evaluation/metrics.py:276-278) 第 276-278 行

```python
"precision": precision_score(valid_targets, binary_valid_predictions, zero_division=0),
"recall": recall_score(valid_targets, binary_valid_predictions, zero_division=0),
"f1": f1_score(valid_targets, binary_valid_predictions, zero_division=0),
```

**分析**：`sklearn` 的 `precision_score` 等函数在**二分类**输入（1D array）下默认 `average="binary"`，即只计算正类的指标。由于 `valid_targets` 和 `binary_valid_predictions` 在这里都是 1D 展平后的数组，所以 `average="binary"` 实际上就是正确的行为——它计算的是"键级别正类"的 precision/recall/f1。

**对比**：[`mini_ghtrans/evaluation/metrics.py`](../mini_ghtrans/evaluation/metrics.py:76-78) 使用了 `average='macro'`，但那是真正的多标签矩阵输入（2D），两者场景不同。

**结论**：**此处定义正确**，因为输入是 1D 展平的二分类数组。

### 问题 2：`precision_micro`/`recall_micro`/`f1_micro` 冗余 ⚠️ 无实际错误

**位置**：[`metrics.py`](../graph_transform/evaluation/metrics.py:279-281) 第 279-281 行

```python
"precision_micro": precision_score(..., average="binary", zero_division=0),
"recall_micro": recall_score(..., average="binary", zero_division=0),
"f1_micro": f1_score(..., average="binary", zero_division=0),
```

**分析**：对于 1D 二分类数组，`average="binary"` 和 `average="micro"` 的结果完全相同。所以这三个指标与上面的 `precision`/`recall`/`f1` 值完全一样，是冗余的。

**结论**：**无计算错误，但存在冗余**。如果未来需要真正的 micro-average，应传入 2D 矩阵并使用 `average="micro"`。

### 问题 3：`auc`/`auc_macro`/`auc_micro`/`auc_weighted` 全部相同 ⚠️ 信息损失

**位置**：[`metrics.py`](../graph_transform/evaluation/metrics.py:308-311) 第 308-311 行

```python
metrics["auc"] = auc
metrics["auc_macro"] = auc
metrics["auc_micro"] = auc
metrics["auc_weighted"] = auc
```

**分析**：由于输入是 1D 展平数组，`roc_auc_score` 只能计算一个全局 AUC。macro/micro/weighted 这些多标签 AUC 变体需要 2D 矩阵输入才能区分。当前实现将它们全部设为同一个值，**丢失了多标签维度的区分信息**。

**对比**：[`mini_ghtrans/evaluation/metrics.py`](../mini_ghtrans/evaluation/metrics.py:100-102) 正确使用了 2D 矩阵计算不同的 AUC 变体：
```python
metrics['auc_macro'] = roc_auc_score(targets, predictions, average='macro')
metrics['auc_micro'] = roc_auc_score(targets, predictions, average='micro')
metrics['auc_weighted'] = roc_auc_score(targets, predictions, average='weighted')
```

**结论**：**这是一个设计缺陷**。如果需要真正的 macro/micro/weighted AUC，应基于 `target_matrix` / `pred_matrix` 计算。

### 问题 4：`class_0_precision`/`class_0_recall`/`class_0_f1` 冗余

**位置**：[`metrics.py`](../graph_transform/evaluation/metrics.py:312-314) 第 312-314 行

```python
metrics["class_0_precision"] = metrics["precision"]
metrics["class_0_recall"] = metrics["recall"]
metrics["class_0_f1"] = metrics["f1"]
```

**分析**：在二分类场景下只有一个类（正类），所以 class_0 指标与全局指标相同。这是为了保持输出格式兼容性而设置的别名。

**结论**：**无错误，仅为格式兼容别名。**

### 问题 5：`pred_matrix` 填充零可能影响 label-based 指标 ⚠️ 潜在偏差

**位置**：[`metrics.py`](../graph_transform/evaluation/metrics.py:264-272) 第 264-272 行

```python
max_len = max((sample.size for sample in sample_probabilities), default=0)
pred_matrix = np.zeros((len(sample_probabilities), max_len), dtype=np.int32)
target_matrix = np.zeros((len(self.sample_targets), max_len), dtype=np.int32)
```

**分析**：由于不同样本的键数不同（变长序列），矩阵用零填充到 `max_len`。对于 label-based 指标，这些填充位置的 gt=0、pred=0 会被当作"真负例"参与计算，可能导致指标偏高。

**对比**：[`arc_dbond/multi_label_metrics.py`](../arc_dbond/multi_label_metrics.py) 的输入是固定宽度的标签矩阵（所有样本标签数相同），不存在此问题。

**结论**：**这是一个潜在的指标偏差**。理想情况下应使用 mask 来排除填充位置，或确保所有样本的标签数相同。不过在实际使用中，如果数据集的序列长度差异不大，影响可能较小。

---

## 四、与 training/metrics.py 的重复问题

[`graph_transform/training/metrics.py`](../graph_transform/training/metrics.py) 和 [`graph_transform/evaluation/metrics.py`](../graph_transform/evaluation/metrics.py) 的内容**几乎完全相同**（包括所有指标函数、`BinaryBondMetrics` 类、排序常量等），存在严重的代码重复。

**差异**：`training/metrics.py` 额外包含 `MetricTracker` 类（第 327-365 行），用于训练过程中跟踪最佳指标。

**建议**：应将共享的指标逻辑提取到一个公共模块，避免后续维护时出现不一致。

---

## 五、总结

| 类别 | 状态 |
|------|------|
| dbond_m 对齐的 example-based 指标 | ✅ 完全正确 |
| dbond_m 对齐的 label-based 指标 | ✅ 完全正确 |
| 基础二分类指标（accuracy/precision/recall/f1） | ✅ 正确 |
| AUC 多标签变体（macro/micro/weighted） | ⚠️ 全部相同，未真正区分 |
| micro-average 指标 | ⚠️ 与 binary 指标冗余 |
| 变长序列零填充对 label-based 指标的影响 | ⚠️ 潜在偏差 |
| 代码重复（training vs evaluation） | ⚠️ 维护风险 |
