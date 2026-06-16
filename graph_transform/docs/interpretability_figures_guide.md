# 可解释性图表解读手册

> 适用图：`residue_pair_matrix.svg`、`occlusion_<sample_id>.svg`、`occlusion_aggregate.svg`
> 生成脚本：`graph_transform/scripts/residue_pair_analysis.py`、`graph_transform/scripts/occlusion_analysis.py`

本手册解释三张论文级可解释性图的**坐标轴、指标含义、好坏阈值**，以及如何把它们串成完整的论证链。

---

## 一、Residue-Pair Chemistry Matrix（残基对化学矩阵）

**文件**：`results/residue_pair_matrix/residue_pair_matrix.svg`
**研究问题**：不同残基对 (X-Y) 的**内在断裂倾向**是多少？模型预测是否与之一致？

### 1.1 坐标轴约定（三个子图共用）

| 轴 | 含义 | 备注 |
|----|------|------|
| **横轴 (列, Y)** | 肽键 C 端残基（即键右侧的残基） | 24 种 AA：`A B C D E F G H I K L M N O P Q R S T V W X Y Z` |
| **纵轴 (行, X)** | 肽键 N 端残基（即键左侧的残基） | 同上 |
| **格子 (X, Y)** | 所有 "X-Y" 肽键（X 在 N 端，Y 在 C 端）的聚合统计 | **非对称**：X-Y ≠ Y-X |

> 例：第 `D` 行、第 `E` 列的格子 = 数据集中所有 "D-E" 肽键的统计（D 在 N 端，E 在 C 端）。

### 1.2 三个子图

#### (a) Empirical cleavage rate `P(broken | X-Y)`  [%]
- **含义**：该残基对的真实断裂频率（来自测试集 `true_multi` 列）
- **数值**：百分比整数，例如 `45` = 该残基对中有 45% 的键真实断裂
- **色阶**：黄→红 (`YlOrRd`)，0% → 100%
- **解读**：**纯化学基线**，反映残基对的内在断裂倾向，与模型无关

#### (b) Model predicted `E[σ(model) | X-Y)`  [%]
- **含义**：模型对该残基对预测的平均断裂概率
- **数值**：百分比整数，与 (a) 同尺度（便于直接对比）
- **色阶**：与 (a) 完全一致
- **解读**：**模型对该化学的"理解"**

#### (c) Difference `(predicted − empirical)`  [signed bias]
- **含义**：模型预测减去真实频率（**百分点**）
- **数值**：带符号 2 位小数，例如 `+0.12`（高估 12 pp）/ `−0.08`（低估 8 pp）
- **色阶**：蓝→白→红 (`RdBu_r`)，−0.5 → +0.5
  - **红色**：模型**高估**了该残基对的断裂倾向
  - **蓝色**：模型**低估**了
  - **白色**：完美匹配

### 1.3 格子标注约定

| 标注 | 含义 | 可信度 |
|------|------|--------|
| 纯数字（如 `45`） | 样本数 N ≥ 50 | **可信** |
| 数字 + `*`（如 `33*`） | N ∈ [10, 50) | 中等，需谨慎 |
| 数字 + `**`（如 `12**`） | N < 10 | **低**，统计不稳定 |
| 灰色斜线填充 | N = 0 | 该残基对在数据中不存在 |

### 1.3.1 关于空格子（重要：本项目数据特性）

本项目为 **D-氨基酸镜像肽数据存储**，**刻意排除了部分标准氨基酸**。
矩阵中的空格子不是 bug，而是设计性数据特性：

**字母表 25 字符** = `#` (padding) + 24 种 AA 代码：
- **标准 20 AA**：`A C D E F G H I K L M N P Q R S T V W Y`
- **非标准 4 AA**：`B` (Asx, Asp/Asn 歧义码)、`O` (Pyrrolysine, 吡咯赖氨酸)、
  `X` (unknown)、`Z` (Glx, Glu/Gln 歧义码)

**测试集 6072 实际出现的 19 种 AA**（5 种字母表成员完全缺失）：

| 缺失 AA | 类型 | 训练集出现次数 | 测试集出现次数 | 缺失原因 |
|---------|------|--------------|--------------|---------|
| **C** (Cys, 半胱氨酸) | 标准 | **0** | **0** | 设计性排除：含 -SH 基易形成二硫键，破坏序列稳定性 |
| **M** (Met, 甲硫氨酸) | 标准 | **0** | **0** | 设计性排除：含硫原子易氧化 |
| **W** (Trp, 色氨酸) | 标准 | **0** | **0** | 设计性排除：含吲哚环，体积大易碎裂 |
| **V** (Val, 缬氨酸) | 标准 | 1,043 | **0** | 训练集极少（仅 0.0003%），测试集恰好无 |
| **Z** (Glx, Glu/Gln) | 非标准 | 1,131 | **0** | 训练集极少，测试集恰好无 |

**实际出现在测试集中的 19 种 AA**：
```
A B D E F G H I K L N O P Q R S T X Y
```

**频次分布**（测试集 96,605 样本 / 2.5M+ 残基）：
- 高频（>10⁶）：K A D G P L Q Y H N T S F E O B
- 中频（10⁴~10⁵）：I
- 低频（<10⁴）：R (5,415), X (1,046)
- 4 种非标准 AA 中：**B (130K) 和 O (135K) 高频**；X (1K) 低频；**Z 在测试集无**

> 因此矩阵中 C/M/W/V/Z 对应的 5 行 + 5 列共 ~215 个格子全为空，
> 占总数 576 的 ~37%。这是**论文应该明确说明的数据特性**，
> 不是缺失数据或 bug。

### 1.3.2 论文中如何描述

**推荐措辞**：
> "The DBond mirror-image peptide dataset employs a 24-letter alphabet
> (20 standard + 4 ambiguity codes B/O/X/Z). Three sulfur-containing or
> bulky aromatic residues (C, M, W) are **by design** excluded to ensure
> sequence stability during MS/MS analysis. V and Z are present in the
> training set but absent in the held-out test fold. The empirical and
> predicted statistics reported here cover the 19 amino acids that
> actually appear in the test set."

### 1.4 好坏阈值

| 全局指标（在 `residue_pair_summary.json`） | 优 | 良 | 差 |
|------|------|------|------|
| `global_pearson_r_empirical_vs_predicted` | ≥ 0.85 | 0.70–0.85 | < 0.70 |
| `global_mae_predicted_vs_empirical` | < 0.05 | 0.05–0.12 | > 0.12 |
| `coverage`（非空格子占比） | > 0.85 | 0.70–0.85 | < 0.70 |

### 1.5 如何读图（论文叙事）

1. **看 (a)**：识别"易断"残基对（深红）vs"稳定"残基对（浅黄）—— 这是化学基线
2. **对比 (a) vs (b)**：模型是否复现了相同的红/黄模式？
3. **看 (c)**：偏差集中在哪里？是否有系统性偏置（如对某类 AA 普遍高估）？
4. **看 `*` 分布**：稀有 AA 的预测是否可信？

---

## 二、Occlusion vs Attention（单样本归因对比）

**文件**：`results/occlusion_attribution/occlusion_<sample_id>.svg` × 15
**研究问题**：每个残基位置 j **因果上**对键 i 的预测有多大贡献？与 attention 是否一致？

### 2.1 坐标轴约定（两子图共用）

| 轴 | 含义 |
|----|------|
| **横轴（列, i）** | 肽键位置 i = 0, 1, ..., L−2（标签 `A-B`, `B-C`, ...，旋转 45°） |
| **纵轴（行, j）** | 残基位置 j = 0, 1, ..., L−1（标签 `j:AA`，如 `0:A`） |
| **格子 (j, i)** | "残基 j 对键 i 的归因强度" |
| **图像尺寸** | 自适应：每键 ≥ 0.45"，最长 28"，避免标签重叠 |

### 2.2 两个子图

#### (a) Occlusion sensitivity  `mean |Δp[j→aa] on bond i|`
- **方法**：把残基 j 突变为 24 种 AA（排除原 AA），记录键 i 预测变化的绝对值，求平均
- **数值**：[0, 1]，越大越敏感
- **色阶**：黄→红，0 → max
- **含义**：**因果归因** —— 残基 j 的改变在因果上多大程度影响键 i 的预测
- **预期模式**：对角线主导（局部），偏离对角线 = 长程依赖

#### (b) Functional-saliency attention  (residue → bond)
- **方法**：从最后一层 attention 提取，转换为残基→键的对应关系
- **数值**：[0, +∞)，归一化到 colorbar 范围
- **色阶**：同 (a)，便于直接对比
- **含义**：**模型内部注意力** —— 残基 j 在前向计算中多大程度"参与"了键 i 的预测

### 2.3 标题关键指标

**Pearson r (attention vs occlusion)** —— 单样本的一致性指标：

| r 范围 | 一致性等级 | 解读 |
|--------|-----------|------|
| r ≥ 0.5 | **strongly consistent** | attention 是可靠的解释，与因果归因强一致 |
| 0.3 ≤ r < 0.5 | **moderately consistent** | attention 部分捕捉了模型决策，但有遗漏 |
| 0 ≤ r < 0.3 | **weakly consistent** | attention 与因果归因弱相关，需谨慎解读 |
| r < 0 | **inversely consistent** | attention 与因果归因方向相反 |

### 2.4 如何读图（单样本）

1. **看 (a) 对角线**：是否对角线最亮？—— 是 = 局部主导；否 = 长程依赖强
2. **看 (a) 整体稀疏度**：少量亮格 = 模型依赖少数关键残基；弥散 = 分布式决策
3. **对比 (a) vs (b)**：亮区是否重合？—— 重合度高 = attention 可信
4. **看 r 值**：r ≥ 0.5 才能说"attention 解释了模型行为"

---

## 三、Occlusion-Attention Consistency（聚合一致性）

**文件**：`results/occlusion_attribution/occlusion_aggregate.svg`
**研究问题**：在所有样本上，attention 整体上是否是可靠的解释？

### 3.1 左图 (a) Global consistency across all samples

| 轴 | 含义 |
|----|------|
| **横轴** | Functional-saliency attention（残基→键），所有样本所有 (j,i) 点拼接 |
| **纵轴** | Occlusion sensitivity（残基→键），同上 |
| **可视化** | 散点（n<5000）/ Hexbin 密度（n≥5000，颜色越深点越密） |

**关键指标**（图标题）：
- **Pearson r**（线性相关性）：衡量 attention 与 occlusion 的线性一致程度
- **Spearman ρ**（秩相关性）：衡量排序一致性，对异常值更鲁棒

| r / ρ 范围 | 解读 |
|-----------|------|
| ≥ 0.5 | 强一致：attention 在群体水平上可靠 |
| 0.3 – 0.5 | **中等一致**：attention 部分可信，但需 occlusion 补充 |
| 0.1 – 0.3 | 弱一致：attention 与因果归因偏离较大 |
| < 0.1 | 无一致：attention 不是有效解释 |

### 3.2 右图 (b) Per-sample consistency distribution

**箱线图**：15 个样本各自的 Pearson r 分布

| 元素 | 含义 |
|------|------|
| 箱体 | r 的 IQR（25%–75% 分位） |
| 箱内黑横线 | **中位数** r |
| 红色散点 | 每个样本的 r 值 |
| 黑色虚线 | **均值** r |
| 橙色点线 | r = 0.5（强一致性阈值） |
| 灰色横线 | r = 0（无相关性） |

**关键阈值**：
- **n_strong / n_valid**（图标题）：r ≥ 0.5 的样本数 / 有效样本数
  - 例：`3/15 samples r ≥ 0.5` → 20% 样本达到强一致
  - 例：`0/15` → 无样本达到强一致

### 3.3 好坏阈值（论文叙事）

| 整体表现 | 中位 r | n_strong / n_valid | 论文措辞建议 |
|---------|--------|-------------------|------------|
| **优** | ≥ 0.5 | ≥ 50% | "attention saliency is strongly consistent with causal occlusion" |
| **良** | 0.35–0.5 | 20–50% | "attention is moderately consistent; occlusion reveals additional structure" |
| **中** | 0.2–0.35 | < 20% | "attention and occlusion capture complementary aspects" |
| **差** | < 0.2 | 0% | "attention diverges from causal attribution; recommend occlusion as primary" |

---

## 四、三张图的论证链（论文整合）

```
残基对矩阵 (Figure X)         单样本 occlusion (Figure Y)        聚合一致性 (Figure Z)
        │                              │                              │
        ▼                              ▼                              ▼
   宏观化学层面              微观单样本层面                      群体验证层面
   模型捕获了化学            attention 是否                      attention 整体可靠性
   (a)↔(b) 模式一致吗?       与因果归因一致吗?                  量化指标
        │                              │                              │
        └────────── 三层共同回答 ──────┴──────────────────────────────┘
                       "模型既学到化学（宏观）
                        又有可解释的局部推理（微观）
                        attention 是部分可靠的解释"
```

---

## 五、`occlusion_aggregate.svg` 读图示例（image_3f36d9.png 验证）

### 5.1 用户读图内容（已确认正确）

> - 左图：Hexbin 密度图，深蓝区集中在 (0, 0) 附近
> - 全局 Pearson r = **+0.324**
> - 全局 Spearman ρ = **+0.362**
> - 右图：箱线图，中位 r = **+0.357**，均值 = **+0.325**
> - 强一致性阈值 r = 0.5 之上：**0 / 15** 样本

### 5.2 验证 + 补充解读

| 用户结论 | 验证 | 补充 |
|---------|------|------|
| "无样本达到 r ≥ 0.5" | ✅ 正确 | 中位 r = 0.357，所有点聚集在 [0.2, 0.45] 区间 |
| "整体缺乏显著一致性" | ⚠️ 部分正确 | 应说"**仅弱到中等一致**"，r ≈ 0.32 不是零相关 |
| "两者结果不匹配" | ⚠️ 需精确化 | 应说"**部分匹配**"（r > 0 表示方向一致，但强度不足） |

### 5.3 论文措辞建议（基于此图数据）

**❌ 避免说**：
- "attention 完全无效" —— 错误，r 显著大于 0
- "attention 与因果归因无关" —— 错误，方向一致
- "attention 是可靠的解释" —— 过强，r < 0.5

**✅ 推荐说**：
- "Functional-saliency attention is **moderately consistent** with causal occlusion
   (global Pearson r = +0.32, Spearman ρ = +0.36), indicating that attention captures
   part of the model's decision-relevant features but is not a complete explanation."
- "Per-sample analysis shows no individual peptide reaches the strong-consistency
   threshold (r ≥ 0.5); median r = +0.36 across 15 samples."
- "We therefore recommend occlusion as a complementary attribution method for
   D-amino acid bond cleavage prediction."

### 5.4 此结果对项目的意义

**正面价值**（这是有用的发现，不是失败）：
1. **诚实呈现**：增加论文可信度，避免"attention 是解释"的过度声明
2. **方法学贡献**：明确推荐 occlusion 作为更可靠的归因方法
3. **D-AA 特殊性**：D-氨基酸的稀疏模式可能使 attention 比 L-AA 更难解释
4. **未来工作**：为后续设计更可解释的架构（如内置 occlusion 监督）提供动机

---

## 附：所有指标的"红绿灯"快速参考

| 指标 | 🟢 优 | 🟡 良 | 🔴 差 |
|------|------|------|------|
| 残基对矩阵：empirical vs predicted Pearson r | ≥ 0.85 | 0.70–0.85 | < 0.70 |
| 残基对矩阵：MAE (predicted vs empirical) | < 0.05 | 0.05–0.12 | > 0.12 |
| Occlusion 单样本 r | ≥ 0.5 | 0.3–0.5 | < 0.3 |
| Occlusion 聚合 global r | ≥ 0.5 | 0.3–0.5 | < 0.3 |
| Occlusion 聚合 n_strong / n_valid | ≥ 50% | 20–50% | < 20% |
