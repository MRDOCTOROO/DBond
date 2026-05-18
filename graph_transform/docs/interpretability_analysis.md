# GraphTransformer 可解释性分析文档

## 概述

本文档详细说明了 GraphTransformer 模型的可解释性分析框架，包括脚本功能、输出结果用途、图表指标解读等内容。该框架旨在打开深度学习模型的"黑盒"，验证模型的预测依据是否符合化学直觉。

---

## 1. 文件结构

```
graph_transform/
├── scripts/
│   └── attention_visualization.py    # 主脚本：注意力可视化与分析
├── utils/
│   ├── attention_extractor.py        # 工具：注意力权重提取
│   └── visualization.py              # 工具：可视化函数库
└── docs/
    └── interpretability_analysis.md  # 本文档
```

---

## 2. 脚本功能说明

### 2.1 主脚本：`attention_visualization.py`

**功能**：
- 加载训练好的 GraphTransformer 模型
- 从测试数据中提取注意力权重
- 生成可视化图表和统计分析

**命令行参数**：

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--config` | str | 必需 | 模型配置文件路径（YAML） |
| `--checkpoint` | str | 必需 | 模型权重文件路径（.pt） |
| `--input_csv` | str | 必需 | 测试数据CSV文件路径 |
| `--output_dir` | str | `results/attention_viz` | 输出目录 |
| `--num_samples` | int | 5 | 案例研究样本数 |
| `--num_stat_samples` | int | 500 | 统计分析样本数 |
| `--sampling_strategy` | str | `stratified` | 抽样策略（random/stratified） |
| `--num_length_bins` | int | 5 | 分层抽样时序列长度分组数 |
| `--skip_case_study` | flag | False | 跳过案例研究，只进行统计分析 |
| `--infer_config` | flag | False | 从检查点推断模型配置 |

**使用示例**：

```bash
# 论文统计分析（推荐）
python graph_transform/scripts/attention_visualization.py \
    --config graph_transform/config/default.yaml \
    --checkpoint checkpoints/best_model.pt \
    --input_csv dataset/test_data.csv \
    --output_dir results/attention_viz_paper \
    --num_samples 3 \
    --num_stat_samples 1000 \
    --sampling_strategy stratified \
    --infer_config
```

### 2.2 工具模块：`attention_extractor.py`

**功能**：
- 从 GraphTransformer 模型中提取注意力权重
- 处理单个样本的格式转换（单数键名 -> 复数键名，适配批处理格式）
- 处理边级别注意力权重的映射

**核心类**：
- `AttentionExtractor`：注意力权重提取器

**核心方法**：
- `extract_attention_weights(batch_data)`：从批次数据提取注意力权重
- `extract_attention_for_sample(sample_data)`：从单个样本提取注意力权重

**为什么单独提取注意力权重？**
- 原始模型的 `forward()` 方法只输出预测结果，不输出注意力权重
- `get_attention_weights()` 方法存在 padding 和 global_node 处理不完整的问题
- 本模块复制了 `forward()` 中完整的节点准备逻辑，确保注意力权重与训练时一致

### 2.3 工具模块：`visualization.py`

**功能**：
- 提供各种可视化函数
- 处理边级别注意力权重到节点矩阵的映射
- 生成论文级别的图表

**核心函数**：

| 函数名 | 功能 | 输出 |
|-------|------|------|
| `plot_attention_heatmap()` | 绘制单层注意力热力图 | `attention_heatmap_layerX.png` |
| `plot_peptide_attention_graph()` | 绘制单层肽段结构图叠加注意力 | `peptide_attention_layerX.png` |
| `plot_peptide_attention_combined()` | **绘制多层肽段注意力合并图（推荐）** | `peptide_attention_combined.png` |
| `plot_attention_head_comparison()` | 绘制单层多头注意力比较图 | `attention_heads_layerX.png` |
| `plot_attention_heads_combined()` | **绘制多层多头注意力合并图（推荐）** | `attention_heads_combined.png` |
| `analyze_attention_patterns()` | 分析注意力模式与断裂关系 | 分析字典 |
| `plot_attention_analysis()` | 绘制综合分析图（4子图） | `comprehensive_analysis.png` |

---

## 3. 输出结果说明

### 3.1 输出文件结构

```
results/attention_viz_paper/
│
├── comprehensive_analysis.png                    # 案例研究综合图（少量样本）
├── comprehensive_analysis_statistical.png        # 统计分析综合图（大样本）[主图]
├── statistical_analysis_results.csv              # 统计结果原始数据
│
├── sample_0/                                     # 案例样本1
│   ├── peptide_attention_combined.png            # [推荐] 多层肽段注意力合并图
│   ├── attention_heads_combined.png              # [推荐] 多层多头注意力合并图
│   ├── attention_heatmap_layer0.png              # Layer 0 注意力热力图
│   ├── attention_heatmap_layer1.png              # Layer 1 注意力热力图
│   ├── peptide_attention_layer0.png              # Layer 0 肽段结构图
│   ├── peptide_attention_layer1.png              # Layer 1 肽段结构图
│   ├── attention_heads_layer0.png                # Layer 0 多头注意力比较
│   ├── attention_heads_layer1.png                # Layer 1 多头注意力比较
│   ├── attention_analysis_layer0.txt             # Layer 0 数值分析结果
│   └── attention_analysis_layer1.txt             # Layer 1 数值分析结果
│
├── sample_1/                                     # 案例样本2
└── sample_2/                                     # 案例样本3
```

**合并图说明**：
- `peptide_attention_combined.png`：将所有层的肽段注意力图横向排列，减少留白，适合论文展示
- `attention_heads_combined.png`：将所有层的多头注意力图横向排列，便于对比不同层的注意力头

### 3.2 各文件用途

| 文件名 | 用途 | 论文使用建议 |
|-------|------|-------------|
| `comprehensive_analysis_statistical.png` | 统计分析主图，展示模型整体机制 | **主图**，基于大样本统计（500-1000个） |
| `comprehensive_analysis.png` | 案例研究综合图 | 补充材料（3-5个样本） |
| `peptide_attention_layerX.png` | 展示模型对具体序列的关注点 | 案例研究图（选取1-2个典型序列） |
| `attention_heads_layerX.png` | 展示不同注意力头的功能分化 | 补充材料 |
| `attention_heatmap_layerX.png` | 展示注意力权重分布 | 补充材料 |
| `attention_analysis_layerX.txt` | 单个样本的数值分析结果 | 参考 |
| `statistical_analysis_results.csv` | 原始统计数据 | 数据可用性声明 |

---

## 4. 图表指标详细解读

### 4.1 综合分析图（comprehensive_analysis_statistical.png）[主图]

该图包含 4 个子图，分别用 **(a)-(d)** 标记，用于论文中引用。

---

#### (a) Attention on Broken vs Intact Bonds（断裂键 vs 完整键的注意力权重）

**图表类型**：分组柱状图

**横轴（X轴）**：`Layer/Head`
- 表示不同的注意力层和头组合
- 例如：`0_0` 表示 Layer 0 Head 0，`1_3` 表示 Layer 1 Head 3
- `avg` 表示所有头的平均值

**纵轴（Y轴）**：`Mean Attention Weight`（平均注意力权重）
- 表示模型对该类键的平均关注度
- 数值越高表示模型越关注该类键
- 范围：[0, +∞)，实际值通常在 [0, 1] 范围内

**颜色编码**：
- 🔴 红色：`Broken Bonds`（断裂键）
- 🔵 蓝色：`Intact Bonds`（完整键）

**指标解读**：
- **红色柱显著高于蓝色柱**：该注意力头更关注断裂键，是 **"断裂检测头"**
- **蓝色柱显著高于红色柱**：该注意力头更关注完整键，是 **"稳定性检测头"**
- **两者相近**：该注意力头对两类键的关注度相似

**好坏标准**：
- ✅ **好**：存在明显的功能分化（部分头关注断裂，部分头关注稳定），说明多头注意力机制实现了功能解耦
- ❌ **差**：所有头的关注度相似，缺乏功能分化，说明模型没有学到有效的注意力模式

---

#### (b) Attention Preference for Broken Bonds（断裂键注意力偏好）

**图表类型**：柱状图

**横轴（X轴）**：`Layer/Head`
- 同 (a)

**纵轴（Y轴）**：`Attention Difference (Broken - Intact)`
- 计算公式：`断裂键平均权重 - 完整键平均权重`
- **正值**（>0）表示更关注断裂键
- **负值**（<0）表示更关注完整键
- 范围：(-∞, +∞)，实际值通常在 [-0.5, 0.5] 范围内

**颜色编码**：
- 🔴 红色：正值（更关注断裂键）
- 🔵 蓝色：负值（更关注完整键）

**指标解读**：
- **红色柱高**：该注意力头专门负责检测易断裂的肽键
- **蓝色柱高**：该注意力头专门关注稳定的肽键结构
- **接近 0**：该注意力头对两类键无明显偏好

**好坏标准**：
- ✅ **好**：存在明显的正负分化，说明模型学会了分工（某些头负责断裂检测，某些头负责稳定检测）
- ❌ **差**：所有值接近 0，说明模型没有学到有效的区分特征

---

#### (c) Attention-Breakage Correlation（注意力-断裂相关性）

**图表类型**：柱状图

**横轴（X轴）**：`Layer/Head`
- 同 (a)

**纵轴（Y轴）**：`Correlation Coefficient`（相关系数）
- 计算方法：Pearson 相关系数
- 范围：[-1, 1]
- 含义：注意力权重与键断裂标签之间的线性相关程度

**颜色编码**：
- 🟢 绿色：正相关（>0）
- 🟠 橙色：负相关（<0）

**指标解读**：
| 相关系数值 | 含义 |
|-----------|------|
| r > 0.5 | 强正相关：注意力越高，越可能断裂 |
| 0.3 < r < 0.5 | 中等正相关 |
| 0.1 < r < 0.3 | 弱正相关 |
| -0.1 < r < 0.1 | 无相关性 |
| r < -0.1 | 负相关：注意力越高，越不容易断裂（异常） |

**好坏标准**：
- ✅ **好**：多个头显示正相关（r > 0.3），说明模型通过"赋予特定化学键高注意力"来预测其断裂
- ⚠️ **一般**：相关性在 0.1-0.3 之间，说明模型学到的特征有一定意义但不够强
- ❌ **差**：相关性接近 0 或为负值，说明模型的注意力机制与预测目标脱节

---

#### (d) Bond Distribution in Analysis（键分布统计）

**图表类型**：分组柱状图

**横轴（X轴）**：`Layer/Head`
- 同 (a)

**纵轴（Y轴）**：`Number of Bonds`（键数量）
- 表示分析样本中各类键的总数量

**颜色编码**：
- 🔴 红色：`Broken Bonds`（断裂键数量）
- 🔵 蓝色：`Intact Bonds`（完整键数量）

**指标解读**：
- 展示数据集中断裂键和完整键的比例
- 通常完整键数量远多于断裂键（**类别不平衡**）
- 有助于理解模型面临的挑战

**好坏标准**：
- 该图主要用于展示数据分布，**无好坏之分**
- 引用时可说明："由于肽段中大多数肽键保持完整，数据存在类别不平衡问题"

---

### 4.2 肽段结构图（peptide_attention_layerX.png）

**图表类型**：双面板图

**左图：肽段序列图**
- **节点**：氨基酸残基（显示单字母代码，如 K, S, O, B 等）
- **边**：肽键连接（相邻残基之间的连线）
- **边宽度**：注意力权重（越宽表示模型越关注）
- **边颜色**：
  - 🔴 红色：预测断裂的键
  - 🔵 蓝色：预测完整的键

**右图：相邻注意力矩阵**
- 热力图显示相邻残基间的注意力权重
- 颜色越深（越红）表示注意力越高

**横轴**：Key Position（键位置，即目标残基位置）
**纵轴**：Query Position（查询位置，即源残基位置）

**颜色编码**：`YlOrRd` 色谱（黄→橙→红）
- 浅黄色：低注意力权重
- 深红色：高注意力权重

**指标解读**：
- **深红色区域**：模型高度关注的肽键（可能是易断裂位点）
- **浅黄色区域**：模型较少关注的肽键（可能是稳定位点）

**好坏标准**：
- ✅ **好**：深红色区域与实际断裂位点匹配
- ❌ **差**：深红色区域与实际断裂位点不匹配

**Layer 0 vs Layer 1 对比**：
- Layer 0 注意力相对均匀（浅色为主），说明浅层在做局部上下文聚合
- Layer 1 注意力高度集中（深色斑块），说明深层在锚定关键断裂位点

---

### 4.3 多头注意力比较图（attention_heads_layerX.png）

**图表类型**：多面板热力图

**每个子图**：一个注意力头的权重矩阵

**横轴**：Key Position（键位置）
**纵轴**：Query Position（查询位置）

**颜色编码**：`viridis` 色谱（深紫→黄绿）
- 深色：低注意力权重
- 亮色：高注意力权重

**指标解读**：
- **对角线高亮**：关注相邻残基（局部特征提取）
- **行/列高亮**：某些残基是"全局信息枢纽"，在影响整条肽链的断裂行为
- **分散高亮**：关注长程相互作用
- **不同头的高亮区域不同**：说明多头注意力实现了功能分化

**好坏标准**：
- ✅ **好**：不同头显示不同的关注模式（功能分化），例如 Head 0 关注局部，Head 1 关注全局
- ❌ **差**：所有头的模式相似，说明多头注意力没有发挥作用

---

### 4.4 注意力热力图（attention_heatmap_layerX.png）

**图表类型**：热力图

**横轴**：Key Position（键位置）
**纵轴**：Query Position（查询位置）

**颜色编码**：`viridis` 色谱
- 深色：低注意力权重
- 亮色：高注意力权重

**指标解读**：
- **主对角线**：自注意力（每个节点对自身的关注）
- **对角线附近**：局部注意力（相邻残基的交互）
- **远离对角线**：长程注意力（远距离残基的交互）
- **稀疏分布**：模型学会了选择性关注

**好坏标准**：
- ✅ **好**：注意力分布与序列结构相关，存在明显的高亮斑块
- ❌ **差**：注意力分布随机或均匀，说明模型没有学到有意义的模式

---

## 5. 统计指标解读

### 5.1 CSV 文件字段说明

`statistical_analysis_results.csv` 包含以下字段：

| 字段名 | 含义 | 取值范围 | 说明 |
|-------|------|---------|------|
| `layer_index` | 注意力层索引 | 0, 1, ... | Layer 0 为浅层，Layer 1 为深层 |
| `head_index` | 注意力头索引 | 0, 1, ... 或 avg | 多头注意力中的具体头 |
| `sequence_length` | 序列长度 | 正整数 | 肽段中氨基酸的数量 |
| `num_broken_bonds` | 断裂键数量 | 非负整数 | 该样本中实际断裂的肽键数 |
| `num_intact_bonds` | 完整键数量 | 非负整数 | 该样本中保持完整的肽键数 |
| `broken_bond_attention_mean` | 断裂键平均注意力 | ≥0 | 模型对断裂键的平均关注度 |
| `broken_bond_attention_std` | 断裂键注意力标准差 | ≥0 | 关注度的离散程度 |
| `intact_bond_attention_mean` | 完整键平均注意力 | ≥0 | 模型对完整键的平均关注度 |
| `intact_bond_attention_std` | 完整键注意力标准差 | ≥0 | 关注度的离散程度 |
| `attention_difference` | 注意力差异 | 可正可负 | broken_mean - intact_mean |
| `correlation` | 相关系数 | [-1, 1] | 注意力与断裂标签的 Pearson 相关系数 |
| `sample_index` | 样本索引 | 正整数 | 在测试集中的位置 |

### 5.2 关键指标阈值

| 指标 | 优秀 | 良好 | 一般 | 较差 |
|-----|------|------|------|------|
| `correlation` | >0.5 | 0.3-0.5 | 0.1-0.3 | <0.1 |
| `attention_difference` | >0.1 | 0.05-0.1 | 0.01-0.05 | <0.01 |

---

## 6. 论文撰写建议

### 6.1 可解释性分析章节结构

```markdown
## Interpretability Analysis

To validate whether the DBond-GT model's predictions align with chemical intuition, 
we extracted and visualized the attention weights from the Graph Attention Network (GAT) layers.

### 6.1 Multi-head Attention Functional Specialization

Figure X presents the statistical analysis of attention patterns across 1000 test samples. 
As shown in (a), certain attention heads (e.g., Head 0, 2, 6) exhibit significantly higher 
attention weights on broken bonds compared to intact bonds, indicating their specialized 
role in detecting cleavage-prone peptide bonds. The attention-breakage correlation analysis 
(Figure Xc) reveals strong positive correlations (r > 0.4) for multiple heads, confirming 
that the model's attention mechanism effectively captures chemically meaningful patterns.

### 6.2 Hierarchical Feature Learning

Figure Y illustrates the attention patterns for a representative peptide sequence. 
In the shallow layer (Layer 0), attention weights are relatively uniformly distributed 
among adjacent residues, indicating basic local context aggregation. In contrast, the 
deep layer (Layer 1) shows highly sparse and specialized attention, with elevated weights 
(highlighted in red) precisely targeting specific, cleavage-prone residue combinations. 
This hierarchical pattern demonstrates the model's ability to progressively refine its 
focus from general local context to critical bond positions.
```

### 6.2 图表引用格式

- 主图：`Figure 5. Interpretability analysis of DBond-GT. (a) Attention on broken vs intact bonds...`
- 案例图：`Figure 6. Case study of peptide sequence K-S-O...B. (a) Layer 0 attention...`

---

## 7. 常见问题

### Q1: 为什么使用边级别注意力权重而不是节点级别？

**A**: GraphTransformer 的 GAT 层返回的是边级别的注意力权重 `[num_edges, num_heads]`，这更符合图注意力网络的本质。我们通过 `edge_index` 将其映射到节点对，构建节点级别的注意力矩阵用于可视化。

### Q2: 分层抽样有什么优势？

**A**: 分层抽样确保不同长度的序列都有代表性，避免短序列或长序列被忽略。这对于分析模型在不同序列长度下的性能很重要，也使统计结果更具代表性。

### Q3: 相关系数低怎么办？

**A**: 低相关系数可能表示：
1. 模型没有学到有效的注意力模式
2. 数据质量问题
3. 需要调整模型架构或超参数

建议检查数据质量、增加训练轮数或调整注意力头数。

### Q4: 案例研究样本如何选择？

**A**: 建议选择：
1. 包含典型断裂模式的序列
2. 不同长度的序列（短、中、长）
3. 断裂键数量适中的序列（不要全断或全不断）

---

## 8. 更新日志

| 日期 | 版本 | 更新内容 |
|-----|------|---------|
| 2026-05-12 | 1.0 | 初始版本，包含完整可解释性分析框架 |

---

**最后更新**: 2026-05-12
