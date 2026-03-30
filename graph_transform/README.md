# Graph Transformer - 蛋白质键级别二分类模型

## 项目概述

本项目实现了一个基于图神经网络（Graph Neural Network, GNN）的蛋白质键级别二分类模型，专门用于预测蛋白质二硫键的断裂位置。模型将蛋白质序列转换为图结构，通过图卷积网络（GCN）和图注意力网络（GAT）进行特征学习，最终对每个氨基酸残基之间的键（bond）进行二分类预测（断裂/未断裂）。

### 主要特点
- **图结构表示**：将蛋白质序列建模为图结构，节点表示氨基酸残基，边表示残基之间的关系
- **混合架构**：结合GCN和GAT的优势，捕获局部和全局信息
- **多源特征**：整合氨基酸序列、物理化学性质、质谱状态和环境信息
- **键级别预测**：对相邻残基之间的键进行细粒度预测
- **完整训练流程**：支持训练、验证、测试、检查点管理和可视化

## 技术架构

### 1. 模型架构

#### 1.1 整体结构
```
输入数据
    ↓
节点编码器 (NodeEncoder)
    ├─ 氨基酸嵌入 (64维)
    ├─ 位置嵌入 (32维)
    ├─ 物理化学性质嵌入 (32维)
    ├─ 状态变量编码 (32维)
    └─ 环境变量编码 (32维)
    ↓
边编码器 (EdgeEncoder)
    ├─ 边类型嵌入 (16维)
    └─ 距离嵌入 (16维)
    ↓
图神经网络层
    ├─ 3层残差GCN (ResidualGCNLayer)
    ├─ 2层GAT (GraphAttentionLayer)
    └─ 层归一化 + Dropout
    ↓
键级别预测头 (Bond Head)
    └─ 二分类输出
```

#### 1.2 模型层数
- **GCN层**：3层（带残差连接）
- **GAT层**：2层（多头注意力）
- **隐藏维度**：256
- **注意力头数**：8
- **Dropout率**：0.1

#### 1.3 关键组件

##### NodeEncoder（节点编码器）
- **氨基酸嵌入**：将20种氨基酸编码为64维向量
- **位置嵌入**：使用位置编码捕获序列位置信息（32维）
- **物理化学性质**：每个氨基酸的疏水性、电荷、极性、分子量（4维→32维）
- **状态变量**：电荷、肽段质量、强度（3维→32维）
- **环境变量**：碰撞能（NCE）、保留时间（RT）（2维→32维）

**状态变量归一化：**
```python
charge = charge * 0.1
pep_mass = pep_mass / 2000.0
intensity = log1p(clamp_min(intensity, 0)) / 20.0
```

**环境变量归一化：**
```python
nce = nce * 0.01
rt = rt * 0.01
```

##### EdgeEncoder（边编码器）
- **边类型**：sequence、distance、functional、long_range、global
- **距离嵌入**：编码残基间的序列距离（最大10）
- **边特征**：可选的原始边特征编码

##### 全局节点（Global Node）
- 整合样本级的状态和环境信息
- 提供全局上下文，增强模型对整体蛋白质结构的理解

##### GraphTransformer主模型
- **GCN层**：使用残差连接和层归一化
- **GAT层**：多头注意力机制，捕获长程依赖
- **键级别预测**：拼接相邻残基特征，通过MLP预测断裂概率

### 2. 图结构构建

#### 2.1 图构建策略
- **sequence**：基于序列邻接关系
- **distance**：基于空间距离（如果可用）
- **hybrid**：结合序列和距离信息
- **knowledge**：基于蛋白质结构知识

#### 2.2 边类型
1. **sequence**：序列邻接边（相邻残基）
2. **distance**：距离边（基于空间距离）
3. **functional**：功能相关边
4. **long_range**：长程边（间隔10的稀疏连接）
5. **global**：全局边（连接到全局节点）

#### 2.3 全局节点配置
```yaml
use_long_range_edges: true    # 开启稀疏长程边
long_range_stride: 10        # 间隔
long_range_hops: 1            # 每个节点连接几跳
use_global_node: true         # 开启全局虚拟节点
```

### 3. 损失函数

#### 3.1 主要损失函数
**二元交叉熵损失（BCEWithLogitsLoss）**
```python
loss = BCEWithLogitsLoss(predictions, targets)
```

#### 3.2 类别不平衡处理

##### Focal Loss
- **α**：0.25（平衡因子）
- **γ**：2.0（聚焦因子）
- **特点**：关注难分类样本，减少简单样本的权重

```python
BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
pt = torch.exp(-BCE_loss)
F_loss = α * (1 - pt)^γ * BCE_loss
```

##### 加权BCE
- **pos_weight**：自动计算正样本权重
- **计算公式**：pos_weight = neg / (pos + ε)
- **上限**：50.0（防止权重过大）

```python
pos_weight = neg / (pos + 1e-6)
pos_weight = clamp(pos_weight, max=50.0)
loss = BCEWithLogitsLoss(predictions, targets, pos_weight=pos_weight)
```

##### Dice Loss
- 用于优化Jaccard相似度
- **smooth**：1e-6（避免除零）

```python
intersection = (predictions * targets).sum()
dice = (2 * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
dice_loss = 1 - dice
```

#### 3.3 组合损失
可选的组合损失（BCE + Focal + Dice）：
```python
total_loss = 0.5 * bce + 0.3 * focal + 0.2 * dice
```

### 4. 训练方法

#### 4.1 优化器配置

##### AdamW（默认）
```yaml
optimizer:
  type: "adamw"
  betas: [0.9, 0.999]
  eps: 1e-8
  weight_decay: 0.0001
```

##### 其他优化器
- **Adam**：自适应矩估计
- **SGD**：带动量的随机梯度下降
- **RMSprop**：自适应学习率优化器

#### 4.2 学习率调度

##### 余弦退火（默认）
```yaml
training:
  scheduler: "cosine"
  warmup_epochs: 10
  min_lr: 0.00001
```

##### 阶梯衰减
```yaml
optimizer:
  step_size: 30
  gamma: 0.1
```

##### 指数衰减
```yaml
optimizer:
  gamma: 0.95
```

##### ReduceLROnPlateau
```yaml
training:
  scheduler: "plateau"
  patience: 10
  factor: 0.5
```

#### 4.3 训练参数
```yaml
training:
  epochs: 60
  batch_size: 512
  learning_rate: 0.0005
  weight_decay: 0.0001
  gradient_clip_norm: 1.0
  early_stopping: true
  patience: 10
  min_delta: 0.001
  validation_split: 0.2
  validation_interval: 1
```

#### 4.4 混合精度训练
```yaml
device:
  use_amp: true
  amp_backend: "native"  # 或 "apex"
```

#### 4.5 数据增强
```yaml
data:
  augmentation: true
  augmentation_prob: 0.3
  max_augmentation_trials: 3
```

#### 4.6 数据缓存
```yaml
data:
  cache_graphs: true
  cache_dir: "cache/graph_data"
```

### 5. 评估指标

#### 5.1 主要指标
- **准确率（Accuracy）**：整体预测正确的比例
- **精确率（Precision）**：预测为正样本中实际为正的比例
- **召回率（Recall）**：实际正样本中被正确预测的比例
- **F1分数**：精确率和召回率的调和平均
- **AUC**：ROC曲线下面积
- **Hamming Loss**：多标签平均汉明损失

#### 5.2 评估配置
```yaml
evaluation:
  metrics: ["accuracy", "precision", "recall", "f1", "auc", "hamming_loss"]
  threshold: 0.5
  threshold_strategy: "fixed"  # "fixed", "adaptive", "optimal"
  save_outputs: true
  output_pred_dir: "result/pred/graph_transform"
  output_metric_dir: "result/metric/graph_transform"
```

### 6. 使用说明

#### 6.1 环境要求
- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.0（GPU训练）
- 依赖包见 `requirements.txt`

#### 6.2 训练模型

##### 基础训练命令
```bash
python graph_transform/scripts/train_graph_model.py \
    --config graph_transform/config/default.yaml
```

##### 自定义训练参数
```bash
python graph_transform/scripts/train_graph_model.py \
    --config graph_transform/config/default.yaml \
    --epochs 100 \
    --batch_size 256 \
    --learning_rate 0.001 \
    --device cuda
```

##### 恢复训练
```bash
python graph_transform/scripts/train_graph_model.py \
    --config graph_transform/config/default.yaml \
    --resume checkpoints/graph_transform/checkpoint_epoch_30.pt
```

##### 指定随机种子
```bash
python graph_transform/scripts/train_graph_model.py \
    --config graph_transform/config/default.yaml \
    --seed 42
```

#### 6.3 评估模型

##### 基础评估命令
```bash
python graph_transform/scripts/evaluate_graph_model.py \
    --config graph_transform/config/default.yaml \
    --checkpoint checkpoints/graph_transform/best_model.pt
```

##### 自定义评估参数
```bash
python graph_transform/scripts/evaluate_graph_model.py \
    --config graph_transform/config/default.yaml \
    --checkpoint checkpoints/graph_transform/best_model.pt \
    --test_csv dataset/test.csv \
    --out_pred_csv result/predictions.csv \
    --out_metric_csv result/metrics.csv \
    --threshold 0.5
```

#### 6.4 查看训练日志
```bash
# 查看训练日志
tail -f logs/graph_transform/training.log

# 启动TensorBoard
tensorboard --logdir tensorboard/graph_transform
```

### 7. 配置文件说明

配置文件位于 `graph_transform/config/default.yaml`，包含以下主要部分：

#### 7.1 模型配置
```yaml
model:
  hidden_dim: 256              # 隐藏维度
  num_attention_heads: 8       # 注意力头数
  dropout: 0.1                 # Dropout率
  use_edge_features: true      # 是否使用边特征
  num_gcn_layers: 3            # GCN层数
  num_gat_layers: 2            # GAT层数
  max_seq_len: 100             # 最大序列长度
  aa_embedding_dim: 64         # 氨基酸嵌入维度
  position_embedding_dim: 32   # 位置嵌入维度
  physicochemical_dim: 32     # 物理化学性质维度
  edge_types: ["sequence", "distance", "functional", "long_range", "global"]
```

#### 7.2 训练配置
```yaml
training:
  epochs: 60                   # 训练轮数
  batch_size: 512              # 批次大小
  learning_rate: 0.0005         # 学习率
  weight_decay: 0.0001         # 权重衰减
  gradient_clip_norm: 1.0      # 梯度裁剪
  scheduler: "cosine"          # 学习率调度器
  warmup_epochs: 10            # 预热轮数
  early_stopping: true         # 早停
  patience: 10                 # 早停耐心值
```

#### 7.3 数据配置
```yaml
data:
  train_csv_path: "dataset/train.csv"
  test_csv_path: "dataset/test.csv"
  max_seq_len: 100
  graph_strategy: "distance"    # 图构建策略
  augmentation: false           # 数据增强
  cache_graphs: true           # 图缓存
  cache_dir: "cache/graph_data"
  num_workers: 4               # 数据加载线程数
  batch_size: 512
```

#### 7.4 损失函数配置
```yaml
loss:
  main_loss: "binary_cross_entropy"  # 主损失函数
  handle_imbalance: true             # 处理类别不平衡
  imbalance_strategy: "focal"       # 不平衡策略
  focal_loss_alpha: 0.25            # Focal Loss α
  focal_loss_gamma: 2.0              # Focal Loss γ
```

#### 7.5 设备配置
```yaml
device:
  auto_detect: true            # 自动检测设备
  device_type: "cuda"          # 设备类型
  gpu_id: 0                     # GPU ID
  use_amp: false                # 混合精度训练
```

#### 7.6 日志配置
```yaml
logging:
  level: "INFO"
  log_dir: "logs/graph_transform"
  log_file: "training.log"
  use_tensorboard: true
  tensorboard_log_dir: "tensorboard/graph_transform"
  save_training_curves: true
```

#### 7.7 调试配置
```yaml
debug:
  debug_mode: false
  check_gradients: false
  check_nan_inf: true
  log_grad_norm: true
  profile_time: false
  profile_memory: false
```

### 8. 技术路线

#### 8.1 完整流程
```
1. 数据预处理
   ├─ 读取CSV文件
   ├─ 解析序列和标签
   └─ 提取特征（氨基酸、状态、环境）

2. 图构建
   ├─ 节点：每个氨基酸残基作为一个节点
   ├─ 边：根据策略构建边（sequence/distance/hybrid/knowledge）
   ├─ 节点特征：编码氨基酸、位置、物理化学性质
   └─ 边特征：编码边类型、距离

3. 特征编码
   ├─ 节点编码器：整合多源节点特征
   ├─ 边编码器：编码边特征
   └─ 全局节点：整合样本级信息

4. 图神经网络
   ├─ GCN层：局部特征聚合
   ├─ GAT层：全局注意力机制
   └─ 残差连接 + 层归一化

5. 键级别预测
   ├─ 提取相邻残基特征
   ├─ 拼接特征向量
   └─ MLP二分类输出

6. 训练与优化
   ├─ 前向传播计算损失
   ├─ 反向传播更新参数
   ├─ 学习率调度
   └─ 早停机制

7. 评估与可视化
   ├─ 计算评估指标
   ├─ 保存预测结果
   ├─ TensorBoard可视化
   └─ 生成训练曲线
```

#### 8.2 关键技术

##### 数据优化
- **图缓存**：避免重复构图，提升训练速度
- **混合精度训练**：降低显存占用，加速训练
- **数据增强**：提升模型泛化能力
- **多线程加载**：并行数据预处理

##### 模型优化
- **残差连接**：缓解梯度消失，训练更深的网络
- **层归一化**：稳定训练过程
- **Dropout**：防止过拟合
- **全局节点**：提供全局上下文信息
- **多头注意力**：捕获不同类型的关系

##### 训练优化
- **梯度裁剪**：防止梯度爆炸
- **学习率预热**：稳定训练初期
- **早停机制**：避免过拟合
- **检查点管理**：支持断点续训
- **TensorBoard**：实时监控训练过程

### 9. 输出文件说明

#### 9.1 检查点文件
```
checkpoints/graph_transform/
├── best_model.pt              # 最佳模型
├── checkpoint_epoch_10.pt     # 定期检查点
├── checkpoint_epoch_20.pt
└── ...
```

#### 9.2 训练日志
```
logs/graph_transform/
├── training.log               # 训练日志
└── diagnostics/               # 诊断信息（如果启用调试）
```

#### 9.3 训练曲线
```
checkpoints/graph_transform/plots/
├── loss_curve.png             # 损失曲线
├── f1_curve.png               # F1曲线
├── learning_rate_curve.png    # 学习率曲线
├── grad_norm_curve.png        # 梯度范数曲线
└── precision_recall_curve.png # 精确率-召回率曲线
```

#### 9.4 TensorBoard日志
```
tensorboard/graph_transform/
├── train/                     # 训练数据
├── val/                       # 验证数据
└── test/                      # 测试数据
```

#### 9.5 评估结果
```
result/
├── metric/
│   └── graph_transform/
│       ├── latest_metric.csv          # 最新指标
│       └── 20260326_123456_..._test_metric.csv  # 归档指标
└── pred/
    └── graph_transform/
        ├── latest.pred.csv            # 最新预测
        └── 20260326_123456_..._test.pred.csv    # 归档预测
```

### 10. 常见问题

#### 10.1 显存不足
- 减小 `batch_size`
- 减小 `hidden_dim`
- 减少模型层数（`num_gcn_layers`, `num_gat_layers`）
- 启用混合精度训练（`use_amp: true`）

#### 10.2 训练速度慢
- 启用图缓存（`cache_graphs: true`）
- 增加 `num_workers`
- 启用混合精度训练
- 减小 `max_seq_len`

#### 10.3 模型过拟合
- 增加 `dropout`
- 增加 `weight_decay`
- 启用数据增强（`augmentation: true`）
- 减小模型复杂度
- 增加训练数据

#### 10.4 类别不平衡严重
- 使用 Focal Loss
- 使用加权 BCE
- 调整 `focal_loss_alpha` 和 `focal_loss_gamma`

### 11. 项目结构

```
graph_transform/
├── config/
│   └── default.yaml              # 配置文件
├── data/
│   ├── graph_dataset.py          # 图数据集
│   ├── optimized_graph_dataset.py # 优化的图数据集（缓存）
│   ├── graph_builder.py          # 图构建器
│   ├── augmentation.py           # 数据增强
│   └── preprocessing.py          # 数据预处理
├── models/
│   ├── graph_transformer.py      # 主模型
│   ├── attention_layers.py       # 注意力层
│   ├── gcn_layers.py            # GCN层
│   └── utils.py                 # 工具函数
├── training/
│   ├── trainer.py               # 训练器
│   ├── loss_functions.py        # 损失函数
│   └── metrics.py               # 训练指标
├── evaluation/
│   ├── evaluator.py             # 评估器
│   └── metrics.py               # 评估指标
├── scripts/
│   ├── train_graph_model.py      # 训练脚本
│   ├── evaluate_graph_model.py  # 评估脚本
│   └── visualize_sample_graph.py # 可视化脚本
└── README.md                    # 本文档
```

### 12. 引用

如果您在研究中使用了本代码，请引用：

```bibtex
@software{graph_transformer_2024,
  title={Graph Transformer for Protein Bond-level Classification},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/DBond}
}
```

### 13. 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

### 14. 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue：https://github.com/yourusername/DBond/issues
- 邮箱：your.email@example.com

---

**最后更新**：2026年3月26日