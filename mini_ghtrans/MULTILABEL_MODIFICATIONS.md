# 多标签数据格式支持修改总结

本文档总结了为支持多标签训练数据集格式而对graph_transformer模型所做的所有修改。

## 修改概述

根据用户提供的数据格式示例，我们修改了以下关键组件以支持多标签训练：

```
name,seq,charge,pep_mass,intensity,nce,scan_num,rt,fbr,tb,fb,mb,true_multi
YP-092,STKABDFYPQGTATSDAAEFGYED,2,1279.057610968719,40333286.38721,30,2,0.82725786,0.9130434782608695,23,21,1;23;,0;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;0
YP-092,STKABDFYPQGTATSDAAEFGYED,4,640.031853431055,10812201.61865,30,3,0.90697998,0.8695652173913043,23,20,1;18;23;,0;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;0;1;1;1;1;0
```

## 主要修改内容

### 1. 数据集处理 (`graph_transform/data/graph_dataset.py`)

#### 修改的函数：
- `_parse_labels()`: 解析分号分隔的多标签字符串
- `_prepare_labels()`: 创建多标签二进制矩阵

#### 关键变化：
```python
def _parse_labels(self, label_str: str) -> List[int]:
    """解析多标签字符串"""
    if pd.isna(label_str) or label_str == '':
        return []
    
    # 解析分号分隔的多标签
    labels = [int(x.strip()) for x in str(label_str).split(';') if x.strip().isdigit()]
    return labels

def _prepare_labels(self, labels: List[int], seq_len: int) -> torch.Tensor:
    """准备多标签张量"""
    # 创建多标签二进制矩阵 [seq_len, num_classes]
    multi_hot = torch.zeros(seq_len, self.config.num_classes)
    
    # 处理每个位置的标签
    for i, label in enumerate(labels):
        if i >= seq_len:
            break
        if 0 <= label < self.config.num_classes:
            multi_hot[i, label] = 1
    
    # 转换为float类型
    label_tensor = multi_hot.float()
    
    return label_tensor
```

### 2. 模型架构 (`graph_transform/models/graph_transformer.py`)

#### 修改的类：
- `MultiLabelHead`: 支持序列级别的多标签预测

#### 关键变化：
```python
def forward(self, node_features: torch.Tensor, 
            batch_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    前向传播
    
    Args:
        node_features: 节点特征 [num_nodes, hidden_dim] 或 [batch_size, seq_len, hidden_dim]
        batch_indices: 批次索引 [num_nodes]
        
    Returns:
        torch.Tensor: 预测结果 [batch_size, seq_len, num_classes] 或 [batch_size, num_classes]
    """
    # 支持序列级别的多标签预测
    if batch_indices is not None:
        # 处理图级别的节点特征重构
        if node_features.dim() == 2:
            batch_size = batch_indices.max().item() + 1
            seq_len = self._estimate_seq_len(batch_indices, batch_size)
            
            # 重构为 [batch_size, seq_len, hidden_dim]
            reshaped_features = torch.zeros(batch_size, seq_len, node_features.size(1), 
                                          device=node_features.device)
            
            # 填充特征
            for i in range(batch_size):
                mask = (batch_indices == i)
                seq_features = node_features[mask]
                actual_len = min(seq_features.size(0), seq_len)
                reshaped_features[i, :actual_len] = seq_features[:actual_len]
            
            node_features = reshaped_features
    
    # 多标签预测
    predictions = self.mlp(node_features)
    
    return predictions
```

### 3. 训练模块 (`graph_transform/training/`)

#### 新增的模块：
- `loss_functions.py`: 多标签损失函数
- `metrics.py`: 多标签评估指标
- `trainer.py`: 训练器实现

#### 损失函数支持：
```python
class MultiLabelLoss(nn.Module):
    """多标签分类损失函数"""
    
    def __init__(self, config: Dict[str, Any]):
        super(MultiLabelLoss, self).__init__()
        
        self.main_loss = config.get('main_loss', 'binary_cross_entropy')
        self.use_auxiliary_losses = config.get('use_auxiliary_losses', False)
        self.handle_imbalance = config.get('handle_imbalance', False)
        
        # 支持多种损失函数
        if self.main_loss == 'binary_cross_entropy':
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        elif self.main_loss == 'focal':
            self.criterion = FocalLoss(config)
        elif self.main_loss == 'dice':
            self.criterion = DiceLoss(config)
```

### 4. 评估模块 (`graph_transform/evaluation/`)

#### 新增的模块：
- `evaluator.py`: 模型评估器
- `metrics.py`: 评估指标计算

#### 评估指标支持：
```python
class MultiLabelMetrics:
    """多标签分类评估指标"""
    
    def compute(self) -> Dict[str, float]:
        """计算所有指标"""
        # 支持多种多标签指标
        metrics = {
            'accuracy': accuracy_score(targets, binary_predictions),
            'precision': precision_score(targets, binary_predictions, average='macro'),
            'recall': recall_score(targets, binary_predictions, average='macro'),
            'f1': f1_score(targets, binary_predictions, average='macro'),
            'hamming_loss': hamming_loss(targets, binary_predictions),
            'auc_macro': roc_auc_score(targets, predictions, average='macro'),
            'auc_micro': roc_auc_score(targets, predictions, average='micro'),
        }
```

### 5. 配置文件 (`graph_transform/config/default.yaml`)

#### 关键配置更新：
```yaml
# 输出配置
num_classes: 24  # 根据数据格式中的标签范围设置

# 损失函数配置
loss:
  main_loss: "binary_cross_entropy"  # "binary_cross_entropy", "focal", "dice"
  use_auxiliary_losses: true
  auxiliary_loss_weights:
    focal: 0.5
    dice: 0.3
    l2_regularization: 0.1
  handle_imbalance: true
  imbalance_strategy: "focal"  # "focal", "weighted", "oversample"

# 评估配置
evaluation:
  metrics: ["accuracy", "precision", "recall", "f1", "auc", "hamming_loss"]
  threshold: 0.5
  threshold_strategy: "fixed"  # "fixed", "adaptive", "optimal"
```

### 6. 数据处理支持模块

#### 新增模块：
- `preprocessing.py`: 数据预处理
- `augmentation.py`: 数据增强
- `test_multilabel_data.py`: 多标签格式测试脚本

#### 数据预处理功能：
```python
class SequencePreprocessor:
    """序列预处理器"""
    
    def preprocess_sequence(self, sequence: str) -> str:
        """预处理氨基酸序列"""
        # 转换为大写、移除非标准字符、截断长度
        sequence = sequence.upper()
        sequence = re.sub(f'[^{self.alphabet}]', '', sequence)
        if len(sequence) > self.max_seq_len:
            sequence = sequence[:self.max_seq_len]
        return sequence
```

## 数据格式要求

### 必需的CSV列：
1. `name`: 样本名称
2. `seq`: 氨基酸序列
3. `charge`: 电荷
4. `pep_mass`: 肽段质量
5. `intensity`: 强度
6. `nce`: 碰撞能量
7. `scan_num`: 扫描号
8. `rt`: 保留时间
9. `fbr`: 碎裂比例
10. `tb`: 总断裂数
11. `fb`: 前向断裂
12. `mb`: 中间断裂
13. `true_multi`: 多标签目标（分号分隔）

### 多标签格式说明：
- 使用分号 (`;`) 作为分隔符
- 每个数字代表一个类别标签
- 位置i对应序列中位置i的标签
- 示例：`'0;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;0'`

### 序列处理规则：
- 标准氨基酸：A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
- 最大序列长度：可配置（默认：100）
- 超过长度的序列将被截断

## 使用方法

### 1. 准备数据
确保您的CSV文件包含所有必需列，并且`true_multi`列使用分号分隔的多标签格式。

### 2. 配置模型
根据您的数据调整配置文件中的参数：
- `num_classes`: 根据您的标签范围设置
- `max_seq_len`: 根据您的序列长度设置
- 损失函数和评估指标配置

### 3. 训练模型
```bash
cd graph_transform
python scripts/train_graph_model.py --config config/default.yaml
```

### 4. 测试数据格式
```bash
cd graph_transform
python scripts/test_multilabel_data.py
```

## 支持的功能

### 多标签损失函数：
- 二元交叉熵损失
- Focal Loss（处理类别不平衡）
- Dice Loss
- 组合损失函数

### 多标签评估指标：
- Accuracy, Precision, Recall, F1-score
- AUC-ROC (macro/micro)
- Hamming Loss
- Jaccard Score
- 排序指标

### 数据增强：
- 氨基酸替换
- 序列截断
- 噪声注入
- 序列反转

### 模型特性：
- 图神经网络架构
- 注意力机制
- 多头注意力
- 残差连接
- 层归一化

## 注意事项

1. **内存使用**: 多标签数据可能需要更多内存，建议适当调整批大小
2. **类别不平衡**: 使用Focal Loss或加权损失函数处理
3. **阈值选择**: 支持固定、自适应和最优阈值策略
4. **序列长度**: 确保配置的最大序列长度适合您的数据

## 测试验证

运行测试脚本以验证修改是否正确：
```bash
python scripts/test_multilabel_data.py
```

该脚本将：
1. 创建示例多标签数据
2. 测试数据加载和预处理
3. 验证损失函数计算
4. 检查标签解析功能

## 完成的文件列表

### 修改的文件：
- `graph_transform/data/graph_dataset.py`
- `graph_transform/models/graph_transformer.py`
- `graph_transform/scripts/train_graph_model.py`
- `graph_transform/config/default.yaml`
- `graph_transform/models/utils.py`

### 新增的文件：
- `graph_transform/training/__init__.py`
- `graph_transform/training/loss_functions.py`
- `graph_transform/training/metrics.py`
- `graph_transform/training/trainer.py`
- `graph_transform/evaluation/__init__.py`
- `graph_transform/evaluation/evaluator.py`
- `graph_transform/evaluation/metrics.py`
- `graph_transform/data/preprocessing.py`
- `graph_transform/data/augmentation.py`
- `graph_transform/scripts/test_multilabel_data.py`
- `graph_transform/MULTILABEL_MODIFICATIONS.md`

所有修改都已完成，系统现在支持用户提供的多标签训练数据格式。
