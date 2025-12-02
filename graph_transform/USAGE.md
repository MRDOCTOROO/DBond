# 图神经网络多标签分类系统使用指南

本指南详细介绍了如何使用图神经网络多标签分类系统进行蛋白质序列分析。

## 目录

1. [环境设置](#环境设置)
2. [快速开始](#快速开始)
3. [配置说明](#配置说明)
4. [训练模型](#训练模型)
5. [模型评估](#模型评估)
6. [推理预测](#推理预测)
7. [高级功能](#高级功能)
8. [故障排除](#故障排除)

## 环境设置

### 系统要求

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (可选，用于GPU加速)
- 8GB+ RAM (推荐16GB+)
- 4GB+ GPU显存 (推荐8GB+)

### 安装依赖

```bash
# 安装PyTorch (根据你的CUDA版本选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install pandas numpy scipy scikit-learn
pip install tqdm tensorboard wandb
pip install networkx matplotlib seaborn
pip install pyyaml argparse logging
```

### 项目结构

```
graph_transform/
├── README.md                    # 项目说明
├── USAGE.md                     # 使用指南 (本文件)
├── config/                      # 配置文件
│   └── default.yaml            # 默认配置
├── models/                      # 模型实现
│   ├── __init__.py
│   ├── graph_transformer.py   # 主模型
│   ├── gcn_layers.py          # 图卷积层
│   ├── attention_layers.py    # 注意力层
│   └── utils.py              # 工具函数
├── data/                        # 数据处理
│   ├── __init__.py
│   ├── graph_dataset.py       # 数据集类
│   └── graph_builder.py      # 图构建器
├── scripts/                     # 脚本
│   └── train_graph_model.py  # 训练脚本
└── training/                    # 训练模块
    ├── __init__.py
    ├── trainer.py            # 训练器
    └── loss.py              # 损失函数
```

## 快速开始

### 1. 准备数据

确保你的数据格式如下：

```csv
seq,charge,pep_mass,nce,rt,fbr,true_multi
ACDEFGHIK,2,1234.5,30.0,12.3,1;2;3;1
KLMNPQRST,1,987.6,25.0,15.7,2;1;0;2
...
```

### 2. 基础训练

```bash
# 使用默认配置训练
cd graph_transform
python scripts/train_graph_model.py --config config/default.yaml

# 指定GPU
python scripts/train_graph_model.py --config config/default.yaml --device cuda

# 自定义参数
python scripts/train_graph_model.py --config config/default.yaml \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 0.001
```

### 3. 监控训练

训练过程中会自动生成TensorBoard日志：

```bash
# 启动TensorBoard
tensorboard --logdir logs/graph_transform

# 在浏览器中访问 http://localhost:6006
```

## 配置说明

### 模型配置

```yaml
model:
  hidden_dim: 256              # 隐藏层维度
  num_attention_heads: 8        # 注意力头数
  dropout: 0.1                 # Dropout率
  num_gcn_layers: 3            # GCN层数
  num_gat_layers: 2            # GAT层数
  max_seq_len: 100             # 最大序列长度
  num_classes: 20              # 分类类别数
```

### 训练配置

```yaml
training:
  epochs: 100                  # 训练轮数
  batch_size: 32               # 批大小
  learning_rate: 0.001         # 学习率
  weight_decay: 0.0001        # 权重衰减
  early_stopping: true          # 早停
  patience: 15                 # 早停耐心值
```

### 数据配置

```yaml
data:
  train_csv_path: "dataset/dbond_m.train.shuffle.csv"
  test_csv_path: "dataset/dbond_m.test.csv"
  max_seq_len: 100
  graph_strategy: "distance"     # sequence, distance, hybrid, knowledge
  augmentation: true
  cache_graphs: false
```

### 图构建策略

- **sequence**: 基于序列连接构建图
- **distance**: 基于预测距离构建图
- **hybrid**: 混合序列和距离信息
- **knowledge**: 基于蛋白质结构知识构建图

## 训练模型

### 完整训练流程

1. **数据预处理**
   ```python
   from data import GraphDataset
   from models.utils import ModelConfig
   
   config = ModelConfig()
   dataset = GraphDataset(
       csv_path="dataset/dbond_m.train.shuffle.csv",
       config=config,
       max_seq_len=100,
       graph_strategy="distance"
   )
   ```

2. **模型初始化**
   ```python
   from models import GraphTransformer
   
   model = GraphTransformer(config)
   model = model.to(device)
   ```

3. **训练配置**
   ```python
   from training import Trainer, MultiLabelLoss
   
   criterion = MultiLabelLoss(config)
   optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
   
   trainer = Trainer(
       model=model,
       optimizer=optimizer,
       criterion=criterion,
       device=device,
       config=config
   )
   ```

4. **开始训练**
   ```python
   # 训练一个epoch
   metrics = trainer.train_epoch(train_loader, epoch)
   
   # 验证
   val_metrics = trainer.validate_epoch(val_loader)
   ```

### 命令行训练

```bash
# 基础训练
python scripts/train_graph_model.py --config config/default.yaml

# 恢复训练
python scripts/train_graph_model.py \
    --config config/default.yaml \
    --resume checkpoints/graph_transform/best_model.pt

# 指定随机种子
python scripts/train_graph_model.py \
    --config config/default.yaml \
    --seed 42
```

### 训练技巧

1. **学习率调度**
   ```yaml
   training:
     scheduler: "cosine"          # cosine, step, exponential, plateau
     warmup_epochs: 10
     min_lr: 0.00001
   ```

2. **梯度裁剪**
   ```yaml
   training:
     gradient_clip_norm: 1.0
   ```

3. **混合精度训练**
   ```yaml
   device:
     use_amp: true
   ```

## 模型评估

### 评估指标

系统支持多种多标签分类指标：

- **准确率** (Accuracy)
- **精确率** (Precision)
- **召回率** (Recall)
- **F1分数** (F1-Score)
- **AUC** (Area Under Curve)
- **汉明损失** (Hamming Loss)

### 运行评估

```python
from evaluation import Evaluator

evaluator = Evaluator(model, device, config)
metrics = evaluator.evaluate(test_loader)

print(f"Test F1: {metrics['f1']:.4f}")
print(f"Test AUC: {metrics['auc']:.4f}")
```

### 批量评估脚本

```bash
# 评估单个模型
python scripts/evaluate_model.py \
    --model_path checkpoints/best_model.pt \
    --data_path dataset/test.csv \
    --config config/default.yaml

# 比较多个模型
python scripts/compare_models.py \
    --model_dir checkpoints/ \
    --data_path dataset/test.csv
```

## 推理预测

### 单样本预测

```python
from models import GraphTransformer
from data import SequenceGraphBuilder

# 加载模型
model = GraphTransformer(config)
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

# 构建图
graph_builder = SequenceGraphBuilder(config)
sequence = "ACDEFGHIKLMNPQRSTVWY"
env_vars = {'charge': 2, 'pep_mass': 1234.5, 'nce': 30.0, 'rt': 12.3, 'fbr': 0.8}

graph_data = graph_builder.build_graph(sequence, env_vars, 'distance')

# 预测
with torch.no_grad():
    batch_data = {k: v.unsqueeze(0) for k, v in graph_data.items()}
    predictions = model(batch_data)
    probabilities = torch.sigmoid(predictions)
    
# 获取预测标签
threshold = 0.5
predicted_labels = (probabilities > threshold).float()
```

### 批量推理

```python
from data import GraphDataset, GraphDataLoader

# 创建数据集
dataset = GraphDataset(csv_path="test.csv", config=config)
dataloader = GraphDataLoader(dataset, batch_size=32, shuffle=False)

# 批量预测
all_predictions = []
all_probabilities = []

model.eval()
with torch.no_grad():
    for batch in dataloader:
        predictions = model(batch)
        probabilities = torch.sigmoid(predictions)
        
        all_probabilities.append(probabilities.cpu())
        all_predictions.append((probabilities > 0.5).float().cpu())

# 合并结果
final_probabilities = torch.cat(all_probabilities, dim=0)
final_predictions = torch.cat(all_predictions, dim=0)
```

## 高级功能

### 数据增强

```yaml
data:
  augmentation: true
  augmentation_prob: 0.3
  max_augmentation_trials: 3
```

支持的数据增强策略：
- 氨基酸替换
- 序列截断
- 噪声注入
- 图扰动

### 模型集成

```yaml
advanced:
  use_ensemble: true
  ensemble_methods: ["voting", "averaging", "stacking"]
```

### 超参数优化

```python
from training import HyperparameterSearch

# 网格搜索
search_config = {
    'learning_rate': [0.001, 0.0001, 0.00001],
    'batch_size': [16, 32, 64],
    'hidden_dim': [128, 256, 512]
}

search = HyperparameterSearch(config, search_config)
best_params = search.run('grid')
```

### 分布式训练

```bash
# 单机多卡训练
torchrun --nproc_per_node=4 scripts/train_graph_model.py \
    --config config/default.yaml \
    --distributed

# 多机训练
torchrun --nnodes=2 --node_rank=0 --master_addr="192.168.1.100" \
    --master_port=1234 --nproc_per_node=4 scripts/train_graph_model.py \
    --config config/default.yaml
```

### 模型量化

```python
# 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# 静态量化
model.eval()
quantized_model = torch.quantization.quantize_static(
    model, calibration_data, {nn.Linear}, dtype=torch.qint8
)
```

## 故障排除

### 常见问题

#### 1. 内存不足

**问题**: CUDA out of memory

**解决方案**:
```yaml
training:
  batch_size: 16              # 减小批大小
data:
  max_seq_len: 50            # 减小序列长度
model:
  hidden_dim: 128            # 减小模型大小
```

#### 2. 训练速度慢

**问题**: 训练速度过慢

**解决方案**:
```yaml
data:
  num_workers: 8             # 增加数据加载进程
  cache_graphs: true          # 缓存图结构
device:
  use_amp: true              # 使用混合精度
performance:
  compile_model: true         # 编译模型 (PyTorch 2.0+)
```

#### 3. 过拟合

**问题**: 验证集性能下降

**解决方案**:
```yaml
training:
  dropout: 0.3               # 增加dropout
  weight_decay: 0.001         # 增加权重衰减
  early_stopping: true         # 启用早停
data:
  augmentation: true           # 启用数据增强
```

#### 4. 类别不平衡

**问题**: 某些类别预测效果差

**解决方案**:
```yaml
loss:
  main_loss: "focal"          # 使用Focal Loss
  handle_imbalance: true       # 处理不平衡
  focal_loss_alpha: 0.25
  focal_loss_gamma: 2.0
```

### 调试模式

```yaml
debug:
  debug_mode: true             # 启用调试模式
  check_gradients: true        # 检查梯度
  check_nan_inf: true          # 检查NaN/Inf
  save_attention_weights: true  # 保存注意力权重
```

### 性能分析

```python
# 内存分析
from models.utils import ModelProfiler

profiler = ModelProfiler()
stats = profiler.profile_model(model, sample_data, device)
print(f"Model size: {stats['model_size_mb']:.2f} MB")
print(f"Peak memory: {stats['peak_memory_mb']:.2f} MB")
```

### 日志分析

```bash
# 查看训练日志
tail -f logs/graph_transform/training.log

# 搜索错误
grep -i error logs/graph_transform/training.log

# 分析损失变化
grep "Train - Loss" logs/graph_transform/training.log | tail -20
```

## 最佳实践

1. **数据准备**
   - 确保数据质量，去除异常值
   - 进行适当的数据归一化
   - 考虑类别不平衡问题

2. **模型选择**
   - 从小模型开始，逐步增加复杂度
   - 使用验证集调整超参数
   - 考虑模型复杂度与性能的平衡

3. **训练策略**
   - 使用学习率预热和衰减
   - 启用梯度裁剪防止梯度爆炸
   - 定期保存检查点

4. **性能优化**
   - 使用GPU加速训练
   - 启用混合精度训练
   - 考虑模型并行或数据并行

5. **结果分析**
   - 分析注意力权重理解模型决策
   - 使用多种评估指标
   - 可视化训练过程和结果

## 更多资源

- [PyTorch官方文档](https://pytorch.org/docs/)
- [PyG (PyTorch Geometric)](https://pytorch-geometric.readthedocs.io/)
- [TensorBoard教程](https://www.tensorflow.org/tensorboard)
- [多标签分类综述](https://arxiv.org/abs/2006.13509)

## 技术支持

如果遇到问题，请：

1. 查看本文档的故障排除部分
2. 检查配置文件是否正确
3. 查看日志文件获取详细错误信息
4. 在GitHub Issues中提交问题和相关日志

---

**更新日期**: 2024年12月  
**版本**: v1.0  
**作者**: Graph Transform Team
