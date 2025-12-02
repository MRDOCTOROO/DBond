# DBond2 - 深度学习模型用于二硫键断裂预测

DBond2是一个基于深度学习的蛋白质组学（Protein Group Learning, PGL）项目，专门用于预测蛋白质中的二硫键（Disulfide Bond, DDBond）断裂位点。该项目使用Transformer架构来同时预测序列中的断裂位点（位置和类型）和环境变量。

## 项目概述

本项目提供了两个主要的深度学习模型：

- **DBond-s (DBond Single-label)**: 单标签模型，适用于单一断裂类型预测
- **DBond-m (DBond Multi-label)**: 多标签模型，能够同时预测多个断裂位置和类型

两个模型都基于Transformer架构，能够：
- 预测蛋白质序列中所有可能的二硫键断裂位置
- DBond-s：识别单一类型的断裂（如：肽键断裂）
- DBond-m：同时识别多种断裂类型（如：二硫键桥、肽键断裂、二硫键断裂）
- 考虑环境变量对断裂的影响

## 项目结构

```
DBond2/
├── LICENSE                          # 许可证文件
├── README.md                        # 项目说明文档
├── dockerfile                       # Docker构建文件
├── .gitignore                       # Git忽略文件
├── 核心模型文件/
│   ├── dbond_s.py                   # 单标签DBond模型架构定义
│   ├── dbond_m.py                   # 多标签DBond模型架构定义
│   ├── data_utils_dbond_s.py        # 单标签模型数据处理工具
│   └── data_utils_dbond_m.py        # 多标签模型数据处理工具
├── 训练和评估脚本/
│   ├── train.dbond_s.py             # 单标签模型训练脚本
│   ├── train.dbond_m.py             # 多标签模型训练脚本
│   ├── evaluate.dbond_s.py          # 单标签模型评估脚本
│   └── evaluate.dbond_m.py          # 多标签模型评估脚本
├── 配置文件/
│   ├── dbond_s_config/default.yaml  # 单标签模型配置文件
│   └── dbond_m_config/default.yaml  # 多标签模型配置文件
├── 数据集/
│   ├── dbond_s.train.shuffle.csv    # 单标签模型训练数据
│   ├── dbond_s.test.csv             # 单标签模型测试数据
│   ├── dbond_m.train.shuffle.csv    # 多标签模型训练数据
│   ├── dbond_m.test.csv             # 多标签模型测试数据
│   └── dataset.fbr.csv              # 格式结合断裂位点信息数据
├── PBCLA工具/
│   ├── pbcla.py                     # 主数据处理和转换工具
│   ├── mgf2csv.dbond_s.py           # 单标签模型MGF到CSV转换
│   ├── mgf2csv.dbond_m.py           # 多标签模型MGF到CSV转换
│   ├── utils.py                     # 工具函数
│   └── mgf_dataset/                 # 示例数据集
│       ├── example.mgf              # 示例MGF文件
│       ├── example.csv              # 示例CSV文件
│       ├── example.multi.csv       # 示例多标签CSV文件
│       └── example_out.mgf          # 示例输出MGF文件
├── 评估指标/
│   └── multi_label_metrics.py       # 多标签分类评估指标计算
├── 输出目录/
│   ├── best_model/                  # 最佳模型权重存储
│   │   ├── dbond_s/                 # 单标签模型最佳权重
│   │   └── dbond_m/                 # 多标签模型最佳权重
│   ├── checkpoint/                  # 模型检查点
│   │   ├── dbond_s/                 # 单标签模型检查点
│   │   └── dbond_m/                 # 多标签模型检查点
│   ├── result/                      # 评估结果存储
│   │   ├── dbond_s/                 # 单标签模型结果
│   │   ├── dbond_m/                 # 多标签模型结果
│   │   └── multi_label_metric/      # 多标签评估结果
│   └── tensorboard/                 # TensorBoard日志
│       ├── dbond_s/                 # 单标签模型日志
│       └── dbond_m/                 # 多标签模型日志
```

## 核心文件功能说明

### 模型架构文件
- **`dbond_s.py`**: 单标签DBond模型实现，包含Transformer编码器、序列嵌入层和断裂预测头
- **`dbond_m.py`**: 多标签DBond模型实现，具有更大的模型容量和更高的预测精度

### 数据处理文件
- **`data_utils_dbond_s.py`**: 单标签模型的数据加载器，支持批量处理、序列填充和多标签编码
- **`data_utils_dbond_m.py`**: 多标签模型的数据加载器，处理更复杂的特征和环境变量

### 训练脚本
- **`train.dbond_s.py`**: 单标签模型训练主程序，支持GPU训练、学习率调度和模型保存
- **`train.dbond_m.py`**: 多标签模型训练主程序，包含完整的训练流程和验证

### 评估脚本
- **`evaluate.dbond_s.py`**: 单标签模型评估程序，计算多标签分类指标
- **`evaluate.dbond_m.py`**: 多标签模型评估程序，提供详细的性能分析

### 数据转换工具
- **`PBCLA/pbcla.py`**: 主数据处理工具，支持多种质谱数据格式的转换
- **`PBCLA/mgf2csv.dbond_s.py`**: 将MGF格式转换为单标签模型所需的CSV格式
- **`PBCLA/mgf2csv.dbond_m.py`**: 将MGF格式转换为多标签模型所需的CSV格式
- **`PBCLA/utils.py`**: 数据转换相关的工具函数

### 配置文件
- **`dbond_s_config/default.yaml`**: 单标签模型的所有配置参数（模型架构、训练超参数、数据路径等）
- **`dbond_m_config/default.yaml`**: 多标签模型的所有配置参数

### 评估指标
- **`multi_label_metrics.py`**: 多标签分类评估指标，包括准确率、精确率、召回率、F1分数等

## 功能特性

### 模型架构特点
- **Transformer编码器**: 使用自注意力机制捕获序列中的长距离依赖关系
- **多标签分类**: 同时预测多个断裂位点和类型
- **批处理支持**: 高效的批量数据处理
- **可配置架构**: 支持不同的模型大小和参数配置
- **自定义字母表**: 支持20种标准氨基酸的编码

### 输入特征
- **蛋白质序列**: 氨基酸序列（字符串格式）
- **电荷状态**: 肽段的电荷数量（整数）
- **质荷质量**: 肽段的质荷比（浮点数）
- **碰撞能量**: 质谱分析中的碰撞能量（整数）
- **扫描编号**: 质谱扫描的标识符（整数）
- **保留时间**: 液相色谱保留时间（浮点数）
- **环境变量**: pH值、温度等实验条件
- **格式结合率**: Fragment Bonding Ratio（浮点数）

### 输出预测
- **断裂位点位置**: 预测序列中所有可能的断裂位置
- **断裂类型**: 识别不同类型的断裂（二硫键桥、肽键断裂等）
- **环境变量影响**: 预测环境条件对断裂概率的影响

## 数据格式说明

### 输入数据格式
训练和测试数据CSV文件包含以下列：

| 列名 | 类型 | 描述 |
|------|------|------|
| name | 字符串 | 样本唯一标识符 |
| seq | 字符串 | 氨基酸序列（如：ACDEFGHIK） |
| charge | 整数 | 电荷状态（通常为2-4） |
| pep_mass | 浮点数 | 质荷质量 |
| nce | 整数 | 碰撞能量 |
| scan_num | 整数 | 扫描编号 |
| rt | 浮点数 | 保留时间（秒） |
| fbr | 浮点数 | 格式结合率 |
| tb | 整数 | 总键数 |
| fb | 整数 | 断裂键数 |
| mb | 字符串 | 键定断裂位置（分号分隔） |
| true_multi | 字符串 | 真实的多标签标注（分号分隔的0/1序列） |

### 标签编码格式
- 使用分号分隔的字符串表示多个断裂位点
- 每个位置用0或1表示是否发生断裂
- 支持不同类型的断裂分类标签
- 示例：`"1;0;1;0;1;0;1;0;1;0"`表示在位置0、2、4、6、8发生断裂

## 使用方法

### 环境要求
- **Python**: 3.6或更高版本
- **PyTorch**: 1.7或更高版本
- **CUDA**: 推荐使用GPU加速训练
- **内存**: 至少8GB内存用于数据加载和模型训练
- **存储**: 足够的磁盘空间存储数据集和模型权重

### 安装依赖

```bash
# 安装PyTorch（根据你的CUDA版本选择）
pip install torch torchvision torchaudio

# 安装其他依赖
pip install pandas numpy pyteomics scikit-learn matplotlib seaborn tensorboard
```

### 快速开始

#### 1. 查看配置文件
```bash
# 查看单标签模型配置
cat dbond_s_config/default.yaml

# 查看多标签模型配置
cat dbond_m_config/default.yaml
```

#### 2. 准备数据集
```bash
# 确保数据集文件存在
ls dataset/
# 应该包含：dbond_s.train.shuffle.csv, dbond_s.test.csv, dbond_m.train.shuffle.csv, dbond_m.test.csv
```

#### 3. 训练模型

**训练单标签模型（DBond-s）**：
```bash
python train.dbond_s.py --config dbond_s_config/default.yaml
```

**训练多标签模型（DBond-m）**：
```bash
python train.dbond_m.py --config dbond_m_config/default.yaml
```

#### 4. 评估模型

**评估单标签模型**：
```bash
python evaluate.dbond_s.py --config dbond_s_config/default.yaml
```

**评估多标签模型**：
```bash
python evaluate.dbond_m.py --config dbond_m_config/default.yaml
```

#### 5. 使用TensorBoard监控训练
```bash
tensorboard --logdir tensorboard/
# 在浏览器中访问 http://localhost:6006
```

### 自定义数据集使用

#### 1. 准备CSV格式数据
创建符合格式要求的CSV文件，包含所有必需的列：
```python
import pandas as pd

# 创建示例数据
data = {
    'name': ['sample_1', 'sample_2'],
    'seq': ['ACDEFGHIK', 'LMNPQRSTV'],
    'charge': [2, 3],
    'pep_mass': [1234.5, 1567.8],
    'nce': [30, 35],
    'scan_num': [1001, 1002],
    'rt': [1200.5, 1300.8],
    'fbr': [0.8, 0.7],
    'tb': [10, 10],
    'fb': [4, 5],
    'mb': ['1;3;5;7', '2;4;6;8;9'],
    'true_multi': ['1;0;1;0;1;0;1;0;1;0', '0;1;0;1;0;1;0;1;0;1']
}

df = pd.DataFrame(data)
df.to_csv('custom_dataset.csv', index=False)
```

#### 2. 使用PBCLA工具转换MGF文件
```bash
# 转换MGF到CSV（小型模型）
python PBCLA/mgf2csv.dbond_s.py --input your_data.mgf --output output.csv

# 转换MGF到CSV（中型模型）
python PBCLA/mgf2csv.dbond_m.py --input your_data.mgf --output output.csv
```

#### 3. 更新配置文件
修改配置文件中的数据路径：
```yaml
data:
  csv_path: "path/to/your/dataset.csv"
  # ... 其他配置保持不变
```

### 配置文件详细说明

#### 数据配置
```yaml
data:
  csv_path: "dataset/dbond_s.train.shuffle.csv"  # 数据文件路径
  max_len: 100                                   # 最大序列长度
  alphabet: "ACDEFGHIKLMNPQRSTVWY"              # 氨基酸字母表
  pad_char: "U"                                  # 填充字符
  batch_size: 32                                 # 批处理大小
```

#### 模型配置
```yaml
model:
  d_model: 256        # Transformer模型维度
  n_head: 8          # 注意力头数
  n_layer: 6         # Transformer层数
  dropout: 0.1       # Dropout率
  vocab_size: 21     # 词汇表大小（20个氨基酸 + 1个填充字符）
```

#### 训练配置
```yaml
training:
  epochs: 100              # 训练轮数
  learning_rate: 0.001     # 初始学习率
  weight_decay: 0.0001     # 权重衰减
  save_interval: 10        # 模型保存间隔
  validation_split: 0.2    # 验证集比例
```

### 评估指标说明

项目提供多种评估指标，均在`multi_label_metrics.py`中实现：

#### 示例级别指标（Example-based Metrics）
- **子集准确率（Subset Accuracy）**: 要求预测标签集合完全匹配真实标签集合
- **示例准确率（Example Accuracy）**: 计算每个样本的准确率然后平均
- **精确率（Precision）**: 预测为正的标签中实际为正的比例
- **召回率（Recall）**: 实际为正的标签中被正确预测的比例
- **F1分数**: 精确率和召回率的调和平均

#### 标签级别指标（Label-based Metrics）
- **宏观平均（Macro）**: 对每个标签计算指标然后平均
- **微观平均（Micro）**: 全局计算指标，不考虑标签差异

#### 使用示例
```python
from multi_label_metrics import example_f1, label_precision_macro

# 假设有真实标签和预测标签
gt = np.array([[1, 0, 1], [0, 1, 0]])  # 真实标签
pred = np.array([[1, 1, 0], [0, 1, 1]])  # 预测标签

# 计算指标
f1_score = example_f1(gt, pred)
precision = label_precision_macro(gt, pred)
```

### 性能优化建议

#### 1. 模型选择
- **单标签模型（DBond-s）**: 
  - 适合快速原型开发和实验
  - 训练时间短，内存需求小
  - 适合资源受限的环境

- **多标签模型（DBond-m）**: 
  - 提供更高的预测精度
  - 需要更多的计算资源和时间
  - 适合生产环境和精度要求高的应用

#### 2. 超参数调优
```yaml
# 学习率调度
training:
  learning_rate: 0.001
  lr_scheduler: "cosine"  # 或 "step", "exponential"
  warmup_epochs: 10

# 正则化
model:
  dropout: 0.1
  weight_decay: 0.0001

# 批处理大小
data:
  batch_size: 32  # 根据GPU内存调整
```

#### 3. 硬件优化
```bash
# 使用多GPU训练
python -m torch.distributed.launch --nproc_per_node=2 train.dbond_m.py

# 混合精度训练
python train.dbond_m.py --mixed_precision
```

### 故障排除

#### 常见数据问题
1. **序列长度超限**：
   ```python
   # 检查序列长度
   max_seq_len = df['seq'].str.len().max()
   print(f"最大序列长度: {max_seq_len}")
   ```

2. **标签格式错误**：
   ```python
   # 验证标签格式
   def validate_labels(label_str):
       labels = label_str.split(';')
       return all(l in ['0', '1'] for l in labels)
   ```

3. **缺失值处理**：
   ```python
   # 检查缺失值
   print(df.isnull().sum())
   # 填充或删除缺失值
   df = df.dropna()  # 或 df.fillna(value)
   ```

#### 常见训练问题
1. **梯度消失/爆炸**：
   ```yaml
   # 梯度裁剪
   training:
     grad_clip_norm: 1.0
   ```

2. **过拟合**：
   ```yaml
   # 增加正则化
   model:
     dropout: 0.2
   training:
     weight_decay: 0.001
   ```

3. **学习率不稳定**：
   ```yaml
   # 学习率调度
   training:
     lr_scheduler: "cosine"
     min_lr: 0.00001
   ```

### 高级功能扩展

#### 1. 自定义模型架构
```python
# 在dbond_s.py中添加新的层
class CustomDBondS(DBondS):
    def __init__(self, config):
        super().__init__(config)
        # 添加自定义层
        self.custom_layer = nn.Linear(config.d_model, config.d_model * 2)
    
    def forward(self, seq, charge, nce, rt):
        x = super().forward(seq, charge, nce, rt)
        x = self.custom_layer(x)
        return x
```

#### 2. 新的数据增强技术
```python
# 在data_utils中添加数据增强
def augment_sequence(seq, augmentation_prob=0.1):
    if random.random() < augmentation_prob:
        # 随机替换氨基酸
        seq = list(seq)
        pos = random.randint(0, len(seq)-1)
        seq[pos] = random.choice(list("ACDEFGHIKLMNPQRSTVWY"))
        return ''.join(seq)
    return seq
```

#### 3. 自定义评估指标
```python
# 在multi_label_metrics.py中添加新指标
def custom_metric(gt, predict):
    """
    自定义评估指标
    """
    # 实现自定义指标计算
    return score
```

### Docker使用

#### 构建Docker镜像
```bash
# 构建Docker镜像
docker build -t dbond_env:v0 .

# 运行容器
docker run -it --gpus all -v $(pwd):/workspace dbond_env:v0 bash
```

#### Dockerfile说明
Dockerfile包含了所有必要的依赖和环境配置，确保项目在不同环境中的一致性。

### 项目贡献指南

#### 如何贡献
1. **Fork项目**并创建功能分支
2. **提交代码**并确保所有测试通过
3. **更新文档**包括README和代码注释
4. **创建Pull Request**并描述更改内容

#### 代码规范
- 遵循PEP 8编码规范
- 添加适当的类型提示
- 编写清晰的文档字符串
- 包含单元测试

### 引用格式

如果您在研究中使用了DBond2，请引用：

```bibtex
@article{DBond2: De novo Deep Learning approach for bond cleavage prediction}
author={Your Name and Collaborators}
title={DBond2: Deep Learning for Disulfide Bond Cleavage Prediction}
journal={Journal of Proteomics}
year={2025}
volume={X}
pages={XXX-XXX}
}
```

### 许可证

本项目采用MIT许可证，详见LICENSE文件。

### 更新日志

#### v1.0.0 (2025-01-02)
- 初始版本发布
- 实现DBond-s和DBond-m两个模型
- 提供完整的数据处理和评估工具
- 支持多种质谱数据格式
- 包含详细的文档和使用示例

#### 未来计划
- [ ] 添加模型可视化工具
- [ ] 支持更多质谱数据格式（如mzML）
- [ ] 优化训练效率和推理速度
- [ ] 添加预训练模型支持
- [ ] 扩展分布式训练功能
- [ ] 开发Web界面
- [ ] 集成更多评估指标
- [ ] 支持实时预测

### 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至：your-email@example.com
- 访问项目主页：https://github.com/your-repo/DBond2

---

**注意**: 本项目仍在积极开发中，欢迎社区贡献和反馈。在使用过程中遇到问题，请及时提交Issue或联系开发团队。
