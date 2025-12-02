# DBond - 基于深度学习的镜像肽肽键断裂预测模型

DBond是基于论文《Optimizing Mirror-Image Peptide Sequence Design for Data Storage via Peptide Bond Cleavage》开发的深度神经网络预测模型，专门用于预测镜像肽在串联质谱（Tandem Mass Spectrometry）过程中的肽键断裂情况。该模型是镜像肽数据存储技术的核心组件，通过预测肽键断裂率来优化序列设计，从而提高数据存储的可靠性。

## 核心目标：优化镜像肽数据存储技术

### 背景问题
镜像肽（由D-氨基酸组成）具有存储密度高、化学稳定性好的特点，是理想的数据存储介质。然而，对镜像肽进行测序（读取数据）极其困难，现有的从头测序（De-novo）算法在处理镜像肽时准确率有限。

### 解决思路
研究发现，质谱图中肽键断裂越多（产生的碎片离子越丰富），测序就越容易。DBond的作用就是通过预测候选镜像肽序列的"肽键断裂率"，来判断该序列是否容易被测序。

### 应用价值
- **序列筛选器**：如果一个序列被DBond预测为具有高断裂率，它就是"易于测序"的优良数据载体
- **数据存储优化**：在数据写入阶段就筛选出最优序列，克服镜像肽测序难的技术瓶颈
- **桥梁作用**：连接"数据编码"与"生物测序"，确保存储的数据未来能够被准确读取

## 模型架构与技术特点

### 输入特征（多维度特征融合）
DBond综合四组特征进行预测：

1. **序列特征 (Sequence Features)**
   - 镜像肽本身的氨基酸序列（由D-氨基酸组成）
   - 决定了肽段的基础理化性质

2. **状态特征 (State Features)**
   - 前体离子的电荷 (Charge)
   - 质荷比 (m/z)
   - 绝对强度 (Intensity)

3. **键特征 (Bond Features)**
   - 肽键在序列中的相对位置（从N端开始计数）

4. **环境特征 (Env Features)**
   - 质谱实验的环境参数
   - 归一化碰撞能量 (NCE)
   - 扫描编号 (Scan Number)

### 网络架构
模型针对不同类型的特征使用专门的处理模块：

- **多头自注意力机制 (Multi-head Self-Attention, MSA)**
  - 处理序列特征，捕捉D-氨基酸之间的依赖关系
  - 提取序列内部的深层信息

- **数值嵌入 (Numerical Embedding)**
  - 处理状态、键和环境特征
  - 数值特征经过归一化和仿射变换后嵌入到高维空间

- **多层感知机 (MLP)**
  - 所有特征处理后拼接输入MLP
  - 输出最终的预测结果

## 预测策略对比

DBond设计了两种预测策略，通过实验验证了单标签策略的优越性：

### 多标签分类策略 (Multi-label Classification) - DBond-m
**策略描述**：
- 一次性预测一条镜像肽中所有肽键的断裂状态
- 输入整条肽段信息，输出包含每个肽键"断裂/未断裂"状态的向量

**劣势**：
- 数据集样本数量相对较少，每个样本对应多个标签
- 解空间非常稀疏，模型难以准确匹配整条肽链的真实断裂模式
- Subset Accuracy较低

### 单标签分类策略 (Single-label Classification) - DBond-s ⭐推荐
**策略描述**：
- 将复杂的整体预测任务分解为多个独立的子任务
- 依次预测每一个肽键的断裂状态
- 输入包含肽段信息和当前要预测的特定肽键位置
- 输出仅针对这一个肽键是否断裂的判断

**优势**：
- 虽然忽略了标签之间的依赖关系，但显著提高了预测性能
- 在独立测试集上表现出色
- 单肽键预测准确率达到**82.42%**

## 在数据存储流程中的应用

DBond在镜像肽数据存储的编码/翻译阶段（Translate Step）发挥关键筛选作用：

```
原始数据 → 二进制编码 → 候选序列生成 → DBond预测筛选 → 最优序列选择 → 物理合成
```

### 详细流程

1. **原始数据编码**
   - 图片、文档等原始数据转换为二进制流

2. **候选规则生成**
   - 二进制数据通过不同映射规则转换为不同的D-氨基酸序列
   - 例如：规则A生成序列AABB，规则B生成序列CCDD

3. **DBond预测与筛选**
   - 将候选序列输入DBond
   - DBond预测各序列的肽键断裂率
   - 系统选择预测断裂率最高的序列及其映射规则

4. **合成与存储**
   - 合成筛选出的最优序列进行物理存储

## 项目结构

```
DBond2/
├── README.md                        # 项目说明文档
├── LICENSE                          # MIT许可证
├── Dockerfile                       # Docker构建文件
├── .gitignore                       # Git忽略文件
│
├── 核心模型文件/
│   ├── dbond_s.py                   # 单标签DBond模型架构定义
│   ├── dbond_m.py                   # 多标签DBond模型架构定义
│   ├── data_utils_dbond_s.py        # 单标签模型数据处理工具
│   └── data_utils_dbond_m.py        # 多标签模型数据处理工具
│
├── 训练和评估脚本/
│   ├── train.dbond_s.py             # 单标签模型训练脚本
│   ├── train.dbond_m.py             # 多标签模型训练脚本
│   ├── evaluate.dbond_s.py          # 单标签模型评估脚本
│   └── evaluate.dbond_m.py          # 多标签模型评估脚本
│
├── 配置文件/
│   ├── dbond_s_config/default.yaml  # 单标签模型配置文件
│   └── dbond_m_config/default.yaml  # 多标签模型配置文件
│
├── 数据集/
│   ├── dbond_s.train.shuffle.csv    # 单标签模型训练数据
│   ├── dbond_s.test.csv             # 单标签模型测试数据
│   ├── dbond_m.train.shuffle.csv    # 多标签模型训练数据
│   ├── dbond_m.test.csv             # 多标签模型测试数据
│   └── dataset.fbr.csv              # 格式结合断裂位点信息数据
│
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
│
├── 评估指标/
│   └── multi_label_metrics.py       # 多标签分类评估指标计算
│
├── graph_transform/                 # 图神经网络扩展模块
│   ├── models/                      # 图模型架构
│   ├── training/                    # 图模型训练
│   ├── evaluation/                  # 图模型评估
│   └── data/                        # 图数据处理
│
└── 输出目录/
    ├── best_model/                  # 最佳模型权重存储
    ├── checkpoint/                  # 模型检查点
    ├── result/                      # 评估结果存储
    └── tensorboard/                 # TensorBoard日志
```

## 核心文件功能说明

### 模型架构文件
- **`dbond_s.py`**: 单标签DBond-s模型实现，基于Transformer架构，用于逐个肽键断裂预测
- **`dbond_m.py`**: 多标签DBond-m模型实现，一次性预测整条肽链的所有肽键断裂状态

### 数据处理文件
- **`data_utils_dbond_s.py`**: 单标签模型数据加载器，处理包含肽键位置信息的特征
- **`data_utils_dbond_m.py`**: 多标签模型数据加载器，处理整条肽链的多标签数据

### 训练脚本
- **`train.dbond_s.py`**: 单标签模型训练，支持GPU加速和学习率调度
- **`train.dbond_m.py`**: 多标签模型训练，包含完整的验证流程

### 评估脚本
- **`evaluate.dbond_s.py`**: 单标签模型评估，计算二分类指标
- **`evaluate.dbond_m.py`**: 多标签模型评估，计算多标签分类指标

## 数据格式说明

### 输入数据格式
训练和测试数据CSV文件包含以下列：

| 列名 | 类型 | 描述 |
|------|------|------|
| name | 字符串 | 样本唯一标识符 |
| seq | 字符串 | D-氨基酸序列（如：ACDEFGHIK） |
| charge | 整数 | 前体离子电荷状态（通常为2-4） |
| pep_mass | 浮点数 | 质荷比 (m/z) |
| nce | 整数 | 归一化碰撞能量 (NCE) |
| scan_num | 整数 | 质谱扫描编号 |
| rt | 浮点数 | 保留时间（秒） |
| intensity | 浮点数 | 前体离子绝对强度 |
| bond_position | 整数 | 当前预测的肽键位置（仅DBond-s） |
| cleavage_label | 整数 | 肽键断裂标签（0/1，仅DBond-s） |
| multi_labels | 字符串 | 所有肽键断裂标签（分号分隔，仅DBond-m） |

### 标签编码格式

**DBond-s（单标签）**：
- `cleavage_label`: 0表示未断裂，1表示断裂
- 每行数据对应一个肽键的预测任务

**DBond-m（多标签）**：
- `multi_labels`: 分号分隔的0/1序列
- 示例：`"1;0;1;0;1;0;1;0;1;0"`表示在位置0、2、4、6、8发生断裂

## 使用方法

### 环境要求
- **Python**: 3.7或更高版本
- **PyTorch**: 1.8或更高版本
- **CUDA**: 推荐使用GPU加速训练
- **内存**: 至少16GB内存用于数据加载和模型训练

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/MRDOCTOROO/DBond.git
cd DBond

# 安装PyTorch（根据CUDA版本选择）
pip install torch torchvision torchaudio

# 安装其他依赖
pip install -r requirements.txt
```

### 快速开始

#### 1. 准备数据集
```bash
# 确保数据集文件存在
ls dataset/
# 应该包含：dbond_s.train.shuffle.csv, dbond_s.test.csv, dbond_m.train.shuffle.csv, dbond_m.test.csv
```

#### 2. 训练模型

**训练单标签模型（推荐，DBond-s）**：
```bash
python train.dbond_s.py --config dbond_s_config/default.yaml
```

**训练多标签模型（DBond-m）**：
```bash
python train.dbond_m.py --config dbond_m_config/default.yaml
```

#### 3. 评估模型

**评估单标签模型**：
```bash
python evaluate.dbond_s.py --config dbond_s_config/default.yaml
```

**评估多标签模型**：
```bash
python evaluate.dbond_m.py --config dbond_m_config/default.yaml
```

#### 4. 使用训练好的模型进行预测

```python
import torch
from dbond_s import DBondS
from data_utils_dbond_s import DBondSDataset

# 加载模型
model = DBondS.load_from_checkpoint('best_model/dbond_s/best_model.pth')
model.eval()

# 准备数据
dataset = DBondSDataset('path/to/your/data.csv')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

# 预测
predictions = []
with torch.no_grad():
    for batch in dataloader:
        seq, charge, nce, rt, bond_pos = batch
        pred = model(seq, charge, nce, rt, bond_pos)
        predictions.extend(pred.cpu().numpy())
```

### 配置文件说明

#### DBond-s配置示例
```yaml
data:
  csv_path: "dataset/dbond_s.train.shuffle.csv"
  max_len: 50
  batch_size: 32
  alphabet: "ACDEFGHIKLMNPQRSTVWY"

model:
  d_model: 256
  n_head: 8
  n_layer: 6
  dropout: 0.1

training:
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  save_interval: 10
```

#### DBond-m配置示例
```yaml
data:
  csv_path: "dataset/dbond_m.train.shuffle.csv"
  max_len: 50
  batch_size: 16  # 多标签模型内存需求更大

model:
  d_model: 512    # 更大的模型容量
  n_head: 8
  n_layer: 8
  dropout: 0.1

training:
  epochs: 150
  learning_rate: 0.0005
```

## 性能对比

### 实验结果
根据论文实验结果，DBond-s在独立测试集上表现优异：

| 模型 | 预测策略 | 准确率 | 优势 |
|------|----------|--------|------|
| DBond-s | 单标签分类 | **82.42%** | 高精度，适合实际应用 |
| DBond-m | 多标签分类 | 较低 | 一次性预测，但精度受限 |
| Prosit | 基线方法 | 较低 | 传统方法，性能一般 |
| PredFull | 基线方法 | 较低 | 传统方法，性能一般 |

### 推荐使用场景

**DBond-s（推荐）**：
- 镜像肂数据存储的实际应用
- 需要高精度预测的场景
- 资源受限的环境

**DBond-m**：
- 研究和实验目的
- 需要快速整链预测的初步筛选
- 计算资源充足的场景

## 数据处理工具

### PBCLA工具使用

```bash
# 转换MGF到CSV（单标签模型）
python PBCLA/mgf2csv.dbond_s.py --input your_data.mgf --output output_s.csv

# 转换MGF到CSV（多标签模型）
python PBCLA/mgf2csv.dbond_m.py --input your_data.mgf --output output_m.csv

# 使用PBCLA主工具
python PBCLA/pbcla.py --mode convert --input data.mgf --output result.csv
```

### 自定义数据处理

```python
import pandas as pd
from data_utils_dbond_s import DBondSDataset

# 创建自定义数据集
def create_custom_dataset():
    data = {
        'name': ['sample_1', 'sample_2'],
        'seq': ['ACDEFGHIK', 'LMNPQRSTV'],
        'charge': [2, 3],
        'pep_mass': [1234.5, 1567.8],
        'nce': [30, 35],
        'scan_num': [1001, 1002],
        'rt': [1200.5, 1300.8],
        'intensity': [1000000, 800000],
        'bond_position': [2, 3],
        'cleavage_label': [1, 0]
    }
    return pd.DataFrame(data)

# 使用自定义数据
df = create_custom_dataset()
dataset = DBondSDataset(df)
```

## 高级功能

### 模型集成
```python
class DBondEnsemble:
    def __init__(self, model_paths):
        self.models = [DBondS.load_from_checkpoint(path) for path in model_paths]
    
    def predict(self, seq, charge, nce, rt, bond_pos):
        predictions = []
        for model in self.models:
            pred = model(seq, charge, nce, rt, bond_pos)
            predictions.append(pred)
        return torch.mean(torch.stack(predictions), dim=0)
```

### 数据增强
```python
def augment_mirror_peptide_data(df, aug_prob=0.1):
    """镜像肽数据增强"""
    augmented_data = []
    
    for _, row in df.iterrows():
        augmented_data.append(row)
        
        if random.random() < aug_prob:
            # 添加噪声到连续特征
            noisy_row = row.copy()
            noisy_row['nce'] += random.uniform(-2, 2)
            noisy_row['intensity'] *= random.uniform(0.9, 1.1)
            augmented_data.append(noisy_row)
    
    return pd.DataFrame(augmented_data)
```

## 故障排除

### 常见问题

1. **内存不足错误**：
   - 减少batch_size
   - 使用梯度累积
   - 增加swap空间

2. **训练不收敛**：
   - 检查学习率设置
   - 增加数据预处理
   - 使用学习率调度器

3. **预测精度低**：
   - 检查数据质量和标签准确性
   - 增加训练数据量
   - 调整模型架构参数

### 性能优化

```yaml
# 优化配置示例
training:
  batch_size: 64
  accumulation_steps: 4  # 梯度累积
  mixed_precision: true  # 混合精度训练
  gradient_clip_norm: 1.0
  
model:
  d_model: 512
  n_head: 16
  n_layer: 12
```

## Docker使用

```bash
# 构建镜像
docker build -t dbond:latest .

# 运行容器（GPU支持）
docker run --gpus all -v $(pwd):/workspace -it dbond:latest

# 训练模型
docker run --gpus all -v $(pwd):/workspace dbond:latest python train.dbond_s.py
```

## 引用格式

如果您在研究中使用了DBond，请引用原始论文：

```bibtex
@article{DBond2024,
  title={Optimizing Mirror-Image Peptide Sequence Design for Data Storage via Peptide Bond Cleavage},
  author={[作者列表]},
  journal={[期刊名称]},
  year={2024},
  volume={[卷号]},
  pages={[页码]}
}
```

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

1. Fork项目并创建功能分支
2. 提交代码并确保所有测试通过
3. 更新相关文档
4. 创建Pull Request并详细描述更改

### 开发规范
- 遵循PEP 8编码规范
- 添加类型提示和文档字符串
- 编写单元测试
- 确保向后兼容性

### v1.0.0
- 初始版本发布
- 实现DBond-s和DBond-m两个模型
- 提供完整的数据处理和评估工具

## 联系方式

- **项目主页**: https://github.com/MRDOCTOROO/DBond
- **问题反馈**: 请在GitHub提交Issue
- **技术讨论**: 欢迎在Discussions中交流

---

**注意**: DBond是一个专门用于镜像肽数据存储优化的研究工具。在使用过程中，请确保理解其在数据存储流程中的作用和局限性。如需技术支持，请通过GitHub Issue联系开发团队。
