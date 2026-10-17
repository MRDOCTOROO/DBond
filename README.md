# DBond — 镜像肽肽键断裂预测（数据存储序列筛选）

DBond 是一个基于深度学习的**镜像肽（D-氨基酸组成）肽键断裂预测**项目。镜像肽具有高存储密度与化学稳定性，
是理想的数据存储介质，但其串联质谱（MS/MS）测序（读回）困难；质谱中肽键断裂越充分、碎片离子越丰富，
序列就越容易被测序。DBond 通过预测候选镜像肽序列的**肽键断裂概率**，在写入阶段筛选"易于测序"的序列，
连接数据编码与生物测序两端。

> 本 README 描述 **2026-10-14 仓库整理后**的当前状态。AI 协作细则、硬规则与实验台账见
> `AGENTS.md`（仅本地）；历史实验结论见 `docs/experiment_log.md`（仅本地）。

---

## 1. 当前主线（论文口径）

| 项目 | 内容 |
|---|---|
| 主模型 | **DBond-GT**：残基图 + 3×残差 GCN + 2×边感知多头 GAT（hidden 256、4 头、dropout 0.1、全局节点） |
| 任务设置 | **pre-synthesis**：屏蔽观测特征 intensity / scan_num / rt，仅用序列、charge、pep_mass、NCE + 15 维理论特征 |
| 模型体改动 | ① 15 维键级理论特征（`data/theory_features.py`）经 Linear(15→256) 拼入键级预测头；② FiLM 条件调制（charge/NCE → γ,β，零初始化） |
| 图构建 | 距离窗口策略：**R = 5**（连接全部 \|i−j\|≤5 的残基对）+ 全局虚拟节点 |
| 训练数据 | 条件组折叠（行 → 唯一 (seq, charge, NCE) 组，软标签 q + `group_uniform` 加权）；**测试侧保持行级** |
| 数据划分 | **5 组重复序列级随机 80/20 划分**（1222/2252/3514/6072/9075；每折内 train/test 零序列重叠）——注意：不是 k-fold CV |
| 集成 | **DBond-GT-Ens** = 同折 5 seed 概率平均（42–46） |
| 主配置 | `graph_transform/config/pre_synthesis_fold1222_theory_film.yaml` |

五组划分上的最终对比（bondacc = padding-free 有效键准确率，%；完整表见 `docs/paper_tables_20260909.md`）：

| 模型 | bondacc | Lab-F1 | q-Brier ↓ | ρ_key |
|---|---|---|---|---|
| DBond-s | 78.30±0.28 | 77.75±0.51 | 7.55 | 76.85 |
| DBond-m | 77.48±0.48 | 76.91±0.64 | 8.25 | 75.36 |
| DBond-AF | 79.86±0.37 | 79.25±0.90 | 7.13 | 81.21 |
| DBond-AF(opt) | 79.93±0.34 | 79.70±0.69 | 6.61 | 81.11 |
| **DBond-GT** | 80.02±0.36 | 79.80±0.64 | 6.69 | 81.01 |
| **DBond-GT-Ens** | **80.38±0.22** | **80.36±0.47** | **6.28** | **81.67** |

**指标口径（重要）**：标签级准确率只以带有效键掩码的 **bondacc** 为准
（`label_accuracy_micro(gt, predict, mask)`）。未经掩码的 accuracy 会被 padding 位 (0,0) 的 TN 灌水
（+3.4~+5.7 点），本仓库不再产生、也不再引用该口径的数字。
原始论文中的 **82.42%** 是旧口径/旧划分下的历史数值，与上表不可直接比较。

---

## 2. 仓库结构

```
DBond/
├── graph_transform/            # 主线：DBond-GT 图模型（模型/训练/评估/数据/脚本/配置）
│   ├── config/                 # 含论文主线配置、md 敏感性扫描配置、archive/ 历史配置
│   ├── scripts/                # 训练、5 组划分并行、评估、折叠、软标签、集成、可解释性、聚合
│   ├── data/ models/ training/ evaluation/ utils/
│   └── README.md               # 模块说明（架构、图构建、脚本入口）
├── ludbond/                    # 对比基线 DBond-s / DBond-m（独立镜像形态，含 5fold 入口）
├── ludbondaf/                  # 对比基线 DBond-AF / DBond-AF(opt)
├── sklearn_baseline/           # 树模型/线性基线（XGBoost/RF/HistGB/LogReg，同口径对照）
├── PBCLA/                      # MGF 解析与 CSV 转换工具 + 示例数据
├── dataset/                    # 随仓库发布的最小基线数据（5 组划分的行级 CSV）
├── figures/                    # 论文图件：源码（.js/.drawio/.py）+ 成品（.pptx/.pdf）+ 图 4 数据
├── result/                     # 运行输出目录（结构用 .gitkeep 保留，内容不入库）
├── docs/                       # 实验台账与论文数值表（仅本地，见 §6）
├── archive/                    # 归档区：被取代的图件/文档/脚本（随仓库保留，见 archive/README.md）
├── run_pre_baselines.py        # 基线 pre 变体批量运行入口
├── run_models_parallel.py      # 多模型 × 多 GPU 矩阵调度
├── scripts/run_baselines_5fold.sh  # 基线 obs 设置 5 组划分批量运行
└── AGENTS.md / CLAUDE.md       # AI 协作说明（仅本地，不入库）
```

## 3. 环境

- Python 3.7+、PyTorch 1.8+（GPU）、CUDA 11.8；依赖见 `requirements.txt`，
  精确版本锁见 `pyproject.toml` + `uv.lock`（推荐 `uv sync`）。
- 训练/评估在 GPU pod 上执行（本项目实际运行环境：A800，`/mnt/pvc/graphtrans/DBond`，结果写入 `result/`、`checkpoints/`）。
- 对比基线镜像自带独立环境（`ludbond/Dockerfile`、`ludbondaf/Dockerfile`，
  亦可 `docker pull LoserLus/dbond_env:latest` / `LoserLus/dbond_af_env:latest`）。
- 本机（无 GPU）只用于图件与仓库维护：`figures/` 下 node 图件脚本需要 `figures/node_modules`（`npm i` 安装 pptxgenjs）。

## 4. 快速开始（复现主线）

```bash
# 0) 数据准备（在 pod 上，项目根目录执行）
#    基线行级数据：dataset/*.csv（已随仓库发布）
#    图模型数据：锚定 5 组划分 → 训练侧条件组折叠 + 软标签 q 真值
python graph_transform/scripts/fold_condition_groups.py      # -> dataset/5fold_folded
python graph_transform/scripts/precompute_soft_labels.py     # q 真值（期望行为口径）

# 1) 五组划分并行训练（5 张 GPU，终端日志）
python graph_transform/scripts/train_5fold_parallel.py \
    --config graph_transform/config/pre_synthesis_fold1222_theory_film.yaml \
    --fold_data_dir dataset/5fold_folded --gpus 0,1,2,3,4

# 2) 单折评估（注意参数是 --checkpoint）
python graph_transform/scripts/evaluate_graph_model.py \
    --config graph_transform/config/pre_synthesis_fold1222_theory_film.yaml \
    --checkpoint checkpoints/graph_transform/<run>/fold_<id>/checkpoints/gt_pre/<ts>/best_model.pt \
    --out_metric_csv result/metric/<name>.csv

# 3) 同折 5-seed 集成推理 + 统一 q 口径复算（跨模型数字都用这两个脚本产出）
python graph_transform/scripts/ensemble_inference.py
python graph_transform/scripts/q_metrics_from_pred_csv.py

# 4) 基线（三个入口按设置选用；同样在 pod 上跑）
python run_pre_baselines.py            # DBond-s/m/AF/AF(opt) 的 pre 变体（论文表行）
bash   scripts/run_baselines_5fold.sh  # 基线的 obs 设置（历史对照行）
```

配置与消融：
- md 敏感性扫描（7 臂）：`pre_synthesis_theory_folded_film_{md3,md4,md5,md7,md8,mddense,notheory}.yaml`，
  用同一 `train_5fold_parallel.py` 逐臂运行（采纳规则：平台期内取最小有效半径）；
- 特征/结构消融：`pre_synthesis_theory_folded_film_{nocharge,nonce,nomass,nomp,noedge,noglobal,gcnonly,...}.yaml`；
- 已证伪的实验方向（辅助头、ranking loss、ASL、9-seed 等）见 `AGENTS.md` 台账，**不要重跑**。

## 5. 论文图件对应关系

| 论文图件 | 源码（`figures/`） | 成品 |
|---|---|---|
| Fig 1 工作流 | `build_workflow_figure.js` | `workflow_mesure.pptx` / `.drawio` / `.svg` |
| Fig 2 残基图构建 | `graph_representation.drawio` | 论文目录内 `graph_representation.drawio2.pdf` |
| Fig 3 模型架构 | `build_architecture_v5.js` | `dbond_gt_architecture_v5.pptx` / `.pdf` |
| Fig 4 残基对化学矩阵 | `graph_transform/scripts/residue_pair_analysis.py` | 数据 `figures/residue_pair_film_fold9075/`；统计复算 `figures/paper_stats_20261013/` |
| SI 可解释性面板 | `graph_transform/scripts/{interpretability,occlusion}_analysis.py` | 汇总脚本 `aggregate_interpretability_occlusion_5fold.py`；解读手册 `graph_transform/docs/interpretability_figures_guide.md` |

PPTX 渲染/导出 PDF 用 `figures/{render_pptx,export_pptx_pdf}.ps1`（PowerPoint COM）。
更早世代的图件源码与成品（panels 系列、v4、chaintograph3、matplotlib 架构图等）已移入 `archive/figures/`。

## 6. 仅本地约定（不入库）

| 路径 | 说明 |
|---|---|
| `docs/` | 实验台账（`experiment_log.md`，**随代码入库**）与论文数值表、工作草稿；其余内容仅本地 |
| `AGENTS.md` / `CLAUDE.md` | AI 协作说明，仅本地使用，不推送到云端 |
| `archive/` | 归档区，随仓库保留（被取代的图件/文档/脚本，见 `archive/README.md`） |
| `实验图/`、`result/`、`checkpoints/`、`results/`、`logs/`、`tensorboard/`、`state/` | 运行产物，不入库 |

## 7. 数据格式

行级 CSV（`dataset/dbond_{s,m}*.csv`）关键列：

| 列 | 含义 |
|---|---|
| `seq` | 镜像肽序列（D-氨基酸） |
| `charge` / `pep_mass` / `nce` / `scan_num` / `rt` / `intensity` | 前体离子与实验条件特征 |
| `bond_position` / `cleavage_label` | 键位置 / 断裂标签（单标签长表） |
| `multi_labels` | 分号分隔的整链断裂标签（多标签模型） |

MGF → CSV：`PBCLA/mgf2csv.dbond_s.py`、`PBCLA/mgf2csv.dbond_m.py`（示例数据在 `PBCLA/mgf_dataset/`，
脚本内默认路径即指向该示例；`ludbond/PBCLA/` 镜像版提供 `--mgf_path/--csv_path` 参数形式）。

## 8. 归档区

`archive/` 存放被取代但保留备查的内容：旧世代架构图（v4、panels、chaintograph3、matplotlib 版）、
旧工作流图、旧规划/评审响应文档、被取代的分析脚本（跨折分析、注意力可视化旧框架等）。
规则见 `archive/README.md`：**归档 = 冻结，不参与流水线、不在论文中引用**；
确证无效的产物（错误口径的消融训练/评估脚本、跑不通的 bench、运行日志）已直接删除，可从 git 历史恢复。

## 9. 引用与许可

论文引用信息请以正式发表版本为准（DBond：Optimizing Mirror-Image Peptide Sequence Design for
Data Storage via Peptide Bond Cleavage）。本项目采用 MIT 许可证，详见 `LICENSE`。

问题与建议请通过本仓库 Issue 反馈。