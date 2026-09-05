# DBond 实验记录：模型改动 ↔ 效果 ↔ 指标

> 维护说明：本文件是实验主线日志。每次模型/数据/损失改动记录一节：改了什么（commit + 配置）、
> 为什么改、结果如何（指标表）、结论与下一步。新实验追加新节，勿删旧记录。
> 评估协议统一：5 折（每卡一折并行），序列级 disjoint 划分，标准输出
> `checkpoints/.../5fold/<ts>/5fold_summary.csv`（mean±std 口径）。
> 指标口径：lab_* 为 padding-free 键级多标签指标（mi=micro, ma=macro），ex_* 为逐样本均值，
> 排序指标（Spearman/Top-K）为肽级、阈值无关。pod 数据根：`/mnt/pvc/graphtrans/DBond`。

最近更新：2026-09-05

---

## 1. 结论速览（GT-pre 主线阶梯）

| 阶梯 | run（cv_root 时间戳） | 相对上一步的单变量 | lab_f1_mi | lab_acc_mi | ex_f1 |
|---|---|---|---|---|---|
| pre md6（锚点基线） | `pre_synthesis/5fold/20260901_101303` | — | 0.7940±0.0070 | 0.7986 | 0.7065 |
| pre md6 + 理论离子特征 | `pre_synthesis/5fold/20260902_073953` | +15 维序列衍生理论键特征 | 0.7950±0.0053 | 0.8000 | 0.7050 |
| full md6（含 intensity/scan，参考上限） | `pre_synthesis/5fold/20260901_113451` | 解除特征屏蔽 | **0.8411**±0.0049 | 0.8470 | 0.7504 |
| 全 GAT + 理论特征 | `pre_synthesis_gat/5fold/20260903_091047` | 3G+2A → 5GAT（等深等参） | 0.7946±0.0056 | 0.7965 | 0.7060 |
| 全 GAT + 辅助头 | `pre_synthesis_gat_aux/5fold/20260903_112113` | +中间层 bond 头 + 肽级比例头 | **0.7972**±0.0062 | 0.7982 | 0.7096 |
| （待跑）+ ASL 主损失 | 配置 `..._gat_aux_asl.yaml` | BCE → ASL | — | — | — |
| （在跑/待评）+ q 软标签 | 配置 `..._gat_aux_soft.yaml` + `dataset/5fold_soft` | 训练标签 → 条件均值 q | — | — | — |

补充指标（3G+2A 时代记录）：

| run | AUC | MCC | Spearman | Top10% | Top20% |
|---|---|---|---|---|---|
| pre md6 | 0.8808 | 0.5973 | 0.7919 | 0.4247 | 0.5840 |
| pre md6 + theory | 0.8818 | 0.6001 | 0.7975 | 0.4329 | 0.6041 |
| full md6 | ~0.93 | — | 0.9379 | — | — |

md3 对照：pre md3 F1 0.7921±0.0054（低于 md6 0.7940，主线定为 md6）。

**要点**：full−pre 差距（F1 +0.046 / Spearman +0.146）全部来自 intensity/scan_num；
结构侧改动（全 GAT、辅助头）目前累计 +0.003 F1，被标签噪声地板约束（见 §2）。

---

## 2. 标签不确定性分析（瓶颈证据，2026-09-05）

脚本 `graph_transform/scripts/analyze_label_uncertainty.py`（pod 已跑，fold 1222 全量）。
数据结构：477,669 张谱图 / 513 条唯一序列（每序列约 1,000 张谱图）。

| 检测量 | 数值 | 含义 |
|---|---|---|
| within-seq 方差 q(1−q) | 0.1755（上限 0.25） | 键标签跨谱图剧烈摇摆 |
| between-seq 方差 | 0.0137 | |
| **ICC(序列)** | **0.0723** | **93% 方差在序列内部** |
| 模糊键（0.1<q<0.9） | 80.8%（强模糊 59.3%） | |
| 平均熵 H(q) | 0.745 bits | |
| 谱图两两 disagreement | 总体 35.95% / **同条件(charge,nce) 14.91%** / 跨条件 37.25% | 同条件 ~15% 为不可约随机碎裂噪声；条件效应占大头 |
| 标签不一致序列 | 513/513（100%） | |

**LOO oracle 上限**（同序列其他谱图的 q 预测当前谱，无泄漏）：

| | acc | F1 | AUC |
|---|---|---|---|
| 序列-only q | 0.718 | 0.694 | 0.805 |
| **条件匹配 q（seq+charge+nce）= pre 特征集 Bayes 上限** | **0.888** | **0.883** | **0.962** |
| 当前 pre 模型 | 0.800 | 0.795 | 0.882 |
| 当前 full 模型 | 0.847 | 0.841 | ~0.93 |

结论：pre 模型距自身特征集上限还有 ~9 个点；换骨干无用（方差不在结构侧），
治法是把条件均值 q 学好（→ q 软标签，§7）。full 模型强是因为 intensity/scan
泄漏了当次谱图的实际碎裂状态。理论特征收益小（+0.002 F1）的解释：标签依赖
实测条件（理论离子"存在"≠"被观测"），但它在排序指标上有效（Spearman +0.006、
Top20% +0.020）——排序只需要 q。

---

## 3. 理论离子特征（2026-09-02 生效）

实现：`graph_transform/data/theory_features.py`（GT）/ `ludbond*/bond_theory_torch.py`（1D 系）。
每键 15 维，纯序列计算（不读 MGF）：prefix/suffix 质量、b1+/b2+/y1+/y2+ 理论 m/z
（H+=1.007276，H2O=18.010565，残基质量 pyteomics 单同位素 + PBCLA 的 B/O/X/Z）、
相对位置、两侧残基质量、H2O-loss（S/T/E/D）、NH3-loss（N/Q/K/R）标记、
Pro 上下文（Xxx-Pro / Pro-Xxx）。GT 侧 Linear(15→hidden) 拼进 bond head；
s/m/af/af_opt 分别为 Encoder theory slot / theory_logit 残差 / theory_proj。

效果（vs pre md6 基线，run 20260902_073953）：F1 +0.001、AUC +0.001、MCC +0.003、
Spearman +0.006、Top10% +0.008、Top20% +0.020。方向全部为正、幅度受噪声地板压制。

---

## 4. 全 GAT 骨干（2026-09-03，commit fb1ee28）

改动：`num_gcn_layers 3→0、num_gat_layers 2→5`（总深度 5、参数量持平 ~133K/层），
配置 `pre_synthesis_5fold_md6_theory_gat.yaml`。同时修复 `gat_only` 消融开关的
深度保持（对称 gcn_only）。结果（run 20260903_091047）：**lab_f1_mi −0.0004（持平，
折间噪声内）**，实际效应是精度换召回（precision −0.012 / recall +0.012）。
结论：骨干选择不改变主线结论；后续动结构优先考虑 GATv2（修 static attention），
但优先级让位于标签问题。

## 5. 辅助头 deep supervision + 肽级比例头（2026-09-03，commit cce2be4）

改动：①第 2/4 层 GAT（索引 1/3）各接轻量 bond 辅助头（[h_src,h_dst,e_ij]→半宽→1，
权重 0.25）；②global node 接 MLP 回归"该肽可观测断裂比例"（MSE，权重 0.2）。
仅训练期参与损失，eval/推理零开销；训练日志新增 aux_bond_loss / peptide_aux_loss。
结果（run 20260903_112113，vs gat）：lab_f1_mi +0.0026、lab_f1_ma +0.0062、ex_f1 +0.004，
由召回驱动。幅度在 5 折 std（±0.006）内、方向各口径一致。配置
`pre_synthesis_5fold_md6_theory_gat_aux.yaml`。

## 6. ASL 主损失（已实现，待跑）

`AsymmetricLoss`（γ⁺=0, γ⁻=2, margin=0.05）注册为 `main_loss: asymmetric`，
对症"0=未观测"弱负标签（压掉易负例梯度）。逐元素项天然支持软目标。
配置 `pre_synthesis_5fold_md6_theory_gat_aux_asl.yaml`。注意标签接近均衡
（grand rate 0.48），预期有限。

## 7. q 软标签 + 双口径评价（2026-09-05，commits e9c9983/76f4eb7/7f41703，在跑）

**软标签路径**：
- `precompute_soft_labels.py`：按 (seq,charge,nce) 组内谱图均值写 `soft_multi` 列。
  train 文件 q 只从本折 train 谱图算（防泄漏）；test 文件也写 q（期望行为真值，
  供评估用，不参与训练）。已生成 `dataset/5fold_soft/`（train ~7.6k 组/折、
  test ~1.9k 组/折、组均 ~50 谱图，mean|q−y|=0.150 —— 与同条件 disagreement
  14.91% 吻合，被平滑的恰是不可预测部分）。
- 数据集 `use_soft_labels` 开关（浮点解析、无列自动回退、缓存指纹含软列、
  与增强互斥）；训练器主损失与辅助头均吃软目标，**验证 loss 与全部评估仍用
  realized 标签**（口径与历史可比）。
- 运行：`train_5fold_parallel.py --config ..._gat_aux_soft.yaml --fold_data_dir dataset/5fold_soft --gpus 0,1,2,3`

**双口径指标**（metrics 新增，q 真值可得时自动输出）：
- realized 口径（不变）：acc/P/R/F1/subset/ex_/lab_ 系 + 肽级 Spearman/Top-K。
- expected-behavior 口径（`q_` 前缀）：键级 q_brier/q_mae/q_rmse/q_pearson/q_spearman；
  肽级 q_spearman_pep / q_ndcg（graded gain=每肽 q 均值）/ q_top10/20_enrichment
  （前 K% 预测肽的真实 q 均值 / 全体均值，>1 即富集）。
- **看点**：q 软标签的主价值在候选序列排序——主看 q_spearman_pep / q_ndcg /
  enrichment；realized 口径 lab_f1_mi 受噪声地板约束（锚点 gat_aux 0.7972）。
- 合成数据单元验证全部通过（含一处行索引 bug 修复）。

---

## 8. ludbond 1D 基线（pre + theory，2026-09-02）

同一协议下四模型（`result/cv/dbond_*_pre_theory/20260902_213611`）：

| 模型 | ex_F1 / F1 |
|---|---|
| dbond_s_pre_theory | 0.7893±0.0043 |
| dbond_m_pre_theory | 0.7118±0.0034 |
| dbond_af_pre_theory | 0.7415±0.0078 |
| dbond_af_opt_pre_theory | 0.7464±0.0079 |

注意：四个 pre 基线（不加理论特征）在本 pod 未跑过（历史基线来自其他机器），
delta 对比需先补跑 `python run_models_parallel.py -m dbond_s_pre dbond_m_pre dbond_af_pre dbond_af_opt_pre`。

## 9. 训练速度记录

见 `docs/training_speed_optimization.md`（/dev/shm 约束、OOM 根因、edge-cache+workers=8
最优、NodeEncoder 向量化 −17%、torch.compile 29×劣化等）。特殊残基 B/O/X/Z 的
核对与修复见 `docs/special_residue_BOXZ_action_plan.md`。

---

## 10. 完整指标对照表（subset_acc → lab_f1_mi，5 折 mean±std）

| 指标 | pre_md6 | +theory | full | gat | gat_aux |
|---|---|---|---|---|---|
| subset_acc | 0.0391±0.0047 | 0.0391±0.0045 | **0.0733**±0.0063 | 0.0379±0.0039 | 0.0385±0.0063 |
| ex_acc | 0.5896±0.0107 | 0.5883±0.0062 | **0.6438**±0.0059 | 0.5892±0.0081 | 0.5935±0.0087 |
| ex_precision | 0.7460±0.0038 | 0.7481±0.0053 | **0.7830**±0.0038 | 0.7390±0.0082 | 0.7361±0.0090 |
| ex_recall | 0.7529±0.0143 | 0.7476±0.0031 | 0.7581±0.0060 | 0.7610±0.0109 | **0.7703**±0.0132 |
| ex_f1 | 0.7065±0.0087 | 0.7050±0.0052 | **0.7504**±0.0045 | 0.7060±0.0074 | 0.7096±0.0068 |
| lab_acc_ma | 0.8042±0.0059 | 0.8055±0.0053 | **0.8494**±0.0045 | 0.8023±0.0078 | 0.8047±0.0069 |
| lab_acc_mi | 0.7986±0.0043 | 0.8000±0.0037 | **0.8470**±0.0034 | 0.7965±0.0038 | 0.7982±0.0041 |
| lab_precision_ma | 0.7561±0.0153 | 0.7651±0.0167 | **0.8160**±0.0186 | 0.7489±0.0136 | 0.7476±0.0134 |
| lab_precision_mi | 0.7814±0.0041 | 0.7841±0.0082 | **0.8403**±0.0056 | 0.7724±0.0082 | 0.7716±0.0087 |
| lab_recall_ma | 0.7509±0.0147 | 0.7521±0.0151 | 0.7974±0.0160 | 0.7682±0.0258 | **0.7773**±0.0216 |
| lab_recall_mi | 0.8072±0.0118 | 0.8063±0.0042 | 0.8419±0.0050 | 0.8183±0.0105 | **0.8247**±0.0132 |
| lab_f1_ma | 0.7440±0.0144 | 0.7449±0.0173 | **0.7982**±0.0154 | 0.7500±0.0222 | 0.7562±0.0199 |
| lab_f1_mi | 0.7940±0.0070 | 0.7950±0.0053 | **0.8411**±0.0049 | 0.7946±0.0056 | 0.7972±0.0062 |

## 11. 下一步优先级（按证据排序）

1. q 软标签五折（在跑）→ 看排序口径（q_spearman_pep/q_ndcg/enrichment）
2. q 有效后：sequence-level cleavage-ratio ranking loss（margin ranking）
3. FiLM 条件调制（use_condition_film，~33K 参数，单变量）——charge×nce 占方差大头，
   现有三条注入路全是加性；FiLM 显式参数化乘性交互
4. GATv2 / bond-centric graph（结构侧候选，单变量逐个来）
5. 5 折 ensemble 推理（5 个 best_model 概率平均，零训练成本）
6. 补跑 4 个 ludbond pre 基线（与 pre_theory 配对算 delta）

## 12. 运行手册（pod 项目根目录，.venv/bin/python）

> `scripts/run_ablation_queue.sh`（旧单卡消融编排）已删除。它做的事：读
> `graph_transform/config/default.yaml` → 内嵌 python 在内存重置 ablation 段只开
> 一个开关 → 写 /tmp 临时 YAML → `CUDA_VISIBLE_DEVICES=<N> python
> graph_transform/scripts/train_5fold.py --config /tmp/xxx.yaml` 单卡顺序跑五折 →
> cat 最新 `checkpoints/graph_transform/5fold/*/5fold_summary.csv`。以下 A/B 是现行替代。

### A. 单配置五折（现行标准，替代旧编排的核心路径）

1. **写配置**：复制最接近的基线 YAML（如 `graph_transform/config/pre_synthesis_5fold_md6.yaml`），
   改三处——`ablation` 段只开目标开关（互斥校验一次只许一个）、输出目录加后缀
   （`training.checkpoint_dir` / `evaluation.output_*_dir` / `logging.*_dir`）、
   `experiment.name`。消融模板参考 `graph_transform/config/archive/ablation_*.yaml`。
2. **跑**（折级并行，每卡一折，4 卡两轮；终端实时流式带 [f折] 前缀）：
   ```bash
   .venv/bin/python graph_transform/scripts/train_5fold_parallel.py \
       --config graph_transform/config/<你的>.yaml --gpus 0,1,2,3
   ```
   数据目录变体（如 q 软标签）加 `--fold_data_dir dataset/5fold_soft`。
3. **看结果**：`<checkpoint_dir>/<tag>/5fold/<时间戳>/5fold_summary.csv`
   （标准三件套 metrics/summary/aggregate）。
4. 图缓存：配置里 `ablation.rebuild_cache: true`（默认开）即等价于旧脚本的清缓存逻辑；
   手动清理删 `cache/graph_data/*.pt`。

### B. 多配置并行（模型级，一卡一任务）

```bash
.venv/bin/python graph_transform/scripts/run_experiments_parallel.py \
    --jobs "名字1:config/一.yaml;名字2:config/二.yaml" --gpus 0,1,2,3
```

### C. ludbond 四基线矩阵（1D 模型）

```bash
.venv/bin/python run_models_parallel.py -m dbond_s_pre dbond_m_pre dbond_af_pre dbond_af_opt_pre --gpus 0,1,2,3
# 理论特征版把 _pre 换成 _pre_theory
```

### D. q 软标签全流程

```bash
.venv/bin/python graph_transform/scripts/precompute_soft_labels.py \
    --fold_dir dataset/5fold --out_fold_dir dataset/5fold_soft
.venv/bin/python graph_transform/scripts/train_5fold_parallel.py \
    --config graph_transform/config/pre_synthesis_5fold_md6_theory_gat_aux_soft.yaml \
    --fold_data_dir dataset/5fold_soft --gpus 0,1,2,3
```

### E. 断点续跑 / 只重建汇总

- 中断后续跑：原命令重跑即可，train_5fold 自动跳过已完成折
  （判定 `best_model.pt` + `latest_test_metric.csv` 同时存在）。
- 只重汇总：`--cv_root <目录> --aggregate_only`。

### F. 评估与推理

- 加载权重补评估（如给旧 run 在 5fold_soft test 上补 q 口径基线）：
  `.venv/bin/python graph_transform/scripts/evaluate_graph_model.py --config <yaml> --model_path <best_model.pt>`
- 合成前候选序列打分（R-01 应用入口）：
  `.venv/bin/python graph_transform/scripts/score_presynthesis.py --config <yaml> --checkpoint <best_model.pt> --sequences cand.txt --output_dir result/presynthesis`
  （可设 `--charges` / `--nces` 预设条件网格）
- 标签不确定性分析：
  `.venv/bin/python graph_transform/scripts/analyze_label_uncertainty.py --inputs dataset/5fold/1222.train.fbr.shuffle.multi.csv dataset/5fold/1222.test.fbr.multi.csv`

## 13. 仓库整理记录（2026-09-05，commit 9f59081+）

- 删除：`mini_ghtrans/`（41 文件最小验证原型）、`arc_dbond/`（13 文件，ludbond 子集）、
  `mgf_cover.py`、`train_5fold copy.py`、`default copy.yaml`、`train_with_preprocessed.py`、
  `test_preprocessing.py`、7 个旧默认输出 CSV、`scripts/run_ablation_queue.sh`（功能并入 §12 手册）。
- 归档：`plans/` → `docs/plans/`；历史 config 4 个 → `graph_transform/config/archive/`。
- 图件：顶层 12 个 PNG/drawio/svg → `figures/`（drawio 受 `*.drawio` ignore 的保持本地未跟踪）。
- 仅取消跟踪（本地保留）：`.claude/`、`.kilo/`、`.zcode/` 会话文件、`CLAUDE.md`、`AGENTS.md`
  （后两者加入 .gitignore；pod 侧 pull 后会从工作区移除，不影响训练）。
- 保留待议：6 个孤立分析脚本（cross_validation_analysis 等）、`graph_transform/outputs/graph_viz/`
  （visualize_sample_graph.py 的可视化产物）、`.drawio-tmp/`（图表生成中间产物）。
