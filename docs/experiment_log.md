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
| ~~软标签(gat_aux底座)~~ | `pre_synthesis_gat_aux_soft/5fold/20260905_094058` | **无效 run**：use_soft_labels 因 config 传递 bug 未生效（ceafc5f 修复），实为 gat_aux 非确定性重跑（F1 0.7954±0.0086） | — | — | — |
| ~~q 软标签(gat_aux底座)·真软标签~~ | `pre_synthesis_gat_aux_soft/5fold/20260906_040542` | 修复后重跑，q 确认生效；**等价性审计证明重复行 q ≡ hard BCE**（§7.2），F1 0.7956±0.0061 与 hard 0.7972 种子噪声内，作为等价性实证对照归档 | 0.7956±0.0061 | 0.7957 | 0.7077 |
| ~~q 软标签·theory 底座~~ | `pre_synthesis_theory_soft/5fold/20260907_053941` | 审计预言逐位兑现：F1 0.7949±0.0053 vs hard 0.7950（Δ=0.0001），q 指标全同（§7.2 实证二） | 0.7949±0.0053 | 0.8006 | 0.7041 |
| 条件组折叠·自检（spectrum w=n_g） | `..._folded_spectrum/5fold/20260907_070840`（fold1222 单折） | 期望≡hard theory：F1 0.7951 vs hard 同折 0.7908（单折种子噪声内；训练日程不同不逐位等）→ 折叠实现正确 | —（单折） | 0.7963 | 0.7066 |
| 条件组折叠·组均匀（uniform 五折） | `..._folded_uniform/5fold/20260907_072754` | 383k 行→7.6k 条件组，w=1；**候选排序主指标无增益**（§14.1） | 0.7954±0.0073 | 0.7975 | 0.7077 |
| 条件组折叠·序列均衡（seqbal 五折） | `..._folded_seqbal/5fold/20260907_091256` | 每序列等权 w=1/K_s；同样无增益（§14.1） | 0.7960±0.0068 | 0.7972 | 0.7085 |
| ~~+ 序列级 ranking loss（rank 五折）~~ | `..._folded_rank/5fold/20260907_124400` | λ=0.3 pairwise；**五折证伪**（q_seq 0.6122 < uniform 0.6171；试点 +0.033 为单折噪声，第二次教训：q_seq 折间 std 0.03-0.05，单折不可判） | 0.7953±0.0069 | 0.7970 | 0.7082 |
| **+ FiLM 条件调制（film 五折）= 当前最佳** | `..._folded_film/5fold/20260907_124432` | [charge,nce]→(γ,β) 乘性调制（~33K 参数零初始化）；F1 **0.7980 新最高**，并把折叠底座掉的键级 q_spearman（0.8038→0.8101）/q_cond（0.8875→0.8901）拉回 hard 行级水平 | **0.7980±0.0057** | 0.8002 | 0.7112 |
| （待跑）+ FiLM + 辅助头 | `..._folded_film_aux.yaml` | 两个已证正交正向机制首次叠加（+0.0026/+0.0026）；deep supervision 层 [1,3]→[1]（2 层 GAT） | — | — | — |
| （待跑）+ ASL 主损失 | 配置 `..._gat_aux_asl.yaml` | BCE → ASL（正率 0.48，预计收益有限，降级） | — | — | — |

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
| **条件匹配 q（seq+charge+nce）= 重复测量一致性上限** | **0.888** | **0.883** | **0.962** |
| 当前 pre 模型 | 0.800 | 0.795 | 0.882 |
| 当前 full 模型 | 0.847 | 0.841 | ~0.93 |

命名修正（20260906 审计）：条件匹配 oracle 利用了同一条测试肽的其他谱图历史，
应称**同序列重复测量条件下的可重复性上限**（repeated-spectrum ceiling），不是
严格意义的"pre 特征集 Bayes 上限"——部署时面对新候选序列没有历史谱图可查。
它仍是任何模型在这些测试行上的精度的上界（若能从特征完美估计 q 即达到），
但 pre 特征能否把未见序列的 q 学到这个水平，是泛化问题，不由此表保证。

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

**事故记录（2026-09-06，ceafc5f 修复）**：run 20260905_094058 的软标签**未生效**——
train/evaluate/smoke 三处把 ModelConfig（仅 model 段属性）而非完整 config dict 传给
GraphDataset，data 段键 `use_soft_labels` 检索不到被静默回退 False，该 run 实为
gat_aux 非确定性重跑。修复：三处改传完整 dict；smoke 增加"batch 必须携带
soft_labels"断言；dataset 增加反向保险丝（有 soft 列但开关未开会打印警告）；
evaluate 支持 `--out_pred_csv /dev/null` 跳过预测输出（批量补评曾因预测归档写满
磁盘配额报 No space left）。

### 7.1 q 口径基线（2026-09-06 补评，4 run × 5 折 best_model 在 5fold_soft test 上）

用 `evaluate_graph_model.py --config <run配置+use_soft_labels> --test_csv dataset/5fold_soft/<fold>.test... 
--out_metric_csv result/metric/q_reeval/<run>_fold<f>.csv --out_pred_csv /dev/null` 补评。
（"soft"列 = 094058 run，即 gat_aux 复跑，可当 gat_aux 的噪声对照。）

| 指标 | theory (3G+2A) | gat (全GAT) | gat_aux | soft(=无效run) |
|---|---|---|---|---|
| q_brier ↓ | **0.0684**±0.0028 | 0.0702 | 0.0686 | 0.0719 |
| q_mae ↓ | **0.1632**±0.0043 | 0.1699 | 0.1678 | 0.1750 |
| q_pearson | **0.7916**±0.0083 | 0.7843 | 0.7899 | 0.7805 |
| q_spearman（键级） | **0.8088**±0.0083 | 0.8007 | 0.8064 | 0.7967 |
| q_spearman_pep（肽级） | **0.9004**±0.0062 | 0.8872 | 0.8891 | 0.8826 |
| q_ndcg | **0.9928** | 0.9918 | 0.9923 | 0.9916 |
| q_top10_enrichment | **1.716**±0.019 | 1.694 | 1.710 | 1.693 |
| q_top20_enrichment | **1.653**±0.009 | 1.630 | 1.640 | 1.629 |
| lab_f1_mi（realized） | 0.7947 | 0.7946 | **0.7972** | 0.7954 |
| spearman_rho（realized 肽级） | **0.7973** | 0.7856 | 0.7874 | 0.7815 |

要点：①**混合骨干在全部 q 口径上优于全 GAT 系**——全 GAT 换骨干轻微损伤排序质量
（与 realized Spearman 下降一致），故新增 theory 底座的软标签配置；②辅助头对 q
也有小幅正贡献（gat_aux > gat）；③q_ndcg≈0.992 已近饱和、区分度弱，主看
q_spearman_pep 与 enrichment；④top10 enrichment ~1.7：按模型排序取前 10% 肽的
真实断裂比例比平均高 71%，直接对应存储筛选价值。

### 7.2 等价性定理：重复行 q 软标签是无效操作（2026-09-06 审计 + 实证）

**定理**：设条件组 g 有 n 张谱图、标签 y_1..y_n，q = mean(y)。pre 特征集下同组
所有行的模型输入完全相同（seq+charge+pep_mass+nce；intensity/scan 被屏蔽），
则对 BCEWithLogits：
`Σ_j BCE(z, y_j) = n·BCE(z, q)` —— 把 q 逐行复制 n 次再平均，与原 hard 标签的
损失函数**完全相同**（非仅期望相同；仅差 4 位小数舍入与 batch 组成的 RNG）。
推论：mixed hard/soft 目标同样无效（组内平均后回到 q）；§7 的软标签路径只改变
了 q 指标的"可见性"，没有改变训练目标。

**实证一**（修复 config bug 后真软标签 run `20260906_040542` vs hard gat_aux）：

| 指标 | gat_aux (hard) | gat_aux_soft（真 q） |
|---|---|---|
| lab_f1_mi | 0.7972±0.0062 | 0.7956±0.0061 |
| q_spearman | 0.8064（补评） | 0.8011±0.0114 |
| q_spearman_pep | 0.8891（补评） | 0.8822±0.0149 |
| q_top10_enrichment | 1.710（补评） | 1.689±0.027 |

**实证二**（theory 底座，`pre_synthesis_theory_soft/5fold/20260907_053941`，五折）：

| 指标 | theory (hard) | theory_soft（真 q） |
|---|---|---|
| lab_f1_mi | 0.7950±0.0053 | **0.7949±0.0053**（Δ=0.0001） |
| lab_acc_mi | 0.8000 | 0.8006 |
| q_spearman | 0.8088（补评） | 0.8086±0.0168 |
| q_spearman_pep | 0.9004（补评） | 0.9010±0.0125 |
| q_top10_enrichment | 1.716（补评） | 1.714±0.027 |

两个底座五折全部落在种子噪声内（theory 底座 F1 差 0.0001，几乎逐位复现）⇒ 定理
完全成立。**结论：要真正改变训练目标，必须折叠条件组并显式选权重（§14）。**

### 7.3 行级肽级 q 指标的口径缺陷（已修复，2026-09-06）

旧行级 q_spearman_pep/q_ndcg/enrichment 把一张谱图当一个 ranking 样本——同组
谱图预测完全相同却被重复计数，指标被**采集频率**加权（组大小中位 61、最大 128）。
单测示例（2 序列/3 组/8 行合成数据）：同一份数据 enrichment 行级 1.714 /
条件级 1.588 / 序列级 1.385——行级虚高来自重复计数。
新增两套口径（metrics.py，trainer/evaluator 均已接入组键透传）：
- `q_*_cond`：去重到唯一 (seq,charge,nce) 条件组；
- `q_*_seq`：序列内对其条件组均匀平均——**合成前候选肽筛选的主指标**。
行级旧口径保留（与 §7.1 基线可比）；§7.1 的补评基线后续需用新口径重算一遍
（重跑 evaluate 即可，便宜）。

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

## 11. 下一步优先级（按证据排序，2026-09-07 折叠五折后修订）

1. **FiLM + 辅助头叠加**（`..._film_aux.yaml`）：两个已证正向机制首次组合，
   目标 ~0.799+；胜出即为 pre-synthesis 定稿生产版
2. 5 折 ensemble 推理：5 个 best_model 概率平均，零训练成本（需写跨折聚合小脚本）
3. FiLM 条件调制（use_condition_film，~33K 参数，单变量）——charge×nce 占方差大头，
   现有三条注入路全是加性；q 泛化差距（§14.1）主要落在条件交互上
4. GATv2 / bond-centric graph（论文扩展位）
5. 补跑 4 个 ludbond pre 基线（与 pre_theory 配对算 delta）
6. 已证伪/降级：~~条件组折叠与重加权~~（无增益 §14.1）、~~q Beta-Binomial 收缩~~、
   ~~ASL~~（正率 0.48 无不平衡可治）、~~序列级 ranking loss~~（五折证伪 §15.5：
   q_seq 0.6122 < uniform 0.6171；注意 fold-1222 单折两次给出假阳性，q_seq 只认五折）

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

### D. q 软标签全流程（已证 ≡ hard BCE，见 §7.2；保留作等价性对照）

```bash
.venv/bin/python graph_transform/scripts/precompute_soft_labels.py \
    --fold_dir dataset/5fold --out_fold_dir dataset/5fold_soft
.venv/bin/python graph_transform/scripts/train_5fold_parallel.py \
    --config graph_transform/config/pre_synthesis_5fold_md6_theory_gat_aux_soft.yaml \
    --fold_data_dir dataset/5fold_soft --gpus 0,1,2,3
```

### D2. 条件组折叠试点（§14，现行主推）

```bash
# 1) 生成折叠数据（依赖 D 步的 5fold_soft；test 文件符号链接不折叠）
.venv/bin/python graph_transform/scripts/fold_condition_groups.py \
    --fold_dir dataset/5fold_soft --out_fold_dir dataset/5fold_folded

# 2) 等价性/正确性单测（纯 CPU，秒级）
.venv/bin/python graph_transform/scripts/test_folded_equivalence.py

# 3) smoke（可选但建议：校验 batch 携带 soft_labels + sample_weights）
CUDA_VISIBLE_DEVICES= .venv/bin/python graph_transform/scripts/smoke_test_pipeline.py \
    --config graph_transform/config/pre_synthesis_fold1222_theory_folded_uniform.yaml

# 4) 三臂并行（一卡一臂，fold 1222 单折）
for arm in spectrum uniform seqbal; do
  CUDA_VISIBLE_DEVICES=$([ $arm = spectrum ] && echo 0 || ([ $arm = uniform ] && echo 1 || echo 2)) \
  nohup .venv/bin/python graph_transform/scripts/train_5fold.py \
      --config graph_transform/config/pre_synthesis_fold1222_theory_folded_${arm}.yaml \
      --folds 1222 --fold_data_dir dataset/5fold_folded \
      > logs/folded_${arm}.log 2>&1 &
done

# 5) q-checkpoint（best_model_q.pt）单独评测：F 步 evaluate --checkpoint <path>
```

### E. 断点续跑 / 只重建汇总

- 中断后续跑：原命令重跑即可，train_5fold 自动跳过已完成折
  （判定 `best_model.pt` + `latest_test_metric.csv` 同时存在）。
- 只重汇总：`--cv_root <目录> --aggregate_only`。

### F. 评估与推理

- 加载权重补评估（如给旧 run 在 5fold_soft test 上补 q 口径基线）：
  `.venv/bin/python graph_transform/scripts/evaluate_graph_model.py --config <yaml> --checkpoint <best_model.pt> --test_csv <csv> --out_metric_csv <csv> --out_pred_csv /dev/null`
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

## 14. 条件组折叠训练（2026-09-06 实现，待跑）

动机与定理见 §7.2：重复行 q 软标签 ≡ hard BCE，唯一出路是把训练样本真正压缩为
唯一条件组并显式定义权重。数据画像（fold 1222 train）：383,295 谱图行 /
7,643 条件组 / 410 序列（每序列 18.6 个条件，组大小中位 61、P10=11、max 128、
singleton 组 76 个占 1%）。

**三种权重语义**（`data.weighting_scheme`，loss 侧逐键展开）：

| scheme | 每组权重 w_g | 训练分布 | 用途 |
|---|---|---|---|
| spectrum | n_g | 谱图事件复现 | **实现自检**：期望 ≡ 行级 hard theory 训练；若 fold-1222 指标显著偏离其 hard 对应值 ⇒ 折叠/加权有 bug |
| group_uniform | 1 | 条件均匀 | 每个候选 (seq,charge,nce) 等权 |
| sequence_balanced | 1/K_s | 序列均衡 | 每条序列等权（410 序列不被采集频率加权），最贴合候选肽筛选 |

**实现清单**（全部本地验证：compileall + 单测 3 项通过）：
- `scripts/fold_condition_groups.py`：5fold_soft → 5fold_folded（每组 first() 行 +
  group_n 列；行数==组数断言；test 符号链接不折叠）。
- `data/graph_dataset.py`：weighting_scheme 解析（缺 group_n 直接报错）；
  sample_weight 注入 3 个 __getitem__（GraphDataset 主路径/缓存完整图/缓存边）；
  collate 汇 sample_weights（混合批报错）。
- `training/loss_functions.py`：BinaryBondLoss(weights=) 加权 BCE 路径
  （仅纯 BCE 可用，handle_imbalance/辅助项开着则报错）。
- `training/trainer.py`：_masked_row_weights（行主序展平对齐 targets）接主损失 +
  bond 辅助头；肽级辅助头按行权重加权 MSE；train/val metrics update 传组键。
- `training/metrics.py`：batch_group_keys()（"seq|charge|nce" 稳定键）；
  q_*_cond（组去重）/ q_*_seq（序列内条件均匀聚合）两套新指标；
  行级旧口径保留（键名不变，与 §7.1 可比）。
- `scripts/train_graph_model.py`：第二 checkpoint **best_model_q.pt**（默认选择
  指标 training.q_checkpoint_metric = q_spearman_pep_cond，仅用 val；早停与最终
  测试仍走 best_model.pt，5fold 汇总管线不动；q 模型用 evaluate --checkpoint 补评）。
- `scripts/test_folded_equivalence.py`：①行级 hard ≡ 折叠+spectrum 权重（1e-6）；
  ②加权归一语义（权重整体缩放不变）；③cond/seq 指标手工值对账 + 行级口径
  重复计数演示（1.714/1.588/1.385）。
- 三个试点配置 `pre_synthesis_fold1222_theory_folded_{spectrum,uniform,seqbal}.yaml`：
  theory 混合骨干（3G+2A，无辅助头）+ 折叠数据 + 各臂权重；训练动力学重标定
  （batch 1024→128、epochs 100→300、patience 10→25、warmup 10→40、t_max 300，
  折叠后 ~48 步/epoch，总优化步数与行级 run 同量级）。

**判读标准**：
1. sanity 臂（spectrum）fold-1222 realized 指标 ≈ theory run 的 fold-1222 值（种子噪声内）→ 实现正确；
2. uniform/seqbal 主看 q_spearman_pep_seq / _cond / top10_enrichment_seq 相对
   theory 基线（需用新口径重补评，D2/§7.3）是否提升；
3. realized F1 允许小幅波动（训练分布变了），不作为本试点的主终点；
4. best_model_q.pt vs best_model.pt 的测试 q 指标差 = 选择口径的净效应。

### 14.1 三臂试点与五折结果（2026-09-07，configs acdac41）

fold-1222 试点（单折，20260907_070840）：

| 臂 | realized F1 | q_pep_cond | q_pep_seq(主) |
|---|---|---|---|
| theory hard 同折（补评） | 0.7908 | 0.8875 | 0.5300 |
| spectrum（w=n_g 自检） | 0.7951 | 0.8793 | 0.5119 |
| uniform（w=1） | 0.7933 | 0.8956 | 0.5497 |
| seqbal（w=1/K_s） | 0.7974 | 0.8935 | 0.5588 |

试点上 uniform/seqbal 的 seq 排序 +0.02/+0.029 看似有效；五折证伪：

| 指标（5 折 mean±std） | theory hard 基线（补评） | uniform（072754） | seqbal（091256） |
|---|---|---|---|
| **q_spearman_pep_seq（主）** | 0.6159 | 0.6171±0.0247 | 0.6160±0.0335 |
| q_spearman_pep_cond | 0.8920 | 0.8875±0.0069 | 0.8864±0.0073 |
| q_spearman（键级） | 0.8088 | 0.8038±0.0100 | 0.8032±0.0111 |
| q_spearman_pep（行级） | 0.9004 | 0.8939±0.0092 | 0.8931±0.0095 |
| q_top10_enrichment_seq | 1.2526 | 1.2283±0.0328 | 1.2179±0.0446 |
| q_top20_enrichment_seq | 1.1858 | 1.2083±0.0270 | 1.2046±0.0283 |
| lab_f1_mi | 0.7950±0.0053 | 0.7954±0.0073 | 0.7960±0.0068 |

**判定**：主指标死平（+0.001/0.000），键级排序一致轻微下降 ~0.006，enrichment 混合
信号。试点提升 = 单折噪声（折间 std 0.025–0.034）。sanity 臂 ≈ hard ⇒ 这是可信的零。

**联合结论（§7.2 + §14.1）**：BCE 目标侧已饱和——同一 (特征→q) 信息无论怎么切行/
加权/软化，学到的排序能力相同。锚点链：序列-only oracle 0.718 < 当前模型 0.800 <
重复测量上限 0.888；当前模型已超过序列平均行为的信息上限（靠 charge/nce 泛化），
剩余差距 = q 对未见序列的泛化差距 ⇒ 下一步转向条件交互建模（FiLM）与直接排序
优化（ranking loss），不再做目标侧工程。

## 15. 第二轮实现：ranking loss + FiLM + ensemble（2026-09-07，待跑）

依据 §14.1 结论（目标侧饱和、差距在 q 对未见序列的泛化）启动三项，用户手动训练。

### 15.1 序列级 pairwise ranking loss（损失侧，单变量）

- 实现：`trainer._sequence_ranking_loss`——行级 score=有效键 sigmoid 均值、
  target=有效标签均值（优先 q）；按 `sequences` 聚合到序列级，对 target 不等
  （>1e-4）的序列对施加 hinge `relu(margin-(s_hi-s_lo))`；配对数归一。
- 开关：`loss.ranking_loss_weight`（0=关，试点 0.3）、`loss.ranking_margin`（0）；
  训练日志新增 `ranking_loss` 列。仅训练路径，val loss/指标口径不变。
- 配置：`pre_synthesis_fold1222_theory_rank.yaml`（uniform 折叠底座 + ranking，
  对照 = uniform 五折 20260907_072754）。
- 单测：test_folded_equivalence.py [4]（违序>0.5 / 顺序=0 / 单序列=0）。

### 15.2 FiLM 条件调制（结构侧，单变量）

- 实现：`model.use_condition_film`——`[charge×0.1, nce×0.01]` → MLP(2→64→2H)
  生成逐样本 (γ,β)，对打包后节点特征做 `h=(1+γ)h+β`；末层零初始化=恒等起步。
  ~33.5K 参数；charge/nce 均在 pre 允许集，无泄漏。
- 配置：`pre_synthesis_fold1222_theory_film.yaml`（uniform 折叠底座 + FiLM）。
- 看点：q_spearman_pep_cond/seq 与 q_brier 相对 uniform 五折的变化——FiLM 直接
  作用在条件交互的泛化上。

### 15.3 集成推理（⚠ 跨折版已作废，2026-09-07 泄漏实证）

**数据审计发现（2026-09-07，经确认属设计协议）**：五个 fold（1222/2252/3514/
6072/9075）是**五种独立随机划分**，不是 K 折交叉验证分区——每折单独训练、对
五份指标取 mean±std；划分间谱图重叠（76-84%）是设计使然。fold 1222 的 test 有
83.57% 出现在 fold 2252 的 train（按 name+scan_num+rt 唯一标识、全行含标签相同）。
fold 内部 train/test 严格互斥（两折抽检 overlap=0，train 无重复谱图），故
**所有"每折单独跑"的单模型结果全部有效**；推论：任何跨折模型混用（集成/蒸馏/
用 A 折模型评 B 折数据）从协议上不成立，评估必须与本折训练同划分。

首版"留一跨折集成"（fold f 用其余 4 折模型）实测 F1 0.8559±0.0034、AUC 0.9407、
q_spearman_pep_seq 0.8945 —— 看似巨大增益，实为记忆效应（其余 4 折模型各自见过
fold f test 的 ~80%），**全部作废**。防泄漏方向当初就想反了：本折模型才是唯一
没见过本折 test 的。

**唯一诚实方案 = 同折同划分 + 不同 seed 集成**（权重初始化/dropout/批序差异）。
`ensemble_inference.py` 已重写为该模式（--checkpoints 列表，文档头写明泄漏依据）。
预期诚实增益幅度远小于作废数字（通常 +0.003~0.01），需实测。

### 15.4 运行命令（用户手动，pod 项目根）

```bash
# 0) 同步代码
git pull

# 1) 本地单测（CPU 秒级）
.venv/bin/python graph_transform/scripts/test_folded_equivalence.py

# 2) smoke（验证 FiLM 前向与 batch 键；rank 臂同样可跑）
CUDA_VISIBLE_DEVICES= .venv/bin/python graph_transform/scripts/smoke_test_pipeline.py     --config graph_transform/config/pre_synthesis_fold1222_theory_film.yaml     --csv dataset/5fold_folded/1222.train.fbr.shuffle.multi.csv --n_rows 512 --device cpu

# 3) 两个试点并行（fold 1222 单折，各 ~6 分钟）
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python graph_transform/scripts/train_5fold.py     --config graph_transform/config/pre_synthesis_fold1222_theory_rank.yaml     --folds 1222 --fold_data_dir dataset/5fold_folded > logs/rank_pilot.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python graph_transform/scripts/train_5fold.py     --config graph_transform/config/pre_synthesis_fold1222_theory_film.yaml     --folds 1222 --fold_data_dir dataset/5fold_folded > logs/film_pilot.log 2>&1 &

# 4) ensemble（同折多种子，诚实口径；先要有多 seed 的 best_model 列表）
.venv/bin/python graph_transform/scripts/ensemble_inference.py     --checkpoints <seed_a>/best_model.pt,<seed_b>/best_model.pt,<seed_c>/best_model.pt     --config graph_transform/config/pre_synthesis_fold1222_theory.yaml     --test_csv dataset/5fold_soft/1222.test.fbr.multi.csv     --out_csv result/metric/ensemble/seedN_fold1222.csv

# 5) 判读：试点主看 fold-1222 上 q_spearman_pep_seq / q_pep_cond 相对
#    uniform 试点（0.5497 / 0.8956）与 theory hard 同折（0.5300 / 0.8875）；
#    任一方向为正再扩五折（对照 ±0.02 噪声标尺）
```

### 15.5 rank / film 试点结果（fold-1222 单折，20260907_105212）

| 指标 | hard 同折 | uniform 试点 | **rank 试点** | film 试点 |
|---|---|---|---|---|
| lab_f1_mi | 0.7908 | 0.7933 | **0.7966** | 0.7971 |
| q_brier ↓ | 0.0668 | 0.0650 | **0.0644** | 0.0666 |
| q_spearman | 0.8093 | 0.8167 | **0.8157** | 0.8119 |
| q_spearman_pep_cond | 0.8875 | 0.8956 | 0.8947 | 0.8879 |
| **q_spearman_pep_seq（主）** | 0.5300 | 0.5497 | **0.5630** | 0.5363 |
| q_top10_enrichment_seq | 1.2251 | 1.2466 | 1.2466 | 1.2194 |

- **rank**：主指标 +0.033 vs hard / +0.013 vs uniform，q_brier 全场最优——方向正确，
  幅度在种子噪声边缘（±0.017），**值得扩五折**确认。
- **film**：与 hard 基本持平（seq +0.006 / cond ±0）——零初始化恒等起步下未学出
  条件交互增益。单折证据太弱，五折便宜（~6 min）可顺带确认；若五折仍平则搁置。
- 两臂 realized F1 均正常（0.797 上下），无训练不稳迹象。
