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
| ~~+ FiLM + 辅助头（film_aux 五折）~~ | `..._folded_film_aux/5fold/20260907_144353` | **叠加证伪**：F1 0.7961 < film 0.7980，q_spearman/q_cond/q_pep 同向下降（辅助头的正收益在全 GAT 行级底座上，与折叠稀疏目标冲突）；唯一亮点 ECE 0.0375（最佳校准），不采纳 | 0.7961±0.0052 | 0.7974 | 0.7072 |

**★ pre-synthesis 定稿版本 = film 臂**（`pre_synthesis_fold1222_theory_film.yaml`，
五折 `..._folded_film/5fold/20260907_124432`）：混合 3G+2A 骨干 + 15 维理论离子
特征 + 条件组折叠（group_uniform）+ FiLM 条件调制。F1 0.7980±0.0057 / AUC 0.8825 /
q_cond 0.8901 / q_seq 0.6252 / top10_enrichment_cond 1.80。部署：排序用各折
best_model_q.pt，分类用 best_model.pt。**模型搜索收敛**（正向：理论特征、辅助头
@gat底座、FiLM；证伪：全GAT、软标签≡hard、折叠重加权、ranking loss、FiLM+aux叠加）。
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

### 15.6 终局汇总：理论特征三口径增益 + 同折多种子集成（2026-09-08）

补评修复：新增 `pre_synthesis_5fold_md6_qeval.yaml`（= md6 配置 + `use_soft_labels:
true`，仅评估用）——此前 md6 无理论臂补评缺全部 q_*（软标签开关未开，评估批次不
携带 soft_labels；9/7 的 theory/gat/gat_aux 补评当时带开关跑的，故有 q 口径）。

#### 15.6.1 dbondGT(md6 混合骨干) ± 理论特征，5 折 mean±std（同折配对补评）

无理论 = pre_synthesis/5fold/20260901_101303；有理论 = 20260902_073953。
产物：`result/metric/q_reeval_cond/{premd6_notheory,theory}_fold*.csv`。

| 指标 | 无理论 | +理论 | Δ |
|---|---|---|---|
| lab_f1_mi（行级） | 0.7940±0.0078 | 0.7950±0.0059 | +0.0010 |
| AUC | 0.8808 | 0.8818 | +0.0010 |
| MCC | 0.5974 | 0.6001 | +0.0027 |
| spearman_rho（realized 肽级） | 0.7919 | 0.7975 | +0.0056 |
| q_brier ↓ | 0.0688±0.0023 | 0.0686±0.0030 | −0.0003 |
| q_spearman（键级） | 0.8068 | 0.8088 | +0.0019 |
| q_spearman_pep（行级肽） | 0.8942±0.0090 | 0.9004±0.0070 | +0.0062 |
| q_spearman_pep_cond（条件组） | 0.8865±0.0063 | 0.8920±0.0065 | +0.0056 |
| **q_spearman_pep_seq（序列级）** | 0.6004±0.0619 | 0.6159±0.0544 | **+0.0155** |
| q_top10_enrichment_seq | 1.2277±0.0562 | 1.2526±0.0278 | +0.0248 |

要点：
- **增益随口径"离训练目标越远"而放大**：行级 F1 +0.001（噪声地板 ±0.006 内，不能
  作为证据）→ 肽级 q +0.006 → 序列级 q +0.016（配对 4/5 折为正，成对 t≈1.5）。
  序列级正是候选筛选主口径——理论特征的价值在"对没见过的序列排序"，不在逐键判别。
- 序列级方差同步收窄（q_seq std 0.062→0.054；enrichment_seq std 0.056→0.028）。

#### 15.6.2 跨架构：ludbond 四基线 ± 理论特征（lab_f1_mi，5 折 mean）

目录 `result/cv/dbond_{s,m,af,af_opt}_pre[_theory]/`（s/m_theory=20260902_213611，
af*_theory=20260902_200428）。

| 架构 | 无理论 | +理论 | Δ |
|---|---|---|---|
| dbond_s（逐键 1D） | 0.7775 | 0.7831 | +0.0056 |
| dbond_m（多标签 1D） | 0.7691 | 0.7755 | +0.0064 |
| dbond_af | 0.7925 | 0.7909 | −0.0016 |
| dbond_af_opt | 0.7970 | 0.7981 | +0.0011 |
| dbondGT（md6 混合骨干） | 0.7940 | 0.7950 | +0.0010 |

要点：理论特征对弱架构（s/m）行级即有 +0.006（s 的 AUC +0.0077）；强架构
（af/af_opt/dbondGT）行级贴噪声地板，增益转移到 q 排序口径（15.6.1）。
**结论：理论特征增益成立**——弱架构看行级、dbondGT 看序列级口径。

#### 15.6.3 同折多种子集成（fold 1222，film 终版配置，seed 42-46 概率平均）

单模型 = `checkpoints/graph_transform/seedens_film/s0-s4` 各自
latest_test_metric.csv；集成 = `result/metric/ensemble_seed5_film_fold1222.csv`。

| 指标 | 单 seed mean±std（n=5） | 集成 | Δ |
|---|---|---|---|
| f1 | 0.7935±0.0025 | 0.7994 | +0.0059 |
| AUC | 0.8802 | 0.8854 | +0.0052 |
| MCC | 0.5962 | 0.6065 | +0.0103 |
| q_brier ↓ | 0.0673 | 0.0629 | −0.0044 |
| q_spearman_pep | 0.8900 | 0.9017 | +0.0117 |
| q_spearman_pep_cond | 0.8816 | 0.8916 | +0.0100 |
| **q_spearman_pep_seq** | 0.5286 | 0.5856 | **+0.0570** |
| q_top10_enrichment_seq | 1.2113 | 1.2334 | +0.0221 |

要点：诚实集成增益与预判一致（F1 +0.006，远小于作废的跨折 +0.06）；增益同样集中
在序列级 q（+0.057，超最好单 seed +0.039），且校准改善（q_brier −0.0044）。
模式与理论特征一致：**泛化噪声主导的口径上，方差缩减类方法收益最大**。

### 15.7 收官判定（2026-09-08）

- **最终模型：film 臂**（3GCN+2GAT + 15 维理论特征 + 条件折叠 group_uniform + FiLM），
  F1 0.7980±0.0057；部署 best_model_q.pt（排序）/ best_model.pt（分类），
  需要更强序列级排序时上 5-seed 概率平均集成。
- 目标侧（软标签/加权/折叠重加权）已证饱和（§7.2/§14.1）；结构侧 FiLM 为唯一存活
  增益；集成为部署侧免费增益。理论特征增益按 15.6.1/15.6.2 口径分层成立。
- 全部对照/消融/补评产物：`result/metric/q_reeval_cond/`、`result/cv/`、
  `result/metric/ensemble_seed5_film_fold1222.csv`。

### 15.8 ludbond 基线 q 口径补评：跨模型 GT 不领先（2026-09-08，重要修正）

工具 `graph_transform/scripts/q_metrics_from_pred_csv.py`：读各 1D 基线每折
`pred/test.pred.csv` 键级长表（dbond_s 变长含 bond_index；dbond_m/af 定宽 W=35 含
padding 位），重排成概率矩阵 → logit 逆变换 → 喂 DBond-GT 同一 metrics 模块
（注意：metrics.compute() 固定按 logits 过 sigmoid，概率必须先逆变换，否则双
sigmoid 全预测正——f1 会掉到 0.64 一档）。对齐三重断言：行长匹配、逐行 true 串
与 true_multi 全等、重建 f1 与该折 lab_f1_mi 全等（20/20 折 Δ≈3e-15）。

五个 pre 模型 q 口径（5 折 mean±std，% 化；同一 5fold_soft test、同一指标实现）：

| 模型 | lab_F1 | q_brier↓ | q_spearman_pep | _cond | **_seq** | top10_seq |
|---|---|---|---|---|---|---|
| dbond_s  | 77.75±0.51 | 7.55±0.12 | 89.53±0.93 | 88.78±0.54 | **65.06±3.69** | 1.260±0.048 |
| dbond_m  | 76.91±0.64 | 8.25±0.32 | 89.41±0.71 | 88.64±0.54 | 63.35±4.35 | 1.239±0.051 |
| dbond_af | 79.25±0.90 | 7.13±0.21 | **90.25±0.93** | **89.52±0.89** | 63.55±5.69 | 1.220±0.055 |
| dbond_af_opt | 79.70±0.69 | **6.61±0.13** | 89.78±0.71 | 88.93±0.67 | 63.29±4.46 | **1.266±0.064** |
| dbond_gt(film) | **79.80±0.64** | 6.69±0.19 | 89.67±0.90 | 89.01±0.82 | 62.52±5.93（最低） | 1.226±0.045 |

**修正 §15.6/15.7 的跨模型表述**：理论特征/FiLM/5-seed 集成的 q_seq 增益全部是
dbondGT **家族内**比较（±特征、±FiLM、±集成）；跨模型看 q 口径无一家领先——
q_pep/cond 以 dbond_af 最高，q_seq 以 dbond_s 最高且跨折最稳（std 3.7），q_brier
以 af_opt 最低；GT 仅 lab_F1 名义最高（Δ=0.001 vs af_opt，噪声内）。"GT 拉开差距
在 q 口径"的说法不成立，撤回。

解释：dbond_s 逐键独立模型正则最强、对条件结构欠拟合反而序列级排序最稳；GT 容量
大，行级判别与校准强但序列级泛化方差大（std 5.9，家族内低折如 theory 臂 fold-1222
=0.5300 拖累均值）。pre 设置下无模型全面占优；GT 的卖点应表述为 obs 设置全面领先
（§10 旧表）+ pre 行级 F1 名义最高 + 校准第二。序列筛选排序基准建议同时报 dbond-s。

## 15.9 肽级辅助头 RankNet 排序试点（rankaux，2026-09-09，证伪）

设计：film 底座（折叠+group_uniform+FiLM）+ 仅肽级辅助头（deep supervision 关），
目标从 MSE 比例回归换为 RankNet 成对排序（行=条件组，|Δq|≥0.05，权重 0.2），
q_checkpoint_metric 同步切到 q_spearman_pep_seq。与已证伪的"主损失序列级
hinge"“film+MSE 辅助头”变量隔离。五折（best_model.pt 口径）：

| 指标 | film 底座 | rankaux | Δ |
|---|---|---|---|
| lab_F1 (%) | 79.80±0.64 | 79.27±0.51 | −0.53 |
| q_brier (×100) | 6.69 | 7.06 | +0.37 |
| q_spearman_pep | 89.67 | 88.65 | −1.02 |
| q_spearman_pep_cond | 89.01 | 88.01 | −1.00 |
| q_spearman_pep_seq | 62.52±5.93 | 58.75±3.55 | −3.77 |

全指标劣于底座，F1 跌幅 0.53 > 0.3 门限 → 证伪，不进主表。至此辅助头三种
变体（主损失 rank / film+MSE 头 / film+rank 头）全部证伪，与"目标侧已饱和"
判断一致：增益应从方差缩减（集成/正则）找，不从辅助目标找。产物：
checkpoints/graph_transform/pre_synthesis_theory_film_rankaux/5fold/20260909_073823。

## 15.10 五折 × 5-seed 集成正式化（2026-09-10，film 臂，seed 42-46）

方法：train_5fold.py --seed 42..46 × --cv_root seedens5_film/s0-s4（每折 5 个
同划分不同 seed 的 best_model 概率平均，ensemble_inference.py 防跨折泄漏）。
产物：result/metric/ensemble_seed5_5fold/{1222..9075}.csv + summary.csv。

| 指标 | GT 单模型(film) | GT ensemble | 跨模型参照 |
|---|---|---|---|
| lab_F1 (%) | 79.80±0.64 | **80.36±0.47** | af_opt 79.70 → 集成明确第一 |
| q_brier (×100↓) | 6.69 | **6.28±0.14** | af_opt 6.61 → 集成第一（超 af_opt） |
| q_mae (×100↓) | 16.60 | 16.72±0.23 | af 16.17 仍第一 |
| q_spearman_pep | 89.67 | 89.87±0.50 | af 90.25 仍第一 |
| q_spearman_pep_cond | 89.01 | 89.23±0.49 | af 89.52 仍第一 |
| q_spearman_pep_seq | 62.52±5.93 | 63.84±3.93 | dbond_s 65.06 仍第一（差 1.22） |
| q_top10_enrichment_seq | 1.226 | 1.236±0.032 | af_opt 1.266 / dbond_s 1.260 仍前二 |

判读：①F1 由名义领先变为明确领先（+0.66 vs af_opt，>noise）；②Brier 全场第一；
③q_seq +1.32（非 fold-1222 单折预示的 +5.7——该折实测 +5.1，其余折 +1~+3，
折间增益不均），std 5.93→3.93 显著收窄；④MAE/pep/cond 全部第二；⑤Top10_seq
1.236 仍第四（1.266/1.260/1.239）。结论："DBond-GT (ensemble)" 行 = 2 项第一 +
4 项第二，论文可标注推理成本 ×5。剩余缺口：q_seq vs dbond_s（−1.2）与
Top10_seq——对症杠杆是正则化上调（序列级方差），非容量/目标侧。

## 15.11 dropout 0.2 试点 + 25-run 单模型参照 + 集成行 ρ_key 补齐（2026-09-10）

**drop02 五折**（vs film 底座，同判据）：

| 指标 | film(0.1) | drop02 | 判 |
|---|---|---|---|
| lab_F1 (%) | 79.80±0.64 | 80.09±0.53 | +0.29（升） |
| q_brier (×100↓) | 6.69 | 6.48±0.15 | 改善 |
| q_spearman_pep / cond | 89.67 / 89.01 | 89.36 / 88.62 | −0.3~−0.4 |
| q_spearman_pep_seq | 62.52±5.93 | 62.09±4.43 | 未过线（std 收窄） |
| q_top10_enrichment_seq | 1.226 | 1.233 | 持平 |

判读：dropout 0.2 改善行级 F1 与校准，但**不动序列级排序**——晋级门限
（q_seq>62.52）未过，对"补 q_seq 缺口"判为无效；行级收益也不足以替换主臂
（F1 已由 ensemble 行第一）。正则化路线对 q_seq 至此无证据支持。

**25-run 单模型参照**（seed 42-46 × 5 折，seedens5_film）：f1 79.49±0.51、
q_brier 6.91±0.21、q_seq 60.05±5.16 → 集成增益 +0.87 F1 / −0.63 Brier /
**+3.79 q_seq**，且单模型 q_seq 的种子间波动（60.05 vs 主 run 62.52）再次
确认 q_seq 噪声 ±2-3，种子级别结论一律不可信。

**集成行 ρ_key 补齐**：81.67±0.52 > af 81.21 → **集成行升至 3 项第一
（lab_F1 / q_brier / ρ_key）+ 3 项第二（MAE/pep/cond）+ q_seq 第二（差
dbond_s 1.22）+ Top10_seq 第四**。剩余缺口仍集中序列级两列。

## 15.12 9-seed 集成（2026-09-11，增益饱和确认，判据收口）

seed 47-50 补跑（s5-s8），9 模型概率平均（vs 5-seed）：

| 指标 | 5-seed | 9-seed | Δ |
|---|---|---|---|
| lab_F1 (%) | 80.36±0.47 | 80.40±0.42 | +0.04 |
| q_brier (×100↓) | 6.28 | 6.24 | −0.04 |
| q_spearman (ρ_key) | 81.67 | 81.76 | +0.09 |
| q_spearman_pep / cond | 89.87 / 89.23 | 89.98 / 89.34 | +0.11 |
| q_spearman_pep_seq | 63.84±3.93 | 64.28±3.73 | **+0.44（<0.5 门限）** |
| q_top10_enrichment_seq | 1.236 | 1.248 | +0.012 |

八项指标同向微升但全部噪声量级；q_seq 增益 25→5→9 seeds 为 60.05→63.84→
64.28，方差缩减曲线明确饱和。**决策：论文主表保留 5-seed 行（成本脚注 ×5），
正文一句"扩到 9 seeds 再 +0.44 q_seq（64.28），与方差缩减饱和一致"。**
与 dbond_s（65.06）终局差距 0.78-1.22，序列级缺口定性为库容量限制（410 条
训练序列），不再追。产物：result/metric/ensemble_seed9_5fold/。

## 15.13 bondacc 口径事故与全链路修复（2026-09-12）

**事故**：ludbond 系（s/m/af/af_opt）全部 lab_acc_ma/mi 由定宽 padding 矩阵
（pad 到 max_len−1≈35）统计，padding 位 (0,0) 以真 negative 身份灌水 accuracy
（实测 +3.4~+5.7 点）；lab_*_ma 同时被全零列稀释。**GT 家族所有 run（film/
theory/uniform/soft/gat_aux/rankaux/drop02/ensemble）不受影响**——GT 管道
BinaryBondMetrics 本来就 padding-free，折叠脚本 test 不折叠（行级 realized）。
受污染的只有"引用基线 cv 汇总 lab_acc_*"的跨模型表：§2.7/§15 旧表、09-12
前版论文表（均已在 docs/paper_tables_20260909.md 修正）。

**基线 padding-free bondacc 重算值**（各折 test.pred.csv → GT 同一模块，剥
padding + 对齐断言；自检 dbond_s fold1222 0.78164 == 其 top-level accuracy）：
dbond_s 81.71→**78.30**、dbond_m 83.19→**77.48**、dbond_af 84.96→**79.86**、
af_opt 85.02→**79.93**；GT 80.02 / Ens 80.38 本就正确。统一口径后排名：
**Ens 80.38 > GT 80.02 > af_opt 79.93 > af 79.86**——bondacc 第一/第二归 GT
家族，此前"af_opt 85 第一"为 padding 假象。表 1 其余 8 列数学上不受 padding
影响（(0,0) 不进 subset/ex/micro-PRF1 计数；f1 已有 20 折 1e-15 校验）。

**源头修复（后续 run 不再产出灌水 lab_acc）**：
- `multi_label_metrics.py`（ludbond/ 与 ludbondaf/ 两份同步）：label_* 全系
  加 `mask=` 参数（有效键位掩码）+ `valid_length_mask()` 构造器；mask=None
  保留旧行为仅向后兼容；**bondacc ≡ label_accuracy_micro(gt,pred,mask)**。
  自测：padded 0.80 → bondacc 0.60（构造用例），macro 剔除全零列。
- 生产管道全部接线：train.dbond_s.py（per-precursor 实际键数）、train.dbond_m.py
  （val+test 两块，复用已有 seq_padding 掩码）、train.dbond_m.exp_af.py（两块）、
  evaluate.dbond_s.py masked_metric（padding 前真实键数）、evaluate.dbond_m.py
  （补 mask 收集）。此后基线重跑的 lab_acc_mi 应 ≈ 其 top-level accuracy（可作
  自检断言）。
- `q_metrics_from_pred_csv.py` 落盘键加 accuracy/hamming_loss——跨模型补评
  自动携带 bondacc。
- **未修（已证伪死代码，任何驱动脚本均未引用，勿再跑勿采信其 lab_acc）**：
  ludbondaf/train.dbond_m.exp_af_rm_{cat,attn,attn_cat}.py 及对应 evaluate.*
  （rm 消融变体，不建 mask）。

**双 checkpoint（F1 选 vs q 选）现状**：机制在（同训双存 best_model.pt /
best_model_q.pt，film 臂按 q_spearman_pep_cond 选点），film 五折 best_model_q.pt
齐全，但 test 侧对比评估**从未跑成**——当时循环因 $CV 未设中断，
result/metric/rankaux_5fold_qsel/ 为空目录。对比命令见 docs/paper_tables
会话记录（evaluate_graph_model.py --checkpoint best_model_q.pt 逐折评估后与
F1 选点行对照）。

## 15.14 双 checkpoint 五折实证：q 选点 vs F1 选点（2026-09-14）

film 臂五折 best_model_q.pt（val q_spearman_pep_cond 选点）补评完成
（result/metric/film_5fold_qsel/），与 F1 选点同折配对：

| 指标 | F1 选点 | q 选点 | 配对 Δ |
|---|---|---|---|
| lab_F1 (%) | 79.80±0.64 | 79.62±0.68 | −0.18（4/5 折降，最多 −0.60） |
| bondacc (%) | 80.02±0.36 | 80.09±0.32 | +0.07 |
| lab_P / lab_R | 77.70 / 82.04 | 78.45 / 80.86 | +0.75 / −1.18（操作点向 precision 移） |
| q_brier | 6.69 | 6.69 | 0（校准不动） |
| q_mae | 16.60 | 16.38 | −0.22 |
| ρ_key / pep / cond | 81.01/89.67/89.01 | 81.17/89.89/89.31 | +0.17/+0.22/+0.31 |
| q_seq | 62.52±5.93 | 63.06±4.64 | +0.54（全部来自 fold1222 +3.65；2 折微降） |
| Top10_seq | 1.2263 | 1.2308 | +0.005 |

**判读**：方向与设计一致（排序口径均值不劣：ρ_key/pep/cond 平均 +0.17/+0.22/+0.31，q_seq +0.54 且方差 5.93→4.64（2 折微降）、precision 端点右移、校准不变），
但幅度全部在折间噪声内（q_seq +0.54 集中于单折，非稳健赢）。**决策：主表
保持 F1 选点行不动；q 选点不进对比表**（避免选型标准混口径），双 checkpoint
作为方法卖点写一句：一次训练产出两个部署端点，排序筛选场景可用 q 选点
（排序指标不劣且 q_seq 方差更小 5.93→4.64），行级判别场景用 F1 选点。



## 15.15 传统机器学习基线（RF/XGBoost/HistGB/LogReg）：小样本假设检验入档（2026-09-14）

**假设**：MiPD513 只有 513 条唯一序列（47.7 万谱图行，93% 标签方差在序列内，
ICC 0.072），小样本低序列多样性下传统树模型可能不逊于深度模型。新增
`sklearn_baseline/`（features.py + run_tree_baselines.py + README.md），在
**与 film 臂完全相同的特征/折/指标**下直接检验。

**parity 口径**：每键 18 维 = 15 理论特征（`compute_bond_theory` 同源）+
charge/pep_mass/nce（pre-synthesis mask 保留项）；标签 = true_multi 展开逐键
0/1（一折 ~1000 万键行）；5 折（1222/2252/3514/6072/9075）+ seed 42+idx +
行级 20% val；评测走 `BinaryBondMetrics`（f1_micro@0.5 + q_* 全套，logit 逆
变换喂入），q 真值 = 测试折 (seq,charge,nce) 组内均值（逐行复刻
`compute_group_soft`，含 %.4f 取整）；pred 落 dbond_s 变长长表，
`q_metrics_from_pred_csv.py` 可直接补评对拍。

**模型**：rf(300树) / histgb(500轮,lr0.1) / xgb(hist,lr0.05,d6,val logloss
early-stop 50；依赖已进 pyproject+uv.lock→xgboost 3.4.1) / logreg 线性对照。
XGB 迭代数按 val logloss 早停（自定义 F1 eval 的 margin/概率入参歧义不冒险），
val F1 照记。

**运行（pod gsj-5090）**：
```bash
cd /mnt/pvc/graphtrans/DBond && git pull && source .venv/bin/activate
uv pip install xgboost        # 勿 uv sync（会剪掉未声明的 torch）
python sklearn_baseline/run_tree_baselines.py --models rf,histgb,xgb,logreg --folds all --xgb_device cuda
# 对拍闭环（本机缺 torch_geometric 跑不了 canonical precompute）：
python graph_transform/scripts/precompute_soft_labels.py --fold_dir dataset/5fold --out_fold_dir dataset/5fold_soft
python graph_transform/scripts/q_metrics_from_pred_csv.py --pred_csv sklearn_baseline/result/cv/xgb/<ts>/fold_1222/pred/test.pred.csv \
  --test_csv dataset/5fold_soft/1222.test.fbr.multi.csv \
  --ref_metric_csv sklearn_baseline/result/cv/xgb/<ts>/fold_1222/metric/test_metric.csv \
  --out_csv sklearn_baseline/result/cv/xgb/<ts>/fold_1222/metric/q_reeval_check.csv
```

**本机预跑（Windows .venv，fold1222 全量，histgb/logreg 全参，rf 降载
100 树+max_samples 0.3——16GB 本机 RF 全量 OOM）**：管线贯通，pred 长表经
q_metrics_from_pred_csv 对拍 f1 重建 Δ=0.00e+00（histgb/rf 双双位级一致）。

| fold1222 全量 | lab_f1_mi | q_brier | ρ_cond | q_seq | Top10_seq |
|---|---|---|---|---|---|
| histgb(500轮) | **0.7874** | 0.0713 | 0.856 | 0.383 | 1.246 |
| rf(100树,0.3采样) | 0.7771 | 0.0790 | 0.855 | 0.356 | 1.221 |
| logreg 对照 | 0.6828 | 0.1274 | 0.809 | 0.150 | 0.934 |
| （同折锚点·主线 GT hard theory） | 0.7908 | — | — | — | — |
| （同折锚点·1D 基线 dbond_s） | 0.7816 | — | — | — | — |

初步判读（锚点已对齐主线 GT，非 1D 基线）：同特征同折下 histgb 0.7874
**低于主线 GT 单折锚点（hard theory 0.7908，folded 变体单折 0.7951）约
0.3~0.8 点，高于 1D 基线 dbond_s（0.7816）**——"513 序列小样本下传统树
模型键级判别可用、但未超主线 GNN"，且训练成本近零；**q_seq 0.35-0.38 vs
GT 家族 61-63**，序列级排序/筛选能力是树模型明显短板（概率分辨率不足），
该口径主线不可替代。正式结论待 pod 5 折
（rf 300 树全量 + xgb cuda）+ 对拍闭环后回填；对照锚点：film 79.80±0.64 /
Ens 80.36 / dbond_s bondacc 78.30 / q_seq dbond_s 65.06 vs GT 62.52。

## 15.16 Example-F1 口径分裂修复（2026-10-08，外审 P0-1 定案）

**问题**：主表 Table 5 的 Example-F1 列混用两种定义——
- 基线（`ludbond/multi_label_metrics.py example_f1`）：先对谱平均 P/R、再调和平均；
- GT（`graph_transform/training/metrics.py _example_f1`）：逐样本 F1 平均（docstring 自述
  「旧实现是先平均 precision/recall 再调和，数值上不等价」）。
GT 71.12±0.50 与基线 74.x 同列不可比（外审核出 P/R 调和值 ~75.4 的矛盾，根因即此）。

**修复**（统一为基线/DBond 系列调和口径，零推理）：GT 每折 Ex-F1 = 2·ex_p·ex_r/(ex_p+ex_r)，
直接由 film 5fold（20260907_124432）逐折 `5fold_metrics.fold_*.csv` 的 ex_precision/ex_recall 计算：

| fold | ex_p | ex_r | ex_f1(samples) | ex_f1(harmonic) |
|---|---|---|---|---|
| 1222 | .7359 | .7723 | .7100 | .7536 |
| 2252 | .7401 | .7663 | .7104 | .7529 |
| 3514 | .7473 | .7697 | .7193 | .7583 |
| 6072 | .7409 | .7637 | .7057 | .7522 |
| 9075 | .7558 | .7505 | .7108 | .7531 |

**samples 口径 71.12±0.50（与旧表逐位吻合，源完整）；harmonic 口径 75.40±0.25（ddof=1）**。
75.40 > AF-opt 74.23 → **Example-F1 最优模型翻转为 DBond-GT**。

**五处同步（20261008 完成）**：① 主表 GT 行 71.12±0.50→**75.40±0.25**（bold 归 GT，
underline 归 AF-opt 74.23，AF 撤 underline）；② §3.1 最优句加 Example-F1（AF-opt 处删该项）；
③ 结论句改「Subset Accuracy 以 AF-opt 最高（5.51%），Example-F1 以 GT 最高（75.40%）」；
④ 摘要最优清单加 Example-F1；⑤ paper_tables_20260909.md 冻结表已注记。
**注意**：GT-Ens 行 71.69 与 9-seed 行仍为 samples 口径（论文未引用；若将来引用须同法重算）。

## 15.17 双侧片段读出和残基化学编码（2026-10-17，代码已验证，训练待运行）

目标是检验：在既有 theory + FiLM 主线上，显式提供断点两侧的上下文、以及特殊残基的结构描述，是否改善有效肽键预测。**本节没有新增性能结果；冻结论文数值保持原样。**

### 实现与实验变量

| 实验臂 | 配置（位于 `graph_transform/config/`） | 新增参数 | 总参数 |
|---|---|---:|---:|
| 原模型对照 | `pre_synthesis_theory_film_arch_control.yaml` | 0 | 1,706,153 |
| 双侧片段 | `pre_synthesis_theory_film_arch_fragment.yaml` | 65,857 | 1,772,010 |
| 化学编码 | `pre_synthesis_theory_film_arch_chemistry.yaml` | 9,024 | 1,715,177 |

三臂均复制 `pre_synthesis_fold1222_theory_film.yaml` 的训练设置，仅改变声明的新分支开关、实验名称说明及输出/cache目录。数据划分、种子、图半径、理论特征、FiLM、BCE、学习率、早停和固定0.5阈值不变。正式比较用5组重复序列级80/20划分；训练条件组折叠、测试逐谱口径保持不变。两分支组合在代码中支持，但暂不列为首轮训练臂。

- **双侧片段**：先将 GNN 的每个真实残基状态投影为64维；对键 `i` 分别计算 N侧 `0..i` 和 C侧 `i+1..L-1` 的均值。将有序的两侧均值、差、乘积、两侧长度和局部相邻残基状态送入小MLP，输出加在原logit上的修正量。池化显式排除global节点与padding，按FP32分别做正反向累积。这里池化的是**经过全图消息传递的上下文状态**，不是独立碎片模拟，也不是 b/y 离子监督。
- **化学编码**：17维固定描述（C/H/N/O/S元素数 + 12个官能团/环特征）经 `17→32→256` 投影加到原节点编码上。保留原AA身份embedding和旧4维物化特征；因此这是检验化学信息的增量，而非替换/纠正原特征。覆盖20种标准残基与 B=D-Dap、O=D-Orn、X=3-(3-pyridyl)-D-Ala、Z=D-Cha。组成采用游离氨基酸减H2O；24项重算质量与项目既有理论质量表相符（容差1e-5 Da）。固定缩放不读取数据统计；不填猜测的pKa、气相碱性或确定碎裂规则。
- 两分支末层均为零初始化；新层在旧模型初始化完成后创建，并独立保存/恢复CPU随机状态，保证共享骨干与单独/组合分支的初始化可比。默认关闭，旧配置与旧权重仍可用。新增checkpoint分支键检查，拒绝新旧分支配置不匹配时的静默部分加载。
- 顺带修复全局节点打包的AMP dtype不一致：scatter前转为残基容器dtype。FP32为恒等操作。**旧FiLM末层的零初始化实际会被后续Xavier覆盖**；本轮保留这一历史行为，三个臂完全一致，未把FiLM初始化修正混入实验。

### 已完成验证

本机 Windows / Python 3.12.8 / PyTorch 2.12.0+cpu，`test_architecture_extensions.py` 的13项测试全部通过。使用真实 GraphBuilder 与 GraphDataLoader collate 的合成异长序列，覆盖开关组合、有无global、池化方向与掩码、化学表、非零分支的批次/观测输入扰动不变性、梯度、CPU BF16、权重往返和错误checkpoint配置拒绝。配置测试确认三臂仅有声明的差异。

另外加载修改前的模型文件，对正式256维、3GCN+2GAT配置进行对拍：关闭新分支时，参数及FP32前向输出在eval和train模式均逐值一致。三份正式配置通过runtime预合成掩码处理和合成批次前向，实际state/env mask为 `[T,T,F]` / `[T,F]`。语法编译与修改文件的 `git diff --check` 通过。没有启动训练、生产评估或GPU作业；CUDA FP16和真实数据管线需在pod验证。

### 手动运行

先将本节列出的源码、配置与测试同步到pod项目 `/mnt/pvc/graphtrans/DBond`。可以使用本次交付的补丁：在项目根目录先 `git apply --check <补丁路径>`，成功后再 `git apply <补丁路径>`；不要对有冲突的远端改动强制覆盖。

在pod项目根目录，先运行小批验证（第二项只读少量真实训练样本，并执行一次优化步骤，供用户手动运行）：

```bash
python graph_transform/scripts/test_architecture_extensions.py
python graph_transform/scripts/smoke_test_pipeline.py \
  --config graph_transform/config/pre_synthesis_theory_film_arch_fragment.yaml \
  --csv dataset/5fold_folded/1222.train.fbr.shuffle.multi.csv \
  --n_rows 16 --batch_size 8 --device cuda
python graph_transform/scripts/smoke_test_pipeline.py \
  --config graph_transform/config/pre_synthesis_theory_film_arch_chemistry.yaml \
  --csv dataset/5fold_folded/1222.train.fbr.shuffle.multi.csv \
  --n_rows 16 --batch_size 8 --device cuda
```

通过后依次运行三个臂（每臂五张卡，每卡一个划分）：

```bash
python graph_transform/scripts/train_5fold_parallel.py \
  --config graph_transform/config/pre_synthesis_theory_film_arch_control.yaml \
  --fold_data_dir dataset/5fold_folded --gpus 0,1,2,3,4
python graph_transform/scripts/train_5fold_parallel.py \
  --config graph_transform/config/pre_synthesis_theory_film_arch_fragment.yaml \
  --fold_data_dir dataset/5fold_folded --gpus 0,1,2,3,4
python graph_transform/scripts/train_5fold_parallel.py \
  --config graph_transform/config/pre_synthesis_theory_film_arch_chemistry.yaml \
  --fold_data_dir dataset/5fold_folded --gpus 0,1,2,3,4
```

若pod的环境未激活，将 `python` 替换为该项目已有的虚拟环境解释器。每臂开始前确认上一臂五个划分均显示OK；现有调度器可能在子任务失败后继续汇总，不能只依据进程退出码判断成功。结果目录为 `checkpoints/graph_transform/pre_synthesis_theory_film_arch_<arm>/5fold/<timestamp>/`，读取5个划分齐全的 `5fold_metrics.csv` 和 `5fold_summary.csv`。

报告每个划分相对control的有效键micro-F1、bondacc、ROC-AUC差值，另列q_seq和q_brier，检查改善是否由单个划分驱动。Example-F1沿用§15.16：每划分先取谱平均P/R再调和，不能直接混用samples-F1列。不要将重复划分当统计独立样本，不根据单折涨跌宣称有效。架构/超参数决策应依据训练内部验证集，测试用于固定方案的最终报告；尚无证据决定是否组合两分支或替换论文主模型。

### storage 知识依据与 project_work_report

`project_goal`：把storage知识库中可迁移的碎片上下文和化学组成思路落实为两个独立、可复现的架构实验。`completion_status`：实现与本机验证完成，训练及性能结论待运行。`target_kb`：`kb:storage`，根目录 `G:/valhalla/valhalla_storage_share_root_20260830`。本轮仅使用当前库，未写回知识库。

`knowledge_used / source_records`（页码为PDF物理页）：

| 实体与资源身份 | 公共表现文件与定位 | 原文支持的事实及迁移边界 |
|---|---|---|
| `ent_000004 → res_000119 → file_000119` | `Library/public_resources/res_000119/authoritative.pdf`，AlphaPeptDeep，第2、10页 | 使用基于化学组成的PTM表示及条件元数据嵌入；本实验把“组成描述”迁移到非标准残基，是项目假设，原文没有验证本项目的D残基收益。 |
| `ent_000036 → res_000151 → file_000151` | `Library/public_resources/res_000151/authoritative.pdf`，Prosit，第11页 | 序列编码、charge/NCE条件融合、按b/y及电荷组织输出并屏蔽不可能通道。本轮保留既有条件输入，只借鉴显式碎片结构；没有复制Prosit网络或新增离子强度监督。 |
| `ent_000024 → res_000140 → file_000140` | `Library/public_resources/res_000140/authoritative.pdf`，Fragmentation Pathways of Protonated Peptides，第3–4页 | 可见碎片涉及断裂、质子保留与前后续反应，不能仅等同局部键断裂。因此提出两侧上下文读出；原文不支持“均值池化必然提高预测”的结论。 |

机器映射来自 `Wiki/Wiki_storage/.registry/machine/entity_resource_map.yaml` 对应 `entity_id/resource_id` 条目，均为 `relation: derived_from`。旧表没有 `map_id/evidence_type/locator` 字段；**map_id缺失，未臆造**，使用原条目的 `evidence_location` 与上述实际PDF页核验补足定位。三资源已核验active成员、未排除/未黑名单，资源注册状态active，authoritative公共副本present；未读取source_copies。

`created_files`：`graph_transform/models/fragment_readout.py`、`graph_transform/models/residue_chemistry.py`、`graph_transform/scripts/test_architecture_extensions.py`、上表三份YAML。

`modified_files`：`graph_transform/models/graph_transformer.py`、`graph_transform/models/utils.py`、`docs/experiment_log.md`（仅追加本节，保留已有修改）。

`unresolved_questions`：实际五组划分效果、跨种子稳定性、CUDA兼容性；离子通道监督需要另行确认原始谱图/离子匹配标注；化学描述不能覆盖所有结构异构或完整碎裂动力学；机器映射旧表缺map_id。`suggested_knowledge_writeback`：取得完整训练证据后，可另行用 `ingest_engineering` 沉淀实验实现与结论；本次未摄入。

## 15.18 架构三臂首轮结果审计（2026-10-18）

**结论：首轮结果不足以采用双侧片段或残基化学编码，也不足以支持直接叠加两者。** F1相对本轮control仅增加0.0512和0.0434个百分点；平均AUC均未提高，概率误差均增加。这是本次实现与训练设置下的结果，不等于否定所有片段建模或化学表示方法。§15.17的训练待运行状态现已更新为“三臂训练完成，结果已审计”；论文主表保留历史模型数值。

### 数据来源与完整性

pod项目根目录 `/mnt/pvc/graphtrans/DBond`，三臂目录均位于 `checkpoints/graph_transform/`：

- control：`pre_synthesis_theory_film_arch_control/5fold/20261017_132121`
- fragment：`pre_synthesis_theory_film_arch_fragment/5fold/20261017_135040`
- chemistry：`pre_synthesis_theory_film_arch_chemistry/5fold/20261017_142454`
- 历史参照：`pre_synthesis_theory_folded_film/5fold/20260907_124432`

本次只读访问pod，未启动训练或模型推理。读取汇总、逐划分CSV、配置、日志、训练曲线和预测CSV首行元数据。15个训练均有唯一主checkpoint、正常完成记录，关键曲线无NaN/Inf；各臂5个划分齐全。逐划分seed一致，保存的train.csv/val.csv字节级SHA256一致；相同划分的历史训练/验证文件也一致。三臂配置差异仅为声明的新开关、名称和产物目录；当前pod模型及指标相关7个源码文件与本机版本匹配。预测元数据指向对应 `best_model.pt`，阈值0.5。这里未重新扫描全量预测或重做推理。

逐指标复算均值、std和num_folds，与原summary相符；验证accuracy=lab_acc_mi=1−hamming_loss及f1=lab_f1_mi。原summary采用ddof=0，以下为与论文一致的**样本标准差ddof=1**。Ex-F1每个划分先计算 `2*ex_precision*ex_recall/(ex_precision+ex_recall)`，再跨划分平均；原 `ex_f1` 列是另一口径，不能直接代入。

### 五组重复序列级划分的描述性结果

准确率、F1、Ex-F1以百分数列示；其余为原尺度。均值±样本标准差，不是置信区间，五个划分也不是统计独立样本。

| 指标 | control | fragment | chemistry |
|---|---:|---:|---:|
| 有效键准确率 | 79.842 ± 0.329 | 79.758 ± 0.462 | 79.834 ± 0.313 |
| 有效键micro-F1 | 79.694 ± 0.566 | 79.745 ± 0.662 | 79.738 ± 0.526 |
| ROC-AUC | 0.880686 ± 0.002944 | 0.880606 ± 0.005342 | 0.880682 ± 0.003050 |
| Ex-F1调和口径 | 75.288 ± 0.373 | 75.081 ± 0.415 | 75.350 ± 0.301 |
| 序列级Spearman | 0.61582 ± 0.05051 | 0.62449 ± 0.07720 | 0.61258 ± 0.05283 |
| q-Brier（越低越好） | 0.067829 ± 0.002062 | 0.069297 ± 0.002606 | 0.068187 ± 0.002485 |
| ECE（越低越好） | 0.04373 ± 0.00565 | 0.05409 ± 0.00881 | 0.04618 ± 0.00743 |

| 划分 | fragment ΔF1（百分点） | chemistry ΔF1（百分点） | fragment Δq_seq | chemistry Δq_seq |
|---|---:|---:|---:|---:|
| 1222 | -0.3979 | +0.0274 | -0.03172 | -0.00950 |
| 2252 | +0.3235 | +0.0348 | +0.02692 | -0.00377 |
| 3514 | +0.3140 | -0.0590 | +0.04686 | -0.00183 |
| 6072 | -0.2006 | +0.1530 | -0.00418 | -0.00444 |
| 9075 | +0.2171 | +0.0608 | +0.00546 | +0.00332 |

### 两个分支的判断

**fragment**：F1在3/5划分提高，但accuracy仅1/5提高，Ex-F1平均下降0.2064个百分点。precision下降0.4331个百分点、recall上升0.6062个百分点，预测阳性率上升0.6731个百分点；这与固定阈值下更偏向阳性的权衡相符，不能由F1小涨推断整体判别变强。q_seq平均+0.008668，主要来自2252、3514；移除3514后，其余四组平均Δq_seq为−0.000881（仅作敏感性描述，不替代正式五组汇总）。q-Brier增加0.001468，4/5划分恶化；ECE由0.04373升至0.05409。概率质量没有净收益。

**chemistry**：F1虽4/5提高，但均值仅+0.0434个百分点，AUC差为−0.00000382。precision下降0.1350个百分点、recall上升0.2563个百分点。q_seq在4/5划分下降，均值−0.003247；q-Brier在4/5划分恶化，ECE在5/5划分恶化。Ex-F1小涨0.0619个百分点，不足以抵消排序和概率误差的代价。

q-Brier是整体概率误差，并非纯标定误差；ECE提供另一个标定诊断。两个分支均增加召回，现有汇总没有证明其错误互补，因此暂不建议直接运行组合臂。后续可以用现有预测按断点位置、片段长度、BOXZ邻键与charge/NCE分层，检验是否存在被全局均值掩盖的局部收益，再决定是否重设计。

从实现看，片段分支池化的是已经经全图消息传递的状态；化学描述也是残基身份的确定性函数。二者可能主要增加已有信息的另一种表达，而没有提供新的监督或观测，这是与结果相容的解释，**尚非已验证的失败原因**。本轮不能据此诊断“过拟合”，也不能由train/val loss直接比较得出过拟合，因为训练使用软q、验证loss使用硬标签。

### 本轮control与历史冻结值

本轮control F1=79.6941%，历史film=79.7979%，相差−0.1038个百分点；accuracy相差−0.1783个百分点，AUC相差−0.001856。数据和名义超参数相同并不意味着重新训练逐值复现；本轮架构归因应对照同期control，不能用较高的历史值替代control，也不能把control漂移归因于新增分支。历史与本轮在2252、3514的主checkpoint epoch分别为84/97与61/61；漂移原因未被本次只读审计完全定位。

另发现现有保存规则将主checkpoint更新与 `min_delta=0.001` 绑定：只有验证F1比已保存值高超过0.1个百分点才替换。若提升小于该值，就算是新的最高F1也不保存。例如control-2252第84轮验证F1比保存的第61轮高0.0583个百分点，但仍使用第61轮；fragment-1222第62轮比保存的第39轮高0.0749个百分点。所有臂遵守同一规则，本轮比较仍按既定协议；但这个选择粒度大于本轮架构平均收益。

若下一轮研究微小收益，应将“保存验证集最高F1”与“早停的min_delta/patience”分开，并对所有比较臂统一重跑；这属于新的训练协议实验，不能在测试结果上事后挑epoch或阈值，也不能保证能提升性能。当前没有修改保存逻辑或启动该实验。

### 本地产物与后续

原始快照、逐划分CSV、训练曲线、配对差值、统一口径均值/标准差和审计结果位于 `F:/codexout/dbond_arch_results_20261018/`。主要入口：`comparison_summary.csv`、`audit_conclusions.json`、`raw_snapshot.json`、`run_audit.json`、`training_details.json`；每臂子目录保存原始汇总与 `verified_metrics.csv` / `delta_vs_control.csv`。这些文件是本次结果快照，不是Valhalla知识库资源；未自动摄入知识库。论文表仅追加实验附注，历史主表不替换。

## 15.19 图传播前的序列顺序编码：第二轮三臂（2026-10-18）

状态：代码、本机/CUDA验证、15次训练及结果审计已全部完成，完整结果和否定性结论见§15.20。用户明确授权本地提交推送、pod拉取、启动训练及完成后自动分析；这次授权覆盖旧AGENTS.md的手工启动训练约定。Valhalla操作为project_work，知识库只读。

### 假设与知识来源

上一轮图传播后的均值汇聚没有稳定收益。本轮检验在图传播前显式编码有方向的残基次序：局部多尺度模式与两端递归上下文。两者都只重表征已有序列，不增加观测特征或标签，不能预先声称提供新信息或模拟了真实碎裂机制。

| 实验臂 | 输入端残差分支 | 新增参数 |
|---|---|---:|
| seq2_control | 无 | 0 |
| seq2_multiscale | AA embedding 64维 → 并行Conv1d核宽3/5/7、各32通道、SiLU → 96维拼接 → Linear(96,256) | 55,648 |
| seq2_bigru | AA embedding 64维 → 单层双向GRU、每方向48维 → Linear(96,256) | 57,664 |

分支加到NodeEncoder输出、在既有FiLM和GCN/GAT之前；末端投影零初始化，保持初始输出及旧权重完全一致，构建分支不消耗主随机数序列。卷积输入/输出显式屏蔽padding，GRU按真实长度打包，不直接编码global node。上一轮fragment/chemistry均关闭。

知识库根目录 `G:/valhalla/valhalla_storage_share_root_20260830`，实际读取公开authoritative.pdf：

- PredFull：`ent_000048 → res_000204 → file_000204`，`Library/public_resources/res_000204/authoritative.pdf`物理第5页及第13页Fig.1，8路序列卷积核宽2–9。原文完整谱图模型约1900万参数；本轮3/5/7和小通道是工程假设，原文未证明其对本项目D肽断键预测有效。
- Prosit：`ent_000036 → res_000151 → file_000151`，`Library/public_resources/res_000151/authoritative.pdf`物理第11页，BiGRU/GRU/attention编码及条件融合。本轮只迁移有方向的顺序编码，不复现其完整网络或离子强度监督。
- PepNet边界核对：`ent_000052 → res_000208 → file_000208`，物理第8–9页，TCN处理的是谱图m/z轴，不作为本轮已知序列卷积的直接证据。

身份链由 `Wiki/Wiki_storage/.registry/machine/entity_resource_map.yaml` 的derived_from条目核验；旧表缺map_id，未补造。三资源均active、本地成员、未excluded/blacklisted且公共副本present。未读取source_copies、未写入知识库。

### 共同训练协议与判读规则

三臂统一使用 `training.checkpoint_selection: max_f1`：每次严格刷新验证F1最高值即保存；早停另用min_delta=0.001/patience=25的锚点。旧配置默认legacy，保留旧保存行为。本轮采用新同期control，不能把相对历史模型的差异单独归因于架构。保存选择状态供恢复核对；原框架没有保存scheduler/AMP scaler/RNG，恢复不宣称等同连续训练。

其他设置固定：5组重复序列级80/20划分、训练内部验证划分及seed规则不变，条件组折叠、q软标签、group_uniform、pre输入屏蔽、theory+FiLM、3GCN+2GAT、固定阈值0.5、300轮上限、原学习率/调度/损失/图结构。配置为 `graph_transform/config/pre_synthesis_theory_film_seq2_{control,multiscale,bigru}.yaml`。

主要判读为有效键micro-F1的五组均值及配对差，附accuracy、AUC、Ex-F1调和口径、q_seq和q-Brier。将“值得进一步复验的分类候选”事先定义为相对同期control平均F1至少+0.20个百分点、至少4/5划分F1提高，同时平均AUC不降、q-Brier不升；这是工程筛选门槛而非显著性检验。未达门槛时如实报告局部变化，不称稳定提升，不用测试集挑阈值/epoch。排序用途另外报告q_seq及概率误差权衡，不以单划分涨幅决定采用。五组共享序列，不进行独立样本t检验。

### 实现与验证

`created_files`：sequence_adapter.py、3份seq2 YAML、test_sequence_adapters.py、training/checkpoint_selection.py、test_checkpoint_selection.py、run_architecture_batch.py、summarize_architecture_batch.py、test_architecture_batch.py（均位于graph_transform对应子目录）。`modified_files`：models/graph_transformer.py、models/utils.py、scripts/train_graph_model.py、scripts/train_5fold_parallel.py、本台账。

本机验证：新分支5项（零初始化、RNG、padding/NaN、梯度、CPU bf16、checkpoint匹配、真实配置参数量）、保存策略10项、批处理/汇总6项、既有架构13项，共34项通过。批处理任一臂训练或汇总失败即停止；固定输出目录并记录配置快照、SHA256、git版本、运行状态；完成后自动校验15个结果并输出均值、ddof=1标准差与配对差。CUDA和真实数据单步验证见下方启动记录，正式训练及结果审计见§15.20。

`unresolved_questions`：CUDA兼容性已验证、完整三臂实际收益已在§15.20报告；单种子之外的稳定性和低样本量下其他迁移结构的适用性仍未解决。`suggested_knowledge_writeback`：完整结果经审计后可另行摄入工程结论；本任务未执行知识库摄入。

### 实际启动记录

北京时间2026-10-18 01:25（pod UTC目录时间20261017_172512）启动，运行根目录 `/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/seq2_batch/20261017_172512`，主PID 200817，启动日志 `logs/seq2_batch_20261017_172512.log`。代码提交 `770ff82b6fa5bf977c9c5e02b18834efcf5e3566`，本地push、pod fast-forward pull后工作区clean；本批次使用GPU0/4，其余GPU已有任务。

CUDA验证：新分支5项测试通过，包括FP16 autocast前后向；两种分支分别通过真实折叠数据16行、batch=8的单步训练，软标签/组权重与pre输入屏蔽断言通过。GPU测试中修正了测试自身的eval模式反向问题（cuDNN GRU要求train模式），全精度跨batch比较关闭TF32；完整图输出允许CUDA原子归约的1e-6舍入误差，权重和RNG仍精确相等。模型实现未因这些测试修正而变化。

启动只读审计：3份配置快照和manifest四脚本SHA256匹配；control 1222/2252的train.csv、val.csv与上一轮对应文件逐字节相同、seed相同、序列重叠为0。1222第16轮F1相较early-stop anchor仅+0.0008422仍保存主checkpoint，bad_validations=2，证实max_f1保存和早停分别生效。批次completed后的正式结果见§15.20；上述启动证据本身不作为性能结论。

## 15.20 序列顺序编码第二轮完成与判定（2026-10-18）

**结论：多尺度序列卷积和BiGRU均未获得整体提升，不进入候选复验，也不替换现有主模型。** 相对本轮同期control，两者有效键micro-F1均值分别−0.0448和−0.0312个百分点，均仅2/5划分提高；平均AUC下降、q-Brier上升。§15.19预先规定的四项筛选条件，两臂均未满足任何一项。此结论限定于本轮小型残差分支、现有数据和训练设置，不是对原论文完整架构或所有序列建模方法的否定。

### 完成情况与证据审计

批次北京时间2026-10-18 01:25:12启动、02:54:35完成，耗时约89分23秒。三臂×五组重复序列级划分共15次训练，三臂returncode均为0，状态文件为completed、error为空。GPU0/4、最大同时2个训练；代码提交 `770ff82b6fa5bf977c9c5e02b18834efcf5e3566`。训练前本地push并在pod完成fast-forward pull；完成状态由本地SSH监视器自动捕获后执行结果下载与分析，没有手动挑选测试epoch或阈值。

运行根目录：`/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/seq2_batch/20261017_172512`。三臂位于 `arms/01_pre_synthesis_theory_film_seq2_control`、`arms/02_pre_synthesis_theory_film_seq2_multiscale`、`arms/03_pre_synthesis_theory_film_seq2_bigru`；各臂 `cv/5fold_metrics.csv` 是原始逐划分汇总。批处理自动生成 `summary/summary.json` 及CSV；本地快照位于 `F:/codexout/dbond_seq2_results_20261017_172512`，原始文件按bytes下载，三个metrics CSV的SHA256均与远端summary记录一致。

完整只读审计379项通过、0项失败：核对训练git版本、脚本/配置SHA256、仅声明的配置差异、15个产物及曲线有限值、连续epoch、三臂逐划分seed与train/val文件SHA256一致、训练内部train/val序列不重叠、pre输入屏蔽、阈值0.5、主checkpoint与历史验证F1严格最高值及首次达到该值的epoch一致；独立重放checkpoint选择/早停状态，并检查metrics与预测CSV首行checkpoint元数据。审计通过不意味着扫描了全部权重或全部预测；本轮未重新推理。指标恒等式及六指标均值/样本标准差另从原始CSV复算。

下表为“主checkpoint epoch / 训练最终epoch”；主checkpoint不是最后一轮，也不是测试集最优轮。

| 实验臂 | 1222 | 2252 | 3514 | 6072 | 9075 |
|---|---:|---:|---:|---:|---:|
| control | 74 / 99 | 84 / 109 | 85 / 86 | 80 / 105 | 83 / 96 |
| multiscale | 74 / 99 | 61 / 86 | 106 / 122 | 80 / 105 | 70 / 76 |
| bigru | 74 / 99 | 61 / 86 | 61 / 86 | 63 / 71 | 71 / 96 |

### 统一口径结果

均值±样本标准差（ddof=1）。所有分类指标仅统计有效肽键；Ex-F1每个划分先由谱平均P/R计算调和均值，再跨划分平均，不能直接使用原CSV的samples-F1列。五组是重复序列级80/20随机划分且共享序列，不视为统计独立样本；每个划分只有一次初始化，不是五种子复验。本轮checkpoint协议改变，因此架构归因只使用同期seq2_control，历史主表保持冻结。

| 指标 | seq2_control | seq2_multiscale | seq2_bigru |
|---|---:|---:|---:|
| 有效键准确率（%） | 79.984 ± 0.373 | 79.895 ± 0.417 | 79.788 ± 0.364 |
| 有效键micro-F1（%） | 79.778 ± 0.640 | 79.733 ± 0.670 | 79.747 ± 0.570 |
| ROC-AUC | 0.882175 ± 0.003550 | 0.881722 ± 0.004414 | 0.880226 ± 0.004037 |
| Ex-F1调和口径（%） | 75.367 ± 0.270 | 75.285 ± 0.350 | 75.406 ± 0.305 |
| 序列级Spearman | 0.62155 ± 0.06124 | 0.62391 ± 0.06162 | 0.61582 ± 0.04955 |
| q-Brier（越低越好） | 0.067317 ± 0.002203 | 0.067807 ± 0.002787 | 0.068297 ± 0.002727 |

| 划分 | multiscale ΔF1（百分点） | BiGRU ΔF1（百分点） | multiscale Δq_seq | BiGRU Δq_seq |
|---|---:|---:|---:|---:|
| 1222 | +0.1877 | +0.1162 | +0.01769 | +0.01334 |
| 2252 | -0.1607 | -0.0405 | -0.00036 | -0.00700 |
| 3514 | -0.0988 | -0.0645 | +0.01661 | -0.01610 |
| 6072 | +0.0140 | -0.3859 | -0.00738 | -0.01936 |
| 9075 | -0.1660 | +0.2187 | -0.01478 | +0.00048 |

### 判读与下一步

多尺度卷积：F1 −0.0448个百分点、accuracy −0.0893个百分点、AUC −0.000453、Ex-F1 −0.0821个百分点。q_seq均值+0.002355，但只有2/5划分提高；q-Brier平均增加0.000491，虽3/5划分小幅降低，2252的较大恶化使总体变差。没有证据支持采用或与上一轮分支叠加。

BiGRU：F1 −0.0312个百分点、accuracy −0.1964个百分点、AUC −0.001948；AUC在4/5划分下降，q-Brier在4/5划分升高，均值+0.000980。q_seq平均−0.005728。Ex-F1虽4/5提高、均值+0.0387个百分点，但谱平均precision下降0.5512个百分点、recall上升0.6822个百分点；这项局部变化不足以抵消有效键分类、排序及概率误差的损失。q-Brier是概率误差，不等同于纯校准误差。

四项门槛是“平均ΔF1≥+0.20个百分点、F1至少4/5正向、平均AUC不降、平均q-Brier不升”，属于工程筛选而非显著性检验。两臂四项均未通过，故不宣称稳定提升、不据此替换论文主表。新分支保留为默认关闭的可复现实验代码。保存最高验证F1与早停分离的功能保持显式配置，旧配置仍采用legacy行为。

结合§15.18和本轮，四种小型结构扩展都没有给出足以采用的整体收益。它们主要重表征同一序列信息；“缺乏更细监督可能是瓶颈”是待验证的解释，不能据此断言已到性能上限或已经证明过拟合。下一项优先研究应是知识库Prosit/PredFull所使用的离子级信息如何引入本项目：按b/y、碎片电荷及必要中性丢失组织可观测性/强度监督，再与键级任务联合训练。其收益尚未验证。

已只读检查数据可行性：当前MiPD513训练CSV没有可直接训练的离子通道或碎片强度数组，`intensity`来自MGF前体PEPMASS[1]（`PBCLA/mgf2csv.dbond_m.py`），不是碎片峰强度；`PBCLA/utils.py`已有b/y×1+/2+×无/H2O/NH3的理论离子生成，但`PBCLA/pbcla.py`最终把匹配结果OR到键级标签，现有true_multi无法反推出每个离子通道。在已检查的项目及pod目录中未定位到完整MiPD513原始MGF及稳定行映射；这不是对所有存储位置的穷尽检查。

本机存在带峰注释的外部MSP：`G:/msp-demo/FTMS_HCD_20_annotated_2019-11-12.msp`，可进一步评估离子级预训练。现有 `graph_transform/scripts/nist_msp_to_dbond_csv.py`未保留通道注释；若使用该资源，须先冻结预训练/验证/外部测试的序列隔离，不能训练后继续把同一NIST数据当独立外测，也不能假定标准L肽的强度规律直接适用于D肽和BOXZ。此次未启动新的数据转换或预训练。

### 交付记录

`project_goal`：结合storage知识实现并在pod完成两个新架构的配对实验，自动取得完成结果并判断提升。`completion_status`：代码、本机及CUDA验证、push/pull、15次训练、结果下载与审计、描述性比较全部完成；性能结论为两种新架构均未获得整体收益。`target_kb`：kb:storage，只读；`knowledge_used/source_records`见§15.19的PredFull/Prosit身份链与实际PDF定位。

本地证据入口：`summary/summary.json`、`audit.json`（379项）、`decision.json`（预定规则逐项判定）、`summary/figures/seq2_paired_differences.png`及同名矢量PDF。图仅显示五组配对差及均值，不绘制独立样本置信区间；已查看最终PNG确认标签完整。论文表文件仅追加本轮实验附注，不替换历史冻结主表。

`unresolved_questions`：跨初始化种子稳定性未复验；未验证更大/不同位置的序列编码器；离子级监督需要原始谱图、通道匹配和数据隔离；当前结论不构成性能上限估计。`suggested_knowledge_writeback`：可将这两轮完整负结果与checkpoint选择修正作为后续工程摄入候选；本次没有写回Valhalla知识库。

## 15.21 读出阶段的条件融合：第三轮配对实验（2026-10-18）

状态：代码、本机/CUDA验证、15次训练及结果审计均已完成，结果见§15.22。本节保留训练前固定的方案与判据，沿用用户此前代码实现、push/pull、训练与分析的授权。知识库操作仍为project_work，仅使用kb:storage，只读。

### 假设、对照与来源

本轮检验charge/NCE在最终键判别特征上的直接非线性交互。现有FiLM位于GNN前，global/state/env/edge路径也已携带条件信息，因此不是增加新观测、不是声称原条件信号消失。上一轮序列CNN/GRU、此前fragment均值和固定化学分支均关闭，旧FiLM的历史初始化行为保持不变。

新模块接在原bond_head第一层Linear→ReLU之后、原Dropout之前的256维键隐表示h上。令x=SiLU(Linear(256,32)(h))，c=Linear(32,32)(SiLU(Linear(2,32)([0.1×charge,0.01×NCE])))。concat与gated分支分别将[x,c]与[x×(1+tanh(c)),c]送入相同的Linear(64,32)→SiLU→Linear(32,1)，输出作为原logit的残差。门值范围为(0,2)，这是本项目对乘性交互的有界实现，不是原论文的原样复制。

| 实验臂 | bond_condition_mode | 新增参数 | 比较问题 |
|---|---|---:|---|
| cond3_control | none | 0 | 同期原模型、重跑统一协议 |
| cond3_concat | concat | 11,489 | 在读出时显式拼接条件是否有收益 |
| cond3_gated | gated | 11,489 | 同容量下乘性调节是否优于拼接 |

末层零初始化，不增加dropout；分支在legacy初始化结束后通过fork_rng构造，预期初始函数、共享权重和主随机状态与control一致，待测试核验。旧bond_head参数名和默认关闭路径不变。两种模式参数形状相同，checkpoint必须另行核对融合模式，禁止仅凭state_dict形状静默加载。

知识库根目录 `G:/valhalla/valhalla_storage_share_root_20260830`，证据均为公开authoritative PDF的物理页码：

| 实体→资源→文件 | 机器映射与原文定位 | 支持的事实及边界 |
|---|---|---|
| ent_000036→res_000151→file_000151 | `Wiki/Wiki_storage/.registry/machine/entity_resource_map.yaml`对应条目；`Library/public_resources/res_000151/authoritative.pdf`第11页，Fragmentation model—Model architecture | Prosit把序列latent与charge/NCE编码逐元素相乘后交给decoder。支持条件乘法融合的先例，没有单独证明本项目的晚期gated优于concat。 |
| ent_000004→res_000119→file_000119 | 同机器map对应条目；`Library/public_resources/res_000119/authoritative.pdf`第10页，MS2 model | AlphaPeptDeep将序列/PTM与charge、NCE、instrument embedding拼接。将其改为DBond最终键读出残差是项目假设，不是该论文的实证结果。 |

实体active；资源为local active、未excluded/blacklisted、registry lifecycle active、公共副本present。旧map缺map_id/evidence_type/locator，未伪造；原文证据位置以上述实际PDF核验补足。未读取source_copies、未写入知识库。

只读检查上一轮control缓存的五份内部train.csv（仅读seq/charge/nce）：每份6098–6179条件组、328–330序列，均覆盖charge={2,3,4,5,6}与NCE={20,30,40,50}全部20个组合；每格至少271–282条序列（各划分的最小值），每序列条件数中位数20。这支持现有数据可训练条件交互，但不保证新分支有效；未用测试标签选择条件划分或函数形式。

### 训练与结果判读预先约定

三份配置：`graph_transform/config/pre_synthesis_theory_film_cond3_{control,concat,gated}.yaml`。沿用seq2_control全部数据/损失/优化参数：五组重复序列级80/20划分、内部验证划分与seed规则、q软标签、group_uniform、theory+FiLM、3GCN+2GAT、pre输入屏蔽、固定阈值0.5、max_f1主checkpoint与独立早停min_delta=0.001/patience=25。各臂仅改变声明的条件模块模式和实验/产物名称；两新臂相同参数量。使用空闲GPU、同臂每划分一个训练；不修改其他作业。

主要指标为有效键micro-F1五组均值和相对同期control的逐划分差；附有效键accuracy、ROC-AUC、Ex-F1调和口径、q_seq与q-Brier，标准差ddof=1。候选筛选规则仍要求同时满足：平均ΔF1≥0.002（+0.20个百分点）、至少4/5划分ΔF1>1e-12、五划分平均ΔAUC≥0、五划分平均Δq-Brier≤0；全部使用未四舍五入值。差值绝对值≤1e-12视为平局。两臂同时通过时均列为待复验候选，不根据本次测试均值指定论文主模型。

另报告gated−concat的逐划分差，作为同容量不同融合函数的描述性比较；各自−control同时包含增加容量/路径的影响，不能全归因为乘法。q-Brier指现有对条件组q的概率误差，并非纯校准误差。五组共享序列，不做独立样本t检验，不称统计显著；单划分单次初始化不能说明跨种子稳定。

本项目多轮筛选已经参考相同测试划分，因此此轮即使预先固定方案，也属于探索性实验；正结果仍需不同初始化及未参与方法筛选的数据验证。训练仅凭内部val选checkpoint，不在test上选epoch、阈值或融合维数；历史冻结论文值保持不变。

### 数据路线的可行性边界

继续检查未发现现有MiPD训练CSV有可直接使用的离子通道；tb/fb/mb/fbr是true_multi的派生统计，不作为新监督来源。此前找到的完整外部MSP有较多序列已被NIST外测使用；按基础序列隔离后，剩余无修饰候选未通过既有质量一致性规则，不能为了预训练跳过质量检查。还有修饰肽候选，但须保留并解析PTM质量和峰注释、重新进行质量与序列隔离审计，不能简单删去修饰当成天然肽。故本轮先运行不需要新标签的条件融合；原始MiPD谱图路径已向用户询问，后续有完整数据再规划离子级监督。

### 实现与验证记录

`created_files`：`graph_transform/models/condition_readout.py`、`graph_transform/scripts/test_condition_readout.py`及三份cond3 YAML。`modified_files`：`graph_transform/models/graph_transformer.py`、`graph_transform/models/utils.py`、本台账。新增模块须验证初始等价/RNG、条件敏感性、屏蔽输入与观测字段不变性、异长/padding/global/批次顺序、梯度与AMP、checkpoint模式匹配；使用真实数据小批验证后才启动正式训练。现有FiLM本身未遵守charge/NCE ablation mask，本轮不混入其修复；新分支单独保证mask语义，完整模型掩码测试需关闭旧FiLM以隔离这一既有边界。

本机Windows/Python3.12/PyTorch CPU：专用测试8项、既有架构回归13项、上一轮序列分支回归5项，共26项通过。覆盖上述行为、CPU BF16前后向、正式三配置仅声明差异及参数量。control=1,706,153参数，concat/gated均为1,717,642。模式标识写入state_dict，CheckpointManager在加载前检查，直接state_dict载入也会拒绝错误/缺失模式标识。git diff --check与四份变更Python语法解析通过；pod CUDA及真实数据单步验证也已完成，见下方实际启动记录。

`project_goal`：验证两个同容量的晚期条件融合方案并自动完成训练与结果判读。`completion_status`：实现、验证、训练与结果判读均已完成，见§15.22。`target_kb`：kb:storage。`unresolved_questions`：本轮总体/跨划分收益已判定为负；跨种子与新留出数据泛化、离子监督的数据来源仍未解决。`suggested_knowledge_writeback`：完整负结果可作为后续工程摄入候选；本轮不自动写回知识库。

### 实际启动记录

代码提交 `e878180edd0951891f58cbde0b4ae7aff7058eb0` 已本地push、pod fast-forward pull。pod CUDA专用8项测试全部通过（含FP16 autocast前后向），concat/gated分别以真实折叠训练CSV的16行、batch=8完成单步训练，软标签、group_uniform组权重和pre输入屏蔽检查通过。

北京时间2026-10-18 14:13:36（UTC 06:13:36）启动，根目录 `/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/cond3_batch/20261018_061336`，主PID 399785，日志 `logs/cond3_batch_20261018_061336.log`。启动前GPU0/1/2均空闲，本批次仅使用这三张卡；三臂依次运行，每臂最多同时3个划分。源码9文件SHA256保存在本地 `F:/codexout/dbond_cond3_launch.json` 和pod `logs/cond3_launch_20261018_061336.json`，完成后核对未变化。

用户并行修改的 `graph_transform/scripts/ensemble_baseline_preds.py` 保持未提交原状，且不在本批训练路径；未触碰其结果或其他GPU作业。批次由脱离SSH终端的后台进程执行。初期本地监视器每30秒检查状态，后按用户要求停止并改成静默完成事件触发（见§15.22）；最终分析只在所有15个结果齐全且批次completed后执行。

### 并行数据诊断补充（不改变本轮方案）

后续一次MSP只读流式诊断发现，上述8,674条无修饰剩余候选（8,270个基础序列，电荷4/5/6分别7,921/706/47条）的质量偏差有一致结构：全部z×Parent−MW在2.0143–2.0148 Da，支持这些记录采用Parent≈(MW+2H)/z的元数据约定；MW与无修饰序列理论质量最大绝对差约0.000524 Da。现有转换脚本对z≥3只补一个质子的分支不能覆盖这一高电荷约定，因此“未通过现过滤”不能直接解释为随机谱图错误或没有可救回数据。

该一致性只是同一条谱记录的头字段交叉检查，不是独立测量证明，更不能证明碎片标签质量。候选已经排除现有NIST外测基础序列，但未来还需排除全513条D肽序列、核验原始来源及碎片注释，并单独冻结新的预训练/验证数据。全部候选Comment中的key=value未发现CE/NCE；未据此推断其他所有头字段缺失，也不把文件名HCD20视为逐谱NCE证据。本轮没有修复/放宽过滤、没有生成预训练集、没有改变原外测。

完整规则、范围、计数和三个元数据例子见 `F:/codexout/dbond_cond3_msp_mass_diagnostic.json`。这为后续预训练提供了一条待QA的具体路线，尚无性能结论。

进一步的一次MSP流式只读QA仅使用原始10份MiPD CSV和现有NIST外测的seq列排除基础序列：核对MiPD共513条、NIST共238,178条；剩余候选没有命中MiPD，仍为8,674条记录/8,270个序列，实际长度11–30。按文件顺序选前100个不同候选序列各第一条谱，共2,660峰；其中严格无中性丢失b/y、碎片电荷1+/2+且可独立计算的注释有2,108个（79.25%），全部绝对误差≤50 ppm，中位1.883、P95 10.492、最大17.811 ppm。其余552峰为中性丢失，未核验不计为错误。计算使用标准b/y理论m/z，没有沿用Parent的特殊约定或直接相信注释里的ppm数值。

此样本不是随机抽样，结果只证明抽查注释的m/z一致性；不证明整库质量、强度可比性、来源独立或逐谱NCE已知，也未完成Parent元数据约定确认。没有生成训练集、修复过滤或改变当前实验。报告和可复现脚本为 `F:/codexout/dbond_cond3_msp_fragment_qa.json`、`F:/codexout/qa_cond3_msp_fragments.py`，报告保存MSP单遍SHA256、序列排除规则及适用范围。

### 重复control的数值一致性检查

cond3与seq2 control的五划分seed及10份内部train/val CSV字节哈希相同，配置仅名称/描述/路径及新增默认关闭的条件分支字段不同；两个训练版本间训练驱动、数据与checkpoint选择代码未变。关闭分支仍使用原bond_head。cond3−seq2的五划分F1差依次为+0.048151、−0.202652、+0.055190、−0.012904、+0.150078个百分点，均值+0.007573个百分点；AUC平均差−0.000141617。单次重复不能估计一般波动分布，以上仅为两次实际运行的差异。

训练代码设置seed，但未找到启用严格确定性的调用。图聚合使用index_add_/scatter_add_/scatter_reduce_，pod本地PyTorch文档提示这些CUDA操作可能不确定；这构成合理的复现性限制，尚未定位此次差异的具体来源，也未比较缓存逐tensor或过程RNG状态。YAML的benchmark字段没有找到对应应用代码，不能据此认定该选项实际开启。详细只读证据见 `F:/codexout/dbond_cond3_control_repeat_diagnostic.json`。当前方案和判据不因这项检查而改变。

## 15.22 晚期条件融合第三轮结果（2026-10-18）

**本轮没有方案同时满足预定四项候选门槛，暂不替换现有主模型。** 完整三臂、每臂五组重复序列级划分，共15次训练已完成。以下报告全部固定方案的结果，不按测试数据修改epoch、阈值或筛除划分。

### 训练完成与审计

北京时间2026-10-18 14:13:36至2026-10-18 15:10:16，耗时56分40秒。GPU0/1/2，三臂returncode均0、批次completed且error为空。训练代码 `e878180edd0951891f58cbde0b4ae7aff7058eb0`；本地push、pod pull、CUDA验证后才启动。远端根目录 `/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/cond3_batch/20261018_061336`；本地证据目录 `F:/codexout/dbond_cond3_results_20261018_061336`。

只读完整审计427项通过、0项失败。核对manifest/配置/训练代码版本，启动时记录的9个源码SHA256及批处理脚本SHA256，三臂配置差异范围，15个checkpoint的配置与持久化融合模式标识，同划分seed与train/val字节级哈希，训练内部序列隔离、pre输入屏蔽、固定阈值0.5；独立重放验证F1最高checkpoint和早停状态，核对预测CSV首行来源及训练曲线有限值。原始指标CSV的哈希在审计时保存，并与下载文件、summary及逐划分audit数值再次绑定核对。未扫描全量模型权重或重做模型推理。

启动审计另核验control前三划分的6个train/val文件与seq2对应文件逐字节相同，seed一致、内部序列重叠0。本轮源码9文件与启动记录一致；另一个任务的ensemble_baseline_preds.py工作区修改不参与本批训练，保持原状。

下表为主checkpoint epoch / 最终训练epoch：

| 实验臂 | 1222 | 2252 | 3514 | 6072 | 9075 |
|---|---:|---:|---:|---:|---:|
| control | 74 / 99 | 68 / 93 | 97 / 122 | 80 / 105 | 82 / 107 |
| concat | 74 / 99 | 61 / 86 | 61 / 86 | 80 / 105 | 77 / 96 |
| gated | 99 / 99 | 61 / 86 | 61 / 86 | 80 / 105 | 83 / 96 |

### 主要判断


晚期条件拼接：相对同期control，平均F1 -0.2012个百分点，F1在0/5划分提高；AUC -0.003032，q_seq -0.012891，q-Brier +0.001917。预定四项门槛通过0/4，未列入待复验候选。

晚期条件乘性调节：相对同期control，平均F1 -0.2672个百分点，F1在0/5划分提高；AUC -0.002661，q_seq -0.000644，q-Brier +0.001689。预定四项门槛通过0/4，未列入待复验候选。

代码版本：`e878180edd0951891f58cbde0b4ae7aff7058eb0`；规则：`docs/experiment_log.md §15.21`。

所有值由 15 份原始划分结果重算并与 summary 校验；下表为均值 ± 样本标准差（ddof=1）。

| 指标 | Control | Late concat | Late gated |
|---|---:|---:|---:|
| 有效键 micro-F1（%） | 79.785 ± 0.626 | 79.584 ± 0.566 | 79.518 ± 0.592 |
| 有效键 accuracy（%） | 79.958 ± 0.402 | 79.716 ± 0.332 | 79.739 ± 0.315 |
| ROC-AUC | 0.882033 ± 0.003927 | 0.879000 ± 0.002932 | 0.879372 ± 0.003267 |
| Ex-F1 调和口径（%） | 75.421 ± 0.346 | 75.192 ± 0.391 | 75.164 ± 0.345 |
| q_seq（序列级 Spearman） | 0.62365 ± 0.06377 | 0.61076 ± 0.05154 | 0.62301 ± 0.03047 |
| q-Brier（越低越好） | 0.067338 ± 0.002387 | 0.069255 ± 0.002304 | 0.069026 ± 0.002440 |

### 相对同期 Control 的逐划分差

| 划分 | Late concat ΔF1（百分点） | Late gated ΔF1（百分点） | Late concat Δq_seq | Late gated Δq_seq |
|---|---:|---:|---:|---:|
| 1222 | -0.20721 | -0.23273 | -0.002866 | +0.051234 |
| 2252 | -0.18896 | -0.13439 | -0.019767 | -0.020105 |
| 3514 | -0.36311 | -0.47618 | -0.040079 | -0.037840 |
| 6072 | -0.05339 | -0.08346 | -0.003280 | -0.009296 |
| 9075 | -0.19314 | -0.40900 | +0.001538 | +0.012784 |

### 四项工程门槛

判定使用未舍入值；F1 获胜要求 ΔF1 > 1e-12。所有门槛必须同时通过。

| 实验臂 | 平均 ΔF1 ≥ +0.20 百分点 | F1 至少 4/5 获胜 | 平均 ΔAUC ≥ 0 | 平均 Δq-Brier ≤ 0 | 待复验候选 |
|---|---:|---:|---:|---:|---|
| Late concat | -0.201162 (未通过) | 0/5 (未通过) | -0.00303244 (未通过) | +0.00191728 (未通过) | 否 |
| Late gated | -0.267150 (未通过) | 0/5 (未通过) | -0.00266135 (未通过) | +0.00168870 (未通过) | 否 |

待复验候选：无。两臂同时通过时同时保留，不依据这次测试均值指定论文主模型。

### 次要描述比较：Late gated − Late concat

三项百分比指标的差值以百分点表示；其余用原尺度。q-Brier 减少计为 gated 获胜；其他指标增加计为获胜。

| 指标差值 | 1222 | 2252 | 3514 | 6072 | 9075 | 均值 ± SD | gated 获胜 / 平局 / 失利 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 有效键 micro-F1（百分点） | -0.0255 | +0.0546 | -0.1131 | -0.0301 | -0.2159 | -0.0660 ± 0.1026 | 1 / 0 / 4 |
| 有效键 accuracy（百分点） | +0.2295 | +0.0607 | -0.0783 | -0.0077 | -0.0902 | +0.0228 ± 0.1305 | 2 / 0 / 3 |
| ROC-AUC | +0.0018940 | +0.0002381 | -0.0001342 | -0.0002828 | +0.0001403 | +0.0003711 ± 0.0008765 | 3 / 0 / 2 |
| Ex-F1 调和口径（百分点） | +0.0403 | +0.1258 | -0.0963 | +0.0552 | -0.2658 | -0.0281 ± 0.1553 | 3 / 0 / 2 |
| q_seq（序列级 Spearman） | +0.054101 | -0.000338 | +0.002239 | -0.006016 | +0.011246 | +0.012246 ± 0.024212 | 3 / 0 / 2 |
| q-Brier（越低越好） | -0.0009337 | -0.0001339 | +0.0000859 | +0.0000416 | -0.0002029 | -0.0002286 ± 0.0004119 | 3 / 0 / 2 |

这是同容量不同融合函数的描述性比较；各臂相对 Control 的差同时含容量/路径效应，不能全归因为乘法。

Ex-F1 先在每划分内由 ex_precision/ex_recall 求调和均值再取均值。q-Brier 是对条件组 q 的概率误差，并非纯校准误差。

五组重复划分共享序列，未进行独立样本 t 检验，不作显著性或置信区间推断。每划分一次初始化不能证明跨种子稳定。

多轮方法筛选已参考相同测试划分，本轮属于探索性实验；正结果仍需不同初始化及未用于方法筛选的数据复验。

### 分支学习诊断与后续方向

完成后的事后只读诊断检查了10个候选主checkpoint：新增分支参数均有限，零初始化的末层权重全部已变为非零，L2范数范围0.3184–0.3738。gated在预先已有的20个charge/NCE组合上，门值合并范围0.6067–1.8712；按abs(tanh(c))≥0.99描述性规则，五划分最高饱和比例为0.0%。这排除了“末层始终全零”及这20条件上的明显门饱和，但不证明分支的因果贡献，也未使用测试样本做消融推理。证据为`learned_readout_diagnostics.json`。

结合前两轮，六种小型结构扩展均未达到采用门槛。本次concat/gated不仅没有达到+0.20个百分点，而且F1在所有五划分均下降；不推荐继续叠加这两个读出分支。下一优先方向是引入离子级监督或预训练信息，但尚不能断言现有瓶颈已被定位。§15.21的外部MSP诊断发现排除现有外测和MiPD序列后仍有8,270个候选序列，前100谱中的2,108个可评估标准b/y峰质量误差均≤50 ppm；这只是有界可行性证据，来源、Parent约定、逐谱NCE及强度质量仍需核验后再定义正式预训练协议。

### 交付与边界

本机26项、pod CUDA专用8项测试通过；两新分支各完成真实折叠数据16行单步训练。完整训练结果由后台批处理自动汇总、本地监视器捕获完成后下载；分析与绘图须通过完整性审计才执行。两个新分支仍通过显式配置选择，历史冻结主表不自动替换。

按用户最新要求，训练期间已停止逐轮进度轮询，改用目录写入事件静默等待；仅完成/失败/连接异常时唤回处理，十分钟超时只作为丢失事件兜底，不读取epoch日志。此次完成事件于北京时间15:10:16返回后才执行最终下载和分析。后续训练沿用完成事件触发方式，避免模型持续查询状态。

证据入口：三臂各自的`cv/5fold_metrics.csv`、`summary/summary.json`、`audit.json`、`decision.json`、`analysis_tables.md`及`summary/figures/cond3_paired_differences.png`/同名PDF。数据路线的可行性诊断保存在`F:/codexout/dbond_cond3_data_feasibility.json`；该诊断不等于穷尽搜索全部原始谱图，也未将质量过滤失败的MSP记录转为训练样本。

`project_goal`：完成两个同容量晚期条件融合方案的实现、push/pull、pod训练和自动判读。`completion_status`：全部完成；性能结论见本节。`target_kb`：kb:storage。`knowledge_used/source_records`：§15.21所列Prosit/AlphaPeptDeep的实体、机器映射、公共资源身份和PDF页。`created_files/modified_files`：见§15.21，另更新本台账与本地论文表附注。`unresolved_questions`：跨种子及未参与筛选的数据上的复验、原始谱图与离子标签来源；多轮结果不构成性能上限证明。`suggested_knowledge_writeback`：可另行摄入完整实现与正/负结果；本轮没有写回知识库。

## 15.23 离子相对强度预训练与迁移：第四轮预定方案（2026-10-18）

状态：用户要求继续优化，沿用本地实现/测试/commit/push、pod pull、训练与完成后分析的授权。使用Valhalla project_work，base/kb:storage，只读。训练提交后采用完成事件静默等待，不逐轮读取日志，不向模型重复报告epoch。当前先冻结数据用途、训练目标和迁移比较，再构建数据及执行验证；实际产物、版本与结果后续追加。

### 研究问题和文献边界

前三轮六种小型结构扩展没有达到采用门槛。本轮检验新增外部离子强度监督能否提供可迁移表征，而非继续增加下游模型容量。一个外部预训练模型供两种迁移范围使用，目标域均保留现有模型结构、参数量和全参数微调协议。外部预训练同时跨标准L肽/特殊D残基化学域及强度/肽键分类任务，收益是待检验假设。

知识库根目录`G:/valhalla/valhalla_storage_share_root_20260830`。机器映射`Wiki/Wiki_storage/.registry/machine/entity_resource_map.yaml`核验`ent_000004 → res_000119 → file_000119`（旧map缺map_id，未补造）；有效local membership、entity及resource lifecycle均active，未excluded/blacklisted，authoritative公共副本present且SHA匹配。实际读取`Library/public_resources/res_000119/authoritative.pdf`：物理第5–6页、Fig.4是AlphaPeptDeep对21类PTM的少样本MS2迁移实证；第10页MS2 model说明逐断裂位b+/b++/y+/y++及修饰中性丢失通道、按每谱最高匹配强度归一、L1损失。该文没有证明本项目的跨化学域/跨输出任务迁移，也没有使用本轮的缺失注释mask规则。

同时核验Prosit（ent_000036→res_000151/file_000151）和PredFull（ent_000048→res_000204/file_000204）的有效公共来源；Prosit第12页具体transfer示例是iRT，不冒充MS2迁移证据；三篇没有找到EMA/SWA做法。未读source_copies，未写回知识库。

### 外部数据用途与固定质量规则

源文件`G:/msp-demo/FTMS_HCD_20_annotated_2019-11-12.msp`。本轮新建独立的library-annotation auxiliary数据集，不改旧NIST外测CSV或既有转换器。仅保留无修饰、标准20AA、长度7–30、报告charge为2–6的候选；从现有NIST外测和原始10份MiPD CSV仅读seq列排除全部序列，并采用I/L合并key作保守隔离与分组。B/O/X/Z是特殊化学残基，不映射成标准氨基酸或通配符。

输入为序列、库中报告charge及序列推导的理论前体m/z。Parent字段不参与监督、过滤或模型输入；这不是确认其约定正确。NCE未知，只用0占位并在node/global/edge全路径屏蔽，预训练关闭FiLM与晚期条件模块。强度/scan/rt不作输入。数据用途明确为库内条件混合下的注释相对强度，不能称已校准NCE下的原始实验谱预测。

逐条核验唯一标准无中性丢失b/y、碎片电荷1+/2+的理论m/z，要求正有限强度和绝对质量误差≤50ppm。输出按N端到C端肽键排列的(L−1)×4数组，通道[b1,b2,y1,y2]；yN映射到0-based键位L−N−1。同槽重复取有效峰最高强度并记录计数，缺失或歧义槽mask=false，绝不当作不存在的负例；每谱按有效匹配槽最高强度归一。无有效槽或无法安全隔离的歧义记录拒绝，所有拒绝原因计数。原`true_multi`仅为GraphDataset兼容占位，不能用于预训练损失或分类评估。

同(seq,charge)重复选择文件中第一条通过固定QA的谱；按保守序列key固定hash划分约90%训练/10%验证，不按模型效果选样本。构建器输出train.csv/val.csv、manifest和QA报告，绑定源MSP SHA256、排除集、CSV哈希、通道mask与序列分组。数据集大小须以完整构建结果为准，不能把先前8,270候选当作已通过全部QA的训练序列数。

### 外部预训练协议

采用现有GraphTransformer共享骨干，四通道临时强度头接原bond_head隐藏表征；输出sigmoid，按有效mask计算L1，每谱先平均有效槽再按谱平均。外部validation loss只用于预训练选点；不使用MiPD测试标签、旧NIST测试标签或其预测。记录未训练初始validation loss，以及仅由外部train拟合的通道常数预测器作为诊断，不据其调参。

固定seed42、batch128、AdamW学习率1e−4、weight_decay1e−4、dropout0.1、5epoch warmup+cosine、最多60epoch；validation maskedL1严格最小者保存，patience10。共一个源预训练模型，不根据下游测试结果选择源epoch。保存可安全weights_only读取的源checkpoint及配置、CSV/manifest SHA、源train/val序列身份、history和epoch。

### 下游三臂与公平比较

1. ion4_control：原模型随机初始化，全部新增分支关闭。
2. ion4_encoder：按白名单迁移经学习的共享序列/图表征，保留条件编码器、FiLM和分类读出的各划分初始化。
3. ion4_bond_context：在encoder范围上增加理论特征投影和bond_head第一隐藏层，最终分类logit层仍保留各划分初始化。

两候选共享同一个预训练checkpoint及SHA，迁移不加载optimizer。标准AA按token语义拷贝，B/O/X/Z、padding及未训练行保留目标初始化；位置编码仅迁移源训练覆盖位置。条件相关的未训练输入列不得随整矩阵静默覆盖，白名单和部分张量范围写入迁移provenance。未知键、形状、模式、SHA或序列隔离不符时失败。来源hash和provenance保存在每划分主/q checkpoint及输出文件，恢复训练与新预训练初始化不得混用。

下游保持此前五组重复序列级80/20划分、内部val seed、group_uniform、q软标签、theory+FiLM、3GCN+2GAT、学习率/损失/调度及全部其他设置一致；有效键口径、固定0.5阈值、max_f1选点、独立earlystop min_delta=.001/patience25，不选择测试epoch或阈值。所有臂下游参数量一致。迁移与control的差包含额外数据和预训练计算；两迁移范围之间的差不混入新的源训练。

主要判据沿用：相对同期control的五划分平均ΔF1≥.002（+0.20个百分点）、F1至少4/5严格提高、平均ΔAUC≥0、平均Δq-Brier≤0；使用未舍入值，差≤1e−12视为平局。附accuracy、Ex-F1调和口径、q_seq，报告mean±sample SD(ddof1)及逐划分配对差；两个候选同时通过时均保留待复验，不按本轮测试均值选择论文主模型。

本项目多轮已参考相同测试划分，本轮属探索性方法筛选。五组共享序列，单初始化不能证明跨种子稳定，不做独立样本t检验。外部预训练排除旧NIST测试序列并不使同源谱库外测成为source-independent验证；未来描述须区分序列留出与来源独立。历史冻结论文值保持不变。

### 数据构建与迁移实现核验

独立目录`dataset/ion4_pretraining_v1`已完成唯一正式构建，原MSP 469,898条、481,006,710 bytes，SHA256=`1c1a8c7ba5299a0312b26f5eb5f8e4974c5a3339cdb4253869067b8e9b4102f1`。原候选8,674谱中额外排除1条I/L等质量序列，留下8,673谱、8,269序列，全部通过冻结的片段质量/强度QA。训练7,817谱/7,447序列，验证856谱/822序列；I/L身份三方交集为0。保留181,766个监督槽，51,804个中性丢失峰不参与监督。独立绝对质量误差median=2.08818、p95=10.22106、max=21.19655 ppm，全部≤50ppm；无非有限/非正强度、歧义、质量失败及同(seq,charge)重复。完整分布与排除计数见manifest.json/build_report.json。实际报告charge分布4+/5+/6+=7,920/706/47；这是高电荷库子集，不声称代表目标域分布。

训练CSV SHA256=`43692bf8b86d3210100e6cf737315df3e7f9fe6db9a5dcafa1ddd4a8d6639efc`；验证CSV SHA256=`2933098b2d1171e3f4f8b95495db5094139cc79edbb85e95af31cc7e5cce6dbd`。数据构建13项合成测试、预训练7项CPU测试通过，正式manifest强化后相关2项复测通过。只读交叉审查确认y对齐、缺失槽mask、位置覆盖及序列隔离；另以全源精确文本查找确认不存在`Mods=0/`矛盾token，通用解析器的该边界不影响本固定源。

encoder白名单为标准AA语义行、实际训练残基位置[0,max_train_length)、physicochemical_encoder、node_encoder融合层的AA/position/physico前128列及共享bias/LayerNorm、GCN和GAT。node融合最后64个条件输入列保持目标初始化；edge_encoder整体、global节点/投影、state/env、FiLM及最终bond_head.3不迁移。bond_context仅额外迁移bond_theory_proj与bond_head.0。位置不包含global节点或未被使用的padding位置。迁移前复核safe checkpoint、固定SHA、代码与数据manifest、全MiPD513排除集及实际下游各split序列，完成全部校验后才拷贝参数；两候选按run保存transfer_provenance.json并写入各类checkpoint。

迁移8项专用测试（含拒绝子场景）与原checkpoint selection 10项测试通过；正式数据经实际loader/driver serializer/transfer的无训练跨模块检查通过。首次CUDA测试中批次大小一致性受TF32影响：同seed关闭TF32后两类global设置最大差由1.01e−4/7.93e−5降至2.98e−8/5.96e−8，原批次比较容差通过。仅该一致性测试关闭并恢复TF32，正式训练精度不改。随后同输入条件扰动测试的bitwise比较出现5.96e−8量级的CUDA scatter归约差：为此增加sanitized输入逐位相等、被屏蔽位精确0和同输入重复对照，CPU输出仍要求bitwise，CUDA输出按rtol1e−6/atol1e−7检验。其余6项CUDA测试通过；仅复测修订单项及真实数据单步，不重复已通过项目。

### 外部预训练完成，固定下游初始化来源

CUDA失败单项修复后通过，真实外部数据16谱完成AMP前向/反向/更新，梯度有限、骨干梯度非零、辅助头权重发生更新。源训练代码版本`57824da9a354cea613dee1bc2adb72c8a747a19d`，已push并在pod pull；运行目录`checkpoints/graph_transform/ion4_pretraining/20261018_082558`，北京时间16:26:00–16:28:19完成60epoch、3,720次optimizer更新，无AMP跳步。完成事件触发后才下载及审计，未读取逐轮进度。

外部验证masked L1初始0.4048580051，train通道median常数预测器0.1765313008，严格最小点第56epoch为0.1228186764。该改善仅说明本辅助任务学到了信号，不构成MiPD分类提高的证据。权重、配置、代码、CSV/manifest、序列身份、逐轮选点与早停重放检查通过；本地审计见`F:/codexout/dbond_ion4_pretrain_results_20261018_082558/pretraining_audit.json`。

两候选固定同一`best_ion_intensity.pt`，SHA256=`2754a6cf380a1a7cf7956a67a51969a0df9c48d7a62b078b2fdbf58732703e03`。新增三份`pre_synthesis_theory_film_ion4_{control,encoder,bond_context}.yaml`，由同期cond3_control克隆，模型字段完全一致；仅实验名/输出目录与三项pretrained初始化配置不同。下游测试结果未参与源checkpoint选择。

`project_goal`：实现并验证离子相对强度预训练及两个迁移范围，完成push/pull、pod训练与事件触发分析。`completion_status`：外部预训练与源审计完成，下游三臂配置已固定，待CUDA单步及15次完整训练。`target_kb`：kb:storage。`knowledge_used/source_records`：本节AlphaPeptDeep公共证据链。`created_files/modified_files`：新增pretraining模块、数据构建/训练/迁移测试脚本、三份ion4配置，最小接入train_graph_model.py并更新本台账。`unresolved_questions`：迁移收益与跨种子泛化，缺失注释和标准L→特殊D域差。`suggested_knowledge_writeback`：完成后可摄入实现与正/负结果，本轮不写知识库。
