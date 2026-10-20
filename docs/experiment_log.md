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

`project_goal`：实现并验证离子相对强度预训练及两个迁移范围，完成push/pull、pod训练与事件触发分析。`completion_status`：外部预训练、下游15次训练及结果审计完成，结果见§15.24。`target_kb`：kb:storage。`knowledge_used/source_records`：本节AlphaPeptDeep公共证据链。`created_files/modified_files`：新增pretraining模块、数据构建/训练/迁移测试脚本、三份ion4配置，最小接入train_graph_model.py并更新本台账。`unresolved_questions`：迁移收益与跨种子泛化，缺失注释和标准L→特殊D域差。`suggested_knowledge_writeback`：完成后可摄入实现与正/负结果，本轮不写知识库。

## 15.24 离子强度预训练迁移第四轮结果（2026-10-18）

**完整三臂15次训练完成，两个迁移范围均未同时通过预定四项分类采用门槛，不替换冻结主模型。** bond_context在F1、AUC及q概率误差上出现改善信号，须与“已达到采用要求”区分；没有事后下调门槛、调整测试阈值或排除失利划分。

### 完成、版本与审计

训练代码`d374c8be98a12783f2d4bdbad2d5491eb3c6eb70`，本地push/pod pull后启动；北京时间2026-10-18 16:32:42–17:27:17，耗时54分35秒，GPU0/1/2，三臂均returncode=0。远端根目录`/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/ion4_batch/20261018_083242`；本地证据`F:/codexout/dbond_ion4_finetune_results_20261018_083242`。只有完成事件返回后才读取汇总和分析，期间未逐轮获取训练状态。

两个候选先在真实fold1222训练数据16行上各执行一次生产Trainer AMP更新，均通过，模型参数量均1,706,153。验证加载不改变RNG，特殊BOXZ、padding及未训练C残基行保留，各条件/edge/global/最终分类层保留目标初始化，所有参数参与后续微调。首个smoke只因检查脚本错误要求cache_dir一致而在训练前停止；修正检查脚本允许独立cache目录后通过，正式训练代码及数据规则未改变。失败与成功检查记录均保留。

完成后只读审计442项通过、0失败：启动源码全部SHA、manifest和配置、相同fold seed/train-val字节哈希、内部序列隔离、输入mask、阈值0.5、严格max-F1选点和独立早停重放、主/q/周期checkpoint迁移provenance及同源SHA、完整白名单和保留范围、预测CSV首行来源及history有限值。原始15行指标经独立CSV复算，与audit/summary数值及SHA一致。没有重做模型推理，也未以测试结果调整选择规则。用户另一个ensemble脚本修改保留原状。

源预训练见§15.23：8,673谱，源验证最佳第56epoch，masked L1=0.1228186764（初始0.4048580051、train-median常数基线0.1765313008），同一源checkpoint供两臂使用；源码和源checkpoint均已绑定SHA。源自然残基库中的19类实际训练残基参与语义行迁移，未观察到的C与特殊残基保持下游初始化。

下表为主checkpoint epoch / 最终训练epoch：

| 实验臂 | 1222 | 2252 | 3514 | 6072 | 9075 |
|---|---:|---:|---:|---:|---:|
| control | 74 / 99 | 84 / 109 | 97 / 122 | 80 / 105 | 83 / 96 |
| encoder | 74 / 84 | 74 / 97 | 40 / 60 | 39 / 64 | 70 / 95 |
| bond_context | 59 / 71 | 61 / 86 | 40 / 65 | 80 / 105 | 70 / 76 |

代码版本：`d374c8be98a12783f2d4bdbad2d5491eb3c6eb70`；规则：`docs/experiment_log.md §15.23`。

所有值由 15 份原始划分结果重算并与 summary 校验；下表为均值 ± 样本标准差（ddof=1）。

| 指标 | Control | Encoder transfer | Bond context transfer |
|---|---:|---:|---:|
| 有效键 micro-F1（%） | 79.837 ± 0.643 | 79.747 ± 0.358 | 80.000 ± 0.429 |
| 有效键 accuracy（%） | 80.032 ± 0.368 | 80.103 ± 0.258 | 80.281 ± 0.168 |
| ROC-AUC | 0.882465 ± 0.003790 | 0.884110 ± 0.003559 | 0.886188 ± 0.002231 |
| Ex-F1 调和口径（%） | 75.494 ± 0.244 | 75.133 ± 0.305 | 75.568 ± 0.175 |
| q_seq（序列级 Spearman） | 0.62496 ± 0.06238 | 0.65865 ± 0.03596 | 0.65163 ± 0.05609 |
| q-Brier（越低越好） | 0.067151 ± 0.002217 | 0.064963 ± 0.002240 | 0.064097 ± 0.002157 |

### 相对同期 Control 的逐划分差

| 划分 | Encoder transfer ΔF1（百分点） | Bond context transfer ΔF1（百分点） | Encoder transfer Δq_seq | Bond context transfer Δq_seq |
|---|---:|---:|---:|---:|
| 1222 | -0.09181 | +0.07258 | +0.083447 | +0.039977 |
| 2252 | +0.16609 | -0.04160 | -0.004997 | +0.005941 |
| 3514 | -0.64494 | -0.13959 | +0.009409 | +0.025977 |
| 6072 | -0.12222 | +0.37324 | +0.029403 | +0.027221 |
| 9075 | +0.24057 | +0.54868 | +0.051191 | +0.034233 |

### 四项工程门槛

判定使用未舍入值；F1 获胜要求 ΔF1 > 1e-12。所有门槛必须同时通过。

| 实验臂 | 平均 ΔF1 ≥ +0.20 百分点 | F1 至少 4/5 获胜 | 平均 ΔAUC ≥ 0 | 平均 Δq-Brier ≤ 0 | 待复验候选 |
|---|---:|---:|---:|---:|---|
| Encoder transfer | -0.090461 (未通过) | 2/5 (未通过) | +0.00164470 (通过) | -0.00218759 (通过) | 否 |
| Bond context transfer | +0.162663 (未通过) | 3/5 (未通过) | +0.00372312 (通过) | -0.00305422 (通过) | 否 |

待复验候选：无。两臂同时通过时同时保留，不依据这次测试均值指定论文主模型。

### 次要描述比较：Bond context transfer − Encoder transfer

三项百分比指标的差值以百分点表示；其余用原尺度。q-Brier 减少计为 bond_context 获胜；其他指标增加计为获胜。

| 指标差值 | 1222 | 2252 | 3514 | 6072 | 9075 | 均值 ± SD | bond_context 获胜 / 平局 / 失利 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 有效键 micro-F1（百分点） | +0.1644 | -0.2077 | +0.5054 | +0.4955 | +0.3081 | +0.2531 ± 0.2939 | 4 / 0 / 1 |
| 有效键 accuracy（百分点） | +0.0815 | -0.1423 | +0.2076 | +0.5041 | +0.2389 | +0.1780 ± 0.2360 | 4 / 0 / 1 |
| ROC-AUC | +0.0023307 | -0.0026360 | +0.0013039 | +0.0075316 | +0.0018619 | +0.0020784 ± 0.0036289 | 4 / 0 / 1 |
| Ex-F1 调和口径（百分点） | +0.3834 | -0.2987 | +0.8169 | +0.8879 | +0.3874 | +0.4354 ± 0.4728 | 4 / 0 / 1 |
| q_seq（序列级 Spearman） | -0.043470 | +0.010938 | +0.016568 | -0.002183 | -0.016957 | -0.007021 ± 0.024133 | 2 / 0 / 3 |
| q-Brier（越低越好） | -0.0014233 | +0.0013364 | -0.0007710 | -0.0023109 | -0.0011643 | -0.0008666 ± 0.0013553 | 4 / 0 / 1 |

这是同一个源 checkpoint 的不同迁移范围的描述性比较；相对 Control 的差包含额外外部数据和预训练计算，下游模型容量保持一致。

Ex-F1 先在每划分内由 ex_precision/ex_recall 求调和均值再取均值。q-Brier 是对条件组 q 的概率误差，并非纯校准误差。

五组重复划分共享序列，未进行独立样本 t 检验，不作显著性或置信区间推断。每划分一次初始化不能证明跨种子稳定。

多轮方法筛选已参考相同测试划分，本轮属于探索性实验；正结果仍需不同初始化及未用于方法筛选的数据复验。

### 解释与下一步限制

仅迁移encoder：F1 −0.090461个百分点、2/5胜；AUC +0.001644702、q-Brier −0.002187594、q_seq +0.033690519。增加bond隐藏表征迁移：F1 +0.162663个百分点、3/5胜；accuracy约+0.249个百分点、AUC +0.003723115、q-Brier −0.003054217、q_seq +0.026669652。后者q-Brier及q_seq均在5/5划分改善，但F1仍未达到+0.20个百分点和4/5胜的预定门槛。

两个范围都改善排序/概率误差，说明外部离子监督比此前六种小分支扩展提供了更明确的辅助指标信号；这仍不证明分类性能稳定提高或找到了唯一瓶颈。bond_context相对encoder的F1平均+0.253124个百分点、4/5胜，是同一源模型迁移范围的描述性结果，不能代替与随机初始化control比较。下游模型容量完全相同，但迁移实验额外使用外部数据和预训练计算。

后续可预先注册较低骨干学习率、原读出正常学习率的迁移保护实验，以检验全参数同学习率微调是否影响预训练表征；这是待验证假设，本轮未执行。保留固定验证选点及测试阈值，不能事后在测试集调阈值追逐F1。若进一步追求可靠主结果，需要未参与方法筛选的数据和不同初始化；五次重叠划分不能视为独立重复实验。缺失注释、高电荷源库及标准L→特殊D化学域差仍存在，不能将本辅助任务称为完整实验谱预测。

### 交付

完整结果、判定表、逐划分PNG/PDF、源审计与CUDA检查保存在本地证据目录及pod批次review目录；本台账与论文表附注追加本轮结果，冻结主表未替换。预训练代码、迁移器、测试及三配置已经提交并推送，所有新行为通过显式配置启用。完成等待沿用事件触发，不实时读取epoch日志。

`project_goal`：实现外部离子相对强度预训练及两种迁移范围，完成push/pull、训练与事件触发判读。`completion_status`：全部完成，两个范围均未通过全部分类采用门槛。`target_kb`：kb:storage。`knowledge_used/source_records`：§15.23 AlphaPeptDeep原始公共PDF及实体/资源映射证据。`created_files/modified_files`：§15.23代码与配置、本台账、论文本地附注和实验分析产物。`unresolved_questions`：域差、跨种子稳定性、较低骨干学习率能否保留迁移收益。`suggested_knowledge_writeback`：可另行摄入完整实现及正/负结果；本轮未写知识库。

## 15.25 完整迁移编码层降低学习率：第五轮预定方案（2026-10-18）

状态：用户要求继续下一轮优化，沿用代码实现、测试、commit/push、pod pull、训练和完成后分析授权。Valhalla project_work，base/kb:storage，只读；项目产物仍在DBond仓库及F:/codexout实验目录。本节在训练前固定方案、比较和判据；训练采用完成事件静默等待，不逐轮查询。

### 研究问题与来源边界

第四轮bond_context有F1及排序/概率误差改善，但未通过全部采用门槛。本轮检验同一预训练来源下，降低已完整迁移编码层的更新速率是否比等学习率微调更合适。不能由现有结果认定发生了灾难性遗忘，也不能预设减慢适配能解决标准L→特殊D及强度→分类的域差。

再次核验storage证据：`ent_000004` active → `.registry/machine/entity_resource_map.yaml:33`（derived_from，无旧map_id，未补造）→ `res_000119` → `file_000119`，公共文件`Library/public_resources/res_000119/authoritative.pdf`，SHA256=`e87e5b9ee3d363c1308828c85d50d73520761cd3efd75e820db3fde8d02f993b`；local membership active、required/excluded/blacklist空、资源active/not_listed/present，注册SHA一致。物理p5–6、Fig.4支持天然肽/PTM的少样本MS2适配；p10 Methods/MS2 model中预训练phase1、phase2/3及21种PTM微调均为lr=1e−5，PTM10epoch、warmup5、dropout.1、L1及cosine warmup。该主文未提供分层学习率/冻结优于等学习率的直接证据。此次0.1倍率是本项目固定工程假设，不称文献已验证策略；未读取source_copies、未写回KB。

### 三臂与唯一新变量

1. lr5_control：随机初始化，与此前同期control相同协议。
2. lr5_uniform：第四轮bond_context范围，所有参数统一原学习率，作为本轮迁移复验和速率比较的对照。
3. lr5_slow：与uniform完全相同的预训练初始化，仅已整张迁移的encoder参数使用0.1×学习率。

uniform与slow固定第四轮同一源：`checkpoints/graph_transform/ion4_pretraining/20261018_082558/best_ion_intensity.pt`，SHA256=`2754a6cf380a1a7cf7956a67a51969a0df9c48d7a62b078b2fdbf58732703e03`。不新增预训练、不按本轮测试重新选源。迁移白名单、全MiPD513/旧NIST排除及I/L身份隔离保持§15.23。两臂所有目标域参数可训练，没有冻结阶段。

slow组精确为已验证transfer_provenance.fully_copied_keys减去四个context张量（bond_theory_proj的weight/bias及bond_head.0的weight/bias），共48张量：GCN/GAT及完全迁移的physicochemical/node融合bias/LayerNorm。只部分拷贝的AA embedding、position embedding、node融合第一层weight整张保持原LR，避免对未预训练的BOXZ/C/保留位置或条件列一起降速。全部读出、条件、独立edge_encoder/global参数保持原LR；GCN/GAT内部边变换属于已完整迁移的编码层，随该层使用低LR。故本轮不称“全部骨干统一降低LR”。

采用真实AdamW参数组而非缩放梯度；普通组放index0，峰值1e−4、下限1e−5，slow组峰值1e−5、下限1e−6。两组共用原warmup40及cosine倍率，整个调度周期保持0.1比例，不能沿用统一绝对min_lr导致slow后程卡在1e−5。weight_decay系数仍1e−4；学习率降低也会降低AdamW解耦衰减的实际步长，效果解释为较慢更新协议，不单独归因为防遗忘。scale=1或缺省必须完全保持原单组optimizer、参数顺序、scheduler和RNG路径；scale<1与resume不混用。

### 固定训练与验证协议

三组模型结构与参数量相同，五组重复序列级80/20划分及内部val seed配对；training条件组折叠、group_uniform、q软标签、test行级、pre输入mask、theory+FiLM、固定阈值0.5不变。最多300epoch、batch128、warmup40、原cosine、max_f1主选点、独立earlystop min_delta=.001/patience25均不改。本轮仅0.1这一预定倍率，不扫0.3等网格，不增加训练预算补偿低LR，不按测试表现调整epoch/阈值。

低LR参数分组、参数名/numel、峰值与floor LR写入optimizer_provenance.json及主/q/周期checkpoint；逐epoch记录实际各组LR供完成后重放核对。测试覆盖AdamW真实更新比例、完整唯一参数覆盖、部分迁移参数归普通组、全周期LR比例/下限、disabled原行为及错误配置失败。再于pod真实16行数据各执行uniform/slow一次CUDA训练步骤，成功后才提交完整15次训练。

### 预定判据

相对本轮随机control，uniform和slow分别沿用四项工程筛选：meanΔF1≥.002（+0.20个百分点）、F1至少4/5严格提高、meanΔAUC≥0、meanΔq-Brier≤0。此外slow要被作为“降低学习率方案”的待复验候选，必须相对同期uniform meanΔF1>0且F1至少4/5严格提高；这项速率收益判断不额外要求对uniform提高.002。报告slow−uniform所有六指标及逐划分配对差，不能用只胜随机control来声称降低LR有效。uniform若四项通过，仅记为原迁移方案复验候选。

所有判定用未舍入值、1e−12平局容差，mean±sample SD(ddof1)；Ex-F1逐划分由谱平均P/R取调和后再均值，q-Brier是对q的概率误差。两种比较均属探索性，五组共享序列且单初始化，不能做独立样本显著性推断。多轮已参考相同测试划分；即使候选通过也不自动更新冻结主表，仍需不同初始化及未参与筛选数据复验。

实现核验：专用optimizer测试7/7、原checkpoint选择回归10/10、迁移provenance保存定点回归1/1，共18项通过。scale=1/缺省的原优化器数值、参数顺序和RNG一致；缺省不新增optimizer metadata或改变原q保存路径。新元数据仅对显式scale字段启用（本轮三臂均显式配置）。预训练SOURCE_FILES的12个文件均未修改。真实CUDA集成检查及运行版本待提交后追加。

`project_goal`：实现并检验完整迁移编码层0.1×LR，完成push/pull、15次训练及事件触发判读。`completion_status`：三臂15次训练、完整审计和结果记录已完成，结果及预定判据结论见§15.26。`target_kb`：kb:storage。`knowledge_used/source_records`：本节AlphaPeptDeep公共PDF证据链；分层LR为工程假设。`created_files/modified_files`：新增training/transfer_optimizer.py、专用测试、三份lr5配置和实验分析工具，最小接入train_graph_model.py，更新本台账。`unresolved_questions`：降低更新速率能否改善目标域分类与稳定性、域差和跨种子复验。`suggested_knowledge_writeback`：完成后可另行摄入实现与结果，本轮不写知识库。

## 15.26 完整迁移编码层低学习率第五轮结果（2026-10-18）

**完整三臂15次训练及审计完成。Uniform 未通过全部四项门槛；Slow 未通过相对随机对照的全部四项门槛，也未通过相对 Uniform 的 method_support。冻结主表保持不变。** 所有判定使用预注册规则和未舍入值。

### 完成、版本与审计

训练代码`34466f27c0ba7cfcea0019d7d8e1d879cfd86bee`；北京时间2026-10-18 18:13:19至2026-10-18 19:12:15，耗时58分55秒，三臂均returncode=0；声明GPU分配`[0, 1, 2]`。远端根目录`/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/lr5_batch/20261018_101319`；本地证据`F:/codexout/dbond_lr5_finetune_results_20261018_101319`。

实现阶段已完成专用optimizer测试7项、checkpoint选择回归10项、迁移provenance保存定点回归1项，共18项通过。已保存的CUDA集成检查`F:/codexout/lr5_cuda_smoke_result.json`（SHA256=`89aedd8d6376eeaf7ef36d741051f949ada21d51b0fbcd13ae0ab323d067e60d`）于北京时间2026-10-18 18:10:36至2026-10-18 18:10:54通过；uniform、slow各使用fold1222的前16条折叠训练样本执行一次生产Trainer AMP优化更新，模型参数量均1,706,153。两者载入相同源初始化，RNG保持不变，梯度和参数有限，partial AA/position/fusion与BOXZ仍使用正常LR；slow的48张量名单与全调度比例验证通过。smoke损失仅为集成检查，不作为模型效果。

完成后只读审计460项全部通过、0失败：启动源码SHA与固定源12文件、manifest/配置、配对seed和内部train/val字节哈希、序列隔离、输入mask、阈值0.5、严格max-F1选点与独立早停重放、所有主/q/周期checkpoint迁移和optimizer provenance、参数组完整唯一覆盖/numel/48张量白名单、实际LR和floor比例、checkpoint extra与运行JSON一致。history第N轮记录该轮训练前实际LR；checkpoint保存的是该轮调度结束后的下一步LR，两种时点分别重放验证。预测来源只检查CSV首行，未重新推理或扫描全部模型权重数值。

结果由`finish_lr5_analysis.py`从原始CSV重算，并核对summary、审计指标及来源哈希。本记录脚本再次验证decision与analysis_tables均对应当前完整输入。

下表为主checkpoint epoch / 最终训练epoch（均为1-based）：

| 实验臂 | 1222 | 2252 | 3514 | 6072 | 9075 |
|---|---:|---:|---:|---:|---:|
| control | 99 / 99 | 84 / 86 | 85 / 86 | 80 / 105 | 83 / 96 |
| uniform | 99 / 124 | 61 / 86 | 40 / 65 | 80 / 105 | 51 / 76 |
| slow | 101 / 102 | 74 / 99 | 98 / 107 | 62 / 87 | 70 / 75 |

### 固定来源与实际干预

Uniform与Slow共用第四轮固定源checkpoint，SHA256=`2754a6cf380a1a7cf7956a67a51969a0df9c48d7a62b078b2fdbf58732703e03`，没有本轮新预训练或按下游测试重选源epoch。两者均为bond_context迁移范围。Slow仅让已整张迁移的48个encoder张量以0.1×LR更新：GCN/GAT及完整迁移的physicochemical、node融合bias/LayerNorm；GCN/GAT内部边变换属于这48张量。部分迁移的AA embedding、position embedding、node融合weight整张保持正常LR，全部读出/条件、独立edge_encoder、global参数也保持正常LR。

普通组峰值1e−4、cosine下限1e−5；slow组峰值1e−5、cosine下限1e−6，共用warmup40与同一cosine倍率。线性warmup初期可以低于cosine下限。使用真实AdamW参数组，未冻结参数或缩放梯度；weight_decay系数仍为1e−4，但降低LR同时减小解耦权重衰减的实际更新步长，因此结果只解释为较慢更新协议，不能单独归因于防止遗忘。

代码版本：`34466f27c0ba7cfcea0019d7d8e1d879cfd86bee`；规则：`docs/experiment_log.md §15.25`。

所有值由 15 份原始划分结果重算并与 summary 校验；下表为均值 ± 样本标准差（ddof=1）。

| 指标 | Control | Uniform LR transfer | Slow encoder transfer |
|---|---:|---:|---:|
| 有效键 micro-F1（%） | 79.800 ± 0.661 | 80.063 ± 0.410 | 79.426 ± 0.567 |
| 有效键 accuracy（%） | 80.087 ± 0.371 | 80.317 ± 0.233 | 79.649 ± 0.435 |
| ROC-AUC | 0.882962 ± 0.003775 | 0.886593 ± 0.003083 | 0.879345 ± 0.004887 |
| Ex-F1 调和口径（%） | 75.425 ± 0.240 | 75.592 ± 0.222 | 74.886 ± 0.213 |
| q_seq（序列级 Spearman） | 0.63098 ± 0.04852 | 0.66685 ± 0.03673 | 0.64457 ± 0.06249 |
| q-Brier（越低越好） | 0.066792 ± 0.002463 | 0.063871 ± 0.002185 | 0.068097 ± 0.002490 |

### 相对同期 Control 的逐划分差

| 划分 | Uniform LR transfer ΔF1（百分点） | Slow encoder transfer ΔF1（百分点） | Uniform LR transfer Δq_seq | Slow encoder transfer Δq_seq |
|---|---:|---:|---:|---:|
| 1222 | +0.49848 | -0.27980 | +0.063612 | -0.002548 |
| 2252 | -0.03382 | -0.53474 | +0.004007 | -0.000431 |
| 3514 | -0.11971 | -0.45879 | +0.033384 | +0.030953 |
| 6072 | +0.31677 | -0.41096 | +0.020695 | +0.035171 |
| 9075 | +0.65461 | -0.18485 | +0.057692 | +0.004821 |

### 四项工程门槛

判定使用未舍入值；F1 获胜要求 ΔF1 > 1e-12。四门槛须同时通过；Slow 还须通过下述相对 Uniform 的 method_support。

| 实验臂 | 平均 ΔF1 ≥ +0.20 百分点 | F1 至少 4/5 获胜 | 平均 ΔAUC ≥ 0 | 平均 Δq-Brier ≤ 0 | 待复验候选 |
|---|---:|---:|---:|---:|---|
| Uniform LR transfer | +0.263269 (通过) | 3/5 (未通过) | +0.00363136 (通过) | -0.00292087 (通过) | 否 |
| Slow encoder transfer | -0.373828 (未通过) | 0/5 (未通过) | -0.00361685 (未通过) | +0.00130494 (未通过) | 否 |

待复验候选：无。Uniform 是迁移结果的同期复验；Slow 只有四门槛和 method_support 同时通过才保留，不据此指定论文主模型。

### 方法支持比较：Slow encoder transfer − Uniform LR transfer

F1、Accuracy、Ex-F1 差值以百分点表示；其余用原尺度。q-Brier 减少计为 Slow 获胜；其他指标增加计为获胜。

| 指标差值 | 1222 | 2252 | 3514 | 6072 | 9075 | 均值 ± SD | Slow 获胜 / 平局 / 失利 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 有效键 micro-F1（百分点） | -0.7783 | -0.5009 | -0.3391 | -0.7277 | -0.8395 | -0.6371 ± 0.2100 | 0 / 0 / 5 |
| 有效键 accuracy（百分点） | -0.7373 | -0.6944 | -0.0529 | -0.8960 | -0.9624 | -0.6686 ± 0.3614 | 0 / 0 / 5 |
| ROC-AUC | -0.0087460 | -0.0065259 | -0.0011461 | -0.0103930 | -0.0094300 | -0.0072482 ± 0.0036963 | 0 / 0 / 5 |
| Ex-F1 调和口径（百分点） | -0.8798 | -0.4893 | -0.5202 | -1.0829 | -0.5620 | -0.7068 ± 0.2620 | 0 / 0 / 5 |
| q_seq（序列级 Spearman） | -0.066160 | -0.004438 | -0.002431 | +0.014475 | -0.052871 | -0.022285 ± 0.035087 | 1 / 0 / 4 |
| q-Brier（越低越好） | +0.0048231 | +0.0039107 | +0.0017840 | +0.0046990 | +0.0059121 | +0.0042258 ± 0.0015400 | 0 / 0 / 5 |

method_support：未通过。要求相对 Uniform 的平均 ΔF1 > 0 且至少 4/5 严格获胜；这里不要求 +0.20 个百分点。

两臂共享源 checkpoint 和 bond_context 迁移范围；只有 48 张完整迁移 encoder 张量使用 0.1× LR，部分迁移 AA/位置/融合张量、读出与条件参数仍用正常 LR。相对随机初始化 Control 的差仍包含外部数据与预训练计算，下游容量保持一致。

Ex-F1 先在每划分内由 ex_precision/ex_recall 求调和均值再取均值。q-Brier 是对条件组 q 的概率误差，并非纯校准误差。

五组重复划分共享序列，未进行独立样本 t 检验，不作显著性或置信区间推断。每划分一次初始化不能证明跨种子稳定。

多轮方法筛选已参考相同测试划分，本轮属于探索性实验；正结果仍需不同初始化及未用于方法筛选的数据复验。

### 结论边界

Uniform 相对随机 Control：ΔF1 +0.263269 个百分点、3/5 胜，ΔAUC +0.003631357、Δq_seq +0.035878144、Δq-Brier -0.002920865。

Slow 相对随机 Control：ΔF1 -0.373828 个百分点、0/5 胜，ΔAUC -0.003616851、Δq_seq +0.013593180、Δq-Brier +0.001304938。

Slow−Uniform：平均 ΔF1 -0.637096 个百分点、0/5 严格获胜，method_support 未通过。该比较只要求平均 ΔF1 > 0 且至少 4/5 胜，不额外要求提高 0.20 个百分点。

Uniform 未通过全部四项门槛；Slow 未通过相对随机对照的全部四项门槛，也未通过相对 Uniform 的 method_support。冻结主表保持不变。 Uniform是原迁移方案的同期复验；Slow必须同时胜过随机对照的四门槛与上述method_support，不能只凭优于随机初始化声称降低LR有效。无论候选是否通过，五组重叠划分、单初始化和多轮已参考相同测试集的限制仍存在；没有独立样本t检验、显著性或跨种子稳定性结论，需要不同初始化及未参与筛选数据复验。

知识依据沿用§15.25预注册时核验的storage公共证据：`ent_000004 → .registry/machine/entity_resource_map.yaml:33 → res_000119 → file_000119`，公共`Library/public_resources/res_000119/authoritative.pdf`，SHA256=`e87e5b9ee3d363c1308828c85d50d73520761cd3efd75e820db3fde8d02f993b`；旧映射无map_id，未补造。AlphaPeptDeep物理p5–6/Fig.4支持天然肽/PTM的少样本MS2适配；p10未提供分层LR或冻结优于统一LR的直接证据。0.1倍率仍为本项目工程假设。高电荷注释源库、缺失槽mask及标准L→特殊D化学域差仍限制外推；同源NIST序列留出不构成来源独立验证。本轮未写回知识库。

`project_goal`：检验完整迁移编码层0.1×LR并完成训练与事件触发判读。`completion_status`：三臂15次训练与审计完成；Uniform 未通过全部四项门槛；Slow 未通过相对随机对照的全部四项门槛，也未通过相对 Uniform 的 method_support。冻结主表保持不变。`target_kb`：kb:storage。`knowledge_used/source_records`：§15.25及上述AlphaPeptDeep公共证据。`created_files/modified_files`：lr5实现/测试/三配置、审计分析产物、本台账和本地论文附注。`unresolved_questions`：跨种子与新数据复验、域差，以及较慢AdamW更新与衰减的共同作用。`suggested_knowledge_writeback`：可另行摄入完整实现及正/负结果；本轮不写知识库。

## 15.27 共享投影 GATv2 注意力第六轮预定方案（2026-10-18）

用户要求借鉴storage文献继续实现和训练，沿用本地commit/push、pod pull、CUDA检查、提交训练及完成事件触发判读的授权。Valhalla project_work为base/kb:storage，只读知识库。本轮检验一个注意力机制变体，不联合改动输入特征、图半径、读出头、标签或预训练方式；两臂随机初始化，各五组重复序列级80/20划分，共10次训练。

### 文献证据及工程假设

依据《How Attentive are Graph Attention Networks?》(ICLR2022)。知识库机器身份链：`ent_000028 → .registry/machine/entity_resource_map.yaml → res_000143 → file_000143`，relation=derived_from；旧映射没有map_id，未补造。entity/local membership/resource均active、无excluded/blacklist；公共`G:/valhalla/valhalla_storage_share_root_20260830/Library/public_resources/res_000143/authoritative.pdf`为authoritative/present，实际SHA256与登记一致：`d316a7d23d735becf0c31d2b73c3426bb2aa5ec592705ce559d79a740abf53ec`。物理第5页§3.3、公式(6)–(7)说明将attention向量投影放到向量非线性之后，使邻居内容排序能依赖query。追溯记录`F:/codexout/attn6_kb_source.json`。

现有score为`LeakyReLU(a_src·Wh_src + a_dst·Wh_dst + edge_bias_ij)`。已有边偏置、向量message gate及条件化节点，因此不能把整网完全等同原文的无边特征标准GAT；对相同关系特征，当前节点内容打分仍可分离。本轮选择保留原共享投影W的受限GATv2变体：`a_v2·LeakyReLU(Wh_src + Wh_dst) + edge_bias_ij`。保持W在source/target/message间共享；这是具体工程适配，不声称复现论文所有参数化或其完整通用表达力定理。标量edge bias移至内容分数之后，区别于旧版位于标量LeakyReLU内部；这个放置变化也是本轮机制干预的一部分。

### 固定实验臂与实现范围

| 实验臂 | 配置 | 模式 | 预期参数量 |
|---|---|---|---:|
| Control | `pre_synthesis_theory_film_attn6_control.yaml` | gat | 1,706,153 |
| GATv2 shared projection | `pre_synthesis_theory_film_attn6_gatv2.yaml` | gatv2 | 1,705,641 |

两份配置基于lr5随机control，移除所有pretrained/transfer设置；只在`model.gat_attention_type`、实验描述及输出/cache路径不同。仅替换两层GAT，保留3层GCN、256维/4头、既有图、理论15维特征、FiLM、边bias/message gate、残差/LayerNorm/dropout及原键级读出。先前fragment、chemistry、CNN/GRU及晚期条件分支均关闭，不使用源预训练checkpoint。

每层原`a[4,128,1]`换为`a_v2[4,64,1]`，共少512参数；无新增死参数，不以扩容解释收益。拟新增`models/attention_v2.py`并最小接入GraphTransformer/ModelConfig，旧`attention_layers.py`保持不变。先执行与旧层一致的构造，再在隔离RNG上下文内创建新的attention向量；核验同seed其余共享权重及后续随机状态一致。默认gat不新增state键，gatv2独有顶层int64标量`gat_attention_mode_code=1`；配置与状态模式不一致、缺失/伪造标志或混合a/a_v2应在加载前拒绝。正式参数量和测试结果以完成记录为准。

### 数据、训练与判据

五组划分固定为1222/2252/3514/6072/9075，各自内部序列级验证划分与seed规则不变。训练使用condition-group folded q软标签和group_uniform，测试保留逐谱标签。pre-synthesis运行时mask为state[T,T,F]、env[T,F]，不输入观测intensity/scan/RT。训练batch128、AdamW学习率1e−4、weight_decay1e−4、warmup40+cosine/min_lr1e−5、最多300epoch、patience25/min_delta.001。主checkpoint严格按验证F1最大保存；早停仍独立按min_delta；测试阈值固定0.5。所有optimizer保持原legacy路径，不新增分组LR。

采用门槛沿用：相对同期Control，五划分平均ΔF1≥.002（+0.20个百分点），至少4/5严格F1提高，平均ΔAUC≥0，平均Δq-Brier≤0；四项须同时满足。使用未舍入数值，差≤1e−12为平局。报告有效键F1/accuracy/AUC、逐fold由ex_precision/ex_recall调和得到的Ex-F1、q_seq和q-Brier，mean±样本SD(ddof1)及逐划分配对差。不调测试阈值、源epoch、划分子集或事后采用门槛。五次划分共享序列且本轮单初始化，多轮方法筛选已参考同一测试划分；不作独立样本显著性或跨种子稳定性结论。即使通过，也只列待复验候选，不自动替换冻结主表。

### 执行、核验及后续数据可用性

先完成CPU针对性测试（真实排名反转、旧默认兼容与共享初始化、梯度/掩码、保存加载模式拒绝），再push/pod pull后做两臂各一次真实fold1222前16行的production Trainer AMP更新。完整批次仅在CUDA检查通过后提交。启动绑定源码/配置SHA与gitrev，选择空闲GPU，不干预其他任务。完成等待使用inotify终态事件，不逐epoch读取训练日志；完成后审计完整10份输出、配对train/val、选点、模式和参数形状，再分析、作图及独立CSV复算。

为后续目标域12通道监督，只读盘点本地项目和pod `/mnt/pvc/graphtrans`（跳过环境、缓存、模型/输出目录）目前仅找到PBCLA示例MGF；未找到完整目标域MGF/mzML。此有限范围结果不证明其他目录或压缩包中不存在原始数据。记录`F:/codexout/attn6_target_spectrum_inventory.json`。用户随后提供原始压缩包`D:/UserData/Downloads/pep_521_mgf.tar.gz`并授权本地检查/解压或传pod；正在独立核验归档内容和谱图字段。本轮attention协议保持不变，不从现有二元CSV伪造离子标签。

实现核验：6项专用CPU测试和1项旧条件分支checkpoint回归均通过；实际参数量为1,706,153/1,705,641，共享参数及构造后CPU RNG逐位配对。独立代码审阅通过，原attention_layers.py未变。真实CUDA单步与正式训练均已完成，实际结果见§15.28。

`project_goal`：验证文献启发的共享投影GATv2变体，完成同步、10次训练及事件触发判读。`completion_status`：两臂10次训练、完整审计和结果记录已完成，结果及预定判据结论见§15.28。`target_kb`：kb:storage。`knowledge_used/source_records`：本节ent_000028/res_000143/file_000143公共PDF身份、SHA和物理第5页。`created_files/modified_files`：新attention_v2模块与专用测试、两份attn6配置、最小模型接入、实验台账及外部审计/完成等待工具。`unresolved_questions`：该受限动态打分是否改善目标域泛化，跨种子/新数据稳定性，以及完整目标域原始谱图位置。`suggested_knowledge_writeback`：完成后可另行摄入工程实现及正负结果，本轮不写知识库。


## 15.28 共享投影GATv2注意力第六轮结果（2026-10-18）

**两臂10次训练与审计完成。GATv2未通过全部四项预定门槛，不列为待复验候选；冻结主表保持不变。** 判定使用§15.27预定规则和未舍入值，没有按测试结果修改阈值、epoch或划分。

### 完成、版本与审计

训练代码`9ae88307ae9801de75d36486d2cc7fb7f36d753b`；北京时间2026-10-18 20:32:08至2026-10-18 21:09:25，耗时37分18秒，两臂均returncode=0；声明GPU分配`[0, 1, 2]`。远端`/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/attn6_batch/20261018_123207`；分析与证据目录`/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/attn6_batch/20261018_123207`。

实现阶段6项GATv2专用CPU测试与1项旧条件分支checkpoint回归均通过。生产CUDA检查`/mnt/pvc/graphtrans/attn6_ops/evidence/attn6_cuda_smoke_result.json`（SHA256=`d913ca6ef77b61b23ebcff6d7a9d210f4d5bbcb4da377c8c042c42eb6417347d`）于北京时间2026-10-18 20:31:40至2026-10-18 20:31:52通过：两臂各使用fold1222前16条折叠训练样本完成一次production Trainer AMP优化更新，attention score/projection及分类头更新、梯度/参数有限，观测输入保持屏蔽。共享初始张量与构造后CPU/CUDA RNG一致；参数量Control为1,706,153、GATv2为1,705,641。smoke损失只检查集成，不代表下游效果。

完成后审计380项全部通过、0失败：启动源码SHA、manifest/两臂配置、同划分seed与train/val字节哈希、内部序列隔离、pre输入mask、0.5阈值、严格验证max-F1选点与独立早停重放、主/q/周期checkpoint的注意力模式和a/a_v2形状及参数量、单组optimizer与无迁移provenance、汇总行与原始指标文件/同evaluation_id归档SHA绑定、预测CSV首行来源及history有限值。预测来源仅检查CSV首行，没有重新推理或扫描全量权重数值。

`finish_attn6_analysis.py`从10份原始结果重算并核对summary、审计指标和来源SHA；本记录脚本再次核对decision及analysis_tables与完整输入。

主checkpoint epoch / 最终训练epoch（1-based）：

| 实验臂 | 1222 | 2252 | 3514 | 6072 | 9075 |
|---|---:|---:|---:|---:|---:|
| control | 99 / 99 | 84 / 109 | 85 / 86 | 80 / 105 | 83 / 96 |
| gatv2 | 74 / 99 | 61 / 86 | 61 / 69 | 71 / 76 | 83 / 96 |

### 实际干预

两臂均随机初始化，无外部预训练和分组学习率。仅两层注意力的内容打分改为`a_v2·LeakyReLU(Wh_src+Wh_dst)+edge_bias_ij`；W仍在source/target/message间共享，旧分数为`LeakyReLU(a_src·Wh_src+a_dst·Wh_dst+edge_bias_ij)`。标量edge bias后置也属于此次干预。边向量message gate、图结构、3层GCN、theory/FiLM、节点编码和键读出保持一致。每层a从[4,128,1]换为a_v2[4,64,1]，共少512参数；这是共享投影受限变体，不能称完整通用GATv2表达力复现或严格等参数比较。

既有fragment/chemistry/sequence adapter/晚期条件分支关闭。训练仍是条件组q软标签、group_uniform、原AdamW单组LR和warmup40+cosine、最多300epoch、独立patience25/min_delta.001、主max_f1及固定0.5阈值。

代码版本：`9ae88307ae9801de75d36486d2cc7fb7f36d753b`；规则：`docs/experiment_log.md §15.27`。

两臂均随机初始化；所有值由 10 份原始划分结果重算并与 summary 校验。下表为均值 ± 样本标准差（ddof=1）。

| 指标 | Control (GAT) | GATv2 |
|---|---:|---:|
| 有效键 micro-F1（%） | 79.769 ± 0.631 | 79.620 ± 0.304 |
| 有效键 accuracy（%） | 80.042 ± 0.352 | 79.754 ± 0.249 |
| ROC-AUC | 0.882735 ± 0.003578 | 0.879857 ± 0.002219 |
| Ex-F1 调和口径（%） | 75.380 ± 0.261 | 75.205 ± 0.291 |
| q_seq（序列级 Spearman） | 0.62984 ± 0.04629 | 0.62134 ± 0.03772 |
| q-Brier（越低越好） | 0.066936 ± 0.002317 | 0.068487 ± 0.002013 |

### 相对同期 Control 的逐划分差

| 划分 | GATv2 ΔF1（百分点） | GATv2 Δq_seq |
|---|---:|---:|
| 1222 | -0.10495 | -0.011433 |
| 2252 | -0.12297 | -0.014782 |
| 3514 | -0.57165 | -0.044557 |
| 6072 | -0.35149 | +0.003868 |
| 9075 | +0.40234 | +0.024437 |

### 四项工程门槛

判定使用未舍入值；F1 获胜要求 ΔF1 > 1e-12。所有门槛必须同时通过。

| 实验臂 | 平均 ΔF1 ≥ +0.20 百分点 | F1 至少 4/5 获胜 | 平均 ΔAUC ≥ 0 | 平均 Δq-Brier ≤ 0 | 待复验候选 |
|---|---:|---:|---:|---:|---|
| GATv2 | -0.149742 (未通过) | 1/5 (未通过) | -0.00287774 (未通过) | +0.00155102 (未通过) | 否 |

待复验候选：无。不依据本次测试均值指定论文主模型。

### 六指标配对差：GATv2 − Control

三项百分比指标的差值以百分点表示；其余用原尺度。q-Brier 减少计为获胜；其他指标增加计为获胜。

| 指标差值 | 1222 | 2252 | 3514 | 6072 | 9075 | 均值 ± SD | 获胜 / 平局 / 失利 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 有效键 micro-F1（百分点） | -0.1049 | -0.1230 | -0.5717 | -0.3515 | +0.4023 | -0.1497 ± 0.3627 | 1 / 0 / 4 |
| 有效键 accuracy（百分点） | -0.5225 | -0.4242 | -0.5817 | -0.2314 | +0.3220 | -0.2876 ± 0.3657 | 1 / 0 / 4 |
| ROC-AUC | -0.0046620 | -0.0040678 | -0.0057202 | -0.0018765 | +0.0019379 | -0.0028777 ± 0.0030361 | 1 / 0 / 4 |
| Ex-F1 调和口径（百分点） | -0.1633 | +0.1412 | -0.3867 | -0.6581 | +0.1907 | -0.1753 ± 0.3578 | 2 / 0 / 3 |
| q_seq（序列级 Spearman） | -0.011433 | -0.014782 | -0.044557 | +0.003868 | +0.024437 | -0.008494 ± 0.025426 | 2 / 0 / 3 |
| q-Brier（越低越好） | +0.0036265 | +0.0019552 | +0.0030374 | +0.0002930 | -0.0011570 | +0.0015510 ± 0.0019746 | 1 / 0 / 4 |

GATv2 改变打分非线性与边偏置的位置，共享 source/destination 投影保持不变。参数量为 GAT 1,706,153、GATv2 1,705,641，相差 −512，并非严格等参数。

Ex-F1 先在每划分内由 ex_precision/ex_recall 求调和均值再取均值。q-Brier 是对条件组 q 的概率误差，并非纯校准误差。

五组重复划分共享序列，未进行独立样本 t 检验，不作显著性或置信区间推断。每划分一次初始化不能证明跨种子稳定。

多轮方法筛选已参考相同测试划分，本轮属于探索性实验；正结果仍需不同初始化及未用于方法筛选的数据复验。

### 结论边界

GATv2 相对随机 Control：ΔF1 -0.149742 个百分点、1/5 胜，ΔAUC -0.002877740、Δq_seq -0.008493531、Δq-Brier +0.001551025。

GATv2未通过全部四项预定门槛，不列为待复验候选；冻结主表保持不变。 通过也只表示达到预设工程筛选门槛，不自动替换论文主模型。五组重复序列级划分共享序列、每划分一次初始化，且多轮方法筛选已参考相同测试集；不报告独立样本t检验、统计显著性或跨种子稳定性，需要未参与筛选数据及不同初始化复验。Ex-F1逐划分由谱平均P/R求调和均值再平均；SD为ddof1；q-Brier是对条件组q的概率误差，不是纯校准误差。

文献证据沿用§15.27：`ent_000028 → .registry/machine/entity_resource_map.yaml → res_000143 → file_000143`，公共authoritative.pdf SHA256=`d316a7d23d735becf0c31d2b73c3426bb2aa5ec592705ce559d79a740abf53ec`，物理第5页§3.3公式(6)–(7)。旧映射缺map_id，未补造。现有edge bias/gate使整网不等同无边特征标准GAT，论文定理不直接作为本模型瓶颈证明。原始谱归档的只读核验是后续离子监督的数据准备，不改变本轮输入、标签或判据；本轮未写回知识库。

### 后续目标域离子监督的数据准备

此项是后续方法的数据准备，与上述attn6两臂结果分开记录。归档清单含2,049个MGF文件；每个文件前至多3谱的有界检查共读取6,126条谱，覆盖目标序列513/513。首部抽样覆盖不代表文件内全部谱均已核验。最初8条精确训练记录的PBCLA union核对为8/8一致，仅属小样本检查，不能替代全量校验。

v1全量构建目标为原fold1222训练侧383,295行、410条序列、7,643个条件组，实际状态`failed`且未发布不完整缓存。首错训练行357546的重建/原始OR bits为125040380/125564668。局部诊断确认自定义解析器与pyteomics的float32峰数组完全相同；y9-NH3、碎片charge=1的理论m/z=994.52037824、已量化峰m/z=994.500488281，20ppm容差0.0198904075648；当前float32标量差-0.0198974609375未命中，同一量化峰的float64差-0.0198899591099命中，并恢复该条谱与原CSV的union一致。

本次明确选择的v2 manifest为`/mnt/pvc/graphtrans/target_ion12_work/target_ion12_fold1222_v2_snapshot/manifest.json`，不会自动改用其他同名失败构建。v2采用固定float32峰量化后，将搜索数组显式提升到float64执行原二分查找；intensity仍为float32，20ppm/eps及PBCLA通道规则不变。这是明确声明的数值兼容方式，不是已核实的历史NumPy环境复原；单条谱和合成边界示例不能证明所有行均兼容。v2最终manifest状态为`passed`：原fold1222训练行383,295/383,295、410条序列、7,643个条件组；扫描归档记录477,669，其中94,374条未加入缓存。全部行的12通道OR与原true_multi精确一致，全部组计数/q重构及前体强度身份核对通过。这是逐行/逐组兼容性校验通过，不是新监督训练结果或化学真值证明。

缓存成员范围只允许原fold1222训练CSV中的精确记录，不加入archive-only谱或使用测试标签；其他四划分未在此缓存准备中覆盖。新12通道监督尚未训练，未参与本轮attn6输入、损失、模型选择或四项门槛。通道匹配是可观测离子的操作性标签，不能直接等同化学断裂真值。

数据准备证据（记录时读取的小型文件；未读取SQLite或原始归档）：

- `attn6_target_archive_inspection.json`：SHA256=`c0a6a1fa9329cde6a33619e2760e3a0f520f5bb7ed0f31011cf047058868b2ff`。
- `attn6_ion12_first_mismatch_diagnostic.json`：SHA256=`b8b6a33a1fe4dd6d4b83e9b495b3e6c5d9cb3511f1d2f3f44fd5785d7e6104d0`。
- `target_ion12_fold1222_v1/manifest.json`：SHA256=`ffdec1c4f837a781cc459b31f0fb817d698f1c7e228ee8c4bf34e4236642e877`。
- `target_ion12_fold1222_v2_snapshot/manifest.json`：SHA256=`19e9825247916b83cd0d78990303622e4da1763c26f3f03767b74051c00abcc7`。


`project_goal`：检验共享投影GATv2并完成10次训练及事件触发判读。`completion_status`：训练、审计与结果记录完成；GATv2未通过全部四项预定门槛，不列为待复验候选；冻结主表保持不变。`target_kb`：kb:storage。`knowledge_used/source_records`：§15.27的ent_000028/res_000143/file_000143公开证据。`created_files/modified_files`：attention_v2与测试、两配置、最小模型接入、审计/分析工具及实验台账。`unresolved_questions`：受限变体的跨种子和新数据泛化，及目标域离子监督适用性。`suggested_knowledge_writeback`：可另行摄入完整工程结果，本轮未写知识库。

## 15.29 目标域12通道碎片匹配辅助监督：第七轮预定方案（2026-10-18）

状态：用户继续优化授权下实现；本节在正式训练前固定目标、两臂、训练及判据。数据处理、数值核验和训练均在pod，本机仅编辑和同步。采用完成事件，不实时读取epoch状态。冻结论文主表不变。

### 依据与研究问题

storage公共来源沿用已核验的实体—资源链：Prosit `ent_000036→res_000151→file_000151`，`Library/public_resources/res_000151/authoritative.pdf`物理第11页，按位置/b-y/电荷组织碎片输出；AlphaPeptDeep `ent_000004→res_000119→file_000119`，同目录`res_000119/authoritative.pdf`物理第10页，MS2模型预测碎片通道及中性丢失（文件SHA见§15.25）。这些工作支持使用碎片层级监督的思路，并未验证本项目的12通道BCE或λ=0.1。此前结构小分支、受限GATv2及外部离子强度迁移没有通过全部采用门槛；不能据此证明监督粒度是唯一瓶颈。本轮检验同一目标域原始谱图提供的细分监督是否改善既有断键任务，不复制文献完整网络，不增加外部预训练。

### 数据与目标

原始来源为用户提供的`pep_521_mgf.tar.gz`，压缩字节SHA256=`1b2172d00f1aa39b13e8aa6e2e7ed4ab64b822aa685fded807b17e8733d3e805`。完整1222训练侧重构已经通过§15.28逐行核验。五份原训练CSV的身份盘点并集为477,669谱/513序列；其中383,295谱/410序列已覆盖，另94,374谱来自103个新增序列。同一序列在其出现的各训练划分中，完整记录集合、多重性和条件组计数完全一致。因此仅补匹配新增记录，按各划分自己的训练条件白名单导出sidecar；不读测试CSV参与目标构建。

每条记录按原TITLE/name、SEQ、scan、charge、NCE、PEPMASS、RT精确身份匹配，并另核对前体强度。B/O/X/Z使用项目PBCLA质量表。峰先量化float32，再用float64搜索运算；仍为原nearest-peak、20ppm、eps=1e−6、最近峰强度>0规则。这是§15.28显式数值兼容方案，不声称恢复了历史NumPy环境。每条原训练谱的12槽OR必须与冻结true_multi逐键一致；不一致即停止，不改主标签或容差。

固定通道顺序：`b1,b2,y1,y2,b1-H2O,b2-H2O,y1-H2O,y2-H2O,b1-NH3,b2-NH3,y1-NH3,y2-NH3`。这里预测的是原算法在导出峰表上的匹配事件，未匹配为操作性0，不能解释为物理离子不存在。一个峰满足多个槽时保留全部命中，不强加一对一匹配；padding以外不按损失或预测删槽。每个条件组 `(seq,charge,NCE)` 的辅助目标为逐键逐通道命中计数除以原group_n。主任务q继续使用原soft_multi，不用OR/独立性假设从通道均值重算q。

五份sidecar分别绑定该fold原始train及folded train SHA，校验完整条件集合、group_n及原union计数。folded soft_multi原来只保留4位小数，因此与未舍入union频率的允许差固定为5.0001e−5；这只用于序列化一致性检查，不改动已存主任务q或逐谱OR的严格一致要求。加载器仅为该fold train及内部序列隔离val附加目标，test保持原行级数据且不加载辅助目标。内部train/val同序列不跨侧；各划分模型独立训练。

### 两臂与唯一干预

- control：原GAT+theory/FiLM随机初始化，增加同样的`Linear(256,12)`辅助头，λ=0。
- aux：完全相同模型与初始化，λ=0.1。

两者均1,709,237参数，比原主模型增加3,084。辅助头作用于原bond_head第一层Linear/ReLU之后、原Dropout之前的隐藏表示；不新增Dropout，初始化位于旧初始化之后并隔离RNG。主输出仍走原Dropout和二分类Linear；训练态辅助logits单独暂存，eval/inference不经过辅助头。原峰、离子目标、匹配强度都不作为模型输入。旧fragment/chemistry/sequence adapter/晚期条件/GATv2分支关闭，无预训练或冻结阶段。

目标为 `L = L_main + 0.1 * mean_condition(mean_valid_bond(mean_12channels(BCEWithLogits(z_ion, q_ion))))`。主loss和原group_uniform权重语义不变（主loss仍按有效键平均）；辅助loss按条件组平均，不乘group_n。control仍计算同头输出/诊断loss，但λ=0时不把辅助loss接入反向图。每epoch单独记录主loss、辅助loss和λ；合成日志保留主有效键均值与辅助条件均值各自分母。辅助验证指标不用于选点；本轮不支持ion12 resume。

### 固定训练与采用门槛

两臂各跑1222/2252/3514/6072/9075五组重复序列级80/20划分，seed42配对、内部序列val20%、训练条件组折叠、测试行级不变。pre输入mask为state[charge,m/z,intensity]=[T,T,F]、env[NCE,scan]=[T,F]；R=5距离图、3GCN+2GAT、hidden256/H4、原15维理论特征和FiLM不变。batch128/drop_last、AdamW lr1e−4/wd1e−4、warmup40、cosine floor1e−5、最多300epoch、patience25/min_delta.001不变。主checkpoint严格验证max-F1，测试阈值固定0.5。λ=0.1只作为一个预先固定工程假设，不扫描网格或按测试结果调阈值/训练时长。

相对本轮control，四项须同时满足：平均ΔF1≥.002（+0.20个百分点）、至少4/5划分严格F1获胜（差>1e−12）、平均ΔAUC≥0、平均Δq-Brier≤0。使用未舍入值，均值±样本SD(ddof=1)；Ex-F1仍为每划分谱平均P/R调和后再均值；报告全部六指标及配对差。仅表示待独立复验候选，不自动替换论文主表。五组划分共享序列、每划分一次初始化且多轮已查看相同测试数据，本轮是探索性筛选；不做独立样本t检验、不宣称显著性或跨种子稳定。

数据准备已完成：`/mnt/pvc/graphtrans/target_ion12_work/ion12_five_train/manifest.json` SHA256=`20ad4e2edf52a392794ebce0f1c847a6f7197f45715449dbaeaca30aa8bd5d8f`，477,669条谱/513序列/9,552个并集条件；新增94,374条全部逐行OR一致，各fold侧条件数依次7,643/7,721/7,649/7,650/7,605，缺失0、原整数计数/n完全一致。训练只使用各自`fold_{fold}/targets.json`和manifest，不直接加载并集谱级SQLite。五份sidecar最大q差均为5e−5，符合既有四位序列化精度。

实现验证包括：head关闭的兼容路径、λ0共享初始化/RNG/主输出与梯度、辅助梯度进入主干、条件归一/padding、checkpoint模式拒绝、完整sidecar绑定、普通/边缓存/全图缓存、test排除及history记录。核心8项及数据接入15项测试在pod通过；五份sidecar已全部passed。真实1222内部train前16组、417有效键的两臂生产Trainer AMP单步均通过：共享初始化/RNG配对、梯度/参数有限、主干及分类头更新，辅助头仅λ=0.1更新。GPU非确定性聚合时同初始化主loss曾差3.93e−5；仅在单步核验启用确定性算法后，两臂主/辅助loss完全一致，正式训练的原deterministic=false协议不变。检查回执`/mnt/pvc/graphtrans/target_ion12_work/ion12_cuda_smoke_result.json`；单步损失不作为效果结果。完成后按固定来源和选点核验再判读，不实时监控训练。

`project_goal`：实现目标域12通道辅助监督，完成push/pull、配对训练与事件触发判读。`completion_status`：代码和数据已同步，配对10次训练及完成分析均成功结束；结果未通过全部预定门槛，详见§15.30。`target_kb`：kb:storage，只读。`knowledge_used/source_records`：本节Prosit/AlphaPeptDeep公共证据及项目PBCLA操作性定义。`created_files/modified_files`：ion12 dataset/loss/head、显式训练接入和两配置、专用测试、pod数据准备与完成分析脚本、本台账。`unresolved_questions`：目标域细分监督能否改善主任务、独立数据与初始化泛化。`suggested_knowledge_writeback`：可另行摄入完整实现与结果，本轮未写知识库。


## 15.30 目标域12通道辅助监督：第七轮配对结果（2026-10-18）

完成：10/10 runs，自动完整性分析通过。训练开始 2026-10-18 21:44:19 +08:00，结束 2026-10-18 22:22:45 +08:00；后处理完成 2026-10-18 22:22:56 +08:00。代码 `265dd6f9e13498136b1fdaf01055b4e1f7113627`，GPU 0,1,2。

Aux 相对同期 control 的平均 ΔF1 为 -0.031643 个百分点，严格获胜 3/5 划分；ΔAUC=-0.000027，Δq-Brier=-0.000493，Δq_seq=+0.004042。未满足全部四项预定门槛，本轮不采纳该候选。不自动替换冻结论文主表。

按§15.29，两臂均为 GAT+theory/FiLM 和同一个 Linear(256,12) 辅助头，均1,709,237参数、随机初始化；唯一训练干预为 λ=0 / 0.1。辅助 BCE 先平均12通道与有效键、再按条件组平均；主损失保持有效键均值。离子目标不作为输入，eval 不运行辅助头；无预训练/迁移。主checkpoint严格按验证最大F1选取，测试阈值固定0.5。

### 六项指标与配对差

| 指标 | Control λ=0 | Aux λ=0.1 | 配对均值差 |
|---|---:|---:|---:|
| micro-F1 (%) | 79.838 ± 0.607 | 79.806 ± 0.625 | -0.032 |
| 有效键 accuracy (%) | 80.013 ± 0.361 | 80.024 ± 0.329 | +0.011 |
| ROC-AUC | 0.882399 ± 0.003525 | 0.882371 ± 0.003172 | -0.000027 |
| Ex-F1 调和口径 (%) | 75.450 ± 0.284 | 75.436 ± 0.242 | -0.014 |
| q_seq | 0.62385 ± 0.05932 | 0.62789 ± 0.04505 | +0.00404 |
| q-Brier (越低越好) | 0.067226 ± 0.002230 | 0.066733 ± 0.002295 | -0.000493 |

F1、accuracy、Ex-F1 差使用百分点，其余原尺度。Ex-F1在每划分先由谱平均P/R求调和均值，再汇总；accuracy仅有效键。q_seq只作描述，不替代四项门槛。

| 划分 | Control F1 (%) | Aux F1 (%) | ΔF1 (pp) |
|---|---:|---:|---:|
| 1222 | 79.7812 | 79.2343 | -0.5469 |
| 2252 | 79.7319 | 79.9856 | +0.2537 |
| 3514 | 80.7799 | 80.7351 | -0.0448 |
| 6072 | 79.8166 | 79.8510 | +0.0344 |
| 9075 | 79.0803 | 79.2257 | +0.1453 |

| 预定工程门槛 | 已验证判定 |
|---|---|
| 平均 ΔF1 ≥ +0.20 pp | 未通过 |
| F1 严格改善至少 4/5 划分 | 未通过 |
| 平均 ΔAUC ≥ 0 | 未通过 |
| 平均 Δq-Brier ≤ 0 | 通过 |

选中epoch/终止epoch（按1222/2252/3514/6072/9075）：control：1222=74/99 / 2252=84/109 / 3514=85/86 / 6072=80/105 / 9075=82/96；aux：1222=59/71 / 2252=84/109 / 3514=76/86 / 6072=83/88 / 9075=83/102。

### 数据、验证与来源

五份原训练集合并集为 477,669 谱 / 513 序列 / 9,552 条件组；各模型仅使用本划分训练白名单 sidecar。内部train/val序列隔离、两臂同fold split字节一致，外部test不加载辅助目标。

并集manifest SHA256：`20ad4e2edf52a392794ebce0f1c847a6f7197f45715449dbaeaca30aa8bd5d8f`。每fold目标绑定如下：

| 划分 | 条件组 | manifest SHA256 | targets SHA256 |
|---|---:|---|---|
| 1222 | 7,643 | `9b02237fc0da11f4b9dcb5fa3cd5f48dece4f9ed5eec9d880a5bf206abbc9625` | `8bc3508b0702c72d6b50b7cae4b394fda04dd410f67ff86e6bafb0d6e6310f87` |
| 2252 | 7,721 | `ac969af3406809a5b6d4ef5c7fd3f73a7f96ef3f1a244501ec69e0c902ca5793` | `d3a7579c75268cde7cbcd7a8da36a5eabea1e7702cfa81d12114c1e0386e273d` |
| 3514 | 7,649 | `3caf31e12e7593383ee61b0356fb89f4b0e124ed676384b6bbc413791970287e` | `2e187855cf266a53337dfe655bfb517bb99b2a70cddcf7b63bf10ade645698a8` |
| 6072 | 7,650 | `e600b487b7ff5f0dc8edb7148fd2a83f5964e18ac5a63bd95a0ac7c1a17515a7` | `c9a806b900d00626917f2096d43dbe0d691c2476d34ad67326eb46a90344af1a` |
| 9075 | 7,605 | `ce0572dcd2651bce83164bcc17f40593cdf1c446d2364b9f82f19b88f9f55fb9` | `88aa1a67d5dee1ec4d1edd5a1f8f4f1fb718c3d237d1ea399bd4144621e7f2b0` |

实施验证：核心8项、数据接入15项（共23项），另有数据构建4项；这些不是新增训练重复。真实训练组上的两臂CUDA AMP单步检查通过，共享初始化/RNG配对；仅smoke启用确定性算法，正式训练仍采用原deterministic=false设置。smoke两份源文件与生产gitblob的差异已严格证明仅为CRLF→LF，其余24份源字节一致；不推广为一般代码等价。本次渲染未重跑测试、推理或独立统计复算。

完成分析绑定launch/全部源码与配置/五fold目标SHA、10run终态、严格max-F1选点、主/辅助损失日志、主/q/periodic目标provenance及原始metric路径/版本化SHA；仅使用已验证decision与analysis_tables生成本节。

结果根：`/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/ion12_batch/20261018_134417`。`integrity.json` SHA256=`2bce026ad3e638c85ae1dab3bf686ec0d27f9d538ea04b818b5c9e07da85c02e`；`decision.json` SHA256=`434be44576cdc1b0e5d0e3c9c188294cce5e7970367c824731c11f28a970371c`；`analysis_tables.md` SHA256=`07673ca7c2d060f10ed10f666de246822c0a80f8f50ea5164bb9ac83a96caed8`。完整输入/输出绑定见rendering_receipt.json。

这五组是共享序列的重复序列级 80/20 随机划分，每划分仅一次初始化；此前已多轮查看同一测试集合，本轮属于探索性筛选。mean±样本 SD（ddof=1）与配对差均为描述性统计，工程门槛不等于统计显著性，不能据此断言跨种子稳定或对独立新序列、其他采集条件/数据集的泛化。离子目标是原峰匹配算法的操作性事件，不等同物理离子存在/不存在或化学断裂真值。

运行补记：两个完成监听器共用固定临时文件，导致其中一个写入完成通知时出现FileNotFoundError；另一个监听器及正文渲染正常exit 0，训练和统计分析均已成功完成。仅修复后续使用的未冻结helper，使各写入者使用独立临时文件；pod针对性并发写及异常清理验证通过。冻结启动快照和全部分析/渲染结果SHA未变，没有重跑训练或复算统计。修复回执为`/mnt/pvc/graphtrans/ion12_ops/corrections/atomic_notification_race_20261018/correction_receipt.json`，SHA256=`c7df24cc2185b97bb9c265ad02f494fd145d0860dd7cbf79023203e5f2795ecf`。

`project_goal`：完成目标域12通道辅助监督的配对训练与预定判读。`completion_status`：10次训练和完成分析已结束；本候选未通过全部门槛，冻结主表不变。`target_kb`：kb:storage，只读。`knowledge_used/source_records`：§15.29的Prosit/AlphaPeptDeep公开证据及PBCLA操作性定义。`unresolved_questions`：独立数据与初始化泛化。`suggested_knowledge_writeback`：可另行摄入工程结果，本轮未写知识库。

### 15.31 结构精简预注册（2026-10-19，待结果）

假设：重复的读出表示及过深的图传播可能增加小样本方差；这是假设，尚无因果证据。
四臂为 `pre_synthesis_slim_{control,lean_head,shallow,no_global}.yaml`：
control 原始 3GCN+2GAT；lean_head 只保留 h_i、h_j 和理论特征（同时删除读出块的 edge/diff/product，图层仍使用边）；shallow 改为 2GCN+1GAT；no_global 仅关闭全局节点。每臂独立随机初始化，不迁移权重；相同种子不代表不同结构的共享参数逐位相同。读出和深度臂是模块级删减，不能区分模块内部各成分的贡献。
沿用原始优化器、学习率、早停、5 组重叠的序列级80/20划分与 padding-free 评价；对照重新训练，不与历史最优直接作配对。
提升标准沿用 ΔF1≥0.002、至少4/5划分严格提高、ΔAUC≥0、ΔqBrier≤0，四项均需满足。
另预注册描述性压缩筛选：参数减少≥10%、平均F1下降≤0.001、至少4/5划分F1下降≤0.002、平均AUC下降≤0.001、平均qBrier增加≤0.001。此标准只产生待独立确认的压缩候选，不等于统计学非劣效证明，也不替换冻结主表。所有臂全部报告，不按最佳测试臂修改阈值。
训练在 pod；完成/失败事件后统一分析，不读取 epoch 进度。

### 15.32 精简实验结果（2026-10-19）

源码 `e832e8ab5d78116f72447129f94b47c865e38e3b`；批次 `checkpoints/graph_transform/slim_batch/20261019_021556`。四臂各5组划分均成功，批次终态 completed；训练结束时间记录为 2026-10-19 03:26:48 UTC。pod 自动后处理完成，原始报告 `/mnt/pvc/graphtrans/slim_ops/{result.json,report.md}`；本地小报告 `F:/codexout/dbond_slim_results_20261019/`。后处理检查训练源码和配置哈希保持不变。沿用有效键 mask 的生产指标，不重新推理。

| 臂 | 参数 | F1 % | ΔF1 百分点 | AUC | q-Brier |
|---|---:|---:|---:|---:|---:|
| control | 1706153 | 79.7626 | 0 | 0.881796 | 0.067407 |
| lean_head | 1509545 | 79.7243 | -0.0383 | 0.879920 | 0.068934 |
| shallow | 1309093 | 79.1440 | -0.6186 | 0.873403 | 0.071701 |
| no_global | 1689257 | 79.0999 | -0.6626 | 0.873074 | 0.072190 |

所有删减臂均未通过预注册提升标准或压缩筛选。lean_head 虽减参11.5%且F1基本保持，但AUC下降约0.001876、q-Brier增加约0.001527，超过压缩筛选容忍值；不能只看F1采纳。浅骨干减参23.3%但F1下降0.6186个百分点；去全局节点只减参1.0%却下降0.6626个百分点。本轮不支持“这些模块整体冗余，删除可提升性能”的假设；不代表所有局部删减均无效，且模块级实验不能归因到被同时删除的单个特征或层。保留原版，冻结主表不变。后续若继续精简，优先单独删除差值、乘积、读出边表示之一，避免直接组合删除。

### 15.33 位置编码移至注意力前相加（2026-10-19，预注册）

用户指定在原模型基础上，将位置编码放在进入注意力层前，相加而非拼接。配置 `pre_synthesis_position_{control,pre_attention_add}.yaml`；保持原始3GCN+2GAT、全局节点、理论特征、FiLM、预测头及完整训练预算。control 延用 input_concat；实验臂从节点编码器输入移除32维位置拼接（192→160），保留学习式绝对位置表32维，经无偏置Linear(32→256)后，在3层GCN结束、第一层GAT开始前只相加一次。真实残基按各自序列位置编码，全局节点和padding不加位置向量；没有额外LayerNorm/dropout/门控。图中的相对序列距离/边编码保留。实验是位置注入方式与时机的组合变更，不能分别归因于时机或相加。

旧配置缺少新开关时完整保留原始计算、初始化和checkpoint结构；新模式使用checkpoint标记，跨模式加载应在修改参数前失败。独立从头初始化，维度变化会改变随机初始化流，不宣称两臂共享参数逐位相同；不迁移简化实验权重。5组重叠序列级80/20划分、mask有效键指标、相同训练/选择协议；沿用提升标准ΔF1≥0.002、≥4/5严格提高、ΔAUC≥0、ΔqBrier≤0，探索性结果不替换冻结主表。所有计算和验证在pod；只在完成/失败事件时后处理，不读epoch进度。

### 15.34 注意力前相加位置编码：完成结果（2026-10-19）

源码 `7be0fa19b27d73495d2905081e414506934dc713`，批次 `checkpoints/graph_transform/slim_batch/20261019_034601`。两臂共10次训练成功，终态完成时间 `2026-10-19T04:23:42.755791+00:00`（北京时间12:23:42）。对照为同轮原结构重训，非历史冻结表的单次结果。两臂各1,706,153参数，其他协议见§15.33。

| 指标 | 原版对照 | 注意力前相加 | 相加−对照 | 改善划分 |
|---|---:|---:|---:|---:|
| 有效肽键准确率 % | 79.907549 | 79.786495 | -0.121054 pp | 1/5 |
| 微平均F1 % | 79.755323 | 79.455137 | -0.300186 pp | 2/5 |
| ROC-AUC | 0.881287 | 0.880677 | -0.000610 | 2/5 |
| 谱图P/R调和F1 % | 75.332588 | 74.832086 | -0.500502 pp | 1/5 |
| q_seq | 0.619729 | 0.594077 | -0.025652 | 1/5 |
| q-Brier（越低越好） | 0.067620 | 0.066379 | -0.001241 | 3/5 |

逐划分F1变化（1222/2252/3514/6072/9075，百分点）：-0.583888 / +0.285808 / -1.213088 / -0.069199 / +0.079435。提升四项门槛仅q-Brier通过，其余三项未通过，不替换原版或冻结主表。q-Brier变好表示平均q拟合误差降低，不等于分类、序列排序同步改善。本次同时改变了注入时机和融合方式，不能分别归因，也不能推广为所有相加位置编码均无效。GCN前段缺少绝对位置可能影响表示学习，只是待验证解释，不能由本轮确认。

核验在pod完成，无新训练或推理：检查10份已成功划分输出、启动源码SHA未变、配置快照SHA、原始metric与summary绑定SHA，并从保存CSV独立复算全部16项臂×指标的均值/样本SD/配对差/胜率（1e-12内一致）。原GT mask评价不变。pod详报 `/mnt/pvc/graphtrans/position_ops/review.md`，核验回执 `review_receipt.json`；本地 `F:/codexout/dbond_position_results_20261019/`。

### 15.35 完成提醒补齐（2026-10-19）

此前仅在pod生成终态和报告，没有建立应用回到对话的提醒，导致用户必须追问。现已创建原对话heartbeat `dbond`（名称DBond训练完成提醒），每60分钟仅检查完成/失败收据，不读epoch进度。扫描器 `/mnt/pvc/graphtrans/dbond_completion_events.py`，去重状态 `/mnt/pvc/graphtrans/dbond_completion_notified.json`。已手工验证终态扫描、当前两轮已报告登记、空队列；应用确认ACTIVE。尚未声称验证过后续定时主动投递；本机应用与SSH可用是运行条件。

后续本对话批次的launcher应在独立 `/mnt/pvc/graphtrans/*_ops/` 目录写 `launch.json`（含root）和 `terminal.json`（含同一root及completed/failed，成功含result/report，失败含error）；root必须位于 `/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/` 下。旧运行文件保留，不复用已完成路径。heartbeat只对scan返回的新终态读小报告并汇报，随后ack去重；无新结果不发进度，禁止自动启动新实验。此机制是每小时补查终态，不是pod事件直接唤醒应用。

### 15.36 融合头交叉注意力后接MLP（2026-10-19，预注册）

用户指定更换逐肽键融合头。原版底座保持3GCN+2GAT、输入端拼接位置编码、FiLM、全局节点、理论特征；不叠加位置相加或结构删减。配置 `pre_synthesis_cross_fusion_{control,cross_attention}.yaml`。对照为原6路拼接1536→256→1。实验为每条有效键的2个端点表示h_i/h_j作为Q，4路e_ij、h_i−h_j、h_i*h_j、theory_proj作为K/V，四头交叉注意力（hidden256，dropout0.1）后做端点残差加LayerNorm，保留有序左右两路展开512，再接MLP512→256→ReLU→Dropout0.1→1。K/V含学习式来源类型向量；Q长度2、K/V长度4，注意力只在该肽键内部的特征来源间计算，不跨键或跨样本。不是用单个K/V的退化注意力，也不是对全部6路做自注意力。

旧配置默认为concat_mlp，不改变原计算、参数名或初始化；交叉头在旧版初始化后替换临时拼接头并使用独立RNG上下文，因此两臂同种子时骨干共享初始化逐位一致，拼接头不留在实验参数中。checkpoint记录头类型、注意力头数和K/V来源，跨结构/语义错配应在加载修改权重前失败。当前不与旧condition_readout或ion12辅助头叠加。

每臂5组重复序列级80/20划分，从头训练，预算、优化器、早停和F1选点一致；指标去除padding，Ex-F1沿用谱图P/R调和口径。预注册采用门槛沿用平均ΔF1≥0.002、F1≥4/5严格改善、平均ΔAUC≥0、平均ΔqBrier≤0，四项均需通过，探索性结果不替换冻结主表。计算和核验仅pod；完成/失败后自动汇总，用已启用heartbeat检查终态提醒，不读取epoch进度。

### 15.37 逐键交叉注意力融合头：完成结果（2026-10-19，heartbeat主动提醒）

源码 `84520e67e211877dafe2d528d8110bcf4447beca`，批次 `checkpoints/graph_transform/cross_fusion_batch/20261019_044452`。两臂共10次训练成功，完成时间2026-10-19 05:23:18 UTC（北京时间13:23:18），05:44 UTC的原对话heartbeat扫描到新完成事件并处理。对照为同轮原版重训，不使用历史最优作配对；原版1,706,153参数，交叉头1,708,713。骨干共享初始化保持相同，位置编码仍为原始输入拼接。

| 指标 | 原版对照 | 交叉注意力+MLP | 相比对照 | 改善划分 |
|---|---:|---:|---:|---:|
| 有效键准确率 % | 79.932782 | 79.785116 | -0.147666 pp | 2/5 |
| 微平均F1 % | 79.724122 | 79.642018 | -0.082104 pp | 2/5 |
| ROC-AUC | 0.881620 | 0.879054 | -0.002566 | 1/5 |
| 谱图P/R调和F1 % | 75.307899 | 75.192885 | -0.115013 pp | 2/5 |
| q_seq | 0.618793 | 0.603088 | -0.015705 | 2/5 |
| q-Brier（越低越好） | 0.067302 | 0.069554 | +0.002253 | 1/5 |

逐划分F1变化（1222/2252/3514/6072/9075，百分点）：+0.086494 / -0.152023 / -0.547393 / -0.150766 / +0.353170。预注册四项采用门槛全部未通过，主要指标均值均无改善；保留原版，冻结主表不变。F1差异较小，且只有5组重叠划分、每组一次初始化，不据此断言统计显著或所有交叉注意力均无效。此次检验的是端点Q对边/差值/乘积/理论特征K/V的指定局部融合实现。

pod核验启动源码SHA、配置快照SHA、原始metric绑定SHA，并从保存CSV复算16项臂×指标均值/样本SD/配对差/改善划分数，1e-12内与汇总一致。沿用生产有效键mask，未重新推理、调阈值或启动训练。详报 `/mnt/pvc/graphtrans/cross_fusion_ops/review.md`，回执 `review_receipt.json`；本地小报告 `F:/codexout/dbond_cross_fusion_results_20261019/`。完整样本SD和逐划分数值见详报。完成提醒按事件键登记去重，不重复发送历史结果。

### 15.38 当前 FiLM 底座的 5GAT 对照与历史正弦项审计（2026-10-19，预注册）

用户回忆早期纯 GAT 效果较好，要求在现有基础上复测。历史 §4 的 5GAT 基于行级 theory 底座，未包含当前条件组折叠和 FiLM；本轮检验当前主线中的骨干选择，不复用历史最好值作对照。配置 `pre_synthesis_pure_gat_{control,gat5}.yaml`：control 为原版3GCN+2GAT，gat5 明确设置0GCN+5GAT；两者总消息传递深度均为5。保留输入拼接位置编码、全局节点、边偏置/门控、理论特征、FiLM及原1536→256→1拼接头，不叠加交叉注意力或其他近轮修改。模型现有实现已支持零GCN，不改前向算法。结构改变会改变参数量与初始化抽样，不宣称等参或共享张量初始化相同。

按同轮两臂、每臂5组重叠序列级80/20划分运行，独立从头训练；种子、数据、优化器、学习率、预算、早停和验证最大F1选点保持原协议。有效键mask、测试阈值0.5、谱图平均P/R调和F1口径不变。预定工程筛选仍为平均ΔF1≥0.002、F1至少4/5严格改善、平均ΔAUC≥0、平均Δq-Brier≤0，全部满足才列为待确认候选；不等于统计显著，不直接替换冻结主表。

代码检查确认 `1+0.2*sin(d/2)` 仍在 `data/graph_builder.py` 的活动距离连边路径及一个当前未被调用的历史矩阵方法中。没有在模型激活中发现该正弦项；GAT的 `leaky_relu_slope=0.2` 是另一个参数。本次仅修正文档注释：旧注释将正弦项描述为no-op不严谨。现有md6预期等价于显式R5、仅相邻边为near；直接删除公式且保留md6会增加距离6的边，同时改变距离2的边型。是否完全等价、变化边数及实际模型层数/AMP更新将在pod审计并保存 `pure_gat_ops/distance_audit.json` 和 `smoke.json`。本批保持原连边算法，避免把骨干和图定义混为一个干预；不为0.2补造物理来源。

训练与数值验证仅pod。helper使用独立 `/mnt/pvc/graphtrans/pure_gat_ops/`，绑定源码/配置SHA，成功或失败写匹配的launch.json和terminal.json，由已有dbond完成提醒读取；不反复查看epoch。结果尚待训练，不能预判纯GAT是否改善。

启动与审计回执：源码 `77dfd4694cb3031713ccf94a1426607788b7a374`，批次 `checkpoints/graph_transform/pure_gat_batch/20261019_082206`，GPU 0/1/2，已提交后台worker，尚无效果结论。pod成功smoke显示control为1,706,153参数、3GCN+2GAT；gat5为1,315,253参数、0GCN+5GAT。两臂各完成真实16组、417有效键的生产Trainer CUDA AMP单步，全部GAT层前向执行且梯度非零、参数更新；有效输入屏蔽、图与标签一致。此次只改配置和注释，未改模型计算；不因单步损失差异判断优劣。

图审计在长度1–32及全局节点开/关的64种情形逐张量验证：活动历史公式与显式R5/相邻独占near规则的edge_index、edge_types、edge_distances、bond_edge_map完全一致，含边缓存路径。长度32且含全局节点时，直接删除公式并保持阈值6会使有向边354→406（新增52条距离6的边），另外60条距离2的已有边改变类型。模型模块未发现sin调用；两处sin位于graph_builder的活动连边和历史矩阵方法。回执 `/mnt/pvc/graphtrans/pure_gat_ops/{distance_audit,smoke,launch}.json`；本地小回执 `F:/codexout/dbond_pure_gat_launch_20261019/`。smoke首次外层SSH包装的退出码转义有误，但pod内smoke已成功保存passed收据；未重复计算，launcher独立核验收据、源码提交和配置SHA后成功提交。完成/失败事件仍由原dbond提醒扫描，不读取epoch进度。

### 15.39 当前 FiLM 底座 5GAT：完成结果（2026-10-19）

源码 `77dfd4694cb3031713ccf94a1426607788b7a374`，批次 `checkpoints/graph_transform/pure_gat_batch/20261019_082206`。两臂共10次训练全部成功，结束时间2026-10-19 09:00:55 UTC（北京时间17:00:55）。用户询问进度后检查pending完成事件，未读取运行中状态或epoch日志。control是同轮原版3GCN+2GAT，gat5是0GCN+5GAT；其余配置及历史正弦连边算法保持原版。参数1,706,153→1,315,253，减少22.9112%。

| 指标 | 同轮原版 | 5层纯GAT | 纯GAT−原版 | 改善划分 |
|---|---:|---:|---:|---:|
| 有效键 accuracy % | 79.808493 | 79.661798 | -0.146695 pp | 2/5 |
| micro-F1 % | 79.693517 | 79.670253 | -0.023264 pp | 2/5 |
| ROC-AUC | 0.880297 | 0.879921 | -0.000376 | 2/5 |
| 谱图P/R调和F1 % | 75.321622 | 75.088146 | -0.233476 pp | 2/5 |
| q_seq | 0.609693 | 0.606280 | -0.003412 | 2/5 |
| q-Brier（越低越好） | 0.068229 | 0.068579 | +0.000351 | 2/5 |

逐划分F1变化（1222/2252/3514/6072/9075，百分点）：-0.049602 / +0.609672 / -0.254158 / -0.797834 / +0.375602。四项预定提升门槛均未通过，保留原版和冻结主表。F1均值差很小，不能据此断言纯GAT显著变差；减参且F1均值接近说明它可作为待确认的轻量化方向，不等于已证明非劣效。没有运行成本/吞吐测量，不由参数减少推出加速或显存减少。5组重叠序列级划分、各一次初始化，只作探索性配对描述。

pod读取完成报告并独立从保存CSV复算16项臂×指标的均值、样本SD、配对差和胜率，1e-12内一致；核验启动源码SHA、helper SHA、配置快照SHA、原始metric SHA及result/summary绑定。沿用生产有效键mask，谱图F1先在每划分由平均P/R求调和均值；没有新训练、推理、阈值调整。详报 `/mnt/pvc/graphtrans/pure_gat_ops/review.md`，回执 `review_receipt.json`；本地 `F:/codexout/dbond_pure_gat_results_20261019/`。完成事件已ack去重，不自动启动其他实验。

### 15.40 折叠是否限制性能：只读数据/验证审计（2026-10-19，重要适用条件修正）

用户提出折叠后样本变少可能压制模型区分能力。本次仅在pod读取已完成pure_gat批次的1222对照配置、实际内部val.csv、折叠train与原始5fold_soft训练CSV；不训练、不推理、不读epoch日志、不修改运行协议。回执 `/mnt/pvc/graphtrans/folded_validation_audit_20261019.json`，本地同名小JSON位于F:/codexout/。

已核验：外部训练383,295谱图→7,643条件组，410序列；内部train为6,098组/328序列，内部val为1,545组/82序列，实际落盘val.csv与配置seed=949822381重建完全一致。对应这些验证序列的原始谱图有77,992行；折叠val的true_multi逐组与原始首行标签完全相同，soft_multi也一致。代码 `Trainer.validate_epoch` 用labels硬标签计算F1、loss，soft_labels只进入q指标；主checkpoint和早停以该F1决定。因此主验证只用每组首张谱图的硬标签，没有使用完整验证谱图。测试仍为未折叠谱图；此处证明验证表示与目标测试分布不同，不证明其导致了多大的性能损失。该选点不稳定隐患应优先做受控核验。

实际训练为group_uniform，每组行权重1并逐有效键展开；它改变了原逐谱训练中组大小n_g的相对权重，不能称为无条件损失等价。当前每个完整训练epoch（drop_last=true、batch128）47次更新；历史重标定不等于已经证明优化器步数/学习率调度/早停匹配。只在同组logit相同、纯BCE、保留组大小权重且比较全量目标时，sum_j BCE(z,y_j)=n_g BCE(z,q_g)严格成立；SGD、自归一小批权重、dropout、选点轨迹另需分析。历史“目标侧已饱和”“换骨干无用”等概括不由该恒等式或单折自检支持。

输入相同前提也需修正：原始7,643组中7,554组的pep_mass并非常数；组内范围中位0.00193846、P90=0.00476394、P99=0.01159223，最大2.83969814，原始唯一(seq,charge,nce,pep_mass)为356,417。多数差异很小且模型将pep_mass除以2000，可能是测量抖动；是否存在有用信息或异常尚未验证，不能直接据此归因。组内q字符串均一致（0组变化）。`fold_condition_groups.fold_frame`关于pep_mass恒定的历史文档假设不适用于当前实测数据，故真实行级输入不能严格视为完全相同；本轮未修代码/重做数据。

目前结论：压缩行数本身不等于丢失监督，但当前折叠流程同时涉及权重、实际输入保留、训练动力学和验证选点，尚无因果证据排除这些影响。建议先保持架构和训练折叠方式，只恢复内部验证序列的全量谱图；随后固定同一序列划分、全量验证与优化步预算，对比逐谱hard BCE、折叠+spectrum及折叠+group_uniform。此处仅记录诊断，不视为用户授权启动新批次，未开新训练，不修改冻结主表。


### 15.41 完整谱图内部验证 + 四状态主预测头：配对实验协议（2026-10-19）

用户明确授权本地实现、commit/push、pod pull、提交训练及完成提醒，覆盖本地AGENTS旧有“只交命令不启动”规则。本轮在原版3GCN+2GAT、FiLM、输入拼接位置编码、全局节点、理论特征及原1536→256 MLP基础上比较binary control与four_state_mlp，不叠加纯GAT、交叉注意力、位置相加或其他近轮结构；保留历史距离连边公式，不改变PBCLA标签定义或峰过滤。

共同验证修复：内部序列划分仍按原稳定fold seed产生，训练继续使用对应折叠条件组和group_uniform（有效键平均，并非纯序列平均）。验证改为dataset/5fold_soft中该外部训练集合、该内部验证序列的全部原始谱图，保留逐谱pep_mass及hard true_multi。恢复前逐条件核验外部训练seq/key集合、group_n、hard均值与四位soft_multi、组内q及文件SHA；外部测试不参与内部验证。两臂均使用完整验证的padding-free micro F1选严格最大值best_model.pt；min_delta=0.001仅用于patience=25早停。每臂最多300 epochs、batch128、lr1e-4、AdamW wd1e-4、warmup40、cosine300，与原训练预算配置一致；实际停止epoch及优化步数可能因早停而异，不宣称实际步数强制相等。eval batch512、worker4、persistent_workers启用为两臂相同吞吐设置，训练drop_last原样、验证/测试drop_last=False。按臂/fold隔离缓存与落盘split。

四状态标签从已逐谱OR核验的ion12只读SQLite恢复，状态固定[none,b_only,y_only,both]；b/y各OR其1+/2+及无/H2O/NH3六通道。逐谱状态>0须等于历史union_bits；逐条件四状态计数之和须等于n，正状态和须与本fold原训练CSV的true_multi整数计数完全一致。每份sidecar只导出该fold训练白名单；并集缓存不作为模型输入，不导出测试监督。没有对现有峰歧义或电荷通道另行过滤，以避免改变任务。目标保留原折叠q的四位序列化值：p0=1-q，三个正状态按原整数计数在正样本中的比例分配q；原始精确计数及序列化误差一并审计。

四状态头仅将原MLP最后Linear(256,1)替换为Linear(256,4)，q-logit=logsumexp(L1:L3)-L0，所以sigmoid(q-logit)=1-softmax(L)[0]。训练使用有效键均值的soft四类CE，等价于原union BCE加上按q加权的正状态条件CE；不再是独立ion12辅助头，无额外loss权重搜索。初始化在原版随机初始化完成后，令L0=0、L1=L2=L3=z-log(3)，保持初始union q、共享骨干参数及后续随机流一致（浮点误差允许范围由pod核验）；没有迁移已训练权重。新checkpoint有模式/状态顺序标记，跨头加载在参数复制前拒绝。val/test只依赖序列和预设/现有条件输入，不加载四状态目标。

5组重叠序列级随机80/20划分，两臂共10次训练，固定test阈值0.5，沿用有效键mask与谱图平均P/R调和F1。提升筛选仍为平均ΔF1≥0.002、至少4/5 F1严格提高、平均ΔAUC≥0、平均ΔqBrier≤0，全部满足才列待确认候选；不是统计显著性门槛。只与本轮修正验证的control对照，不用历史原版替代同轮对照；两臂实验不能单独估计验证修复的因果收益。冻结论文主表保持原数值，尚无本轮效果结论。

所有数据恢复、QA、CUDA smoke及训练在pod。代码含prepare_four_state_targets.py、test_four_state.py、four_state_ops.py；后者在后台训练终止后复算保存CSV的小报告并输出配对launch.json/terminal.json，接入既有dbond完成提醒。运行中不反复读epoch/状态，不自动追加实验。正式启动需pod的全量标签QA、五划分配对验证SHA、loss/mask/初始化/跨模式加载与CUDA更新核验通过。


启动与pod核验补记：训练源码4da48eeab50decd0a31fbf64f2f179478a6624a7，已push并在pod fast-forward pull。批次checkpoints/graph_transform/four_state_batch/20261019_124645，后台worker PID1195953，选择当时空闲GPU0/1/2/4；GPU3的其他负载未触碰。两臂共10次训练已提交，尚无本轮效果结论。配对事件目录/mnt/pvc/graphtrans/four_state_ops/{launch,terminal}.json；结束后生成result.json、report.md和review_receipt.json，原dbond提醒已从PAUSED恢复ACTIVE，每60分钟仅扫描pending终态。当前对话target_thread_id=01a151ed-0d59-76a3-9634-aec14ee56eb0，未新建重复提醒。

全量标签QA通过：477,669逐谱状态OR完全一致，9,552并集条件组；各fold只导出其训练白名单7,643/7,721/7,649/7,650/7,605组。四类整数计数和=n、正类和=原true_multi计数，folded q最大序列化误差5e-5。250,102个原匹配多通道碰撞峰保留，不改变历史操作性标签。预处理manifest /mnt/pvc/graphtrans/four_state_targets_v1/preparation.json SHA256=ced81fc73f9fe2d2637cf3be549b351beee2b55b3823c426c1ea698323f824de。原SQLite只读SHA复核通过，无外部测试CSV参与标签构建。

五组内部train/val的序列完全隔离，且与各自外部test序列隔离；两臂train.csv/val.csv字节SHA逐划分一致。实际训练条件组依次6,098/6,179/6,140/6,147/6,161；完整验证谱图依次77,992/76,804/76,626/74,085/69,957，每划分82验证序列。原版1,706,153参数，四状态1,706,924参数；各臂真实16组/427有效键、生产AMP单步均完成参数更新，四状态所有输出梯度有限且非零；512谱验证批生产精度路径峰值已分配显存1,021.137MB（仅smoke测量，不作为训练峰值或速度结论）。loss分解、含NaN padding排除、主q一致、目标不参与推理、跨模式加载拒绝且参数未被部分修改均通过。

数值核验限定：两次初期预检分别暴露测试入口logger未初始化和TF32下过严的初始输出比较。测试入口已修；pod诊断显示相同隐藏输入的1输出/4输出GEMM在生产TF32下可有约1.4e-3的logit差异，关闭TF32后约1e-6。最终smoke用highest FP32检验初始union关系，最大logit差8.344650268554688e-7，并恢复原high/TF32设置后运行AMP更新与512谱eval。保留“数学初始化相同、共享参数和RNG相同”的结论，不声称生产浮点输出逐位相等。所有错误在正式提交前解决，标签预处理未重跑。小回执保存F:/codexout/dbond_four_state_launch_20261019/；运行中不读epoch或status。

### 15.42 四状态头完成结果及完整指标导出（2026-10-20）

§15.41批次两臂10次训练全部完成，训练源码4da48eeab50decd0a31fbf64f2f179478a6624a7；完成时间2026-10-19 16:07:41 UTC（北京时间2026-10-20 00:07:41）。原对话heartbeat于00:37收到新完成事件，已在pod独立核验原始test metric与5split CSV、均值/样本SD/配对差/胜率，并ack去重。两臂均为修复后的全谱验证，不与历史最佳值直接配对。

| 指标 | 同轮binary control | 四状态头 | 差值 | 改善划分 |
|---|---:|---:|---:|---:|
| 有效键micro-F1 % | 79.675943 | 80.015219 | +0.339277 pp | 4/5 |
| 有效键accuracy % | 79.817481 | 80.044721 | +0.227240 pp | 3/5 |
| ROC-AUC | 0.880185 | 0.883109 | +0.002924 | 5/5 |
| 谱图P/R调和F1 % | 75.234822 | 75.697745 | +0.462923 pp | 4/5 |
| q-Brier ↓ | 0.068051 | 0.066439 | -0.001612 | 4/5 |
| q_seq | 0.613053 | 0.623920 | +0.010866 | 4/5 |

四项预定工程门槛全部通过，列为待复验候选，不视为统计显著或跨种子确认。逐划分F1差（1222/2252/3514/6072/9075，pp）：+0.220817/-0.071538/+0.540565/+0.391925/+0.614615。谱图precision略降0.117080pp、recall上升1.072865pp；AUC、q误差和序列排序也改善。不能单独量化修复验证的因果效果。参数1,706,153→1,706,924，无成本/速度结论。五组重叠序列划分、每组一次初始化，继续保持探索性表述。

pod报告/mnt/pvc/graphtrans/four_state_ops/report.md，独立核验heartbeat_review_20261020.json。用户要求完整终端文字后，新增report_four_state_metrics.py只读取已完成且SHA绑定的CSV导出全部保存指标、原始比例尺度、alias/定义说明及逐划分F1；完整文本为同目录all_metrics.txt。外部test仅评估历史union任务，未生成四类分类准确率，不能将union accuracy解释成四类accuracy。冻结论文主表原文不改，本地paper_tables末尾同步追加本节探索性结果。

### 15.43 四状态头正状态条件损失权重对照（2026-10-20，预注册）

用户明确授权在四状态版上继续优化并提交pod训练和完成提醒。三臂pre_synthesis_four_state_weight_{control,half,double}.yaml均为原成功四状态结构，α依次1/0.5/2，重训3×5=15次；不复用已训练权重或历史对照。唯一科学干预为loss.four_state_conditional_weight：L=BCE(q_union)+α[-sum_{s>0} t_s log softmax(L_positive)_s]。正状态项已含q权重，不能再重复乘q。α是预先固定的优化超参数，不是物理常数；同时向下/向上测试，不按测试结果决定试验值。

α=1精确保留旧版四类CE计算表达式与默认路径，其他值用稳定union logit BCE加条件CE；q=0/1无除零，目标、头结构、推理q=1-Pnone、参数量1,706,924均不变。三臂共享全量初始化和初始随机流；模型和输入一致。训练保持group_uniform折叠、完整原谱内部验证、严格max-F1选点、原优化器/学习率/早停/预算，实际步数可能因早停不同。验证和测试不使用α或四状态目标计算union指标，不调阈值。

沿用五组重叠序列级80/20划分和去padding指标。两个候选分别与同轮α=1比较，全部报告；每个候选需同时满足平均ΔF1≥0.002、F1≥4/5严格改善、ΔAUC≥0、Δq-Brier≤0才列待确认候选。两个预设候选带来额外筛选选择，结果仍属探索性，不据此改冻结主表或声称显著。当前不预判调低还是调高更好。

计算、验证和训练均在pod；本机仅编辑/同步/读取小报告。预检包含默认与显式α=1旧CE及梯度逐位一致、α=0/0.5/1/2分解和梯度关系、极端logit/q端点/污染padding、五划分train/val SHA与成功批次一致、同初始化/三个生产AMP单步。独立事件目录/mnt/pvc/graphtrans/four_state_weight_ops/使用配对launch.json/terminal.json；完成后核验15个run的实际α、来源及指标汇总，通过原dbond提醒回本对话，不读epoch进度。

启动补记：训练源码fae6c33cf571e0abda5071ace3fde96062f32832已push、pod fast-forward pull，2026-10-20 01:51:38 UTC（北京时间09:51:38）提交批次checkpoints/graph_transform/four_state_weight_batch/20261020_015138，worker PID1262590，选择空闲GPU0/1/2/4。pod语法编译及预检全部通过，5组train/val SHA逐一与原成功四状态批次一致；三臂真实16组/427有效键均完成生产AMP更新，共享完整初始化逐位一致，参数均1,706,924。preflight、launch小回执同步F:/codexout/dbond_four_state_weight_launch_20261020/。原dbond提醒检查时为PAUSED，按用户本轮完成提醒请求恢复同一任务至ACTIVE；保留每60分钟终态扫描，并明确本批control为α=1四状态版、两个候选全报。未新建重复提醒，训练运行中不读epoch或状态。仅ledger启动补记后续提交，pod训练源码保持fae6c33，不在训练中pull。

### 15.44 四状态损失权重：完成结果（2026-10-20）

§15.43批次checkpoints/graph_transform/four_state_weight_batch/20261020_015138，源码fae6c33cf571e0abda5071ace3fde96062f32832，15次训练全部完成（2026-10-20 06:46:31 UTC）。已由06:49 UTC heartbeat主动提醒并ack，本节补齐台账；同轮四状态α=1作唯一配对对照。

| 指标 | α=1 | α=0.5 | Δ / 改善划分 | α=2 | Δ / 改善划分 |
|---|---:|---:|---:|---:|---:|
| 有效键micro-F1 % | 79.963332 | 79.980133 | +0.016801 pp / 4/5 | 79.852187 | -0.111144 pp / 2/5 |
| 有效键accuracy % | 80.029613 | 80.008566 | -0.021046 pp / 3/5 | 79.845290 | -0.184323 pp / 1/5 |
| ROC-AUC | 0.883107 | 0.882872 | -0.000234 / 2/5 | 0.881616 | -0.001491 / 0/5 |
| 谱图P/R调和F1 % | 75.535827 | 75.600931 | +0.065104 pp / 3/5 | 75.382081 | -0.153746 pp / 1/5 |
| q-Brier ↓ | 0.066173 | 0.067000 | +0.000828 / 0/5 | 0.066983 | +0.000810 / 1/5 |
| q_seq | 0.625702 | 0.625323 | -0.000378 / 1/5 | 0.629552 | +0.003850 / 4/5 |

两个候选均未通过全部采用门槛。α=0.5的F1增益很小且所有划分q-Brier变差；α=2的AUC全部下降。保留α=1。pod完成225项独立核验，绑定原metric/汇总CSV、实际运行配置、targets和全谱验证来源；均值/样本SD/配对差/差值SD/胜率及逐划分P/R调和一致。完整报告/mnt/pvc/graphtrans/four_state_weight_ops/report.md，核验heartbeat_review_20261020.json；本地F:/codexout/dbond_four_state_weight_results_20261020/。论文冻结表不替换。

### 15.45 四状态α=1：电荷感知的两侧碎片组成（2026-10-20，预注册）

用户授权继续基于storage选择新方案、实现、push/pod pull、配对训练和完成提醒；明确固定α=1。本轮仅检验一个新输入分支，不叠加交叉注意力、位置改动、预训练或辅助损失。以同轮从头重训四状态α=1为对照，两个臂分别5组重复序列级80/20划分，共10次训练。沿用group_uniform折叠训练、完整原谱内部验证、严格max-F1选点、原优化器/预算/早停、有效键mask和固定0.5测试阈值。训练/验证/测试划分及targets保持原成功批次相同，不用历史最好模型替代对照。

知识依据：storage当前root G:/valhalla/valhalla_storage_share_root_20260830，base/kb:storage，project_work只读。bootstrap的全局自检因缺少无关的workflows/kb/antibody_design_ingest.md而未完成；未修补技能或伪称启动成功。独立检查当前root/KB状态和project_work资格均满足，按该contract执行只读来源核验。实体ent_000024(active)→.registry/machine/entity_resource_map.yaml:243(derived_from，旧表无map_id，未补造)→res_000140(active本地成员、无required/excluded/blacklist、资源active/not_listed)→file_000140(authoritative/present)，公共Library/public_resources/res_000140/authoritative.pdf；pod复核SHA256=814b51192664603b3b764766b436451b6117c041f84ef619f679e70da25a9e9c。实际提取物理p3–4 mobile-proton及PIC讨论：碱性残基组成、前体电荷、肽长影响质子迁移，Arg/Lys不同，理论定性且不能覆盖全部碎裂过程。详细source_records.json与提取证据保存/mnt/pvc/graphtrans/four_state_charge_ops/evidence/；未读source_copies、未写KB。

工程假设：四状态b/y可见性可能受两侧组成与电荷关系影响。当前15维理论特征没有显式累积的两侧碱性组成；本轮新增17维描述：前/后缀R、K/B/O一级胺、H、X吡啶基、D、E计数各6维（除完整肽长）；两侧长度/32；charge/10；(charge−Arg总数)/肽长；(charge−R/K/B/O/H/X侧链基团总数)/肽长。后两个是有符号描述，不叫实际移动质子数，不做正负截断或硬预测规则。B/O/X基团外推、缩放、特征集合和网络接法均为项目假设，综述未验证D肽收益，也未给本实验pKa/气相碱性常数。

新增17→32→SiLU→3小分支，输出[0, Δb_only, Δy_only, Δboth]加到原四状态logits，none作为参考；新增675参数（1,706,924→1,707,599）。不取骨干隐向量，不增加dropout；最后线性权重/偏置零初始化，构建在原初始化之后并隔离RNG，初始函数和共享参数保持原对照。alpha=1四类CE、状态顺序及推理union q不变。Checkpoint显式检查分支键、形状和特征版本，跨模式strict=False也须在拷贝权重前失败。

采用门槛预先固定：平均ΔF1≥0.002、F1至少4/5严格改善、平均ΔAUC≥0、平均Δq-Brier≤0，全部满足才列待确认候选。五组划分重叠、各一次初始化、多轮探索使用同一外部测试集，不能声称确认性显著；冻结论文主表不变。完成报告全报F1/accuracy/AUC/谱图P-R调和F1/q-Brier/q_seq及逐划分差值与胜率。

本机仅代码/配置编辑、git同步及小报告；计算/验证/训练全部pod。训练前检查独立字符计数参考、B/O/X/D/E语义、padding及批次不变性、空键、两臂共享初始化、pod CPU初始eval/train预测及同次AMP前向中零残差/union保持（跨前向CUDA归约差异仅诊断）、跨模式checkpoint安全、生产α=1损失与两步分支梯度，以及5组train/val SHA和序列隔离。独立four_state_charge_ops/{launch,terminal}.json终态事件，训练结束自动分析汇总，不读epoch状态。现有dbond heartbeat已改每120分钟、ACTIVE、仍回原对话，仅扫描pending终态并去重；未建立pod直接唤醒应用的回调，不冒充即时通知。

project_goal：完成有KB依据的新输入方案和配对训练提交。completion_status：代码与证据已准备，pod预检及正式训练尚待完成。target_kb：kb:storage。knowledge_used/source_records：本节及evidence/source_records.json。created_files：charge_partition.py、两份charge配置、test_four_state_charge.py、four_state_charge_ops.py。modified_files：graph_transformer.py、models/utils.py、本台账和本地论文表附注。unresolved_questions：D肽迁移边界、跨种子稳定性、多轮探索偏差。suggested_knowledge_writeback：完整结果可后续另行ingest_engineering；本轮不写回。

启动与预检补记：最终训练源码6e9274405fc27880d6a577df0775a531079fe837已push、pod fast-forward pull。2026-10-20 10:07:10 UTC（北京时间18:07:10）已提交checkpoints/graph_transform/four_state_charge_batch/20261020_100710，worker PID1302440，使用当时空闲GPU0/1/2/4，GPU3其他负载未触碰。两臂共10次训练；当前没有效果结论，不读epoch或运行状态。完成/失败目录/mnt/pvc/graphtrans/four_state_charge_ops/，结果在report.md/result.json，完成后review_receipt.json绑定原始指标和来源。

pod预检全部通过：五组train/val SHA逐一与原成功四状态批次相同，序列与外部test隔离；每臂真实16组/427有效键。对照1,706,924参数，特征臂1,707,599；α均1，生产AMP对照1步、候选2步，输出层和后续隐藏层梯度有限且非零。CPU初始eval/train预测逐位一致，共享参数与构建RNG逐位一致。初次过严的跨前向AMP逐位相等断言未通过；诊断中候选−对照与对照重复前向最大logit差均0.001953125，生产CUDA归约非确定。现以pod CPU检验数学初始函数，同时在候选同次AMP前向捕获原头logits，证实新增残差严格为0、union logits与所捕获原头精确一致；生产确定性/精度设置未改变。不宣称跨前向AMP逐位相等。

预检和启动收据同步F:/codexout/dbond_four_state_charge_launch_20261020/。本轮交付状态：证据核验、实现、git push/pull、pod数值预检和正式训练提交均完成；实验效果及独立完成核验仍待终态。比较检验的是整个新增特征分支，不能仅凭收益分离额外容量、优化参数化和化学描述的贡献。dbond原对话heartbeat为ACTIVE、每120分钟仅扫描pending终态，报告后ack，不新建重复提醒。本机应用运行及SSH可用是提醒条件；当前没有pod直接唤醒应用的即时回调。仅本台账补记另行提交，训练中不pull。

### 15.46 电荷/片段组成分支完成（2026-10-20）

§15.45批次four_state_charge_batch/20261020_100710已于2026-10-20 13:41:03 UTC完成，源码6e9274405fc27880d6a577df0775a531079fe837。原对话13:49 UTC heartbeat在pod对十个原始逐划分metric CSV完成461项独立校验（均值/样本SD/配对差/胜率、来源hash、实际配置及配对train/val SHA），已提醒并ack。对照为同轮四状态α=1，不是历史binary模型。

| 指标 | 同轮四状态对照 | +组成特征 | Δ / 改善划分 |
|---|---:|---:|---:|
| 有效键micro-F1 % | 79.987457 | 80.029839 | +0.042382 pp / 3/5 |
| 有效键accuracy % | 80.002819 | 80.048668 | +0.045849 pp / 4/5 |
| ROC-AUC | 0.882994 | 0.883101 | +0.000107 / 4/5 |
| 谱图P/R调和F1 % | 75.569189 | 75.725351 | +0.156162 pp / 3/5 |
| q-Brier ↓ | 0.066454 | 0.066488 | +0.000034 / 3/5 |
| q_seq | 0.628259 | 0.621483 | -0.006776 / 2/5 |

未通过全部预设门槛，不采纳该分支，后续回到无新增电荷特征的四状态α=1。只对该具体实现作结论，不否定化学信息本身。报告/mnt/pvc/graphtrans/four_state_charge_ops/report.md，独立收据heartbeat_review_20261020.json，本地F:/codexout/dbond_four_state_charge_results_20261020/。冻结论文主表不替换。

### 15.47 四状态CE + union Brier，内部验证选择λ（2026-10-20，预注册）

用户明确授权实现、git push、pod pull训练及完成提醒。保持原四状态头α=1，关闭use_charge_partition_features，不改图、FiLM、输入特征、参数量或推理union。新增逐有效键损失L=CE4(t,P)+λ*(sigmoid(logsumexp(z_positive)-z_none)-(1-t_none))^2；两项共用原mask、group_uniform权重和归一分母。λ=0跳过新增算术，保持旧CE值/梯度路径；参数是有限非负数且非bool。训练外的验证/测试loss和全部指标不加此项。

数学依据：对固定概率标签，CE和概率平方误差均有合理概率估计目标（Gneiting & Raftery 2007, DOI10.1198/016214506000001437）；组合改变有限容量/优化过程中的相对误差惩罚，不提供新观测、不保证收益。当前q仍为同(seq,charge,NCE)重复观测均值，四类比例和4位q不变，不做label smoothing或先验收缩。训练有效键group_uniform与谱图频次加权q-Brier不同，不能承诺两者完全对齐。

预先固定λ候选{0,0.25,1,4}，是几何间隔的优化超参数网格，不是物理常数。每个外部划分独立选择，不跨划分混用标签：每候选从头训练，使用相同fold稳定seed、相同初始化/最大epoch300/AdamW学习率与早停规则；各候选epoch可因早停不同。仍按内部完整原谱验证集strict max micro-F1保存best_model.pt。只读取该点checkpoint.metrics.q_brier，以未舍入q-Brier最小者选λ，精确平局取小λ，允许回退λ0。q_seq与外部测试指标均不参与λ或checkpoint选择。五组重复序列级80/20划分各4次，共20次训练。

本轮不用会自动打开外部test的旧5fold训练驱动；直接生成每候选fold配置，强制test_csv_path=None，在创建dataset前协议校验，并安装Python文件audit屏障，候选训练和选权重阶段禁止打开*.test.fbr.* CSV。checkpoint.extra_state记录实际α/λ、协议、延后测试状态和完整原始/解析配置hash。训练receipt绑定checkpoint/config/内部验证来源；缺候选、非有限指标、异常来源均失败，不静默跳过。全部五组选择落盘selection.json（候选内部指标、选中checkpoint/config SHA、目标和预检来源）后，才创建外部测试数据集。之后只评估同轮λ0与验证集选择的模型；某组选中0时直接复用对照结果，不重复推理。未选候选不读取测试指标。

外部评估均固定0.5、batch512、allow_target_aware=False、去padding；报告F1/accuracy/AUC/谱图P-R调和F1/q-Brier/q_seq、逐划分差值/胜率及选中λ。沿用采用门槛：平均ΔF1>=0.002、F1至少4/5严格改善、平均ΔAUC>=0、平均Δq-Brier<=0，全部满足才列待复验候选。这检验内部验证选参策略，而非预先固定某λ；五组划分重叠、单次初始化、多轮外部测试探索，不宣称统计显著，不替换冻结主表。

全部数值、数据处理、预检和训练均在pod，本机仅编辑/同步/查看小报告。预检覆盖原CE λ0值/梯度逐位等价、独立MSE公式/梯度、q端点/极端logits/污染padding/权重/非法配置、五组旧train/val SHA、禁止外部test访问、内部选参完整性/NaN/精确平局/0回退、相同完整初始化和CPU/CUDA RNG、真实16组427有效键的生产AMP两臂更新。新事件目录/mnt/pvc/graphtrans/four_state_brier_ops/采用配对launch.json/terminal.json；完成后原dbond每120分钟仅扫描终态、核验并提醒原对话，不读epoch进度。当前状态：代码、pod预检和正式训练提交已完成，尚无效果结论（启动回执见下）。

启动与预检补记：训练源码d69127282f8184172ec7e9e1d333ed8b318f96da已push并在pod fast-forward pull。2026-10-20 14:28:37 UTC（北京时间22:28:37）提交批次checkpoints/graph_transform/four_state_brier_batch/20261020_142837，worker PID1324382，使用提交时空闲GPU0/1/2/4。五组划分各λ=0/0.25/1/4共20次训练；对照为同轮λ0四状态α=1，关闭新增电荷特征。现阶段只有提交与预检结果，尚无性能结论。提交后不读取epoch日志或运行状态，不在pod训练中pull。

pod预检全部通过：λ0与原CE值/梯度逐位一致；独立CE+Brier参考与梯度、极端logits、q端点、污染padding、权重归一化、非法参数/协议拒绝均通过；内部选参完整性、NaN拒绝、精确平局取小λ和0回退、禁止外部test访问均通过。五组train/val SHA逐一与原成功四状态批次相同；真实16组427有效键的生产AMP两臂更新均通过，完整模型初始化及CPU/CUDA初始RNG一致，参数量均1,706,924。另以1222内部train/val运行隔离的单epoch λ1集成canary，验证真实checkpoint保存/重载、实际α/λ/配置与来源hash及训练receipt绑定；外部test CSV读取数为0。canary不计入20次正式训练，不用于效果判断，不作为正式模型起点。

提交回执/mnt/pvc/graphtrans/four_state_brier_ops/launch.json；预检smoke.json和integration_canary/passed.json；三份小回执同步F:/codexout/dbond_four_state_brier_launch_20261020/。正式批次先锁定全部五组内部验证选择到selection.json，再评估λ0对照与选中策略；未选λ不做外部测试。终态生成result.json、report.md、review_receipt.json，再写terminal.json供原对话dbond提醒扫描。原提醒已确认ACTIVE、每120分钟，仅扫描pending完成/失败并报告后ack；完成核验明确检查选择锁定与同轮对照，不用历史binary或电荷分支替代。本机应用及SSH可用是提醒条件，不声称已有pod直接唤醒应用的即时回调。
