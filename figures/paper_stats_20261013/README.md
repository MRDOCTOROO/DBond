# paper_stats_20261013 — 审稿证据包（三项待核验证据 + panel_c 重生成）

本目录保存 2026-10-13 图表审计/投稿检查中"证据待核验"三项的**计算脚本与结果**，
以及 SI 图 S1(c) 的重生成记录。所有脚本只读取已存数据，不重跑模型训练/推理
（panel_c 重生成仅做 500 样本的注意力前向提取，X 版脚本见下）。

## 1. 主图4：274 对 / r=0.839 / MAE=5.0 pp
- 脚本：`fig4_recompute.py`；结果：`fig4_recompute_results.json`
- 数据：`../residue_pair_film_fold9075/residue_pair_stats.csv`（fold 9075、film 臂测试集，
  `residue_pair_summary.json` 记录 total_samples=100,539，与 SI 折表 9075 test 一致）
- 复算（与绘图代码同口径，N≥50 的 274 个残基对）：**r = 0.8387、MAE = 5.03 pp**；
  576 格 = 274(N≥50) + 302(N=0)，无 0<N<50 格；|偏差|≤10 pp 占 87.6%。

## 2. 五组划分的解释性汇总
- 脚本：`interp_fivefold.py`；结果：`interp_fivefold_results.json`
- 数据：`fold_{1222,2252,3514,6072,9075}/occlusion_summary.json`
  （pod `result/interp_film/fold_*/occlusion/` 的镜像）
- 复算：全局 Pearson **0.283±0.054**、全局 Spearman **0.355±0.025**、
  逐样本 r 的划分级均值 **0.291±0.056**；强一致性（r≥0.5）**0/0/0/0/1 = 1/750**
  —— 与 SI S9 及正文数值逐位一致。

## 3. ppm 统计与序列聚类 CI
- `ppm_check.py` + `ppm_results.txt`：MiPD513 数据集 pep_mass（观测前体 m/z）与
  理论 $(M+zH)/z$ 的偏差：2,427 个 (序列,电荷) 组合，中位 +0.38 ppm、p5–p95
  [−2.3, +5.2]、99% 以上在 ±10 ppm 内（18 组 >10 ppm，集中于 z≥5 的个别记录）。
- `cluster_ci.py` + `cluster_ci_results.json`：遮掩一致性逐样本 r 的**序列整簇
  bootstrap**（80 条唯一序列为整簇）；总区间 [0.353,0.378]（样本级 [0.356,0.375]）；
  Charge 2 与 4–5 不重叠（聚类后仍成立）；低/中 FBR 样本级不重叠、聚类后边缘重叠
  （[0.322,0.363] vs [0.362,0.401]）。以上为 SI S9 相关表述的数字来源。

## panel_c 重生成（SI 图 S1(c)，2026-10-13）
- 脚本：`regen_panel_c_9075.py`（pod 执行；scratch 内需带同版本 `visualization.py`）；
  结果：`panel_c_effect.json`
- 变更：图内不再标注 p 值；Cohen's d 的 95% CI 改为序列整簇 bootstrap；
  标题第二行取收窄后的 "Attention = functional saliency"。
- 一致性核对：与原始 500 样本完全对齐（n_broken=6176、n_intact=6835、d=−0.243、
  AUC=0.438 逐位相同）；新 CI [−0.294, −0.192]（原逐键口径 [−0.277, −0.208]）。