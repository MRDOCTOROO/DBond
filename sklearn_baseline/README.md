# sklearn_baseline — 传统机器学习基线（RF / XGBoost / HistGB / LogReg）

**动机**：MiPD513 语料只有 513 条唯一镜像肽序列（47.7 万谱图行，~1000 张/序列），
93% 的标签方差在序列内部（ICC 0.072）。小样本、低序列多样性场景下，传统树模型
可能不逊于深度模型——本目录在**与最新 film 臂完全相同的特征、数据折与评测口径**
下直接检验该假设。

## 特征与协议对齐（parity 口径）

| 项 | 对齐对象 | 说明 |
|---|---|---|
| 逐键特征（15 维） | `graph_transform/data/theory_features.py::compute_bond_theory` | 纯序列理论碎片特征，与 film 臂 `use_theory_features: true` 同源 |
| 条件特征 | charge / pep_mass / nce | film 臂 pre-synthesis mask 保留项；intensity/scan_num/rt 属合成后信息，排除 |
| 训练数据 | `dataset/5fold/{fold}.train.fbr.shuffle.multi.csv` | 5 个 sequence-level fold（1222/2252/3514/6072/9075），train/test 序列不相交 |
| 标签 | `true_multi` 展开的逐键 0/1 | 每键一行（dbond_s 行口径），一折约 1000 万键行 |
| val 划分 | train 行级随机 20%，seed = 42 + fold_index | 协议同 ludbond `_5fold_common`（RNG 实现不同：sklearn vs torch） |
| 指标 | `graph_transform/training/metrics.py::BinaryBondMetrics` | f1_micro(≡lab_f1_mi)@0.5 + accuracy/hamming + q_* 全套，GT/film 同一实现 |
| q 真值 | 测试折内 (seq,charge,nce) 组内逐键均值 | 逐行复刻 `precompute_soft_labels.compute_group_soft`（含 %.4f 取整） |
| pred 长表 | dbond_s 变长格式 | `fold_*/pred/test.pred.csv` 可直接喂 `q_metrics_from_pred_csv.py` |

## 文件

- `features.py`：折 CSV → 键级特征矩阵（理论特征按唯一序列缓存，513 条只需算一次）
- `run_tree_baselines.py`：主入口，5 折训练 + 评测 + 汇总（mean±std, ddof=0，对齐 `aggregate_5fold`）
- `result/cv/{model}/{ts}/fold_{id}/{pred,metric}/ + 5fold_{metrics,summary}.csv`

## 模型

| 名称 | 实现 | 备注 |
|---|---|---|
| `rf` | RandomForestClassifier(n_estimators=300, n_jobs=-1) | |
| `histgb` | HistGradientBoostingClassifier(max_iter=500, lr=0.1) | sklearn 内置 LightGBM 风格，零额外依赖 |
| `xgb` | XGBClassifier(hist, lr=0.05, depth=6, early_stop 50 on val logloss) | 需 `xgboost>=2.0`（已进 pyproject/uv.lock，xgboost 3.4.1） |
| `logreg` | StandardScaler + LogisticRegression | 线性对照 |

## 环境配置（pod：gsj-5090.devpod）

```bash
cd /mnt/pvc/graphtrans/DBond
git pull                       # 拉取 sklearn_baseline/ + pyproject + uv.lock
source .venv/bin/activate
uv pip install xgboost         # 从 lock 安装 xgboost（勿用 uv sync——会剪掉未声明的 torch 等）
```

本机（仓库 `.venv`，Windows）：sklearn/pandas/numpy/torch-cpu 已齐，可直接跑除 xgb 外的模型。

## 运行

```bash
# pod 正式：全部模型 × 5 折（xgb 上 GPU）
python sklearn_baseline/run_tree_baselines.py --models rf,histgb,xgb,logreg --folds all --xgb_device cuda

# 本机冒烟（截断行数 + 小 RF）
python sklearn_baseline/run_tree_baselines.py --models rf,histgb,logreg --folds 1222 \
    --limit_rows 20000 --rf_n_estimators 100

# 常用参数
#   --folds 1222,2252        子集折（调试）
#   --force_new              已有 test_metric.csv 也重跑（默认断点续跑）
#   --no-save-pred           不落 pred 长表（每折~95MB/模型）
#   --rf_max_samples 0.3     RF 每树 bootstrap 降载（~16GB 本机全量 10M 键行会 OOM；pod 大内存可省略）
```

耗时参考：特征构建 ~10s/折；RF(100树) 全折 CPU（6 核）约 1 小时；HistGB ~10-20 分/折；
LR 分钟级。pod 核多 + xgb cuda 会显著更快。

## 输出与交叉校验

每折每模型产出：
- `fold_{id}/pred/test.pred.csv`：`evaluation_id,threshold,bond_index,true,pred,pred_prob`
  （dbond_s 变长长表，行序 = 测试 CSV 行序）
- `fold_{id}/metric/test_metric.csv`：metric,value 两列（compute() 全量键）
- `fold_{id}/metric/feature_importance.csv`（RF/XGB 有）

**pod 端全链路对拍**（本机因缺 torch_geometric 无法跑 canonical `precompute_soft_labels`，
runner 内的 q 是其逐行复刻，建议在 pod 做一次对拍闭环）：

```bash
python graph_transform/scripts/precompute_soft_labels.py --fold_dir dataset/5fold --out_fold_dir dataset/5fold_soft
python graph_transform/scripts/q_metrics_from_pred_csv.py \
    --pred_csv sklearn_baseline/result/cv/xgb/<ts>/fold_1222/pred/test.pred.csv \
    --test_csv dataset/5fold_soft/1222.test.fbr.multi.csv \
    --ref_metric_csv sklearn_baseline/result/cv/xgb/<ts>/fold_1222/metric/test_metric.csv \
    --out_csv sklearn_baseline/result/cv/xgb/<ts>/fold_1222/metric/q_reeval_check.csv
# 期望：[f1 校验:OK] Δ<1e-6（重建 f1_micro == 参考 f1_micro）
```

## 参考对照（docs/experiment_log.md §15，5 折 mean±std，%为×100）

| 模型 | lab_F1 | bondacc | q_brier | ρ_pep/cond | q_seq | Top10_seq |
|---|---|---|---|---|---|---|
| film 臂（GT, F1 选点） | 79.80±0.64 | 80.02±0.36 | 6.69 | 89.67/89.01 | 62.52±5.93 | 1.2263 |
| GT 5-seed 集成 | 80.36±0.47 | 80.38 | 6.28 | — | 63.84 | — |
| dbond_s（padding-free 重算） | — | 78.30 | — | — | 65.06 | — |
| dbond_af / af_opt（重算） | — | 79.86 / 79.93 | — | — | — | — |
| histgb（本机 fold1222 预跑） | 78.74 | — | 7.13 | 85.60(cond) | 38.27 | 1.246 |
| rf（本机预跑，100树+0.3采样降载） | 77.71 | — | 7.90 | 85.47(cond) | 35.57 | 1.221 |
| logreg（本机预跑） | 68.28 | — | 12.74 | 80.85(cond) | 14.95 | 0.934 |
| （同折锚点·主线 GT hard theory，§1） | 79.08 | — | — | — | — | — |
| （同折锚点·1D 基线 dbond_s） | 78.16 | — | — | — | — | — |
| rf/histgb/xgb 5 折正式 | 待 pod 跑（本目录 result/cv/） | | | | | |

判读口径提醒（§15.8）：lab_F1 跨模型可比；q_seq 家族间结论不宜直接外推跨模型。
本机预跑判读：histgb 同折低于主线 GT 约 0.3~0.8 点、高于 1D 基线 dbond_s；
q_seq（序列排序）远低于 GT 家族 61-63——主线 GT 在排序/筛选口径不可替代。

## 已知边界

- XGB 迭代数由 **val logloss** early stopping 决定（非 val F1）：自定义 F1 eval 的
  入参形式（margin/概率）在 xgboost 版本间有歧义，不冒险；val F1 仍照常记录。
- q 真值为 runner 内联复刻（12 行 groupby 逻辑），pod 端可用上面的对拍命令闭环验证。
- 本轮只跑 parity 特征；`features.py` 预留 `extended`（两侧氨基酸 one-hot 等）供后续消融。
- RF/HistGB/LR 无 early stopping，固定超参直接拟 80% 训练子集（与 ludbond 训练量一致）。
