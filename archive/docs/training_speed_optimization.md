# 训练速度优化探索与落地记录

> 创建：2026-08-30（devpod 实测）
> 环境：devpod 4× RTX 4090D 24GB，torch 2.12.0+cu130，AMP on，batch=1024，fold 1222 train 310,910 行
> 基线口径：s/batch = forward + backward + optimizer step（GPU 同步后计时）

---

## 1. 结论速览

| 优化项 | 结果 | 收益 | 状态 |
|---|---|---|---|
| NodeEncoder 序列编码批量向量化 | 0.2413 → 0.1999 s/batch | **-17%** | ✅ 已落地 |
| torch.compile | 6.97 s/batch | -29x（不可用） | ❌ 放弃 |
| DataLoader workers>0 | shm 64MB 硬限制崩溃 | — | ❌ 不可行 |
| 关闭每 epoch force_gc / empty_cache | 配置项 | 小（<2%） | ✅ 已落地 |
| tqdm 每 batch loss.item() 降频 | 代码 | 小 | ✅ 已落地 |
| NaN 检查（_ensure_finite_tensor） | 实测非瓶颈 | ~0 | 保持现状 |
| grad_norm .item() 同步 | 实测非瓶颈 | ~0 | 保持现状 |

**训练时间影响**：epoch 时间约 128s → 约 106s（-17%）；100-epoch 单模型约 3.5h → 约 2.9h。

---

## 2. 瓶颈定位过程（profile 实测）

### 2.1 优化前 forward 分解（batch=1024, E≈533K, N≈28.9K）

```text
node_encode     0.0744s  (48%)  ← 绝对热点
gcn_total       0.0364s  (23%)
gat_total       0.0324s  (21%)
edge_encode     0.0111s  ( 7%)
global_node     0.0004s
forward_total   0.1562s
```

`node_encode` 占近一半，但其中 **0.072s 是 CPU 侧逐序列 Python 循环 + `pad_sequence`**（每个序列单独建 tensor、查表、append，1024 次循环 + 多次内核启动），而非 GPU 计算本身。

### 2.2 优化后 forward 分解

```text
node_encode     0.0018s  ( 2%)  ← -97%
gcn_total       0.0305s  (43%)
gat_total       0.0266s  (37%)
edge_encode     0.0112s  (16%)
forward_total   0.0718s  (减半)
```

### 2.3 backward 侧

forward 73ms / backward+step 127ms（backward 占 batch 总时间 63%）。实测排除两个嫌疑：
- `_ensure_finite_tensor`（每 batch 多次 `isfinite().all()` + `.item()`）：关掉后 0.1993 vs 0.1999，**非瓶颈**；
- grad_norm 每 batch `.item()`：同样**非瓶颈**。

backward 慢的真实原因是 GAT 的 `scatter_reduce_` / `index_add_` 反向传播 + AMP `unscale_/clip/step` 在 **E≈53 万边量级** 上的 kernel 开销——属架构/数据量决定，需改动模型结构或减少边数才能进一步压缩（见第 4 节可选方向）。

---

## 3. 已落地修改

### 3.1 `graph_transform/models/graph_transformer.py` — `_encode_sequences` 批量向量化

- 原实现：逐序列 `np.frombuffer → torch.as_tensor → lookup → append` + `pad_sequence`（1024 次 Python 循环 + 1024 次内核启动）；
- 新实现：单次 `[B, max_len]` byte 缓冲（padding 填 `pad_char`，查表得 padding token 0，与原 `padding_value=0` 语义一致）+ 单次 `ascii_lookup[byte_tensor]` 查表；
- 非法字符检测改为 2D 坐标定位（`bad_row, bad_col`）；
- 收益：node_encode 0.0744 → 0.0018s，batch 0.2413 → 0.1999s（**-17%**）。

### 3.2 `graph_transform/training/trainer.py` — tqdm 更新降频

- `pbar.set_postfix` 内 `loss.item()` 是 GPU 同步点，原每 batch 执行；改为仅 `log_interval`（默认 10）的 batch 更新，数值口径不变。

### 3.3 `graph_transform/config/pre_synthesis_5fold.yaml` — 关闭每 epoch GC/缓存清理

- `force_gc_on_epoch_end / force_gc_on_eval_end / empty_cache_on_epoch_end` 由 true → false（调试项，排查显存时再手动开启）。

---

## 4. 已评估不可行 / 不建议

| 方向 | 结论 | 原因 |
|---|---|---|
| torch.compile | 6.97 s/batch（-29x） | 模型含 `scatter_reduce_`/`index_add_` 等动态 scatter，compile 大量 graph break，退化为逐算子 eager + 编译开销 |
| DataLoader workers>0 | 崩溃 | 容器 /dev/shm 仅 64MB，batch=1024 单 batch 约 50MB（distance 图 E=533K 的 edge 张量），workers×prefetch 必超；`file_system` 共享策略仍走 `shm_open`，无法绕过 |
| 移除 NaN 检查 / grad_norm 同步 | 无收益 | 实测非瓶颈（<0.3%），保留安全网 |

---

## 5. DataLoader worker 并行实测与 shm 限制（2026-08-30 补充）

### 5.1 背景：为什么 workers=0

容器 `/dev/shm` 仅 64MB，PyTorch 多 worker 通过 shm 传 sample。distance 图（maxdist=10）每样本约 40KB 张量，batch=1024 ≈ **40MB**。实测：

| 配置 | fetch+collate | 结果 |
|---|---|---|
| workers=0（当前） | 86ms/iter（GPU 等待） | 稳定 |
| workers=1（短跑 6 iter） | 0.2ms（完全隐藏） | 看似可行 |
| workers=1（长跑 30 iter） | — | **shm 累积崩溃** |
| workers=2/4/8 | — | shm 必崩 |

结论：**workers>0 在 maxdist=10 + batch=1024 下不可用**；根因是 batch 字节（40MB）超过 shm 预算（×prefetch×worker 后必超）。

### 5.2 减边 → 字节下降 → worker 并行可行（数据）

| 图策略 | 边/序列 | batch1024 字节 | shm 下可开 worker（prefetch=2） |
|---|---|---|---|
| distance maxdist=10 | 636（含 global 边） | ~40MB | 0 |
| distance maxdist=6 | 354 | ~22MB | 1~2 |
| sequence（仅相邻键） | 126 | ~8MB | 4+ |

**减边同时带来三重收益**：GPU 计算降（GCN/GAT 与 E 线性相关，maxdist 10→6 约 -44%）、batch 字节降（worker 可行，fetch 隐藏）、显存降。

### 5.3 与 dbond-s/m 的对照（参考问题）

- **dbond-s/m/af 没有 distance 图/边概念**：它们是 transformer 全连接 self-attention（每序列 L×L 注意力对，L≤32），键预测用 `bond_pos` 特征（ludbond config `bond_index_col_name: bond_pos`）+ AF 的相邻位置拼接（`dbond_m_exp_af.py:278`）。不存在 `max_distance` 参数，也没有稀疏边张量。
- 它们的"局部上下文"等价物是：相邻键聚合（1-hop）+ 全连接 attention（全局）。GT 的 distance 图（maxdist=10 的 636 边）在信息上**远超** dbond-s/m 的局部邻接需求——减到 maxdist=6 甚至 sequence graph 更贴近 dbond-s/m 的信息结构。

### 5.4 通用处理方式（for 循环 / GPU 等待数据的解法）

1. **DataLoader 多 worker**（标准方案）：worker 进程并行 `__getitem__` + 主进程预取（prefetch），GPU 计算与取数重叠。受 shm/内存限制。
2. **线程（threading）**：collate 是 CPU-bound + GIL，线程并行收益有限（numpy/torch 部分释放 GIL）；PyTorch 的 DataLoader worker 是**进程**模型，不是线程。
3. **减少 batch 字节**（本仓库的关键杠杆）：减边/降精度 → worker 数可开大 → fetch 完全隐藏（实测 workers=1 时 fetch 0.2ms vs 86ms）。
4. **collate 移入 worker**：PyTorch 默认 collate 在主进程，无法直接并行；需自定义 IterableDataset/预 collate，工程复杂。
5. **本项目推荐实验**：`max_distance: 10 → 6`（或 sequence graph）+ `num_workers: 2` + `prefetch_factor: 1`，同 seed 单折对照精度（F1/PR-AUC/MCC）与速度。

---

## 5.5 max_distance 对照实测（2026-08-30，fold 1222，10 epoch 短跑）

### 边数/字节（seq_len=32，含 global 边）

| max_distance | 保留跳数 | 非global边/序列 | batch1024 字节 | shm 下可开 worker |
|---|---|---|---|---|
| 2（=sequence 拓扑） | 1 跳 | 62 | 3.9MB | 4+ |
| 3 | 1+2 跳 | 122 | 5.8MB | 4+ |
| 6 | 1~5 跳 | 290 | 11.1MB | 2 |
| 10（当前） | 1~11 跳 | 572 | 19.9MB | 0 |

> maxdist=2 的 distance 图与 sequence 图**拓扑一致**（仅相邻边），区别只是边类型编号（distance≤2 用 type=1 vs sequence 用 type=0）；maxdist=3 比 sequence 多一层 2-hop 稀疏连接。
> 注：`_build_distance_edges` 的折叠距离 `seq_dist*(1+0.2*sin(seq_dist*0.5))` 使 maxdist=10 实际保留到 11 跳。

### 速度与精度（md10 基线 vs md3，fold 1222 test，各 10 epoch）

| 指标 | md10 + workers=0 | md3 + workers=2 | 差异 |
|---|---|---|---|
| FetchWait | 78ms | **2ms**（隐藏） | -97% |
| Forward | 130ms | **29ms** | -78% |
| 单 batch | 256ms | **75.5ms** | -70% |
| **epoch 时间** | ~128s | **~45s** | **-65%（2.8x）** |
| valid-bond F1 | 0.7848 | 0.7778 | -0.007 |
| ROC-AUC | 0.8784 | 0.8704 | -0.008 |
| PR-AUC | 0.8577 | 0.8451 | -0.013 |
| MCC | 0.5888 | 0.5762 | -0.013 |
| ECE | 0.0410 | 0.0427 | +0.002 |

**结论**：maxdist=3（+workers=2）用 ~0.007 F1 的代价换 2.8 倍训练速度。10 epoch 短跑口径下精度基本持平；**建议作为主线配置跑完整训练 + md6 对照**验证收敛后精度。maxdist=2 与 sequence 图等价、信息最少，不推荐单独用。

---

## 5.6 新容器实测（2026-08-30：4× RTX 5090 32GB + /dev/shm 32G）

容器重建后：4090D（24G 时间片、shm 64M）→ **5090（32G 物理独占、shm 32G 内存盘）**。shm 限制解除，DataLoader 多 worker 完全可用。

### 5090 + shm32G 基准（batch=1024, fold 1222 train）

| 配置 | fetch | gpu | 单 batch |
|---|---|---|---|
| md10 workers=0 | 86ms（等数据） | 129ms | 215ms |
| **md10 workers=8** | **0.3ms（隐藏）** | 142ms | **142ms** |
| **md3 workers=8** | **0.3ms** | **54.5ms** | **54.7ms** |

**结论**：
1. **shm 32G 解锁后 workers=8 使 fetch 完全隐藏**（86ms→0.3ms），md10 单 batch 215→142ms（-34%）；
2. 5090 本身比 4090D 快 ~35%（md10 gpu 129ms vs ~200ms）；
3. **md3 + workers=8 = 54.7ms/batch** → epoch ≈ 17s+val ≈ **~25s/epoch**，比最初基线（4090D md10 workers0 128s/epoch）**快 5 倍**；
4. shm 已不是约束：md3/md6/md10 均可开 workers=8，**速度收益来源变为纯 GPU 计算量**（边数）。

### 建议的新配置基线（模板）

```yaml
model:
  max_distance: 3          # 从 10 改为 3（-71% 边，精度 10ep 下 -0.007 F1）
data:
  num_workers: 8           # shm 32G 下安全（原 0）
  prefetch_factor: 2
```

> 精度对照（md3 vs md10 的 0.7778 vs 0.7848）来自旧容器 4090D 的 10-epoch 短跑；新容器上建议重跑同口径确认后，再切模板。

---

## 5.7 完整 5 折结果与并行调度（2026-09-01，5090 + md3 + workers=8）

### 5 折完整训练（每折 early stopping 在 11-26 epoch 收敛，100 epoch 上限）

| 指标 | mean ± std | min / max |
|---|---|---|
| **valid-bond F1** | **0.7921 ± 0.0054** | 0.7848 / 0.7991 |
| accuracy | 0.7940 ± 0.0054 | 0.7852 / 0.8021 |
| ROC-AUC | 0.8759 ± 0.0058 | 0.8675 / 0.8853 |
| PR-AUC | 0.8557 ± 0.0098 | 0.8430 / 0.8722 |
| MCC | 0.5901 ± 0.0096 | 0.5747 / 0.6041 |
| Brier / ECE | 0.1461 / 0.0503 | — |
| ex_f1 / lab_f1_mi / subset_acc | 0.6745 / 0.7703 / 0.0343 | — |

- 结果落盘：`result/metric/graph_transform/5fold_md3_summary.csv`；
- **总耗时 32 分钟（4 卡并行）**：4 折并行（~20min）+ 第 5 折（~12min）。early stopping 使实际远快于 100-epoch 上限。

### 并行调度 bug 与修复（cv_root 冲突）

- **问题**：`train_5fold.py` 的 cv_root 用秒级 timestamp；4 折同时启动（同秒）会生成相同 cv_root 并互相覆盖，5fold_metrics.csv 只剩最后一折。
- **修复**：`train_5fold_parallel.py` 为每折生成独立 config（`checkpoint_dir` 加 `5fold_par/<fold_id>` 后缀），cv_root 天然隔离；日志落 `logs/5fold_par_<ts>/fold<id>_gpu<n>.log`。
- 本次 5 折数据已在冲突目录中无损合并（各 fold 的 `fold_<id>/metrics/gt_pre/latest_test_metric.csv` 独立存在）。

---

## 5.8 并行内存 OOM 根因与模型级并行对照（2026-09-01）

### OOM 根因：全图缓存 × 多路并行 × fork COW

容器内存曾超 64GB 被 OOM kill，机制：

1. `cache_full_graphs=true` 把全部样本图张量载入 RAM（实测磁盘：md10 train=5.3G / md6=3.2G / md3=1.9G / seq=1.4G，加 val/test）；
2. `torch.load` 反序列化瞬时 ~2x；
3. **致命项**：`num_workers=8` fork 出的 worker 触碰缓存对象（`__getitem__` 读 `cached_data` 改引用计数）触发 COW 页拷贝，单实验进程族最坏 ~7GB×8 ≈ 56GB；4 路并行叠加破限额。

**修复（正确组合）**：shm=32G 后回到 **edge-cache 模式**（`cache_full_graphs: false` + `workers=8`）——父进程只持 136KB 边结构缓存，edge_attr 由 worker 实时计算。4 路并行实测内存 ~10GB、无 OOM，速度不降（fetch 仍被 worker 隐藏）。

### 并行冲突第二课：checkpoint run_id 秒级碰撞

`train_graph_model` 的 checkpoint 目录用秒级 timestamp，多个 job 同秒启动共享同一 `best_model.pt` 互相覆盖（表现为加载时 distance_embedding shape mismatch）。`run_experiments_parallel.py` 已修复：每 job 的 checkpoint/指标/预测/日志目录按 name 隔离（`exp_par/<name>` 后缀）。

### 模型级并行对照（4 卡各跑一个图配置，fold 1222，valid-bond 口径）

| 模型 | epoch 时间 | F1 | ROC-AUC | PR-AUC | MCC | ECE |
|---|---|---|---|---|---|---|
| seq（仅相邻边，5ep） | 28.5s | 0.7836 | 0.8709 | 0.8425 | 0.5808 | 0.0383 |
| md3 | ~36s | 0.7816 | 0.8702 | 0.8445 | 0.5781 | 0.0397 |
| **md6** | 41.8s | **0.7850** | 0.8740 | 0.8494 | 0.5851 | **0.0334** |
| md10 | 56.7s | 0.7795 | **0.8791** | **0.8584** | **0.5862** | 0.0428 |

（seq 只跑了 5 epoch，为配置遗留；其余 10 epoch）

**解读**：
- F1 四者差距 ≤0.005（≈折间 std），短训练预算下基本等价；md6 综合最优（F1 最高 + 校准最好）；
- md10 的 AUC/PR-AUC 略高（+0.005~0.009）——远距边对**排序能力**有微小贡献，但 epoch 时间贵 36%；
- 结论：**md6 是速度-精度平衡点**；若追求极致 AUC 且不在意时间可用 md10。5 折完整验证应基于 md6 或 md3。

### 并行方案结论（保留两种）

1. **模型级并行**（`run_experiments_parallel.py`）：不同 config/模型各占一卡横向对比，目录按 job 隔离；
2. **折级并行**（`train_5fold_parallel.py`）：每卡一折跑完整 5 折，cv_root 按折隔离，跑完汇聚 mean±std；
3. **DDP（多卡合力跑单模型）不可行**：单卡 batch1024 GPU 已饱和，拆 batch 到多卡不会更快（batch 2048 实测吞吐持平）；workers=8 已是数据加载最优（16/32 反而慢）。

---

## 6. 可选后续方向（未落地，需权衡精度/收益）

1. **减 distance 图边数**：`max_distance=10 → 6`（E 636→354，-44%）或 sequence graph（126 边）——**同时提速 + 解锁 worker 并行**，是收益最大的可选方向，需单折对照精度。
2. **batch=512 + workers=2**：batch 减半 → shm 25MB×2<64MB，但改变收敛行为，需对照。
3. **GAT 中间张量降档**：`edge_attr` 每层 `.to(fp16)` 重复拷贝（E×256），可在 edge_encode 后统一转 fp16 一次；收益中等（约 5-10%），有数值精度风险，需对照。
4. **GCN/GAT 层数/宽度精简**：hidden 256 → 192 等，属精度取舍。
5. **多卡并行**：4×4090D 未利用（当前单卡）。DDP + 每卡独立 fold 最简单；单卡已 2.9h/100ep，多卡收益主要在 5 折并行（每折一卡）。

---

## 7. 复现

```bash
# 基准（devpod）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python graph_transform/scripts/_bench_train.py --mode baseline --iters 12
# profile
CUDA_VISIBLE_DEVICES=0 .venv/bin/python graph_transform/scripts/_bench_train.py --mode profile
# 边数/batch 字节统计
.venv/bin/python /tmp/edge_stat.py   # 或 graph_transform/scripts/_edge_stat.py
```

> 注意：`_bench_train.py` 为基准用临时脚本，非生产入口。
