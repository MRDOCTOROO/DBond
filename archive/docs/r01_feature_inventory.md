# R-01 · DBond-GT 输入特征清单（Feature Inventory）

建立日期：2026-08-21。**全部结论来自代码逐层溯源，非论文描述。** 行号以当前工作区文件为准。

---

## 1. 数据流总览

```text
MGF 谱图文件
  │  PBCLA/mgf2csv.dbond_m.py（L36-45 逐字段提取）
  ▼
CSV（必需列，graph_dataset.py:89）:
  seq, charge, pep_mass, intensity, nce, <env 第二维: scan_num 或 rt>, true_multi
  │  GraphDataset._extract_sample_features（graph_dataset.py:154-168，原值直读，无变换）
  ▼
sample_features dict
  ├─ state_vars = [charge, pep_mass, intensity]        （graph_dataset.py:113）
  ├─ env_vars   = [nce, secondary_env]                 （graph_dataset.py:114）
  └─ graph_builder.build_graph(seq, sample_features)   → edge_attr（8 维，含 5 个 state/env 维）
  ▼
NodeEncoder.forward（graph_transformer.py:141-204）
  x_i = [ aa_emb(64) ‖ pos_emb(32) ‖ physchem(32) ‖ state_MLP(32) ‖ env_MLP(32) ]
       → Linear(192→256) + LayerNorm + ReLU + Dropout
  ▼
GCN ×3 → GAT ×2 → bond_head（相邻残基对 → 每键 1 个 logit，sigmoid + BCE）
```

## 2. 原始字段 → CSV 列（mgf2csv.dbond_m.py:36-45）

| CSV 列 | MGF 原始字段 | 物理含义 | 级别 | 单位/数值域 | 合成前可获得？ |
|---|---|---|---|---|---|
| `seq` | `params.seq` | D-氨基酸序列（24 字母表，含 B/O/X/Z 特殊残基） | candidate | — | **✓ 序列本身** |
| `charge` | `params.charge[0]` | 前体离子电荷 | precursor | 整数 | **✓ 可预设**（条件） |
| `pep_mass` | `params.pepmass[0]` | 前体 m/z（MGF 规范 PEPMASS[0]=m/z；语义待云端核对，见 §6） | precursor | Da（归一化按 /2000 设计） | **✓ 可由序列+charge 理论计算** |
| `intensity` | `params.pepmass[1]` | **前体离子峰强度（MS1 survey 中前体峰的 intensity），不是碎片峰强度** | precursor/spectrum | counts，log1p 后 /20 | **❌ 合成后采集才知道** |
| `nce` | `params.nce` | 归一化碰撞能量（采集参数） | spectrum | %（整数） | **✓ 可预设**（条件） |
| `scan_num` | `params.scans` | 采集顺序号（编码 LC 梯度位置与批次/顺序的代理） | spectrum | 整数，log1p 后 /20 | **❌** |
| `rt` | `params.rtinseconds` | 保留时间。**注意**：字段名叫 seconds 但数值域 0.7–60.6（见 rt_5fold.yaml 注释），单位疑为分钟，待核 | spectrum | min?（归一化 ×0.01） | **❌** |

> **R-03 的核心疑问在此解决**：`intensity` 的确切定义 = MGF `PEPMASS` 元组第二个元素 = 前体离子强度。论文中"ion-mass distribution and peak-intensity levels"若指碎片峰强度分布，则与实现不符，需改写。

`true_multi` 标签由 PBCLA 从序列理论碎片与实验峰匹配生成（pbcla.py:39-45 调 `utils.py` 的 `mass.fast_mass`，容差 `get_threshold(ppm=20)`）；**pep_mass 列不参与标签生成**。

## 3. 模型注入路径（三条，全部已内建 per-feature mask）

state/env 类协变量在**三处**进入模型，`state_feature_mask=[charge,pep_mass,intensity]`、`env_feature_mask=[nce,second]` 在三处同步生效：

### 路径 A：节点级（NodeEncoder）
- `_encode_state`（graph_transformer.py:235-257）：归一化 `charge×0.1`、`pep_mass/2000`、`log1p(max(intensity,0))/20` → **乘 state_feature_mask**（L256）
- `_encode_environmental`（L259-275）：`nce×0.01`、`scan_num→log1p(x)/20`（或 `rt×0.01`）→ **乘 env_feature_mask**（L274）
- 归一化后经 `state_encoder`/`env_encoder`（Linear3→64→ReLU→Dropout→64→32），expand 到每个残基位置，拼入 x_i

### 路径 B：全局节点（graph_transformer.py:434-443, 510-521）
`global_context = [state_encoder 输出 ‖ env_encoder 输出]`（**已是 mask 后的编码**）→ `global_node_proj`（Linear64→256）→ 虚拟全局节点参与 message passing

### 路径 C：边级（graph_builder.py:335-380）
`edge_attr` 8 维 = `[edge_type, edge_distance, 1/(1+distance), charge×0.1, pep_mass/2000, log1p(intensity)/20, nce×0.01, second_env_norm]`，后 5 维按同一套 mask 置 0（L352-373）

**结论（回答第六步问题）**：intensity / RT / scan 是普通特征向量分量，注入维度固定（3+2），**没有任何架构耦合**——不需要改输入维度或数据处理管道，现有 mask 机制（`apply_ablation_config`，train_graph_model.py:186-290）屏蔽后等价于"该特征不输入模型"。mask 在归一化**之后**乘，故屏蔽位置精确为 0，无残留信息。

## 4. DBond-GT-obs 输入清单（= 当前完整模型，重新定义任务名即可，无需重训）

```text
序列侧（合成前可得）：seq（aa_emb + physchem）、position
条件侧（合成前可预设）：charge、NCE、pep_mass(可理论计算)
实测侧（合成后才有）：  intensity（前体强度）、scan_num 或 rt（env 第二维，二选一）
```
任务定义：**给定一张已采集 MS/MS 谱的实际采集条件（前体电荷、m/z、前体强度、NCE、采集顺序/保留时间），预测各 backbone position 是否获得 fragment-ion support** —— post-acquisition / retrospective, condition-specific prediction。

## 5. DBond-GT-pre 输入清单（待实现，R-01 新增实验）

```text
保留：seq（aa_emb + physchem + position）、charge、NCE、pep_mass(理论计算)
删除：intensity、scan_num/rt（env 第二维整体屏蔽）
实现：state_feature_mask = [True, True, False]
      env_feature_mask   = [True, False]
```
- 注入点不变、维度不变（屏蔽位置恒 0），模型结构零改动。
- **必须 `rebuild_cache: true`**：图缓存 edge_attr 里固化了 5 个 state/env 数值（§3 路径 C），mask 不同则数值不同，复用旧缓存会把实测信息泄漏进 pre 模型。
- charge/NCE 在**打分时**用统一预设网格注入（训练时仍用每条谱的实际 charge/NCE——它们是合法预设条件，不是实测协变量）。

## 6. pep_mass 语义与理论计算（决策 D2）——✅ 已确认 = 前体 m/z（2026-08-22 云端核对）

云端抽样（1222.train）：charge=2 行 pep_mass=1272.15 → 中性质量 ≈ 2×1272.15−2×1.007 ≈ 2542 Da；charge=5 行 625.30 → ≈ 3121 Da。pep_mass 列数值域 379–1953（describe：mean 792.5），即 m/z 量级而非 30 肽中性质量（~2500–3200 Da）量级 → **pep_mass = 前体 m/z（PEPMASS[0]）**。

- PBCLA 生成标签时**不用** pep_mass（理论碎片全部由序列 + pyteomics 计算，utils.py:38-59）。
- pre 版理论计算（score_presthesis.py 已实现）：`m/z_theoretical(seq, c) = pyteomics mass.fast_mass(seq, charge=c)`，B/O/X/Z 残基组成由 PBCLA/utils.py 导入时更新（import 副作用）。
- ppm 偏差核对：`score_presynthesis.py --reference_csv dataset/5fold/1222.train.fbr.shuffle.multi.csv` 会输出理论 m/z vs 观测 pep_mass 的 median/p5/p95/max ppm。

## 7. charge × NCE 预设网格（任务 11）——✅ 已由云端统计锁定（2026-08-21）

云端统计命令（留档复现）：

```bash
python - <<'EOF'
import pandas as pd
df = pd.read_csv("dataset/5fold/1222.train.fbr.shuffle.multi.csv")
print(df["charge"].value_counts().sort_index())
print(df["nce"].value_counts().sort_index())
print(df.groupby(["charge","nce"]).size().sort_values(ascending=False))
EOF
```

统计结果（训练集 1222.train，共 383,295 谱）：

- **charge ∈ {2, 3, 4, 5, 6}**：31,825 / 88,212 / 91,653 / 93,869 / 77,736
- **NCE ∈ {20, 30, 40, 50}**：90,495 / 98,386 / 97,780 / 96,634（粒度为 10，**数据中不存在 25/35 等中间值**）
- 联合分布：5×4 = 20 个组合全部存在，每格 7,479–24,051 条谱，近似均匀

**锁定网格：charge {2,3,4,5,6} × NCE {20,30,40,50}，20 点等权平均。**

理由：
- 完整覆盖训练分布，无任何网格点落在训练支持之外（避免外推）；注意 ChatGPT 示例中的 NCE={20,25,30,35} 含 25/35，本数据集不存在这两个值，会造成外推，**不采用**。
- 分布近似均匀 → 等权（方案 A）与频率加权（方案 B）几乎等价，选等权：最简单、完全可复现、天然满足"所有候选完全相同网格与权重"的 R-02 要求。
- charge=2 谱数最少（31,825）但仍有 3 万+，网格点无支持不足问题。

## 8. ChatGPT 16 任务清单覆盖状态

| # | 任务 | 状态 |
|---|------|------|
| 1-2 | 列出全部输入特征 + 原始字段 | ✅ 本文档 §2 |
| 3-7 | intensity/RT/scan/charge/NCE 实际定义 | ✅ 本文档 §2（rt 单位存疑待核） |
| 8 | 特征注入位置 | ✅ 本文档 §3（三条路径） |
| 9-10 | obs / pre 输入清单 | ✅ 本文档 §4 §5 |
| 11 | charge×NCE 网格 | ✅ 已锁定：{2,3,4,5,6}×{20,30,40,50} 等权 20 点（§7） |
| 12-16 | 改代码 → 训练 → 无谱推理测试 → 唯一性验证 → Table 6 | 后续阶段（revision_todo.md R-01 1.1-1.7） |

## 9. 遗留核对项

1. ~~pep_mass 列语义~~ ✅ 已确认 = 前体 m/z（§6，2026-08-22）。
2. ~~rt 单位~~ ✅ 已定案 = **分钟**（2026-08-23 云端核实：0.70–60.58、均值 30.2、四分位 15.9/44.2，均匀覆盖 ~60 min LC 梯度；MGF 字段名 rtinseconds 系误导性命名，data dictionary 按分钟写）。
3. ~~env 第二维最终采用~~ ✅ 已定案 scan_num（2026-08-23：20260421_181316base/fold_1222/config.yaml:69）。pre 版中两者均被屏蔽。
4. ~~name 列有无 run 标识~~ ✅ 已定案：name = 肽编号（CO-/YP-/CS- 前缀），为**数据集/样本类型标识**（用户确认：不同类型肽，序列特征可区分），非采集/合成批次 → 数据无 run/batch 标识，leave-run-out/within-run permutation 不可行（limitation 路线）。intensity 零缺失、零零值（2026-08-23 核实）。
