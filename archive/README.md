# archive/ — 归档区（2026-10-14 整理）

本目录存放**已被取代、但保留备查**的图件、文档与脚本。约定了三条规则：

1. **归档 = 冻结**。这里的文件不再维护、不参与任何流水线、不要在论文/报告中引用；
   需要复现历史结果时再按 `archive/README.md` 的说明取用。
2. **归档 ≠ 垃圾**。真正的无效产物（空壳、错口径、跑不通的脚本、运行日志）已在整理时
   直接删除（可从 git 历史 `git log --diff-filter=D -- <path>` 找到）。
3. **这里随仓库走**（与 `graph_transform/config/archive/` 同一惯例），所以归档文件仍在
   git 中、云端可见；只是从活跃目录（`figures/`、`docs/`、代码目录）移出，避免与当前
   论文资产混淆。

## 目录结构

```
archive/
├── figures/                    # 被取代的图件源码与产物（pre-20261006 系列）
│   ├── build_architecture{,_simple,_minimal,_panels,_panels_v3,_v4}.js
│   ├── dbond_gt_architecture*.{pptx,pdf,drawio,svg,png}     # v5 之前的架构图各代
│   ├── build_chaintograph3.js / chaintograph3.{pptx,pdf}    # 未采用的图2重制版
│   ├── model_architecture_schematic.py / model_architecture2.*  # v5 之前的 matplotlib 架构图
│   ├── graph_representation_R20.*, graph-structures-code-consistent.drawio,
│   │   graph_transform_diagram.drawio, model_architecture.drawio,
│   │   graph_representation2.drawio（与在用件字节相同，属重复件）
│   ├── comprehensive_analysis.png / mesure.png / model.png / shili.png  # 原始论文期旧图
│   └── strip_residue_pair_footer.py（一次性美化脚本，路径写死在本机）
├── docs/                       # 旧规划 / 草稿 / 评审响应材料
│   ├── plans/*.md              # 2026-03~04 的审计与规划文档
│   ├── graph_construction.{yaml,tex,drawio,svg}  # 图2早期草图（Jul）
│   ├── r16_/r20_response_materials.tex           # 评审轮响应材料
│   ├── notation_section.tex, r01_feature_inventory.md,
│   │   session_work_report_20260909.md, training_speed_optimization.md
│   └── graph_transform/{figure_analysis,interpretability_analysis}.md  # 旧可解释性框架文档
├── scripts/                    # 被取代/孤立的脚本（能读、能参考，不要再跑）
│   ├── aggregate_dbond_gt_ranking.py, infer_graph_model.py, generate_ranking_figures.py,
│   │   compare_scan_num_rt.py, cross_validation_analysis.py, attention_visualization.py,
│   │   scan_ablation_runs.py, report_dataset_lengths.py,
│   │   audit_and_merge_private_dataset.py, visualize_sample_graph.py
│   └── dbond_m_exp_af_rm_{attn,cat,attn_cat}.py  # 已证伪消融的模型定义（训练/评估脚本已删除）
├── drawio_tmp/                 # 旧图件的 drawio skill 边车文件（spec/arch.json + 生成脚本）
│   ├── graph-structures/       # “与代码一致”图尝试（已被 graph_representation.drawio 取代）
│   └── graph_representation/   # R20 评审轮变体（20261006 决定改用非 R20 旧图）
└── ci/                         # build_docker_image.yml：构建根目录 Dockerfile，而本仓库没有该文件
```

## 取用说明

- 归档脚本按原路径写死了相对路径（`graph_transform/scripts/...`、`import r20_metrics` 等），
  直接运行会因路径变化而失败；要复用请先复制回原目录再跑。
- 被删除文件（`ludbondaf/{train,evaluate}.dbond_m.exp_af_rm_*.py`、`_bench_train.py`、
  `sklearn_baseline/local_run_fold1222.log`、`graph_transform/outputs/graph_viz/*`）可从 git 取回：
  整理提交前它们仍在 `HEAD` 里，直接 `git show HEAD:<path> > <path>`；提交后在历史中查找，例如
  `git log --oneline --diff-filter=D --name-only | grep rm_attn` 定位提交，再用
  `git show <commit>^:<path> > <path>` 还原。