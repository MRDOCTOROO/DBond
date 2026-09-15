#!/usr/bin/env python3
"""
NIST msp 肽谱库 → DBond-GT 外部测试集转换脚本

把 NIST 注释版 consensus msp（如 FTMS_HCD_20_annotated_2019-11-12.msp）转成
与 dataset/5fold/*.csv 完全同 schema 的测试 CSV，可直接喂
evaluate_graph_model.py --test_csv 做域外（MiPD513 之外）泛化评估。

关键设计：
- 标签 100% 复用 PBCLA：直接调用 PBCLA/pbcla.py 的 pbcla()（把 msp 峰列表装进
  它接受的 {params.seq, m/z array, intensity array} 结构），阈值 20 ppm、
  离子类型 b/y ± H2O/NH3、1+/2+、键位映射规则与训练标签生成完全一致，零二次实现。
- pep_mass 按合成前协议用 pyteomics 理论计算（R-01 决策 D2：pre 特征集下
  pep_mass = 理论前体 m/z）；与 msp 实测 Parent 的 ppm 偏差写入报告作 sanity。
- intensity / scan_num / rt 填 0：pre_synthesis 消融下这两路特征被 mask
  （train_graph_model.apply_ablation_config: state=[T,T,F], env=[T,F]），
  数值不影响模型输入；若将来评 full 模型需另议。
- 过滤链（每步计数进报告）：Mods=0（无修饰）→ 字母表 ⊆ 20 标准氨基酸
  （排除 B/O/X/Z——与本项目 D-Dap/D-Orn/吡啶丙氨酸/D-Cha 特殊残基码冲突，
  排除 U/J 等非标准码；C 仅在无修饰时允许，理论质量一致）→ 肽长过滤
  → charge ∈ [2,6]（训练分布）→ (seq,charge) 去重 → 剔除与 MiPD513
  （dataset/5fold/*.csv 全部序列）重复的序列并报告重叠数（同源序列不构成
  "未见序列"）。

用法（项目根目录）：
  python graph_transform/scripts/nist_msp_to_dbond_csv.py \
      --msp /path/FTMS_HCD_20_annotated_2019-11-12.msp \
      --out_csv dataset/nist_external/nist_hcd20_ext_test.csv \
      --report dataset/nist_external/prep_report.txt \
      --nce 20 --workers 8

  小样本冒烟：加 --limit 3000
  条件网格版（探索用，每肽 × 每个 NCE 一行）：--nce_grid 20,30,40,50
"""

import argparse
import importlib.util
import logging
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PBCLA_DIR = os.path.join(REPO_ROOT, "PBCLA")

# 项目 24 字符表 = '#' + ABCDEFGHIKLMNOPQRSTVWXYZ，其中 B/O/X/Z 已被镜像肽特殊残基
# （D-Dap/D-Orn/吡啶丙氨酸/D-Cha）占用；标准肽序列若含 B/O/X/Z/U/J 会被模型误读，
# 一律剔除。C 允许（仅 Mods=0 的无修饰肽进入，理论质量一致）。
ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWY")

CSV_COLUMNS = [
    "name", "seq", "charge", "pep_mass", "intensity", "nce", "scan_num",
    "rt", "fbr", "tb", "fb", "mb", "true_multi",
]

_PARENT_RE = re.compile(r"Parent=([0-9]+\.?[0-9]*)")
_MODS_RE = re.compile(r"Mods=(\d+)")
# 质子质量，与 PBCLA/理论离子特征一致（H+ = 1.007276）
H_PROTON = 1.00727646


def load_mipd513_seqs(fold_dir):
    """从 dataset/5fold/*.csv 收集 MiPD513 全部唯一序列（用于重叠剔除）。"""
    import pandas as pd
    seqs = set()
    if not os.path.isdir(fold_dir):
        print(f"[warn] 折叠目录不存在，跳过 MiPD513 重叠剔除: {fold_dir}")
        return seqs
    for fn in sorted(os.listdir(fold_dir)):
        if fn.endswith(".csv"):
            df = pd.read_csv(os.path.join(fold_dir, fn), usecols=["seq"])
            seqs.update(df["seq"].astype(str).str.strip().unique())
    return seqs


def parse_msp_stream(msp_path, limit=0):
    """流式解析 msp，产出 (seq, charge, parent, mods, mz_list, inten_list)。

    峰行：m/z<TAB>intensity<TAB>"annotation"（annotation 可缺失/不引号，忽略）。
    """
    n_entries = 0
    seq = None
    charge = None
    parent = None
    mods = None
    mz_list, inten_list = [], []

    with open(msp_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("Name:"):
                # 先交出上一条
                if seq is not None:
                    yield (seq, charge, parent, mods, mz_list, inten_list)
                    n_entries += 1
                    if limit and n_entries >= limit:
                        return
                seq, charge, parent, mods = None, None, None, None
                mz_list, inten_list = [], []
                name = line[5:].strip()
                try:
                    peptide, ch = name.rsplit("/", 1)
                    charge = int(ch)
                except ValueError:
                    charge = None
                seq = peptide.strip().upper() if "/" in name else None
            elif line.startswith("Comment:"):
                m = _PARENT_RE.search(line)
                parent = float(m.group(1)) if m else None
                m = _MODS_RE.search(line)
                mods = int(m.group(1)) if m else None
            elif line.startswith("Num peaks"):
                continue
            elif seq is not None and line and line[0].isdigit():
                parts = line.split("\t") if "\t" in line else line.split()
                try:
                    mz_list.append(float(parts[0]))
                    inten_list.append(float(parts[1]))
                except (ValueError, IndexError):
                    continue
        # 文件末尾最后一条
        if seq is not None:
            yield (seq, charge, parent, mods, mz_list, inten_list)


def process_entry(task):
    """worker：PBCLA 打标。输入为已通过全部廉价过滤的条目（pep_mass 已在主进程算好）。"""
    import numpy as np

    seq, charge, pep_mass_th, mz_list, inten_list = task
    # 与 PBCLA 完全一致的打标：构造 pbcla() 接受的最小谱图结构
    order = sorted(range(len(mz_list)), key=lambda i: mz_list[i])
    mz_arr = np.array([mz_list[i] for i in order], dtype="float64")
    inten_arr = np.array([inten_list[i] for i in order], dtype="float64")
    sp = {"params": {"seq": seq}, "m/z array": mz_arr, "intensity array": inten_arr}
    from pbcla import pbcla  # noqa: E402  (PBCLA_DIR 已加入 sys.path)
    tb, fb, mb = pbcla(sp)

    missing = {int(x) for x in mb.split(";") if x}
    true_multi = ";".join("0" if k in missing else "1" for k in range(1, tb + 1))

    return {
        "seq": seq, "charge": charge, "pep_mass": pep_mass_th,
        "tb": tb, "fb": fb, "mb": mb, "true_multi": true_multi,
    }


def worker_init():
    sys.path.insert(0, PBCLA_DIR)
    logging.disable(logging.INFO)  # pbcla() 每序列一条 INFO，批量场景静音


def self_check_labels(csv_path):
    """用项目 label_validation 校验输出 CSV（fail-fast 口径与 GraphDataset 一致）。"""
    spec = importlib.util.spec_from_file_location(
        "label_validation",
        os.path.join(REPO_ROOT, "graph_transform", "data", "label_validation.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import pandas as pd
    df = pd.read_csv(csv_path)
    report = mod.label_error_report(df)
    if len(report):
        report.to_csv(csv_path + ".bad_labels.csv", index=False)
        raise SystemExit(f"[FAIL] {len(report)} 行标签未通过项目校验，"
                         f"明细见 {csv_path}.bad_labels.csv")
    required = ["seq", "charge", "pep_mass", "intensity", "nce", "scan_num", "true_multi"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[FAIL] 缺必需列: {missing}")
    print(f"[self-check] {len(df)} 行全部通过 label_validation 与列 schema 检查")


def main():
    ap = argparse.ArgumentParser(description="NIST msp → DBond-GT 外部测试 CSV")
    ap.add_argument("--msp", required=True)
    ap.add_argument("--out_csv",
                    default="dataset/nist_external/nist_hcd20_ext_test.csv")
    ap.add_argument("--report", default=None,
                    help="默认 <out_csv 同目录>/prep_report.txt")
    ap.add_argument("--nce", type=int, default=20,
                    help="NCE 条件值（默认 20，取自文件名 NIST HCD NCE 桶）")
    ap.add_argument("--nce_grid", default=None,
                    help="如 20,30,40,50：每肽 × 每 NCE 一行（探索用，默认关）")
    ap.add_argument("--min_len", type=int, default=7)
    ap.add_argument("--max_len", type=int, default=30,
                    help="≤ 模型 max_seq_len=32，且与 MiPD513 长度域对齐")
    ap.add_argument("--min_charge", type=int, default=2)
    ap.add_argument("--max_charge", type=int, default=6)
    ap.add_argument("--max_ppm", type=float, default=50.0,
                    help="理论前体 m/z 与 msp Parent 的 |ppm| 上限（剔除同位素错选/"
                         "注释错误条目；默认 50 ppm）")
    ap.add_argument("--drop_c", action="store_true", help="剔除含 Cys 的肽")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（冒烟用）")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--no_self_check", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    from pyteomics import mass as ptmass

    out_csv = os.path.join(REPO_ROOT, args.out_csv) if not os.path.isabs(args.out_csv) else args.out_csv
    report_path = args.report or os.path.join(os.path.dirname(out_csv), "prep_report.txt")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    nce_values = ([int(x) for x in args.nce_grid.split(",")]
                  if args.nce_grid else [args.nce])

    mipd_seqs = load_mipd513_seqs(os.path.join(REPO_ROOT, "dataset", "5fold"))

    stats = Counter()
    ppm_samples, fbr_samples, pos_bonds, tot_bonds = [], [], 0, 0
    len_hist, charge_hist = Counter(), Counter()
    overlap_examples, rows = [], []
    seen = set()

    def cheap_ok(seq, charge, mods, mz_list):
        if mods != 0:
            stats["drop_mods"] += 1
            return False
        if charge is None or not (args.min_charge <= charge <= args.max_charge):
            stats["drop_charge"] += 1
            return False
        n = len(seq)
        if not (args.min_len <= n <= args.max_len):
            stats["drop_len"] += 1
            return False
        if not set(seq) <= ALLOWED_AA:
            stats["drop_alphabet"] += 1
            return False
        if args.drop_c and "C" in seq:
            stats["drop_cys"] += 1
            return False
        if "(" in seq or not mz_list:
            stats["drop_malformed"] += 1
            return False
        return True

    entries = parse_msp_stream(args.msp, limit=args.limit)
    batch = []
    BATCH = 256

    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=worker_init) as pool:
        futures = []

        def drain(futures):
            nonlocal pos_bonds, tot_bonds
            for fut in futures:
                rec = fut.result()
                rows.append(rec)
                fb, tb = rec["fb"], rec["tb"]
                fbr_samples.append(fb / tb if tb else 0.0)
                pos_bonds += fb
                tot_bonds += tb
                len_hist[len(rec["seq"])] += 1
                charge_hist[rec["charge"]] += 1
            futures.clear()

        for seq, charge, parent, mods, mz_list, inten_list in entries:
            stats["entries_seen"] += 1
            if seq is None:
                stats["drop_malformed"] += 1
                continue
            if not cheap_ok(seq, charge, mods, mz_list):
                continue
            key = (seq, charge)
            if key in seen:
                stats["drop_duplicate"] += 1
                continue
            seen.add(key)
            if seq in mipd_seqs:
                stats["drop_mipd513_overlap"] += 1
                if len(overlap_examples) < 20:
                    overlap_examples.append(seq)
                continue
            # pep_mass = 标准理论前体 m/z (M+zH)/z（与 MiPD513 的 PEPMASS[0] 口径一致）。
            # ppm 校验参照须按本库的记录口径：实测发现 z>=3 条目的 Parent 系统性
            # 少记一个质子（= (M+(z-1)H)/z，z=2 正常），故 z>=3 用该口径做参照，
            # pep_mass 本身仍写入正确的 (M+zH)/z。
            m_neutral = float(ptmass.calculate_mass(sequence=seq))
            pep_mass_th = (m_neutral + charge * H_PROTON) / charge
            parent_ref = (m_neutral + (charge - 1) * H_PROTON) / charge if charge >= 3 else pep_mass_th
            ppm = ((parent - parent_ref) / parent_ref * 1e6) if parent else float("nan")
            if parent is None or abs(ppm) > args.max_ppm:
                stats["drop_ppm"] += 1
                continue
            stats["kept"] += 1
            ppm_samples.append(ppm)
            batch.append((seq, charge, pep_mass_th, mz_list, inten_list))
            if len(batch) >= BATCH:
                futures.append(pool.submit(process_entry, batch.pop(0)))
                if len(futures) >= args.workers * 8:
                    drain(futures)
                if stats["kept"] % 20000 < BATCH:
                    print(f"[{time.time()-t0:6.0f}s] kept={stats['kept']} "
                          f"seen={stats['entries_seen']}", flush=True)
        if batch:
            for item in batch:
                futures.append(pool.submit(process_entry, item))
        drain(futures)

    # 写 CSV（每个 NCE 条件一行）
    import pandas as pd
    out_rows = []
    for rec in rows:
        for nce in nce_values:
            out_rows.append({
                "name": rec["seq"],
                "seq": rec["seq"],
                "charge": rec["charge"],
                "pep_mass": rec["pep_mass"],
                "intensity": 0.0,   # pre_synthesis 下被 mask，填 0
                "nce": nce,
                "scan_num": 0,      # 同上
                "rt": 0.0,          # 同上
                "fbr": rec["fb"] / rec["tb"] if rec["tb"] else 0.0,
                "tb": rec["tb"],
                "fb": rec["fb"],
                "mb": rec["mb"],
                "true_multi": rec["true_multi"],
            })
    pd.DataFrame(out_rows, columns=CSV_COLUMNS).to_csv(out_csv, index=False)

    # 报告
    import numpy as np
    ppm_arr = np.array([p for p in ppm_samples if p == p])
    n_seqs = len({r["seq"] for r in rows})
    rep = []
    rep.append("NIST msp → DBond-GT 外部测试集 转换报告")
    rep.append(f"  源文件: {args.msp}")
    rep.append(f"  输出:   {out_csv}  (rows={len(out_rows)}, 唯一(seq,charge)={len(rows)}, "
               f"唯一序列={n_seqs})")
    rep.append(f"  NCE 条件: {nce_values}（外部谱图真实 NCE 未知，取文件名 NCE 桶/网格）")
    rep.append(f"  耗时: {time.time()-t0:.0f}s, workers={args.workers}")
    rep.append("")
    rep.append("过滤漏斗（条目级）:")
    rep.append(f"  msp 条目读入          : {stats['entries_seen']}")
    for k, label in [("drop_mods", " Mods!=0 剔除"), ("drop_charge", " charge 域外剔除"),
                     ("drop_len", " 长度域外剔除"), ("drop_alphabet", " 非标准字母剔除(B/O/X/Z/U/J/修饰符)"),
                     ("drop_cys", " 含C剔除"), ("drop_malformed", " 畸形剔除"),
                     ("drop_duplicate", " (seq,charge) 重复"), ("drop_mipd513_overlap", " 与MiPD513序列重复"),
                     ("drop_ppm", f" 前体m/z偏差>{args.max_ppm}ppm")]:
        if stats[k]:
            rep.append(f"  {label:28s}: {stats[k]}")
    rep.append(f"  保留                   : {stats['kept']}")
    rep.append("")
    rep.append(f"与 MiPD513 重叠样例(≤20): {overlap_examples}")
    rep.append("")
    rep.append(f"键级正标签率(断裂率)   : {pos_bonds/max(tot_bonds,1):.4f}  "
               f"(MiPD513 训练分布 grand rate ≈ 0.48)")
    rep.append(f"肽级 fbr 均值±std      : {np.mean(fbr_samples):.4f} ± {np.std(fbr_samples):.4f}")
    if len(ppm_arr):
        rep.append(f"理论前体 m/z ppm 偏差   : median={np.median(ppm_arr):+.2f} "
                   f"p95(|ppm|)={np.percentile(np.abs(ppm_arr),95):.2f} "
                   f"max(|ppm|)={np.max(np.abs(ppm_arr)):.2f}")
    rep.append(f"电荷分布               : {dict(sorted(charge_hist.items()))}")
    top_len = ", ".join(f"{k}:{v}" for k, v in sorted(len_hist.items()))
    rep.append(f"长度分布               : {top_len}")
    rep.append("")
    rep.append("注意事项:")
    rep.append("  0) 本库 z>=3 条目的 Parent 系统性少记一个质子（=(M+(z-1)H)/z，z=2 正常），")
    rep.append("     ppm 校验已按该口径对齐；CSV 的 pep_mass 始终为标准 (M+zH)/z 理论值。")
    rep.append("  1) 外部序列为 L 型标准肽；模型按非手性 HCD 碎裂近似迁移，论文需声明手性差异。")
    rep.append("  2) 标签来自 NIST consensus 谱（已去噪峰），与 MiPD513 原始谱打标相比")
    rep.append("     随机 20ppm 窗口误匹配更少，正标签率偏高属数据源性质而非实现差异。")
    rep.append("  3) 外部数据无同条件重复测量，q_* 指标不可算，仅 realized 口径可比。")
    rep.append("  4) intensity/scan_num/rt=0：pre_synthesis 消融下被 mask，不影响输入。")
    rep.append("  5) nce 为条件输入取 NCE 桶值（20），在训练网格 {20,30,40,50} 内。")
    report_text = "\n".join(rep)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(report_text)

    if not args.no_self_check:
        self_check_labels(out_csv)


if __name__ == "__main__":
    main()
