#!/usr/bin/env python3
"""特殊残基 B/O/X/Z 数据统计审计脚本（P0-3 固化）。

输入：fold CSV 目录（或单个 CSV）
输出：stdout 报告 + 可选 CSV：
  1. 每个特殊残基（B/O/X/Z）出现次数 / 出现行数 / 唯一序列数；
  2. 每残基的键位置分布（键位置 = 该残基在序列中的位置 i，参与 bond i-1 与 i）；
  3. 含特殊残基的序列比例。

用法（devpod，项目根目录）：
    .venv/bin/python graph_transform/scripts/report_special_residues.py \
        --fold_dir dataset/5fold \
        --output result/metric/special_residues.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import pandas as pd

SPECIAL = set("BOXZ")


def scan_csv(path: str) -> Tuple[int, Counter, Counter, int, Counter]:
    """返回 (n_rows, 残基次数, 残基唯一序列数, 含特殊残基的序列数, 位置分布)。

    位置分布: {aa: Counter(bond_position_index)}，bond 位置从 1 开始
    （残基 i 与 i+1 之间的键为 bond i，i 从 1 到 len-1）。
    """
    total_aa = Counter()
    unique_seqs = Counter()  # aa -> set size（用 Counter 计数唯一序列）
    pos_dist: Dict[str, Counter] = defaultdict(Counter)
    special_seq_count = 0
    n_rows = 0
    seen_seqs: Dict[str, set] = defaultdict(set)

    for chunk in pd.read_csv(path, usecols=["seq"], chunksize=200000):
        n_rows += len(chunk)
        for seq in chunk["seq"].astype(str):
            seq = seq.strip()
            if not seq:
                continue
            for i, c in enumerate(seq):
                if c in SPECIAL:
                    total_aa[c] += 1
                    seen_seqs[c].add(seq)
                    # 键位置：该残基作为 src 参与 bond(i+1)，作为 dst 参与 bond(i)
                    if i > 0:
                        pos_dist[c][i] += 1  # bond i (1-based)
                    if i < len(seq) - 1:
                        pos_dist[c][i + 1] += 1  # bond i+1 (1-based)
            if any(c in SPECIAL for c in seq):
                special_seq_count += 1

    for c in SPECIAL:
        unique_seqs[c] = len(seen_seqs.get(c, set()))
    return n_rows, total_aa, unique_seqs, special_seq_count, pos_dist


def main() -> None:
    parser = argparse.ArgumentParser(description="Report B/O/X/Z special residue statistics")
    parser.add_argument("--fold_dir", default="dataset/5fold", help="fold CSV 目录（*.csv）")
    parser.add_argument("--csv", default=None, help="单个 CSV（优先于 --fold_dir）")
    parser.add_argument("--output", default=None, help="可选：明细 CSV 输出路径")
    args = parser.parse_args()

    if args.csv:
        paths = [args.csv]
    else:
        paths = sorted(glob.glob(os.path.join(args.fold_dir, "*.csv")))
    if not paths:
        print(f"ERROR: no CSV found ({args.fold_dir or args.csv})", file=sys.stderr)
        sys.exit(1)

    total_rows = 0
    grand_aa = Counter()
    grand_unique = Counter()
    grand_special_seqs = 0
    grand_pos: Dict[str, Counter] = defaultdict(Counter)

    print("=== 特殊残基 B/O/X/Z 统计 ===")
    for p in paths:
        n_rows, aa, uniq, special_seqs, pos = scan_csv(p)
        total_rows += n_rows
        grand_aa.update(aa)
        for c in SPECIAL:
            grand_unique[c] += uniq[c]
        grand_special_seqs += special_seqs
        for c, counter in pos.items():
            grand_pos[c].update(counter)
        print(f"{os.path.basename(p)}: rows={n_rows}, "
              f"B={aa['B']}, O={aa['O']}, X={aa['X']}, Z={aa['Z']}, "
              f"special_seqs={special_seqs}")

    print(f"\n合计 rows={total_rows}")
    print(f"{'aa':>3} {'出现次数':>12} {'唯一序列数':>10}")
    for c in "BOXZ":
        print(f"{c:>3} {grand_aa[c]:>12} {grand_unique[c]:>10}")
    print(f"含特殊残基的序列数={grand_special_seqs}")

    print("\n键位置分布（bond 位置 1..len-1 内的出现次数，仅特殊残基参与键）:")
    for c in "BOXZ":
        if not grand_pos[c]:
            continue
        top = grand_pos[c].most_common(8)
        desc = ", ".join(f"bond{i}:{n}" for i, n in top)
        print(f"  {c}: {desc}")

    if args.output:
        rows = []
        for c in "BOXZ":
            for pos, n in sorted(grand_pos[c].items()):
                rows.append({"aa": c, "bond_position": pos, "count": n})
        out = pd.DataFrame(rows, columns=["aa", "bond_position", "count"])
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        out.to_csv(args.output, index=False)
        print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
