#!/usr/bin/env python3
"""
一键执行 mini_ghtrans 的最小数据验证流程：
1. 生成小数据
2. 训练
3. 评估
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the mini_ghtrans end-to-end validation flow.")
    parser.add_argument("--config", default="mini_ghtrans/config/mini_debug.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    python = sys.executable
    config_path = Path(args.config)

    run([python, "mini_ghtrans/scripts/create_mini_dataset.py"])
    run([python, "mini_ghtrans/scripts/train_graph_model.py", "--config", str(config_path)])

    checkpoint_candidates = sorted(
        Path("mini_ghtrans/checkpoints").glob("**/best_model.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    if not checkpoint_candidates:
        raise FileNotFoundError("No best_model.pt found under mini_ghtrans/checkpoints")
    checkpoint_path = checkpoint_candidates[-1]

    run(
        [
            python,
            "mini_ghtrans/scripts/evaluate_graph_model.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--out_pred_csv",
            "mini_ghtrans/result/pred/pred.csv",
            "--out_metric_csv",
            "mini_ghtrans/result/metric/metric.csv",
        ]
    )


if __name__ == "__main__":
    main()
