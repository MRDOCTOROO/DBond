#!/usr/bin/env python3
"""
Export a sample sequence graph (edge_index) to CSV and DOT for visualization.
"""

import argparse
import os
import random
import sys
from typing import Dict, Any, Tuple

import pandas as pd
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.graph_builder import SequenceGraphBuilder
from models.utils import ModelConfig


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_sample_row(
    data: pd.DataFrame,
    max_seq_len: int,
    index: int,
    seed: int
) -> Tuple[int, pd.Series]:
    filtered = data[data["seq"].str.len() <= max_seq_len].reset_index(drop=True)
    if len(filtered) == 0:
        raise ValueError("No sequences within max_seq_len after filtering.")

    if index is None:
        random.seed(seed)
        index = random.randrange(len(filtered))
    elif index < 0 or index >= len(filtered):
        raise IndexError(f"Index out of range: {index} (0..{len(filtered) - 1})")

    return index, filtered.iloc[index]


def build_graph_for_row(
    row: pd.Series,
    model_config: ModelConfig,
    strategy: str
) -> Dict[str, Any]:
    sequence = str(row["seq"])
    env_vars = {
        "charge": float(row.get("charge", 0.0)),
        "pep_mass": float(row.get("pep_mass", 0.0)),
        "nce": float(row.get("nce", 0.0)),
        "rt": float(row.get("rt", 0.0)),
        "fbr": float(row.get("fbr", 0.0)),
    }

    graph_builder = SequenceGraphBuilder(model_config)
    graph_data = graph_builder.build_graph(sequence, env_vars, strategy)

    return {
        "sequence": sequence,
        "edge_index": graph_data["edge_index"],
        "edge_types": graph_data["edge_types"],
        "edge_distances": graph_data["edge_distances"],
    }


def export_edge_list_csv(
    output_dir: str,
    sample_id: int,
    edge_index,
    edge_types,
    edge_distances
) -> str:
    csv_path = os.path.join(output_dir, f"sample_{sample_id}_edges.csv")
    rows = []
    for i in range(edge_index.size(1)):
        src = int(edge_index[0, i])
        dst = int(edge_index[1, i])
        edge_type = int(edge_types[i])
        edge_dist = int(edge_distances[i])
        rows.append((src, dst, edge_type, edge_dist))

    df = pd.DataFrame(rows, columns=["src", "dst", "edge_type", "edge_distance"])
    df.to_csv(csv_path, index=False)
    return csv_path


def export_dot(
    output_dir: str,
    sample_id: int,
    sequence: str,
    edge_index,
    edge_types,
    edge_distances
) -> str:
    dot_path = os.path.join(output_dir, f"sample_{sample_id}.dot")
    num_nodes = len(sequence)

    # Build undirected edge set (collapse symmetric edges)
    edge_set = set()
    for i in range(edge_index.size(1)):
        src = int(edge_index[0, i])
        dst = int(edge_index[1, i])
        edge_type = int(edge_types[i])
        edge_dist = int(edge_distances[i])
        a, b = (src, dst) if src <= dst else (dst, src)
        edge_set.add((a, b, edge_type, edge_dist))

    with open(dot_path, "w", encoding="utf-8") as f:
        f.write("graph G {\n")
        f.write('  node [shape=circle, fontsize=10];\n')

        for i in range(num_nodes):
            aa = sequence[i]
            f.write(f'  {i} [label="{i}:{aa}"];\n')

        for a, b, edge_type, edge_dist in sorted(edge_set):
            f.write(f'  {a} -- {b} [label="t{edge_type} d{edge_dist}"];\n')

        f.write("}\n")

    return dot_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a sample graph to CSV and DOT."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="graph_transform/config/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Override CSV path (default: config data.train_csv_path)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["sequence", "distance", "hybrid"],
        help="Override graph strategy",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Row index after filtering by max_seq_len",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="graph_transform/outputs/graph_viz",
        help="Output directory",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]
    model_config = ModelConfig(config["model"])

    csv_path = args.csv or data_config["train_csv_path"]
    max_seq_len = data_config.get("max_seq_len", 100)
    strategy = args.strategy or data_config.get("graph_strategy", "distance")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    data = pd.read_csv(csv_path)
    sample_id, row = select_sample_row(data, max_seq_len, args.index, args.seed)
    graph_info = build_graph_for_row(row, model_config, strategy)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_out = export_edge_list_csv(
        args.output_dir,
        sample_id,
        graph_info["edge_index"],
        graph_info["edge_types"],
        graph_info["edge_distances"],
    )
    dot_out = export_dot(
        args.output_dir,
        sample_id,
        graph_info["sequence"],
        graph_info["edge_index"],
        graph_info["edge_types"],
        graph_info["edge_distances"],
    )

    print(f"Sample index: {sample_id}")
    print(f"Sequence length: {len(graph_info['sequence'])}")
    print(f"Graph strategy: {strategy}")
    print(f"Edge list CSV: {csv_out}")
    print(f"DOT file: {dot_out}")
    print("Render example (Graphviz): dot -Tpng sample_x.dot -o sample_x.png")


if __name__ == "__main__":
    main()
