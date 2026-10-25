#!/usr/bin/env python3
"""从主模型 fold config 快照派生 DBond-GT-pre 权威配置（R-01）。

用法（云端，在 ~/graphtrans/DBond 下）：

    python graph_transform/scripts/make_pre_synthesis_config.py \
        --source checkpoints/graph_transform/5fold/20260421_181316base/fold_1222/config.yaml \
        --output graph_transform/config/pre_synthesis_5fold.yaml

只做三类改动，其余字段（架构、超参、数据路径、seed 等）逐字节保留，
保证 DBond-GT-pre 与主模型的唯一差异就是特征 mask：

1. ablation 段：全部消融开关置 false，仅 pre_synthesis=true、rebuild_cache=true、tag=gt_pre；
2. model 段：state_feature_mask/env_feature_mask 写回全 true 占位
   （实际 mask 由 apply_ablation_config 在运行时强制写入，此处保持源文件原样即可）；
3. 输出目录：checkpoint/metric/pred/log/tensorboard 全部重定向到 *_synthesis 隔离目录。

若 --source 本身带任何消融开关（非 baseline 快照），脚本会拒绝并报错。
"""

from __future__ import annotations

import argparse
import copy
import sys

import yaml

EXCLUSIVE_FLAGS = [
    'use_sequence_graph', 'use_hybrid_graph', 'disable_global_node',
    'gcn_only', 'gat_only', 'no_message_passing', 'no_edge_attr', 'no_state_env',
    'baseline_no_state_env', 'state_charge_only', 'state_mass_intensity_only',
    'env_nce_only', 'env_scan_num_only', 'state_mass_only', 'state_intensity_only',
    'env_rt_only', 'pre_synthesis',
]

DIR_REDIRECTS = [
    ('training', 'checkpoint_dir', 'checkpoints/graph_transform/pre_synthesis'),
    ('evaluation', 'output_pred_dir', 'result/pred/graph_transform/pre_synthesis'),
    ('evaluation', 'output_metric_dir', 'result/metric/graph_transform/pre_synthesis'),
    ('logging', 'log_dir', 'logs/graph_transform/pre_synthesis'),
    ('logging', 'tensorboard_log_dir', 'tensorboard/graph_transform/pre_synthesis'),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive DBond-GT-pre config from main-model fold config snapshot")
    parser.add_argument('--source', required=True, help='主模型 fold config.yaml 快照（如 20260421_181316base/fold_*/config.yaml）')
    parser.add_argument('--output', required=True, help='输出的 pre 配置 yaml 路径')
    args = parser.parse_args()

    with open(args.source, 'r', encoding='utf-8') as f:
        source = yaml.safe_load(f)

    ablation = source.get('ablation', {}) or {}
    active = [flag for flag in EXCLUSIVE_FLAGS if ablation.get(flag, False)]
    if active:
        sys.exit(
            f"[make_pre] 源配置不是 baseline 快照（消融开关已激活: {active}）。\n"
            f"请改用 20260421_181316base 下 fold_*/config.yaml（主模型训练时落盘的原样配置）。"
        )

    derived = copy.deepcopy(source)

    # 1) ablation 段：唯一实验变量
    new_ablation = {flag: False for flag in EXCLUSIVE_FLAGS}
    new_ablation.update({
        'tag': 'gt_pre',
        'base_experiment_name': None,
        'pre_synthesis': True,
        'rebuild_cache': True,
    })
    derived['ablation'] = new_ablation

    # 2) 输出目录重定向（不与主模型结果混写）
    for section, key, value in DIR_REDIRECTS:
        derived.setdefault(section, {})[key] = value

    # 3) 实验标识
    derived.setdefault('experiment', {})['name'] = 'graph_transform_pre_synthesis'
    derived['experiment']['description'] = (
        'DBond-GT-pre (R-01): derived from '
        f'{args.source}; only ablation.pre_synthesis=true (state_mask=[T,T,F], env_mask=[T,F])'
    )

    with open(args.output, 'w', encoding='utf-8') as f:
        yaml.safe_dump(derived, f, sort_keys=False, allow_unicode=True)

    # 报告与源文件的差异键，人工可审
    flat_src, flat_out = {}, {}

    def flatten(d, prefix=''):
        for k, v in d.items():
            key = f'{prefix}.{k}' if prefix else k
            if isinstance(v, dict):
                flatten(v, key)
            else:
                flat_out[key] = v

    flatten(derived)

    def flatten_src(d, prefix=''):
        for k, v in d.items():
            key = f'{prefix}.{k}' if prefix else k
            if isinstance(v, dict):
                flatten_src(v, key)
            else:
                flat_src[key] = v

    flatten_src(source)

    changed = [k for k in flat_out if flat_out.get(k) != flat_src.get(k)]
    added = [k for k in flat_out if k not in flat_src]
    print(f"[make_pre] 源配置: {args.source}")
    print(f"[make_pre] 输出配置: {args.output}")
    print(f"[make_pre] 改动键 ({len(changed)}):")
    for k in sorted(changed):
        print(f"    {k}: {flat_src.get(k)!r} -> {flat_out[k]!r}")
    if added:
        print(f"[make_pre] 新增键 ({len(added)}):")
        for k in sorted(added):
            print(f"    {k} = {flat_out[k]!r}")
    print("[make_pre] 其余字段与源快照完全一致（单变量保证）。")


if __name__ == '__main__':
    main()
