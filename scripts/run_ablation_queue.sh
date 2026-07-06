#!/usr/bin/env bash
# 消融实验顺序编排脚本（单卡环境）
#
# 用途：在另一张显卡上顺序跑多个消融的五折实验，不污染 default.yaml。
# 工作原理：读 default.yaml 拿基础参数，在内存里强制重置整个 ablation 段，
#           只开目标开关，生成临时 config 跑完后删除。
#
# 用法：
#   bash scripts/run_ablation_queue.sh
#
# 可调环境变量（在脚本顶部修改）：
#   DEVICE_ID=1          # 使用的显卡编号（CUDA_VISIBLE_DEVICES）
#   ABLATIONS="no_edge_attr sequence_graph"  # 要跑的消融开关名
#
# 注意：
#   - 每个消融会跑完整 5 折后才进入下一个
#   - 输出目录由 ablation tag 自动区分（checkpoints/.../no_edge_attr/、.../sequence_graph/）
#   - 跑前会清理对应图策略的旧缓存（sequence/hybrid 的边缓存必须重建）
#   - 临时 config 文件存到 /tmp，跑完自动删除

set -euo pipefail

# ============== 可调配置 ==============
DEVICE_ID="${DEVICE_ID:-1}"                                    # 显卡编号，按你的环境改
ABLATIONS="${ABLATIONS:-no_edge_attr sequence_graph}"          # 要跑的消融开关，空格分隔
CONFIG_SRC="graph_transform/config/default.yaml"               # 基线配置来源
CACHE_DIR="cache/graph_data"                                   # 图缓存目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=========================================="
echo " 消融实验编排"
echo " 显卡: CUDA_VISIBLE_DEVICES=${DEVICE_ID}"
echo " 消融队列: ${ABLATIONS}"
echo " 项目根目录: ${PROJECT_ROOT}"
echo "=========================================="

# 检查基线配置存在
if [[ ! -f "${CONFIG_SRC}" ]]; then
    echo "[ERROR] 基线配置 ${CONFIG_SRC} 不存在"
    exit 1
fi

# 每个 ablation 对应的图缓存清理规则
# 只有改变图结构的消融（sequence/hybrid）才需要清缓存
clear_graph_cache() {
    local ablation="$1"
    case "${ablation}" in
        sequence_graph|use_sequence_graph)
            echo "  [cache] 清理 sequence 图缓存"
            rm -f "${CACHE_DIR}"/edges_sequence_*.pt 2>/dev/null || true
            rm -f "${CACHE_DIR}"/*_sequence_*_cache.pt 2>/dev/null || true
            ;;
        hybrid_graph|use_hybrid_graph)
            echo "  [cache] 清理 hybrid 图缓存"
            rm -f "${CACHE_DIR}"/edges_hybrid_*.pt 2>/dev/null || true
            rm -f "${CACHE_DIR}"/*_hybrid_*_cache.pt 2>/dev/null || true
            ;;
        no_edge_attr|no_state_env|no_message_passing|gcn_only|gat_only|disable_global_node)
            echo "  [cache] 无需清理（不改变图拓扑）"
            ;;
        *)
            echo "  [cache] 未知消融 ${ablation}，不清理缓存"
            ;;
    esac
}

# 生成临时配置：读 default.yaml，强制重置 ablation 段，只开目标开关
make_temp_config() {
    local ablation="$1"
    local out="/tmp/dbond_ablation_${ablation}_$$.yaml"

    python3 - "$CONFIG_SRC" "${out}" "${ablation}" <<'PYEOF'
import sys
import yaml
import copy

src_path, out_path, ablation = sys.argv[1], sys.argv[2], sys.argv[3]

with open(src_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# 所有消融开关的完整清单
ALL_SWITCHES = [
    "use_sequence_graph", "use_hybrid_graph", "disable_global_node",
    "gcn_only", "gat_only",
    "no_message_passing", "no_edge_attr", "no_state_env",
]

# 强制重建 ablation 段：所有开关置 false，再开目标
ablation_cfg = cfg.setdefault("ablation", {})
ablation_cfg["tag"] = None  # null = 自动推导 tag
ablation_cfg["base_experiment_name"] = None
for sw in ALL_SWITCHES:
    ablation_cfg[sw] = False
ablation_cfg["rebuild_cache"] = True

# 开启目标消融开关
if ablation not in ALL_SWITCHES:
    raise ValueError(
        f"未知消融开关: {ablation}\n"
        f"支持的开关: {ALL_SWITCHES}"
    )
ablation_cfg[ablation] = True

# 显式确保图策略与骨干参数是干净基线值（防止 default.yaml 被改脏）
cfg.setdefault("model", {})["num_gcn_layers"] = 0
cfg.setdefault("model", {})["num_gat_layers"] = 5
cfg.setdefault("data", {})["graph_strategy"] = "distance"
cfg.setdefault("model", {})["use_long_range_edges"] = False

with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print(f"  [config] 临时配置写入 {out_path}")
print(f"  [config] 开关: {ablation}=true, 其余=false")
PYEOF
    echo "${out}"
}

# ============== 顺序执行 ==============
for ABLATION in ${ABLATIONS}; do
    echo ""
    echo "------------------------------------------"
    echo "[$(date '+%H:%M:%S')] 开始消融: ${ABLATION}"
    echo "------------------------------------------"

    # 1. 清理图缓存（仅对改变拓扑的消融）
    clear_graph_cache "${ABLATION}"

    # 2. 生成临时配置
    TEMP_CONFIG=$(make_temp_config "${ABLATION}")

    # 3. 跑 5 折（隔离显卡）
    echo "  [run] CUDA_VISIBLE_DEVICES=${DEVICE_ID} python train_5fold.py --config ${TEMP_CONFIG}"
    CUDA_VISIBLE_DEVICES="${DEVICE_ID}" \
        python graph_transform/scripts/train_5fold.py \
        --config "${TEMP_CONFIG}"

    # 4. 清理临时配置
    rm -f "${TEMP_CONFIG}"
    echo "  [done] ${ABLATION} 五折完成，临时配置已删除"

    # 5. 找到并打印这次的汇总结果
    LATEST_SUMMARY=$(ls -t checkpoints/graph_transform/5fold/*/5fold_summary.csv 2>/dev/null | head -1 || true)
    if [[ -n "${LATEST_SUMMARY}" && -f "${LATEST_SUMMARY}" ]]; then
        echo "  [summary] ${ABLATION} 汇总:"
        cat "${LATEST_SUMMARY}"
    fi
done

echo ""
echo "=========================================="
echo " 全部消融完成: ${ABLATIONS}"
echo "=========================================="
