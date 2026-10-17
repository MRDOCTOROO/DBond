#!/usr/bin/env bash
# 基线多种子集成批次：m/af/af_opt × base_seed 43-46，5 折（s 跳过）
# 成员拼装：各模型已有 seed42 正式 run + 本批 4 个新 seed = 5-seed 集成（GT-Ens 同规格）
# 启动方式（agent 已执行）：
#   cd /mnt/pvc/graphtrans/DBond && nohup setsid bash logs/ens_baselines/run_batch.sh \
#       > logs/ens_baselines/master.log 2>&1 < /dev/null &
set -u
cd /mnt/pvc/graphtrans/DBond
source .venv/bin/activate
mkdir -p logs/ens_baselines
echo "[batch] start $(date '+%F %T %Z')"

run_chain() {
  local gpu=$1; shift
  for j in "$@"; do
    local mdl=${j%%:*} seed=${j##*:} script
    case $mdl in
      m)      script=ludbond/train_dbond_m_pre_5fold.py ;;
      af)     script=ludbondaf/train_dbond_af_pre_5fold.py ;;
      af_opt) script=ludbondaf/train_dbond_af_opt_pre_5fold.py ;;
    esac
    echo "[chain gpu=$gpu] $(date '+%F %T') START $mdl seed=$seed"
    CUDA_VISIBLE_DEVICES=$gpu python "$script" --base_seed "$seed" \
        > "logs/ens_baselines/${mdl}_s${seed}.log" 2>&1
    echo "[chain gpu=$gpu] $(date '+%F %T') DONE  $mdl seed=$seed rc=$?"
  done
}

run_chain 1 af:43 af:44 af_opt:45 m:45 &
run_chain 2 af:45 af:46 af_opt:46 m:46 &
run_chain 4 af_opt:43 af_opt:44 m:43 m:44 &
wait
echo "[batch] all chains finished $(date '+%F %T %Z')"

# 自动聚合：发现 5 个 seed 成员 → 概率平均 → 统一口径评测 → 打印对照表
python graph_transform/scripts/ensemble_baseline_preds.py \
    --models dbond_m_pre dbond_af_pre dbond_af_opt_pre --dry_run
python graph_transform/scripts/ensemble_baseline_preds.py \
    --models dbond_m_pre dbond_af_pre dbond_af_opt_pre \
    --out_dir result/metric/ens_baselines

touch logs/ens_baselines/ALL_DONE
echo "[batch] ALL_DONE marker written $(date '+%F %T %Z')"