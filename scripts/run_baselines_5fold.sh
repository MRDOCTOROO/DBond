#!/usr/bin/env bash
# 四个基线(DBond-m / DBond-s / DBond-AF / DBond-AF-opt)五折顺序复现脚本(对齐 DBond-GT 协议)
#
# 用途: 在 DBond-GT 完全相同的协议下顺序跑基线的 5 折实验,
#        解决审稿 R-05/R-06 的公平性硬伤, 输出 mean ± SD 供表 3 同台对比。
#
# 常见用法:
#   # 1) 只跑指定模型(最常用: s/m 已跑完, 只补新的 af + af_opt)
#   MODELS="dbond_af dbond_af_opt" bash scripts/run_baselines_5fold.sh
#
#   # 2) 跑全部四个(已完成的会被 SKIP_DONE 或续跑逻辑自动跳过)
#   bash scripts/run_baselines_5fold.sh
#
#   # 3) 中断后再次运行 = 自动续跑(已完成 fold 跳过, 未完成的从头训练)
#   bash scripts/run_baselines_5fold.sh
#
# 复用机制(三层, 从粗到细):
#   1) SKIP_DONE=1 (默认开): 若某模型 state 里的 cv_root 下已有 5fold_summary.csv,
#      整个模型跳过(连 python 都不启动)。这是"模型级"复用, 最省时。
#      想强制重算某模型: SKIP_DONE=0, 或删 state/{model}.cvroot。
#   2) 续跑(--resume_from): SKIP_DONE 没跳过时, 若 state/{model}.cvroot 存在且目录还在,
#      自动 --resume_from 指向它; 训练脚本对每个 fold 检查 test_metric.csv,
#      已完成的 fold 跳过训练, 只跑未完成的。这是"fold 级"复用。
#   3) EVAL_ONLY=1: 不训练, 只用每个 fold 已有 best_model 重算 test 指标。
#      适用指标口径更新后补算(如 dbond_s 聚合 bug 修复后重算 example/label 指标)。
#
# 强制重跑:
#   FORCE_NEW=1 SKIP_DONE=0 bash scripts/run_baselines_5fold.sh   # 忽略已完成 fold 全部重跑(仍写入已记录的 cv_root)
#   # 或彻底重跑某模型: 删 state/{model}.cvroot 再跑(会生成新 cv_root)
#
# 仅评估(复用已训练 best_model, 不重训, 只重算 test 指标):
#   EVAL_ONLY=1 bash scripts/run_baselines_5fold.sh
#   EVAL_ONLY=1 MODELS="dbond_s" bash scripts/run_baselines_5fold.sh   # 只给 dbond_s 补指标
#
# 可调环境变量(在脚本顶部或调用前 export):
#   DEVICE_ID=0             显卡编号(CUDA_VISIBLE_DEVICES)
#   MODELS="dbond_m dbond_s dbond_af dbond_af_opt"   要跑的模型, 空格分隔(默认全部四个)
#   FOLD_DATA_DIR=dataset/5fold         5fold 数据目录
#   SKIP_DONE=1             1=模型级跳过(已有 5fold_summary.csv 则整个模型跳过, 默认开)
#   FORCE_NEW=0             1=强制重跑(忽略已完成 fold; 需配合 SKIP_DONE=0 才生效)
#   EVAL_ONLY=0             1=仅评估(不训练, 用已有 best_model 重算 test 指标)
#
# 输出:
#   result/cv/{dbond_m,dbond_s,dbond_af,dbond_af_opt}/{timestamp}/
#     ├── fold_{1222,2252,3514,6072,9075}/{best_model,checkpoint,metric,pred,tensorboard}/
#     ├── 5fold_metrics.csv     (每 fold 一行)
#     └── 5fold_summary.csv     (mean ± std, 表 3 用)
#   state/{model}.cvroot       (续跑状态文件, 记录 cv_root 路径)

set -uo pipefail   # 注意: 不用 -e, 让单个模型失败不中断后续模型(便于分别排查/续跑)

# ============== 可调配置 ==============
DEVICE_ID="${DEVICE_ID:-0}"
MODELS="${MODELS:-dbond_m dbond_s dbond_af dbond_af_opt}"
FOLD_DATA_DIR="${FOLD_DATA_DIR:-dataset/5fold}"
FORCE_NEW="${FORCE_NEW:-0}"
# EVAL_ONLY=1: 仅评估模式, 不训练, 用每个 fold 已有 best_model 重算 test 指标(复用训练成果)
#              需每个模型已有 state/{model}.cvroot(指向含 best_model 的旧 cv_root)
EVAL_ONLY="${EVAL_ONLY:-0}"
# SKIP_DONE=1: 模型级复用 — 若 state/{model}.cvroot 指向的 cv_root 下已有 5fold_summary.csv,
#              整个模型跳过(连 python 都不启动)。默认开, 省 s/m 这种已完成模型的无谓启动。
#              FORCE_NEW=1 或 EVAL_ONLY=1 时自动关(SKIP_DONE 不该生效)。
SKIP_DONE="${SKIP_DONE:-1}"
if [ "${FORCE_NEW}" = "1" ] || [ "${EVAL_ONLY}" = "1" ]; then
  SKIP_DONE=0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

STATE_DIR="${PROJECT_ROOT}/state"
mkdir -p "${STATE_DIR}"

# 每个模型的 config 路径 + 入口脚本
declare -A MODEL_CONFIG=(
  [dbond_m]="ludbond/dbond_m_config/default.yaml"
  [dbond_s]="ludbond/dbond_s_config/default.yaml"
  [dbond_af]="ludbondaf/dbond_m_exp_af_config/default.yaml"
  [dbond_af_opt]="ludbondaf/dbond_m_exp_af_config/af_opt.yaml"
)
declare -A MODEL_ENTRY=(
  [dbond_m]="ludbond/train_dbond_m_5fold.py"
  [dbond_s]="ludbond/train_dbond_s_5fold.py"
  [dbond_af]="ludbondaf/train_dbond_af_5fold.py"
  [dbond_af_opt]="ludbondaf/train_dbond_af_opt_5fold.py"
)

export CUDA_VISIBLE_DEVICES="${DEVICE_ID}"

FORCE_NEW_FLAG=""
if [ "${FORCE_NEW}" = "1" ]; then
  FORCE_NEW_FLAG="--force_new"
fi

EVAL_ONLY_FLAG=""
if [ "${EVAL_ONLY}" = "1" ]; then
  EVAL_ONLY_FLAG="--eval_only"
fi

echo "========================================================"
echo " 基线 5 折顺序复现(对齐 DBond-GT 协议)"
echo "   模型:   ${MODELS}"
echo "   数据:   ${FOLD_DATA_DIR}"
echo "   GPU:    ${DEVICE_ID}"
echo "   强制重跑: ${FORCE_NEW}"
echo "   仅评估: ${EVAL_ONLY}"
echo "   模型级跳过(SKIP_DONE): ${SKIP_DONE}"
echo "========================================================"

# 整体退出码: 记录哪些模型失败 / 跳过
FAILED_MODELS=""
SKIPPED_MODELS=""

for MODEL in ${MODELS}; do
  CONFIG="${MODEL_CONFIG[${MODEL}]}"
  ENTRY="${MODEL_ENTRY[${MODEL}]}"
  STATE_FILE="${STATE_DIR}/${MODEL}.cvroot"

  echo ""
  echo "--------------------------------------------------------"
  echo " [${MODEL}] config=${CONFIG}  entry=${ENTRY}"
  echo "--------------------------------------------------------"

  if [ ! -f "${CONFIG}" ]; then
    echo " [${MODEL}] 跳过: config 不存在 ${CONFIG}"
    FAILED_MODELS="${FAILED_MODELS} ${MODEL}"
    continue
  fi
  if [ ! -f "${ENTRY}" ]; then
    echo " [${MODEL}] 跳过: 入口脚本不存在 ${ENTRY}"
    FAILED_MODELS="${FAILED_MODELS} ${MODEL}"
    continue
  fi

  # 模型级复用(SKIP_DONE): 若 state 里记录的 cv_root 下已有 5fold_summary.csv,
  # 说明该模型 5 折已全部跑完, 整个跳过(连 python 都不启动), 省 s/m 这种已完成模型的无谓启动。
  if [ "${SKIP_DONE}" = "1" ] && [ -f "${STATE_FILE}" ]; then
    SAVED_CVROOT_CHECK="$(cat "${STATE_FILE}")"
    if [ -d "${SAVED_CVROOT_CHECK}" ] && [ -f "${SAVED_CVROOT_CHECK}/5fold_summary.csv" ]; then
      echo " [${MODEL}] 跳过(SKIP_DONE): 已有 5fold_summary.csv → ${SAVED_CVROOT_CHECK}/5fold_summary.csv"
      echo "        (想重跑: SKIP_DONE=0, 或 FORCE_NEW=1, 或删 state/${MODEL}.cvroot)"
      SKIPPED_MODELS="${SKIPPED_MODELS} ${MODEL}"
      continue
    fi
  fi

  # eval_only 模式: 必须有 state 记录的 cv_root(best_model 在里面)
  if [ "${EVAL_ONLY}" = "1" ]; then
    if [ ! -f "${STATE_FILE}" ]; then
      echo " [${MODEL}] 跳过: eval_only 需要 state/${MODEL}.cvroot(无训练记录, 请先正常训练)"
      FAILED_MODELS="${FAILED_MODELS} ${MODEL}"
      continue
    fi
    SAVED_CVROOT="$(cat "${STATE_FILE}")"
    if [ ! -d "${SAVED_CVROOT}" ]; then
      echo " [${MODEL}] 跳过: state 记录的 cv_root 不存在: ${SAVED_CVROOT}"
      FAILED_MODELS="${FAILED_MODELS} ${MODEL}"
      continue
    fi
    RESUME_FLAG="--resume_from ${SAVED_CVROOT}"
    echo " [${MODEL}] eval_only 模式: cv_root=${SAVED_CVROOT} (只评估不训练)"
  else
    # 续跑判定: 若有 state 文件且对应 cv_root 还在, 用 --resume_from
    RESUME_FLAG=""
    if [ -f "${STATE_FILE}" ]; then
      SAVED_CVROOT="$(cat "${STATE_FILE}")"
      if [ -d "${SAVED_CVROOT}" ]; then
        RESUME_FLAG="--resume_from ${SAVED_CVROOT}"
        echo " [${MODEL}] 续跑模式: cv_root=${SAVED_CVROOT}"
      else
        echo " [${MODEL}] state 记录的 cv_root 已不存在, 全新跑"
        rm -f "${STATE_FILE}"
      fi
    fi
  fi

  # 跑该模型的 5 折
  # 用 python 调用; 若是全新跑(无 RESUME_FLAG), 训练脚本会生成新 cv_root,
  # 但入口脚本目前只在内存里用, 我们从日志/输出里抓 cv_root 写入 state。
  # 为可靠抓 cv_root, 用 tee 同时输出到日志, 再 grep。
  LOG_FILE="${STATE_DIR}/${MODEL}.run.log"
  echo " [${MODEL}] 开始训练, 日志: ${LOG_FILE}"

  set +e
  python "${ENTRY}" \
    --config "${CONFIG}" \
    --fold_data_dir "${FOLD_DATA_DIR}" \
    ${FORCE_NEW_FLAG} \
    ${EVAL_ONLY_FLAG} \
    ${RESUME_FLAG} \
    2>&1 | tee "${LOG_FILE}"
  RC=${PIPESTATUS[0]}
  set -o pipefail

  # 抓取本次的 cv_root(从日志的 "5fold cv_root:" 行), 写入 state 供下次续跑
  CVROOT_LINE="$(grep '5fold cv_root:' "${LOG_FILE}" | tail -1)"
  if [ -n "${CVROOT_LINE}" ]; then
    CVROOT_PATH="$(echo "${CVROOT_LINE}" | sed -E 's/.*5fold cv_root: ([^ ]+).*/\1/')"
    echo "${CVROOT_PATH}" > "${STATE_FILE}"
    echo " [${MODEL}] cv_root 已记录到 state: ${CVROOT_PATH}"
  fi

  if [ ${RC} -eq 0 ]; then
    echo " [${MODEL}] 完成 ✓"
    # 检查 5fold_summary.csv 是否生成(完成的标志)
    SUMMARY="$(cat "${STATE_FILE}" 2>/dev/null)/5fold_summary.csv"
    if [ -f "${SUMMARY}" ]; then
      echo " [${MODEL}] 5fold_summary.csv 已生成: ${SUMMARY}"
    fi
  else
    echo " [${MODEL}] 失败(退出码 ${RC}), 已记录 cv_root, 可重跑本脚本续跑"
    FAILED_MODELS="${FAILED_MODELS} ${MODEL}"
  fi
done

echo ""
echo "========================================================"
# 结果汇总: 列出所有模型的 5fold_summary.csv 位置(含跳过/失败/完成)
echo " 结果汇总(各自 5fold_summary.csv):"
for MODEL in ${MODELS}; do
  STATE_FILE="${STATE_DIR}/${MODEL}.cvroot"
  STATUS_TAG=""
  case " ${SKIPPED_MODELS} " in *" ${MODEL} "*) STATUS_TAG="[跳过]";; esac
  case " ${FAILED_MODELS} " in *" ${MODEL} "*) STATUS_TAG="[失败]";; esac
  if [ -f "${STATE_FILE}" ]; then
    echo "   ${MODEL} ${STATUS_TAG}: $(cat "${STATE_FILE}")/5fold_summary.csv"
  else
    echo "   ${MODEL} ${STATUS_TAG}: (无 state 记录)"
  fi
done
echo "========================================================"
if [ -z "${FAILED_MODELS}" ]; then
  echo " 全部完成 ✓ (跳过: ${SKIPPED_MODELS:-无})"
  exit 0
else
  echo " 以下模型未完成, 请重跑本脚本续跑: ${FAILED_MODELS}"
  echo " (跳过: ${SKIPPED_MODELS:-无})"
  exit 1
fi
