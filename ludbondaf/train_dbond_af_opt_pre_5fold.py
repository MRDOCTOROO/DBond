"""DBond-AF-opt-pre 5fold 训练(R-01 pre 变体, 对齐 DBond-GT-pre)。

与 train_dbond_af_opt_5fold.py 的唯一差异: 默认配置指向 af_opt_pre.yaml
(zero_fill intensity/scan_num; hidden=64/dropout=0.4 等 AF-opt 既有超参不变),
结果写到 result/cv/dbond_af_opt_pre/, 不与 dbond_af_opt 混。

用法(云端, 在仓库根目录运行):
  python ludbondaf/train_dbond_af_opt_pre_5fold.py --config ludbondaf/dbond_m_exp_af_config/af_opt_pre.yaml \
      --fold_data_dir dataset/5fold
  python ludbondaf/train_dbond_af_opt_pre_5fold.py --folds 1222   # 调试单 fold

结果输出:
  ./result/cv/dbond_af_opt_pre/{timestamp}/fold_{id}/{best_model,metric,pred,...}/
  ./result/cv/dbond_af_opt_pre/{timestamp}/5fold_metrics.csv  (每 fold 一行)
  ./result/cv/dbond_af_opt_pre/{timestamp}/5fold_summary.csv  (mean ± std)
"""
import os
import sys

# ===== 跨目录复用 ludbond/_5fold_common.py =====
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))           # .../ludbondaf
_REPO_ROOT = os.path.dirname(_THIS_DIR)                          # 仓库根
_LUDBOND_DIR = os.path.join(_REPO_ROOT, 'ludbond')
sys.path.insert(0, _LUDBOND_DIR)
from _5fold_common import run_5fold, build_argparser

# DBond-AF-opt 多标签 fold 文件后缀(与 dbond_af_opt 相同数据)
TRAIN_SUFFIX = '.train.fbr.shuffle.multi.csv'
TEST_SUFFIX = '.test.fbr.multi.csv'
MODEL_NAME = 'dbond_af_opt_pre'
DEFAULT_CONFIG = 'dbond_m_exp_af_config/af_opt_pre.yaml'
TRAIN_MODULE_NAME = 'train.dbond_m.exp_af'


def main():
    parser = build_argparser(MODEL_NAME, DEFAULT_CONFIG)
    args = parser.parse_args()

    run_5fold(
        base_config_path=args.config,
        fold_data_dir=args.fold_data_dir,
        train_module_name=TRAIN_MODULE_NAME,
        train_suffix=TRAIN_SUFFIX,
        test_suffix=TEST_SUFFIX,
        model_name=MODEL_NAME,
        base_seed=args.base_seed,
        folds=args.folds,
        force_new=args.force_new,
        train_module_dir=_THIS_DIR,
        resume_from=args.resume_from,
        eval_only=args.eval_only,
    )


if __name__ == '__main__':
    main()
