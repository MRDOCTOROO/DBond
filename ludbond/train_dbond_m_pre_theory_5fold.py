"""DBond-m-pre+theory 5fold 训练(R-01 pre 变体, 对齐 DBond-GT-pre)。

与 train_dbond_m_5fold.py 的唯一差异: 默认配置指向 dbond_m_config/pre.yaml
(zero_fill intensity/scan_num), 结果写到 result/cv/dbond_m_pre_theory/, 不与 dbond_m 混。

用法(云端):
  python ludbond/train_dbond_m_pre_theory_5fold.py --config ludbond/dbond_m_config/pre.yaml \
      --fold_data_dir dataset/5fold
  python ludbond/train_dbond_m_pre_5fold.py --folds 1222   # 调试单 fold

结果输出:
  ./result/cv/dbond_m_pre_theory/{timestamp}/fold_{id}/{best_model,metric,pred,...}/
  ./result/cv/dbond_m_pre_theory/{timestamp}/5fold_metrics.csv  (每 fold 一行)
  ./result/cv/dbond_m_pre_theory/{timestamp}/5fold_summary.csv  (mean ± std)
"""
from _5fold_common import run_5fold, build_argparser

import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# DBond-m 多标签 fold 文件后缀(与 dbond_m 相同数据)
TRAIN_SUFFIX = '.train.fbr.shuffle.multi.csv'
TEST_SUFFIX  = '.test.fbr.multi.csv'
MODEL_NAME   = 'dbond_m_pre_theory'
# 默认配置相对本脚本目录解析(CWD 无关; run_5fold 按 CWD 打开相对路径会踩坑)
DEFAULT_CONFIG = os.path.join(_THIS_DIR, 'dbond_m_config', 'pre_theory.yaml')


def main():
    parser = build_argparser(MODEL_NAME, DEFAULT_CONFIG)
    args = parser.parse_args()
    run_5fold(
        base_config_path=args.config,
        fold_data_dir=args.fold_data_dir,
        train_module_name='train.dbond_m',
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
