"""DBond-s 5fold 训练(对齐 DBond-GT 协议)。

用法(云端):
  python ludbond/train_dbond_s_5fold.py --config ludbond/dbond_s_config/default.yaml \
      --fold_data_dir dataset/5fold
  python ludbond/train_dbond_s_5fold.py --folds 1222   # 调试单 fold

数据(需先 copy 到 {fold_data_dir}, 来自 G:\\Download):
  {fold_id}.train.shuffle.csv   (单标签 train)
  {fold_id}.test.csv            (单标签 test)
  fold_id ∈ {1222,2252,3514,6072,9075}

结果输出:
  ./result/cv/dbond_s/{timestamp}/fold_{id}/{best_model,metric,pred,...}/
  ./result/cv/dbond_s/{timestamp}/5fold_metrics.csv  (每 fold 一行)
  ./result/cv/dbond_s/{timestamp}/5fold_summary.csv  (mean ± std)
"""
from _5fold_common import run_5fold, build_argparser

# 训练模块所在目录(本脚本自身目录, 即 ludbond/), 用绝对路径供 importlib 定位, 不依赖 CWD
import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# DBond-s 单标签 fold 文件后缀
TRAIN_SUFFIX = '.train.shuffle.csv'
TEST_SUFFIX  = '.test.csv'
MODEL_NAME   = 'dbond_s'
DEFAULT_CONFIG = 'dbond_s_config/default.yaml'


def main():
    parser = build_argparser(MODEL_NAME, DEFAULT_CONFIG)
    args = parser.parse_args()
    run_5fold(
        base_config_path=args.config,
        fold_data_dir=args.fold_data_dir,
        train_module_name='train.dbond_s',
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
