"""DBond-AF (attention-based feature fusion) 5fold 训练(对齐 DBond-GT 协议)。

用法(云端, 在仓库根目录运行):
  python ludbondaf/train_dbond_af_5fold.py --config ludbondaf/dbond_m_exp_af_config/default.yaml \
      --fold_data_dir dataset/5fold
  python ludbondaf/train_dbond_af_5fold.py --folds 1222   # 调试单 fold

数据(需先 copy 到 {fold_data_dir}, 来自 G:\\Download; 与 DBond-m 同源多标签):
  {fold_id}.train.fbr.shuffle.multi.csv   (多标签 train)
  {fold_id}.test.fbr.multi.csv            (多标签 test)
  fold_id ∈ {1222,2252,3514,6072,9075}

结果输出:
  ./ludbondaf/result/cv/dbond_af/{timestamp}/fold_{id}/{best_model,metric,pred,...}/
  ./ludbondaf/result/cv/dbond_af/{timestamp}/5fold_metrics.csv  (每 fold 一行)
  ./ludbondaf/result/cv/dbond_af/{timestamp}/5fold_summary.csv  (mean ± std)

注意: 本脚本复用 ludbond/_5fold_common.py 的通用 5fold 逻辑。
      训练模块 train.dbond_m.exp_af.py 用 importlib 按 CWD 加载, 故本脚本会 chdir
      到 ludbondaf/ 目录, 并把 config 路径转成绝对路径以兼容。
"""
import os
import sys

# ===== 跨目录复用 ludbond/_5fold_common.py =====
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))           # .../ludbondaf
_REPO_ROOT = os.path.dirname(_THIS_DIR)                          # 仓库根
_LUDBOND_DIR = os.path.join(_REPO_ROOT, 'ludbond')
sys.path.insert(0, _LUDBOND_DIR)
from _5fold_common import run_5fold, build_argparser

# DBond-AF 多标签 fold 文件后缀(与 DBond-m 同源, 同名)
TRAIN_SUFFIX = '.train.fbr.shuffle.multi.csv'
TEST_SUFFIX = '.test.fbr.multi.csv'
MODEL_NAME = 'dbond_af'
DEFAULT_CONFIG = 'dbond_m_exp_af_config/default.yaml'
TRAIN_MODULE_NAME = 'train.dbond_m.exp_af'


def main():
    parser = build_argparser(MODEL_NAME, DEFAULT_CONFIG)
    args = parser.parse_args()

    # config 路径转绝对(因为接下来要 chdir 到 ludbondaf, 让 importlib 能找到训练模块)
    config_abs = args.config if os.path.isabs(args.config) else os.path.abspath(args.config)
    fold_dir_abs = args.fold_data_dir if os.path.isabs(args.fold_data_dir) else os.path.abspath(args.fold_data_dir)

    # chdir 到 ludbondaf: _5fold_common 的 importlib 用相对路径加载 train.dbond_m.exp_af.py
    os.chdir(_THIS_DIR)

    run_5fold(
        base_config_path=config_abs,
        fold_data_dir=fold_dir_abs,
        train_module_name=TRAIN_MODULE_NAME,
        train_suffix=TRAIN_SUFFIX,
        test_suffix=TEST_SUFFIX,
        model_name=MODEL_NAME,
        base_seed=args.base_seed,
        folds=args.folds,
        force_new=args.force_new,
    )


if __name__ == '__main__':
    main()
