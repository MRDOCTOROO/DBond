"""DBond-AF-opt (超参优化版) 5fold 训练(对齐 DBond-GT 协议)。

用法(云端, 在仓库根目录运行):
  python ludbondaf/train_dbond_af_opt_5fold.py --config ludbondaf/dbond_m_exp_af_config/af_opt.yaml \
      --fold_data_dir dataset/5fold
  python ludbondaf/train_dbond_af_opt_5fold.py --folds 1222   # 调试单 fold

与 train_dbond_af_5fold.py 的唯一区别:
  - config 默认指向 af_opt.yaml(hidden=64, dropout=0.4, bs=1024, attn=1)
  - MODEL_NAME = 'dbond_af_opt'(结果写到 result/cv/dbond_af_opt/, 不与 dbond_af 混)
  训练模块仍是 train.dbond_m.exp_af.py(模型架构不变, 仅超参不同)。

数据(与 DBond-m / DBond-AF 同源多标签, 需先 copy 到 {fold_data_dir}):
  {fold_id}.train.fbr.shuffle.multi.csv   (多标签 train)
  {fold_id}.test.fbr.multi.csv            (多标签 test)
  fold_id ∈ {1222,2252,3514,6072,9075}

结果输出(相对 CWD, 即仓库根):
  ./result/cv/dbond_af_opt/{timestamp}/fold_{id}/{best_model,metric,pred,...}/
  ./result/cv/dbond_af_opt/{timestamp}/5fold_metrics.csv  (每 fold 一行)
  ./result/cv/dbond_af_opt/{timestamp}/5fold_summary.csv  (mean ± std)

复用 ludbond/_5fold_common.py 的通用 5fold 逻辑(importlib 按本脚本所在目录绝对定位)。
"""
import os
import sys

# ===== 跨目录复用 ludbond/_5fold_common.py =====
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))           # .../ludbondaf
_REPO_ROOT = os.path.dirname(_THIS_DIR)                          # 仓库根
_LUDBOND_DIR = os.path.join(_REPO_ROOT, 'ludbond')
sys.path.insert(0, _LUDBOND_DIR)
from _5fold_common import run_5fold, build_argparser

# DBond-AF-opt 多标签 fold 文件后缀(与 DBond-m / DBond-AF 同源, 同名)
TRAIN_SUFFIX = '.train.fbr.shuffle.multi.csv'
TEST_SUFFIX = '.test.fbr.multi.csv'
MODEL_NAME = 'dbond_af_opt'   # 区别于 dbond_af, 结果独立目录归档
DEFAULT_CONFIG = 'dbond_m_exp_af_config/af_opt.yaml'
TRAIN_MODULE_NAME = 'train.dbond_m.exp_af'   # 模型架构不变, 仅超参由 config 控制


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
        train_module_dir=_THIS_DIR,   # 训练模块所在目录(绝对路径), importlib 据此定位
        resume_from=args.resume_from,
        eval_only=args.eval_only,
    )


if __name__ == '__main__':
    main()
