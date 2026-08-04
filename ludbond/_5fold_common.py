"""5fold 封装通用逻辑(DBond-s / DBond-m 共用)。

仿 graph_transform/scripts/train_5fold.py, 把 ludbond 的单 fold 训练脚本
(train.dbond_s.py / train.dbond_m.py) 的 main(config, run_id) 逐 fold 调用,
在 DBond-GT 完全相同的协议(5 个 sequence-level split + seed 42+fold_index +
val random_split + val_f1 选模型)下重跑基线, 并汇总 mean ± std。

协议对齐(DBond-GT):
- 5 个 fold: 1222/2252/3514/6072/9075(sequence-level split, test 互不重叠)
- seed: base_seed(42) + fold_index(0~4)
- val: 从 train random_split 20%(在 train.dbond_*.py 的 main 内完成)
- 模型选择: val_f1(在 train.dbond_*.py 的 main 内完成)
- 汇总: test 指标 mean ± std(ddof=0) + min/max/num_folds
"""
import os
import copy
import yaml
import argparse
import datetime
import logging
import pandas
import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s[%(levelname)s]:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.Formatter.converter = lambda *args: (datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=8)).timetuple()

# 5 个 sequence-level split(与 DBond-GT 完全一致)
FOLD_IDS = ['1222', '2252', '3514', '6072', '9075']
DEFAULT_FOLD_DIR = 'dataset/5fold'
DEFAULT_BASE_SEED = 42  # 与 DBond-GT 一致; 实际 seed = base_seed + fold_index

# 不参与 5fold 聚合的指标前缀/字段(与 GT train_5fold.py 一致)
SUMMARY_METRIC_EXCLUDE_PREFIXES = ('gpu_mem_',)
SUMMARY_METRIC_EXCLUDE_KEYS = {'best_model_path'}


def beijing_now():
    return datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=8)


def make_fold_config(base_config:dict, fold_id:str, fold_index:int, fold_dir:str,
                     train_suffix:str, test_suffix:str, base_seed:int,
                     model_name:str, cv_root:str)->dict:
    """为一个 fold 生成专属 config: 注入数据路径 / seed / 输出目录。

    train/test 路径 = {fold_dir}/{fold_id}{train_suffix|test_suffix}
    """
    cfg = copy.deepcopy(base_config)
    fold_dir = fold_dir.rstrip('/')

    cfg['csv']['train_dataset_path'] = f'{fold_dir}/{fold_id}{train_suffix}'
    cfg['csv']['test_dataset_path'] = f'{fold_dir}/{fold_id}{test_suffix}'
    # validation_dataset_path 已弃用(val 由 random_split 切出), 置空避免误用
    cfg['csv']['validation_dataset_path'] = ''

    # seed 对齐 DBond-GT: 42 + fold_index
    cfg['train_args']['seed'] = base_seed + fold_index

    # 每个 fold 独立的输出目录
    run_root = os.path.join(cv_root, f'fold_{fold_id}')
    cfg['output'] = {
        'best_model_dir':    os.path.join(run_root, 'best_model'),
        'checkpoint_dir':    os.path.join(run_root, 'checkpoint'),
        'result_metric_dir': os.path.join(run_root, 'metric'),
        'result_pred_dir':   os.path.join(run_root, 'pred'),
        'tensorboard_dir':   os.path.join(run_root, 'tensorboard'),
    }
    cfg['_fold_id'] = fold_id
    cfg['_fold_index'] = fold_index
    cfg['_run_root'] = run_root
    return cfg


def should_aggregate_metric(metric_name:str)->bool:
    """判断某指标是否参与 5fold 聚合(排除路径类字段 + gpu_mem 前缀)。"""
    if metric_name in SUMMARY_METRIC_EXCLUDE_KEYS:
        return False
    if metric_name.startswith(SUMMARY_METRIC_EXCLUDE_PREFIXES):
        return False
    return True


def aggregate_5fold(per_fold_metrics:list, output_dir:str, model_name:str)->tuple:
    """把 5 个 fold 的 test 指标聚合成 mean ± std(ddof=0) + min/max/num_folds。

    per_fold_metrics: list[dict], 每个 dict 含 fold_id/seed/best_val_f1 + 各 test 指标。
    输出:
      {output_dir}/5fold_metrics.csv  (每 fold 一行)
      {output_dir}/5fold_summary.csv  (指标 × {mean,std,min,max,num_folds})
    返回 (metrics_df, summary_df)。
    """
    metrics_df = pandas.DataFrame(per_fold_metrics)
    metrics_csv = os.path.join(output_dir, '5fold_metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    logging.info(f'save 5fold per-fold metrics: {metrics_csv}')

    # 聚合(对齐 GT train_5fold.py:264-286)
    agg_rows = []
    metric_cols = [c for c in metrics_df.columns if should_aggregate_metric(c)]
    for metric in metric_cols:
        series = pandas.to_numeric(metrics_df[metric], errors='coerce').dropna()
        if len(series) == 0:
            continue
        agg_rows.append({
            'metric': metric,
            'mean': float(series.mean()),
            'std':  float(series.std(ddof=0)),
            'min':  float(series.min()),
            'max':  float(series.max()),
            'num_folds': int(series.shape[0]),
        })
    summary_df = pandas.DataFrame(agg_rows)
    summary_csv = os.path.join(output_dir, '5fold_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    logging.info(f'save 5fold summary: {summary_csv}')
    return metrics_df, summary_df


def run_5fold(base_config_path:str, fold_data_dir:str, train_module_name:str,
              train_suffix:str, test_suffix:str, model_name:str, base_seed:int,
              folds:list=None, force_new:bool=False, train_module_dir:str='.',
              resume_from:str=None, eval_only:bool=False)->tuple:
    """5fold 主流程。

    base_config_path: 基础 config yaml 路径(如 ludbond/dbond_s_config/default.yaml)
    fold_data_dir: 5fold 数据目录(如 dataset/5fold)
    train_module_name: 训练模块名('train.dbond_s' 或 'train.dbond_m'), 需有 main(config, run_id)
    train_suffix / test_suffix: fold 文件后缀
    model_name: 'dbond_s' / 'dbond_m'
    folds: 可选, 子集 fold id 列表(调试用); None=全部 5 折
    train_module_dir: 训练模块所在目录(如 'ludbond' / 'ludbondaf'), 用于 importlib 绝对定位, 不依赖 CWD
    resume_from: 可选, 续跑模式 — 指定旧的 cv_root 目录(如 result/cv/dbond_m/20260802_234055),
                 已完成(test_metric.csv 存在)的 fold 跳过, 未完成的从头训练。结果继续写入该目录。
                 None=全新跑, 生成带时间戳的新 cv_root。
    eval_only: 仅评估模式 — 不训练, 用每个 fold 已有的 best_model_*.pt 重算 test 指标。
               用于复用训练成果, 只更新评估口径(如指标补全后重算)。需配合 resume_from 指定旧 cv_root。
    """
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # cv_root: 续跑模式用指定目录, 否则生成带时间戳的新目录
    if eval_only and not resume_from:
        raise ValueError('eval_only 模式必须配合 --resume_from 指定旧 cv_root(best_model 在旧目录里)')
    if resume_from:
        cv_root = resume_from
        if not os.path.isdir(cv_root):
            raise FileNotFoundError(f'resume_from 目录不存在: {cv_root}')
    else:
        timestamp = beijing_now().strftime('%Y%m%d_%H%M%S')
        cv_root = os.path.join(f'./result/cv/{model_name}', timestamp)
    os.makedirs(cv_root, exist_ok=True)
    mode_tag = '(eval_only)' if eval_only else ('(resume)' if resume_from else '(new)')
    logging.info(f'5fold cv_root: {cv_root} {mode_tag}')

    # 动态 import 训练模块(注意 .py 文件名带点, 用 importlib)
    # 用 train_module_dir 解析绝对路径, 不依赖 CWD
    import importlib.util
    module_file = os.path.join(train_module_dir, f'{train_module_name}.py')
    if not os.path.exists(module_file):
        raise FileNotFoundError(f'训练模块不存在: {module_file} (train_module_dir={train_module_dir})')
    spec = importlib.util.spec_from_file_location(train_module_name, module_file)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    if not hasattr(train_module, 'main'):
        raise AttributeError(f'{train_module_name} 缺少 main(config, run_id) 函数')
    if eval_only and not hasattr(train_module, 'evaluate_only'):
        raise AttributeError(f'{train_module_name} 缺少 evaluate_only(config, run_id) 函数(eval_only 模式需要)')

    fold_ids = folds if folds is not None else FOLD_IDS
    per_fold_metrics = []

    for fold_index, fold_id in enumerate(tqdm.tqdm(fold_ids, desc='5fold')):
        logging.info('='*20 + f' fold {fold_id} (index={fold_index}, seed={base_seed+fold_index}) ' + '='*20)
        fold_config = make_fold_config(
            base_config, fold_id, fold_index, fold_data_dir,
            train_suffix, test_suffix, base_seed, model_name, cv_root)

        if eval_only:
            # 仅评估模式: 用已有 best_model 重算 test 指标, 不训练(复用训练成果)
            run_id = f'fold_{fold_id}'
            logging.info(f'fold {fold_id} [eval_only]: 重载 best_model 评估 test')
            fold_metrics = train_module.evaluate_only(fold_config, run_id=run_id)
        else:
            # 断点续跑判定: 若该 fold 已有 test_metric.csv 且非 force_new, 跳过训练直接读结果
            metric_csv = os.path.join(fold_config['_run_root'], 'metric', 'test_metric.csv')
            if os.path.exists(metric_csv) and not force_new:
                logging.info(f'fold {fold_id} 已有结果, 跳过(force_new=False): {metric_csv}')
                fold_metrics = _read_metric_csv(metric_csv)
            else:
                run_id = f'fold_{fold_id}'
                fold_metrics = train_module.main(fold_config, run_id=run_id)

        # 补充 fold 元信息
        fold_metrics['fold_id'] = fold_id
        fold_metrics['seed'] = base_seed + fold_index
        per_fold_metrics.append(fold_metrics)

    # 汇总
    metrics_df, summary_df = aggregate_5fold(per_fold_metrics, cv_root, model_name)
    logging.info('='*20 + ' 5fold summary ' + '='*20)
    print(summary_df.to_string(index=False))
    return metrics_df, summary_df


def _read_metric_csv(metric_csv:str)->dict:
    """从 test_metric.csv 读回指标(metric,value 两列)。"""
    df = pandas.read_csv(metric_csv)
    return dict(zip(df['metric'], df['value']))


def build_argparser(model_name:str, default_config:str)->argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f'{model_name} 5fold 训练(对齐 DBond-GT 协议)')
    p.add_argument('--config', type=str, default=default_config, help='基础 config yaml 路径')
    p.add_argument('--fold_data_dir', type=str, default=DEFAULT_FOLD_DIR, help='5fold 数据目录')
    p.add_argument('--base_seed', type=int, default=DEFAULT_BASE_SEED, help='基础 seed(实际 seed = base_seed + fold_index)')
    p.add_argument('--folds', type=str, nargs='*', default=None, help='子集 fold id(调试用, 默认全部 5 折)')
    p.add_argument('--force_new', action='store_true', help='忽略已有结果强制重跑(仍写入 resume_from 指定的目录, 或新目录)')
    p.add_argument('--resume_from', type=str, default=None,
                   help='续跑: 指定旧 cv_root 目录(如 result/cv/dbond_m/20260802_234055)。'
                        '已完成(test_metric.csv 存在)的 fold 跳过, 未完成的从头训练。')
    p.add_argument('--eval_only', action='store_true',
                   help='仅评估模式: 不训练, 用每个 fold 已有 best_model 重算 test 指标。'
                        '需配合 --resume_from 指定旧 cv_root(写回原目录)。复用训练成果。')
    return p
