import torch
import yaml
from dbond_s import Model as Net
from dbond_s import focal_loss
from data_utils_dbond_s import PepDataset,collate_callback
from sklearn.metrics import recall_score,precision_score,accuracy_score,confusion_matrix,ConfusionMatrixDisplay,f1_score,roc_auc_score,average_precision_score
from torch.utils.tensorboard import SummaryWriter
import tqdm
from typing import List,Callable,Dict
import numpy
import datetime
import argparse
import random
import os
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s[%(levelname)s]:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.Formatter.converter = lambda *args: (datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=8)).timetuple()


MODEL = 'dbond_s'
# Beijing clock
def get_beijing_time():
    return datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=8)

now = datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=8)
# format
run_time = now.strftime("%Y_%m_%d_%H_%M")

tensorboard_log_pattern = './tensorboard/{model}/{time}_{status}_{tag}'
checkpoint_path_pattern = './checkpoint/{model}/{time}_{tag}_{epoch}.pt'
model_weight_path_pattern = './best_model/{model}/{time}_{tag}_{epoch}.pt'
model_weight_dir_pattern =  './best_model/{model}'
model_weight_dir = model_weight_dir_pattern.format(model = MODEL)


def set_seed(seed:int):
    """固定随机种子(对齐 DBond-GT 协议: 5fold 时 seed = 42 + fold_index)"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)


def build_optimizer(model, config)->torch.optim.Optimizer:
    if config['train_args']['optimizer'].lower() == 'sgd':
        return torch.optim.SGD(model.parameters(),**config['train_args']['optimizer_args'])
    elif config['train_args']['optimizer'].lower() == 'adam':
        return torch.optim.Adam(model.parameters(),**config['train_args']['optimizer_args'])
    raise ValueError(f"unknown optimizer: {config['train_args']['optimizer']}")


def build_loss_func(config)->Callable:
    if config['train_args']['loss_type'].lower() == 'ce':
        return lambda logits,labels:torch.nn.functional.cross_entropy(logits,labels,reduction='mean',**config['train_args']['loss_args'])
    elif config['train_args']['loss_type'].lower() == 'focal':
        return lambda logits,labels:focal_loss(logits,labels,reduction='mean',**config['train_args']['loss_args'])
    raise ValueError(f"unknown loss_type: {config['train_args']['loss_type']}")


def early_stopping(patience=5, delta=1e-4)->Callable[[float],bool]:
    """对'越大越好'的指标(如 val_f1): 严格提升(> best+delta)才更新 best 并重置计数,
    否则计数。与 DBond-GT 早停语义一致(修复 best drift bug)。"""
    best_metric = None
    counter = 0
    early_stop = False

    def check_stop(metric:float)->bool:
        nonlocal best_metric, counter, early_stop

        # 与 DBond-GT 早停语义一致: 严格提升(metric > best + delta)才更新 best 并重置计数,
        # 否则计数。修复原 bug(原 else 分支会把 best 重置为当前 metric, 缓慢下降曲线下永不触发早停)。
        if best_metric is None or metric > best_metric + delta:
            best_metric = metric
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                early_stop = True

        return early_stop

    return check_stop


def save_checkpoint(save_path, metric, status, model, optimizer, config, epoch):
    checkpoint_dict:dict = {}
    checkpoint_dict.update(config)
    checkpoint_dict['optimizer_state_dict'] = optimizer.state_dict()
    checkpoint_dict['model_state_dict'] = model.state_dict()
    checkpoint_dict['train_args']['save_epoch'] = epoch
    checkpoint_dict['metric'] = metric
    checkpoint_dict['status'] = status
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint_dict,save_path)
    return


def _run_epoch(epoch, status, writer, dataloader, dataset_len, model, loss_func,
               device, config, optimizer=None)->Dict:
    """跑一个 epoch 的 train 或 validation。
    optimizer 非 None 时反传训练, None 时仅前向(eval)。
    """
    if status.lower() == 'train':
        model.train()
    elif status.lower() == 'validation':
        model.eval()
    else:
        return None

    loss_sum = []
    preds = []
    preds_probs = []
    trues = []

    with tqdm.tqdm(dataloader, total =len(dataloader),unit='batch') as loop:
        for seq_index_batch,seq_padding_mask_batch,bond_index_batch,bond_vec_batch,state_vec_batch,env_vec_batch,label_real_batch in loop:
            loop.set_description(f"{status.capitalize()} Epoch [{epoch}/{config['train_args']['epoch']}]")

            seq_index_batch=seq_index_batch.to(device)
            seq_padding_mask_batch=seq_padding_mask_batch.to(device)
            bond_index_batch=bond_index_batch.to(device)
            bond_vec_batch=bond_vec_batch.to(device)
            state_vec_batch=state_vec_batch.to(device)
            env_vec_batch=env_vec_batch.to(device)
            label_real_batch = label_real_batch.to(device)
            if status.lower() == 'train':
                model.zero_grad()
            label_predict_batch = model.forward(seq_index_batch=seq_index_batch,
                                                                seq_padding_mask_batch=seq_padding_mask_batch,
                                                                bond_index_batch=bond_index_batch,
                                                                bond_vec_batch=bond_vec_batch,
                                                                state_vec_batch=state_vec_batch,
                                                                env_vec_batch=env_vec_batch)
            loss:torch.Tensor = loss_func(label_predict_batch,label_real_batch)

            if status.lower() == 'train' and optimizer is not None:
                loss.backward()
                optimizer.step()

            loss_sum.append(label_real_batch.shape[0]*loss.item())

            label_prob_batch = torch.nn.functional.softmax(label_predict_batch,dim=1)

            label_predict_batch = label_predict_batch.argmax(dim=1)

            preds.extend(label_predict_batch.detach().cpu().numpy())
            preds_probs.extend(label_prob_batch[:,1].detach().cpu().numpy())
            trues.extend(label_real_batch.detach().cpu().numpy())
            loop.set_postfix({'loss':loss.item()} )


    mean_loss = numpy.sum(loss_sum)/dataset_len

    sklearn_accuracy = accuracy_score(trues, preds)
    sklearn_auc = roc_auc_score(trues,preds_probs)
    sklearn_ap = average_precision_score(trues,preds_probs)

    sklearn_precision_label_0,sklearn_precision_label_1 = precision_score(trues, preds, average=None)
    sklearn_recall_label_0,sklearn_recall_label_1 = recall_score(trues, preds, average=None)
    sklearn_f1_label_0,sklearn_f1_label_1 = f1_score(trues, preds, average=None)

    # val_f1: 正类(label=1, 断裂键)的 F1, 等价 sklearn f1_score(average='binary'),
    # 与 DBond-GT task-level binary F1 口径对齐。固定阈值 0.5(单标签 argmax)。
    val_f1 = sklearn_f1_label_1

    metrics_dict = {
        'Loss':mean_loss,
        'accuracy':sklearn_accuracy,
        'AUC':sklearn_auc,
        'AP':sklearn_ap,
        'precision':(sklearn_precision_label_0+sklearn_precision_label_1)/2,
        'recall':(sklearn_recall_label_0+sklearn_recall_label_1)/2,
        'f1':(sklearn_f1_label_0+sklearn_f1_label_1)/2,
        'Label 0: precision':sklearn_precision_label_0,
        'Label 1: precision':sklearn_precision_label_1,
        'Label 0: recall':sklearn_recall_label_0,
        'Label 1: recall':sklearn_recall_label_1,
        'Label 0: f1':sklearn_f1_label_0,
        'Label 1: f1':sklearn_f1_label_1,
        # DBond-GT 协议用 val_f1 选 best model(不再用 AUC)
        'val_f1':val_f1,
    }
    for k,v in metrics_dict.items():
         writer.add_scalar(k,v,epoch)

    return metrics_dict


def main(config:dict, run_id:str=None)->Dict:
    """训练一个 fold 并返回 test 指标。可被单次入口和 5fold 脚本调用。

    config: 已注入数据路径/seed/输出目录的完整配置。
    run_id: 可选运行标识, 用于结果归档目录命名(如 fold_1222)。None 时用时间戳。

    协议对齐(DBond-GT):
    - seed 来自 config['train_args']['seed'](5fold 时 = 42 + fold_index)
    - val 从 train 内 random_split 切 20%(不再读外部 val CSV)
    - best model 按 val_f1 选(不再用 val_auc)
    - 训练结束重载 best model 在 test 上评估, 返回 test 指标
    """
    # ===== seed =====
    seed = config['train_args']['seed']
    set_seed(seed)

    # ===== run id / 输出目录 =====
    if run_id is None:
        run_id = get_beijing_time().strftime("%Y_%m_%d_%H_%M")
    model_weight_dir_cfg = config.get('output', {}).get(
        'best_model_dir', model_weight_dir_pattern.format(model=MODEL))
    checkpoint_dir = config.get('output', {}).get(
        'checkpoint_dir', f'./checkpoint/{MODEL}/{run_id}')
    result_metric_dir = config.get('output', {}).get(
        'result_metric_dir', f'./result/metric/{MODEL}/{run_id}')
    result_pred_dir = config.get('output', {}).get(
        'result_pred_dir', f'./result/pred/{MODEL}/{run_id}')
    tensorboard_dir = config.get('output', {}).get(
        'tensorboard_dir', tensorboard_log_pattern.format(
            model=MODEL, time=run_id, status='run', tag=config['tag']))
    for d in [model_weight_dir_cfg, checkpoint_dir, result_metric_dir, result_pred_dir, tensorboard_dir]:
        os.makedirs(d, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('='*10+str(device)+'='*10)

    # ===== train / val 切分(对齐 DBond-GT: 从 train random_split 20%) =====
    full_train_dataset = PepDataset(config, split='train')
    validation_split = config['train_args'].get('validation_split', 0.2)
    train_size = int((1 - validation_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size], generator=generator)
    logging.info(f"train_size={train_size}, val_size={val_size} (random_split from train, seed={seed})")

    train_dataloader = torch.utils.data.DataLoader(
                        train_dataset,
                        shuffle=False,
                        pin_memory=True,
                        batch_size=config['train_args']['batch_size'],
                        collate_fn=collate_callback,
                        num_workers=config['train_args']['dataloader_workers'])

    validation_dataloader = torch.utils.data.DataLoader(
                        validation_dataset,
                        pin_memory=True,
                        shuffle=False,
                        batch_size=config['train_args']['batch_size'],
                        collate_fn=collate_callback,
                        num_workers=config['train_args']['dataloader_workers'])

    # ===== test 数据(训练后评估用) =====
    test_dataset = PepDataset(config, split='test')
    test_dataloader = torch.utils.data.DataLoader(
                        test_dataset,
                        pin_memory=True,
                        shuffle=False,
                        batch_size=config['train_args']['batch_size'],
                        collate_fn=collate_callback,
                        num_workers=config['train_args']['dataloader_workers'])

    # ===== model / optimizer / loss =====
    model = Net(config)
    logging.info(str(model))
    model.to(device)
    optimizer = build_optimizer(model, config)
    loss_func = build_loss_func(config)

    train_writer = SummaryWriter(tensorboard_dir+f'__train')
    validation_writer = SummaryWriter(tensorboard_dir+f'__validation')

    # ===== 模型选择: 按 val_f1 选(对齐 DBond-GT) =====
    best_validation_f1 = 0.0
    best_validation_model_path = ''
    epoch_cnt_to_save = int(config['train_args']['save_per_epoch'])
    early_stop = early_stopping(**config['train_args']['early_stopping'])

    # ===== 训练循环 =====
    for epoch in range(config['train_args']['epoch']):
        train_metrics_dict = _run_epoch(epoch, 'train', train_writer, train_dataloader,
                                        len(train_dataset), model, loss_func, device, config, optimizer)
        validation_metrics_dict = _run_epoch(epoch, 'validation', validation_writer, validation_dataloader,
                                             len(validation_dataset), model, loss_func, device, config, None)
        logging.info(f"{'#'*10} validation val_f1 {validation_metrics_dict['val_f1']:.4} {'#'*10}")
        if epoch % epoch_cnt_to_save == 0:
            save_path = os.path.join(checkpoint_dir, f'{MODEL}_{run_id}_{config["tag"]}_epoch{epoch}.pt')
            save_checkpoint(save_path, validation_metrics_dict, 'validation', model, optimizer, config, epoch)
            logging.info(f'save checkpoint: {save_path}')

        if early_stop(validation_metrics_dict['val_f1']):
            logging.info(f"{'#'*10} early stop {'#'*10}")
            logging.info(f"{'#'*10} epoch: [{epoch}/{config['train_args']['epoch']}] {'#'*10}")
            logging.info(f"{'#'*10} best validation val_f1 {best_validation_f1:.4} {'#'*10}")
            logging.info(f"{'#'*10} validation val_f1 {validation_metrics_dict['val_f1']:.4} {'#'*10}")
            break
        # best model: val_f1 越大越好(对齐 DBond-GT)
        if validation_metrics_dict['val_f1'] > best_validation_f1:
            best_validation_f1 = validation_metrics_dict['val_f1']
            save_path = os.path.join(model_weight_dir_cfg, f'best_model_{config["tag"]}_epoch{epoch}.pt')
            save_checkpoint(save_path, validation_metrics_dict, 'validation', model, optimizer, config, epoch)
            logging.info(f'save model weight: {save_path}')
            if best_validation_model_path != '' and best_validation_model_path != save_path:
                try:
                    os.remove(best_validation_model_path)
                    logging.info(f'remove success: {best_validation_model_path}')
                except Exception as e:
                    logging.info(f'remove failed: {best_validation_model_path}\nerror: {e}')
            best_validation_model_path = save_path

    train_writer.close()
    validation_writer.close()

    # ===== 重载 best model 在 test 上评估(对齐 DBond-GT) =====
    test_metrics = _evaluate_on_test(model, best_validation_model_path, test_dataloader, test_dataset,
                                     loss_func, device, len(test_dataset), config,
                                     result_metric_dir, result_pred_dir, run_id)
    test_metrics['best_val_f1'] = best_validation_f1
    test_metrics['best_model_path'] = best_validation_model_path
    return test_metrics


def evaluate_only(config: dict, run_id: str = None) -> Dict:
    """仅评估模式(不训练): 扫 best_model_dir 找已有的 best_model_*.pt, 重载后在 test 上评估。

    用于复用已训练好的 best_model, 只补跑 test 评估(如指标口径更新后重算)。
    config: 需含 output.best_model_dir / output.result_metric_dir / output.result_pred_dir + csv.test_dataset_path。
    返回 test 指标 dict(与 main 返回值同结构, 但无 best_val_f1 — best_val_f1 来自训练过程)。
    """
    if run_id is None:
        run_id = get_beijing_time().strftime("%Y_%m_%d_%H_%M")
    best_model_dir = config.get('output', {}).get(
        'best_model_dir', model_weight_dir_pattern.format(model=MODEL))
    result_metric_dir = config.get('output', {}).get('result_metric_dir', f'./result/metric/{MODEL}/{run_id}')
    result_pred_dir = config.get('output', {}).get('result_pred_dir', f'./result/pred/{MODEL}/{run_id}')
    for d in [result_metric_dir, result_pred_dir]:
        os.makedirs(d, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('='*10 + f'{device} (eval_only)' + '='*10)

    # 扫 best_model_dir 找 best_model_*.pt(取修改时间最新的一个)
    import glob
    candidates = sorted(glob.glob(os.path.join(best_model_dir, 'best_model_*.pt')),
                        key=os.path.getmtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f'best_model_dir 下无 best_model_*.pt: {best_model_dir}')
    best_model_path = candidates[0]
    logging.info(f'[eval_only] 使用 best_model: {best_model_path}')

    # 构造 test 数据 + 模型(从 checkpoint 恢复 config, 保证模型结构一致)
    test_dataset = PepDataset(config, split='test')
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, pin_memory=True, shuffle=False,
        batch_size=config['train_args']['batch_size'],
        collate_fn=collate_callback,
        num_workers=config['train_args']['dataloader_workers'])
    model = Net(config).to(device)
    loss_func = build_loss_func(config)

    test_metrics = _evaluate_on_test(model, best_model_path, test_dataloader, test_dataset,
                                     loss_func, device, len(test_dataset), config,
                                     result_metric_dir, result_pred_dir, run_id)
    test_metrics['best_model_path'] = best_model_path
    return test_metrics


def _evaluate_on_test(model, best_model_path, test_dataloader, test_dataset, loss_func, device,
                      dataset_len, config, result_metric_dir, result_pred_dir, run_id)->Dict:
    """重载 best checkpoint, 在 test 上评估。
    单标签预测按 seq 聚合回多标签后, 补充 example/label 级指标(subset_acc/ex_f1/lab_f1),
    以便与 DBond-GT / DBond-m 同口径对比。固定阈值 0.5(argmax)。
    """
    if best_model_path == '' or not os.path.exists(best_model_path):
        logging.warning("best model not found, skip test evaluation")
        return {}
    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    loss_sum = []
    preds = []
    preds_probs = []
    trues = []
    bond_indices = []  # 每条预测对应的键位置, 用于聚合回多标签
    with tqdm.tqdm(test_dataloader, total=len(test_dataloader), unit='batch') as loop:
        for seq_index_batch,seq_padding_mask_batch,bond_index_batch,bond_vec_batch,state_vec_batch,env_vec_batch,label_real_batch in loop:
            loop.set_description(f"Test [{run_id}]")
            seq_index_batch=seq_index_batch.to(device)
            seq_padding_mask_batch=seq_padding_mask_batch.to(device)
            bond_index_batch=bond_index_batch.to(device)
            bond_vec_batch=bond_vec_batch.to(device)
            state_vec_batch=state_vec_batch.to(device)
            env_vec_batch=env_vec_batch.to(device)
            label_real_batch = label_real_batch.to(device)
            with torch.no_grad():
                label_predict_batch = model.forward(seq_index_batch=seq_index_batch,
                                                    seq_padding_mask_batch=seq_padding_mask_batch,
                                                    bond_index_batch=bond_index_batch,
                                                    bond_vec_batch=bond_vec_batch,
                                                    state_vec_batch=state_vec_batch,
                                                    env_vec_batch=env_vec_batch)
                loss = loss_func(label_predict_batch, label_real_batch)
            loss_sum.append(label_real_batch.shape[0]*loss.item())
            label_prob_batch = torch.nn.functional.softmax(label_predict_batch, dim=1)
            label_predict_batch = label_predict_batch.argmax(dim=1)
            preds.extend(label_predict_batch.detach().cpu().numpy())
            preds_probs.extend(label_prob_batch[:,1].detach().cpu().numpy())
            trues.extend(label_real_batch.detach().cpu().numpy())
            bond_indices.extend(bond_index_batch.detach().cpu().numpy())

    mean_loss = numpy.sum(loss_sum)/dataset_len

    # ---- 单标签 task-level 指标 ----
    sklearn_accuracy = accuracy_score(trues, preds)
    sklearn_auc = roc_auc_score(trues, preds_probs)
    sklearn_ap = average_precision_score(trues, preds_probs)
    sklearn_precision_label_0,sklearn_precision_label_1 = precision_score(trues, preds, average=None)
    sklearn_recall_label_0,sklearn_recall_label_1 = recall_score(trues, preds, average=None)
    sklearn_f1_label_0,sklearn_f1_label_1 = f1_score(trues, preds, average=None)
    val_f1 = sklearn_f1_label_1

    # ---- 聚合回多标签: 按 precursor(seq,charge,pep_mass,nce,scan_num) 分组, 用 bond_index 对齐键位置 ----
    # 关键: 不能只按 seq 分组。MS 数据里同一 seq 有几百个 precursor(不同 charge/NCE/scan),
    # 每个 precursor 有自己独立的断裂标签。只按 seq 会把几百个 precursor 压成 1 个 example,
    # 且后写覆盖前写, example 级指标全部失真。precursor 粒度与 DBond-m / DBond-AF / DBond-GT
    # 的多标签 example(一行一个 precursor)完全对齐, 才能公平对比表 3。
    import multi_label_metrics
    seq_col = test_dataset.seq_col_name
    bond_idx_col = test_dataset.bond_index_col_name
    # precursor key 列名取自 config 的 state/env 变量(charge, pep_mass 在 state; nce, scan_num 在 env)
    state_cols = list(config['csv'].get('state_var_col_name', []))
    env_cols = list(config['csv'].get('env_var_col_name', []))
    # 按 (seq, charge, pep_mass, nce, scan_num) 组装 precursor key; 容错: 缺列则退化为更窄 key
    precursor_cols = []
    for c in ['charge', 'pep_mass', 'nce', 'scan_num']:
        if c in state_cols or c in env_cols:
            precursor_cols.append(c)
    df = test_dataset.df
    n_rows = len(df)
    # 构造每行的 precursor key(同 precursor 的所有 bond 行归到同一 example)
    if precursor_cols:
        key_tuples = list(zip(*[df[c].values for c in precursor_cols], df[seq_col].values))
        unique_precursors, ex_inverse = numpy.unique(key_tuples, return_inverse=True)
    else:
        # 退化: 无 precursor 列时按 seq(不推荐, 仅兜底)
        unique_precursors, ex_inverse = numpy.unique(df[seq_col].values, return_inverse=True)
    n_examples = len(unique_precursors)
    logging.info(f"[dbond_s aggregate] n_bond_rows={n_rows}, n_examples(precursor)={n_examples}, "
                 f"avg bonds/example={n_rows/max(n_examples,1):.1f}, precursor_cols={precursor_cols}")
    # 重建每个 example 的多标签向量(按该 precursor 实际键数, 不 pad 到 max_len)
    from collections import defaultdict
    ex_true = defaultdict(dict)
    ex_pred = defaultdict(dict)
    for i in range(n_rows):
        ex = int(ex_inverse[i])
        bi = int(bond_indices[i])
        ex_true[ex][bi] = int(trues[i])
        ex_pred[ex][bi] = int(preds[i])
    # 转成定长矩阵(键数 = 所有 example 中最大 bond_index+1, 与 DBond-m/AF 的 max_len-1 pad 口径一致)
    ex_keys = sorted(ex_true.keys())
    if len(ex_keys) > 0:
        max_bonds = max(max(ex_true[k].keys()) for k in ex_keys) + 1
        gt_mat = numpy.zeros((len(ex_keys), max_bonds), dtype=int)
        pred_mat = numpy.zeros((len(ex_keys), max_bonds), dtype=int)
        for r, k in enumerate(ex_keys):
            for bi, v in ex_true[k].items():
                gt_mat[r, bi] = v
            for bi, v in ex_pred[k].items():
                pred_mat[r, bi] = v
        # 完整 example/label 指标(与 DBond-m / DBond-AF 同口径, 便于表 3 同台对比)
        subset_acc = multi_label_metrics.example_subset_accuracy(gt_mat, pred_mat)
        ex_acc = multi_label_metrics.example_accuracy(gt_mat, pred_mat)
        ex_precision = multi_label_metrics.example_precision(gt_mat, pred_mat)
        ex_recall = multi_label_metrics.example_recall(gt_mat, pred_mat)
        ex_f1 = multi_label_metrics.example_f1(gt_mat, pred_mat)
        lab_acc_ma = multi_label_metrics.label_accuracy_macro(gt_mat, pred_mat)
        lab_acc_mi = multi_label_metrics.label_accuracy_micro(gt_mat, pred_mat)
        lab_precision_ma = multi_label_metrics.label_precision_macro(gt_mat, pred_mat)
        lab_precision_mi = multi_label_metrics.label_precision_micro(gt_mat, pred_mat)
        lab_recall_ma = multi_label_metrics.label_recall_macro(gt_mat, pred_mat)
        lab_recall_mi = multi_label_metrics.label_recall_micro(gt_mat, pred_mat)
        lab_f1_ma = multi_label_metrics.label_f1_macro(gt_mat, pred_mat)
        lab_f1_mi = multi_label_metrics.label_f1_micro(gt_mat, pred_mat)
    else:
        subset_acc = ex_acc = ex_precision = ex_recall = ex_f1 = 0.0
        lab_acc_ma = lab_acc_mi = lab_precision_ma = lab_precision_mi = 0.0
        lab_recall_ma = lab_recall_mi = lab_f1_ma = lab_f1_mi = 0.0

    metrics_dict = {
        'Loss': mean_loss,
        'accuracy': sklearn_accuracy,
        'AUC': sklearn_auc,
        'AP': sklearn_ap,
        'precision': (sklearn_precision_label_0+sklearn_precision_label_1)/2,
        'recall': (sklearn_recall_label_0+sklearn_recall_label_1)/2,
        'f1': (sklearn_f1_label_0+sklearn_f1_label_1)/2,
        'Label 1: f1': sklearn_f1_label_1,
        'val_f1': val_f1,
        # 多标签聚合指标(与 DBond-GT / DBond-m / DBond-AF 同口径)
        'subset_acc': subset_acc,
        'ex_acc': ex_acc,
        'ex_precision': ex_precision,
        'ex_recall': ex_recall,
        'ex_f1': ex_f1,
        'lab_acc_ma': lab_acc_ma,
        'lab_acc_mi': lab_acc_mi,
        'lab_precision_ma': lab_precision_ma,
        'lab_precision_mi': lab_precision_mi,
        'lab_recall_ma': lab_recall_ma,
        'lab_recall_mi': lab_recall_mi,
        'lab_f1_ma': lab_f1_ma,
        'lab_f1_mi': lab_f1_mi,
    }

    # 输出 metric csv
    import csv
    metric_csv = os.path.join(result_metric_dir, 'test_metric.csv')
    with open(metric_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        for k, v in metrics_dict.items():
            w.writerow([k, float(v)])
    logging.info(f'save test metric: {metric_csv}')

    # 输出 pred csv(单标签: 每行一个键, 含 bond_index/true/pred/pred_prob)
    pred_csv = os.path.join(result_pred_dir, 'test.pred.csv')
    with open(pred_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['evaluation_id', 'threshold', 'bond_index', 'true', 'pred', 'pred_prob'])
        for i in range(n_rows):
            w.writerow([run_id, 0.5, int(bond_indices[i]), int(trues[i]), int(preds[i]), float(preds_probs[i])])
    logging.info(f'save test pred: {pred_csv}')

    return metrics_dict


# ===== 单次训练入口(保留原 CLI 兼容) =====
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str,action='store',default='/workspace/dbond_s_config/default.yaml',help='path to config')
    args = parser.parse_args()
    logging.info('='*10+'Args'+'='*10)
    for k,v in vars(args).items():
        logging.info(f'{k:15}\t{v}')

    with open(str(args.config), 'r') as stream:
        config = yaml.safe_load(stream)

    test_metrics = main(config)
    logging.info(f"{'#'*10} final test metrics {'#'*10}")
    for k, v in test_metrics.items():
        logging.info(f'{k:20}\t{v}')
