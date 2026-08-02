import torch
import yaml
from dbond_m import Model as Net
from dbond_m import multilabel_categorical_crossentropy
from data_utils_dbond_m import PepDataset,collate_callback
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

MODEL = 'dbond_m'
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
        return lambda logits,labels:torch.nn.functional.multilabel_soft_margin_loss(logits,labels)
    elif config['train_args']['loss_type'].lower() == 'zlpr':
        return lambda logits,labels:multilabel_categorical_crossentropy(labels,logits)
    raise ValueError(f"unknown loss_type: {config['train_args']['loss_type']}")


def early_stopping(patience=5, delta=1e-4)->Callable[[float],bool]:
    """对'越大越好'的指标(如 val_f1): 严格提升(> best+delta)才更新 best 并重置计数,
    否则计数。与 DBond-GT 早停语义一致(原代码用 loss 时方向是反的且有 best drift bug)。"""
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


def main(config:dict, run_id:str=None)->Dict:
    """训练一个 fold 并返回 test 指标。可被单次入口和 5fold 脚本调用。

    config: 已注入数据路径/seed/输出目录的完整配置。
    run_id: 可选运行标识, 用于结果归档目录命名(如 fold_1222)。None 时用时间戳。

    协议对齐(DBond-GT):
    - seed 来自 config['train_args']['seed'](5fold 时 = 42 + fold_index)
    - val 从 train 内 random_split 切 20%(不再读外部 val CSV)
    - best model 按 val_f1 选(不再用 val_loss)
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

    def _train_step(model, loss):
        model.zero_grad()
        loss.backward()
        optimizer.step()

    def _process_with_step(epoch, status, writer, dataloader, dataset_len):
        """process 的训练版本, 绑定真实 optimizer.step"""
        if status.lower() == 'train':
            model.train()
        else:
            model.eval()
        loss_sum = []
        predict = []
        predict_probs = []
        gt = []
        masks = []  # bond 级有效位 mask, 用于 val_f1 屏蔽 padding(对齐 DBond-GT label_mask)
        with tqdm.tqdm(dataloader, total=len(dataloader), unit='batch') as loop:
            for seq_index_batch,seq_padding_mask_batch,state_vec_batch,env_vec_batch,label_real_batch in loop:
                loop.set_description(f"{status.capitalize()} Epoch [{epoch}/{config['train_args']['epoch']}]")
                seq_index_batch=seq_index_batch.to(device)
                seq_padding_mask_batch=seq_padding_mask_batch.to(device)
                state_vec_batch=state_vec_batch.to(device)
                env_vec_batch=env_vec_batch.to(device)
                label_real_batch = label_real_batch.to(device)
                logits_predict_batch = model.forward(seq_index_batch=seq_index_batch,
                                                    seq_padding_mask_batch=seq_padding_mask_batch,
                                                    state_vec_batch=state_vec_batch,
                                                    env_vec_batch=env_vec_batch)
                loss:torch.Tensor = loss_func(logits_predict_batch,label_real_batch)
                if status.lower() == 'train':
                    _train_step(model, loss)
                loss_sum.append(label_real_batch.shape[0]*loss.item())
                label_prob_batch = torch.nn.functional.sigmoid(logits_predict_batch)
                label_predict_batch = (label_prob_batch > 0.5).long()
                predict.extend(label_predict_batch.detach().cpu().numpy())
                predict_probs.extend(label_prob_batch.detach().cpu().numpy())
                gt.extend(label_real_batch.detach().cpu().numpy())
                # bond i 连残基 i 与 i+1; 残基 i 非 padding 则该键有效。label pad 到 max_len-1 列,
                # mask 取 seq_padding_mask 前 max_len-1 位的补码, 与 label 列数对齐。
                masks.append((~seq_padding_mask_batch[:, :-1]).cpu().numpy())
                loop.set_postfix({'loss':loss.item()})
        mean_loss = numpy.sum(loss_sum)/dataset_len
        import multi_label_metrics
        gt = numpy.vstack(gt)
        predict = numpy.vstack(predict)
        mask = numpy.vstack(masks)
        subset_acc = multi_label_metrics.example_subset_accuracy(gt, predict)
        ex_acc = multi_label_metrics.example_accuracy(gt, predict)
        ex_precision = multi_label_metrics.example_precision(gt, predict)
        ex_recall = multi_label_metrics.example_recall(gt, predict)
        ex_f1 = multi_label_metrics.example_f1(gt, predict)
        lab_acc_ma = multi_label_metrics.label_accuracy_macro(gt, predict)
        lab_acc_mi = multi_label_metrics.label_accuracy_micro(gt, predict)
        lab_precision_ma = multi_label_metrics.label_precision_macro(gt, predict)
        lab_precision_mi = multi_label_metrics.label_precision_micro(gt, predict)
        lab_recall_ma = multi_label_metrics.label_recall_macro(gt, predict)
        lab_recall_mi = multi_label_metrics.label_recall_micro(gt, predict)
        lab_f1_ma = multi_label_metrics.label_f1_macro(gt, predict)
        lab_f1_mi = multi_label_metrics.label_f1_micro(gt, predict)
        # val_f1 只算有效键(屏蔽 padding), 与 DBond-GT valid-key 展平 F1 同口径
        val_f1 = f1_score(gt[mask], predict[mask], zero_division=0)
        metrics_dict = {
            'Loss':mean_loss,
            "subset_acc":subset_acc,
            "ex_acc":ex_acc,
            "ex_precision":ex_precision,
            "ex_recall":ex_recall,
            "ex_f1":ex_f1,
            "lab_acc_ma":lab_acc_ma,
            "lab_acc_mi":lab_acc_mi,
            "lab_precision_ma":lab_precision_ma,
            "lab_precision_mi":lab_precision_mi,
            "lab_recall_ma":lab_recall_ma,
            "lab_recall_mi":lab_recall_mi,
            "lab_f1_ma":lab_f1_ma,
            "lab_f1_mi":lab_f1_mi,
            "val_f1":val_f1,
        }
        for k,v in metrics_dict.items():
            writer.add_scalar(k,v,epoch)
        return metrics_dict

    # ===== 训练循环 =====
    for epoch in range(config['train_args']['epoch']):
        train_metrics_dict = _process_with_step(epoch, 'train', train_writer, train_dataloader, len(train_dataset))
        validation_metrics_dict = _process_with_step(epoch, 'validation', validation_writer, validation_dataloader, len(validation_dataset))
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
    test_metrics = _evaluate_on_test(model, best_validation_model_path, test_dataloader,
                                     loss_func, device, len(test_dataset), config,
                                     result_metric_dir, result_pred_dir, run_id)
    test_metrics['best_val_f1'] = best_validation_f1
    test_metrics['best_model_path'] = best_validation_model_path
    return test_metrics


def _evaluate_on_test(model, best_model_path, test_dataloader, loss_func, device,
                      dataset_len, config, result_metric_dir, result_pred_dir, run_id)->Dict:
    """重载 best checkpoint, 在 test 上评估, 输出 metric csv + pred csv。固定 0.5 阈值。"""
    if best_model_path == '' or not os.path.exists(best_model_path):
        logging.warning("best model not found, skip test evaluation")
        return {}
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    loss_sum = []
    predict = []
    predict_probs = []
    gt = []
    masks = []  # bond 级有效位 mask, 用于 val_f1 屏蔽 padding(对齐 DBond-GT label_mask)
    with tqdm.tqdm(test_dataloader, total=len(test_dataloader), unit='batch') as loop:
        for seq_index_batch,seq_padding_mask_batch,state_vec_batch,env_vec_batch,label_real_batch in loop:
            loop.set_description(f"Test [{run_id}]")
            seq_index_batch=seq_index_batch.to(device)
            seq_padding_mask_batch=seq_padding_mask_batch.to(device)
            state_vec_batch=state_vec_batch.to(device)
            env_vec_batch=env_vec_batch.to(device)
            label_real_batch = label_real_batch.to(device)
            with torch.no_grad():
                logits_predict_batch = model.forward(seq_index_batch=seq_index_batch,
                                                    seq_padding_mask_batch=seq_padding_mask_batch,
                                                    state_vec_batch=state_vec_batch,
                                                    env_vec_batch=env_vec_batch)
                loss = loss_func(logits_predict_batch, label_real_batch)
            loss_sum.append(label_real_batch.shape[0]*loss.item())
            label_prob_batch = torch.nn.functional.sigmoid(logits_predict_batch)
            label_predict_batch = (label_prob_batch > 0.5).long()
            predict.extend(label_predict_batch.detach().cpu().numpy())
            predict_probs.extend(label_prob_batch.detach().cpu().numpy())
            gt.extend(label_real_batch.detach().cpu().numpy())
            masks.append((~seq_padding_mask_batch[:, :-1]).cpu().numpy())

    import multi_label_metrics
    gt = numpy.vstack(gt)
    predict = numpy.vstack(predict)
    predict_probs = numpy.vstack(predict_probs)
    mask = numpy.vstack(masks)
    mean_loss = numpy.sum(loss_sum)/dataset_len

    metrics_dict = {
        'Loss': mean_loss,
        "subset_acc": multi_label_metrics.example_subset_accuracy(gt, predict),
        "ex_acc": multi_label_metrics.example_accuracy(gt, predict),
        "ex_precision": multi_label_metrics.example_precision(gt, predict),
        "ex_recall": multi_label_metrics.example_recall(gt, predict),
        "ex_f1": multi_label_metrics.example_f1(gt, predict),
        "lab_acc_ma": multi_label_metrics.label_accuracy_macro(gt, predict),
        "lab_acc_mi": multi_label_metrics.label_accuracy_micro(gt, predict),
        "lab_precision_ma": multi_label_metrics.label_precision_macro(gt, predict),
        "lab_precision_mi": multi_label_metrics.label_precision_micro(gt, predict),
        "lab_recall_ma": multi_label_metrics.label_recall_macro(gt, predict),
        "lab_recall_mi": multi_label_metrics.label_recall_micro(gt, predict),
        "lab_f1_ma": multi_label_metrics.label_f1_macro(gt, predict),
        "lab_f1_mi": multi_label_metrics.label_f1_micro(gt, predict),
        "val_f1": f1_score(gt[mask], predict[mask], zero_division=0),
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

    # 输出 pred csv(true / pred / pred_prob 展平, 对齐 GT latest_test.pred.csv 风格)
    pred_csv = os.path.join(result_pred_dir, 'test.pred.csv')
    with open(pred_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['evaluation_id', 'threshold', 'true', 'pred', 'pred_prob'])
        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                w.writerow([run_id, 0.5, int(gt[i, j]), int(predict[i, j]), float(predict_probs[i, j])])
    logging.info(f'save test pred: {pred_csv}')

    return metrics_dict


# ===== 单次训练入口(保留原 CLI 兼容) =====
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str,action='store',default='/workspace/dbond_m_config/default.yaml',help='path to config')
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
