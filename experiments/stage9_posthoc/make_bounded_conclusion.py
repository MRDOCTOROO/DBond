"""Pod-only scalar synthesis for the stage-9 failed-export hierarchy batch.

Does not train, select a checkpoint, change source artifacts, or promote a model.
The strict posthoc audit remains unverified; this report isolates only the
preregistered F1-gate conclusion supported by formal CSV and saved-weight replay.
"""

import datetime
import hashlib
import json
from pathlib import Path

OPS = Path('/mnt/pvc/graphtrans/four_state_virtual_hierarchy_ops')
AUDIT = OPS / 'posthoc_review_v2_20261024'
OUT = AUDIT / 'bounded_f1_conclusion_v2.md'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(name):
    return json.loads((AUDIT / name).read_text(encoding='utf-8'))


result = read('result.json')
receipt = read('review_receipt.json')
status = read('audit_status.json')
repeat_1222 = read('repeat_1222_result.json')
repeat_3514 = read('repeat_3514_result.json')
repeat_3514_gpu2 = read('repeat_3514_gpu2_result.json')
launch = json.loads((OPS / 'launch.json').read_text(encoding='utf-8'))
terminal = json.loads((OPS / 'terminal.json').read_text(encoding='utf-8'))

require(status['status'] == 'unverified_full_precision_disagreement'
        and receipt['original_terminal_status'] == terminal['status'] == 'failed',
        'Strict audit or original failed terminal changed')
require(result['gitrev'] == launch['gitrev']
        == '5d7b27ec7c894d60f3cd6d8708a9c3cc84b3b4ed'
        and result['root'] == launch['root'] == terminal['root'],
        'Batch identity changed')
require(sha(OPS / 'terminal.json') == receipt['original_terminal']['original_terminal_sha256'],
        'Original terminal bytes changed')

arms = ('single', 'grouped_skip', 'grouped_strict')
folds = ('1222', '2252', '3514', '6072', '9075')
replay = result['posthoc_audit']['full_precision_replay']
max_f1_drift = max(abs(replay[a][f]['delta_from_original']['f1'])
                   for a in arms for f in folds)
require(max_f1_drift < 5e-5, 'F1 replay exceeded declared audit tolerance')

rows = []
for arm in arms:
    formal = sum(result['folds'][arm][fold]['f1'] for fold in folds) / len(folds)
    checked = sum(replay[arm][fold]['full_precision']['f1'] for fold in folds) / len(folds)
    rows.append((arm, formal, checked))

comparisons = []
for arm in arms[1:]:
    formal_delta = (next(row[1] for row in rows if row[0] == arm) - rows[0][1])
    replay_delta = (next(row[2] for row in rows if row[0] == arm) - rows[0][2])
    formal_wins = sum(result['folds'][arm][f]['f1'] > result['folds']['single'][f]['f1']
                      for f in folds)
    replay_wins = sum(replay[arm][f]['full_precision']['f1']
                      > replay['single'][f]['full_precision']['f1'] for f in folds)
    conservative_upper = formal_delta + 2 * 5e-5
    require(conservative_upper < .002 and formal_wins < 4 and replay_wins < 4,
            'Bounded no-F1-gate result no longer holds')
    comparisons.append((arm, formal_delta, replay_delta, formal_wins, replay_wins,
                        conservative_upper))

proof = result['recovery']['grouped_skip']['1222']['prediction_recomputation']
formal_precision = result['folds']['grouped_skip']['1222']['ex_precision']
export_precision = proof['computed']['ex_precision']
repeat_precision = repeat_1222['repeat_replay']['ex_precision']
require(abs(formal_precision - export_precision) > proof['tolerances']['ex_precision']
        and abs(formal_precision - repeat_precision) <= proof['tolerances']['ex_precision'],
        'Export/precision diagnostic identity changed')
formal_qseq = repeat_3514['formal']['q_seq']
first_qseq = repeat_3514['first_replay']['q_seq']
second_qseq = repeat_3514['repeat_replay']['q_seq']
gpu2_qseq = repeat_3514_gpu2['repeat_replay']['q_seq']
require(first_qseq != formal_qseq and second_qseq == formal_qseq
        and gpu2_qseq == formal_qseq,
        'Sequence-rank repeat diagnostic changed')

lines = [
    '# 层级虚拟节点：失败终态后的限定结论',
    '',
    f'生成时间：{datetime.datetime.now(datetime.timezone.utc).isoformat()}。',
    '',
    '**原自动分析终态仍为 failed；15次训练均已完成，事后严格全指标审计仍标记 unverified。** '
    '本补充报告只判断预先约定的 F1 工程门槛；不能把它读作整批结果已通过验收。',
    '',
    '## F1：两种层级连接均未达到提升门槛',
    '',
    '| 结构 | 正式F1均值 | 保存权重复评F1均值 | 相对single正式差值 | 复评差值 | 正式/复评改善划分 |',
    '|---|---:|---:|---:|---:|---:|',
]
for arm, formal, checked in rows:
    if arm == 'single':
        lines.append(f'| {arm} | {formal*100:.6f}% | {checked*100:.6f}% | 基线 | 基线 | — |')
    else:
        item = next(comp for comp in comparisons if comp[0] == arm)
        lines.append(f'| {arm} | {formal*100:.6f}% | {checked*100:.6f}% | '
                     f'{item[1]*100:+.6f} pp | {item[2]*100:+.6f} pp | '
                     f'{item[3]}/5、{item[4]}/5 |')
lines += [
    '',
    '预设门槛：相对同轮single的平均F1至少+0.2个百分点、至少4/5划分F1改善，'
    '同时AUC不下降且q-Brier不变差。两种层级结构在前两项F1要求就已失败。',
    f'15份权重复评与正式F1的最大绝对差为 {max_f1_drift:.9g}（0—1比例）。'
    '若每臂每折F1误差不超过本次事后审计选用的±0.00005容差，'
    '则得到下列有条件的差值包络；该容差并非已证明的误差界限，也不是统计置信区间：',
]
for arm, formal_delta, checked_delta, formal_wins, replay_wins, upper in comparisons:
    lines.append(f'- {arm}：在上述容差条件下的平均F1差值上界 {upper*100:+.6f} pp，'
                 '仍远低于+0.2 pp；正式和复评的改善划分均不足4/5。')
lines += [
    '',
    '## 其余指标：仅作带审计缺口的描述',
    '',
    '原正式CSV中，grouped_skip相对single的AUC平均+0.00251746、q-Brier平均−0.00201738、'
    'q_seq平均+0.01692503，但谱图P/R调和F1约−0.003438个百分点；'
    'grouped_strict的F1、AUC、q-Brier均值均较差。'
    '这些是同轮配对描述值，不能据此宣布层级方案整体更优。完整八指标的均值、样本SD、'
    '逐划分配对差及资源数据见同目录 `report.md`、`result.json` 和看板。',
    '',
    '## 导出核验缺口与复评证据',
    '',
    f'- grouped_skip/1222：逐谱图CSV重算的谱图精度 {export_precision:.12f}，'
    f'正式指标 {formal_precision:.12f}，差 {abs(export_precision-formal_precision):.9g}，'
    f'超过原分析容差 {proof["tolerances"]["ex_precision"]:.0e}。'
    f'两次保存权重复评为 {repeat_1222["first_replay"]["ex_precision"]:.12f}、'
    f'{repeat_precision:.12f}，均落在原容差内；这不能修复已保存导出文件。',
    f'- grouped_skip/3514：序列Spearman正式值 {formal_qseq:.12f}，'
    f'首次GPU0复评 {first_qseq:.12f}，再次GPU0复评 {second_qseq:.12f}，'
    f'原物理GPU2复评 {gpu2_qseq:.12f}。后两次与正式值相同；'
    '近并列排序对浮点扰动敏感是合理解释，但具体算子原因未证实。',
    '- 复评中部分接近0.5的键在不同推理轮次翻转；原预测CSV和正式指标来自两次独立推理。'
    '本审计未证明某个具体GPU算子有缺陷，也没有重写原CSV或放宽原失败终态。',
    '',
    '## 归档与解释边界',
    '',
    f'- 原训练源码：`{result["gitrev"]}`；原训练root：`{result["root"]}`。',
    f'- 原 `terminal.json` SHA256：`{sha(OPS / "terminal.json")}`；'
    f'严格审计 `review_receipt.json` SHA256：`{sha(AUDIT / "review_receipt.json")}`。',
    f'- 两份单折重复复评JSON SHA256：`{sha(AUDIT / "repeat_1222_result.json")}`、'
    f'`{sha(AUDIT / "repeat_3514_result.json")}`；原GPU2重复：'
    f'`{sha(AUDIT / "repeat_3514_gpu2_result.json")}`。',
    '- 15份checkpoint的路径、SHA、保存配置、训练/划分种子以及来源绑定：'
    '`checkpoint_registry.json`、`review_receipt.json`。',
    '- 五组是重叠划分且每组仅一次训练初始化；组节点按序列位置划分，'
    '不代表真实空间折叠。F1未过门槛不等于层级结构在所有任务上无效。'
    '保留原single-k5候选，不自动替换主模型或启动下一轮。',
    '',
]
require(not OUT.exists(), 'Refusing to overwrite bounded conclusion')
OUT.write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps({'path': str(OUT), 'sha256': sha(OUT),
                  'f1_gate': 'both_candidates_fail',
                  'strict_audit': status['status']}))
