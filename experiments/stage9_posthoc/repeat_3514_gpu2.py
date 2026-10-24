"""One checkpoint-only repeat of the stage-9 posthoc inference (no training)."""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, '/mnt/pvc/graphtrans/DBond/graph_transform/scripts')
import audit_virtual_hierarchy_posthoc as audit

result = json.loads((audit.POSTHOC / 'result.json').read_text())
saved = result['recovery']['grouped_skip']['3514']
proof = saved['prediction_recomputation']
runtime = yaml.safe_load(Path(saved['config']).read_text())
repeated = audit.full_precision_evaluate(
    runtime,
    Path(saved['checkpoint']),
    Path(proof['test_csv']),
    Path(proof['prediction_csv']),
    2,
    'repeat_grouped_skip_3514_gpu2',
)
first = result['posthoc_audit']['full_precision_replay']['grouped_skip']['3514']['full_precision']
formal = result['folds']['grouped_skip']['3514']
out = {
    'arm': 'grouped_skip',
    'fold': '3514',
    'formal': {key: formal[key] for key in audit.METRIC_NAMES},
    'first_replay': {key: first[key] for key in audit.METRIC_NAMES},
    'repeat_replay': {key: repeated[key] for key in audit.METRIC_NAMES},
    'repeat_replay_metadata': {key: value for key, value in repeated.items() if key not in audit.METRIC_NAMES},
}
(audit.POSTHOC / 'repeat_3514_gpu2_result.json').write_text(json.dumps(out, indent=2))
print(json.dumps({'formal_q_seq': formal['q_seq'],
                  'first_q_seq': first['q_seq'],
                  'repeat_q_seq': repeated['q_seq']}))
