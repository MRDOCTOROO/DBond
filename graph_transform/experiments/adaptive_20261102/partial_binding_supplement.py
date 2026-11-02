"""Supplement source/config/target bindings for completed arms only."""
import copy,hashlib,json
from pathlib import Path
import torch,yaml
OPS=Path('/mnt/pvc/graphtrans/dbond_adaptive_research_20261102_ops')
OUT=OPS/'completed_reviews/first_three_arms'
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
d=json.loads((OUT/'result.json').read_text());smoke=json.loads((OPS/'smoke.json').read_text());root=Path(d['root'])
proof={}
for arm in d['arms']:
    proof[arm]={}
    for fold in ('1222','2252','3514','6072','9075'):
        cfg=yaml.safe_load((root/arm/f'fold_{fold}/config.yaml').read_text())
        control=yaml.safe_load((root/'full'/f'fold_{fold}/config.yaml').read_text())
        normalized=copy.deepcopy(cfg)
        if arm=='no_theory':
            assert cfg['model']['use_theory_features'] is False
            normalized['model']['use_theory_features']=control['model']['use_theory_features']
        if arm=='no_message_passing':
            assert cfg['model']['num_gcn_layers']==cfg['model']['num_gat_layers']==0
            for key in ('num_gcn_layers','num_gat_layers'):normalized['model'][key]=control['model'][key]
        a=json.dumps(normalized,sort_keys=True).replace('adaptive_'+arm,'adaptive_full').replace('/'+arm+'/','/full/')
        assert a==json.dumps(control,sort_keys=True),'Undeclared protocol difference'
        manifest=Path(cfg['data']['four_state_targets_manifest']);target=json.loads(manifest.read_text())
        sidecar=manifest.parent/target['sidecar']['path']
        assert sha(manifest)==smoke['folds'][fold]['four_state_manifest_sha256']
        assert sha(sidecar)==target['sidecar']['sha256']
        assert sha(cfg['data']['train_csv_path'])==target['source_folded_train']['sha256']
        p=d['provenance'][arm]['folds'][fold]['max_f1']['checkpoint']
        saved=torch.load(p,map_location='cpu',weights_only=False)
        split=Path(saved['extra_state']['row_validation_provenance']['split_train_csv'])
        assert sha(split)==smoke['folds'][fold]['train_sha256']
        proof[arm][fold]=dict(declared_config_changes_only=True,target_manifest_sha256=sha(manifest),
            target_sidecar_sha256=sha(sidecar),training_split_sha256=sha(split),outer_train_sha256=sha(cfg['data']['train_csv_path']))
receipt=dict(status='passed',result_sha256=sha(OUT/'result.json'),script_sha256=sha(Path(__file__)),bindings=proof)
(OUT/'source_binding_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(dict(status='passed',arms=d['arms'])))
