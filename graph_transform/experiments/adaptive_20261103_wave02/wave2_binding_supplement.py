"""Second-wave single-factor, reference, data and terminal bindings."""
import copy,hashlib,json
from pathlib import Path
import torch,yaml
OPS=Path('/mnt/pvc/graphtrans/dbond_adaptive_regularization_wave2_20261103_ops')
OUT=OPS/'completed_reviews/complete_wave2'
PREVIOUS=Path('/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/adaptive_research_batch/20261102_155846')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
d=json.loads((OUT/'result.json').read_text());launch=json.loads((OPS/'launch.json').read_text());smoke=json.loads((OPS/'smoke.json').read_text());root=Path(d['root'])
assert root==Path(launch['root']) and d['gitrev']==launch['gitrev'] and sha(OPS/'smoke.json')==launch['smoke_sha256']
reuse=json.loads((OPS/'reference_reuse_receipt.json').read_text());assert reuse['status']=='passed' and sha(OPS/'reference_reuse_receipt.json')==launch['reference_reuse_sha256']
PATCHES={'full':{},'lr_3e4':{'training.learning_rate':.0003},'dropout_03':{'model.dropout':.3},'weight_decay_1e3':{'training.weight_decay':.001},'hidden128_k5':{'model.hidden_dim':128}}
proof={}
for arm in d['arms']:
    proof[arm]={}
    for fold in ('1222','2252','3514','6072','9075'):
        tr=(PREVIOUS if arm=='full' else root)/arm/f'fold_{fold}'
        cfg=yaml.safe_load((tr/'config.yaml').read_text())
        control=yaml.safe_load((PREVIOUS/'full'/f'fold_{fold}/config.yaml').read_text());normalized=copy.deepcopy(cfg)
        for dotted,expected in PATCHES[arm].items():
            section,key=dotted.split('.');assert cfg[section][key]==expected
            normalized[section][key]=control[section][key]
        normalized=json.dumps(normalized,sort_keys=True).replace('adaptive_'+arm,'adaptive_full').replace('/'+arm+'/','/full/').replace(str(root),str(PREVIOUS))
        assert normalized==json.dumps(control,sort_keys=True),'Undeclared configuration difference'
        assert cfg['model']['num_attention_heads']==4 and cfg['model']['node_conv_mode']=='single_k5' and cfg['model']['bond_head_type']=='four_state_mlp'
        manifest=Path(cfg['data']['four_state_targets_manifest']);target=json.loads(manifest.read_text());sidecar=manifest.parent/target['sidecar']['path']
        assert sha(manifest)==smoke['folds'][fold]['four_state_manifest_sha256'] and sha(sidecar)==target['sidecar']['sha256']
        assert sha(cfg['data']['train_csv_path'])==target['source_folded_train']['sha256']
        ck=d['provenance'][arm]['folds'][fold]['max_f1']['checkpoint'];saved=torch.load(ck,map_location='cpu',weights_only=False)
        split=Path(saved['extra_state']['row_validation_provenance']['split_train_csv'])
        assert sha(split)==smoke['folds'][fold]['train_sha256']
        proof[arm][fold]=dict(reference_reused=arm=='full',single_factor_only=True,training_split_sha256=sha(split),target_sidecar_sha256=sha(sidecar))
terminal=json.loads((OPS/'terminal.json').read_text())
assert terminal['status']=='completed' and terminal['root']==str(root) and terminal['gitrev']==d['gitrev']
assert sha(OPS/'result.json')==terminal['result_sha256']
receipt=dict(status='passed',result_sha256=sha(OUT/'result.json'),original_result_sha256=sha(OPS/'result.json'),terminal_sha256=sha(OPS/'terminal.json'),script_sha256=sha(Path(__file__)),bindings=proof)
(OUT/'source_binding_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(dict(status='passed',arms=d['arms'])))
