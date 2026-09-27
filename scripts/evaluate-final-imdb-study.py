"""Evaluate frozen checkpoints exactly once; never train or select using test rows."""
import argparse,hashlib,json
from pathlib import Path
import torch
from lib.text_tensor_model import TextTensorModel
parser=argparse.ArgumentParser();parser.add_argument('--root',default='output/imdb-final');root=Path(parser.parse_args().root)
sha=lambda b:hashlib.sha256(b).hexdigest()
frozen_raw=(root/'frozen.json').read_bytes();frozen=json.loads(frozen_raw)
manifest=json.loads((root/'holdout-manifest.json').read_text());assert sha(frozen_raw)==manifest['frozenSha256']
raw=(root/'holdout-numeric.json').read_bytes();assert sha(raw)==manifest['numericSha256']
assert not (root/'test-results.json').exists(),'Test already evaluated; do not retune or replace results.'
# Check all checkpoint hashes before evaluating any one of them.
for job in frozen['jobs']:
    assert sha((root/'templates'/f"{job['name']}.json").read_bytes())==job['templateSha256']
    assert sha((root/'runs'/f"{job['name']}-weights.json").read_bytes())==job['weightsSha256']
torch.set_num_threads(4)
rows=json.loads(raw)['rows'];lengths=torch.tensor([len(r['ids']) for r in rows]);context=frozen['protocol']['context']
ids=torch.tensor([r['ids']+[0]*(context-len(r['ids'])) for r in rows]);targets=torch.tensor([r['target'] for r in rows],dtype=torch.float32)
results=[]
with torch.no_grad():
 for job in frozen['jobs']:
    template=json.loads((root/'templates'/f"{job['name']}.json").read_text());weights=json.loads((root/'runs'/f"{job['name']}-weights.json").read_text())
    for node in template['graph']['nodes']:
        if node['id'] in weights:node['params']['value']=weights[node['id']]
    model=TextTensorModel(template);model.eval();total=0.;probabilities=[]
    for start in range(0,len(rows),128):
        stop=min(start+128,len(rows));prediction,loss=model(ids[start:stop,:int(lengths[start:stop].max())],lengths[start:stop],targets[start:stop])
        assert bool(torch.isfinite(loss)) and bool(torch.isfinite(prediction).all())
        total+=float(loss)*(stop-start);probabilities.extend(prediction.tolist())
    hits=sum((p>=.5)==bool(r['target']) for p,r in zip(probabilities,rows))
    result={**job,'test':{'loss':total/len(rows),'accuracy':hits/len(rows),'examples':len(rows)},'probabilities':probabilities}
    results.append(result);print(job['name'],result['test'],flush=True)
(root/'test-results.json').write_text(json.dumps({'frozenSha256':sha(frozen_raw),'holdoutSha256':sha(raw),'results':results},indent=2))
