"""Freeze validation-selected recipes before evaluating the held-out final partition."""
from pathlib import Path
import hashlib,json,math,statistics
import numpy as np
root=Path('output/housing-pilot')
if (root/'frozen.json').exists() or (root/'final-results.json').exists():raise RuntimeError('Final study already exists.')
names=[f'{kind}-{seed}' for kind in ['all-l1-0','all-l1-0.05','top2','top3'] for seed in [137,211,307]]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
frozen={'selection':json.loads((root/'selection.json').read_text()),'sourceSha256':sha(root/'source.json'),'jobs':[{'name':n,'templateSha256':sha(root/'templates'/f'{n}.json'),'runSha256':sha(root/'runs'/f'{n}.json')} for n in names]}
(root/'frozen.json').write_text(json.dumps(frozen,indent=2))
source=json.loads((root/'source.json').read_text());stats=json.loads((root/'stats.json').read_text());ids=source['final']
x=(np.asarray(source['x'])[ids]-stats['mean'])/stats['scale'];y=np.asarray(source['y'])[ids]
results=[]
for name in names:
 run=json.loads((root/'runs'/f'{name}.json').read_text());pred=x[:,run['features']]@run['weights']+run['bias'][0];mse=float(np.mean((pred-y)**2))
 results.append({'name':name,'examples':len(ids),'mse':mse,'rmseDollars':math.sqrt(mse)*100000,'maeDollars':float(np.mean(abs(pred-y)))*100000})
(root/'final-results.json').write_text(json.dumps({'frozenSha256':sha(root/'frozen.json'),'results':results},indent=2))
for kind in ['all-l1-0','all-l1-0.05','top2','top3']:
 rows=[r for r in results if r['name'].rsplit('-',1)[0]==kind];print(kind,statistics.mean(r['mse'] for r in rows),statistics.mean(r['rmseDollars'] for r in rows))
