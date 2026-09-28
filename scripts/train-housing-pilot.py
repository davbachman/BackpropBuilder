"""Explicit from-scratch linear regression, using the app's fitted statistics and initialization."""
import copy,json,time
from pathlib import Path
import torch
root=Path('output/housing-pilot')
if (root/'frozen.json').exists():raise RuntimeError('Study is frozen.')
source=json.loads((root/'source.json').read_text());stats=json.loads((root/'stats.json').read_text())
torch.set_num_threads(2)
x=(torch.tensor(source['x'],dtype=torch.float32)-torch.tensor(stats['mean']))/torch.tensor(stats['scale']);y=torch.tensor(source['y'],dtype=torch.float32).reshape(-1,1)
train=torch.tensor(source['train']);validation=torch.tensor(source['validation'])
def run(template):
 name=template['name'];path=root/'runs'/f'{name}.json'
 if path.exists():return json.loads(path.read_text())
 graph=template['graph'];nodes={n['id']:n for n in graph['nodes']};columns=template['features'];strength=nodes['loss']['params'].get('regularizationStrength',0)
 w=torch.tensor(nodes['output_w']['params']['value']['data'],dtype=torch.float32,requires_grad=True).reshape(-1,1).detach().requires_grad_(True);b=torch.tensor(nodes['output_b']['params']['value']['data'],dtype=torch.float32,requires_grad=True)
 xt=x[train][:,columns];xv=x[validation][:,columns];yt=y[train];yv=y[validation]
 initial=((xt[:1]@w+b-yt[:1])**2).mean()+strength*w.abs().sum()
 if abs(float(initial.detach())-template['expectedLoss'])>1e-4:raise ValueError(('initial mismatch',name,float(initial.detach()),template['expectedLoss']))
 opt=torch.optim.AdamW([w,b],lr=graph['learningRate'],weight_decay=0)
 best=float('inf');stale=0;best_epoch=0;history=[];start=time.perf_counter()
 for epoch in range(template['epochs']+1):
  with torch.no_grad():tr=float(((xt@w+b-yt)**2).mean());val=float(((xv@w+b-yv)**2).mean())
  if val<best-graph['training']['minDelta']:best=val;checkpoint=(w.detach().clone(),b.detach().clone());best_epoch=epoch;stale=0
  else:stale+=1
  history.append({'epoch':epoch,'train':tr,'validation':val})
  if stale>=graph['training']['patience'] or epoch==template['epochs']:break
  opt.zero_grad();loss=((xt@w+b-yt)**2).mean()+strength*w.abs().sum();loss.backward();opt.step()
 w,b=checkpoint
 result={'name':name,'features':columns,'strength':strength,'weights':w.flatten().tolist(),'bias':b.tolist(),'bestEpoch':best_epoch,'completed':epoch,'validation':best,'train':float(((xt@w+b-yt)**2).mean()),'seconds':time.perf_counter()-start,'history':history}
 path.write_text(json.dumps(result));print(name,round(best,5),best_epoch,[round(z,3) for z in result['weights']],flush=True);return result
if __name__=='__main__':
 for row in json.loads((root/'manifest.json').read_text()):run(json.loads((root/'templates'/(row['name']+'.json')).read_text()))
