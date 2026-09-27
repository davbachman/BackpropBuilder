import hashlib,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).parent/'lib'))
from curriculum_tensor_model import CurriculumModel
r=Path('output/semester-pilot');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();torch.set_num_threads(4)
if (r/'final-results.json').exists():raise SystemExit('Final results already exist. Do not retune against these examples.')
frozen=json.loads((r/'frozen.json').read_text());results=[]
for job in frozen['jobs']:
 name=job['name'];assert sha(r/'templates'/(name+'.json'))==job['templateSha256'];assert sha(r/'runs'/(name+'.json'))==job['weightsSha256']
 template=json.loads((r/'templates'/(name+'.json')).read_text());run=json.loads((r/'runs'/(name+'.json')).read_text());rows=json.loads((r/'final'/(name+'.json')).read_text());model=CurriculumModel(template)
 for key,p in model.parameter_map.items():p.data.copy_(torch.tensor(run['weights'][key]['data']).reshape(p.shape))
 model.eval();loss_sum=correct=total=last_sum=last_correct=0;predictions=[]
 kind=next(n for n in template['graph']['nodes'] if n['type']=='loss')['params']['loss']
 with torch.no_grad():
  for start in range(0,len(rows),128):
   batch=rows[start:start+128];features=[torch.tensor([x['features'][i]['data'] for x in batch],dtype=torch.float32).reshape(len(batch),*batch[0]['features'][i]['shape']) for i in range(len(batch[0]['features']))];target=torch.tensor([x['target']['data'] for x in batch],dtype=torch.float32).reshape(len(batch),*batch[0]['target']['shape'])
   pred,_,loss=model(features,target);loss_sum+=float(loss)*len(batch)
   if kind=='cross-entropy':
    ids=pred.argmax(-1).reshape(target.shape);correct+=int((ids==target).sum());total+=target.numel();predictions.extend(ids.tolist())
    last_sum+=float(F.cross_entropy(pred[:,-1,:],target.reshape(len(batch),-1)[:,-1].long()))*len(batch);last_correct+=int((pred[:,-1,:].argmax(-1)==target.reshape(len(batch),-1)[:,-1]).sum())
   elif kind=='binary-cross-entropy':correct+=int(((pred>=.5).reshape(target.shape)==target).sum());total+=target.numel();predictions.extend(pred.reshape(-1).tolist())
   else:predictions.extend(pred.reshape(-1).tolist())
 result={'name':name,'loss':loss_sum/len(rows),'accuracy':correct/total if total else None,'examples':len(rows),'dataSha256':sha(r/'final'/(name+'.json'))}
 if name.startswith('alice'):result.update({'lastPositionLoss':last_sum/len(rows),'lastPositionAccuracy':last_correct/len(rows)})
 if kind=='mse':
  scale=json.loads((r/'data'/('mpg.json' if name.startswith('mpg') else 'concrete.json')).read_text())['scaling']['targetScale']
  if name.startswith('mpg-small'):scale*=json.loads((r/'data/mpg-small-scaling.json').read_text())['targetScale']
  result['rmseOriginalUnits']=(result['loss']**.5)*scale
 results.append(result);print(name,result,flush=True)
(r/'final-results.json').write_text(json.dumps({'frozenSha256':sha(r/'frozen.json'),'results':results},indent=2))
