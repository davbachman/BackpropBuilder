"""Freeze validation-selected comparisons, then evaluate untouched test stories."""
import collections,hashlib,importlib.util,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
spec=importlib.util.spec_from_file_location('pilot',Path(__file__).with_name('pilot-babi.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
root=m.ROOT;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
selection={0:['counts-mlp','mean','ordered-mlp'],1:['counts-mlp','mean','ordered-mlp','sentence-mlp','memory-1'],4:['counts-mlp','ordered-mlp','transformer-1','transformer-2'],2:['memory-1','memory-2'],100:['memory-1','memory-2','memory-2:assisted','retrieval-1:extended','retrieval-2:extended','retrieval-3:extended']}
jobs=[]
for task,kinds in selection.items():
 for entry in kinds:
  kind,_,tag=entry.partition(':')
  for seed in [137,211,307]:
   name=f'qa{task}-{kind}-seed{seed}'+('-'+tag if tag else '')
   result=json.loads((root/'runs'/f'{name}.json').read_text());assert not result['testUsed']
   assert result['dataSha256']==sha(root/'data'/f'qa{task}.json')
   jobs.append({'name':name,'task':task,'kind':kind,'tag':tag,'seed':seed,'validation':result['validation'],'weightsSha256':sha(root/'runs'/f'{name}.pt'),'resultSha256':sha(root/'runs'/f'{name}.json'),'dataSha256':result['dataSha256']})
frozen={'timestamp':time.time(),'jobs':jobs,'sourceSha256':sha(Path(__file__).with_name('pilot-babi.py')),'selection':'Chosen on validation: confirm the easy order bridge, question-directed attention on qa1, qa4 transformer reliability, and the harder two-fact cases. All three seeds included; no selection by official test.','limitations':'Sentence-boundary preprocessing is shared by sentence-mlp and memory models. Assisted models additionally use supporting-fact training labels; never provided at inference. Extended models receive 100 epochs rather than 30.'}
assert not (root/'frozen.json').exists(),'Already frozen/evaluated; do not overwrite.'
with (root/'frozen.json').open('x') as f:json.dump(frozen,f,indent=2)
torch.set_num_threads(4);all_results=[];manifests={};errors={}
with torch.no_grad():
 for task in selection:
  data=json.loads((root/'data'/f'qa{task}.json').read_text());index={t:i for i,t in enumerate(data['vocabulary'])}
  if task in (0,100):raw=json.loads((root/'data'/f'qa{task}-test.json').read_text())
  else:raw=m.parse(root/'raw'/f'qa{task}_{m.TASKS[task]}_test.txt')
  rows=[r for r in raw if len(r['tokens'])<=data['length']]
  for r in rows:r['ids']=[index.get(t,1) for t in r['tokens']];r['target']=data['classes'].index(r['answer'])
  train_signatures={tuple(r['tokens']) for r in data['rows'] if r['split']=='train'}
  validation_signatures={tuple(r['tokens']) for r in data['rows'] if r['split']=='validation'}
  novel=[tuple(r['tokens']) not in train_signatures|validation_signatures for r in rows]
  manifests[task]={'examples':len(rows),'excludedTooLong':len(raw)-len(rows),'maxTokens':data['length'],'novelExamples':sum(novel),'exactTrainingDuplicates':sum(tuple(r['tokens']) in train_signatures for r in rows),'novelDefinition':'Full token sequence absent from both training and validation; story-group splits are also preserved.','sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()}
  ids=torch.tensor([r['ids']+[0]*(data['length']-len(r['ids'])) for r in rows]);targets=torch.tensor([r['target'] for r in rows])
  encode=lambda text:[index.get(t,1) for t in m.tokens(text)]
  assert all(len(encode(f))<=10 for r in rows for f in r['facts']+[r['question']])
  pad=lambda text:encode(text)+[0]*(10-len(encode(text)))
  assert max(len(r['facts']) for r in rows)<=64
  facts=torch.tensor([[pad(f) for f in r['facts']]+[[0]*10]*(64-len(r['facts'])) for r in rows]);questions=torch.tensor([pad(r['question']) for r in rows])
  for job in [j for j in jobs if j['task']==task]:
   assert sha(root/'runs'/f"{job['name']}.pt")==job['weightsSha256']
   model=m.PilotModel(job['kind'],len(data['vocabulary']),data['length'],len(data['classes']),question_id=index['<question>'],answer_ids=[index[c] for c in data['classes']])
   model.load_state_dict(torch.load(root/'runs'/f"{job['name']}.pt",weights_only=True));model.eval();predictions=[];total=0.;attention_correct=collections.Counter();attention_total=0
   for start in range(0,len(rows),128):
    stop=min(start+128,len(rows));x=ids[start:stop];x=x[:,:int(x.ne(0).sum(1).max())];f=facts[start:stop];f=f[:,:int(f.ne(0).any(-1).sum(-1).max())]
    logits=model(x,f,questions[start:stop]);assert torch.isfinite(logits).all();total+=float(F.cross_entropy(logits,targets[start:stop]))*(stop-start);predictions.extend(logits.argmax(-1).tolist())
    if task==100 and job['kind']=='memory-2':
     # Score factual retrieval after prediction, never feed these annotations to the model.
     for j,r in enumerate(rows[start:stop]):
      obj=m.tokens(r['question'])[-2];ownerfact=next(i for i,f in enumerate(r['facts']) if obj in m.tokens(f) and 'picked' in m.tokens(f));owner=m.tokens(r['facts'][ownerfact])[0];locationfact=next(i for i,f in enumerate(r['facts']) if m.tokens(f)[0]==owner and 'moved' in m.tokens(f))
      for h,label in enumerate([ownerfact,locationfact]):attention_correct[h]+=int(model.attention_scores[h][j].argmax()==label)
      attention_total+=1
   correct=[p==r['target'] for p,r in zip(predictions,rows)]
   result={**job,'test':{'accuracy':sum(correct)/len(rows),'loss':total/len(rows),'examples':len(rows),'novelAccuracy':sum(c for c,n in zip(correct,novel) if n)/sum(novel) if any(novel) else None,'novelExamples':sum(novel)},'predictions':predictions}
   if attention_total:result['supportAccuracy']={str(h):v/attention_total for h,v in attention_correct.items()}
   all_results.append(result)
   errors[job['name']]=[{'facts':r['facts'],'question':r['question'],'answer':r['answer'],'prediction':data['classes'][p]} for p,r in zip(predictions,rows) if p!=r['target']][:5]
   print(job['name'],result['test'],flush=True)
(root/'test-results.json').write_text(json.dumps({'frozenSha256':sha(root/'frozen.json'),'manifests':manifests,'results':all_results},indent=2));(root/'error-examples.json').write_text(json.dumps(errors,indent=2))
