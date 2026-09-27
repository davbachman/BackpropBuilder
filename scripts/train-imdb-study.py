"""Controlled CPU PyTorch study of the same graphs used by the browser prototype.
Validation chooses epochs; fresh official-test rows are evaluated only after selection.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
from lib.text_tensor_model import TextTensorModel

parser=argparse.ArgumentParser()
parser.add_argument('--root',default='output/accelerated-imdb')
parser.add_argument('--epochs',type=int,default=12)
parser.add_argument('--seeds',default='137,211,307')
parser.add_argument('--models',default='counts-linear,mean,attention,position-attention,transformer')
args=parser.parse_args()
root=Path(args.root);(root/'runs').mkdir(exist_ok=True)
torch.set_num_threads(4)
data=json.loads((root/'numeric-data.json').read_text())
rows=data['rows'];context=data['maxLength']
ids=torch.tensor([row['ids']+[0]*(context-len(row['ids'])) for row in rows],dtype=torch.long)
lengths=torch.tensor([len(row['ids']) for row in rows]);targets=torch.tensor([row['target'] for row in rows],dtype=torch.float32)
splits={name:torch.tensor([i for i,row in enumerate(rows) if row['split']==name]) for name in ['train','validation','final-test']}
def batch(indices):
    count=int(lengths[indices].max())
    return ids[indices,:count],lengths[indices],targets[indices]
@torch.no_grad()
def evaluate(model,indices):
    model.eval();total=0.;hits=0
    for group in indices.split(128):
        prediction,loss=model(*batch(group));total+=float(loss)*len(group);hits+=int(((prediction>=.5)==targets[group]).sum())
    return {'loss':total/len(indices),'accuracy':hits/len(indices),'examples':len(indices)}
def order(epoch):
    # Match the app's seeded Fisher-Yates shuffle; architecture and initialization do not change data order.
    values=splits['train'].tolist();seed=42+epoch
    for i in range(len(values)-1,0,-1):
        seed=(1664525*seed+1013904223)&0xffffffff;j=int(seed/4294967296*(i+1));values[i],values[j]=values[j],values[i]
    return torch.tensor(values)
for seed in map(int,args.seeds.split(',')):
 for kind in args.models.split(','):
    template=json.loads((root/'templates'/f'{kind}-{seed}.json').read_text())
    model=TextTensorModel(template)
    _,initial=model(*batch(torch.tensor([0])))
    if abs(float(initial.detach())-template['expectedLoss'])>2e-5: raise ValueError('App/PyTorch initial loss mismatch')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01,betas=(.9,.999),eps=1e-8)
    start=time.perf_counter();best=evaluate(model,splits['validation']);best_state=copy.deepcopy(model.state_dict());best_epoch=0;stale=0
    reports=[{'epoch':0,'validation':best}]
    for epoch in range(1,args.epochs+1):
        model.train();online=0.
        for group in order(epoch-1).split(32):
            optimizer.zero_grad();_,loss=model(*batch(group));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();online+=float(loss.detach())*len(group)
        val=evaluate(model,splits['validation']);improved=val['loss']<best['loss']-.001
        if improved: best=val;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
        else: stale+=1
        reports.append({'epoch':epoch,'onlineTrainingLoss':online/len(splits['train']),'validation':val,'improved':improved})
        print(kind,seed,epoch,'train',round(online/len(splits['train']),4),'val',round(val['loss'],4),'accuracy',round(val['accuracy'],4),'seconds',round(time.perf_counter()-start,1),flush=True)
        if stale>=3:break
    model.load_state_dict(best_state)
    result={'kind':kind,'seed':seed,'backend':'PyTorch CPU','learningRate':.0003,'optimizer':'AdamW','weightDecay':.01,'batchSize':32,'clipNorm':1,'patience':3,'minDelta':.001,'maxEpochs':args.epochs,'bestEpoch':best_epoch,'completed':epoch,'seconds':time.perf_counter()-start,'train':evaluate(model,splits['train']),'validation':best,'test':evaluate(model,splits['final-test']),'reports':reports}
    (root/'runs'/f'{kind}-{seed}-results.json').write_text(json.dumps(result,indent=2))
    (root/'runs'/f'{kind}-{seed}-weights.json').write_text(json.dumps(model.graph_weights()))
    print('FINISHED',kind,seed,'best',best_epoch,'test',result['test'],flush=True)
