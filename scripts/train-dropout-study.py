"""Dropout/learning-rate search. Selection uses validation only; never evaluates final-test rows."""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
from lib.text_tensor_model import TextTensorModel

parser=argparse.ArgumentParser()
parser.add_argument('--source',default='output/accelerated-imdb')
parser.add_argument('--root',default='output/imdb-dropout')
parser.add_argument('--confirmation-only',action='store_true',help='Extend an existing screening run with missing confirmation controls.')
args=parser.parse_args()
root=Path(args.root);(root/'runs').mkdir(exist_ok=True)
protocol=json.loads((root/'protocol.json').read_text())
torch.set_num_threads(4)
data=json.loads((Path(args.source)/'numeric-data.json').read_text())
# No final-test inputs, labels, metrics, or model selection enter this experiment.
rows=[r for r in data['rows'] if r['split'] in ('train','validation')]
context=data['maxLength']
ids=torch.tensor([r['ids']+[0]*(context-len(r['ids'])) for r in rows],dtype=torch.long)
lengths=torch.tensor([len(r['ids']) for r in rows]);targets=torch.tensor([r['target'] for r in rows],dtype=torch.float32)
splits={name:torch.tensor([i for i,r in enumerate(rows) if r['split']==name]) for name in ('train','validation')}
assert len(splits['train'])==protocol['training'] and len(splits['validation'])==protocol['validation']
def batch(indices):
    return ids[indices,:int(lengths[indices].max())],lengths[indices],targets[indices]
@torch.no_grad()
def evaluate(model,indices):
    model.eval();total=0.;hits=0
    for group in indices.split(128):
        prediction,loss=model(*batch(group));total+=float(loss)*len(group)
        hits+=int(((prediction>=.5)==targets[group]).sum())
    return {'loss':total/len(indices),'accuracy':hits/len(indices),'examples':len(indices)}
def order(epoch):
    values=splits['train'].tolist();seed=42+epoch
    for i in range(len(values)-1,0,-1):
        seed=(1664525*seed+1013904223)&0xffffffff;j=int(seed/4294967296*(i+1));values[i],values[j]=values[j],values[i]
    return torch.tensor(values)
def run(seed,rate,lr):
    name=f'seed{seed}-p{rate:g}-lr{lr:g}'
    torch.manual_seed(104729+seed)
    template=json.loads((root/'templates'/f'transformer-{seed}-p{rate:g}.json').read_text())
    model=TextTensorModel(template);model.eval()
    _,initial=model(*batch(torch.tensor([0])))
    if abs(float(initial.detach())-template['expectedLoss'])>2e-5:raise ValueError('Initial graph mismatch')
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=protocol['weightDecay'],betas=(.9,.999),eps=1e-8)
    start=time.perf_counter();best=evaluate(model,splits['validation']);best_state=copy.deepcopy(model.state_dict());best_epoch=0;stale=0
    reports=[{'epoch':0,'validation':best}]
    for epoch in range(1,protocol['maxEpochs']+1):
        model.train();online=0.
        for group in order(epoch-1).split(protocol['batchSize']):
            optimizer.zero_grad();_,loss=model(*batch(group))
            if not bool(torch.isfinite(loss)): raise ValueError('Nonfinite training loss '+name)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['clipNorm']);optimizer.step()
            online+=float(loss.detach())*len(group)
        val=evaluate(model,splits['validation']);improved=val['loss']<best['loss']-protocol['minDelta']
        if improved:best=val;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
        else:stale+=1
        reports.append({'epoch':epoch,'onlineTrainingLossWithDropout':online/len(splits['train']),'validation':val,'improved':improved})
        print(name,epoch,'val',round(val['loss'],4),'accuracy',round(val['accuracy'],4),flush=True)
        if stale>=protocol['patience']:break
    model.load_state_dict(best_state)
    result={'name':name,'seed':seed,'dropoutRate':rate,'learningRate':lr,'backend':'PyTorch CPU','torchVersion':torch.__version__,'dropoutSeed':104729+seed,'bestEpoch':best_epoch,'completed':epoch,'seconds':time.perf_counter()-start,'train':evaluate(model,splits['train']),'validation':best,'reports':reports,'finalTestUsed':False}
    (root/'runs'/f'{name}-results.json').write_text(json.dumps(result,indent=2))
    (root/'runs'/f'{name}-weights.json').write_text(json.dumps(model.graph_weights()))
    print('FINISHED',name,'validation',best,'seconds',round(result['seconds'],1),flush=True)
    return result
screen=[json.loads((root/'runs'/f"seed{protocol['screeningSeed']}-p{p:g}-lr{lr:g}-results.json").read_text()) if args.confirmation_only else run(protocol['screeningSeed'],p,lr) for p in protocol['rates'] for lr in protocol['learningRates']]
ranked=sorted(screen,key=lambda r:(r['validation']['loss'],r['dropoutRate'],r['learningRate']))
selected=[(r['dropoutRate'],r['learningRate']) for r in ranked[:2]]
baseline=(0,.0003)
# Match the selected learning rate without dropout to separate the two effects.
confirmation=list(dict.fromkeys(selected+[baseline]+[(0,lr) for _,lr in selected]))
(root/'selection.json').write_text(json.dumps({'screeningRanking':[r['name'] for r in ranked],'selected':selected,'confirmationIncludingBaseline':confirmation,'criterion':'lowest validation loss','finalTestUsed':False},indent=2))
for seed in protocol['confirmationSeeds']:
    for p,lr in confirmation:
        if not args.confirmation_only or not (root/'runs'/f'seed{seed}-p{p:g}-lr{lr:g}-results.json').exists():run(seed,p,lr)
