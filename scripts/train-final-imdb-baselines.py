"""Train matched baselines without accessing any final-test reviews."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import torch
from lib.text_tensor_model import TextTensorModel

parser=argparse.ArgumentParser()
parser.add_argument('--root',default='output/imdb-final')
args=parser.parse_args();root=Path(args.root);(root/'runs').mkdir(exist_ok=True)
protocol=json.loads((root/'protocol.json').read_text())
torch.set_num_threads(4)

def run_setting(size,vocabulary,kind):
    key=f'n{size}-v{vocabulary}'
    raw=(root/'data'/f'{key}-numeric.json').read_bytes()
    stats=json.loads((root/'data'/f'{key}-stats.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==stats['numericSha256']
    rows=json.loads(raw)['rows'];context=protocol['context']
    assert all(r['split'] in ('train','validation') for r in rows)
    ids=torch.tensor([r['ids']+[0]*(context-len(r['ids'])) for r in rows],dtype=torch.long)
    lengths=torch.tensor([len(r['ids']) for r in rows]);targets=torch.tensor([r['target'] for r in rows],dtype=torch.float32)
    splits={name:torch.tensor([i for i,r in enumerate(rows) if r['split']==name]) for name in ('train','validation')}
    assert len(splits['train'])==size and len(splits['validation'])==protocol['validation']
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
        values=splits['train'].tolist();state=42+epoch
        for i in range(len(values)-1,0,-1):
            state=(1664525*state+1013904223)&0xffffffff;j=int(state/4294967296*(i+1));values[i],values[j]=values[j],values[i]
        return torch.tensor(values)
    for seed in protocol['seeds']:
        name=f'{kind}-seed{seed}'
        torch.manual_seed(104729+seed)
        template=json.loads((root/'templates'/f'{name}.json').read_text())
        model=TextTensorModel(template);model.eval()
        _,initial=model(*batch(torch.tensor([0])))
        if abs(float(initial.detach())-template['expectedLoss'])>2e-5:raise ValueError('Initial graph mismatch '+name)
        optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learningRate'],weight_decay=protocol['weightDecay'],betas=(.9,.999),eps=1e-8)
        start=time.perf_counter();best=evaluate(model,splits['validation']);best_state=copy.deepcopy(model.state_dict());best_epoch=0;stale=0;updates=0
        reports=[{'epoch':0,'validation':best}]
        for epoch in range(1,protocol['maxEpochs']+1):
            model.train();online=0.
            for group in order(epoch-1).split(protocol['batchSize']):
                optimizer.zero_grad();_,loss=model(*batch(group))
                if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss '+name)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['clipNorm']);optimizer.step();updates+=1
                online+=float(loss.detach())*len(group)
            val=evaluate(model,splits['validation']);improved=val['loss']<best['loss']-protocol['minDelta']
            if improved:best=val;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
            else:stale+=1
            reports.append({'epoch':epoch,'updates':updates,'onlineTrainingLoss':online/size,'validation':val,'improved':improved})
            print(name,epoch,'validation',round(val['loss'],4),round(val['accuracy'],4),flush=True)
            if stale>=protocol['patience']:break
        model.load_state_dict(best_state)
        result={'kind':kind,'name':name,'trainingSize':size,'vocabularySize':vocabulary,'seed':seed,'backend':'PyTorch CPU','torchVersion':torch.__version__,'numericSha256':stats['numericSha256'],'parameterCount':sum(p.numel() for p in model.parameters()),'bestEpoch':best_epoch,'completed':epoch,'updates':updates,'seconds':time.perf_counter()-start,'train':evaluate(model,splits['train']),'validation':best,'reports':reports,'finalTestUsed':False}
        (root/'runs'/f'{name}-results.json').write_text(json.dumps(result,indent=2))
        (root/'runs'/f'{name}-weights.json').write_text(json.dumps(model.graph_weights()))
        print('FINISHED',name,best,'seconds',round(result['seconds'],1),flush=True)

for kind in protocol['models']:
    if kind != 'transformer-two-head':run_setting(20000,4000,kind)
