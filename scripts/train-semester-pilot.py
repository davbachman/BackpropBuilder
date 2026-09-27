"""Train the exact exported app graphs; select by validation data loss.
Native acceleration is checked against the app's traced and browser engines.
"""
import argparse,copy,json,random,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parent/'lib'))
from curriculum_tensor_model import CurriculumModel
ROOT=Path('output/semester-pilot')
if (ROOT/'frozen.json').exists():raise RuntimeError('Pilot is frozen. Use a separate checkout/output directory for a new study.')

def train(name):
    path=ROOT/'runs'/f'{name}.json'
    if path.exists():print('EXISTS',name,flush=True);return
    template=json.loads((ROOT/'templates'/f'{name}.json').read_text());rows=json.loads((ROOT/'numeric'/f'{name}.json').read_text())
    model=CurriculumModel(template);settings=template['graph']['training'];torch.set_num_threads(4)
    features=[torch.tensor([r['features'][i]['data'] for r in rows],dtype=torch.float32).reshape(len(rows),*rows[0]['features'][i]['shape']) for i in range(len(rows[0]['features']))]
    target=torch.tensor([r['target']['data'] for r in rows],dtype=torch.float32).reshape(len(rows),*rows[0]['target']['shape'])
    splits={s:torch.tensor([i for i,r in enumerate(rows) if r['split']==s]) for s in ['train','validation']}
    task=next(n for n in template['graph']['nodes'] if n['type']=='loss')['params']['loss']
    def evaluate(ids):
        model.eval();total=correct=0
        with torch.no_grad():
            for chunk in ids.split(128):
                pred,_,loss=model([f[chunk] for f in features],target[chunk]);total+=float(loss)*len(chunk)
                if task=='cross-entropy':correct+=int((pred.argmax(-1).reshape(-1)==target[chunk].reshape(-1)).sum())
                elif task=='binary-cross-entropy':correct+=int(((pred>=.5).reshape(-1)==target[chunk].reshape(-1)).sum())
        return {'loss':total/len(ids),'accuracy':correct/(len(ids)*target[0].numel()) if task!='mse' else None,'examples':len(ids)}
    opt=torch.optim.AdamW(model.parameters(),lr=template['graph']['learningRate'],weight_decay=settings['weightDecay'])
    first=splits['train'][:1];model.eval();_,loss,_=model([f[first] for f in features],target[first]);
    if abs(float(loss.detach())-template['expectedLoss'])>1e-4:raise ValueError(('Initial loss mismatch',name,float(loss.detach()),template['expectedLoss']))
    # Save independent one-example derivatives plus three controlled updates for app parity.
    loss.backward();gradients={k:p.grad.detach().reshape(-1).tolist() for k,p in model.parameter_map.items()};initial=copy.deepcopy(model.state_dict())
    for _ in range(3):
        opt.zero_grad();_,loss,_=model([f[first] for f in features],target[first]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),settings['clipNorm']);opt.step()
    parity={'loss':template['expectedLoss'],'gradients':gradients,'weights':model.graph_weights()}
    (ROOT/'runs'/f'{name}-parity.json').write_text(json.dumps(parity))
    model.load_state_dict(initial);opt=torch.optim.AdamW(model.parameters(),lr=template['graph']['learningRate'],weight_decay=settings['weightDecay'])
    best=evaluate(splits['validation']);state=copy.deepcopy(model.state_dict());best_epoch=0;stale=0;reports=[];start=time.perf_counter()
    for epoch in range(1,template['epochs']+1):
        model.train();order=splits['train'].tolist();random.Random(42+epoch).shuffle(order)
        for offset in range(0,len(order),template['batch']):
            ids=order[offset:offset+template['batch']];opt.zero_grad();_,loss,_=model([f[ids] for f in features],target[ids]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),settings['clipNorm']);opt.step()
        val=evaluate(splits['validation']);tr=evaluate(splits['train']);reports.append({'epoch':epoch,'train':tr,'validation':val})
        if val['loss']<best['loss']-settings['minDelta']:best=val;state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
        else:stale+=1
        if epoch==1 or epoch%5==0:print(name,epoch,round(val['loss'],4),val['accuracy'],round(time.perf_counter()-start,1),flush=True)
        if stale>=settings['patience']:break
    model.load_state_dict(state)
    result={'name':name,'bestEpoch':best_epoch,'validation':best,'train':evaluate(splits['train']),'seconds':time.perf_counter()-start,'reports':reports,'weights':model.graph_weights()}
    path.write_text(json.dumps(result));print('DONE',name,best_epoch,best,flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prefix',default='');args=parser.parse_args()
    for row in json.loads((ROOT/'manifest.json').read_text()):
        if row['name'].startswith(args.prefix):train(row['name'])
