import json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parent/'lib'))
from curriculum_tensor_model import CurriculumModel
root=Path('output/semester-pilot');torch.set_num_threads(4)
names=[r['name'] for r in json.loads((root/'manifest.json').read_text())] if sys.argv[1]=='--all' else [sys.argv[1]]
for name in names:
 t=json.loads((root/'templates'/f'{name}.json').read_text());rows=json.loads((root/'numeric'/f'{name}.json').read_text());r=next(r for r in rows if r['split']=='train')
 m=CurriculumModel(t);features=[torch.tensor(f['data'],dtype=torch.float32).reshape(1,*f['shape']) for f in r['features']];target=torch.tensor(r['target']['data'],dtype=torch.float32).reshape(1,*r['target']['shape']);opt=torch.optim.AdamW(m.parameters(),lr=t['graph']['learningRate'],weight_decay=t['graph']['training']['weightDecay']);steps=[]
 for i in range(4):
  opt.zero_grad();p,l,_=m(features,target);l.backward();steps.append({'loss':float(l.detach()),'prediction':p.detach().reshape(-1).tolist(),'gradients':{k:v.grad.reshape(-1).tolist() for k,v in m.parameter_map.items()}})
  if i<3:torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
 (root/'runs'/f'{name}-steps.json').write_text(json.dumps({'steps':steps,'weights':m.graph_weights()}))
