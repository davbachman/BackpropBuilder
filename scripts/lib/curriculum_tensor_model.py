"""Independent fixed-shape interpreter of editable app graphs for pilot studies."""
import torch
from torch import nn
from torch.nn import functional as F
from text_tensor_model import TextTensorModel

class CurriculumModel(TextTensorModel):
    def __init__(self,template):
        # Parent requires textData but the interpreter also handles numeric CSVs.
        source=next(n for n in template['graph']['nodes'] if n['type']=='dataset')
        source['params'].setdefault('textData',{})
        super().__init__(template)
    def forward(self,features,target):
        values={};batch=target.shape[0]
        target_slot=self.source['params'].get('customCsv',{}).get('targetColumn',len(features))
        signals=[(x,['batch']+['feature']*(x.ndim-1)) for x in features]
        signals.insert(target_slot,(target,['batch']+['feature']*(target.ndim-1)))
        def read(e):return signals[e.get('sourceSlot',0)] if e['source']==self.source['id'] else values[e['source']]
        prediction=loss=data_loss=None
        for key in self.order:
            n=self.nodes[key];p=n['params'];kind=p.get('transform') if n['type']=='tensor-transform' else n['type']
            if kind=='dataset':continue
            args=[read(e) for e in self.inputs[key]];a=[x[0] for x in args]
            axes=next((x[1] for x in args if 'batch' in x[1]),args[0][1] if args else [])
            if kind in ('weight','bias'):v=self.parameter_map[key];axes=['feature']*v.ndim
            elif kind in ('input','target'):v=a[0] if a else torch.tensor(p['value']['data']).reshape(p['value']['shape'])
            elif kind=='standardize':v=(a[0]-torch.tensor(p['standardization']['mean'],dtype=a[0].dtype))/torch.tensor(p['standardization']['scale'],dtype=a[0].dtype)
            elif kind=='embedding':v=F.embedding(a[1].long(),a[0]);axes=args[1][1]+['feature']
            elif kind=='matmul':v=a[0]@a[1];axes=args[0][1][:-1]+args[1][1][-1:]
            elif kind=='arithmetic':v=self.arithmetic(self.expressions[key],a)
            elif kind=='add':v=sum(a)
            elif kind=='multiply':v=a[0]*a[1]
            elif kind=='activation':v={'relu':F.relu,'sigmoid':torch.sigmoid,'tanh':torch.tanh,'identity':lambda x:x}[p.get('activation','identity')](a[0])
            elif kind=='reshape':
                batched=args[0][1][0]=='batch';shape=([batch] if batched else [])+p['shape'];v=a[0].reshape(shape);axes=(['batch'] if batched else [])+['feature']*len(p['shape'])
            elif kind=='slice':
                axis=p.get('axis',0)+int(args[0][1][0]=='batch');v=a[0].narrow(axis,p.get('start',0),p['end']-p.get('start',0))
            elif kind=='transpose':
                batched=args[0][1][0]=='batch';perm=p['axes'];perm=[0]+[i+1 for i in perm] if batched else perm;v=a[0].permute(perm);axes=[args[0][1][i] for i in perm]
            elif kind=='mean':
                axis=p.get('axis',0)+int(args[0][1][0]=='batch');v=a[0].mean(axis,keepdim=p.get('keepDims',False));axes=list(axes)
                if not p.get('keepDims',False):axes.pop(axis)
            elif kind=='concat':v=torch.cat(a,dim=p.get('axis',1)+int(args[0][1][0]=='batch'))
            elif kind=='softmax':v=a[0].softmax(-1)
            elif kind=='causal-mask':v=a[0].masked_fill(torch.triu(torch.ones_like(a[0],dtype=torch.bool),diagonal=1),-1e9)
            elif kind=='layer-norm':v=F.layer_norm(a[0],(a[0].shape[-1],),a[1],a[2],p.get('epsilon',1e-5))
            elif kind=='dropout':v=F.dropout(a[0],p.get('dropoutRate',.1),self.training)
            elif kind=='loss':
                prediction=a[0];actual=a[1]
                if p['loss']=='cross-entropy':v=F.cross_entropy(prediction.reshape(-1,prediction.shape[-1]),actual.reshape(-1).long())
                elif p['loss']=='binary-cross-entropy':v=F.binary_cross_entropy(prediction.clamp(1e-7,1-1e-7),actual.reshape(prediction.shape))
                elif p['loss']=='mse':v=F.mse_loss(prediction,actual.reshape(prediction.shape))
                else:raise ValueError(p['loss'])
                data_loss=v
                if p.get('regularization','none')!='none':
                    ids=p.get('regularizationParameterIds',[i for i in self.parameter_map if self.nodes[i]['type']=='weight'])
                    v=v+p.get('regularizationStrength',0)*sum((self.parameter_map[i].abs() if p['regularization']=='l1' else self.parameter_map[i].square()/2).sum() for i in ids)
                loss=v
            else:raise ValueError(kind)
            values[key]=v,axes
        return prediction,loss,data_loss
