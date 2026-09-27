"""From-scratch bAbI architecture pilot. No pretrained models or attention modules.
Official train stories are split as groups; official test stays unopened in training.
"""
import argparse,collections,copy,csv,hashlib,json,math,os,random,re,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(os.environ.get('BABI_PILOT_OUTPUT','output/babi-pilot'))
TASKS={1:'single-supporting-fact',2:'two-supporting-facts',4:'two-arg-relations'}
KINDS=['counts-linear','counts-mlp','mean','ordered-mlp','attention-no-position','attention-position','transformer-1','transformer-2']

def tokens(text):return re.findall(r"\w+|[^\w\s]",text.lower())
def parse(path):
    rows=[];facts=[];story=-1
    for line in path.read_text().splitlines():
        number,text=line.split(' ',1)
        if number=='1':facts=[];story+=1
        if '\t' not in text:facts.append(text);continue
        question,answer,support=text.split('\t')
        # No previous questions/answers or support annotations enter the features.
        sequence=tokens(' '.join(facts))+['<question>']+tokens(question)
        rows.append({'story':story,'facts':list(facts),'question':question.strip(),'answer':answer,'tokens':sequence,'supportCount':len(support.split())})
    return rows

def prepare():
    if (ROOT/'frozen.json').exists():raise ValueError('Study is frozen; use a new output directory.')
    (ROOT/'data').mkdir(exist_ok=True);(ROOT/'runs').mkdir(exist_ok=True)
    protocol={'tasks':[4,1,2],'maxTokens':256,'width':32,'hidden':64,'heads':2,'batch':64,'learningRate':.001,'weightDecay':.01,'clipNorm':1.,'maxEpochs':30,'patience':5,'minDelta':.001,'screenSeed':137,'confirmationSeeds':[211,307],'selection':'Inspect validation across all prespecified variants; confirm promising transitions across seeds; do not access official test until configurations frozen.','successTarget':.95,'trainingSource':'bAbI en-10k official train','sourceUrl':'https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz','archiveSha256':hashlib.sha256((ROOT/'raw/tasks.tar.gz').read_bytes()).hexdigest()}
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for task,title in TASKS.items():
        raw=parse(ROOT/'raw'/f'qa{task}_{title}_train.txt')
        groups=sorted(set(r['story'] for r in raw));random.Random(731).shuffle(groups);validation=set(groups[:len(groups)//5])
        rows=[{**r,'split':'validation' if r['story'] in validation else 'train'} for r in raw if len(r['tokens'])<=256]
        counts=collections.Counter(t for r in rows if r['split']=='train' for t in r['tokens'])
        vocabulary=['<pad>','<unk>']+sorted(counts,key=lambda t:(-counts[t],t));index={t:i for i,t in enumerate(vocabulary)}
        classes=sorted(set(r['answer'] for r in rows if r['split']=='train'))
        for r in rows:r['ids']=[index.get(t,1) for t in r['tokens']];r['target']=classes.index(r['answer'])
        assert not ({r['story'] for r in rows if r['split']=='train'}&{r['story'] for r in rows if r['split']=='validation'})
        data={'task':task,'vocabulary':vocabulary,'classes':classes,'length':max(len(r['ids']) for r in rows),'rows':rows,'excludedLongExamples':len(raw)-len(rows),'originalQuestions':len(raw)}
        (ROOT/'data'/f'qa{task}.json').write_text(json.dumps(data))
        with (ROOT/'data'/f'qa{task}.csv').open('w') as file:
            out=csv.writer(file);out.writerow(['story_id','text','label','split'])
            out.writerows([r['story'],' '.join(r['facts'])+' '+r['question'],r['answer'],r['split']] for r in rows)
        print('PREPARED',task,'train',sum(r['split']=='train' for r in rows),'validation',sum(r['split']=='validation' for r in rows),'length',data['length'],'vocabulary',len(vocabulary),'excluded',data['excludedLongExamples'],flush=True)

class PilotModel(nn.Module):
    def __init__(self,kind,vocabulary,length,classes,width=32,hidden=64,question_id=None,answer_ids=None):
        super().__init__();self.kind=kind;self.length=length;self.vocabulary=vocabulary;self.width=width;self.question_id=question_id
        self.p=nn.ParameterDict()
        def weight(name,shape,scale=None):
            value=torch.empty(*shape)
            if scale is None:nn.init.xavier_uniform_(value)
            else:nn.init.uniform_(value,-scale,scale)
            self.p[name]=nn.Parameter(value)
        def linear(name,inputs,outputs):weight(name+'_w',(inputs,outputs));self.p[name+'_b']=nn.Parameter(torch.zeros(outputs))
        if kind in ('retrieval-1','retrieval-2','retrieval-3'):
            self.hops=int(kind[-1]);self.answer_ids=answer_ids
            if answer_ids is None:raise ValueError('Retrieval output ties to answer word embeddings')
            for h in range(self.hops+1):
                weight(f'emb{h}',(vocabulary,width),.1)
                weight(f'time{h}',(64,width),.01)
            self.p['word_position']=nn.Parameter(torch.ones(10,width))
            self.p['output_b']=nn.Parameter(torch.zeros(classes))
            return
        if kind in ('sentence-mlp','memory-1','memory-2','memory-3'):
            weight('embedding',(vocabulary,width),.1)
            linear('fact',10*width,width);linear('question',10*width,width)
            weight('time',(64,width),.1)
            if kind=='sentence-mlp':linear('hidden',65*width,hidden)
            else:
                linear('hidden',width,hidden)
                self.hops=int(kind[-1])
                for h in range(self.hops):
                    for name in ['q','k','v']:weight(f'h{h}_{name}',(width,width))
            linear('output',hidden,classes)
            return
        if kind.startswith('counts'):
            if kind=='counts-linear':linear('output',vocabulary,classes)
            else:linear('hidden',vocabulary,hidden);linear('output',hidden,classes)
            return
        weight('embedding',(vocabulary,width),.1)
        if kind=='ordered-mlp':linear('hidden',length*width,hidden)
        else:linear('hidden',width,hidden)
        linear('output',hidden,classes)
        self.blocks=2 if kind.endswith('transformer-2') else 1
        if kind in ('attention-position','transformer-1','transformer-2') or kind.startswith('question-transformer'):weight('position',(length,width),.1)
        if 'attention' in kind or 'transformer' in kind:
            for b in range(self.blocks):
                for name in ['q','k','v','o']:weight(f'b{b}_{name}',(width,width))
                if 'transformer' in kind:
                    for name in ['norm1','norm2']:
                        self.p[f'b{b}_{name}_g']=nn.Parameter(torch.ones(width));self.p[f'b{b}_{name}_b']=nn.Parameter(torch.zeros(width))
                    linear(f'b{b}_ff1',width,2*width);linear(f'b{b}_ff2',2*width,width)
    def linear(self,x,name):return x@self.p[name+'_w']+self.p[name+'_b']
    def forward(self,ids,facts=None,question=None):
        self.attention_scores=[]
        mask=ids.ne(0);batch,length=ids.shape
        if self.kind in ('retrieval-1','retrieval-2','retrieval-3'):
            if facts is None or question is None:raise ValueError('Retrieval requires fact boundaries and a question')
            fm=facts.ne(0).any(-1)
            q=(F.embedding(question,self.p['emb0'])*question.ne(0)[...,None]).sum(1)
            for h in range(self.hops):
                keys=(F.embedding(facts,self.p[f'emb{h}'])*facts.ne(0)[...,None]*self.p['word_position']).sum(-2)+self.p[f'time{h}'][:facts.shape[1]]
                values=(F.embedding(facts,self.p[f'emb{h+1}'])*facts.ne(0)[...,None]*self.p['word_position']).sum(-2)+self.p[f'time{h+1}'][:facts.shape[1]]
                a=(keys*q[:,None,:]).sum(-1).masked_fill(~fm,-1e9).softmax(-1)
                q=q+(a[:,:,None]*values).sum(1)
            return q@self.p[f'emb{self.hops}'][self.answer_ids].T+self.p['output_b']
        if self.kind in ('sentence-mlp','memory-1','memory-2','memory-3'):
            if facts is None or question is None:raise ValueError('Structured models require sentence boundaries and a separate question')
            fact_mask=facts.ne(0).any(-1)
            f=F.embedding(facts,self.p['embedding'])*facts.ne(0)[...,None]
            f=F.relu(self.linear(f.flatten(-2),'fact'))+self.p['time'][:facts.shape[1]]
            q=F.embedding(question,self.p['embedding'])*question.ne(0)[...,None]
            q=F.relu(self.linear(q.flatten(-2),'question'))
            if self.kind=='sentence-mlp':
                f=f*fact_mask[:,:,None];x=torch.cat([F.pad(f,(0,0,0,64-f.shape[1])).flatten(1),q],-1)
            else:
                for h in range(self.hops):
                    scores=((f@self.p[f'h{h}_k'])*(q@self.p[f'h{h}_q'])[:,None,:]).sum(-1)/math.sqrt(self.width)
                    scores=scores.masked_fill(~fact_mask,-1e9);self.attention_scores.append(scores)
                    a=scores.softmax(-1)
                    q=q+(a[:,:,None]*(f@self.p[f'h{h}_v'])).sum(1)
                x=q
            return self.linear(F.relu(self.linear(x,'hidden')),'output')
        if self.kind.startswith('counts'):
            x=torch.zeros(batch,self.vocabulary,device=ids.device).scatter_add(1,ids,mask.float())
            if self.kind=='counts-mlp':x=F.relu(self.linear(x,'hidden'))
            return self.linear(x,'output')
        x=F.embedding(ids,self.p['embedding'])*mask[:,:,None]
        if self.kind=='ordered-mlp':
            x=F.pad(x,(0,0,0,self.length-length)).reshape(batch,-1)
        else:
            if 'position' in self.p:x=x+self.p['position'][:length]
            if 'attention' in self.kind or 'transformer' in self.kind:
                for b in range(self.blocks):
                    prefix=f'b{b}_';residual=x
                    if 'transformer' in self.kind:x=F.layer_norm(x,(self.width,),self.p[prefix+'norm1_g'],self.p[prefix+'norm1_b'])
                    q,k,v=[(x@self.p[prefix+n]).reshape(batch,length,2,self.width//2).transpose(1,2) for n in ['q','k','v']]
                    scores=(q@k.transpose(-2,-1))/math.sqrt(self.width//2)
                    probabilities=scores.masked_fill(~mask[:,None,None,:],-1e9).softmax(-1)
                    x=(probabilities@v).transpose(1,2).reshape(batch,length,self.width)@self.p[prefix+'o']
                    if 'transformer' in self.kind:
                        x=residual+x;z=F.layer_norm(x,(self.width,),self.p[prefix+'norm2_g'],self.p[prefix+'norm2_b'])
                        x=x+self.linear(F.relu(self.linear(z,prefix+'ff1')),prefix+'ff2')
            pool=mask
            if self.kind.startswith('question-transformer'):
                pool=mask & (ids.eq(self.question_id).cumsum(1)>0) & ids.ne(self.question_id)
            x=(x*pool[:,:,None]).sum(1)/pool.sum(1,keepdim=True)
        return self.linear(F.relu(self.linear(x,'hidden')),'output')

def tensors(data):
    ids=torch.tensor([r['ids']+[0]*(data['length']-len(r['ids'])) for r in data['rows']]);targets=torch.tensor([r['target'] for r in data['rows']])
    splits={s:torch.tensor([i for i,r in enumerate(data['rows']) if r['split']==s]) for s in ['train','validation']}
    return ids,targets,splits

def train(tasks,kinds,seeds,epochs,patience=5,tag="",support_supervision=False):
    if (ROOT/'frozen.json').exists():raise ValueError('Study is frozen; use a new output directory.')
    protocol=json.loads((ROOT/'protocol.json').read_text());protocol['supportSupervision']=support_supervision;protocol['patience']=patience;protocol['maxEpochs']=epochs;torch.set_num_threads(4)
    for task in tasks:
        data=json.loads((ROOT/'data'/f'qa{task}.json').read_text());ids,targets,splits=tensors(data)
        index={t:i for i,t in enumerate(data['vocabulary'])}
        encode=lambda text:[index.get(t,1) for t in tokens(text)]
        padded=lambda text:(encode(text)+[0]*10)[:10]
        assert all(len(encode(f))<=10 for r in data['rows'] for f in r['facts']+[r['question']])
        assert max(len(r['facts']) for r in data['rows'])<=64
        facts=torch.tensor([[padded(f) for f in r['facts']]+[[0]*10]*(64-len(r['facts'])) for r in data['rows']])
        questions=torch.tensor([padded(r['question']) for r in data['rows']])
        support=None
        if support_supervision:
            assert task==100,'Auxiliary labels are defined only for the controlled two-fact bridge'
            labels=[]
            for r in data['rows']:
                obj=tokens(r['question'])[-2]
                first=next(i for i,f in enumerate(r['facts']) if obj in tokens(f) and 'picked' in tokens(f))
                owner=tokens(r['facts'][first])[0]
                second=next(i for i,f in enumerate(r['facts']) if tokens(f)[0]==owner and 'moved' in tokens(f))
                labels.append([first,second])
            support=torch.tensor(labels)
        def batch(indices):
            x=ids[indices];f=facts[indices];count=int(f.ne(0).any(-1).sum(-1).max())
            return x[:,:int(x.ne(0).sum(1).max())],targets[indices],f[:,:count],questions[indices]
        @torch.no_grad()
        def evaluate(model,indices):
            model.eval();total=0.;hits=0
            for group in indices.split(128):
                x,y,f,q=batch(group);logits=model(x,f,q);total+=float(F.cross_entropy(logits,y))*len(group);hits+=int((logits.argmax(-1)==y).sum())
            return {'loss':total/len(indices),'accuracy':hits/len(indices),'examples':len(indices)}
        for kind in kinds:
            for seed in seeds:
                name=f'qa{task}-{kind}-seed{seed}'+('-'+tag if tag else '');path=ROOT/'runs'/f'{name}.json'
                if path.exists():
                    old=json.loads(path.read_text())
                    assert old['budgetEpochs']==epochs and old.get('trainingProtocol',{}).get('patience',5)==patience,'Existing run uses different settings; use a new tag.'
                    assert old['dataSha256']==hashlib.sha256((ROOT/'data'/f'qa{task}.json').read_bytes()).hexdigest(),'Existing run uses different data.'
                    print('EXISTS',name,flush=True);continue
                torch.manual_seed(seed);model=PilotModel(kind,len(data['vocabulary']),data['length'],len(data['classes']),question_id=data['vocabulary'].index('<question>'),answer_ids=[data['vocabulary'].index(c) for c in data['classes']])
                opt=torch.optim.AdamW(model.parameters(),lr=protocol['learningRate'],weight_decay=protocol['weightDecay'])
                start=time.perf_counter();best=evaluate(model,splits['validation']);state=copy.deepcopy(model.state_dict());best_epoch=0;stale=0;reports=[]
                for epoch in range(1,epochs+1):
                    model.train();order=splits['train'].tolist();random.Random(42+epoch).shuffle(order)
                    for group in torch.tensor(order).split(protocol['batch']):
                        x,y,f,q=batch(group);opt.zero_grad();loss=F.cross_entropy(model(x,f,q),y)
                        if support_supervision:
                            assert len(model.attention_scores)==2
                            loss=loss+.5*sum(F.cross_entropy(scores,support[group,h]) for h,scores in enumerate(model.attention_scores))
                        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite '+name)
                        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),protocol['clipNorm']);opt.step()
                    val=evaluate(model,splits['validation']);improved=val['loss']<best['loss']-protocol['minDelta']
                    if improved:best=val;state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
                    else:stale+=1
                    reports.append({'epoch':epoch,'validation':val,'improved':improved})
                    print(name,epoch,round(val['loss'],4),round(val['accuracy'],4),flush=True)
                    if stale>=protocol['patience']:break
                model.load_state_dict(state)
                result={'name':name,'trainingProtocol':protocol,'task':task,'kind':kind,'seed':seed,'bestEpoch':best_epoch,'completed':epoch,'budgetEpochs':epochs,'seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in model.parameters()),'validation':best,'train':evaluate(model,splits['train']),'reports':reports,'testUsed':False,'dataSha256':hashlib.sha256((ROOT/'data'/f'qa{task}.json').read_bytes()).hexdigest()}
                torch.save(state,ROOT/'runs'/f'{name}.pt');path.write_text(json.dumps(result,indent=2));print('FINISHED',name,result['validation'],round(result['seconds'],1),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','train']);parser.add_argument('--tasks',default='4,1,2');parser.add_argument('--models',default=','.join(KINDS));parser.add_argument('--seeds',default='137');parser.add_argument('--epochs',type=int,default=30);parser.add_argument('--patience',type=int,default=5);parser.add_argument('--tag',default='');parser.add_argument('--support-supervision',action='store_true');args=parser.parse_args()
    if args.mode=='prepare':prepare()
    else:train(list(map(int,args.tasks.split(','))),args.models.split(','),list(map(int,args.seeds.split(','))),args.epochs,args.patience,args.tag,args.support_supervision)
