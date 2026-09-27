"""Controlled two-fact bAbI-style bridge: object owner + owner's location.
No drops/transfers/repeated moves; three questions per world stay in one split.
"""
import collections,csv,importlib.util,json,random
from pathlib import Path
spec=importlib.util.spec_from_file_location('pilot',Path(__file__).with_name('pilot-babi.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
if (m.ROOT/'frozen.json').exists():raise ValueError('Study is frozen; use a new output directory.')
rng=random.Random(123987);people=['Mary','John','Sandra','Daniel','Julie','Bill'];places=['kitchen','garden','bedroom','bathroom','hallway','office'];objects=['apple','milk','football','book'];worlds={}
while len(worlds)<4000:
    assignments=list(zip(rng.sample(people,3),rng.sample(objects,3),[rng.choice(places) for _ in range(3)]));key=tuple(sorted(assignments))
    worlds[key]=assignments
items=list(worlds.values());rng.shuffle(items);rows=[];test=[]
for group,assignments in enumerate(items):
    split='train' if group<3000 else 'validation' if group<3500 else 'test'
    facts=[f'{p} picked up the {o}.' for p,o,l in assignments]+[f'{p} moved to the {l}.' for p,o,l in assignments];rng.shuffle(facts)
    for person,obj,location in assignments:
        q=f'Where is the {obj}?';r={'story':group,'facts':facts,'question':q,'answer':location,'tokens':m.tokens(' '.join(facts))+['<question>']+m.tokens(q),'supportCount':2,'split':split}
        (test if split=='test' else rows).append(r)
counts=collections.Counter(t for r in rows if r['split']=='train' for t in r['tokens']);vocabulary=['<pad>','<unk>']+sorted(counts,key=lambda t:(-counts[t],t));index={t:i for i,t in enumerate(vocabulary)};classes=sorted(set(r['answer'] for r in rows))
for r in rows+test:r['ids']=[index.get(t,1) for t in r['tokens']];r['target']=classes.index(r['answer'])
data={'task':100,'vocabulary':vocabulary,'classes':classes,'length':max(len(r['ids']) for r in rows),'rows':rows,'excludedLongExamples':0,'originalQuestions':len(rows),'syntheticBridge':True,'generation':'Three people each pick up a distinct object and move once. Six shuffled facts. Split by underlying world assignment, all three questions kept together.'}
(m.ROOT/'data/qa100.json').write_text(json.dumps(data));(m.ROOT/'data/qa100-test.json').write_text(json.dumps(test))
with (m.ROOT/'data/qa100.csv').open('w') as f:
    out=csv.writer(f);out.writerow(['story_id','text','label','split']);out.writerows([r['story'],' '.join(r['facts'])+' '+r['question'],r['answer'],r['split']] for r in rows)
print('Prepared two-fact bridge:',len(rows),'train/validation;',len(test),'test. Split by distinct world.')
