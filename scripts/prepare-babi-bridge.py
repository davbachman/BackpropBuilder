"""A sizeable controlled bAbI-style bridge, explicitly not an official bAbI task.
Matched reversal pairs have identical bags but different answers; split by pair.
"""
import collections,csv,importlib.util,json,random
from pathlib import Path
spec=importlib.util.spec_from_file_location('pilot',Path(__file__).with_name('pilot-babi.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
if (m.ROOT/'frozen.json').exists():raise ValueError('Study is frozen; use a new output directory.')
rng=random.Random(20260927);people=['Mary','John','Sandra','Daniel'];places=['kitchen','garden','bedroom','bathroom','hallway','office'];pairs={}
while len(pairs)<6000:
    person=rng.choice(people);locations=[rng.choice(places)]
    for _ in range(4):locations.append(rng.choice([p for p in places if p!=locations[-1]]))
    if locations[0]==locations[-1]:continue
    key=(person,*min(tuple(locations),tuple(reversed(locations))))
    pairs[key]=(person,locations)
items=list(pairs.values());rng.shuffle(items);rows=[];test=[]
for group,(person,places) in enumerate(items):
    split='train' if group<4000 else 'validation' if group<5000 else 'test'
    for locations in [places,list(reversed(places))]:
        facts=[f'{person} moved to the {place}.' for place in locations];q=f'Where is {person}?'
        row={'story':group,'facts':facts,'question':q,'answer':locations[-1],'tokens':m.tokens(' '.join(facts))+['<question>']+m.tokens(q),'supportCount':1,'split':split}
        (test if split=='test' else rows).append(row)
counts=collections.Counter(t for r in rows if r['split']=='train' for t in r['tokens']);vocabulary=['<pad>','<unk>']+sorted(counts,key=lambda t:(-counts[t],t));index={t:i for i,t in enumerate(vocabulary)};classes=sorted(set(r['answer'] for r in rows))
for r in rows+test:r['ids']=[index.get(t,1) for t in r['tokens']];r['target']=classes.index(r['answer'])
for a,b in zip(rows[::2],rows[1::2]):assert collections.Counter(a['tokens'])==collections.Counter(b['tokens']) and a['answer']!=b['answer']
data={'task':0,'vocabulary':vocabulary,'classes':classes,'length':max(len(r['ids']) for r in rows),'rows':rows,'excludedLongExamples':0,'originalQuestions':len(rows),'syntheticBridge':True,'generation':'Five moves by one person; matched reversed histories; split by reversal pair; 10,000 train/validation plus 2,000 held-out questions.'}
(m.ROOT/'data/qa0.json').write_text(json.dumps(data));(m.ROOT/'data/qa0-test.json').write_text(json.dumps(test))
with (m.ROOT/'data/qa0.csv').open('w') as f:
    out=csv.writer(f);out.writerow(['story_id','text','label','split']);out.writerows([r['story'],' '.join(r['facts'])+' '+r['question'],r['answer'],r['split']] for r in rows)
print('Prepared',len(rows),'bridge questions and',len(test),'held-out questions; no identical histories across splits.')
