"""Prepare Assignment 1 data and independent least-squares/gradient-descent checks."""
import csv, hashlib, io, json, random, urllib.request, zipfile
from pathlib import Path
import numpy as np
root=Path('output/assignment-01-mpg');root.mkdir(parents=True,exist_ok=True)
raw=urllib.request.urlopen('https://archive.ics.uci.edu/static/public/9/auto+mpg.zip').read()
lines=zipfile.ZipFile(io.BytesIO(raw)).read('auto-mpg.data').decode().splitlines()
rows=[]
for i,line in enumerate(lines):
    cells=line.split()
    if len(cells)<8:continue
    rows.append({'sourceIndex':i,'weight':float(cells[4]),'year':float(cells[6])+1900,'mpg':float(cells[0])})
order=list(range(len(rows)));random.Random(20260929).shuffle(order)
for rank,i in enumerate(order):rows[i]['split']='train' if rank<318 else 'test'
with (root/'mpg.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['weight_lb','model_year','mpg','split'])
    w.writerows([r['weight'],r['year'],r['mpg'],r['split']] for r in rows)
(root/'data.json').write_text(json.dumps({'version':'mpg-01-v1','rows':rows,'sourceSha256':hashlib.sha256(raw).hexdigest()},indent=2))
results={}
for part in ['one_feature','two_features','quadratic']:
    x=np.array([[1,(r['weight']-3000)/1000]+([(r['year']-1976)/10] if part!='one_feature' else [])+([((r['weight']-3000)/1000)**2] if part=='quadratic' else []) for r in rows]);y=np.array([r['mpg'] for r in rows]);train=np.array([r['split']=='train' for r in rows]);test=~train
    best=np.linalg.lstsq(x[train],y[train],rcond=None)[0]
    runs=[]
    for seed in [137,211,307]:
        params=np.random.default_rng(seed).uniform(-.1,.1,x.shape[1])
        for epoch in range(1000):params-=.05*2*x[train].T@(x[train]@params-y[train])/train.sum()
        runs.append({'seed':seed,'weights':params.tolist(),'trainMSE':float(np.mean((x[train]@params-y[train])**2)),'validationMSE':float(np.mean((x[test]@params-y[test])**2))})
    results[part]={'leastSquares':best.tolist(),'leastSquaresTrainMSE':float(np.mean((x[train]@best-y[train])**2)),'leastSquaresValidationMSE':float(np.mean((x[test]@best-y[test])**2)),'runs':runs}
(root/'pilot.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
