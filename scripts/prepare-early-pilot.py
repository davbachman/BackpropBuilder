"""Real source data, deterministic splits, training-only standardization.
Generated CSVs and recipes stay under ignored output/semester-pilot.
"""
import csv,io,json,random,urllib.request,zipfile
from pathlib import Path
import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer,load_digits
from sklearn.model_selection import train_test_split
import xlrd
ROOT=Path('output/semester-pilot')
if (ROOT/'frozen.json').exists():raise RuntimeError('Pilot is frozen. Use a separate checkout/output directory for a new study.')
(ROOT/'data').mkdir(parents=True,exist_ok=True)
def fetch(url):
    with urllib.request.urlopen(url) as r:return r.read()
def prepare(name,x,y,task):
    x=np.asarray(x,float);y=np.asarray(y)
    indices=np.arange(len(y));train,other=train_test_split(indices,test_size=.3,random_state=731,stratify=y if task!='regression' else None)
    val,test=train_test_split(other,test_size=.5,random_state=982,stratify=y[other] if task!='regression' else None)
    mean=x[train].mean(0);scale=x[train].std(0);scale[scale==0]=1;x=(x-mean)/scale
    ym,ys=(float(y[train].mean()),float(y[train].std())) if task=='regression' else (0,1)
    y=(y-ym)/ys
    rows=[(i,'train') for i in train]+[(i,'validation') for i in val]+[(i,'final-test') for i in test]
    payload={'name':name,'task':task,'features':x.shape[1],'classLabels':None if task=='regression' else [str(int(c)) for c in sorted(set(y))], 'scaling':{'mean':mean.tolist(),'scale':scale.tolist(),'targetMean':ym,'targetScale':ys},'rows':[{'x':x[i].tolist(),'y':float(y[i]),'split':s,'sourceIndex':int(i)} for i,s in rows]}
    (ROOT/'data'/f'{name}.json').write_text(json.dumps(payload))
    with (ROOT/'data'/f'{name}.csv').open('w') as f:
        w=csv.writer(f);w.writerow([f'x{i+1}' for i in range(x.shape[1])]+['target','split']);w.writerows([*x[i],float(y[i]) if task=='regression' else 'class'+str(int(y[i])),'train' if i in train else 'test'] for i in [*train,*val])
    print(name,len(train),len(val),len(test),flush=True)
for name,loader in [('cancer',load_breast_cancer),('iris',load_iris),('digits',load_digits)]:
    d=loader();prepare(name,d.data,d.target,'binary-classification' if name=='cancer' else 'classification')
raw=fetch('https://archive.ics.uci.edu/static/public/9/auto+mpg.zip');z=zipfile.ZipFile(io.BytesIO(raw));lines=z.read('auto-mpg.data').decode().splitlines();rows=[line.split()[:8] for line in lines if '?' not in line.split()[:8]];a=np.array(rows,float)
# Continuous/numeric predictors; omit categorical origin, unique car name.
prepare('mpg',a[:,1:7],a[:,0],'regression')
raw=fetch('https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip');z=zipfile.ZipFile(io.BytesIO(raw));name=next(n for n in z.namelist() if n.endswith('.xls'))
# Preserve the original source workbook; fit normalization on training rows only.
(ROOT/'data/concrete.xls').write_bytes(z.read(name))
book=xlrd.open_workbook(file_contents=z.read(name));sheet=book.sheet_by_index(0);a=np.array([sheet.row_values(i) for i in range(1,sheet.nrows)],float);prepare('concrete',a[:,:8],a[:,8],'regression')
