"""Prepare raw California housing columns; fit no feature scaling outside the app."""
from pathlib import Path
import csv,json
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
root=Path('output/housing-pilot');root.mkdir(parents=True,exist_ok=True)
if (root/'frozen.json').exists():raise RuntimeError('Use a new output directory for another frozen study.')
data=fetch_california_housing(data_home=str(root/'raw'))
train,other=train_test_split(list(range(len(data.target))),test_size=.3,random_state=731)
validation,test=train_test_split(other,test_size=.5,random_state=982)
for name,indices in [('housing',list(train)+list(validation)),('final-test',test)]:
 with (root/(name+'.csv')).open('w') as f:
  writer=csv.writer(f);writer.writerow([*data.feature_names,'MedHouseValue','split'])
  training=set(train)
  writer.writerows([*data.data[i],data.target[i],'train' if i in training else 'test'] for i in indices)
(root/'source.json').write_text(json.dumps({'features':data.feature_names,'train':train,'validation':validation,'final':test,'x':data.data.tolist(),'y':data.target.tolist(),'targetUnits':'100000 USD','source':'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html'}))
print(len(train),len(validation),len(test))
