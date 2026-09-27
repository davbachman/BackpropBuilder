"""Freeze recipes and validation-selected weights before final evaluation."""
import hashlib,json
from pathlib import Path
r=Path('output/semester-pilot');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if (r/'frozen.json').exists():raise SystemExit('Already frozen; refusing to change the selected comparison.')
selected=[]
for row in json.loads((r/'manifest.json').read_text()):
 n=row['name']
 if n.startswith('iris-linear') or n.startswith('mpg-l1') or n.startswith('mpg-l2') or (n.startswith('mpg-small') and not any(n.startswith('mpg-small-l2-'+x+'-') for x in ['0','0.1','1'])):continue
 for folder in ['templates','runs','numeric']:
  if not (r/folder/(n+'.json')).exists():raise ValueError('Missing '+folder+'/'+n)
 selected.append({'name':n,'templateSha256':sha(r/'templates'/(n+'.json')),'weightsSha256':sha(r/'runs'/(n+'.json')),'numericSha256':sha(r/'numeric'/(n+'.json'))})
(r/'frozen.json').write_text(json.dumps({'selection':'All early baselines and improved models, validation-refined Iris rate, all QA/Alice variants, and small-MPG ridge 0/0.1/1, each at three seeds. Full-MPG regularization remains a validation-only screen. No tuning on the final examples. Official bAbI test has been evaluated in the preceding pilot; this is a compatibility confirmation, not a new untouched benchmark.','jobs':selected},indent=2))
print('Frozen',len(selected),'checkpoints')
