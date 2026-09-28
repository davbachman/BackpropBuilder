"""Build uploadable packages; generated data and instructor models stay ignored."""
import hashlib, json, shutil, zipfile
from pathlib import Path
src=Path('assignments/01-mpg');out=Path('output/assignment-01-mpg')
student=out/'student-files';student.mkdir(exist_ok=True)
for name in ['canvas.html']:shutil.copy(src/name,student/name)
shutil.copy(out/'mpg.csv',student/'mpg.csv')
with zipfile.ZipFile(out/'gradescope-mpg-01-v3.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in ['setup.sh','run_autograder','runner.py','grader.py']:z.write(src/name,name)
    z.write(out/'data.json','data.json')
with zipfile.ZipFile(out/'canvas-student-files.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in ['canvas.html','mpg.csv']:z.write(student/name,name)
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'gradescope-mpg-01-v3.zip',out/'canvas-student-files.zip',out/'mpg.csv']}
(out/'package-sha256.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(out/'gradescope-mpg-01-v3.zip') as z:
    assert set(z.namelist())=={'setup.sh','run_autograder','runner.py','grader.py','data.json'}
    assert (z.getinfo('run_autograder').external_attr>>16)&0o111
print(json.dumps(manifest,indent=2))
