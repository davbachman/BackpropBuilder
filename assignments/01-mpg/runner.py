#!/usr/bin/env python3
"""Run each model in its own bounded process; always report partial results."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ.get('AUTOGRADER_ROOT', '/autograder'))
source, submission, results = root/'source', root/'submission', root/'results'
results.mkdir(parents=True, exist_ok=True)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, str(source))
tests = []
def limits():
    if sys.platform.startswith('linux'):
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (12, 12))
try:
    from grader import PARTS, PART_POINTS
    for part in PARTS:
        try:
            completed = subprocess.run(
                [sys.executable, str(source/'grader.py'), part, str(submission/(part+'.json')), str(source/'data.json')],
                capture_output=True, text=True, timeout=15, check=True, preexec_fn=limits)
            report = json.loads(completed.stdout)
            tests.extend(report['tests'])
        except Exception:
            tests.append({'name': part+' — evaluation', 'score': 0, 'max_score': PART_POINTS[part],
                          'output': 'Unable to evaluate this model within the limits. Use a saved project with at most 100 blocks and small tensors.',
                          'visibility': 'visible'})
    output = {'tests': tests, 'score': sum(t['score'] for t in tests),
              'output': 'Assignment mpg-01-v3. Predictions were recomputed from saved parameters on the original dataset. Saved scores and losses were ignored.',
              'output_format': 'text', 'test_output_format': 'text', 'test_name_format': 'text',
              'visibility': 'visible', 'stdout_visibility': 'hidden'}
except Exception:
    output = {'score': 0, 'output': 'Autograder setup failed. Contact the instructor; this is not a model-quality result.',
              'output_format': 'text', 'visibility': 'visible', 'stdout_visibility': 'hidden'}
(results/'results.json').write_text(json.dumps(output, allow_nan=False))
