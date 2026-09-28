import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from grader import model_result, PARTS, PART_POINTS

ROOT = Path(__file__).resolve().parents[2]/'output/assignment-01-mpg'
DATA = json.loads((ROOT/'data.json').read_text())

class GraderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'model.json'
    def tearDown(self):
        self.temp.cleanup()
    def model(self, part='quadratic', prefix=''):
        return json.loads((ROOT/'models'/(prefix+part+'.json')).read_text())
    def grade(self, model, part='quadratic'):
        self.path.write_text(json.dumps(model))
        return model_result(self.path, part, DATA)
    def score(self, result):
        return sum(t['score'] for t in result['tests'])
    def test_reference_models_match_independent_app(self):
        reference = json.loads((ROOT/'app-pilot.json').read_text())
        for part in PARTS:
            result = self.grade(self.model(part), part)
            self.assertEqual(self.score(result), PART_POINTS[part], result)
            app = next(r for r in reference if r['name']==part and r['seed']==137)
            self.assertAlmostEqual(result['metrics']['train'], app['train'], places=7)
            self.assertAlmostEqual(result['metrics']['test'], app['validation'], places=7)
    def test_untrained_still_receives_structure_credit(self):
        self.assertEqual(self.score(self.grade(self.model(prefix='initial-'))), 20)
    def test_standardized_models_match_app_and_reject_invalid_stats(self):
        reference = json.loads((ROOT/'standardized-app-pilot.json').read_text())
        for part in PARTS:
            model = json.loads((ROOT/'standardized-models'/(part+'.json')).read_text())
            result = self.grade(model, part)
            self.assertEqual(self.score(result), PART_POINTS[part], result)
            app = next(r for r in reference if r['name'] == part)
            self.assertAlmostEqual(result['metrics']['train'], app['train'], places=7)
            self.assertAlmostEqual(result['metrics']['test'], app['validation'], places=7)
            node = next(n for n in model['state']['graph']['nodes'] if n['type'] == 'standardize')
            for stats in [None, {'mean':[0], 'scale':[0], 'count':318}, {'mean':[0], 'scale':[1], 'count':0}]:
                node['params']['standardization'] = stats
                self.assertEqual(self.score(self.grade(model, part)), 5)
    def test_reported_scores_values_and_dataset_overrides_are_ignored(self):
        model = self.model()
        model['state']['currentLoss'] = 0
        for node in model['state']['graph']['nodes']:
            node['value']={'shape':[],'data':[999]}
            node['params']['datasetValues']=[{'shape':[],'data':[0]}]*3
        self.assertEqual(self.score(self.grade(model)), 40)
    def test_target_leakage_rejected_even_if_cancelled(self):
        model=self.model()
        g=model['state']['graph']
        pred=next(n for n in g['nodes'] if n['id']=='prediction')
        pred['params']['expression']+=' + x5 - x5'
        g['edges'].append({'id':'leak','source':'data','sourceSlot':2,'target':'prediction','inputSlot':4})
        self.assertEqual(self.score(self.grade(model)),5)
    def test_wrong_model_form(self):
        self.assertEqual(self.score(self.grade(self.model('quadratic'),'two_features')),5)
    def test_modified_data_rejected(self):
        model=self.model();model['state']['graph']['nodes'][0]['params']['customCsv']['rows'][1][2]='0'
        self.assertEqual(self.score(self.grade(model)),0)
    def test_missing_scaling_rejected(self):
        model=self.model('one_feature')
        next(n for n in model['state']['graph']['nodes'] if n['id']=='scaled-weight')['params']['expression']='x1'
        self.assertEqual(self.score(self.grade(model,'one_feature')),5)
    def test_equivalent_scaling_accepted(self):
        model=self.model('one_feature');nodes=model['state']['graph']['nodes']
        next(n for n in nodes if n['id']=='scaled-weight')['params']['expression']='(x1 - 3000) / 2000'
        next(n for n in nodes if n['id']=='w0')['params']['value']['data'][0]*=2
        self.assertEqual(self.score(self.grade(model,'one_feature')),30)
    def test_matrix_implementation_accepted(self):
        model=self.model('two_features');g=model['state']['graph'];nodes={n['id']:n for n in g['nodes']}
        def node(key,kind,params):
            return {'id':key,'type':kind,'label':key,'position':{'x':0,'y':0},'params':params}
        w=[nodes['w0']['params']['value']['data'][0],nodes['w1']['params']['value']['data'][0]]
        g['nodes']=[nodes[k] for k in ['data','scaled-weight','scaled-year','bias','loss']]+[
            node('x','tensor-transform',{'transform':'reshape','shape':[1,1]}),
            node('z','tensor-transform',{'transform':'reshape','shape':[1,1]}),
            node('features','concat',{'axis':1}),
            node('weights','weight',{'value':{'shape':[2,1],'data':w}}),
            node('product','matmul',{}),node('prediction','arithmetic',{'expression':'x1 + x2'})]
        wiring=[('data','scaled-weight',0,0),('data','scaled-year',0,1),
                ('scaled-weight','x',0,0),('scaled-year','z',0,0),
                ('x','features',0,0),('z','features',1,0),
                ('features','product',0,0),('weights','product',1,0),
                ('product','prediction',0,0),('bias','prediction',1,0),
                ('prediction','loss',0,0),('data','loss',1,2)]
        g['edges']=[{'id':str(i),'source':a,'target':b,'inputSlot':c,'sourceSlot':d} for i,(a,b,c,d) in enumerate(wiring)]
        self.assertEqual(self.score(self.grade(model,'two_features')),30)
    def test_malformed_missing_and_resource_limits(self):
        self.path.write_text('{')
        self.assertEqual(self.score(model_result(self.path,'quadratic',DATA)),0)
        self.assertEqual(self.score(model_result(self.path.with_name('absent.json'),'quadratic',DATA)),0)
        for change in ['cycle','tensor','expression','nonfinite']:
            model=self.model();g=model['state']['graph']
            if change=='cycle':g['edges'].append({'source':'prediction','target':'scaled-weight','inputSlot':1})
            if change=='tensor':next(n for n in g['nodes'] if n['id']=='w0')['params']['value']={'shape':[1000000000],'data':[0]}
            if change=='expression':next(n for n in g['nodes'] if n['id']=='scaled-weight')['params']['expression']='__import__("os").system("false")'
            if change=='nonfinite':next(n for n in g['nodes'] if n['id']=='w0')['params']['value']['data'][0]=float('nan')
            self.assertLess(self.score(self.grade(model)),15)
    def test_packaged_runner_complete_and_partial(self):
        root=Path(self.temp.name)
        shutil.copytree(Path(__file__).parent,root/'source')
        shutil.copy(ROOT/'data.json',root/'source/data.json')
        (root/'submission').mkdir()
        for part in PARTS:shutil.copy(ROOT/'models'/(part+'.json'),root/'submission')
        env={**os.environ,'AUTOGRADER_ROOT':str(root)}
        subprocess.run([sys.executable,str(root/'source/runner.py')],env=env,check=True)
        self.assertEqual(json.loads((root/'results/results.json').read_text())['score'],100)
        (root/'submission/two_features.json').write_text('{bad')
        subprocess.run([sys.executable,str(root/'source/runner.py')],env=env,check=True)
        self.assertEqual(json.loads((root/'results/results.json').read_text())['score'],70)

if __name__=='__main__':unittest.main()
