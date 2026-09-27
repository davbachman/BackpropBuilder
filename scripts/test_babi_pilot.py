"""Independent behavioral checks for the native bAbI pilot, not a browser parity claim."""
import importlib.util,unittest,tempfile
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('pilot',Path(__file__).with_name('pilot-babi.py'));pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
class PilotChecks(unittest.TestCase):
    def test_parser_excludes_future_facts_prior_answers_and_support_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'qa.txt';path.write_text('1 Mary went to the garden.\n2 Where is Mary?\tgarden\t1\n3 John went to the kitchen.\n4 Where is John?\tkitchen\t3\n1 Sandra went to the office.\n2 Where is Sandra?\toffice\t1\n')
            rows=pilot.parse(path)
        self.assertEqual([r['story'] for r in rows],[0,0,1]);self.assertEqual(len(rows[0]['facts']),1);self.assertNotIn('kitchen',rows[0]['tokens']);self.assertEqual(rows[1]['tokens'].count('garden'),1);self.assertNotIn('3',rows[1]['tokens'])
    def test_order_blind_controls_are_permutation_invariant(self):
        torch.manual_seed(42);x=torch.tensor([[2,3,2,5],[5,2,3,2]])
        for kind in ['counts-linear','counts-mlp','mean','attention-no-position']:
            m=pilot.PilotModel(kind,10,8,3,width=4,hidden=5)
            torch.testing.assert_close(m(x)[0],m(x)[1],atol=1e-6,rtol=1e-5)
    def test_padding_is_ignored_and_gradients_are_finite(self):
        torch.manual_seed(42)
        for kind in pilot.KINDS:
            m=pilot.PilotModel(kind,10,8,3,width=4,hidden=5)
            a=m(torch.tensor([[2,3]]));b=m(torch.tensor([[2,3,0,0],[4,5,6,7]]))[:1]
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
            torch.nn.functional.cross_entropy(b,torch.tensor([1])).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()),kind)
    def test_ordered_models_can_distinguish_permutations(self):
        torch.manual_seed(42);x=torch.tensor([[2,3,2,5],[5,2,3,2]])
        for kind in ['ordered-mlp','attention-position','transformer-1','transformer-2']:
            m=pilot.PilotModel(kind,10,8,3,width=4,hidden=5)
            self.assertGreater(float((m(x)[0]-m(x)[1]).abs().max().detach()),1e-7)
    def test_structured_models_ignore_padding_and_have_finite_gradients(self):
        torch.manual_seed(42)
        ids=torch.tensor([[2,3,9,4]])
        facts=torch.tensor([[[2,3]+[0]*8,[4,5]+[0]*8]])
        question=torch.tensor([[4]+[0]*9])
        for kind in ['sentence-mlp','memory-1','memory-2','memory-3','retrieval-1','retrieval-2','retrieval-3']:
            model=pilot.PilotModel(kind,12,8,3,width=4,hidden=5,question_id=9,answer_ids=[5,6,7])
            a=model(ids,facts,question)
            b=model(ids,torch.nn.functional.pad(facts,(0,0,0,2)),question)
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
            torch.nn.functional.cross_entropy(b,torch.tensor([1])).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),kind)
    def test_question_readout_has_same_initial_weights_as_mean_readout(self):
        torch.manual_seed(137);a=pilot.PilotModel('transformer-2',12,8,3,width=4,hidden=5)
        torch.manual_seed(137);b=pilot.PilotModel('question-transformer-2',12,8,3,width=4,hidden=5,question_id=9)
        for key in a.state_dict():torch.testing.assert_close(a.state_dict()[key],b.state_dict()[key])
        out=b(torch.tensor([[2,3,9,4,5]]));self.assertTrue(torch.isfinite(out).all())
if __name__=='__main__':torch.set_num_threads(1);unittest.main()
