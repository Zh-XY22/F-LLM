"""CPU tests: python -m unittest discover -s tests."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from fllm.method import prediction_loss,observer_loss,residual,predict,lip
from fllm.source import score_series
from fllm.evaluate import reject_existing_evaluation
from autotimes.data_provider.data_loader import Dataset_TSF
from autotimes.utils.tools import strtobool


class CoreTests(unittest.TestCase):
    def test_patch_l2_and_physical_mask(self):
        p=torch.zeros(1,48);p[0,:2]=torch.tensor([3.,4.]);p[0,24]=12.
        self.assertEqual(prediction_loss(p,torch.zeros_like(p),'PAPER_L2').item(),8.5)
        mask=torch.ones(1,24);mask[0,1]=0
        loss=observer_loss(p[:,:24],torch.zeros(1,24),mask,torch.tensor([[2.]]),'L2_PHYSICAL')
        self.assertEqual(loss.item(),6.)

    def test_zero_history_and_score(self):
        x=torch.zeros(2,48);r,s,_=residual(x,torch.ones_like(x),x)
        self.assertTrue(torch.isfinite(s).all());self.assertEqual(r.abs().sum().item(),0)
        values=score_series(np.zeros((2,18)),np.zeros((2,18)))
        self.assertTrue(np.array_equal(values[:,:5],np.zeros((2,5))))

    def test_rollout_states(self):
        class Plant(nn.Module):
            def forward(self,x,*unused):return x*.8+1.
        class Observer(nn.Module):
            def forward(self,r):return torch.ones_like(r)*.3
        x=torch.arange(48,dtype=torch.float32).reshape(1,48)/10
        a=predict(Plant(),{'o':Observer()},x,torch.ones_like(x),48,amp=False)
        b=predict(Plant(),{'o':Observer()},x,torch.ones_like(x),48,alpha=0,amp=False)
        self.assertTrue(torch.equal(a['RAW'],b['RAW']))
        self.assertTrue(torch.equal(b['RAW'],b['o__CLOSED_LOOP']))
        self.assertTrue(torch.equal(a['o__POSTHOC'][:,:24],a['o__CLOSED_LOOP'][:,:24]))
        self.assertFalse(torch.equal(a['o__POSTHOC'][:,24:],a['o__CLOSED_LOOP'][:,24:]))

    def test_local_penalty_rng_and_graph(self):
        torch.manual_seed(71)
        head=nn.Sequential(nn.Linear(8,12),nn.Tanh(),nn.Dropout(.3),nn.Linear(12,24)).train()
        h=torch.ones(2,2,8,requires_grad=True);before=torch.get_rng_state().clone()
        loss,_,delta,xi=lip(head,h,torch.Generator().manual_seed(902021),0.)
        loss.backward()
        self.assertGreater(loss.item(),0);self.assertTrue(torch.isfinite(h.grad).all())
        self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertTrue(head.training)
        self.assertTrue((delta.norm(dim=-1)<xi.squeeze(-1)).all())

    def test_question_mark_and_existing_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'tiny.tsf'
            values=[str(i) if i!=5 else '?' for i in range(144)]
            path.write_text('@attribute series_name string\n@frequency monthly\n@horizon 18\n@missing true\n@equallength false\n@data\ns:'+','.join(values)+'\n')
            ds=Dataset_TSF(directory,data_path=path.name,size=[48,24,24])
            self.assertEqual(len(ds.timeseries[0]),143);self.assertTrue(np.isfinite(ds.timeseries[0]).all())
            state=root/'tasks/H18_EVALUATION/status.json';state.parent.mkdir(parents=True);state.write_text('{}')
            with self.assertRaises(FileExistsError):reject_existing_evaluation(root)
        self.assertEqual(strtobool('TRUE'),1)
        with self.assertRaisesRegex(ValueError,"invalid truth value 'maybe'"):strtobool('Maybe')


if __name__=='__main__':unittest.main()
