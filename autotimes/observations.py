import random
"""Non-random observations around the unmodified official training operations."""
import csv
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch

ROOT = Path(os.environ.get('RUN_RECORD_DIR', '.'))

def dump(name, value):
    ROOT.mkdir(parents=True, exist_ok=True)
    p = ROOT / name; t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); t.replace(p)

def initial(model, args):
    state = {n: p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    assert state and all(not n.startswith('llama.') for n in state)
    linear = lambda module: [(m.in_features,m.out_features) for m in module.modules() if isinstance(m,torch.nn.Linear)]
    p=args.token_len;h=args.mlp_hidden_dim;n=args.mlp_hidden_layers
    expected=([(p,4096)],[(4096,p)]) if n==0 else ([(p,h)]+[(h,h)]*max(0,n-2)+[(h,4096)],[(4096,h)]+[(h,h)]*max(0,n-2)+[(h,p)])
    assert (linear(model.encoder),linear(model.decoder)) == expected
    assert all(torch.isfinite(v).all() for v in state.values())
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict':state,'seed':int(os.environ['RUN_SEED']), 'stage':'before training','args':vars(args)}, ROOT / 'initial_adapter.pt')
    dump('actual_config.json',dict(args=vars(args),seed=int(os.environ['RUN_SEED']),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),trainable=[dict(name=n,shape=list(v.shape),dtype=str(v.dtype),numel=v.numel()) for n,v in state.items()],trainable_count=sum(v.numel() for v in state.values()),encoder=str(model.encoder),decoder=str(model.decoder)))

class Recorder:
    def __init__(self, experiment, optimizer, scaler, early, dataset, loader):
        expected = len(dataset)
        assert len(dataset)==expected and len(loader)==(expected//loader.batch_size if loader.drop_last else math.ceil(expected/loader.batch_size)), (len(dataset),len(loader),expected)
        self.expected_batches = len(loader)
        self.exp=experiment; self.opt=optimizer; self.scaler=scaler; self.early=early
        experiment.model.run_failure_recorder=self
        self.updates=0;self.rows=[];self.best_epoch=None;self.start=time.time()
        self.hook=optimizer.register_step_post_hook(self.updated)
        save=early.save_checkpoint
        def observed_save(*args, **kwargs):
            result=save(*args,**kwargs); self.best_epoch=self.epoch; return result
        early.save_checkpoint=observed_save
        dump('loader_check.json',dict(samples=len(dataset),batches=len(loader),drop_last=loader.drop_last,batch_size=loader.batch_size,sampler=type(loader.sampler).__name__,num_workers=loader.num_workers))
    def updated(self,*args): self.updates+=1
    def begin(self,epoch):
        self.epoch=epoch;self.epoch_start=time.time();self.before=self.updates;self.batches=0
        self.lr=self.opt.param_groups[0]['lr'];self.scale_start=self.scaler.get_scale()
    def remember(self,x,y,prediction):
        self.current_batch=dict(history=x.detach(),target=y.detach(),prediction=prediction.detach())
    def check_prediction(self,loss):
        if not torch.isfinite(self.current_batch["prediction"]).all():
            from failure_capture import save
            save(self,loss,reason="NONFINITE_PLANT_OUTPUT")
            raise FloatingPointError("NONFINITE_PLANT_OUTPUT")
    def before_step(self,loss):
        if not torch.isfinite(loss).item():
            from failure_capture import save
            save(self,loss)
            dump('failure.json',dict(reason='NONFINITE_LOSS',epoch=self.epoch,batch=self.batches+1,updates=self.updates-self.before,scale=self.scaler.get_scale()))
            raise FloatingPointError('NONFINITE_LOSS')
        self.step_updates=self.updates;self.scale_before=self.scaler.get_scale()
    def after_step(self):
        self.batches+=1
        if self.updates==self.step_updates:
            with (ROOT/'amp_events.jsonl').open('a') as f:
                f.write(json.dumps(dict(epoch=self.epoch,batch=self.batches,scale_before=self.scale_before,scale_after=self.scaler.get_scale(),event='AMP_SKIP'))+'\n')
    def end(self,train,val,test,path):
        assert all(math.isfinite(float(x)) for x in (train,val,test))
        bad=[n for n,p in self.exp.model.named_parameters() if p.requires_grad and not torch.isfinite(p).all()]
        if bad:
            dump('failure.json',dict(reason='NONFINITE_PARAMETER',names=bad,epoch=self.epoch));raise FloatingPointError(str(bad))
        updates=self.updates-self.before;assert self.batches==self.expected_batches and 0<=updates<=self.expected_batches
        self.rows.append(dict(epoch=self.epoch,train_loss=float(train),source_validation_loss=float(val),source_validation_alias_loss=float(test),lr=self.lr,loader_batches=self.batches,optimizer_updates=updates,amp_skips=self.batches-updates,scale_before=self.scale_start,scale_after=self.scaler.get_scale(),best_epoch=self.best_epoch,early_stop=self.early.early_stop,wall_seconds=time.time()-self.epoch_start,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
        with (ROOT/'training.tsv').open('w') as f:
            w=csv.DictWriter(f,self.rows[0].keys(),delimiter='\t');w.writeheader();w.writerows(self.rows)
    def finish(self,checkpoint):
        self.hook.remove()
        dump('train_complete.json',dict(status='COMPLETE',checkpoint=str(Path(checkpoint).resolve()),best_epoch=self.best_epoch,epochs=len(self.rows),updates=sum(r['optimizer_updates'] for r in self.rows),skips=sum(r['amp_skips'] for r in self.rows),stop_reason='OFFICIAL_EARLY_STOPPING' if self.early.early_stop else 'MAXIMUM_EPOCHS',wall_seconds=time.time()-self.start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),resume='Full epoch-boundary state saved in resume.pt; no mid-epoch exact resume claimed'))

