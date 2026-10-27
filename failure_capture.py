"""Non-random forensic snapshot only when a source loss becomes nonfinite."""
import json
import os
from pathlib import Path
import random
import numpy as np
import torch


def save(recorder,loss,phase='TRAIN',validation_batch=None,reason='NONFINITE_LOSS'):
    root=Path(os.environ['RUN_RECORD_DIR']);model=recorder.exp.model
    batch=getattr(recorder,'current_batch',{})
    state=dict(adapter={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
               optimizer=recorder.opt.state_dict(),scaler=recorder.scaler.state_dict(),epoch=recorder.epoch,
               batch=recorder.batches+1 if phase=='TRAIN' else validation_batch,phase=phase,updates=recorder.updates,current_batch={k:v.detach().cpu().clone() for k,v in batch.items()},
               python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
               cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               resume_eligible=False,reason='Forensic mid-epoch snapshot; native DataLoader worker state is not a full resume state')
    torch.save(state,root/'numerical_failure_snapshot.pt')
    record=dict(reason=reason,phase=phase,loss=str(float(loss.detach())),epoch=recorder.epoch,batch=recorder.batches+1 if phase=='TRAIN' else validation_batch,
                updates=recorder.updates,scale=recorder.scaler.get_scale(),
                tensor_finite={k:bool(torch.isfinite(v).all()) for k,v in batch.items()},
                trainable_parameter_finite=all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad),
                snapshot=str(root/'numerical_failure_snapshot.pt'),not_a_resumable_mid_epoch_state=True)
    (root/'numerical_failure.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')


def check():
    import tempfile
    from types import SimpleNamespace
    with tempfile.TemporaryDirectory() as directory:
        old=os.environ.get('RUN_RECORD_DIR');os.environ['RUN_RECORD_DIR']=directory
        try:
            model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters())
            recorder=SimpleNamespace(exp=SimpleNamespace(model=model),opt=optimizer,scaler=torch.cuda.amp.GradScaler(enabled=False),epoch=1,batches=0,updates=0,current_batch=dict(history=torch.ones(1,2),target=torch.zeros(1,1),prediction=torch.full((1,1),float('nan'))))
            before=torch.get_rng_state().clone();save(recorder,torch.tensor(float('nan')))
            assert torch.equal(before,torch.get_rng_state())
            state=torch.load(Path(directory)/'numerical_failure_snapshot.pt',map_location='cpu')
            assert state['batch']==1 and not state['resume_eligible']
            assert all(torch.equal(v,state['adapter'][k]) for k,v in model.state_dict().items())
            assert json.loads((Path(directory)/'numerical_failure.json').read_text())['tensor_finite']==dict(history=True,target=True,prediction=False)
        finally:
            if old is None:os.environ.pop('RUN_RECORD_DIR',None)
            else:os.environ['RUN_RECORD_DIR']=old
    print('CPU failure snapshot and unchanged Torch RNG PASS')


if __name__=='__main__':check()

