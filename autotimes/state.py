"""Epoch-boundary state: no random draws and no frozen-backbone serialization."""
import os
from pathlib import Path
import random
import numpy as np
import torch

def save(model,optimizer,scheduler,scaler,early,recorder,epoch):
    root=Path(os.environ['RUN_RECORD_DIR'])
    state=dict(adapter={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),epoch=epoch,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),early={k:getattr(early,k) for k in ['counter','best_score','early_stop','val_loss_min']},rows=recorder.rows,best_epoch=recorder.best_epoch,updates=recorder.updates)
    tmp=root/'resume.tmp.pt';torch.save(state,tmp);tmp.replace(root/'resume.pt')

def restore(model,optimizer,scheduler,scaler,early,recorder):
    path=Path(os.environ['RUN_RECORD_DIR'])/'resume.pt'
    if not path.exists():return 0
    state=torch.load(path,map_location='cpu');expected={n for n,p in model.named_parameters() if p.requires_grad}
    assert set(state['adapter'])==expected
    result=model.load_state_dict(state['adapter'],strict=False)
    assert not result.unexpected_keys and all(n.startswith('llama.') for n in result.missing_keys)
    optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
    for k,v in state['early'].items():setattr(early,k,v)
    recorder.rows=state['rows'];recorder.best_epoch=state['best_epoch'];recorder.updates=state['updates']
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    return 10 if early.early_stop else state['epoch']

