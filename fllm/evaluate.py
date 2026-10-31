"""Evaluate the fixed example with the executed predictor and scorer."""
from pathlib import Path
import numpy as np
import torch
from .runtime import read,write,task
from .train import make_model
from .method import load_adapter,observer,predict,finite
from .source import score_series,aggregate
from .scoring import independent


def reject_existing_evaluation(root):
    if (Path(root)/'tasks/H18_EVALUATION/status.json').exists():
        raise FileExistsError('Evaluation task state already exists; choose a new output directory')


def evaluate(root,plant_checkpoint,observer_checkpoint):
    root=Path(root);config=read(root/'config.json');ev=config['evaluation']
    reject_existing_evaluation(root)
    with task(root,'H18_EVALUATION','EVALUATION') as out:
        if out is None:raise FileExistsError('Evaluation already complete; choose a new output directory')
        model,capture=make_model(root,config['plant_seed'])
        load_adapter(model,torch.load(plant_checkpoint,map_location='cpu',weights_only=True))
        model.eval().requires_grad_(False)
        obs=observer().cuda()
        obs.load_state_dict(torch.load(observer_checkpoint,map_location='cpu',weights_only=True))
        obs.eval().requires_grad_(False)
        data=root/'data/target_h18';keys=read(data/'keys.json')
        x=np.load(data/'history.npy',mmap_mode='r');xm=np.load(data/'history_mask.npy',mmap_mode='r')
        y=np.load(data/'target.npy',mmap_mode='r');ym=np.load(data/'target_mask.npy',mmap_mode='r')
        h=ev['scored_points'];assert h==18 and ev['generated_points_per_step']==24
        assert len(keys)==len(x)==len(y)==48000 and y.shape==(48000,h)
        arrays={name:np.lib.format.open_memmap(out/(name+'.npy'),mode='w+',dtype=np.float32,shape=y.shape)
                for name in ['RAW','POSTHOC','CLOSED_LOOP']}
        online={name:[] for name in arrays}
        for start in range(0,len(x),ev['batch_size']):
            stop=start+ev['batch_size']
            xx=torch.from_numpy(np.array(x[start:stop],copy=True)).cuda()
            mm=torch.from_numpy(np.array(xm[start:stop],copy=True)).cuda()
            values=predict(model,{'observer':obs},xx,mm,h,ev['feedback_factor'],ev['alpha'],config['plant']['use_amp'])
            assert torch.equal(values['observer__POSTHOC'],values['observer__CLOSED_LOOP'])
            finite(list(values.values()),'NONFINITE_TARGET_PREDICTION')
            for name,a in arrays.items():
                v=values['RAW' if name=='RAW' else 'observer__'+name].cpu().numpy()
                a[start:start+len(v)]=v
                online[name].append(score_series(v,y[start:stop],ym[start:stop],ev['smape_epsilon']))
        for a in arrays.values():a.flush()
        arrays.clear();metrics={};checks={}
        for name in online:
            metrics[name]=aggregate(np.concatenate(online[name]))
            recomputed=aggregate(independent(out/(name+'.npy'),data/'target.npy',data/'target_mask.npy',h,{'FULL':(0,h)},ev['smape_epsilon'])['FULL'])
            errors={k:abs(metrics[name][k]-recomputed[k]) for k in ['smape','mse','mae','sse','sae']}
            assert all(errors[k]<=1e-12 for k in ['smape','mse','mae'])
            assert all(errors[k]<=1e-6+1e-12*abs(metrics[name][k]) for k in ['sse','sae'])
            checks[name]=dict(status='PASS',absolute_errors=errors)
        write(out/'series_keys.json',keys)
        write(out/'bindings.json',dict(plant=str(Path(plant_checkpoint).resolve()),observer=str(Path(observer_checkpoint).resolve()),
            target=str(data/'target.npy'),mask=str(data/'target_mask.npy'),config=str(root/'config.json')))
        write(out/'metrics.json',metrics);write(out/'independent_recompute.json',checks)
