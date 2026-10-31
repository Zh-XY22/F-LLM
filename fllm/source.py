from pathlib import Path
import gc
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from .runtime import read,write,table,task,lease
from .data import Source
from .method import observer,observer_loss,residual,forward,finite,seed,rng_state,restore_rng,load_adapter
from .train import make_model,torch_save

def score_series(pred,truth,mask=None,epsilon=1e-8):
    pred=np.asarray(pred,dtype=np.float64);truth=np.asarray(truth,dtype=np.float64)
    mask=np.ones_like(truth) if mask is None else np.asarray(mask,dtype=np.float64)
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():raise FloatingPointError('NONFINITE_SCORING_INPUT')
    count=mask.sum(-1)
    if not (count>0).all():raise ValueError('EMPTY_TARGET_MASK')
    e=pred-truth;sse=(e*e*mask).sum(-1);sae=(np.abs(e)*mask).sum(-1)
    smape=(200*np.abs(e)/np.maximum(np.abs(pred)+np.abs(truth),epsilon)*mask).sum(-1)/count
    return np.stack((smape,sse/count,sae/count,sse,sae,count),axis=-1)


def aggregate(values):
    return dict(smape=float(values[:,0].mean()),mse=float(values[:,3].sum()/values[:,5].sum()),
        mae=float(values[:,4].sum()/values[:,5].sum()),sse=float(values[:,3].sum()),sae=float(values[:,4].sum()),
        samples=len(values),elements=int(values[:,5].sum()))


def build_cache(root,plant_id):
    cid=plant_id+'__cache'
    with task(root,cid,'CACHE',[plant_id]) as out:
        if out is None:return
        plant_root=Path(root)/'tasks'/plant_id
        assert read(plant_root/'status.json')['status']=='COMPLETE'
        result=read(plant_root/'result.json')
        checkpoint=Path(result['checkpoint'])
        cfg=read(Path(root)/'config.json')
        model,capture=make_model(root,cfg['plant_seed'])
        load_adapter(model,torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval().requires_grad_(False)
        manifest=dict(plant_id=plant_id,checkpoint=str(checkpoint),source_only=True,normalization='sigma of valid history; residual truth-prediction /sigma',splits={})
        for split,batch_size in [('train',256),('val',16)]:
            ds=Source(root,split)
            if (out/split/'manifest.json').exists():
                prior=read(out/split/'manifest.json')
                assert prior['samples']==len(ds) and prior['plant_checkpoint']==str(checkpoint)
                for name,shape in prior['shapes'].items():
                    array=np.load(out/split/(name+'.npy'),mmap_mode='r')
                    assert list(array.shape)==shape and np.isfinite(array).all()
                manifest['splits'][split]=dict(path=str(out/split),samples=len(ds),batch_size=batch_size)
                continue
            stage=out/(split+'.partial');stage.mkdir(exist_ok=True)
            names=['residual','target','future_mask','residual_mask','sigma','raw','truth','history_mask']
            shapes={name:(len(ds),1 if name=='sigma' else 48 if name=='history_mask' else 24) for name in names}
            arrays={name:np.lib.format.open_memmap(stage/(name+'.npy'),mode='w+',dtype=np.float32,shape=shape) for name,shape in shapes.items()}
            count=0;max_roundtrip=0.
            with torch.no_grad():
                for x,y,ids in DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0):
                    x=x.cuda();y=y.cuda();mask=torch.ones_like(x);g=forward(model,x,cfg['plant']['use_amp'])
                    r,s,rm=residual(x,mask,g);truth=y[:,-24:];p=g[:,-24:];target=(truth-p)/s
                    values=dict(residual=r,target=target,future_mask=torch.ones_like(target),residual_mask=rm,
                                sigma=s,raw=p,truth=truth,history_mask=mask)
                    finite(list(values.values()),'NONFINITE_CACHE')
                    assert ids.tolist()==list(range(count,count+len(x)))
                    for name,v in values.items():arrays[name][count:count+len(x)]=v.cpu().numpy()
                    max_roundtrip=max(max_roundtrip,float((target*s+p-truth).abs().max()))
                    count+=len(x)
            assert count==len(ds)
            for name,a in arrays.items():
                a.flush()
                assert a.shape==shapes[name] and np.isfinite(a).all()
            arrays.clear()
            write(stage/'keys.json',ds.keys)
            write(stage/'manifest.json',dict(samples=count,shapes=shapes,plant_checkpoint=str(checkpoint),split=split,
                physical_roundtrip_max_absolute_error=max_roundtrip,batch_size=batch_size,source_labels=split,
                observer_training_allowed=split=='train',keys=str(Path(root)/'data'/('source_'+split)/'keys.json')))
            os.replace(stage,out/split)
            manifest['splits'][split]=dict(path=str(out/split),samples=count,batch_size=batch_size)
        write(out/'cache_manifest.json',manifest)
        del model,capture;gc.collect();torch.cuda.empty_cache()


def cache_arrays(path):
    return {n:np.load(Path(path)/(n+'.npy'),mmap_mode='r') for n in ['residual','target','future_mask','residual_mask','sigma','raw','truth']}


@torch.no_grad()
def validate(obs,arrays,out,epoch):
    obs.eval();corrections=[]
    for begin in range(0,len(arrays['raw']),256):
        r=torch.from_numpy(np.array(arrays['residual'][begin:begin+256],copy=True)).cuda()
        s=torch.from_numpy(np.array(arrays['sigma'][begin:begin+256],copy=True)).cuda()
        rm=torch.from_numpy(np.array(arrays['residual_mask'][begin:begin+256],copy=True)).cuda()
        corrections.append((s*obs(r)*(rm.sum(-1,keepdim=True)>0)).cpu().numpy())
    c=np.concatenate(corrections);raw=np.asarray(arrays['raw']);truth=np.asarray(arrays['truth'])
    corrected=raw+c
    np.save(out/f'val_epoch{epoch:02d}_correction.npy',c,allow_pickle=False)
    records=[]
    for horizon in (18,24):
        values=score_series(corrected[:,:horizon],truth[:,:horizon]);base=score_series(raw[:,:horizon],truth[:,:horizon])
        np.save(out/f'val_epoch{epoch:02d}_h{horizon}_series.npy',values,allow_pickle=False)
        e=truth[:,:horizon].astype(np.float64)-raw[:,:horizon].astype(np.float64)
        cc=c[:,:horizon].astype(np.float64);energy=(cc*cc).mean();alignment=(e*cc).mean()
        # Float32 reported addition has rounding; evaluate exact decomposition in float64 separately.
        exact=score_series(raw[:,:horizon].astype(np.float64)+cc,truth[:,:horizon])
        diff=aggregate(exact)['mse']-aggregate(base)['mse']
        discrepancy=diff-(energy-2*alignment)
        if abs(discrepancy)>1e-8*max(1,abs(diff),energy,abs(alignment)):raise ValueError('SSE_DECOMPOSITION_MISMATCH')
        denom=np.sqrt(np.mean(e*e)*energy)
        centered_denom=np.sqrt(np.sum((e-e.mean())**2)*np.sum((cc-cc.mean())**2))
        row=dict(epoch=epoch,horizon=horizon,**aggregate(values),raw_smape=aggregate(base)['smape'],
            raw_mse=aggregate(base)['mse'],raw_mae=aggregate(base)['mae'],delta_smape=aggregate(values)['smape']-aggregate(base)['smape'],
            delta_mse=aggregate(values)['mse']-aggregate(base)['mse'],delta_mae=aggregate(values)['mae']-aggregate(base)['mae'],
            correction_bias=float(cc.mean()),correction_rms=float(np.sqrt(energy)),correction_energy=float(energy),
            twice_alignment=float(2*alignment),direction_cosine=float(alignment/denom) if denom else None,
            correlation=float(np.sum((e-e.mean())*(cc-cc.mean()))/centered_denom) if centered_denom else None,
            direction_agreement=float(np.mean(np.sign(e)==np.sign(cc))),decomposition_absolute_error=float(abs(discrepancy)),
            finite=True)
        records.append(row)
    return records


def train_observer(root,entry):
    oid=entry['task_id'];pid=entry['plant_id'];cid=pid+'__cache'
    with task(root,oid,'OBSERVER',[cid]) as out:
        if out is None:return
        cache=Path(root)/'tasks'/cid
        assert read(cache/'status.json')['status']=='COMPLETE'
        arrays=cache_arrays(cache/'train');val=cache_arrays(cache/'val')
        cfg=read(Path(root)/'config.json');oc=cfg['observer']
        obs_seed=cfg['observer_seed'];seed(obs_seed);obs=observer().cuda()
        initial={n:v.detach().cpu().clone() for n,v in obs.state_dict().items()}
        init_root=Path(root)/'initialization'/f'observer{obs_seed}'
        with lease(init_root):
            path=init_root/'observer.pt'
            if path.exists():
                old=torch.load(path,map_location='cpu',weights_only=True);assert all(torch.equal(initial[k],old[k]) for k in initial)
            else:torch_save(path,initial)
        torch_save(out/'initial_observer.pt',initial)
        optimizer=torch.optim.AdamW(obs.parameters(),lr=oc['learning_rate'],weight_decay=oc['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=oc['cosine_epochs'],eta_min=oc['scheduler_eta_min'])
        gen=torch.Generator().manual_seed(obs_seed)
        tensor_names=['residual','target','future_mask','residual_mask','sigma']
        ds=TensorDataset(*[torch.from_numpy(np.array(arrays[n],copy=True)) for n in tensor_names],torch.arange(len(arrays['raw'])))
        loader=DataLoader(ds,batch_size=oc['batch_size'],shuffle=True,drop_last=False,num_workers=0,generator=gen)
        # Reproduce original first-five RNG rule: model construction consumes the seeded stream.
        start=time.time();history=[];validations=[];first=1
        write(out/'effective_config.json',dict(config=cfg,task_id=oid,source_cache=str(cache/'train'),
            val_cache=str(cache/'val'),initialization=str(init_root/'observer.pt')))
        if (out/'resume.pt').exists():
            # Only this command's local resume file contains Python/NumPy RNG objects.
            state=torch.load(out/'resume.pt',map_location='cpu',weights_only=False);obs.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler']);restore_rng(state['rng']);gen.set_state(state['loader_rng'])
            history=state['history'];validations=state['validations'];first=state['epoch']+1
        for epoch in range(first,oc['epochs']+1):
            if epoch>oc['cosine_epochs']:
                for group in optimizer.param_groups:group['lr']=oc['learning_rate_after_cosine']
            obs.train();total=0.;count=0;updates=0;order=[]
            for r,target,mask,rm,sigma,ids in loader:
                r,target,mask,rm,sigma=[v.cuda() for v in (r,target,mask,rm,sigma)]
                pred=obs(r)*(rm.sum(-1,keepdim=True)>0)
                loss=observer_loss(pred,target,mask,sigma,oc['loss']);finite([loss,pred],'OBSERVER_NONFINITE_FORWARD')
                optimizer.zero_grad(set_to_none=True);loss.backward();finite([p.grad for p in obs.parameters()],'OBSERVER_NONFINITE_GRADIENT')
                optimizer.step();finite(list(obs.parameters()),'OBSERVER_NONFINITE_PARAMETERS')
                total+=float(loss)*len(r);count+=len(r);updates+=1;order.extend(ids.tolist())
            if epoch<=oc['cosine_epochs']:scheduler.step()
            assert sorted(order)==list(range(len(ds)))
            np.save(out/f'epoch{epoch:02d}_sample_rows.npy',np.asarray(order,np.int64),allow_pickle=False)
            history.append(dict(task_id=oid,epoch=epoch,loss=oc['loss'],training_loss=total/count,samples=count,
                                optimizer_updates=updates,AMP_skips=0,next_lr=optimizer.param_groups[0]['lr']))
            torch_save(out/f'epoch{epoch:02d}.pt',{n:v.detach().cpu().clone() for n,v in obs.state_dict().items()})
            if epoch in oc['validation_epochs']:
                validations.extend(validate(obs,val,out,epoch));table(out/'source_validation.tsv',validations)
            table(out/'training.tsv',history)
            torch_save(out/'resume.pt',dict(model=obs.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                rng=rng_state(),loader_rng=gen.get_state(),history=history,validations=validations,epoch=epoch))
        write(out/'result.json',dict(task_id=oid,epochs=oc['epochs'],optimizer_updates=sum(r['optimizer_updates'] for r in history),
            wall_seconds=time.time()-start,source_train=len(ds),source_val=len(val['raw']),finite=True,
            selection_checkpoints=[str(out/f'epoch{e:02d}.pt') for e in oc['validation_epochs']]))
        print(oid,'COMPLETE',time.time()-start,flush=True)

