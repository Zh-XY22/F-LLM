from pathlib import Path
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .runtime import read,write,table,task,lease
from .data import Source
from .method import seed,isolated_rng,rng_state,restore_rng,lip,prediction_loss,forward,grad_vector,finite,adapter,load_adapter

class Capture:
    def __init__(self,model):
        self.h=None
        self.handle=model.decoder.register_forward_pre_hook(self.capture)
    def capture(self,module,args):self.h=args[0]


def make_model(root,seed_value):
    from fllm.common import model_args
    from autotimes.models.AutoTimes_Llama import Model
    seed(seed_value)
    model=Model(model_args(read(Path(root)/'config.json'),root)).cuda()
    assert all(not p.requires_grad for p in model.llama.parameters())
    assert all(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.decoder.parameters())
    return model,Capture(model)


def batch(ds,indices):
    return torch.from_numpy(np.stack([ds[i][0] for i in indices])).cuda(),torch.from_numpy(np.stack([ds[i][1] for i in indices])).cuda()


def torch_save(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    torch.save(value,tmp);os.replace(tmp,path)


def initialize_record(root,model,seed_value):
    dest=Path(root)/'initialization'/f'plant{seed_value}'
    with lease(dest):
        values=adapter(model)
        p=dest/'adapter.pt'
        if p.exists():
            old=torch.load(p,map_location='cpu',weights_only=True)
            assert set(values)==set(old) and all(torch.equal(values[k],old[k]) for k in values)
        else:torch_save(p,values)
    return str(p)


def stats(ratio,delta,xi,kappa):
    r=ratio.detach().double().cpu().numpy().reshape(-1)
    return dict(ratio_q00=float(np.min(r)),ratio_q25=float(np.quantile(r,.25)),ratio_q50=float(np.quantile(r,.5)),
                ratio_q75=float(np.quantile(r,.75)),ratio_q95=float(np.quantile(r,.95)),ratio_max=float(np.max(r)),
                hinge_active=float((r>kappa).mean()),delta_norm_mean=float(delta.detach().norm(dim=-1).mean()),
                delta_to_radius_mean=float((delta.detach().norm(dim=-1)/xi.detach().squeeze(-1)).mean()))


def probe(root,model,capture,objective,kappa,weight,tag):
    with isolated_rng():
        modes=[(m,m.training) for m in model.modules()]
        model.eval()
        x,y=batch(Source(root,'train'),read(Path(root)/'calibration_keys.json')['probe_rows'])
        config=read(Path(root)/'config.json');lc=config['lipschitz']
        generator=torch.Generator(device='cuda').manual_seed(lc['noise_seed'])
        pred=forward(model,x,config['plant']['use_amp']);h=capture.h
        p=prediction_loss(pred,y,objective)
        l,r,d,xi=lip(model.decoder,h,generator,kappa,lc['radius_multiplier'],lc['interior_fraction'],lc['denominator_epsilon'],lc['direction_epsilon'])
        decoder=list(model.decoder.parameters())
        gp=grad_vector(p,decoder);gl=grad_vector(l,decoder)
        ge=grad_vector(p,list(model.encoder.parameters()))
        finite([pred,h,p,l,gp,gl,ge], 'NONFINITE_PROBE')
        assert gp.norm()>0 and ge.norm()>0
        assert all(v.grad is None and not v.requires_grad for v in model.llama.parameters())
        pn=float(gp.norm());ln=float(gl.norm())
        record=dict(tag=tag,objective=objective,kappa=kappa,lambda_lip=weight,prediction_loss=float(p),lip_loss=float(l),
                    g_pred_decoder=pn,g_lip_decoder=ln,g_pred_encoder=float(ge.norm()),
                    weighted_gradient_ratio=weight*ln/pn,gradient_cosine=float(torch.dot(gp,gl)/(gp.norm()*gl.norm())) if ln>0 else None,
                    activity='ACTIVE' if ln>0 else 'INACTIVE',backbone_frozen=True,optimizer_steps=0,**stats(r,d,xi,kappa))
        for module,mode in modes:module.training=mode
        return record


def plant(root,entry):
    pid=entry['task_id']
    with task(root,pid,'PLANT',entry['depends_on'].split(';')) as out:
        if out is None:return
        cfg=read(Path(root)/'config.json');pcfg=cfg['plant'];lc=cfg['lipschitz']
        objective=pcfg['loss'];weight=lc['lambda_lip'];kappa=lc['kappa']
        seed_value=cfg['plant_seed']
        model,capture=make_model(root,seed_value)
        init=initialize_record(root,model,seed_value)
        torch_save(out/'initial_adapter.pt',adapter(model))
        write(out/'effective_config.json',dict(config=cfg,task_id=pid,initialization=init,
              selection='native source VAL batch-mean; equality updates best'))
        train_ds=Source(root,'train');val_ds=Source(root,'val')
        generator=torch.Generator().manual_seed(seed_value)
        train_loader=DataLoader(train_ds,batch_size=pcfg['batch_size'],shuffle=True,drop_last=False,num_workers=pcfg['num_workers'],generator=generator)
        val_generator=torch.Generator().manual_seed(seed_value+10000)
        val_loader=DataLoader(val_ds,batch_size=pcfg['batch_size'],shuffle=False,num_workers=pcfg['num_workers'],generator=val_generator)
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=pcfg['learning_rate'],weight_decay=pcfg['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=pcfg['scheduler_tmax'],eta_min=pcfg['scheduler_eta_min'])
        scaler=torch.cuda.amp.GradScaler(enabled=pcfg['use_amp'])
        lip_generator=torch.Generator(device='cuda').manual_seed(lc['noise_seed'])
        seed(seed_value)
        history=[];probes=[];best=float('inf');best_epoch=0;bad=0;first=1;total_updates=0;total_skips=0
        if (out/'resume.pt').exists():
            # This local file is generated by this command and includes full RNG state.
            state=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
            load_adapter(model,state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
            scaler.load_state_dict(state['scaler']);restore_rng(state['rng']);generator.set_state(state['loader_rng']);val_generator.set_state(state['val_rng']);lip_generator.set_state(state['lip_rng'])
            history=state['history'];probes=state['probes'];best=state['best'];best_epoch=state['best_epoch'];bad=state['bad'];first=state['epoch']+1
            total_updates=state['updates'];total_skips=state['skips']
        else:probes.append(probe(root,model,capture,objective,kappa,weight,'INITIALIZATION'))
        start=time.time()
        for epoch in range(first,pcfg['train_epochs']+1):
            if bad>=pcfg['patience']:break
            model.train();preds=[];lips=[];totals=[];ratios=[];delta_norm=[];updates=0;skips=0;order=[]
            epoch_start=time.time()
            for x,y,ids in train_loader:
                x=x.cuda();y=y.cuda();optimizer.zero_grad(set_to_none=True)
                pred=forward(model,x,pcfg['use_amp']);h=capture.h
                p=prediction_loss(pred,y,objective)
                l=p.new_zeros(())
                if weight:
                    l,r,d,xi=lip(model.decoder,h,lip_generator,kappa,lc['radius_multiplier'],lc['interior_fraction'],lc['denominator_epsilon'],lc['direction_epsilon'])
                    ratios.extend(r.detach().cpu().numpy().reshape(-1).tolist());delta_norm.append(float(d.norm(dim=-1).mean()))
                total=p+weight*l
                finite([x,y,pred,p,l,total],'NONFINITE_PLANT_FORWARD')
                scale=scaler.get_scale();scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                valid=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad)
                scaler.step(optimizer);scaler.update()
                skip=scaler.get_scale()<scale
                if not valid and not skip:raise FloatingPointError('UNHANDLED_NONFINITE_GRADIENT')
                if skip:skips+=1
                else:updates+=1
                finite([p for p in model.parameters() if p.requires_grad],'NONFINITE_UPDATED_ADAPTER')
                assert all(p.grad is None for p in model.llama.parameters())
                order.extend(ids.tolist());preds.append(float(p));lips.append(float(l));totals.append(float(total))
            np.save(out/f'epoch{epoch:02d}_sample_rows.npy',np.asarray(order,np.int64),allow_pickle=False)
            assert sorted(order)==list(range(len(train_ds)))
            model.eval();val=[]
            with torch.no_grad():
                for x,y,ids in val_loader:
                    pred=forward(model,x.cuda(),pcfg['use_amp']).cpu()
                    loss=prediction_loss(pred,y,objective);finite([pred,loss],'NONFINITE_SOURCE_VAL')
                    val.append(float(loss))
            validation=float(np.mean(np.asarray(val,dtype=np.float32)))
            improved=validation<=best
            if improved:
                best=validation;best_epoch=epoch;bad=0;torch_save(out/'best_adapter.pt',adapter(model))
            else:bad+=1
            torch_save(out/f'epoch{epoch:02d}.pt',adapter(model))
            total_updates+=updates;total_skips+=skips
            record=dict(task_id=pid,epoch=epoch,prediction_loss=float(np.mean(preds)),lip_loss=float(np.mean(lips)),
                weighted_lip_loss=weight*float(np.mean(lips)),total_loss=float(np.mean(totals)),source_val=validation,
                lambda_lip=weight,kappa=kappa,radius_multiplier=lc['radius_multiplier'],optimizer_updates=updates,AMP_skips=skips,
                best_epoch=best_epoch,checkpoint_saved=improved,patience_counter=bad,learning_rate=optimizer.param_groups[0]['lr'],
                samples=len(order),wall_seconds=time.time()-epoch_start,
                ratio_q25=float(np.quantile(ratios,.25)) if ratios else None,
                ratio_q50=float(np.quantile(ratios,.5)) if ratios else None,
                ratio_q75=float(np.quantile(ratios,.75)) if ratios else None,
                ratio_q95=float(np.quantile(ratios,.95)) if ratios else None,
                hinge_active=float(np.mean(np.asarray(ratios)>kappa)) if ratios else None,
                delta_norm_mean=float(np.mean(delta_norm)) if delta_norm else None)
            history.append(record);table(out/'training.tsv',history)
            if epoch==1:probes.append(probe(root,model,capture,objective,kappa,weight,'EPOCH1'))
            if bad<pcfg['patience']:scheduler.step()
            torch_save(out/'resume.pt',dict(model=adapter(model),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),rng=rng_state(),loader_rng=generator.get_state(),val_rng=val_generator.get_state(),
                lip_rng=lip_generator.get_state(),history=history,probes=probes,best=best,best_epoch=best_epoch,bad=bad,
                epoch=epoch,updates=total_updates,skips=total_skips))
            print(pid,record,flush=True)
        if total_updates<1:raise FloatingPointError('NO_EFFECTIVE_OPTIMIZER_UPDATE')
        load_adapter(model,torch.load(out/'best_adapter.pt',map_location='cpu',weights_only=True))
        probes.append(probe(root,model,capture,objective,kappa,weight,'SELECTED_CHECKPOINT'))
        table(out/'sensitivity_probes.tsv',probes)
        write(out/'result.json',dict(task_id=pid,epochs=len(history),best_epoch=best_epoch,best_source_val=best,
            optimizer_updates=total_updates,AMP_skips=total_skips,checkpoint=str(out/'best_adapter.pt'),
            wall_seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),initialization=init))

