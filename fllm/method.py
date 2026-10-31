"""Paired losses and three rollout paths. Predictions accept no future labels."""
import contextlib
import random
import numpy as np
import torch
from torch import nn


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(value):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch'])
    if value['cuda']: torch.cuda.set_rng_state_all(value['cuda'])


@contextlib.contextmanager
def isolated_rng():
    state=rng_state()
    try: yield
    finally: restore_rng(state)


@contextlib.contextmanager
def deterministic(decoder):
    modes=[(m,m.training) for m in decoder.modules()]
    decoder.eval()
    try: yield
    finally:
        for m,training in modes:m.training=training


def lip(decoder,hidden,generator,kappa,radius=.02,interior_fraction=.5,denominator_epsilon=1e-8,direction_epsilon=1e-12):
    with torch.autocast(device_type=hidden.device.type,enabled=False),deterministic(decoder):
        h=hidden.float()
        direction=torch.randn(h.shape,device=h.device,dtype=torch.float32,generator=generator)
        direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(direction_epsilon)
        xi=radius*h.detach().square().mean(-1,keepdim=True).sqrt()
        delta=(interior_fraction*xi*direction).detach()
        dn=delta.norm(dim=-1)
        if not bool(((dn>0)&(dn<xi.squeeze(-1))).all()):raise FloatingPointError('INVALID_LOCAL_BALL')
        ratio=(decoder(h+delta)-decoder(h)).norm(dim=-1)/dn.clamp_min(denominator_epsilon)
        loss=torch.relu(ratio-kappa).square().mean()
    return loss,ratio,delta,xi


def prediction_loss(pred,target,objective):
    pred=pred.float();target=target.float()
    if objective=='SMAPE':return (200*(pred-target).abs()/(pred.abs()+target.abs()+1e-8)).mean()
    if objective=='PAPER_L2':return (pred-target).reshape(len(pred),-1,24).norm(dim=-1).mean()
    raise ValueError(objective)


def observer():
    return nn.Sequential(nn.Linear(24,192),nn.Tanh(),nn.Dropout(.1),nn.Linear(192,24))


def observer_loss(pred,target,mask,sigma,kind):
    diff=(pred-target)*mask
    if kind=='MSE_NORMALIZED':return (diff.square().sum(-1)/mask.sum(-1).clamp_min(1)).mean()
    if kind=='L2_NORMALIZED':return diff.norm(dim=-1).mean()
    if kind=='L2_PHYSICAL':return (diff*sigma).norm(dim=-1).mean()
    raise ValueError(kind)


def residual(x,mask,g):
    count=mask.sum(-1,keepdim=True).clamp_min(1)
    mean=(x*mask).sum(-1,keepdim=True)/count
    sigma=(((x-mean).square()*mask).sum(-1,keepdim=True)/count+1e-5).sqrt()
    rm=mask[:,24:]
    return (x[:,24:]-g[:,:24])*rm/sigma,sigma,rm


def correction(obs,x,mask,g,alpha=1.):
    r,s,rm=residual(x,mask,g)
    return alpha*s*obs(r)*(rm.sum(-1,keepdim=True)>0)


def forward(model,x,amp=True):
    with torch.autocast(device_type=x.device.type,enabled=amp):
        return model(x.unsqueeze(-1),None,None,None).squeeze(-1).float()


@torch.no_grad()
def predict(model,observers,history,mask,scored_points,feedback_factor=1.,alpha=1.,amp=True):
    raw_state=history.clone();raw_mask=mask.clone()
    states={name:history.clone() for name in observers};masks={name:mask.clone() for name in observers}
    output={'RAW':[]}
    for name in observers:output[name+'__POSTHOC']=[];output[name+'__CLOSED_LOOP']=[]
    for step in range((scored_points+23)//24):
        g=forward(model,raw_state,amp);p=g[:,-24:]
        output['RAW'].append(p)
        for name,obs in observers.items():
            c=correction(obs,raw_state,raw_mask,g,alpha)
            output[name+'__POSTHOC'].append(p+c)
            # All first-patch states are identical; later feedback states are never shared.
            cg=g if step==0 else forward(model,states[name],amp)
            cp=cg[:,-24:];cc=correction(obs,states[name],masks[name],cg,alpha)
            output[name+'__CLOSED_LOOP'].append(cp+cc)
            states[name]=torch.cat((states[name][:,24:],cp+feedback_factor*cc),-1)
            masks[name]=torch.cat((masks[name][:,24:],torch.ones_like(cp)),-1)
        raw_state=torch.cat((raw_state[:,24:],p),-1)
        raw_mask=torch.cat((raw_mask[:,24:],torch.ones_like(p)),-1)
    return {name:torch.cat(parts,-1)[:,:scored_points] for name,parts in output.items()}


def grad_vector(loss,params,retain_graph=True):
    gradients=torch.autograd.grad(loss,params,retain_graph=retain_graph,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1).float() for p,g in zip(params,gradients)])


def finite(values,name):
    if not all(bool(torch.isfinite(v).all()) for v in values):raise FloatingPointError(name)


def adapter(model):
    return {n:v.detach().cpu().clone() for n,v in model.state_dict().items() if not n.startswith('llama.')}


def load_adapter(model,value):
    expected=set(adapter(model))
    if set(value)!=expected:raise ValueError('ADAPTER_KEY_MISMATCH')
    model.load_state_dict(value,strict=False)
