import numpy as np

def independent(pred_path,target_path,mask_path,horizon,segments,epsilon):
    p=np.load(pred_path,mmap_mode='r');y=np.load(target_path,mmap_mode='r');m=np.load(mask_path,mmap_mode='r')
    assert p.shape==y.shape==m.shape and p.shape[1]==horizon
    per={name:[] for name in segments}
    # Separate chunked implementation: divide only valid positions, then sum each series.
    for begin in range(0,len(y),257):
        for name,(a,b) in segments.items():
            pp=np.array(p[begin:begin+257,a:b],dtype=np.float64)
            yy=np.array(y[begin:begin+257,a:b],dtype=np.float64)
            valid=np.array(m[begin:begin+257,a:b],dtype=np.float64)
            if not np.isfinite(pp).all() or not np.isfinite(yy).all():raise FloatingPointError('RECOMPUTE_NONFINITE')
            counts=valid.sum(1);assert (counts>0).all()
            diff=pp-yy;absolute=np.abs(diff)
            denom=np.abs(pp)+np.abs(yy);denom[denom<epsilon]=epsilon
            point=np.zeros_like(pp);np.divide(200*absolute,denom,out=point,where=valid>0)
            sse=np.sum(np.multiply(np.multiply(diff,diff),valid),axis=1)
            sae=np.sum(np.multiply(absolute,valid),axis=1)
            per[name].append(np.column_stack((point.sum(1)/counts,sse/counts,sae/counts,sse,sae,counts)))
    return {name:np.concatenate(parts) for name,parts in per.items()}

