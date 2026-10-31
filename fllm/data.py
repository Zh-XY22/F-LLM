"""Native source boundaries; explicit physical coordinates and immutable keys."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .runtime import read, write, table


class Source(Dataset):
    def __init__(self, root, split):
        path=Path(root)/'data'/('source_'+split)
        self.x=np.load(path/'history.npy',mmap_mode='r')
        self.y=np.load(path/'labels.npy',mmap_mode='r')
        self.keys=read(path/'keys.json')
    def __len__(self): return len(self.x)
    def __getitem__(self,index):
        return np.array(self.x[index],copy=True),np.array(self.y[index],copy=True),index


def prepare_source(root):
    root=Path(root)
    from autotimes.data_provider.data_loader import Dataset_TSF
    from autotimes.utils.tools import convert_tsf_to_dataframe
    cfg=read(root/'config.json')
    path=Path(cfg['source_tsf'])
    frame,*metadata=convert_tsf_to_dataframe(str(path))
    series=[]
    for i,row in frame.iterrows():
        values=np.asarray(row.series_value,dtype=np.float32)
        missing=int(np.isnan(values).sum())
        values=values[~np.isnan(values)]
        if not np.isfinite(values).all(): raise ValueError(f'NONFINITE_SOURCE:{i}')
        pad=max(72-len(values),0)
        length=len(values)+pad
        series.append(dict(series_id=str(row.series_name),series_index=int(i),length=len(values),
                           removed_missing=missing,left_padding=pad,
                           train_windows=max(0,length-119),
                           last_train_label_end=length-48-pad if length>=120 else None,
                           val_origin=length-48-pad if length>=96 else None,
                           val_future_end=length-24-pad if length>=96 else None,
                           val_status='ELIGIBLE_NATIVE' if length>=96 else 'NO_NATIVE_LEGAL_VAL_WINDOW'))
    splits={}
    for split in ('train','val'):
        ds=Dataset_TSF(root_path=str(path.parent),data_path=path.name,flag=split,size=[48,24,24],drop_short=False)
        x=[]; y=[]; keys=[]
        for index in range(len(ds)):
            xx,yy,_,_=ds[index]
            si=int(ds.series_ids[index]); local=int(index-ds.length_offsets[index])
            meta=series[si]
            origin=(local+48 if split=='train' else meta['length']+meta['left_padding']-48)
            physical=origin-meta['left_padding']
            native_values=ds.timeseries[si]
            assert np.array_equal(np.asarray(xx).reshape(48),native_values[origin-48:origin])
            assert np.array_equal(np.asarray(yy).reshape(48),native_values[origin-24:origin+24])
            assert physical+24 <= meta['length']-(48 if split=='train' else 24)
            if split=='train': assert physical+24 <= meta['length']-48
            x.append(np.asarray(xx).reshape(48)); y.append(np.asarray(yy).reshape(48))
            keys.append(dict(row=index,series_id=meta['series_id'],series_index=si,cut_point=origin,
                             forecast_origin=physical,history_start=physical-48,
                             supervised_start=physical-24,supervised_end=physical+24,
                             future_start=physical,future_end=physical+24,first18_end=physical+18))
        out=root/'data'/('source_'+split);out.mkdir(parents=True,exist_ok=True)
        np.save(out/'history.npy',np.asarray(x,dtype=np.float32),allow_pickle=False)
        np.save(out/'labels.npy',np.asarray(y,dtype=np.float32),allow_pickle=False)
        write(out/'keys.json',keys)
        splits[split]=dict(samples=len(keys),keys=str(out/'keys.json'),history=str(out/'history.npy'),labels=str(out/'labels.npy'))
    assert splits['train']['samples']==sum(s['train_windows'] for s in series)
    assert len(set(s['series_id'] for s in series))==len(series)
    write(root/'source_split_manifest.json',dict(source_path=str(path),series=series,splits=splits,
        train_labels_disjoint_from_val_future=True,native_missing='Remove NaN values exactly as native loader; coordinates after removal; count recorded',
        native_val='L-48 origin, predict24; first18 selection; final24 source values unused',
        train='Native full48 shifted labels end no later than L-48; observer future inherits last24',
        source_val_not_observer_train=True))
    indices=torch.randperm(splits['train']['samples'],generator=torch.Generator().manual_seed(2022))[:128].tolist()
    write(root/'calibration_keys.json',dict(rows=indices,batches=[indices[i:i+16] for i in range(0,128,16)],
        probe_rows=indices[:16],source_keys=str(root/'data/source_train/keys.json')))


def prepare_target(root):
    root=Path(root)
    from autotimes.utils.tools import convert_tsf_to_dataframe
    cfg=read(root/'config.json')
    target,*metadata=convert_tsf_to_dataframe(cfg['target_tsf'])
    assert len(target)==48000
    target_rows=[]; groups={18:[]}
    for i,row in target.iterrows():
        v=np.asarray(row.series_value,dtype=np.float32); missing=int(np.isnan(v).sum());v=v[~np.isnan(v)]
        if not np.isfinite(v).all() or len(v)<18: raise ValueError(f'NONFINITE_OR_SHORT_TARGET:{i}')
        for h in (18,):
            origin=len(v)-18
            hist=v[max(0,origin-48):origin]
            xx=np.zeros(48,np.float32);mask=np.zeros(48,np.float32)
            if len(hist): xx[-len(hist):]=hist;mask[-len(hist):]=1
            key=dict(series_id=str(row.series_name),series_index=int(i),forecast_origin=origin,
                     history_start=max(0,origin-48),history_valid=len(hist),label_end=origin+h,
                     length=len(v),removed_missing=missing)
            groups[h].append((xx,mask,v[origin:origin+h],key))
            target_rows.append(dict(**key,scope='H18',eligible=True,reason=''))
    for h,values in groups.items():
        out=root/'data'/f'target_h{h}';out.mkdir(parents=True,exist_ok=True)
        for name,idx in [('history',0),('history_mask',1),('target',2)]:
            np.save(out/(name+'.npy'),np.stack([v[idx] for v in values]),allow_pickle=False)
        np.save(out/'target_mask.npy',np.ones((len(values),h),np.float32),allow_pickle=False)
        write(out/'keys.json',[v[3] for v in values])
        assert len(set(v[3]['series_id'] for v in values))==len(values)
    write(root/'target_split_manifest.json',dict(target_path=cfg['target_tsf'],expected_series=48000,
        formal_count=len(groups[18]),series=target_rows,
        values_used_in_selection=False,old_target_results_exposed=True))
