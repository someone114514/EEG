"""Full sequential validation coverage for Band-SSL/classification gradient alignment."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import pandas as pd
import torch
from torch.nn import functional as F
from chbmit_groupkfold.data import load_rows, WindowDataset
from torch.utils.data import DataLoader
from chbmit_groupkfold.meta_model import CHBMetaTTTModel
from chbmit_groupkfold.transforms import deterministic_band_view
from chbmit_groupkfold.meta_train import set_seed


def main():
    p=argparse.ArgumentParser(); p.add_argument('--fold',type=int,required=True);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--source-root',type=Path,default=Path('/root/b_false_alarm_atlas/outputs/reports/meta-ttt-chbmit-5fold-v1'))
    p.add_argument('--output-root',type=Path,default=Path('/root/b_false_alarm_atlas/outputs/reports/band-ttt-v2/continuous_gradient_audit_v2'))
    a=p.parse_args(); a.output_root.mkdir(parents=True,exist_ok=True); out=a.output_root/f'fold{a.fold}_batches.parquet'; done=a.output_root/f'fold{a.fold}_completed.json'
    if done.exists(): print(done.read_text()); return
    set_seed(3407+a.fold); rows=load_rows(a.fold,'validation')
    # Every window is covered exactly once and no batch crosses a recording.
    batches=[]
    for _, group in rows.groupby(['patient','recording'],sort=False):
        ids=group.index.to_list()
        batches.extend(ids[i:i+a.batch_size] for i in range(0,len(ids),a.batch_size))
    loader=DataLoader(WindowDataset(rows),batch_sampler=batches,num_workers=a.workers,pin_memory=True,
                      persistent_workers=a.workers>0,prefetch_factor=4 if a.workers>0 else None)
    model=CHBMetaTTTModel(Path(__file__).parent/'pretrained_weights/pretrained_weights.pth').cuda().eval(); state=torch.load(a.source_root/'runs'/f'meta_band_fold{a.fold}_seed3407'/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model'],strict=True)
    named=model.adaptive_named_parameters('band'); shared=[(n,p) for n,p in named.items() if n.startswith('backbone.')]; names=[n for n,_ in shared]; params=[p for _,p in shared]; records=[]; offset=0; started=time.monotonic()
    for bi,(x,y,sids) in enumerate(loader):
        n=len(y); meta=rows.iloc[offset:offset+n]; offset+=n; x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
        view,band=deterministic_band_view(x,list(sids)); ssl=F.cross_entropy(model(view,mode='band'),band); cls=F.binary_cross_entropy_with_logits(model(x,mode='detect'),y)
        gs=torch.autograd.grad(ssl,params,retain_graph=True);gc=torch.autograd.grad(cls,params)
        dot=sum((u.double()*v.double()).sum() for u,v in zip(gs,gc));sn=sum(u.double().square().sum() for u in gs).sqrt();cn=sum(v.double().square().sum() for v in gc).sqrt();cos=float(dot/(sn*cn)) if sn>0 and cn>0 else None
        records.append({'fold':a.fold,'batch':bi,'row_start':offset-n,'row_end':offset,'rows':n,'patient_first':str(meta.patient.iloc[0]),'patient_last':str(meta.patient.iloc[-1]),'recording_first':str(meta.recording.iloc[0]),'recording_last':str(meta.recording.iloc[-1]),'positive_rows':int(y.sum()),'cosine':cos,'dot':float(dot),'ssl_norm':float(sn),'classification_norm':float(cn),'ssl_loss':float(ssl.detach()),'classification_loss':float(cls.detach())})
        if (bi+1)%250==0: print(f'fold{a.fold} {offset}/{len(rows)}',flush=True)
    frame=pd.DataFrame(records);frame.to_parquet(out,index=False); payload={'status':'complete','fold':a.fold,'windows':len(rows),'batches':len(frame),'cosine_mean':float(frame.cosine.mean()),'negative_batch_fraction':float((frame.cosine<0).mean()),'elapsed_s':time.monotonic()-started,'split':'validation','test_partition_read':False,'labels_used_for_adaptation':False,'aggregation':'chronological contiguous batches, strictly contained within patient and recording','complete_coverage':int(frame.rows.sum())==len(rows),'cross_recording_batches':int(((frame.patient_first!=frame.patient_last)|(frame.recording_first!=frame.recording_last)).sum())};done.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload),flush=True)
if __name__=='__main__':main()
