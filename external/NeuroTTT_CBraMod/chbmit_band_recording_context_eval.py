"""Causal recording-context Band-TTT evaluation on fixed v1 checkpoints."""
from __future__ import annotations
import argparse,json,time,re
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.nn import functional as F
from torch.func import functional_call
from sklearn.metrics import average_precision_score,roc_auc_score
from chbmit_groupkfold.data import load_rows,WindowDataset
from chbmit_groupkfold.meta_model import CHBMetaTTTModel
from chbmit_groupkfold.transforms import deterministic_band_view
from chbmit_groupkfold.meta_evaluate import score_probabilities,select_validation_threshold

REC=Path('/root/b_false_alarm_atlas/manifests/recordings.parquet');SEIZ=Path('/root/b_false_alarm_atlas/manifests/seizures.parquet')

def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);p.add_argument('--steps',type=int,choices=[1,3,5],default=1);p.add_argument('--split',choices=['validation','test'],required=True);p.add_argument('--allow-test',action='store_true');p.add_argument('--source-root',type=Path,default=Path('/root/b_false_alarm_atlas/outputs/reports/meta-ttt-chbmit-5fold-v1'));p.add_argument('--output-root',type=Path,default=Path('/root/b_false_alarm_atlas/outputs/reports/band-ttt-v2/recording_context'));a=p.parse_args()
 if a.split=='test' and not a.allow_test:raise PermissionError('test requires validation lock')
 out=a.output_root/f'k{a.steps}'/f'fold{a.fold}';parts=out/f'{a.split}_recordings';parts.mkdir(parents=True,exist_ok=True);marker=out/f'{a.split}_completed.json'
 if marker.exists():print(marker.read_text());return
 rows=load_rows(a.fold,a.split);pre=Path(__file__).parent/'pretrained_weights/pretrained_weights.pth';model=CHBMetaTTTModel(pre).cuda().eval();ckpt=a.source_root/'runs'/f'meta_band_fold{a.fold}_seed3407'/'best.pt';state=torch.load(ckpt,map_location='cpu',weights_only=False);model.load_state_dict(state['model'],strict=True);alpha=float(state['alpha']);source=model.adaptive_named_parameters('band');dataset=WindowDataset(rows);started=time.monotonic();done_recs={p.stem for p in parts.glob('*.parquet')}
 for ri,((patient,recording),g) in enumerate(rows.groupby(['patient','recording'],sort=False)):
  safe=re.sub(r'[^A-Za-z0-9_.-]','_',str(recording));partpath=parts/f'{safe}.parquet'
  if safe in done_recs:continue
  ids=g.index.to_list(); current={n:p.detach().clone().requires_grad_(True) for n,p in source.items()};probs=[];next_support=0;last_support_end=-float('inf')
  for local,idx in enumerate(ids):
   query=rows.iloc[idx]
   # Consume the latest eligible support only when it does not overlap the
   # previously consumed support. Query itself is never used before prediction.
   while next_support<local and float(rows.iloc[ids[next_support]].end)<=float(query.start):
    support=rows.iloc[ids[next_support]]
    if float(support.start)>=last_support_end:
     sx,_,sid=dataset[idx-(local-next_support)];sx=sx[None].cuda();view,label=deterministic_band_view(sx,[sid])
     for _ in range(a.steps):
      loss=F.cross_entropy(functional_call(model,current,(view,),{'mode':'band'},strict=False),label);grads=torch.autograd.grad(loss,tuple(current.values()),allow_unused=True);current={n:(v if grad is None else v-alpha*grad).detach().requires_grad_(True) for (n,v),grad in zip(current.items(),grads)}
     last_support_end=float(support.end)
    next_support+=1
   qx,_,_=dataset[idx];qx=qx[None].cuda()
   with torch.no_grad():logit=functional_call(model,current,(qx,),{'mode':'detect'},strict=False)
   probs.append(float(torch.sigmoid(logit.float()).cpu()))
  part=g[['patient','recording','start','end','label']].copy();part['probability']=probs;part['updates']=sum(1 for i in range(1,len(ids)) if float(rows.iloc[ids[i]].start)>=float(rows.iloc[ids[0]].start)+10*i);part.to_parquet(partpath,index=False);print(f'f{a.fold} {a.split} recording {ri+1} rows={len(g)}',flush=True)
 table=pd.concat([pd.read_parquet(p) for p in sorted(parts.glob('*.parquet'))],ignore_index=True).sort_values(['patient','recording','start'],kind='stable');assert len(table)==len(rows);table.to_parquet(out/f'{a.split}_probabilities.parquet',index=False);recordings=pd.read_parquet(REC);seizures=pd.read_parquet(SEIZ);y=table.label.to_numpy();prob=table.probability.to_numpy();base={'status':'complete','fold':a.fold,'steps':a.steps,'split':a.split,'context':'causal_recording_persistent_nonoverlap_support','rows':len(table),'alpha':alpha,'window_auprc':float(average_precision_score(y,prob)),'window_auroc':float(roc_auc_score(y,prob)),'elapsed_s':time.monotonic()-started,'test_partition_read':a.split=='test'}
 if a.split=='validation':_,base['selected_event_operating_point']=select_validation_threshold(table,seizures,recordings)
 else:base['selected_event_operating_point']=score_probabilities(table,seizures,recordings,float(json.loads((out/'validation_completed.json').read_text())['selected_event_operating_point']['threshold']))
 marker.write_text(json.dumps(base,indent=2)+'\n');print(json.dumps(base),flush=True)
if __name__=='__main__':main()
