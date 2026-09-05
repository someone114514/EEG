"""Validation/test evaluator for fixed-checkpoint independent K-step Band-TTT."""
from __future__ import annotations
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import average_precision_score,roc_auc_score
from chbmit_groupkfold.data import load_rows,make_eval_loader
from chbmit_groupkfold.meta_model import CHBMetaTTTModel
from chbmit_groupkfold.transforms import deterministic_band_view
from chbmit_groupkfold.band_v2 import BandStepSizes,independent_batch_logits
from chbmit_groupkfold.meta_evaluate import score_probabilities,select_validation_threshold
DEFAULT_RECORDINGS=Path('/root/b_false_alarm_atlas/manifests/recordings.parquet')
DEFAULT_SEIZURES=Path('/root/b_false_alarm_atlas/manifests/seizures.parquet')

def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);p.add_argument('--steps',type=int,choices=[1,3,5],required=True);p.add_argument('--split',choices=['validation','test'],required=True);p.add_argument('--allow-test',action='store_true');p.add_argument('--batch-size',type=int,default=32);p.add_argument('--workers',type=int,default=8);p.add_argument('--source-root',type=Path,default=Path('/root/b_false_alarm_atlas/outputs/reports/meta-ttt-chbmit-5fold-v1'));p.add_argument('--output-root',type=Path,default=Path('/root/b_false_alarm_atlas/outputs/reports/band-ttt-v2/independent'));a=p.parse_args()
 if a.split=='test' and not a.allow_test:raise PermissionError('test requires locked validation selection and --allow-test')
 d=a.output_root/f'k{a.steps}'/f'fold{a.fold}';d.mkdir(parents=True,exist_ok=True);marker=d/f'{a.split}_completed.json'
 if marker.exists():print(marker.read_text());return
 rows=load_rows(a.fold,a.split);loader=make_eval_loader(rows,batch_size=a.batch_size,workers=a.workers);pre=Path(__file__).parent/'pretrained_weights/pretrained_weights.pth';model=CHBMetaTTTModel(pre).cuda().eval();state=torch.load(a.source_root/'runs'/f'meta_band_fold{a.fold}_seed3407'/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model'],strict=True);rates=BandStepSizes(model.adaptive_named_parameters('band').keys(),False,float(state['alpha'])).cuda();ys=[];ps=[];started=time.monotonic();parts=d/f'{a.split}_parts';parts.mkdir(exist_ok=True);existing=sorted(parts.glob('part_*.parquet'));resume_rows=sum(len(pd.read_parquet(p,columns=['label'])) for p in existing);seen=0;part_index=len(existing)
 for bi,(x,y,sids) in enumerate(loader):
  if seen < resume_rows:
   if seen+len(y)>resume_rows:raise RuntimeError('partial part is not batch aligned')
   seen+=len(y);continue
  x=x.cuda(non_blocking=True);view,labels=deterministic_band_view(x,list(sids))
  with torch.enable_grad():logits=independent_batch_logits(model,x,view,labels,rates,a.steps)
  ys.append(y.numpy());ps.append(torch.sigmoid(logits.float()).detach().cpu().numpy());seen+=len(y)
  if len(ys)>=100 or seen==len(rows):
   n=sum(map(len,ys));part=rows.iloc[seen-n:seen][['patient','recording','start','end','label']].copy();part['probability']=np.concatenate(ps);part.to_parquet(parts/f'part_{part_index:05d}.parquet',index=False);part_index+=1;ys=[];ps=[];print(f'{a.split} f{a.fold} k{a.steps} checkpoint {seen}/{len(rows)}',flush=True)
 table=pd.concat([pd.read_parquet(p) for p in sorted(parts.glob('part_*.parquet'))],ignore_index=True);assert len(table)==len(rows);y=table.label.to_numpy();prob=table.probability.to_numpy();table.to_parquet(d/f'{a.split}_probabilities.parquet',index=False);recordings=pd.read_parquet(DEFAULT_RECORDINGS);seizures=pd.read_parquet(DEFAULT_SEIZURES)
 base={'status':'complete','fold':a.fold,'steps':a.steps,'split':a.split,'rows':len(rows),'alpha':float(state['alpha']),'context':'independent','step_parameterization':'checkpoint global scalar','window_auprc':float(average_precision_score(y,prob)),'window_auroc':float(roc_auc_score(y,prob)),'elapsed_s':time.monotonic()-started,'source_checkpoint':str(a.source_root/'runs'/f'meta_band_fold{a.fold}_seed3407'/'best.pt')}
 if a.split=='validation':
  sweep,selected=select_validation_threshold(table,seizures,recordings);sweep.to_parquet(d/'validation_threshold_sweep.parquet',index=False);base['selected_event_operating_point']=selected
 else:
  lock=json.loads((d/'validation_completed.json').read_text());base['selected_event_operating_point']=score_probabilities(table,seizures,recordings,float(lock['selected_event_operating_point']['threshold']))
 marker.write_text(json.dumps(base,indent=2,allow_nan=True)+'\n');print(json.dumps(base),flush=True)
if __name__=='__main__':main()
