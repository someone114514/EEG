"""Aggregate the completed Joint-TTT, MT3-style, and label-prior runs.

This is a read-only reporting script.  It never chooses a threshold and never
opens a new test probability file for scoring; it consumes per-run manifests
written by the one-pass evaluators.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import pandas as pd

ROOT = Path("/root/b_false_alarm_atlas")
OUT = ROOT / "outputs/reports/cbramod-ttt-method-comparison-v1"

def read(path):
    return json.loads(path.read_text()) if path.exists() else None

def pooled(rows):
    if not rows: return {}
    tp=sum(float(x.get("true_positive_events",0)) for x in rows)
    fa=sum(float(x.get("false_alarm_events",0)) for x in rows)
    truth=sum(float(x.get("truth_events",0)) for x in rows)
    hours=sum(float(x.get("nonseizure_hours",0)) for x in rows)
    delays=[float(x["detection_delay_mean_s"]) for x in rows if x.get("detection_delay_mean_s") is not None]
    return {"true_positive_events":tp,"false_alarm_events":fa,"truth_events":truth,
            "event_sensitivity":tp/truth if truth else float("nan"),
            "fa_per_24h":fa*24/hours if hours else float("nan"),
            "nonseizure_hours":hours,
            "detection_delay_mean_s":sum(delays)/len(delays) if delays else float("nan")}

def ttt_summary(namespace):
    base=ROOT/"outputs/reports"/namespace/"evaluation"
    records=[]; frozen=[]; adapted=[]
    for path in sorted(base.glob("fold*_seed*/manifest.json")):
        m=read(path)
        if not m: continue
        records.append({"fold":m.get("fold"),"seed":m.get("seed"),"status":"complete","threshold":m.get("threshold"),"source_checkpoint_sha256":m.get("source_checkpoint_sha256")})
        frozen.extend(m.get("test_frozen_metrics",[])); adapted.extend(m.get("test_adapted_metrics",[]))
    return {"namespace":namespace,"runs":len(records),"run_records":records,"frozen_pooled":pooled(frozen),"adapted_pooled":pooled(adapted)}

def prior_summary(namespace):
    base=ROOT/"outputs/reports"/namespace/"evaluation"
    records=[]; test=[]; gates=[]
    for path in sorted(base.glob("fold*_seed*/manifest.json")):
        m=read(path)
        if not m: continue
        records.append({"fold":m.get("fold"),"seed":m.get("seed"),"status":m.get("status"),"gate":m.get("validation_selected",{}).get("gate"),"gate_selection_status":m.get("gate_selection_status")})
        test.extend(m.get("test_metrics",[])); gates.append(m.get("validation_selected",{}))
    return {"namespace":namespace,"runs":len(records),"run_records":records,"test_pooled":pooled(test),"selected_gates":gates}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payload={"release_id":"cbramod-ttt-method-comparison-v1","joint_ttt":ttt_summary("cbramod-joint-ttt-v1-formal"),"meta_ttt":ttt_summary("cbramod-meta-ttt-v1-formal"),"label_prior":prior_summary("cbramod-label-prior-tta-v1-formal")}
    (OUT/"summary.json").write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=True)+"\n")
    rows=[]
    for key in ("joint_ttt","meta_ttt"):
        x=payload[key]
        for condition in ("frozen_pooled","adapted_pooled"):
            rows.append({"method":key,"condition":condition,**x[condition]})
    rows.append({"method":"label_prior","condition":"gated_postprocess",**payload["label_prior"]["test_pooled"]})
    pd.DataFrame(rows).to_csv(OUT/"pooled_metrics.csv",index=False)
    (OUT/"manifest.json").write_text(json.dumps({"release_id":payload["release_id"],"status":"complete","source_namespaces":[payload[k]["namespace"] for k in ("joint_ttt","meta_ttt","label_prior")],"test_selection":"all thresholds/gates are recorded as validation-only in per-run manifests","outputs":["summary.json","pooled_metrics.csv"]},indent=2,sort_keys=True)+"\n")
    print(json.dumps(payload,indent=2,sort_keys=True,allow_nan=True))
if __name__=="__main__": main()

