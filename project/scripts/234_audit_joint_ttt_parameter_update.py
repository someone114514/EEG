"""Audit that one Joint-TTT update changes the frozen detector parameters."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/root/b_false_alarm_atlas")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluator = load(ROOT / "scripts/214_evaluate_joint_ttt.py", "joint_evaluator")
module = evaluator.load_adaptation_module()
windows, recordings, _ = module.load_tables()
device = torch.device("cuda")
_, _, adapter, head = evaluator.load_ttt_checkpoint(module, 0, 17)
adapter.to(device); head.to(device)
patient = evaluator.fold_patients(0, "validation")[0]
recording_rows = recordings[recordings.patient_id.astype(str) == patient].sort_values("recording_id")
for recording in recording_rows.recording_id.astype(str):
    anchors = module.anchor_rows(windows, patient, recording)
    if len(anchors):
        break
else:
    raise RuntimeError("no validation anchors")
relative = str(recording_rows.loc[recording_rows.recording_id.astype(str) == recording, "relative_path"].iloc[0])
view = np.load(module.signal_path(relative), mmap_mode="r", allow_pickle=False)
times = anchors.start.to_numpy(float)[:12]
contexts = [module.context_from_view(view, float(anchor) - module.CONTEXT_HISTORY_S) for anchor in times]
before_probability = module.score_contexts(adapter, head, contexts, device)
before = {name: parameter.detach().cpu().clone() for name, parameter in adapter.backbone.named_parameters()}
before_projection = {name: parameter.detach().cpu().clone() for name, parameter in adapter.projection.named_parameters()}
before_head = {name: parameter.detach().cpu().clone() for name, parameter in head.named_parameters()}
adapter.backbone.requires_grad_(True)
adapter.projection.requires_grad_(False)
head.requires_grad_(False)
optimizer = torch.optim.AdamW(
    [parameter for parameter in adapter.backbone.parameters() if parameter.requires_grad],
    lr=1e-5,
    weight_decay=1e-5,
)
raw = np.ascontiguousarray(np.stack([context[-1] for context in contexts], axis=0))
loss, grad_norm = module.ttt_update(
    adapter, optimizer, raw, device, np.random.default_rng(20260831), micro_windows=len(raw)
)
after = {name: parameter.detach().cpu().clone() for name, parameter in adapter.backbone.named_parameters()}
after_projection = {name: parameter.detach().cpu().clone() for name, parameter in adapter.projection.named_parameters()}
after_head = {name: parameter.detach().cpu().clone() for name, parameter in head.named_parameters()}
changed = [
    (name, float(np.max(np.abs(after[name].numpy() - before[name].numpy()))),
     float(np.linalg.norm(after[name].numpy() - before[name].numpy())))
    for name in before
    if not torch.equal(before[name], after[name])
]
adapter.eval(); head.eval()
after_probability = module.score_contexts(adapter, head, contexts, device)
print({
    "patient": patient,
    "recording": recording,
    "windows": len(raw),
    "trainable_parameter_count": sum(parameter.numel() for parameter in adapter.backbone.parameters() if parameter.requires_grad),
    "loss": loss,
    "grad_norm_pre_clip": grad_norm,
    "changed_tensor_count": len(changed),
    "total_tensor_count": len(before),
    "max_parameter_abs_change": max((item[1] for item in changed), default=0.0),
    "sum_parameter_l2_change": sum(item[2] for item in changed),
    "max_probability_abs_change": float(np.max(np.abs(before_probability - after_probability))),
    "mean_probability_abs_change": float(np.mean(np.abs(before_probability - after_probability))),
    "projection_changed_tensor_count": sum(not torch.equal(before_projection[name], after_projection[name]) for name in before_projection),
    "head_changed_tensor_count": sum(not torch.equal(before_head[name], after_head[name]) for name in before_head),
})
report = {
    "patient": patient,
    "recording": recording,
    "windows": len(raw),
    "trainable_parameter_count": sum(parameter.numel() for parameter in adapter.backbone.parameters() if parameter.requires_grad),
    "loss": loss,
    "grad_norm_pre_clip": grad_norm,
    "changed_tensor_count": len(changed),
    "total_tensor_count": len(before),
    "max_parameter_abs_change": max((item[1] for item in changed), default=0.0),
    "sum_parameter_l2_change": sum(item[2] for item in changed),
    "max_probability_abs_change": float(np.max(np.abs(before_probability - after_probability))),
    "mean_probability_abs_change": float(np.mean(np.abs(before_probability - after_probability))),
    "threshold": 0.001,
    "direct_threshold_crossings": int(np.count_nonzero((before_probability < 0.001) != (after_probability < 0.001))),
    "before_probability_min": float(np.min(before_probability)),
    "before_probability_max": float(np.max(before_probability)),
    "after_probability_min": float(np.min(after_probability)),
    "after_probability_max": float(np.max(after_probability)),
}
audit_dir = ROOT / "outputs/reports/cbramod-joint-ttt-v1-formal/audit_detailed"
audit_dir.mkdir(parents=True, exist_ok=True)
(audit_dir / "single_update_parameter_probe.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
if not changed or not np.isfinite(loss) or not np.isfinite(grad_norm):
    raise SystemExit("TTT update did not produce a finite parameter change")
