"""Numerically validate and benchmark the deduplicated Joint-TTT scorer."""
from __future__ import annotations

import importlib.util
import time
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
adapter.to(device).eval(); head.to(device).eval()
patient = evaluator.fold_patients(0, "validation")[0]
recording_rows = recordings[recordings.patient_id.astype(str) == patient].sort_values("recording_id")
for recording in recording_rows.recording_id.astype(str):
    anchors = module.anchor_rows(windows, patient, recording)
    if len(anchors):
        break
else:
    raise RuntimeError(f"no anchors for {patient}")
relative = str(recording_rows.loc[recording_rows.recording_id.astype(str) == recording, "relative_path"].iloc[0])
view = np.load(module.signal_path(relative), mmap_mode="r", allow_pickle=False)
times = anchors.start.to_numpy(float)
block_ids = np.floor((times - module.WARMUP_S) / module.BLOCK_S).astype(int)
block_rows = anchors.iloc[np.flatnonzero(block_ids == np.unique(block_ids)[0])].copy()
times = block_rows.start.to_numpy(float)
contexts = [module.context_from_view(view, float(anchor) - module.CONTEXT_HISTORY_S) for anchor in times]

torch.cuda.synchronize(); started = time.perf_counter()
legacy = module.score_contexts(adapter, head, contexts, device)
torch.cuda.synchronize(); legacy_seconds = time.perf_counter() - started
torch.cuda.synchronize(); started = time.perf_counter()
fast, raw = evaluator.score_block_fast(module, adapter, head, view, times, device)
torch.cuda.synchronize(); fast_seconds = time.perf_counter() - started

result = {
    "patient": patient,
    "recording": recording,
    "anchors": len(times),
    "legacy_seconds": legacy_seconds,
    "fast_seconds": fast_seconds,
    "speedup": legacy_seconds / fast_seconds,
    "max_abs_difference": float(np.max(np.abs(legacy - fast))),
    "mean_abs_difference": float(np.mean(np.abs(legacy - fast))),
    "raw_inputs_identical": bool(np.array_equal(raw, np.stack([context[-1] for context in contexts]))),
}
print(result)
if result["max_abs_difference"] > 1e-5 or not result["raw_inputs_identical"]:
    raise SystemExit("fast scorer failed equivalence tolerance")
