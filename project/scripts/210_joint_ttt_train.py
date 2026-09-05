"""Joint source training for the CBraMod test-time-training experiment.

This is a separate experiment from the existing same-patient TTT/oracle
runs. The source encoder is trained with the seizure task and CBraMod's
masked-patch reconstruction objective simultaneously. At target time only
the reconstruction objective may update the encoder; target labels are never
read by this script.

The command runs one immutable fold/seed unit. A queue can later launch the
same command for the 15 outer-fold/seed units.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from bfa.models.cbramod_adapter import CBraModAdapter
from bfa.models.shared_head import SharedContextHead

ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
OUT = ROOT / "outputs/reports" / os.environ.get("JOINT_TTT_NAMESPACE", "cbramod-joint-ttt-v1")
WINDOWS = ROOT / "manifests/windows.parquet"
CV_MANIFEST = ROOT / "manifests/groupkfold_cv_v1/cv_manifest.json"
FOLD_DIR = ROOT / "manifests/groupkfold_cv_v1"
PRETRAINED = ROOT / "third_party/CBraMod/pretrained_weights/pretrained_weights.pth"
SOURCE_ROOT = ROOT / "runs/v3-groupkfold-confirmatory-v1/cbramod"
RATE = 200
WINDOW_S = 10
STRIDE_S = 2
CONTEXT_WINDOWS = 31
CONTEXT_S = 70
DEFAULT_BATCH = 8
DEFAULT_MICRO = max(1, int(os.environ.get("JOINT_TTT_MICRO_CONTEXTS", "2")))
DEFAULT_MASK_PATCHES = 5
SEEDS = (17, 42, 3407)

def now() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n")
    os.replace(temp, path)

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

def device_from_arg(value: str | None) -> torch.device:
    device = torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device

def source_checkpoint(fold: int, seed: int) -> Path:
    path = SOURCE_ROOT / f"split{fold}_seed{seed}_main/checkpoints/step_05000.pt"
    if path.exists(): return path
    fallback = SOURCE_ROOT / "split0_seed17_main/checkpoints/step_05000.pt"
    if fold == 0 and seed == 17 and fallback.exists(): return fallback
    raise FileNotFoundError(path)

def fold_assignments(fold: int) -> dict[str, Any]:
    path = FOLD_DIR / f"fold_{fold}.json"
    payload = json.loads(path.read_text())
    for key in ("train", "validation", "test"):
        if not isinstance(payload.get(key), list) or not payload[key]:
            raise ValueError(f"fold manifest lacks non-empty {key}")
    return payload

def load_source_rows(fold: int, split: str) -> pd.DataFrame:
    if split not in {"train", "validation"}:
        raise ValueError("joint TTT source loader cannot read outer test rows")
    patients = set(str(x) for x in fold_assignments(fold)[split])
    table = pd.read_parquet(WINDOWS)
    table = table[table.patient.astype(str).isin(patients)].copy()
    table = table[(table.label.isin([0.0, 1.0])) & (~table.warmup.astype(bool))].copy()
    table = table.sort_values(["patient", "recording", "start"], kind="stable").reset_index(drop=True)
    if table.empty or set(table.label.astype(int)) != {0, 1}:
        raise RuntimeError(f"{split} rows do not contain both labels for fold {fold}")
    return table

class SignalCache:
    def __init__(self, max_items: int = 128) -> None:
        self.max_items = max_items; self._items: OrderedDict[str, np.ndarray] = OrderedDict()
    def get(self, relative_path: str) -> np.ndarray:
        if relative_path in self._items:
            value = self._items.pop(relative_path); self._items[relative_path] = value; return value
        path = Path(os.environ.get("BFA_CACHE_ROOT", "/mnt/d/EEGData/bfa_cache_v3_official_noclip/cbramod")) / Path(relative_path).with_suffix(".npy")
        if not path.is_file(): raise FileNotFoundError(path)
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.ndim != 2 or value.shape[0] != 16: raise ValueError(f"unexpected cached EEG shape {value.shape}: {path}")
        self._items[relative_path] = value
        while len(self._items) > self.max_items: self._items.popitem(last=False)
        return value
    def context(self, row: Any) -> np.ndarray:
        view = self.get(str(row.relative_path)); end_s = float(row.end)
        start = int(round((end_s - CONTEXT_S) * RATE)); length = int(CONTEXT_S * RATE)
        if start < 0 or start + length > view.shape[1]: raise ValueError(f"context outside cache for {row.recording} at {end_s}")
        context = view[:, start:start + length]
        windows = np.stack([context[:, i * STRIDE_S * RATE:i * STRIDE_S * RATE + WINDOW_S * RATE] for i in range(CONTEXT_WINDOWS)], axis=0)
        if windows.shape != (CONTEXT_WINDOWS, 16, WINDOW_S * RATE): raise ValueError(f"bad context shape {windows.shape}")
        return windows.astype(np.float32, copy=False)

def sample_rows(table: pd.DataFrame, rng: np.random.Generator, batch_size: int) -> pd.DataFrame:
    if batch_size < 2 or batch_size % 2: raise ValueError("batch_size must be even and >=2")
    positives, negatives = table[table.label == 1.0], table[table.label == 0.0]
    selected = pd.concat([positives.iloc[rng.integers(0, len(positives), size=batch_size // 2)], negatives.iloc[rng.integers(0, len(negatives), size=batch_size // 2)]], ignore_index=True)
    return selected.iloc[rng.permutation(len(selected))].reset_index(drop=True)

def build_models(checkpoint: Path, device: torch.device) -> tuple[CBraModAdapter, SharedContextHead]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(state.get("update", -1)) != 5000: raise ValueError(f"source checkpoint is not step 5000: {checkpoint}")
    adapter = CBraModAdapter(PRETRAINED, train_backbone=True)
    adapter.projection.load_state_dict({"weight": state["encoder"]["projection.weight"], "bias": state["encoder"]["projection.bias"]})
    head = SharedContextHead(); head.load_state_dict(state["head"], strict=True)
    adapter.to(device); head.to(device); adapter.train(); head.train(); return adapter, head

def detection_loss(adapter: CBraModAdapter, head: SharedContextHead, contexts: np.ndarray, labels: np.ndarray, device: torch.device, micro_contexts: int) -> tuple[torch.Tensor, float]:
    total: torch.Tensor | None = None; losses: list[tuple[float, int]] = []
    for begin in range(0, len(contexts), micro_contexts):
        batch = torch.from_numpy(np.ascontiguousarray(contexts[begin:begin + micro_contexts])).to(device=device, dtype=torch.float32)
        features = adapter.backbone_features(batch.reshape(-1, 16, 10, RATE))
        logits = head(adapter.projection(features).reshape(batch.shape[0], CONTEXT_WINDOWS, 16, 128))
        target = torch.from_numpy(labels[begin:begin + len(batch)]).to(device=device, dtype=torch.float32)
        value = F.binary_cross_entropy_with_logits(logits, target)
        if not torch.isfinite(value): raise RuntimeError("non-finite detection loss")
        total = value * (len(batch) / len(contexts)) if total is None else total + value * (len(batch) / len(contexts))
        losses.append((float(value.detach().cpu()), len(batch)))
    if total is None: raise RuntimeError("empty detection batch")
    return total, sum(value * count for value, count in losses) / len(contexts)

def reconstruction_loss(adapter: CBraModAdapter, contexts: np.ndarray, device: torch.device, rng: np.random.Generator, mask_patches: int) -> tuple[torch.Tensor, float]:
    chosen = rng.integers(0, CONTEXT_WINDOWS, size=len(contexts)); raw = np.stack([contexts[i, chosen[i]] for i in range(len(contexts))], axis=0)
    batch = torch.from_numpy(np.ascontiguousarray(raw)).to(device=device, dtype=torch.float32)
    mask = torch.zeros((len(batch), 16, 10), device=device, dtype=torch.bool)
    for index in range(len(batch)): mask[index, :, rng.choice(10, size=min(mask_patches, 10), replace=False)] = True
    output = adapter.backbone(batch.reshape(-1, 16, 10, RATE), mask=mask); expanded = mask.unsqueeze(-1).expand_as(output)
    value = F.mse_loss(output[expanded], batch.reshape(-1, 16, 10, RATE)[expanded])
    if not torch.isfinite(value): raise RuntimeError("non-finite reconstruction loss")
    return value, float(value.detach().cpu())

def validation_loss(adapter: CBraModAdapter, head: SharedContextHead, rows: pd.DataFrame, cache: SignalCache, device: torch.device, micro_contexts: int, max_rows: int, seed: int) -> float:
    rng = np.random.default_rng(seed); selected = []
    for label in (1.0, 0.0):
        group = rows[rows.label == label]
        if not group.empty:
            selected.append(group.iloc[np.sort(rng.choice(len(group), size=min(max_rows, len(group)), replace=False))])
    if len(selected) < 2: return float("nan")
    sample = pd.concat(selected, ignore_index=True); contexts = np.stack([cache.context(row) for row in sample.itertuples(index=False)], axis=0)
    adapter.eval(); head.eval()
    with torch.inference_mode(): _, value = detection_loss(adapter, head, contexts, sample.label.to_numpy(dtype=np.float32), device, micro_contexts)
    adapter.train(); head.train(); return value

def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.fold not in range(5) or args.seed not in SEEDS: raise ValueError("invalid fold or seed")
    if args.updates < 1: raise ValueError(args.updates)
    set_seed(args.seed); device = device_from_arg(args.device); out = OUT / "runs" / f"fold{args.fold}_seed{args.seed}"
    if out.exists(): raise RuntimeError(f"refusing to overwrite existing run: {out}")
    out.mkdir(parents=True); fold = fold_assignments(args.fold); fit = load_source_rows(args.fold, "train"); validation = load_source_rows(args.fold, "validation"); checkpoint = source_checkpoint(args.fold, args.seed)
    adapter, head = build_models(checkpoint, device)
    optimizer = torch.optim.AdamW([{ "params": list(adapter.backbone.parameters()), "lr": args.backbone_lr }, { "params": list(adapter.projection.parameters()) + list(head.parameters()), "lr": args.head_lr }], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.updates); cache = SignalCache(args.max_cached_recordings); rng = np.random.default_rng(args.seed + 210000); history = []; started = time.monotonic(); best_val = float("inf"); best_update = 0; best_state = None
    for update in range(1, args.updates + 1):
        rows = sample_rows(fit, rng, args.batch_size); contexts = np.stack([cache.context(row) for row in rows.itertuples(index=False)], axis=0); labels = rows.label.to_numpy(dtype=np.float32); optimizer.zero_grad(set_to_none=True)
        det_tensor, det_value = detection_loss(adapter, head, contexts, labels, device, args.micro_contexts); rec_tensor, rec_value = reconstruction_loss(adapter, contexts, device, rng, args.mask_patches); total = det_tensor + args.reconstruction_weight * rec_tensor
        if not torch.isfinite(total): raise RuntimeError(f"non-finite total loss at update {update}")
        total.backward(); trainable = [p for p in list(adapter.parameters()) + list(head.parameters()) if p.requires_grad]; grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip).detach().cpu())
        if not math.isfinite(grad_norm): raise RuntimeError(f"non-finite gradient at update {update}")
        optimizer.step(); scheduler.step(); record = {"update": update, "detection_loss": det_value, "reconstruction_loss": rec_value, "total_loss": float(total.detach().cpu()), "grad_norm_pre_clip": grad_norm, "positive_n": int(labels.sum()), "negative_n": int(len(labels) - labels.sum()), "elapsed_s": time.monotonic() - started}
        if not args.smoke and (update % args.validation_every == 0 or update == args.updates):
            record["validation_detection_loss"] = validation_loss(adapter, head, validation, cache, device, args.micro_contexts, args.validation_max_rows, args.seed + update); value = float(record["validation_detection_loss"])
            if math.isfinite(value) and value < best_val:
                best_val, best_update = value, update; best_state = {"adapter": {n: t.detach().cpu().clone() for n, t in adapter.state_dict().items()}, "head": {n: t.detach().cpu().clone() for n, t in head.state_dict().items()}}
        history.append(record)
        if update == 1 or update % max(1, min(100, args.updates)) == 0 or update == args.updates: atomic_json(out / "progress.json", {"status": "running", "fold": args.fold, "seed": args.seed, "update": update, "total_updates": args.updates, "best_validation_loss": best_val if math.isfinite(best_val) else None, "best_update": best_update, "updated_utc": now()})
    if best_state is not None: adapter.load_state_dict(best_state["adapter"], strict=True); head.load_state_dict(best_state["head"], strict=True)
    torch.save({"encoder": adapter.state_dict(), "head": head.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "update": args.updates, "selected_by_validation": best_state is not None, "best_validation_update": best_update, "source_checkpoint": str(checkpoint.relative_to(ROOT))}, out / "checkpoint.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2, allow_nan=True) + "\n")
    manifest = {"release_id": "cbramod-joint-ttt-v1", "status": "smoke_complete" if args.smoke else "training_complete", "dataset": "CHB-MIT", "fold": args.fold, "seed": args.seed, "outer_test_read": False, "outer_test_used_for_selection": False, "train_patients": fold["train"], "validation_patients": fold["validation"], "test_patients_recorded_only": fold["test"], "source_checkpoint": str(checkpoint.relative_to(ROOT)), "source_checkpoint_sha256": sha256(checkpoint), "pretrained_cbramod_sha256": sha256(PRETRAINED), "windows_manifest": str(WINDOWS.relative_to(ROOT)), "windows_manifest_sha256": sha256(WINDOWS), "cv_manifest": str(CV_MANIFEST.relative_to(ROOT)), "cv_manifest_sha256": sha256(CV_MANIFEST), "objective": "joint seizure BCE plus masked CBraMod patch reconstruction", "reconstruction_weight": args.reconstruction_weight, "mask_patches": args.mask_patches, "batch_size": args.batch_size, "micro_contexts": args.micro_contexts, "backbone_lr": args.backbone_lr, "head_lr": args.head_lr, "weight_decay": args.weight_decay, "requested_updates": args.updates, "selected_by_validation": best_state is not None, "best_validation_loss": best_val if math.isfinite(best_val) else None, "best_validation_update": best_update if best_state is not None else None, "train_rows_available": int(len(fit)), "validation_rows_available": int(len(validation)), "test_rows_loaded": False, "test_labels_loaded": False, "threshold_source": "not applicable during source training; selected later on validation only", "checkpoint": str((out / "checkpoint.pt").relative_to(ROOT)), "checkpoint_sha256": sha256(out / "checkpoint.pt"), "history": str((out / "history.json").relative_to(ROOT)), "created_utc": now()}
    atomic_json(out / "manifest.json", manifest); atomic_json(OUT / "latest_progress.json", {"status": manifest["status"], "fold": args.fold, "seed": args.seed, "update": args.updates, "updated_utc": now()}); return manifest

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("smoke", "train")); parser.add_argument("--fold", type=int, required=True); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--updates", type=int, default=25); parser.add_argument("--device"); parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH); parser.add_argument("--micro-contexts", type=int, default=DEFAULT_MICRO); parser.add_argument("--mask-patches", type=int, default=DEFAULT_MASK_PATCHES); parser.add_argument("--reconstruction-weight", type=float, default=0.05); parser.add_argument("--backbone-lr", type=float, default=1e-5); parser.add_argument("--head-lr", type=float, default=1e-4); parser.add_argument("--weight-decay", type=float, default=1e-5); parser.add_argument("--grad-clip", type=float, default=1.0); parser.add_argument("--validation-every", type=int, default=100); parser.add_argument("--validation-max-rows", type=int, default=64); parser.add_argument("--max-cached-recordings", type=int, default=128); args = parser.parse_args(); args.smoke = args.mode == "smoke"
    if args.smoke and args.updates > 25: raise ValueError("smoke is capped at 25 updates")
    print(json.dumps(run(args), indent=2, sort_keys=True, allow_nan=True))

if __name__ == "__main__": main()
