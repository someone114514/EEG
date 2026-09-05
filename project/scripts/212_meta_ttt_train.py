"""First-order MT3-style meta-training for unlabeled CBraMod TTT.

Each episode uses unlabeled support EEG for one reconstruction update and a
labelled query episode from a different recording of the same source patient
to train the initialization.  The query is never an outer-test row; target
adaptation is implemented by a separate evaluation stage.  Only the last two
CBraMod transformer blocks, projection, and detection head are meta-trained
to limit drift and memory use.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
OUT = ROOT / "outputs/reports" / os.environ.get("META_TTT_NAMESPACE", "cbramod-meta-ttt-v1")
SEEDS = (17, 42, 3407)


def load_joint_module():
    path = ROOT / "scripts/210_joint_ttt_train.py"
    spec = importlib.util.spec_from_file_location("joint_ttt", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n")
    os.replace(temporary, path)


def now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def configure_trainable(adapter: torch.nn.Module, head: torch.nn.Module) -> list[torch.nn.Parameter]:
    for parameter in adapter.parameters(): parameter.requires_grad_(False)
    for name, parameter in adapter.backbone.named_parameters():
        parameter.requires_grad_(name.startswith("encoder.layers.10.") or name.startswith("encoder.layers.11."))
    for parameter in adapter.projection.parameters(): parameter.requires_grad_(True)
    for parameter in head.parameters(): parameter.requires_grad_(True)
    return [p for p in list(adapter.parameters()) + list(head.parameters()) if p.requires_grad]


def sample_task_rows(table, rng, patient, batch_size):
    unit = table[table.patient.astype(str) == str(patient)]
    if len(unit.recording.unique()) < 2:
        return None, None
    recordings = np.asarray(sorted(unit.recording.astype(str).unique()))
    rng.shuffle(recordings)
    midpoint = max(1, len(recordings) // 2)
    support_recs = set(recordings[:midpoint]); query_recs = set(recordings[midpoint:])
    if not query_recs:
        query_recs = {recordings[-1]}; support_recs = set(recordings[:-1])
    support = unit[unit.recording.astype(str).isin(support_recs)]
    query = unit[unit.recording.astype(str).isin(query_recs)]
    return support, query


def sample_rows_for_meta(table, rng, batch_size, balanced: bool = True):
    """Sample rows robustly when a patient's recording has one label only.

    The meta query is a detection objective, so an all-background or
    all-seizure recording is valid support.  The source module's balanced
    sampler intentionally rejects that case; using it here would silently
    discard exactly the patient/recording episodes needed for a
    patient-adaptation objective.
    """
    if table is None or len(table) == 0:
        return None
    if balanced and set(table.label.astype(float).unique()) == {0.0, 1.0}:
        return load_joint_module().sample_rows(table, rng, batch_size)
    indices = rng.integers(0, len(table), size=batch_size)
    return table.iloc[indices].reset_index(drop=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.fold not in range(5) or args.seed not in SEEDS:
        raise ValueError("invalid fold or seed")
    if args.updates < 1:
        raise ValueError(args.updates)
    m = load_joint_module(); m.set_seed(args.seed)
    device = m.device_from_arg(args.device)
    out = OUT / "runs" / f"fold{args.fold}_seed{args.seed}"
    if out.exists(): raise RuntimeError(f"refusing to overwrite existing run: {out}")
    out.mkdir(parents=True)
    fold = m.fold_assignments(args.fold)
    fit = m.load_source_rows(args.fold, "train")
    checkpoint = m.source_checkpoint(args.fold, args.seed)
    adapter, head = m.build_models(checkpoint, device)
    trainable = configure_trainable(adapter, head)
    optimizer = torch.optim.AdamW(trainable, lr=args.outer_lr, weight_decay=args.weight_decay)
    cache = m.SignalCache(args.max_cached_recordings)
    rng = np.random.default_rng(args.seed + 212000)
    patients = np.asarray(sorted(fit.patient.astype(str).unique()))
    history = []
    started = time.monotonic()
    for update in range(1, args.updates + 1):
        chosen_patient = str(patients[int(rng.integers(len(patients)))])
        support_rows, query_rows = sample_task_rows(fit, rng, chosen_patient, args.batch_size)
        if support_rows is None or query_rows is None:
            continue
        # Support only needs valid EEG for the unlabeled reconstruction step;
        # query may legitimately be single-class for a recording.  Do not
        # drop such episodes or force synthetic positives/negatives.
        support = sample_rows_for_meta(support_rows, rng, args.batch_size, balanced=False)
        query = sample_rows_for_meta(query_rows, rng, args.batch_size, balanced=True)
        if support is None or query is None:
            continue
        support_x = np.stack([cache.context(row) for row in support.itertuples(index=False)], axis=0)
        query_x = np.stack([cache.context(row) for row in query.itertuples(index=False)], axis=0)
        support_y = support.label.to_numpy(dtype=np.float32); query_y = query.label.to_numpy(dtype=np.float32)
        optimizer.zero_grad(set_to_none=True)
        support_rec, support_rec_value = m.reconstruction_loss(adapter, support_x, device, rng, args.mask_patches)
        support_grads = torch.autograd.grad(support_rec, trainable, allow_unused=True)
        backup = [p.detach().clone() for p in trainable]
        with torch.no_grad():
            for parameter, gradient in zip(trainable, support_grads):
                if gradient is not None: parameter.add_(-args.inner_lr * gradient)
        query_before_tensor, query_before = m.detection_loss(adapter, head, query_x, query_y, device, args.micro_contexts)
        # The temporary support update is differentiated only to first order;
        # query gradients are assigned to the pre-update initialization after
        # restoring it. This is the memory-bounded FOMAML approximation.
        query_grads = torch.autograd.grad(query_before_tensor, trainable, allow_unused=True)
        with torch.no_grad():
            for parameter, original in zip(trainable, backup): parameter.copy_(original)
        for parameter, gradient in zip(trainable, query_grads):
            parameter.grad = None if gradient is None else gradient.detach().clone()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip).detach().cpu())
        if not math.isfinite(grad_norm): raise RuntimeError(f"non-finite meta gradient at update {update}")
        optimizer.step()
        record = {"update": update, "patient": chosen_patient, "support_reconstruction_loss": support_rec_value, "query_detection_loss_after_update": float(query_before), "query_has_positive": bool((query_y > 0.5).any()), "query_has_negative": bool((query_y <= 0.5).any()), "query_grad_norm_pre_clip": grad_norm, "elapsed_s": time.monotonic() - started}
        history.append(record)
        if update == 1 or update % max(1, min(100, args.updates)) == 0 or update == args.updates:
            atomic_json(out / "progress.json", {"status": "running", "fold": args.fold, "seed": args.seed, "update": update, "total_updates": args.updates, "updated_utc": now()})
    torch.save({"encoder": adapter.state_dict(), "head": head.state_dict(), "optimizer": optimizer.state_dict(), "update": args.updates, "meta_algorithm": "first_order_MT3", "source_checkpoint": str(checkpoint.relative_to(ROOT))}, out / "checkpoint.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2, allow_nan=True) + "\n")
    manifest = {"release_id": "cbramod-meta-ttt-v1", "status": "smoke_complete" if args.smoke else "training_complete", "dataset": "CHB-MIT", "fold": args.fold, "seed": args.seed, "meta_algorithm": "first_order_MT3", "support_objective": "masked CBraMod patch reconstruction", "query_objective": "seizure BCE after one support update", "outer_test_read": False, "outer_test_used_for_selection": False, "train_patients": fold["train"], "validation_patients": fold["validation"], "test_patients_recorded_only": fold["test"], "source_checkpoint": str(checkpoint.relative_to(ROOT)), "source_checkpoint_sha256": m.sha256(checkpoint), "pretrained_cbramod_sha256": m.sha256(m.PRETRAINED), "cv_manifest": str(m.CV_MANIFEST.relative_to(ROOT)), "cv_manifest_sha256": m.sha256(m.CV_MANIFEST), "requested_updates": args.updates, "inner_lr": args.inner_lr, "outer_lr": args.outer_lr, "batch_size": args.batch_size, "micro_contexts": args.micro_contexts, "trainable_parameter_count": int(sum(p.numel() for p in trainable)), "test_rows_loaded": False, "test_labels_loaded": False, "checkpoint": str((out / "checkpoint.pt").relative_to(ROOT)), "checkpoint_sha256": m.sha256(out / "checkpoint.pt"), "history": str((out / "history.json").relative_to(ROOT)), "created_utc": now()}
    atomic_json(out / "manifest.json", manifest); atomic_json(OUT / "latest_progress.json", {"status": manifest["status"], "fold": args.fold, "seed": args.seed, "update": args.updates, "updated_utc": now()}); return manifest


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("smoke", "train")); parser.add_argument("--fold", type=int, required=True); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--updates", type=int, default=25); parser.add_argument("--device"); parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--micro-contexts", type=int, default=1); parser.add_argument("--mask-patches", type=int, default=5); parser.add_argument("--inner-lr", type=float, default=1e-5); parser.add_argument("--outer-lr", type=float, default=1e-4); parser.add_argument("--weight-decay", type=float, default=1e-5); parser.add_argument("--grad-clip", type=float, default=1.0); parser.add_argument("--max-cached-recordings", type=int, default=128); args = parser.parse_args(); args.smoke = args.mode == "smoke"
    if args.smoke and args.updates > 25: raise ValueError("smoke is capped at 25 updates")
    print(json.dumps(run(args), indent=2, sort_keys=True, allow_nan=True))

if __name__ == "__main__": main()
