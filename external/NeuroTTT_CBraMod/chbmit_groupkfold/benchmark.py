from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import DEFAULT_CACHE, DEFAULT_FOLDS, DEFAULT_WINDOWS, WindowDataset, load_rows
from .model import CHBJointModel
from .train import compute_losses, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("detection_only", "band_joint", "mask_joint"), default="mask_joint")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--fold-root", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--pretrained", type=Path, default=Path(__file__).resolve().parents[1] / "pretrained_weights/pretrained_weights.pth")
    args = parser.parse_args()
    set_seed(3407)
    rows = load_rows(args.fold, "train", args.windows, args.fold_root)
    rng = np.random.default_rng(3407)
    parts = []
    for label in (0.0, 1.0):
        group = rows[rows.label == label]
        parts.append(group.iloc[np.sort(rng.choice(len(group), size=args.batch_size // 2, replace=False))])
    chosen = pd.concat(parts, ignore_index=True)
    dataset = WindowDataset(chosen, args.cache_root)
    batch = [dataset[index] for index in range(len(dataset))]
    signal = torch.stack([row[0] for row in batch]).cuda(non_blocking=True)
    target = torch.stack([row[1] for row in batch]).cuda(non_blocking=True)
    model = CHBJointModel(args.pretrained).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-2)
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for step in range(args.steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            detection, auxiliary, _ = compute_losses(model, signal, target, args.condition)
            loss = detection if auxiliary is None else detection + auxiliary
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
    stable = timings[1:] or timings
    print(json.dumps({
        "condition": args.condition,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "seconds_per_update": float(np.median(stable)),
        "windows_per_second": float(args.batch_size / np.median(stable)),
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated() / 2**20),
        "peak_reserved_mib": float(torch.cuda.max_memory_reserved() / 2**20),
        "loss": float(loss.detach().cpu()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
