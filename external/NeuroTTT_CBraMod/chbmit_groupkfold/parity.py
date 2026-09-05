from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import DEFAULT_CACHE, DEFAULT_FOLDS, DEFAULT_WINDOWS, WindowDataset, load_rows
from .evaluate import FastScalarEpisodicTTT, module_hash, scalar_ttt_batch, set_seed, vmap_ttt_batch
from .model import CHBJointModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=("band", "mask"), required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--fold-root", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--pretrained", type=Path, default=Path(__file__).resolve().parents[1] / "pretrained_weights/pretrained_weights.pth")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = "band_joint" if args.objective == "band" else "mask_joint"
    checkpoint = args.source_root / "runs" / f"{source}_fold{args.fold}_seed{args.seed}" / "best.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    rows = load_rows(args.fold, "validation", args.windows, args.fold_root)
    rng = np.random.default_rng(args.seed + 99)
    parts = []
    for label in (0.0, 1.0):
        group = rows[rows.label == label]
        parts.append(group.iloc[np.sort(rng.choice(len(group), size=max(1, args.samples // 2), replace=False))])
    rows = pd.concat(parts, ignore_index=True).iloc[: args.samples]
    dataset = WindowDataset(rows, args.cache_root)
    batch = [dataset[index] for index in range(len(dataset))]
    signal = torch.stack([row[0] for row in batch]).cuda()
    sample_ids = [row[2] for row in batch]

    scalar_model = CHBJointModel(args.pretrained).cuda()
    scalar_model.load_state_dict(state["model"], strict=True)
    vector_model = CHBJointModel(args.pretrained).cuda()
    vector_model.load_state_dict(state["model"], strict=True)
    fast_model = CHBJointModel(args.pretrained).cuda()
    fast_model.load_state_dict(state["model"], strict=True)
    scalar_before = module_hash(scalar_model)
    vector_before = module_hash(vector_model)
    fast_before = module_hash(fast_model)
    set_seed(args.seed)
    scalar = scalar_ttt_batch(scalar_model, signal, sample_ids, args.objective, 1e-4)
    set_seed(args.seed)
    vector_error = None
    try:
        vector = vmap_ttt_batch(vector_model, signal, sample_ids, args.objective, 1e-4)
    except Exception as error:  # audit the unsupported optimized path without hiding it
        vector = None
        vector_error = f"{type(error).__name__}: {error}"
    set_seed(args.seed)
    adapter = FastScalarEpisodicTTT(fast_model, args.objective, 1e-4)
    fast = adapter.batch(signal, sample_ids)
    adapter.finalize()
    result = {
        "objective": args.objective,
        "samples": len(signal),
        "scalar": scalar.tolist(),
        "vmap": vector.tolist() if vector is not None else None,
        "vmap_error": vector_error,
        "fast_scalar": fast.tolist(),
        "maximum_absolute_probability_difference": float(np.max(np.abs(scalar - vector))) if vector is not None else None,
        "fast_scalar_maximum_absolute_probability_difference": float(np.max(np.abs(scalar - fast))),
        "scalar_restored": module_hash(scalar_model) == scalar_before,
        "vmap_unmutated": module_hash(vector_model) == vector_before,
        "fast_scalar_restored": module_hash(fast_model) == fast_before,
        "strict_probability_parity_1e4": bool(np.max(np.abs(scalar - vector)) <= 1e-4) if vector is not None else False,
        "fast_scalar_strict_probability_parity_1e4": bool(np.max(np.abs(scalar - fast)) <= 1e-4),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
