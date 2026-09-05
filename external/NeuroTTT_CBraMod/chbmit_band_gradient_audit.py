"""Validation-only Band-SSL/classification gradient diagnostics; no model writes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.func import functional_call

from chbmit_groupkfold.data import load_rows, WindowDataset
from chbmit_groupkfold.meta_model import CHBMetaTTTModel
from chbmit_groupkfold.meta_train import set_seed
from chbmit_groupkfold.transforms import deterministic_band_view


def cosine_stats(a, b):
    if not a:
        return {"cosine": None, "ssl_norm": 0., "classification_norm": 0., "dot": 0.}
    dot = sum((x.double() * y.double()).sum() for x, y in zip(a, b))
    aa = sum(x.double().square().sum() for x in a).sqrt()
    bb = sum(y.double().square().sum() for y in b).sqrt()
    return {"cosine": float(dot / (aa * bb)) if aa > 0 and bb > 0 else None,
            "ssl_norm": float(aa), "classification_norm": float(bb), "dot": float(dot)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, default=Path('/root/b_false_alarm_atlas/outputs/reports/meta-ttt-chbmit-5fold-v1'))
    p.add_argument('--output-root', type=Path, default=Path('/root/b_false_alarm_atlas/outputs/reports/band-ttt-v2/gradient_audit'))
    p.add_argument('--folds', type=int, nargs='+', default=list(range(5)))
    p.add_argument('--per-class', type=int, default=16)
    args = p.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    set_seed(3407)
    torch.set_num_threads(4)
    pretrained = Path(__file__).parent / 'pretrained_weights/pretrained_weights.pth'
    for fold in args.folds:
        marker = args.output_root / f'fold{fold}_completed.json'
        if marker.exists():
            print(f'fold{fold}: already complete', flush=True)
            continue
        started = time.monotonic()
        frame = load_rows(fold, 'validation')
        rng = np.random.default_rng(3407 + fold)
        chosen = []
        # Equal class counts, round-robin patient sampling; not a population estimate.
        for label in (0, 1):
            subset = frame[frame.label == label]
            groups = [g.index.to_numpy() for _, g in subset.groupby('patient', sort=True)]
            for i in range(args.per_class):
                chosen.append(int(rng.choice(groups[i % len(groups)])))
        selected = frame.loc[chosen].reset_index(drop=True)
        selected.to_json(args.output_root / f'fold{fold}_samples.json', orient='records', indent=2)
        dataset = WindowDataset(selected)
        state = torch.load(args.source_root / 'runs' / f'meta_band_fold{fold}_seed3407' / 'best.pt', map_location='cpu', weights_only=False)
        model = CHBMetaTTTModel(pretrained).cuda().eval()
        model.load_state_dict(state['model'], strict=True)
        alpha = float(state['alpha'])
        adaptive = model.adaptive_named_parameters('band')
        names, params = list(adaptive), list(adaptive.values())
        shared = [i for i, n in enumerate(names) if n.startswith('backbone.')]
        blocks = sorted(set('.'.join(names[i].split('.')[:4]) for i in shared))
        records = []
        for i in range(len(dataset)):
            signal, target, sample_id = dataset[i]
            signal, target = signal.unsqueeze(0).cuda(), target.unsqueeze(0).cuda()
            transformed, band = deterministic_band_view(signal, [sample_id])
            ssl_loss = F.cross_entropy(model(transformed, mode='band'), band)
            gs = torch.autograd.grad(ssl_loss, params, allow_unused=True)
            before = F.binary_cross_entropy_with_logits(model(signal, mode='detect'), target)
            gc = torch.autograd.grad(before, params, allow_unused=True)
            indices = [k for k in shared if gs[k] is not None and gc[k] is not None]
            updated = {n: w - alpha * g if g is not None else w for n, w, g in zip(names, params, gs)}
            with torch.no_grad():
                after = F.binary_cross_entropy_with_logits(functional_call(model, updated, (signal,), {'mode': 'detect'}, strict=False), target)
                ssl_after = F.cross_entropy(functional_call(model, updated, (transformed,), {'mode': 'band'}, strict=False), band)
            global_stats = cosine_stats([gs[k] for k in indices], [gc[k] for k in indices])
            row = {'fold': fold, 'patient': selected.iloc[i].patient, 'sample_id': sample_id,
                   'label': int(target.item()), 'band': int(band.item()), 'alpha': alpha,
                   **global_stats, 'classification_before': float(before.detach()),
                   'classification_after': float(after), 'classification_delta': float(after-before.detach()),
                   'first_order_predicted_delta': -alpha * global_stats['dot'],
                   'ssl_before': float(ssl_loss.detach()), 'ssl_after': float(ssl_after)}
            for block in blocks:
                ids = [k for k in indices if names[k].startswith(block + '.')]
                row.update({f'{block}.{key}': value for key, value in cosine_stats([gs[k] for k in ids], [gc[k] for k in ids]).items()})
            head = [g for n, g in zip(names, gs) if n.startswith('band_head.') and g is not None]
            row['ssl_head_gradient_norm'] = float(sum(g.double().square().sum() for g in head).sqrt()) if head else 0.
            records.append(row)
            if (i+1) % 8 == 0:
                print(f'fold{fold} {i+1}/{len(dataset)}', flush=True)
        result = pd.DataFrame(records)
        result.to_csv(args.output_root / f'fold{fold}_gradients.csv', index=False)
        payload = {'status': 'complete', 'fold': fold, 'samples': len(result), 'split': 'validation',
                   'test_partition_read': False, 'labels_used_for_adaptation': False,
                   'precision': 'float32 diagnostic; no AMP', 'sampling': 'class-balanced, patient-round-robin',
                   'alpha_from': 'best.pt', 'alpha': alpha, 'elapsed_s': time.monotonic()-started,
                   'cosine_mean': float(result.cosine.mean()), 'cosine_median': float(result.cosine.median()),
                   'negative_cosine_fraction': float((result.cosine < 0).mean()),
                   'classification_loss_improved_fraction': float((result.classification_delta < 0).mean()),
                   'ssl_loss_improved_fraction': float((result.ssl_after < result.ssl_before).mean())}
        marker.write_text(json.dumps(payload, indent=2)+'\n')
        print(json.dumps(payload), flush=True)
        del updated, model, state, gs, gc
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
