"""Real-model exact multi-step and learned-step-size gradient smoke checks."""
from pathlib import Path
import json
import torch
from torch.nn import functional as F
from chbmit_groupkfold.meta_model import CHBMetaTTTModel
from chbmit_groupkfold.data import load_rows, WindowDataset
from chbmit_groupkfold.transforms import deterministic_band_view
from chbmit_groupkfold.band_v2 import BandStepSizes, adapt_band, predict_query, eligible_context


def main():
    torch.set_num_threads(4)
    torch.manual_seed(3407)
    model = CHBMetaTTTModel(Path(__file__).parent/'pretrained_weights/pretrained_weights.pth').cuda().eval()
    rows = load_rows(0, 'train')
    ds = WindowDataset(rows.iloc[:32])
    x, _, sid = ds[0]
    q, y, _ = ds[20]
    x, q, y = x[None].cuda(), q[None].cuda(), y[None].cuda()
    transformed, labels = deterministic_band_view(x, [sid])
    original = {n: p.detach().clone() for n, p in model.adaptive_named_parameters('band').items()}
    results = []
    for layered in (False, True):
        for steps in (1, 3, 5):
            model.zero_grad(set_to_none=True)
            rates = BandStepSizes(original.keys(), layered=layered).cuda()
            updated, losses = adapt_band(model, transformed, labels, rates, steps, differentiable=True)
            outer = F.binary_cross_entropy_with_logits(predict_query(model, updated, q), y)
            outer.backward()
            gradients = {n: None if p.grad is None else float(p.grad) for n, p in rates.raw.items()}
            for name, value in gradients.items():
                if layered and steps == 1 and name == 'band_head':
                    assert value is None or value == 0.0
                else:
                    assert value is not None and torch.isfinite(torch.tensor(value)), (steps, name, value)
            assert all(torch.equal(p.detach(), original[n]) for n, p in model.adaptive_named_parameters('band').items())
            result = {'steps': steps, 'layered': layered, 'outer_loss': float(outer.detach()),
                      'step_size_gradients': gradients, 'source_unmutated': True}
            results.append(result)
            print(json.dumps(result), flush=True)
            del updated, outer
    support = eligible_context(rows, rows.iloc[20])
    assert (support.end <= rows.iloc[20].start).all()
    out = Path('/root/b_false_alarm_atlas/outputs/reports/band-ttt-v2/smoke.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'status': 'passed', 'checks': results, 'test_partition_read': False,
                               'full_training_complete': False}, indent=2)+'\n')


if __name__ == '__main__':
    main()
