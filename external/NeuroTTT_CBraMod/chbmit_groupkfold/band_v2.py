"""Band-only experimental inner loop. Does not modify the frozen v1 pipeline."""
from __future__ import annotations

import math
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call
from .meta_train import second_order_sdp


class BandStepSizes(nn.Module):
    """One scalar or one scalar per adaptive block/head, shared across steps."""
    def __init__(self, names, layered=False, initial=1e-4, lower=1e-6, upper=1e-3):
        super().__init__()
        if not lower < initial < upper:
            raise ValueError('initial step size outside bounds')
        self.lower, self.upper = lower, upper
        self.name_to_group = {}
        for name in names:
            if name.startswith('backbone.encoder.layers.'):
                group = '_'.join(name.split('.')[:4])
            elif name.startswith('band_head.'):
                group = 'band_head'
            else:
                raise ValueError(f'unexpected adaptive parameter {name}')
            self.name_to_group[name] = group if layered else 'global'
        z = (initial-lower)/(upper-lower)
        self.raw = nn.ParameterDict({g: nn.Parameter(torch.tensor(math.log(z/(1-z))))
                                     for g in sorted(set(self.name_to_group.values()))})

    def for_name(self, name):
        return self.lower + (self.upper-self.lower)*torch.sigmoid(self.raw[self.name_to_group[name]])


def adapt_band(model, transformed, band_labels, rates, steps, *, current=None, differentiable=False):
    """Update on unlabeled support only. The caller owns reset/carry boundaries.

    Classification targets are deliberately absent. For independent windows,
    call separately for each sample/episode, not on a mean across patients.
    """
    if steps not in (1, 3, 5):
        raise ValueError('registered matrix uses 1/3/5 steps')
    state = dict(model.adaptive_named_parameters('band') if current is None else current)
    losses = []
    for _ in range(steps):
        with second_order_sdp(transformed.device) if differentiable else nullcontext():
            logits = functional_call(model, state, (transformed,), {'mode': 'band'}, strict=False)
            loss = F.cross_entropy(logits, band_labels)
            grads = torch.autograd.grad(loss, tuple(state.values()), create_graph=differentiable,
                                        allow_unused=True)
            state = {name: value if grad is None else value-rates.for_name(name)*grad
                     for (name, value), grad in zip(state.items(), grads)}
        losses.append(loss.detach())
        if not differentiable:
            state = {name: value.detach().requires_grad_(True) for name, value in state.items()}
    return state, losses


def eligible_context(rows, query, size=8):
    """Same-patient/recording past support with no raw-signal query overlap."""
    if size < 1:
        raise ValueError('context size must be positive')
    eligible = rows[(rows.patient == query.patient) & (rows.recording == query.recording)
                    & (rows.end <= query.start)]
    return eligible.sort_values('end', kind='stable').tail(size)


def predict_query(model, state, query_signal):
    return functional_call(model, state, (query_signal,), {'mode': 'detect'}, strict=False)


def independent_batch_logits(model, signal, transformed, labels, rates, steps):
    """Vectorized independent K-step adaptation, with one state per sample."""
    from torch.func import grad, vmap
    source = model.adaptive_named_parameters('band')
    def loss_one(state, sample, label):
        return F.cross_entropy(functional_call(model,state,(sample[None],),{'mode':'band'},strict=False),label[None])
    gradients=vmap(grad(loss_one),in_dims=(None,0,0),randomness='same')(source,transformed,labels)
    state={n:p[None]-rates.for_name(n)*gradients[n] for n,p in source.items()}
    for _ in range(1,steps):
        gradients=vmap(grad(loss_one),in_dims=(0,0,0),randomness='same')(state,transformed,labels)
        state={n:p-rates.for_name(n)*gradients[n] for n,p in state.items()}
    def predict_one(current,sample):
        return functional_call(model,current,(sample[None],),{'mode':'detect'},strict=False).squeeze(0)
    return vmap(predict_one,in_dims=(0,0),randomness='same')(state,signal)
