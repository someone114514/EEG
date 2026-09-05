from __future__ import annotations

import hashlib
import random

import torch

BANDS = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))


def balanced_band_labels(batch: int, device: torch.device, generator: torch.Generator | None = None) -> torch.Tensor:
    base = torch.arange(batch, device=device) % len(BANDS)
    return base[torch.randperm(batch, device=device, generator=generator)]


def band_reject_view(signal: torch.Tensor, labels: torch.Tensor, *, sampling_hz: int = 200, jitter: float = 0.10) -> torch.Tensor:
    # The upstream torchaudio CUDA IIR kernel fails on the RTX 5090 build used
    # for this experiment.  A vectorized FFT stop-band implements the same
    # pretext label without a CPU round trip or sample mixing.
    spectrum = torch.fft.rfft(signal.float(), dim=-1, norm="backward")
    output_spectrum = spectrum.clone()
    frequencies = torch.fft.rfftfreq(signal.shape[-1], d=1.0 / sampling_hz).to(signal.device)
    nyquist = sampling_hz / 2.0
    for label, (base_low, base_high) in enumerate(BANDS):
        selected = labels == label
        if not torch.any(selected):
            continue
        width = base_high - base_low
        low = max(0.1, base_low + random.uniform(-jitter, jitter) * width)
        high = min(nyquist * 0.98, base_high + random.uniform(-jitter, jitter) * width)
        high = max(low + 0.1, high)
        rejected = (frequencies >= low) & (frequencies <= high)
        output_spectrum[selected] = output_spectrum[selected].masked_fill(rejected, 0)
    return torch.fft.irfft(output_spectrum, n=signal.shape[-1], dim=-1, norm="backward").to(signal.dtype)


def patch_mask(batch: int, channels: int, patches: int, ratio: float, device: torch.device, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.rand((batch, channels, patches), device=device, generator=generator) < float(ratio)


def _digest(sample_id: str) -> bytes:
    return hashlib.sha256(sample_id.encode("utf-8")).digest()


def deterministic_band_view(signal: torch.Tensor, sample_ids: list[str], *, sampling_hz: int = 200, jitter: float = 0.10) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-invariant band labels and jitter for validation/test TTT."""
    if len(signal) != len(sample_ids):
        raise ValueError("sample ids do not align")
    spectrum = torch.fft.rfft(signal.float(), dim=-1, norm="backward")
    frequencies = torch.fft.rfftfreq(signal.shape[-1], d=1.0 / sampling_hz).to(signal.device)
    rejected = torch.zeros((len(signal), len(frequencies)), dtype=torch.bool, device=signal.device)
    labels: list[int] = []
    for index, sample_id in enumerate(sample_ids):
        digest = _digest(sample_id)
        label = digest[0] % len(BANDS)
        labels.append(label)
        low, high = BANDS[label]
        width = high - low
        low += ((digest[1] / 255.0) * 2.0 - 1.0) * jitter * width
        high += ((digest[2] / 255.0) * 2.0 - 1.0) * jitter * width
        low = max(0.1, low)
        high = min(sampling_hz * 0.49, max(low + 0.1, high))
        rejected[index] = (frequencies >= low) & (frequencies <= high)
    spectrum = spectrum.masked_fill(rejected[:, None, None, :], 0)
    filtered = torch.fft.irfft(spectrum, n=signal.shape[-1], dim=-1, norm="backward").to(signal.dtype)
    return filtered, torch.tensor(labels, dtype=torch.long, device=signal.device)


def deterministic_patch_mask(sample_ids: list[str], channels: int, patches: int, ratio: float, device: torch.device) -> torch.Tensor:
    rows = []
    for sample_id in sample_ids:
        seed = int.from_bytes(_digest(sample_id)[:8], "little")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        rows.append(torch.rand((channels, patches), generator=generator) < ratio)
    return torch.stack(rows).to(device=device, non_blocking=True)


def temporal_rearrange_view(signal: torch.Tensor, *, p_shuffle: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Create the upstream two-chunk temporal-order pretext view.

    ``signal`` is ``[B, C, S, P]``.  The two contiguous chunks are swapped
    for a balanced random half of the batch.  The label is 0 for the original
    order and 1 for the swapped order.  A two-chunk task is used because it is
    the configuration in the CHB downstream model and is identifiable without
    introducing a new pretext definition.
    """
    if signal.ndim != 4:
        raise ValueError(f"expected [B,C,S,P], got {tuple(signal.shape)}")
    batch = signal.shape[0]
    flat = signal.reshape(batch, signal.shape[1], -1)
    labels = (torch.rand(batch, device=signal.device) < float(p_shuffle)).long()
    # Keep the task balanced in every source batch rather than relying on an
    # approximate Bernoulli proportion.
    half = batch // 2
    labels.zero_()
    if half:
        labels[torch.randperm(batch, device=signal.device)[:half]] = 1
    midpoint = flat.shape[-1] // 2
    if midpoint <= 0:
        raise ValueError("temporal view needs at least two time chunks")
    first, second = flat[..., :midpoint], flat[..., midpoint:]
    swapped = torch.cat([second, first], dim=-1)
    output = torch.where(labels[:, None, None].bool(), swapped, flat)
    return output.reshape_as(signal), labels


def deterministic_temporal_view(signal: torch.Tensor, sample_ids: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic temporal-order view for validation/test and parity audits."""
    if len(signal) != len(sample_ids):
        raise ValueError("sample ids do not align")
    batch = signal.shape[0]
    flat = signal.reshape(batch, signal.shape[1], -1)
    labels = torch.tensor([_digest(sample_id)[0] & 1 for sample_id in sample_ids], dtype=torch.long, device=signal.device)
    midpoint = flat.shape[-1] // 2
    first, second = flat[..., :midpoint], flat[..., midpoint:]
    swapped = torch.cat([second, first], dim=-1)
    output = torch.where(labels[:, None, None].bool(), swapped, flat)
    return output.reshape_as(signal), labels
