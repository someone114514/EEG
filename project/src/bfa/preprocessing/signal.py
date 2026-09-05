from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import mne
import numpy as np
from scipy.signal import periodogram, resample_poly

from bfa.preprocessing.channels import CHANNELS, MissingCanonicalChannel, canonical_order

MODEL_RATES = {"singlem": 128, "cbramod": 200, "tcn_gat": 256}


def load_canonical_uv(path: Path) -> tuple[np.ndarray, float]:
    data_uv, sampling_hz = load_canonical_raw_uv(path)
    data_uv = mne.filter.notch_filter(
        data_uv.astype(np.float64), sampling_hz, freqs=[60.0], verbose="ERROR"
    )
    data_uv = mne.filter.filter_data(
        data_uv, sampling_hz, 0.5, 45.0, verbose="ERROR"
    )
    return data_uv.astype(np.float32), sampling_hz


def load_canonical_raw_uv(path: Path) -> tuple[np.ndarray, float]:
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    sampling_hz = float(raw.info["sfreq"])
    data_v = raw.get_data()
    try:
        order = canonical_order(raw.ch_names)
        data_uv = data_v[order] * 1e6
    except MissingCanonicalChannel:
        data_uv = derive_bipolar_uv(raw.ch_names, data_v)
    return data_uv.astype(np.float32), sampling_hz


def _electrode_name(channel: str) -> str:
    name = channel.upper().strip()
    if name.endswith("-CS2"):
        name = name[:-4]
    if name == "01":
        name = "O1"
    return name


def derive_bipolar_uv(channel_names: list[str], data_v: np.ndarray) -> np.ndarray:
    """Derive the frozen bipolar montage from a common-reference recording."""
    electrodes: dict[str, int] = {}
    for index, channel in enumerate(channel_names):
        electrode = _electrode_name(channel)
        if electrode not in electrodes and not electrode.startswith("--"):
            electrodes[electrode] = index
    required = {electrode for channel in CHANNELS for electrode in channel.split("-")}
    missing = sorted(required - electrodes.keys())
    if missing:
        raise MissingCanonicalChannel(
            f"cannot derive canonical bipolar montage; missing electrodes: {missing}"
        )
    return np.stack(
        [
            (data_v[electrodes[left]] - data_v[electrodes[right]]) * 1e6
            for left, right in (channel.split("-") for channel in CHANNELS)
        ]
    )


def model_view(data_uv: np.ndarray, sampling_hz: float, model: str) -> np.ndarray:
    if model not in MODEL_RATES:
        raise ValueError(f"unknown model: {model}")
    target = MODEL_RATES[model]
    ratio = Fraction(target / sampling_hz).limit_denominator(10_000)
    output = resample_poly(data_uv, ratio.numerator, ratio.denominator, axis=-1)
    if model == "singlem":
        scaled = output * 1e-2
        clipped = np.mean(np.abs(scaled) > 1.5)
        if clipped > 0.01:
            raise ValueError(f"SingLEM clipping fraction {clipped:.4f} exceeds 1%")
        output = np.clip(scaled, -1.5, 1.5)
    elif model == "cbramod":
        output = output / 100.0
    return output.astype(np.float32)


def dominant_frequency(signal: np.ndarray, sampling_hz: float) -> float:
    frequencies, power = periodogram(signal, fs=sampling_hz)
    return float(frequencies[int(np.argmax(power))])
