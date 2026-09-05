from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import mne
import numpy as np
from scipy.signal import resample_poly

from bfa.preprocessing.channels import CHANNELS, MissingCanonicalChannel


LEGACY_TO_MODERN = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
MODEL_VIEW_CONFIG = {
    "singlem": {"sampling_hz": 128.0, "bandpass_hz": (0.5, 50.0), "scale": 0.01},
    "cbramod": {"sampling_hz": 200.0, "bandpass_hz": (0.3, 75.0), "scale": 0.01},
    "tcn_gat": {"sampling_hz": 256.0, "bandpass_hz": (0.5, 45.0), "scale": 1.0},
}


def normalize_tuh_electrode(name: str) -> str:
    value = name.upper().strip()
    if value.startswith("EEG "):
        value = value[4:]
    for suffix in ("-REF", "-LE", "-AR", "-AVG"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    value = value.replace(" ", "")
    return LEGACY_TO_MODERN.get(value, value)


def derive_tuh_bipolar_uv(channel_names: list[str], data_v: np.ndarray) -> np.ndarray:
    if data_v.ndim != 2 or data_v.shape[0] != len(channel_names):
        raise ValueError("TUH data and channel names are not aligned")
    electrodes: dict[str, int] = {}
    for index, name in enumerate(channel_names):
        normalized = normalize_tuh_electrode(name)
        electrodes.setdefault(normalized, index)
    required = {electrode for pair in CHANNELS for electrode in pair.split("-")}
    missing = sorted(required - set(electrodes))
    if missing:
        raise MissingCanonicalChannel(f"cannot derive TUH bipolar montage; missing electrodes: {missing}")
    return np.stack([
        (data_v[electrodes[left]] - data_v[electrodes[right]]) * 1e6
        for left, right in (pair.split("-") for pair in CHANNELS)
    ]).astype(np.float32)


def load_tuh_canonical_raw_uv(path: Path) -> tuple[np.ndarray, float, list[str]]:
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    data_uv = derive_tuh_bipolar_uv(raw.ch_names, raw.get_data())
    return data_uv, float(raw.info["sfreq"]), list(raw.ch_names)


def tuh_model_view(raw_uv: np.ndarray, sampling_hz: float, model: str) -> np.ndarray:
    if model not in MODEL_VIEW_CONFIG:
        raise ValueError(f"unknown model: {model}")
    config = MODEL_VIEW_CONFIG[model]
    filtered = raw_uv.astype(np.float64)
    if sampling_hz / 2 > 61.0:
        filtered = mne.filter.notch_filter(filtered, sampling_hz, freqs=[60.0], verbose="ERROR")
    low, high = config["bandpass_hz"]
    safe_high = min(float(high), sampling_hz / 2 - 0.5)
    if safe_high <= float(low):
        raise ValueError(f"sampling rate {sampling_hz} is incompatible with bandpass {low}-{high}")
    filtered = mne.filter.filter_data(filtered, sampling_hz, float(low), safe_high, verbose="ERROR")
    target = int(config["sampling_hz"])
    ratio = Fraction(target / sampling_hz).limit_denominator(10_000)
    output = resample_poly(filtered, ratio.numerator, ratio.denominator, axis=-1)
    output = output * float(config["scale"])
    if not np.isfinite(output).all():
        raise ValueError(f"non-finite {model} view")
    return output.astype(np.float32)


def tuh_quality_features(raw_uv: np.ndarray, sampling_hz: float) -> np.ndarray:
    window_samples = int(round(10 * sampling_hz))
    stride_samples = int(round(2 * sampling_hz))
    starts = np.arange(0, raw_uv.shape[-1] - window_samples + 1, stride_samples)
    output = np.empty((len(starts), raw_uv.shape[0], 3), dtype=np.float32)
    frequencies = np.fft.rfftfreq(window_samples, d=1.0 / sampling_hz)
    peak = (frequencies >= 59) & (frequencies <= 61)
    shoulders = ((frequencies >= 55) & (frequencies < 59)) | ((frequencies > 61) & (frequencies <= 65))
    channel_extreme = np.quantile(np.abs(raw_uv), 0.999, axis=-1)
    for offset in range(0, len(starts), 128):
        batch_starts = starts[offset : offset + 128]
        windows = np.stack([raw_uv[:, start : start + window_samples] for start in batch_starts])
        differences = np.diff(windows, axis=-1)
        flat_fraction = np.mean(differences == 0, axis=-1)
        extreme = np.abs(windows[..., :-1]) >= channel_extreme[None, :, None]
        clipping_fraction = np.mean((differences == 0) & extreme, axis=-1)
        spectrum = np.abs(np.fft.rfft(windows, axis=-1)) ** 2
        line_noise = spectrum[..., peak].sum(axis=-1) / (
            spectrum[..., shoulders].sum(axis=-1) + np.finfo(np.float32).eps
        )
        output[offset : offset + len(batch_starts), :, 0] = line_noise
        output[offset : offset + len(batch_starts), :, 1] = clipping_fraction
        output[offset : offset + len(batch_starts), :, 2] = flat_fraction
    if not np.isfinite(output).all():
        raise ValueError("non-finite TUH quality features")
    return output
