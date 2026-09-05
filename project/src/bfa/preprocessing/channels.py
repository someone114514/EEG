from __future__ import annotations

import re

CHANNELS = (
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
)


class MissingCanonicalChannel(ValueError):
    pass


def normalize_channel(name: str) -> str:
    normalized = name.upper().replace("EEG ", "").replace("-REF", "").replace(" ", "")
    return re.sub(r"-\d+$", "", normalized)


def canonical_order(channel_names: list[str] | tuple[str, ...]) -> list[int]:
    normalized: dict[str, int] = {}
    for index, name in enumerate(channel_names):
        normalized.setdefault(normalize_channel(name), index)
    missing = [channel for channel in CHANNELS if channel not in normalized]
    if missing:
        raise MissingCanonicalChannel(f"missing canonical channels: {missing}")
    return [normalized[channel] for channel in CHANNELS]
