from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

PARTITIONS = ("train", "validation", "test")
CAPACITIES = {"train": 13, "validation": 4, "test": 5}
FRACTIONS = {"train": 0.60, "validation": 0.20, "test": 0.20}


@dataclass
class SplitState:
    n: float = 0.0
    seizures: float = 0.0
    hours: float = 0.0
    strata: dict[str, float] = field(default_factory=dict)


def split_cost(state: SplitState, targets: SplitState) -> float:
    eps = 1e-9
    patient_err = abs(state.n / max(targets.n, eps) - 1.0)
    seizure_err = abs(state.seizures / max(targets.seizures, eps) - 1.0)
    hours_err = abs(state.hours / max(targets.hours, eps) - 1.0)
    strata_err = sum(
        abs(state.strata.get(key, 0.0) / max(value, eps) - 1.0)
        for key, value in targets.strata.items()
        if value > 0
    )
    return 2.0 * patient_err + seizure_err + hours_err + 0.5 * strata_err


def _add_patient(state: SplitState, row: pd.Series, strata: list[str]) -> SplitState:
    return SplitState(
        n=state.n + 1,
        seizures=state.seizures + float(row["seizures"]),
        hours=state.hours + float(row["hours"]),
        strata={key: state.strata.get(key, 0.0) + float(row[key]) for key in strata},
    )


def make_group_split(patient_stats: pd.DataFrame, seed: int) -> dict[str, list[str]]:
    required = {"patient_id", "seizures", "hours"}
    missing = required - set(patient_stats.columns)
    if missing:
        raise ValueError(f"patient_stats missing columns: {sorted(missing)}")
    if len(patient_stats) != 22 or patient_stats.patient_id.nunique() != 22:
        raise ValueError("the frozen CHB-MIT split requires exactly 22 canonical patients")

    frame = patient_stats.sort_values("patient_id").reset_index(drop=True)
    strata = sorted(column for column in frame.columns if column.startswith("stratum_"))
    totals = SplitState(
        n=22,
        seizures=float(frame.seizures.sum()),
        hours=float(frame.hours.sum()),
        strata={key: float(frame[key].sum()) for key in strata},
    )
    targets = {
        part: SplitState(
            n=float(CAPACITIES[part]),
            seizures=totals.seizures * FRACTIONS[part],
            hours=totals.hours * FRACTIONS[part],
            strata={key: value * FRACTIONS[part] for key, value in totals.strata.items()},
        )
        for part in PARTITIONS
    }
    states = {part: SplitState(strata={key: 0.0 for key in strata}) for part in PARTITIONS}
    assignments = {part: [] for part in PARTITIONS}
    order = np.random.default_rng(seed).permutation(len(frame))

    for index in order:
        row = frame.iloc[int(index)]
        candidates: list[tuple[float, int, str, SplitState]] = []
        for tie_order, part in enumerate(PARTITIONS):
            if len(assignments[part]) >= CAPACITIES[part]:
                continue
            proposed = _add_patient(states[part], row, strata)
            delta = split_cost(proposed, targets[part]) - split_cost(states[part], targets[part])
            candidates.append((delta, tie_order, part, proposed))
        _, _, selected, proposed_state = min(candidates, key=lambda item: (item[0], item[1]))
        assignments[selected].append(str(row.patient_id))
        states[selected] = proposed_state

    return {part: sorted(assignments[part]) for part in PARTITIONS}


def write_split_manifest(split: dict[str, list[str]], seed: int, destination: Path) -> None:
    split_bytes = json.dumps(split, sort_keys=True, separators=(",", ":")).encode()
    payload = {"seed": seed, "split_sha256": sha256(split_bytes).hexdigest(), **split}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _patient_stats(recordings: pd.DataFrame, seizures: pd.DataFrame) -> pd.DataFrame:
    stats = (
        recordings.groupby("patient_id", as_index=False)
        .agg(hours=("duration_s", lambda values: float(values.sum()) / 3600.0))
    )
    counts = seizures.groupby("patient_id").size().rename("seizures")
    stats = stats.merge(counts, on="patient_id", how="left").fillna({"seizures": 0})
    stats["seizures"] = stats.seizures.astype(int)
    stats["stratum_high_burden"] = (stats.seizures >= stats.seizures.median()).astype(int)
    stats["stratum_long_recording"] = (stats.hours >= stats.hours.median()).astype(int)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, default=Path("manifests"))
    args = parser.parse_args()
    recordings = pd.read_parquet(args.manifests / "recordings.parquet")
    seizures = pd.read_parquet(args.manifests / "seizures.parquet")
    stats = _patient_stats(recordings, seizures)
    for seed in args.seeds:
        split = make_group_split(stats, seed)
        write_split_manifest(split, seed, args.out / f"split_{seed}.json")


if __name__ == "__main__":
    main()
