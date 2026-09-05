from __future__ import annotations

import json
import re
from pathlib import Path

import mne
import pandas as pd

from bfa.provenance import sha256_file

PRIMARY_EXCLUDED_CASES = frozenset({"chb24"})


def canonical_patient_id(case_id: str) -> str:
    normalized = case_id.strip().lower()
    return "chb01_21" if normalized in {"chb01", "chb21"} else normalized


def is_primary_case(case_id: str) -> bool:
    """Return whether a case belongs to the protocol-v3 frozen 22-subject cohort."""
    return case_id.strip().lower() not in PRIMARY_EXCLUDED_CASES


def parse_summary_intervals(text: str) -> list[tuple[str, float, float]]:
    current = ""
    starts: dict[int, float] = {}
    rows: list[tuple[str, float, float]] = []
    next_implicit_index = 1
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("File Name:"):
            current = line.split(":", 1)[1].strip()
            starts = {}
            next_implicit_index = 1
            continue
        start_match = re.match(
            r"Seizure(?:\s+(\d+))?\s+Start Time:\s*([0-9.]+)\s+seconds?",
            line,
            flags=re.IGNORECASE,
        )
        if start_match:
            index = int(start_match.group(1) or next_implicit_index)
            starts[index] = float(start_match.group(2))
            continue
        end_match = re.match(
            r"Seizure(?:\s+(\d+))?\s+End Time:\s*([0-9.]+)\s+seconds?",
            line,
            flags=re.IGNORECASE,
        )
        if end_match:
            index = int(end_match.group(1) or next_implicit_index)
            if not current or index not in starts:
                raise ValueError(f"orphan seizure end line: {line}")
            rows.append((current, starts[index], float(end_match.group(2))))
            next_implicit_index = index + 1
    return rows


def _summary_index(root: Path) -> dict[str, list[tuple[float, float]]]:
    index: dict[str, list[tuple[float, float]]] = {}
    for summary in sorted(root.rglob("*summary*.txt")):
        text = summary.read_text(encoding="utf-8", errors="replace")
        for recording_id, start_s, end_s in parse_summary_intervals(text):
            index.setdefault(recording_id, []).append((start_s, end_s))
    return index


def scan_dataset(
    root: Path, audit_path: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    seizure_index = _summary_index(root)
    recordings: list[dict[str, object]] = []
    seizures: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    excluded: list[str] = []

    for edf_path in sorted(root.rglob("*.edf")):
        case_id = edf_path.parent.name.lower()
        if not is_primary_case(case_id):
            excluded.append(edf_path.relative_to(root).as_posix())
            continue
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
            sampling_hz = float(raw.info["sfreq"])
            duration_s = float(raw.n_times / sampling_hz)
            patient_id = canonical_patient_id(case_id)
            recording_id = edf_path.name
            recordings.append(
                {
                    "patient_id": patient_id,
                    "source_case_id": case_id,
                    "recording_id": recording_id,
                    "relative_path": edf_path.relative_to(root).as_posix(),
                    "sampling_hz": sampling_hz,
                    "duration_s": duration_s,
                    "channels": list(raw.ch_names),
                    "file_bytes": edf_path.stat().st_size,
                    "sha256": sha256_file(edf_path),
                }
            )
            for start_s, end_s in seizure_index.get(recording_id, []):
                if not 0 <= start_s < end_s <= duration_s:
                    raise ValueError(
                        f"invalid seizure interval {recording_id}: {start_s}-{end_s}/{duration_s}"
                    )
                seizures.append(
                    {
                        "patient_id": patient_id,
                        "recording_id": recording_id,
                        "start_s": start_s,
                        "end_s": end_s,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - every unreadable EDF must enter the audit
            errors.append({"path": str(edf_path), "error": f"{type(exc).__name__}: {exc}"})

    recordings_frame = pd.DataFrame(recordings)
    seizures_frame = pd.DataFrame(seizures)
    destination = audit_path or root / "data_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "root": str(root),
                "recording_count": len(recordings),
                "seizure_count": len(seizures),
                "primary_excluded_cases": sorted(PRIMARY_EXCLUDED_CASES),
                "excluded_recording_count": len(excluded),
                "excluded_recordings": excluded,
                "errors": errors,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return recordings_frame, seizures_frame
