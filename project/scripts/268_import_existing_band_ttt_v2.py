"""Register mathematically identical existing Window results in the final v2 release."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
OUT = ROOT / "outputs" / "reports" / "band-ttt-v2-fold01"
V1 = ROOT / "outputs" / "reports" / "meta-ttt-chbmit-5fold-v1"
EARLY_V2 = ROOT / "outputs" / "reports" / "band-ttt-v2" / "independent"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n")
    os.replace(temporary, path)


def config(manifest: dict, config_id: str) -> dict:
    return next(item for item in manifest["configurations"] if item["config_id"] == config_id)


def base(manifest: dict, config_id: str, fold: int, split: str, source_probability: Path, source_payload: dict) -> dict:
    checkpoint = V1 / "runs" / f"meta_band_fold{fold}_seed3407" / "best.pt"
    frozen_lock = json.loads((V1 / "evaluation" / "meta_band_frozen" / f"fold{fold}_seed3407" / "validation_metrics.json").read_text())
    frozen_threshold = float(frozen_lock["selected_event_operating_point"]["threshold"])
    return {
        "release_id": "band-ttt-v2-fold01", "status": f"{split}_complete", "config": config(manifest, config_id),
        "fold": fold, "seed": 3407, "split": split, "rows": int(source_payload["rows"]),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "learned_alpha": float(source_payload.get("alpha", source_payload.get("ttt_lr", 0.0))),
        "probability_path": str(source_probability), "probability_sha256": sha256(source_probability),
        "elapsed_s": float(source_payload["elapsed_s"]), "gpu_peak_mib": float(source_payload.get("gpu_peak_mib", 0.0)),
        "test_labels_used_for_adaptation": False, "create_graph": False,
        "official_record_order_sha256": manifest["record_order_sha256"],
        "frozen_baseline_threshold": frozen_threshold,
        "threshold_source": "validation_only", "imported_without_recomputation": True,
        "import_source": str(source_probability), "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    manifest = json.loads((OUT / "frozen_manifest.json").read_text())
    imported = []
    # Formal v1 is exactly Window + K=1 + Global SGD, including validation-locked test.
    config_id = "window_k1_global_sgd"
    for fold in (0, 1):
        source_dir = V1 / "evaluation" / "meta_band_ttt" / f"fold{fold}_seed3407"
        destination = OUT / "evaluation" / config_id / f"fold{fold}_seed3407"
        for split, filename in (("validation", "validation_metrics.json"), ("test", "test_completed.json")):
            payload = json.loads((source_dir / filename).read_text())
            probability = source_dir / f"{split}_probabilities.parquet"
            result = base(manifest, config_id, fold, split, probability, payload)
            result["selected_event_operating_point"] = payload["selected_event_operating_point"]
            result["matched_frozen_threshold_metrics"] = payload["selected_event_operating_point"]
            result["test_evaluation_count"] = 0 if split == "validation" else 1
            if split == "test":
                frozen_test = json.loads((V1 / "evaluation" / "meta_band_frozen" / f"fold{fold}_seed3407" / "test_completed.json").read_text())
                result["existing_frozen_baseline"] = frozen_test["selected_event_operating_point"]
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_dir / "test_patient_waterfall.csv", destination / "test_patient_waterfall.csv")
                shutil.copy2(source_dir / "test_patient_waterfall.csv", destination / "test_patient_waterfall_matched_frozen_threshold.csv")
            atomic_json(destination / f"{split}_completed.json", result)
            imported.append({"config": config_id, "fold": fold, "split": split})
    # Early v2 K=3 Global validation used the same checkpoint, deterministic view,
    # global learned alpha, independent reset, and exact K-step functional update.
    config_id = "window_k3_global_sgd"
    for fold in (0, 1):
        source_dir = EARLY_V2 / "k3" / f"fold{fold}"
        payload = json.loads((source_dir / "validation_completed.json").read_text())
        probability = source_dir / "validation_probabilities.parquet"
        destination = OUT / "evaluation" / config_id / f"fold{fold}_seed3407"
        result = base(manifest, config_id, fold, "validation", probability, payload)
        result["selected_event_operating_point"] = payload["selected_event_operating_point"]
        result["matched_frozen_threshold_metrics"] = payload["selected_event_operating_point"]
        result["test_evaluation_count"] = 0
        atomic_json(destination / "validation_completed.json", result)
        imported.append({"config": config_id, "fold": fold, "split": "validation"})
    atomic_json(OUT / "imported_existing_results.json", {"status": "complete", "count": len(imported), "items": imported})
    print(json.dumps({"status": "complete", "imported_jobs": len(imported)}, indent=2))


if __name__ == "__main__":
    main()
