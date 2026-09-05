"""Build the reproducibility manifest for the staged migration bundle."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPO = Path("/root/b_false_alarm_atlas")
SOURCE_EXTERNAL = Path("/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/work/NeuroTTT/CBraMod")
SOURCE_RELEASE = SOURCE_REPO / "outputs/reports/meta-ttt-chbmit-5fold-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def size(path: Path) -> int:
    return path.stat().st_size


def source_for(bundle_path: Path) -> Path | None:
    rel = bundle_path.relative_to(ROOT)
    rel_text = rel.as_posix()
    if rel_text.startswith("external/NeuroTTT_CBraMod/"):
        return SOURCE_EXTERNAL / rel_text.removeprefix("external/NeuroTTT_CBraMod/")
    if rel_text.startswith("project/src/"):
        return SOURCE_REPO / "src" / rel_text.removeprefix("project/src/")
    if rel_text.startswith("project/scripts/"):
        return SOURCE_REPO / "scripts" / rel_text.removeprefix("project/scripts/")
    if rel_text == "project/pyproject.toml":
        return SOURCE_REPO / "pyproject.toml"
    if rel_text == "project/README.md":
        return Path("/mnt/c/Users/User/Documents/ChatGPT/EEG_ZiquanBaoBao/README.md")
    if rel_text == "project/BAND_TTT_V2_MATRIX.md":
        return Path("/mnt/c/Users/User/Documents/ChatGPT/EEG_ZiquanBaoBao/BAND_TTT_V2_MATRIX.md")
    if rel_text == "environment/versions.txt":
        return SOURCE_REPO / "environment/versions.txt"
    if rel_text.startswith("manifests/"):
        return SOURCE_REPO / rel_text
    if rel_text.startswith("results/meta-ttt-chbmit-5fold-v1/"):
        return SOURCE_RELEASE / rel_text.removeprefix("results/meta-ttt-chbmit-5fold-v1/")
    if rel_text.startswith("artifacts/meta-ttt-chbmit-5fold-v1/"):
        return SOURCE_RELEASE / rel_text.removeprefix("artifacts/meta-ttt-chbmit-5fold-v1/")
    return None


def source_hash_override(bundle_path: Path) -> str | None:
    path = ROOT / "provenance" / "pre_post_patch_hashes.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    rel = bundle_path.relative_to(ROOT).as_posix()
    for item in payload.get("files", []):
        if item.get("bundle_path") == rel:
            return item.get("pre_patch_sha256")
    return None


def git_info() -> dict[str, object]:
    result: dict[str, object] = {}
    for key, command in (
        ("revision", ["git", "-C", str(SOURCE_REPO), "rev-parse", "HEAD"]),
        ("status_porcelain", ["git", "-C", str(SOURCE_REPO), "status", "--short"]),
    ):
        try:
            result[key] = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            result[key] = None
    return result


def data_record(label: str, path: Path, sample: Path | None = None) -> dict[str, object]:
    record: dict[str, object] = {"label": label, "path": str(path), "exists": path.exists()}
    if path.exists():
        try:
            record["bytes"] = sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.is_dir() else path.stat().st_size
            record["files"] = sum(1 for item in path.rglob("*") if item.is_file()) if path.is_dir() else 1
        except OSError as exc:
            record["scan_error"] = str(exc)
    if sample is not None:
        record["sample"] = str(sample)
        record["sample_exists"] = sample.is_file()
        if sample.is_file():
            record["sample_bytes"] = sample.stat().st_size
            record["sample_sha256"] = sha256(sample)
    return record


def main() -> None:
    files: list[dict[str, object]] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel in {"migration_manifest.json", "data_manifest.json"} or rel.endswith(".zip"):
            continue
        if any(part == "__pycache__" for part in path.parts):
            continue
        source = source_for(path)
        override = source_hash_override(path)
        item: dict[str, object] = {
            "bundle_path": rel,
            "bytes": size(path),
            "bundle_sha256": sha256(path),
            "required": True,
            "kind": "artifact" if rel.startswith("artifacts/") else "result" if rel.startswith("results/") else "code" if (rel.endswith(".py") or rel.endswith(".sh") or rel.startswith("external/") or rel.startswith("project/")) else "metadata",
        }
        if source is not None:
            item["source_path"] = str(source)
            item["source_exists"] = source.is_file()
            item["source_sha256"] = override or (sha256(source) if source.is_file() else None)
        files.append(item)

    manifest = {
        "bundle_id": "metaTTT_migration_20260905",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": str(SOURCE_REPO),
        "source_external": str(SOURCE_EXTERNAL),
        "source_release": str(SOURCE_RELEASE),
        "git": git_info(),
        "path_portability": "default absolute paths were changed only in files listed in provenance/pre_post_patch_hashes.json; see provenance/path_patches.diff",
        "files": files,
    }
    (ROOT / "migration_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    data = {
        "data_policy": "raw data and the precomputed cache stay outside the code/artifact archive; copy them with tools/copy_external_data.sh",
        "records": [
            data_record("chbmit_raw_edf", Path("/mnt/d/EEGData/chbmit-1.0.0"), Path("/mnt/d/EEGData/chbmit-1.0.0/chb01/chb01_01.edf")),
            data_record("cbramod_precomputed_cache", Path("/mnt/d/EEGData/bfa_cache_v3_official_noclip/cbramod"), Path("/mnt/d/EEGData/bfa_cache_v3_official_noclip/cbramod/chb01/chb01_01.npy")),
        ],
        "cache_contract": {"channels": 16, "sample_rate_hz": 200, "window_seconds": 10, "dtype": "float32", "unit": "microvolts/100", "normalization": "no additional scaling"},
    }
    (ROOT / "data_manifest.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(files), "manifest": str(ROOT / "migration_manifest.json"), "data_manifest": str(ROOT / "data_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
