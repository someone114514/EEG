"""Verify the required files in a built migration bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "migration_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    checked = 0
    for item in manifest["files"]:
        if not item.get("required", True):
            continue
        path = root / item["bundle_path"]
        if not path.is_file():
            failures.append(f"missing: {item['bundle_path']}")
            continue
        checked += 1
        actual = sha256(path)
        expected = item.get("bundle_sha256")
        if expected and actual != expected:
            failures.append(f"hash mismatch: {item['bundle_path']} expected={expected} actual={actual}")
    print(json.dumps({"checked_required": checked, "failures": failures, "status": "passed" if not failures else "failed"}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
