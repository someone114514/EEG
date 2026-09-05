from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from bfa.contracts import RunIdentity


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def write_run_identity(
    path: Path, config: dict[str, Any], split_hash: str, seed: int
) -> RunIdentity:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    config_hash = sha256(config_bytes).hexdigest()
    run_id = sha256(f"{config_hash}:{split_hash}:{seed}:{commit}".encode()).hexdigest()[:16]
    identity = RunIdentity(run_id, split_hash, seed, commit, config_hash)
    payload = {**asdict(identity), "config": config}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return identity
