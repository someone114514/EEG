from __future__ import annotations

import json
from hashlib import sha256
from typing import Any


def cache_key(edf_sha256: str, preprocessing_config: dict[str, Any]) -> str:
    config = json.dumps(
        preprocessing_config, sort_keys=True, separators=(",", ":")
    ).encode()
    return sha256(edf_sha256.encode() + b":" + config).hexdigest()
