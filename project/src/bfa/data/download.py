from __future__ import annotations

import subprocess
from pathlib import Path

CHBMIT_URL = "https://physionet.org/files/chbmit/1.0.0/"


def download_chbmit(destination: Path) -> None:
    """Download CHB-MIT with wget using recursive, resumable semantics."""
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "wget",
            "-r",
            "-N",
            "-c",
            "-np",
            "--cut-dirs=3",
            "-nH",
            CHBMIT_URL,
        ],
        cwd=destination,
        check=True,
    )
