"""Apply only path-portability edits to the staged current source tree.

The algorithm/model code is copied first.  This script records hashes and a
unified diff, then changes default absolute paths to environment-driven paths.
It is intentionally limited to the files that are used by the current
MetaTTT/Band-TTT entry points.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external" / "NeuroTTT_CBraMod"
PROJECT = ROOT / "project"

TARGETS = [
    EXTERNAL / "chbmit_groupkfold" / name
    for name in (
        "data.py",
        "evaluate.py",
        "meta_evaluate.py",
        "meta_model.py",
        "meta_summarize.py",
        "meta_train.py",
        "preflight.py",
        "summarize.py",
        "train.py",
    )
] + [
    EXTERNAL / name
    for name in (
        "chbmit_groupkfold_meta_eval_queue.py",
        "chbmit_groupkfold_meta_preflight.py",
        "chbmit_groupkfold_meta_queue.py",
        "chbmit_groupkfold_meta_smoke.py",
    )
] + [
    PROJECT / "scripts" / name
    for name in (
        "110_preflight_no_duplicate_overlap.py",
        "202_cbramod_same_patient_adaptation.py",
        "210_joint_ttt_train.py",
        "212_meta_ttt_train.py",
        "214_evaluate_joint_ttt.py",
        "264_freeze_band_ttt_v2.py",
        "265_evaluate_band_ttt_v2.py",
        "266_summarize_band_ttt_v2.py",
        "267_queue_band_ttt_v2.py",
        "268_import_existing_band_ttt_v2.py",
        "270_queue_band_ttt_v2_paired.py",
    )
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_imports(text: str, names: set[str]) -> str:
    missing = [name for name in sorted(names) if not re.search(rf"^import {re.escape(name)}\s*$", text, re.MULTILINE)]
    if not missing:
        return text
    lines = text.splitlines(keepends=True)
    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    if insert_at < len(lines) and lines[insert_at].startswith('"""'):
        # Do not try to parse a multiline module docstring here; all current
        # files place imports after the future-import or at the top.  The
        # fallback below inserts after a future import when present.
        end = insert_at
        if lines[end].count('"""') < 2:
            end += 1
            while end < len(lines) and '"""' not in lines[end]:
                end += 1
            end += 1
        insert_at = end
    future = [i for i, line in enumerate(lines) if line.startswith("from __future__ import")]
    if future:
        insert_at = future[-1] + 1
    lines[insert_at:insert_at] = [f"import {name}\n" for name in missing]
    return "".join(lines)


def patch(text: str) -> str:
    original = text
    needs_os = False
    needs_sys = False

    def root_expr(match: re.Match[str]) -> str:
        nonlocal needs_os
        needs_os = True
        suffix = match.group(1)
        if suffix:
            return 'Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas")) / ' + repr(suffix.lstrip("/"))
        return 'Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))'

    text = re.sub(r'Path\("/root/b_false_alarm_atlas([^" ]*)"\)', root_expr, text)

    def cache_expr(match: re.Match[str]) -> str:
        nonlocal needs_os
        needs_os = True
        return 'Path(os.environ.get("BFA_CACHE_ROOT", "/mnt/d/EEGData/bfa_cache_v3_official_noclip/cbramod"))'

    text = re.sub(r'Path\("/mnt/d/EEGData/bfa_cache_v3_official_noclip/cbramod"\)', cache_expr, text)

    def raw_expr(match: re.Match[str]) -> str:
        nonlocal needs_os
        needs_os = True
        return 'Path(os.environ.get("BFA_RAW_ROOT", "/mnt/d/EEGData/chbmit-1.0.0"))'

    text = re.sub(r'Path\("/mnt/d/EEGData/chbmit-1\.0\.0"\)', raw_expr, text)

    def external_expr(match: re.Match[str]) -> str:
        nonlocal needs_os
        needs_os = True
        return 'Path(os.environ.get("NEUROTTT_CODE_ROOT", "/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/work/NeuroTTT/CBraMod"))'

    text = re.sub(r'Path\("/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/work/NeuroTTT/CBraMod"\)', external_expr, text)

    def python_expr(match: re.Match[str]) -> str:
        nonlocal needs_os, needs_sys
        needs_os = True
        needs_sys = True
        return 'Path(os.environ.get("META_TTT_PYTHON", sys.executable))'

    text = re.sub(r'Path\("/root/b_false_alarm_atlas/\.venv/bin/python"\)', python_expr, text)
    if needs_os:
        text = ensure_imports(text, {"os"})
    if needs_sys:
        text = ensure_imports(text, {"sys"})
    return text if text != original else original


def main() -> None:
    patch_dir = ROOT / "provenance"
    patch_dir.mkdir(parents=True, exist_ok=True)
    source_hashes: list[dict[str, str]] = []
    diff: list[str] = []
    changed = 0
    for path in TARGETS:
        if not path.is_file():
            continue
        before = path.read_text(encoding="utf-8")
        after = patch(before)
        rel = path.relative_to(ROOT).as_posix()
        source_hashes.append({
            "bundle_path": rel,
            "pre_patch_sha256": sha256_bytes(before.encode()),
            "post_patch_sha256": sha256_bytes(after.encode()),
            "changed": str(before != after).lower(),
        })
        if before != after:
            changed += 1
            path.write_text(after, encoding="utf-8", newline="")
            diff.extend(difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile=f"a/{rel}", tofile=f"b/{rel}",
            ))
    (patch_dir / "pre_post_patch_hashes.json").write_text(
        json.dumps({"path_patch_version": 1, "files": source_hashes}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (patch_dir / "path_patches.diff").write_text("".join(diff), encoding="utf-8")
    print(json.dumps({"target_files": len(source_hashes), "changed_files": changed, "diff": str(patch_dir / "path_patches.diff")}, indent=2))


if __name__ == "__main__":
    main()
