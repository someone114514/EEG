from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256


@dataclass(frozen=True)
class MatrixRun:
    suite: str
    configuration: str
    model: str
    split_seed: int
    model_seed: int
    run_id: str


def content_addressed_run_id(
    *,
    configuration: str,
    model: str,
    split_seed: int,
    model_seed: int,
    config_sha256: str,
    split_sha256: str,
    code_commit: str,
) -> str:
    payload = {
        "configuration": configuration,
        "model": model,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "config_sha256": config_sha256,
        "split_sha256": split_sha256,
        "code_commit": code_commit,
    }
    token = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return f"{configuration}-{model}-split{split_seed}-seed{model_seed}-{token}"


def _runs(
    *,
    suite: str,
    configuration: str,
    models: Iterable[str],
    model_seeds: Iterable[int],
    split_seeds: Iterable[int],
    config_sha256: str = "test-config",
    split_sha256: str = "test-split",
    code_commit: str = "test-commit",
) -> list[MatrixRun]:
    return [
        MatrixRun(
            suite=suite,
            configuration=configuration,
            model=model,
            split_seed=int(split_seed),
            model_seed=int(model_seed),
            run_id=content_addressed_run_id(
                configuration=configuration,
                model=model,
                split_seed=int(split_seed),
                model_seed=int(model_seed),
                config_sha256=config_sha256,
                split_sha256=split_sha256,
                code_commit=code_commit,
            ),
        )
        for split_seed in split_seeds
        for model_seed in model_seeds
        for model in models
    ]


def build_unified_matrix(
    *,
    models: Iterable[str],
    model_seeds: Iterable[int],
    split_seeds: Iterable[int],
    **identity: str,
) -> list[MatrixRun]:
    return _runs(
        suite="mandatory",
        configuration="unified",
        models=models,
        model_seeds=model_seeds,
        split_seeds=split_seeds,
        **identity,
    )


def build_native_matrix(
    *,
    models: Iterable[str],
    model_seeds: Iterable[int],
    split_seed: int,
    **identity: str,
) -> list[MatrixRun]:
    return _runs(
        suite="mandatory",
        configuration="native",
        models=models,
        model_seeds=model_seeds,
        split_seeds=[split_seed],
        **identity,
    )


def as_records(runs: Iterable[MatrixRun]) -> list[dict[str, object]]:
    return [asdict(run) for run in runs]
