"""Pinned Hugging Face dataset loading boundaries."""

from collections.abc import Callable
from typing import Any

from datasets import load_dataset

from ppmoe.config import DatasetSpec


DatasetLoader = Callable[..., Any]


def _load(spec: DatasetSpec, loader: DatasetLoader) -> Any:
    return loader(
        spec.repo_id,
        name=spec.name,
        split=spec.split,
        revision=spec.revision,
        streaming=spec.streaming,
    )


def stream_fineweb(
    spec: DatasetSpec,
    *,
    loader: DatasetLoader = load_dataset,
) -> Any:
    if not spec.streaming:
        raise ValueError("FineWeb-Edu must use streaming=True")
    return _load(spec, loader)

def load_wikitext_test(
    spec: DatasetSpec,
    *,
    loader: DatasetLoader = load_dataset,
) -> Any:
    if spec.name != "wikitext-103-raw-v1" or spec.split != "test":
        raise ValueError("WikiText evaluation requires wikitext-103-raw-v1/test")
    if spec.streaming:
        raise ValueError("WikiText test must be materialized, not streamed")
    return _load(spec, loader)
