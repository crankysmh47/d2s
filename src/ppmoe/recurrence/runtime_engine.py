"""Heavyweight CUDA execution engine for recurrence production stages.

Only :mod:`ppmoe.recurrence.production` imports this module, and only after an
orchestrated stage has passed provenance/resource prerequisites.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import TYPE_CHECKING

import torch

from .data import Example, RenderedExample, load_retention_chunks, render_example
from .evaluate import evaluate_examples, evaluate_retention, generate_answer
from .gates import (
    CudaCostLedger,
    classify_informative_families,
    confirmation_decision,
    derive_a3_steps,
    discovery_decision,
    measure_training_arms,
    paired_accuracy_bootstrap,
    paired_retention_bootstrap,
    summarize_exact_scores,
)
from .lora import install_adapters
from .model import RecurrentQwen3
from .provenance import sha256_file, sha256_payload, write_immutable_json
from .train import (
    _make_adapter_optimizer,
    capture_initialization,
    load_checkpoint,
    load_initialization,
    train_arm,
    write_completed_training_manifest,
)

if TYPE_CHECKING:
    from .experiment import StageContext


_ARMS = ("a1", "a2", "a3")
_MODES = {"a1": "single", "a2": "repeat", "a3": "single"}


def _json(path: Path) -> dict[str, object]:
    from .experiment import _json_canonical, _read_test_bytes_once

    return _json_canonical(path, _read_test_bytes_once(path, trusted_root=path.parents[1]))


def _source(context: "StageContext") -> tuple[Path, str]:
    from .experiment import _source_snapshot_path, _stage_path

    record = _json(_stage_path(context.run_root, "source"))
    payload = record.get("payload")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("revision"), str):
        raise RuntimeError("source stage lacks pinned production model evidence")
    return _source_snapshot_path(context.run_root, payload), str(payload["revision"])


def _load_stock(context: "StageContext") -> tuple[torch.nn.Module, object]:
    from .experiment import _lazy_transformers_loader

    if not torch.cuda.is_available():
        raise RuntimeError("registered production runtime requires CUDA")
    snapshot, revision = _source(context)
    tokenizer_cls, model_cls = _lazy_transformers_loader()
    tokenizer = tokenizer_cls.from_pretrained(
        str(snapshot), revision=revision, local_files_only=True
    )
    stock = model_cls.from_pretrained(
        str(snapshot), revision=revision, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
    )
    stock.to(torch.device("cuda:0"))
    stock.config.use_cache = False
    return stock, tokenizer


def _rows(context: "StageContext", split: str, tokenizer: object) -> tuple[RenderedExample, ...]:
    from .experiment import _read_test_bytes_once

    if split == "test":
        raise RuntimeError("sealed test rows must be supplied as authorized bytes")
    raw = _read_test_bytes_once(
        context.run_root / "data" / f"{split}.jsonl", trusted_root=context.run_root
    )
    return _render_rows(raw, split, tokenizer)


def _render_rows(raw: bytes, split: str, tokenizer: object) -> tuple[RenderedExample, ...]:
    values: list[RenderedExample] = []
    for line in raw.decode("utf-8").splitlines():
        item = json.loads(line)
        if not isinstance(item, Mapping) or set(item) != {
            "id", "family", "split", "structure_id", "prompt", "answer", "latent_spec"
        } or item.get("split") != split:
            raise ValueError(f"frozen {split} row is malformed")
        example = Example(
            id=str(item["id"]), family=str(item["family"]), split=str(item["split"]),
            structure_id=str(item["structure_id"]), prompt=str(item["prompt"]),
            answer=str(item["answer"]), latent_spec=dict(item["latent_spec"]),
        )
        values.append(render_example(tokenizer, example))
    if not values or len({row.example.id for row in values}) != len(values):
        raise ValueError(f"frozen {split} rows are empty or duplicated")
    return tuple(values)


def _signatures(context: "StageContext", arm: str, seed: int, initialization: str) -> dict[str, str]:
    return {
        **dict(context.signatures), "arm": arm, "seed": str(seed),
        "initialization_hash": initialization,
    }


def _ensure_initialization(context: "StageContext", seed: int) -> tuple[str, str]:
    path = context.run_root / "initializations" / f"{seed}.pt"
    stock, _ = _load_stock(context)
    wrapper = RecurrentQwen3(stock, context.config.loop.layer_start, context.config.loop.layer_stop)
    install_adapters(
        wrapper.stock_model,
        context.config.loop.layer_start,
        context.config.loop.layer_stop,
        seed,
    )
    initialization = capture_initialization(path, wrapper, seed)
    digest = sha256_file(path)
    del wrapper, stock
    torch.cuda.empty_cache()
    return initialization, digest


def _adapted(context: "StageContext", arm: str, seed: int) -> tuple[RecurrentQwen3, object, str]:
    if context.stage in {"discover", "confirm"}:
        from .production import load_passed_pilot
        load_passed_pilot(context)
    stock, tokenizer = _load_stock(context)
    wrapper = RecurrentQwen3(stock, context.config.loop.layer_start, context.config.loop.layer_stop)
    wrapper.tokenizer = tokenizer
    install_adapters(
        wrapper.stock_model,
        context.config.loop.layer_start,
        context.config.loop.layer_stop,
        seed,
    )
    initialization = load_initialization(
        context.run_root / "initializations" / f"{seed}.pt", wrapper, expected_seed=seed
    )
    return wrapper, tokenizer, initialization


def _unadapted(context: "StageContext") -> tuple[RecurrentQwen3, object]:
    stock, tokenizer = _load_stock(context)
    wrapper = RecurrentQwen3(stock, context.config.loop.layer_start, context.config.loop.layer_stop)
    wrapper.tokenizer = tokenizer
    return wrapper, tokenizer


def _vram() -> dict[str, int]:
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "total_vram_bytes": int(torch.cuda.get_device_properties(0).total_memory),
    }


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _validated_records(
    context: "StageContext", path: Path, *, arm: str, seed: int, split: str
) -> list[dict[str, object]]:
    """Load a completed evaluator vector only through its signed summary."""
    summary_path = path.with_name(f"{path.stem}.summary.json")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = _records(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"validated {arm} seed-{seed} evidence is missing") from error
    if (
        not isinstance(summary, Mapping)
        or summary.get("schema_version") != 1
        or summary.get("complete") is not True
        or summary.get("arm") != arm.upper()
        or summary.get("seed") != seed
        or summary.get("split") != split
        or summary.get("signatures") != dict(context.signatures)
        or summary.get("record_count") != len(rows)
        or summary.get("expected_count") != len(rows)
        or summary.get("records_sha256") != sha256_file(path)
        or any(row.get("split") != split for row in rows)
    ):
        raise RuntimeError(f"validated {arm} seed-{seed} evidence does not reconcile")
    return rows


def _charge_once(context: "StageContext", attempt_id: str, seconds: float) -> None:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("chargeable CUDA interval is invalid")
    path = context.run_root / "cuda-cost.jsonl"
    ledger = (
        CudaCostLedger.load(path, context.config.resources.maximum_cuda_seconds)
        if path.exists() else CudaCostLedger(path, context.config.resources.maximum_cuda_seconds)
    )
    existing = next((entry for entry in ledger.entries if entry.attempt_id == attempt_id), None)
    if existing is not None:
        if existing.stage != context.stage or not math.isclose(existing.elapsed_seconds, seconds, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError("CUDA ledger replay does not match durable stage evidence")
        return
    ledger.record(attempt_id, context.stage, seconds, "passed")


def _arm_preflight(context: "StageContext") -> None:
    from .experiment import _preflight_resources, default_dependencies

    _preflight_resources(context, default_dependencies())


def _hide_diagnostic_summary(result: Mapping[str, object]) -> None:
    """Keep diagnostic evidence while excluding it from registered aggregates."""
    value = result.get("summary_path")
    if not isinstance(value, str):
        return
    path = Path(value)
    destination = path.with_suffix(".evidence")
    if destination.exists():
        if destination.read_bytes() != path.read_bytes():
            raise RuntimeError("diagnostic summary replay mismatch")
        path.unlink()
    else:
        path.replace(destination)


def _persist_accuracy_bootstrap(
    root: Path,
    result: Mapping[str, object],
    records: Mapping[tuple[str, int], Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    comparisons = result.get("comparisons")
    if not isinstance(comparisons, Mapping) or set(comparisons) != {"A2-F0", "A2-A1", "A2-A3"}:
        raise ValueError("accuracy bootstrap comparisons are incomplete")
    reference = next(iter(comparisons.values()))
    if not isinstance(reference, Mapping) or not isinstance(reference.get("sample_indices"), Mapping):
        raise ValueError("accuracy bootstrap sample matrix is missing")
    first_rows = next(iter(records.values()))
    families: dict[str, list[str]] = {}
    for row in first_rows:
        families.setdefault(str(row["family"]), []).append(str(row["example_id"]))
    families = {name: sorted(values) for name, values in sorted(families.items())}
    matrix_payload = {
        "families": {
            family: {
                "ids": identifiers,
                "indices": reference["sample_indices"][family],
            }
            for family, identifiers in families.items()
        }
    }
    matrix_path = root / "bootstrap" / "9047-indices.json"
    matrix_hash = write_immutable_json(matrix_path, matrix_payload)
    compact: dict[str, object] = {}
    vector_hashes: dict[str, str] = {}
    for name, item in comparisons.items():
        if not isinstance(item, Mapping) or not isinstance(item.get("bootstrap"), list):
            raise ValueError("accuracy bootstrap vector is missing")
        vector_path = root / "bootstrap" / f"accuracy-{name}.json"
        vector_hash = write_immutable_json(
            vector_path, {"comparison": name, "values": item["bootstrap"]}
        )
        vector_hashes[name] = vector_hash
        compact[name] = {
            "vector_path": vector_path.relative_to(root).as_posix(),
            "vector_sha256": vector_hash,
            "matrix_sha256": matrix_hash,
            "evaluation_seeds": [47, 48],
            "point": item["point"], "ci_low": item["ci_low"], "ci_high": item["ci_high"],
            "passed": float(item["point"]) >= 0.02 and float(item["ci_low"]) > 0.0,
        }
    return {
        "kind": "accuracy_bootstrap", "seed": 9047, "replicates": 10000,
        "bootstrap_sha256": sha256_payload(vector_hashes),
        "paired_sample_indices_path": matrix_path.relative_to(root).as_posix(),
        "paired_sample_indices_sha256": matrix_hash,
        "comparisons": compact,
    }


def _persist_retention_bootstrap(
    root: Path,
    results: Mapping[int, Mapping[str, object]],
    chunk_ids: Sequence[int],
) -> dict[str, object]:
    if set(results) != {47, 48}:
        raise ValueError("retention bootstrap requires both registered seeds")
    reference = results[47]
    sample_indices = reference.get("sample_indices")
    if not isinstance(sample_indices, Mapping) or set(sample_indices) != {"retention"}:
        raise ValueError("retention bootstrap matrix is missing")
    if results[48].get("sample_indices") != sample_indices:
        raise ValueError("retention seeds must share the registered paired matrix")
    matrix_path = root / "bootstrap" / "9157-indices.json"
    matrix_hash = write_immutable_json(matrix_path, {
        "families": {"retention": {"ids": [str(index) for index in chunk_ids], "indices": sample_indices["retention"]}}
    })
    compact: dict[str, object] = {}
    vector_hashes: dict[str, str] = {}
    for seed, item in sorted(results.items()):
        name = f"A2-F0@{seed}"
        vector_path = root / "bootstrap" / f"retention-{seed}.json"
        vector_hash = write_immutable_json(
            vector_path, {"comparison": name, "seed": seed, "values": item["bootstrap"]}
        )
        vector_hashes[name] = vector_hash
        compact[name] = {
            "seed": seed, "evaluation_seeds": [seed],
            "vector_path": vector_path.relative_to(root).as_posix(),
            "vector_sha256": vector_hash, "matrix_sha256": matrix_hash,
            "point": item["point"], "ci_low": item["ci_low"], "ci_high": item["ci_high"],
            "passed": float(item["point"]) <= 1.02 and float(item["ci_high"]) <= 1.02,
        }
    return {
        "kind": "retention_bootstrap", "seed": 9157, "replicates": 10000,
        "bootstrap_sha256": sha256_payload(vector_hashes),
        "paired_sample_indices_path": matrix_path.relative_to(root).as_posix(),
        "paired_sample_indices_sha256": matrix_hash,
        "comparisons": compact,
    }


def _slim_bootstrap(value: object) -> object:
    """Remove persisted vector/matrix bodies from stage JSON payloads."""
    if isinstance(value, Mapping):
        return {
            str(key): _slim_bootstrap(item)
            for key, item in value.items()
            if key not in {"bootstrap", "sample_indices"}
        }
    if isinstance(value, list):
        return [_slim_bootstrap(item) for item in value]
    return value


def _score(
    context: "StageContext", model: RecurrentQwen3, rows: Sequence[RenderedExample],
    chunks: Sequence[object], *, arm: str, seed: int, mode: str, prefix: str,
    split: str = "dev", output_root: Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    root = output_root or context.run_root
    score_path = root / "scores" / f"{prefix}.jsonl"
    retention_path = root / "retention-scores" / f"{prefix}.jsonl"
    exact = evaluate_examples(
        model, rows, mode, score_path, arm=arm.upper(), seed=seed,
        signatures=context.signatures,
        eos_token_id=int(getattr(model.stock_model.config, "eos_token_id")),
        max_new_tokens=context.config.decoding.max_new_tokens,
    )
    retention = evaluate_retention(
        model, chunks, mode, retention_path, arm=arm.upper(), seed=seed,
        signatures=context.signatures, split=split,
    )
    if exact.get("complete") is not True or retention.get("complete") is not True:
        raise RuntimeError(f"{arm.upper()} seed {seed} evaluation is incomplete")
    elapsed = sum(float(row["latency_seconds"]) for row in _records(Path(str(exact["records_path"]))))
    elapsed += sum(float(row["latency_seconds"]) for row in _records(Path(str(retention["records_path"]))))
    _charge_once(context, f"{context.stage}:{prefix}:evaluation", elapsed)
    return exact, retention


def _train(context: "StageContext", arm: str, seed: int, steps: int) -> dict[str, object]:
    manifest_path = context.run_root / "training" / f"{arm}-{seed}.json"
    if manifest_path.is_file():
        from .production import _load_training_manifest, load_passed_pilot
        pilot = load_passed_pilot(context)
        value = dict(_load_training_manifest(context.run_root, arm, seed, int(pilot["n3"])))
        checkpoint = context.run_root / "checkpoints" / arm / str(seed) / "latest.pt"
        if (
            value.get("signatures") != dict(context.signatures)
            or value.get("initialization_hash") != pilot["initialization_hashes"][str(seed)]
            or not checkpoint.is_file()
            or sha256_file(checkpoint) != value.get("checkpoint_sha256")
        ):
            raise RuntimeError(f"completed training manifest is not bound for {arm} seed {seed}")
        return value
    wrapper, tokenizer, initialization = _adapted(context, arm, seed)
    train_rows = _rows(context, "train", tokenizer)
    signatures = _signatures(context, arm, seed, initialization)
    result = train_arm(
        wrapper, train_rows, arm, seed, steps,
        context.run_root / "checkpoints" / arm / str(seed),
        context.run_root / "events.jsonl", signatures, stage=context.stage,
    )
    latest = context.run_root / "checkpoints" / arm / str(seed) / "latest.pt"
    value = {
        "schema_version": 1, "status": "complete", "arm": arm, "seed": seed,
        "signatures": dict(context.signatures), "checkpoint_sha256": sha256_file(latest),
        **result,
    }
    write_completed_training_manifest(manifest_path, value)
    _charge_once(
        context, f"{context.stage}:{arm}:{seed}:training",
        float(value["synchronized_cuda_seconds"]),
    )
    del wrapper
    torch.cuda.empty_cache()
    return value


def _pilot(context: "StageContext") -> Mapping[str, object]:
    hashes: dict[str, str] = {}
    identities: dict[str, str] = {}
    for seed in (47, 48):
        identities[str(seed)], hashes[str(seed)] = _ensure_initialization(context, seed)
    wrapper_a1, tokenizer, initialization = _adapted(context, "a1", 47)
    wrapper_a2, _, initialization_a2 = _adapted(context, "a2", 47)
    if initialization_a2 != initialization:
        raise RuntimeError("pilot arms do not share the seed-47 initialization")
    train_rows = _rows(context, "train", tokenizer)
    first = train_rows[0]
    device = next(wrapper_a1.parameters()).device
    common = {
        "input_ids": torch.tensor([first.train_input_ids], dtype=torch.long, device=device),
        "labels": torch.tensor([first.labels], dtype=torch.long, device=device),
    }
    pilot_started = time.perf_counter()
    timings = measure_training_arms({
        "a1": (wrapper_a1, {**common, "mode": "single"}, _make_adapter_optimizer(wrapper_a1)),
        "a2": (wrapper_a2, {**common, "mode": "repeat"}, _make_adapter_optimizer(wrapper_a2)),
    })
    torch.cuda.synchronize()
    _charge_once(context, "pilot:training-timing", time.perf_counter() - pilot_started)
    n3 = derive_a3_steps(timings["a1"], timings["a2"])
    del wrapper_a1, wrapper_a2
    torch.cuda.empty_cache()

    disposable = context.run_root / "pilot-disposable"
    disposable.mkdir(parents=True, exist_ok=True)
    evidence_root = context.run_root / "pilot-evidence"
    frozen, tokenizer = _unadapted(context)
    dev = _rows(context, "dev", tokenizer)
    sample = tuple(row for family in ("expressions", "lists", "graphs") for row in [item for item in dev if item.example.family == family][:4])
    longest = max(dev, key=lambda row: len(row.prompt_input_ids))
    chunks = load_retention_chunks(context.run_root / "retention")
    generation_times: list[float] = []
    retention_times: list[float] = []
    for arm, mode in (("f0", "single"), ("f1", "repeat"), ("f2", "damped")):
        exact = evaluate_examples(
            frozen, sample, mode, evidence_root / "scores" / f"pilot-{arm}.jsonl",
            arm=arm.upper(), seed=47, signatures=context.signatures,
            eos_token_id=int(getattr(frozen.stock_model.config, "eos_token_id")),
            max_new_tokens=context.config.decoding.max_new_tokens,
        )
        if exact.get("complete") is not True:
            raise RuntimeError(f"pilot {arm.upper()} generation is incomplete")
        generation_times.extend(float(row["latency_seconds"]) for row in _records(Path(str(exact["records_path"]))))
        _charge_once(
            context, f"pilot:{arm}:generation",
            sum(float(row["latency_seconds"]) for row in _records(Path(str(exact["records_path"])))),
        )
        _hide_diagnostic_summary(exact)
    forced = generate_answer(
        frozen,
        torch.tensor([longest.prompt_input_ids], dtype=torch.long, device=next(frozen.parameters()).device),
        "single",
        int(getattr(frozen.stock_model.config, "eos_token_id")),
        32,
        stop_on_eos=False,
    )
    forced_seconds = float(forced["latency_seconds"])
    if forced.get("generated_token_count") != 32:
        raise RuntimeError("pessimistic generation timing did not complete 32 tokens")
    _charge_once(context, "pilot:longest-prompt-forced-32", forced_seconds)
    pilot_retention = evaluate_retention(
        frozen, chunks[:4], "single",
        evidence_root / "retention-scores" / "pilot-f0.jsonl",
        arm="F0", seed=47, signatures=context.signatures, split="pilot",
    )
    if pilot_retention.get("complete") is not True:
        raise RuntimeError("pilot retention measurement is incomplete")
    retention_times.extend(
        float(row["latency_seconds"])
        for row in _records(Path(str(pilot_retention["records_path"])))
    )
    _charge_once(context, "pilot:f0:retention", sum(retention_times))
    _hide_diagnostic_summary(pilot_retention)
    (disposable / "DISPOSABLE").write_text("pilot-only\n", encoding="utf-8")
    payload = {
        "schema_version": 1, "n3": n3,
        "training_samples": {name: list(values) for name, values in timings.items()},
        "seconds_per_training_update": {
            "a1": statistics.median(timings["a1"]), "a2": statistics.median(timings["a2"]),
        },
        "seconds_per_generation_example": max([forced_seconds, *generation_times]),
        "forced_generation_seconds": forced_seconds,
        "seconds_per_retention_chunk": max(retention_times),
        "longest_prompt_tokens": len(longest.prompt_input_ids),
        "generation_record_count": 36, "retention_measurement_count": 4,
        "disposable_relative_path": "pilot-disposable",
        "initialization_hash": initialization, "initialization_hashes": identities,
        "initialization_sha256": hashes,
        **_vram(),
    }
    write_immutable_json(context.run_root / "pilot.json", payload)
    return payload


def _frozen_diagnostic(
    candidate: Sequence[Mapping[str, object]],
    baseline: Sequence[Mapping[str, object]],
    candidate_retention: Sequence[Mapping[str, object]],
    baseline_retention: Sequence[Mapping[str, object]],
    memory: Mapping[str, int],
) -> dict[str, object]:
    """Build the registered non-selective G3 comparison against F0."""
    candidate_summary = summarize_exact_scores(candidate, baseline)
    baseline_summary = summarize_exact_scores(baseline)
    candidate_families = candidate_summary["families"]
    baseline_families = baseline_summary["families"]
    assert isinstance(candidate_families, Mapping)
    assert isinstance(baseline_families, Mapping)
    retention = paired_retention_bootstrap(candidate_retention, baseline_retention)
    return {
        "macro_accuracy_delta": float(candidate_summary["macro_accuracy"])
        - float(baseline_summary["macro_accuracy"]),
        "family_accuracy_deltas": {
            family: float(candidate_families[family]["accuracy"])
            - float(baseline_families[family]["accuracy"])
            for family in sorted(baseline_families)
        },
        "global_invalid_rate_delta": float(candidate_summary["global_invalid_rate"])
        - float(baseline_summary["global_invalid_rate"]),
        "family_invalid_rate_deltas": {
            family: float(candidate_families[family]["invalid_rate"])
            - float(baseline_families[family]["invalid_rate"])
            for family in sorted(baseline_families)
        },
        "transitions": candidate_summary["transitions"],
        "retention_ratio": _slim_bootstrap(retention),
        "latency_seconds": {
            "generation": sum(float(row["latency_seconds"]) for row in candidate),
            "retention": sum(float(row["latency_seconds"]) for row in candidate_retention),
        },
        "memory": dict(memory),
    }


def _frozen(context: "StageContext") -> Mapping[str, object]:
    model, tokenizer = _unadapted(context)
    dev = _rows(context, "dev", tokenizer)
    chunks = load_retention_chunks(context.run_root / "retention")
    summaries: dict[str, object] = {}
    exact_rows: dict[str, list[dict[str, object]]] = {}
    retention_rows: dict[str, list[dict[str, object]]] = {}
    memory: dict[str, dict[str, int]] = {}
    for arm, mode in (("f0", "single"), ("f1", "repeat"), ("f2", "damped")):
        exact, retention = _score(context, model, dev, chunks, arm=arm, seed=47, mode=mode, prefix=f"frozen-{arm}")
        summaries[arm] = {
            "exact": summarize_exact_scores(_records(Path(str(exact["records_path"])))),
            "retention": retention,
        }
        exact_rows[arm] = _records(Path(str(exact["records_path"])))
        retention_rows[arm] = _records(Path(str(retention["records_path"])))
        memory[arm] = _vram()
    classification = classify_informative_families(summaries["f0"]["exact"])
    informative = classification["passes"] is True
    decision = {
        "status": "passed" if informative else "stopped_scientific",
        "reason": "informative" if informative else "uninformative_benchmark",
        "passed": informative,
        "gates": classification,
    }
    diagnostics = {
        arm: _frozen_diagnostic(
            exact_rows[arm], exact_rows["f0"], retention_rows[arm],
            retention_rows["f0"], memory[arm],
        )
        for arm in ("f1", "f2")
    }
    return {
        "schema_version": 1, "decision": decision, "summaries": summaries,
        "diagnostics": diagnostics, **_vram(),
    }


def _pilot_payload(context: "StageContext") -> Mapping[str, object]:
    from .production import load_passed_pilot
    return load_passed_pilot(context)


def _discovery(context: "StageContext") -> Mapping[str, object]:
    pilot = _pilot_payload(context)
    n3 = int(pilot["n3"])
    training: dict[str, dict[str, object]] = {}
    for arm in _ARMS:
        _arm_preflight(context)
        training[arm] = _train(context, arm, 47, n3 if arm == "a3" else 1000)
    records: dict[tuple[str, int], list[dict[str, object]]] = {}
    retention_rows: dict[str, list[dict[str, object]]] = {}
    summaries: dict[str, object] = {}
    for arm in _ARMS:
        model, tokenizer, _ = _adapted(context, arm, 47)
        optimizer = _make_adapter_optimizer(model)
        load_checkpoint(context.run_root / "checkpoints" / arm / "47", model, optimizer, _signatures(context, arm, 47, str(training[arm]["initialization_hash"])))
        exact, retention = _score(context, model, _rows(context, "dev", tokenizer), load_retention_chunks(context.run_root / "retention"), arm=arm, seed=47, mode=_MODES[arm], prefix=f"{arm}-47-dev")
        records[(arm.upper(), 47)] = _records(Path(str(exact["records_path"])))
        retention_rows[arm] = _records(Path(str(retention["records_path"])))
        summaries[arm] = summarize_exact_scores(records[(arm.upper(), 47)])
        del model
        torch.cuda.empty_cache()
    f0 = _validated_records(
        context, context.run_root / "scores" / "frozen-f0.jsonl",
        arm="f0", seed=47, split="dev",
    )
    f0_retention = _validated_records(
        context, context.run_root / "retention-scores" / "frozen-f0.jsonl",
        arm="f0", seed=47, split="dev",
    )
    records[("F0", 47)] = f0
    accuracy = paired_accuracy_bootstrap(records)
    retention_bootstrap = paired_retention_bootstrap(retention_rows["a2"], f0_retention)
    ratio = float(training["a3"]["synchronized_cuda_seconds"]) / float(training["a2"]["synchronized_cuda_seconds"])
    timing = {
        "a3_a2_realized_training_time_ratio": ratio,
        "seed": 47, "sample_count": 1,
        "metadata": {"units": "seconds", "synchronized": True, "measurement": "complete_production_training", "arms": ["A2", "A3"]},
        "a2_realized_training_time_seconds": training["a2"]["synchronized_cuda_seconds"],
        "a3_realized_training_time_seconds": training["a3"]["synchronized_cuda_seconds"],
    }
    decision = discovery_decision(accuracy, retention_bootstrap, timing)
    payload = {
        "schema_version": 1, "training": training, "summaries": summaries,
        "accuracy": _slim_bootstrap(accuracy), "retention": _slim_bootstrap(retention_bootstrap),
        "timing": timing, "decision": decision,
        "selection_sha256": None, "data_manifest_sha256": None, **_vram(),
    }
    if decision["status"] == "passed":
        data_manifest = json.loads((context.run_root / "data" / "manifest.json").read_text(encoding="utf-8"))
        selection = {
            "schema_version": 1, "status": "passed", "selected_condition": "A2",
            "signatures": dict(context.signatures),
            "test_sha256": str(data_manifest["split_sha256"]["test"]), "n3": n3,
            "dev_score_hashes": {
                "F0": sha256_file(context.run_root / "scores" / "frozen-f0.jsonl"),
                **{arm.upper(): sha256_file(context.run_root / "scores" / f"{arm}-47-dev.jsonl") for arm in _ARMS},
            },
            "data_manifest_sha256": sha256_file(context.run_root / "data" / "manifest.json"),
        }
        payload["selection_sha256"] = write_immutable_json(
            context.run_root / "selection.json", selection
        )
        payload["data_manifest_sha256"] = selection["data_manifest_sha256"]
    return payload


def _confirmation(context: "StageContext") -> Mapping[str, object]:
    pilot = _pilot_payload(context)
    n3 = int(pilot["n3"])
    training: dict[str, dict[str, object]] = {}
    for arm in _ARMS:
        _arm_preflight(context)
        training[arm] = _train(context, arm, 48, n3 if arm == "a3" else 1000)
    # Authorization is deliberately repeated after training and immediately
    # before parsing sealed bytes; this rejects checkpoint/selection mutation.
    from .production import authorize_confirmation_test_read
    sealed_test_bytes = authorize_confirmation_test_read(context)
    all_records: dict[tuple[str, int], list[dict[str, object]]] = {}
    retention: dict[int, object] = {}
    retention_rows: dict[tuple[str, int], list[dict[str, object]]] = {}
    retention_summaries: dict[str, object] = {}
    timing: dict[int, object] = {}
    for seed in (47, 48):
        for arm in _ARMS:
            model, tokenizer, initialization = _adapted(context, arm, seed)
            optimizer = _make_adapter_optimizer(model)
            load_checkpoint(context.run_root / "checkpoints" / arm / str(seed), model, optimizer, _signatures(context, arm, seed, initialization))
            test_rows = _render_rows(sealed_test_bytes, "test", tokenizer)
            exact = evaluate_examples(model, test_rows, _MODES[arm], context.run_root / "scores" / f"{arm}-{seed}-test.jsonl", arm=arm.upper(), seed=seed, signatures=context.signatures, eos_token_id=int(getattr(model.stock_model.config, "eos_token_id")), max_new_tokens=context.config.decoding.max_new_tokens)
            if exact.get("complete") is not True:
                raise RuntimeError(f"{arm.upper()} seed {seed} sealed evaluation is incomplete")
            all_records[(arm.upper(), seed)] = _records(Path(str(exact["records_path"])))
            _charge_once(
                context, f"confirm:{arm}:{seed}:test-generation",
                sum(float(row["latency_seconds"]) for row in all_records[(arm.upper(), seed)]),
            )
            if seed == 48:
                r = evaluate_retention(model, load_retention_chunks(context.run_root / "retention"), _MODES[arm], context.run_root / "retention-scores" / f"{arm}-48-confirm.jsonl", arm=arm.upper(), seed=48, signatures=context.signatures, split="test")
                if r.get("complete") is not True:
                    raise RuntimeError(f"{arm.upper()} seed 48 retention evaluation is incomplete")
                retention_rows[(arm.upper(), 48)] = _records(Path(str(r["records_path"])))
                retention_summaries[f"{arm.upper()}@48"] = r
                _charge_once(
                    context, f"confirm:{arm}:48:retention",
                    sum(float(row["latency_seconds"]) for row in retention_rows[(arm.upper(), 48)]),
                )
            del model
            torch.cuda.empty_cache()
        a2_time = float(json.loads((context.run_root / "training" / f"a2-{seed}.json").read_text())["synchronized_cuda_seconds"])
        a3_time = float(json.loads((context.run_root / "training" / f"a3-{seed}.json").read_text())["synchronized_cuda_seconds"])
        timing[seed] = {
            "seed": seed, "sample_count": 1,
            "metadata": {"units": "seconds", "synchronized": True, "measurement": "complete_production_training", "arms": ["A2", "A3"]},
            "a2_realized_training_time_seconds": a2_time,
            "a3_realized_training_time_seconds": a3_time,
            "a3_a2_realized_training_time_ratio": a3_time / a2_time,
        }
    # The deterministic F0 vector is produced once and reused for both paired
    # seed comparisons.  It must never be regenerated under a second seed.
    model, tokenizer = _unadapted(context)
    test_rows = _render_rows(sealed_test_bytes, "test", tokenizer)
    exact = evaluate_examples(model, test_rows, "single", context.run_root / "scores" / "f0-test.jsonl", arm="F0", seed=47, signatures=context.signatures, eos_token_id=int(getattr(model.stock_model.config, "eos_token_id")), max_new_tokens=context.config.decoding.max_new_tokens)
    f0_test_rows = _records(Path(str(exact["records_path"])))
    all_records[("F0", 47)] = f0_test_rows
    _charge_once(
        context, "confirm:f0:test-generation",
        sum(float(row["latency_seconds"]) for row in f0_test_rows),
    )
    del model
    torch.cuda.empty_cache()

    f0_retention_rows = _validated_records(
        context, context.run_root / "retention-scores" / "frozen-f0.jsonl",
        arm="f0", seed=47, split="dev",
    )
    for arm in _ARMS:
        retention_rows[(arm.upper(), 47)] = _validated_records(
            context,
            context.run_root / "retention-scores" / f"{arm}-47-dev.jsonl",
            arm=arm, seed=47, split="dev",
        )
    for seed in (47, 48):
        retention[seed] = paired_retention_bootstrap(
            retention_rows[("A2", seed)], f0_retention_rows
        )
    accuracy = paired_accuracy_bootstrap(all_records)
    per_seed = {}
    for seed in (47, 48):
        seed_bootstrap = paired_accuracy_bootstrap({
            key: rows for key, rows in all_records.items()
            if key[1] == seed or key[0] == "F0"
        })
        per_seed[seed] = {
            **seed_bootstrap,
            **{f"a2_family_deltas_vs_{ref}": seed_bootstrap["comparisons"][f"A2-{ref.upper()}"]["family_deltas"] for ref in ("f0", "a1")},
        }
    accuracy["per_seed"] = per_seed
    accuracy["accuracy_ci_lows"] = {name: value["ci_low"] for name, value in accuracy["comparisons"].items()}
    decision = confirmation_decision(accuracy, retention, timing)
    accuracy_bootstrap = _persist_accuracy_bootstrap(context.run_root, accuracy, all_records)
    retention_bootstrap = _persist_retention_bootstrap(
        context.run_root, retention,
        [int(row["chunk_index"]) for row in retention_rows[("A2", 47)]],
    )
    return {
        "schema_version": 1, "training": training, "accuracy": _slim_bootstrap(accuracy),
        "retention": _slim_bootstrap(retention), "timing": timing, "decision": decision,
        "retention_summaries": retention_summaries,
        "accuracy_bootstrap": accuracy_bootstrap,
        "retention_bootstrap": retention_bootstrap,
        **_vram(),
    }


def execute_stage(
    context: "StageContext", stage: str, *, sealed_test_bytes: bytes | None = None
) -> Mapping[str, object]:
    """Execute exactly one registered production stage."""
    started = time.perf_counter()
    try:
        if stage == "pilot":
            return _pilot(context)
        if stage == "frozen":
            return _frozen(context)
        if stage == "discover":
            return _discovery(context)
        if stage == "confirm":
            if sealed_test_bytes is not None:
                raise RuntimeError("sealed bytes may not be injected before confirmation training")
            return _confirmation(context)
        raise ValueError(f"unsupported production runtime stage: {stage}")
    except Exception as error:
        from .train import FreshProcessRequired, _is_cuda_process_fault
        if isinstance(error, FreshProcessRequired):
            # Task 9 already persisted the exact attempt-local CUDA interval,
            # checkpoint identity, and retry status for training faults.
            raise
        if _is_cuda_process_fault(error):
            from .provenance import append_event
            diagnostics = {
                "message": str(error), "error_type": type(error).__name__,
                "cuda_process_fault": True,
                "chargeable_cuda_seconds": max(0.0, time.perf_counter() - started),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)) if torch.cuda.is_available() else 0,
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)) if torch.cuda.is_available() else 0,
            }
            append_event(context.run_root / "events.jsonl", {
                "kind": "stage_attempt", "stage": stage, "attempt": context.attempt,
                "payload": diagnostics, "outcome": "fresh_process_required",
            })
            raise FreshProcessRequired(str(error)) from error
        raise


def execute_child_stage(request: Mapping[str, object], stage: str) -> Mapping[str, object]:
    """Reconstruct the authenticated fresh process and emit bound retry claims."""
    from .config import RecurrenceConfig
    from .experiment import StageContext

    if request.get("stage") != stage or request.get("attempt") != 2:
        raise RuntimeError("child request stage/attempt mismatch")
    config_path = Path(str(request["config_path"])).resolve()
    signatures = request.get("signatures")
    if not isinstance(signatures, Mapping):
        raise RuntimeError("child request signatures are invalid")
    context = StageContext(
        stage, RecurrenceConfig.load(config_path), config_path,
        Path(str(request["artifact_root"])).resolve(), Path(str(request["run_root"])).resolve(),
        dict(signatures), 2, lambda path: path.read_bytes(),
    )
    payload = dict(execute_stage(context, stage))
    from .production import validate_stage_payload
    validate_stage_payload(stage, payload, context)
    payload.update({
        "worker_pid": os.getpid(), "attempt": request["attempt"],
        "checkpoint_path": request["checkpoint_path"],
        "checkpoint_digest": request["checkpoint_digest"],
        "checkpoint_sha256": request["checkpoint_sha256"],
        "checkpoint_attempt": request["checkpoint_attempt"],
        "checkpoint_update": request["checkpoint_update"],
        "checkpoint_event_id": request["checkpoint_event_id"],
        "checkpoint_event_index": request["checkpoint_event_index"],
        "checkpoint_arm": request["checkpoint_arm"],
        "checkpoint_seed": request["checkpoint_seed"],
        "checkpoint_relative_path": request["checkpoint_relative_path"],
        "checkpoint_signature": request["checkpoint_signature"],
        "dispatcher_stage": request["stage"], "dispatcher_ref": request["worker_ref"],
        "dispatcher_identity": request["worker_identity"],
        "checkpoint_proof": request["checkpoint_proof"],
    })
    return payload
