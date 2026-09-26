"""Deterministic V1 benchmark generation and exclusive split freezing."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re
from typing import Mapping, Sequence

from .config_v1 import V1Config
from .data import (
    Example,
    _compact_json,
    _example_payload,
    _expression_tree,
    _shortest_path_length,
    _thaw_json,
    interpret,
    render_example,
)
from .provenance import sha256_payload, validate_revision


FAMILIES = ("expressions", "lists", "graphs")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def make_pilot(config: V1Config, family: str, candidate: int, forbidden_ids: set[str]) -> tuple[Example, ...]:
    """Generate one candidate's pilot with its independent registered PRNG stream."""
    if family not in FAMILIES:
        raise ValueError(f"unknown benchmark family {family!r}")
    if isinstance(candidate, bool) or candidate not in (1, 2):
        raise ValueError("candidate must be 1 or 2")
    if not isinstance(forbidden_ids, set) or any(not isinstance(value, str) for value in forbidden_ids):
        raise TypeError("forbidden_ids must be a set of IDs")
    seed = config.data.pilot_seeds[family][candidate - 1]
    return _generate(config, family, candidate, "pilot", config.data.pilot_per_candidate, seed, set(forbidden_ids))


def make_selected_splits(
    config: V1Config, choices: Mapping[str, int], forbidden_ids: set[str]
) -> dict[str, tuple[Example, ...]]:
    """Generate independent development and sealed data at selected difficulties."""
    if set(choices) != set(FAMILIES):
        raise ValueError("choices must select one candidate for each family")
    for family, choice in choices.items():
        if isinstance(choice, bool) or choice not in (1, 2):
            raise ValueError(f"invalid candidate choice for {family}")
    if not isinstance(forbidden_ids, set) or any(not isinstance(value, str) for value in forbidden_ids):
        raise TypeError("forbidden_ids must be a set of IDs")
    seen = set(forbidden_ids)
    output: dict[str, tuple[Example, ...]] = {}
    for split, counts, seed_map in (
        ("dev", config.data.development_per_family, config.data.development_seeds),
        ("sealed", config.data.sealed_per_family, config.data.sealed_seeds),
    ):
        rows: list[Example] = []
        for family in FAMILIES:
            generated = _generate(config, family, choices[family], split, counts, seed_map[family], seen)
            rows.extend(generated)
            seen.update(row.id for row in generated)
        output[split] = tuple(rows)
    return output


def _generate(
    config: V1Config,
    family: str,
    candidate: int,
    split: str,
    count: int,
    seed: int,
    forbidden_ids: set[str],
) -> tuple[Example, ...]:
    rng = random.Random(seed)
    accepted: list[Example] = []
    seen = set(forbidden_ids)
    while len(accepted) < count:
        accepted_index = len(accepted)
        for _ in range(config.data.maximum_attempts_per_example):
            row = _candidate(config, family, candidate, split, accepted_index, rng)
            if row is None or row.id in seen:
                continue
            accepted.append(row)
            seen.add(row.id)
            break
        else:
            raise ValueError(
                f"could not accept {family}/{split} example {accepted_index} after "
                f"{config.data.maximum_attempts_per_example} attempts (forbidden or rejected candidates)"
            )
    return tuple(accepted)


def _candidate(
    config: V1Config, family: str, candidate: int, split: str, accepted_index: int, rng: random.Random
) -> Example | None:
    if family == "expressions":
        leaves_count = config.data.expression_leaf_counts[candidate - 1]
        shape = "chain" if accepted_index % 2 == 0 else "balanced"
        latent: dict[str, object] = {
            "shape": shape,
            "leaves": [rng.randrange(10) for _ in range(leaves_count)],
            "operators": [rng.choice(("+", "-", "*")) for _ in range(leaves_count - 1)],
        }
        answer_value = interpret(latent, family)
        if abs(answer_value) > config.data.maximum_absolute_expression_answer:
            return None
        expression = _expression_tree(latent["leaves"], latent["operators"], shape)[1]  # type: ignore[arg-type]
        structure_id = f"{shape}{leaves_count}"
        prompt = f"Compute {expression}. Return only the integer answer."
        answer = str(answer_value)
    elif family == "lists":
        item_count = config.data.list_item_counts[candidate - 1]
        operation_count = config.data.list_operation_counts[candidate - 1]
        latent = {
            "input": [rng.randrange(10) for _ in range(item_count)],
            "ops": [rng.choice(("ROTL1", "REVERSE", "ADD1")) for _ in range(operation_count)],
        }
        operations = latent["ops"]
        if any(pair == ("REVERSE", "REVERSE") for pair in zip(operations, operations[1:])):  # type: ignore[arg-type]
            return None
        answer_value = interpret(latent, family)
        if answer_value == latent["input"]:
            return None
        ops_text = ",".join(operations)  # type: ignore[arg-type]
        structure_id = f"n{item_count}:{ops_text}"
        prompt = (
            f"Start with {_compact_json(latent['input'])}. Apply {ops_text} from left to right. ROTL1 moves the first element "
            "to the end; REVERSE reverses the list; ADD1 adds 1 modulo 10 to each element. "
            "Return only the final list."
        )
        answer = _compact_json(answer_value)
    else:
        vertex_count = config.data.graph_vertex_counts[candidate - 1]
        vertices = [chr(ord("A") + index) for index in range(vertex_count)]
        edges = [
            f"{source}->{target}"
            for source in vertices
            for target in vertices
            if source != target and rng.random() < config.data.graph_edge_probability
        ]
        edges.sort()
        start, end = rng.sample(vertices, 2)
        latent = {"n": vertex_count, "edges": edges, "start": start, "end": end}
        answer_value = interpret(latent, family)
        wanted = "yes" if accepted_index % 2 == 0 else "no"
        if answer_value != wanted:
            return None
        distance = _shortest_path_length(vertex_count, edges, start, end)
        structure_id = f"n{vertex_count}:d{distance if distance is not None else 'unreachable'}"
        edges_text = ", ".join(edges) if edges else "none"
        prompt = (
            f"Directed edges: {edges_text}. Is there a directed path from {start} to {end}? "
            "Return only yes or no."
        )
        answer = answer_value
    identifier = sha256_payload({"family": family, "latent_spec": latent})
    return Example(identifier, family, split, structure_id, prompt, answer, latent)


def freeze_v1_split(
    path: Path,
    examples: Sequence[Example],
    tokenizer: object,
    *,
    revision: str,
    tokenizer_sha256: str,
) -> dict[str, object]:
    """Exclusively create a deterministic split JSONL and its provenance sidecar."""
    try:
        validate_revision(revision)
    except ValueError as error:
        raise ValueError("freeze requires an immutable tokenizer revision") from error
    if not isinstance(tokenizer_sha256, str) or _SHA256.fullmatch(tokenizer_sha256) is None:
        raise ValueError("freeze requires a full tokenizer SHA-256")
    if not examples:
        raise ValueError("split must contain at least one example")
    if isinstance(examples, (str, bytes)):
        raise ValueError("examples must be a sequence of Example records")
    if any(not isinstance(row, Example) for row in examples):
        raise ValueError("split contains a malformed row")
    splits = {row.split for row in examples}
    if len(splits) != 1:
        raise ValueError("split must be well formed and homogeneous")
    split = next(iter(splits))
    if split not in {"pilot", "dev", "sealed"}:
        raise ValueError("split must be pilot, dev, or sealed")
    identifiers: set[str] = set()
    rendered_lengths: list[tuple[int, int]] = []
    encoded_rows: list[bytes] = []
    for row in examples:
        if row.family not in FAMILIES:
            raise ValueError("split row has an unknown family")
        if row.id in identifiers:
            raise ValueError("split contains duplicate IDs")
        identifiers.add(row.id)
        expected_id = sha256_payload({"family": row.family, "latent_spec": _thaw_json(row.latent_spec)})
        if row.id != expected_id:
            raise ValueError("split row does not match its canonical ID")
        evaluated = interpret(row.latent_spec, row.family)
        expected_answer = evaluated if row.family == "graphs" else _compact_json(evaluated)
        if row.answer != expected_answer:
            raise ValueError("split row answer is inaccurate")
        if not row.prompt or not row.structure_id:
            raise ValueError("split row has a malformed prompt or structure ID")
        rendered = render_example(tokenizer, row)
        prompt_tokens = len(rendered.prompt_input_ids)
        if prompt_tokens + 256 > 512:
            raise ValueError("rendered prompt exceeds V1 prompt-plus-generation token budget")
        answer_tokens = len(rendered.train_input_ids) - prompt_tokens - 1
        rendered_lengths.append((prompt_tokens, prompt_tokens + answer_tokens))
        payload = json.dumps(_example_payload(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        encoded_rows.append(payload.encode("utf-8") + b"\n")
    raw = b"".join(encoded_rows)
    digest = sha256(raw).hexdigest()
    destination = Path(path)
    manifest_path = Path(str(destination) + ".manifest.json")
    if destination.is_symlink() or manifest_path.is_symlink():
        raise ValueError("split and manifest paths may not be symlinks")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "split": split,
        "count": len(examples),
        "sha256": digest,
        "ordered_ids": [row.id for row in examples],
        "tokenizer_revision": revision,
        "tokenizer_sha256": tokenizer_sha256,
        "max_prompt_tokens": max(prompt for prompt, _ in rendered_lengths),
        "max_prompt_plus_answer_tokens": max(total for _, total in rendered_lengths),
    }
    manifest_raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
    created_data = False
    try:
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created_data = True
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        fd = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(manifest_raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if created_data:
            destination.unlink(missing_ok=True)
        raise
    except BaseException:
        if created_data:
            destination.unlink(missing_ok=True)
        raise
    return manifest
