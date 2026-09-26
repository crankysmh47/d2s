"""V2 list prompt rendering and deterministic training-view transformation."""

from __future__ import annotations

import json
from hashlib import sha256
import os
from pathlib import Path
import random
import re
import stat
from typing import Mapping, Sequence

from .config_v2 import V2Config
from .data import Example, _thaw_json, interpret
from .data import _compact_json, _example_payload, _expression_tree, _shortest_path_length, render_example
from .provenance import sha256_payload, validate_revision


_OPERATION_DEFINITIONS = {
    "ROTL1": "ROTL1 moves the first element to the end",
    "REVERSE": "REVERSE reverses the list",
    "ADD1": "ADD1 adds 1 modulo 10 to each element",
}
_FAMILIES = {"expressions", "lists", "graphs"}
_FAMILY_ORDER = ("expressions", "lists", "graphs")
_V2_TOKENIZER_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
_FREEZE_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def render_v2_list_prompt(latent_spec: Mapping[str, object]) -> str:
    """Render the fixed V2 list prompt from a list specification alone."""
    if not isinstance(latent_spec, Mapping) or set(latent_spec) != {"input", "ops"}:
        raise ValueError("list latent specification must contain exactly input and ops")
    values, operations = latent_spec["input"], latent_spec["ops"]
    if (
        not isinstance(values, (list, tuple)) or not values
        or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 9 for value in values)
    ):
        raise ValueError("list input must contain integers from zero through nine")
    if (
        not isinstance(operations, (list, tuple)) or not operations
        or any(not isinstance(operation, str) or operation not in _OPERATION_DEFINITIONS for operation in operations)
    ):
        raise ValueError("list ops must contain registered operation names")
    compact_input = json.dumps(list(values), separators=(",", ":"))
    operation_names = ", ".join(operations)
    definitions = "; ".join(_OPERATION_DEFINITIONS[name] for name in dict.fromkeys(operations))
    return (
        f"Initial list: {compact_input}. Operations to apply in order: {operation_names}. "
        "Apply exactly these operations once each, from left to right; do not apply any operation not listed. "
        f"Definitions for listed operations: {definitions}. Return only the final list."
    )


def make_pilot_v2(
    config: V2Config, family: str, candidate: int, forbidden_ids: set[str]
) -> tuple[Example, ...]:
    """Generate one registered V2 pilot with a private, deterministic PRNG stream."""
    if family not in _FAMILY_ORDER:
        raise ValueError(f"unknown benchmark family {family!r}")
    if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate not in (1, 2):
        raise ValueError("candidate must be 1 or 2")
    _validate_forbidden_ids(forbidden_ids)
    seed = config.data.pilot_seeds[family][candidate - 1]
    return _generate_v2(
        config, family, candidate, "pilot", config.data.pilot_per_candidate, seed, set(forbidden_ids)
    )


def make_selected_splits_v2(
    config: V2Config, choices: Mapping[str, int], forbidden_ids: set[str]
) -> dict[str, tuple[Example, ...]]:
    """Generate selected V2 development and sealed data, excluding prior IDs."""
    if not isinstance(choices, Mapping) or set(choices) != set(_FAMILY_ORDER):
        raise ValueError("choices must select one candidate for each family")
    for family in _FAMILY_ORDER:
        choice = choices[family]
        if isinstance(choice, bool) or not isinstance(choice, int) or choice not in (1, 2):
            raise ValueError(f"invalid candidate choice for {family}")
    _validate_forbidden_ids(forbidden_ids)

    seen = set(forbidden_ids)
    output: dict[str, tuple[Example, ...]] = {}
    for split, count, seeds in (
        ("dev", config.data.development_per_family, config.data.development_seeds),
        ("sealed", config.data.sealed_per_family, config.data.sealed_seeds),
    ):
        rows: list[Example] = []
        for family in _FAMILY_ORDER:
            rows.extend(_generate_v2(config, family, choices[family], split, count, seeds[family], seen))
        output[split] = tuple(rows)
    return output


def _validate_forbidden_ids(forbidden_ids: set[str]) -> None:
    if not isinstance(forbidden_ids, set) or any(not isinstance(value, str) for value in forbidden_ids):
        raise TypeError("forbidden_ids must be a set of IDs")


def _generate_v2(
    config: V2Config,
    family: str,
    candidate: int,
    split: str,
    count: int,
    seed: int,
    seen: set[str],
) -> tuple[Example, ...]:
    rng = random.Random(seed)
    accepted: list[Example] = []
    while len(accepted) < count:
        accepted_index = len(accepted)
        for _ in range(config.data.maximum_attempts_per_example):
            row = _candidate_v2(config, family, candidate, split, accepted_index, rng)
            if row is None or row.id in seen:
                continue
            accepted.append(row)
            seen.add(row.id)
            break
        else:
            raise ValueError(
                f"could not accept V2 {family}/{split} example {accepted_index} after "
                f"{config.data.maximum_attempts_per_example} attempts (forbidden or rejected candidates)"
            )
    return tuple(accepted)


def _candidate_v2(
    config: V2Config, family: str, candidate: int, split: str, accepted_index: int, rng: random.Random
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
        structure_id = f"n{item_count}:{','.join(operations)}"  # type: ignore[arg-type]
        prompt = render_v2_list_prompt(latent)
        answer = _compact_json(answer_value)
    elif family == "graphs":
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
    else:
        raise ValueError(f"unknown benchmark family {family!r}")

    identifier = sha256_payload({"family": family, "latent_spec": latent})
    return Example(identifier, family, split, structure_id, prompt, answer, latent)


def freeze_v2_split(
    path: Path,
    examples: Sequence[Example],
    tokenizer: object,
    *,
    revision: str,
    tokenizer_sha256: str,
) -> dict[str, object]:
    """Exclusively create a deterministic V2 JSONL split and provenance sidecar."""
    try:
        validate_revision(revision)
    except ValueError as error:
        raise ValueError("V2 freeze requires an immutable tokenizer revision") from error
    if revision != _V2_TOKENIZER_REVISION:
        raise ValueError("V2 freeze requires the registered tokenizer revision")
    if not isinstance(tokenizer_sha256, str) or _FREEZE_SHA256.fullmatch(tokenizer_sha256) is None:
        raise ValueError("V2 freeze requires a full tokenizer SHA-256")
    if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence) or not examples:
        raise ValueError("V2 split must contain at least one example")
    if any(not isinstance(row, Example) for row in examples):
        raise ValueError("V2 split contains a malformed row")
    splits = {row.split for row in examples}
    if len(splits) != 1:
        raise ValueError("V2 split must be well formed and homogeneous")
    split = next(iter(splits))
    if split not in {"pilot", "dev", "sealed"}:
        raise ValueError("V2 split must be pilot, dev, or sealed")

    identifiers: set[str] = set()
    rendered_lengths: list[tuple[int, int]] = []
    encoded_rows: list[bytes] = []
    for row in examples:
        if row.family not in _FAMILIES:
            raise ValueError("V2 split row has an unknown family")
        if row.id in identifiers:
            raise ValueError("V2 split contains duplicate IDs")
        identifiers.add(row.id)
        latent = _thaw_json(row.latent_spec)
        expected_id = sha256_payload({"family": row.family, "latent_spec": latent})
        if row.id != expected_id:
            raise ValueError("V2 split row does not match its canonical ID")
        try:
            evaluated = interpret(row.latent_spec, row.family)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("V2 split row latent specification is malformed") from error
        expected_answer = evaluated if row.family == "graphs" else _compact_json(evaluated)
        if row.answer != expected_answer:
            raise ValueError("V2 split row answer is inaccurate")
        if not row.prompt or not row.structure_id:
            raise ValueError("V2 split row has a malformed prompt or structure ID")
        if row.family == "lists" and row.prompt != render_v2_list_prompt(row.latent_spec):
            raise ValueError("V2 split row list prompt does not match the frozen V2 list prompt")
        rendered = render_example(tokenizer, row)
        prompt_tokens = len(rendered.prompt_input_ids)
        if prompt_tokens + 256 > 512:
            raise ValueError("V2 rendered prompt exceeds prompt-plus-generation token budget")
        answer_tokens = len(rendered.train_input_ids) - prompt_tokens - 1
        rendered_lengths.append((prompt_tokens, prompt_tokens + answer_tokens))
        payload = json.dumps(_example_payload(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        encoded_rows.append(payload.encode("utf-8") + b"\n")

    raw = b"".join(encoded_rows)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "split": split,
        "count": len(examples),
        "sha256": sha256(raw).hexdigest(),
        "ordered_ids": [row.id for row in examples],
        "tokenizer_revision": revision,
        "tokenizer_sha256": tokenizer_sha256,
        "max_prompt_tokens": max(prompt for prompt, _ in rendered_lengths),
        "max_prompt_plus_answer_tokens": max(total for _, total in rendered_lengths),
    }
    manifest_raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
    destination = Path(path)
    manifest_path = Path(str(destination) + ".manifest.json")
    _reject_v2_reparse_ancestors(destination, manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_v2_reparse_ancestors(destination, manifest_path)

    created_data = False
    created_manifest = False
    try:
        _reject_v2_reparse_ancestors(destination, manifest_path)
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created_data = True
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        _reject_v2_reparse_ancestors(destination, manifest_path)
        descriptor = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created_manifest = True
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(manifest_raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if created_manifest:
            manifest_path.unlink(missing_ok=True)
        if created_data:
            destination.unlink(missing_ok=True)
        raise
    return manifest


def _reject_v2_reparse_ancestors(*targets: Path) -> None:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    checked: set[Path] = set()
    for target in targets:
        absolute = Path(os.path.abspath(target))
        for ancestor in (*reversed(absolute.parents), absolute):
            if ancestor in checked:
                continue
            checked.add(ancestor)
            try:
                metadata = os.lstat(ancestor)
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(metadata.st_mode) or bool(
                getattr(metadata, "st_file_attributes", 0) & reparse_flag
            ):
                raise ValueError(f"V2 split output path contains a symlink or reparse point: {ancestor}")


def make_training_view_v2(train_rows: Sequence[Example]) -> tuple[Example, ...]:
    """Copy verified train examples, applying only the frozen V2 list prompt repair."""
    if isinstance(train_rows, (str, bytes)) or not isinstance(train_rows, Sequence) or not train_rows:
        raise ValueError("training rows must be a non-empty sequence of Example records")
    view: list[Example] = []
    for row in train_rows:
        if not isinstance(row, Example):
            raise ValueError("training view contains a malformed row")
        if (
            not isinstance(row.id, str) or not row.id
            or not isinstance(row.family, str) or row.family not in _FAMILIES
            or not isinstance(row.split, str) or row.split != "train"
            or not isinstance(row.structure_id, str) or not row.structure_id
            or not isinstance(row.prompt, str) or not row.prompt
            or not isinstance(row.answer, str) or not row.answer
            or not isinstance(row.latent_spec, Mapping)
        ):
            raise ValueError("training view contains a malformed row")
        latent = _thaw_json(row.latent_spec)
        if row.id != sha256_payload({"family": row.family, "latent_spec": latent}):
            raise ValueError("training row ID does not match its canonical latent specification")
        try:
            expected_value = interpret(row.latent_spec, row.family)
            expected_answer = expected_value if row.family == "graphs" else (
                str(expected_value) if row.family == "expressions" else json.dumps(expected_value, separators=(",", ":"))
            )
            if row.family == "lists":
                prompt = render_v2_list_prompt(row.latent_spec)
            else:
                prompt = row.prompt
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("training row latent specification is malformed") from error
        if row.answer != expected_answer:
            raise ValueError("training row answer does not match its latent specification")
        view.append(Example(row.id, row.family, row.split, row.structure_id, prompt, row.answer, latent))
    return tuple(view)
