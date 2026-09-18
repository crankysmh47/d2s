"""Deterministic synthetic exact-answer data for the recurrence V0 protocol."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re
import tempfile
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Sequence

from .config import RecurrenceConfig
from .provenance import canonical_json, sha256_payload, validate_revision


FAMILIES = ("expressions", "lists", "graphs")
SPLITS = ("train", "dev", "test")
_SPLIT_SEEDS = {
    "train": {"expressions": 101, "lists": 102, "graphs": 103},
    "dev": {"expressions": 201, "lists": 202, "graphs": 203},
    "test": {"expressions": 301, "lists": 302, "graphs": 303},
}
_INTEGER = re.compile(r"-?(0|[1-9][0-9]*)\Z")
_FULL_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_GENERATOR_CONTRACT: dict[str, object] = {
    "schema_version": 1,
    "family_order": list(FAMILIES),
    "split_order": list(SPLITS),
    "split_seeds": _SPLIT_SEEDS,
    "maximum_attempt_multiplier": 100,
    "counts": {
        "train": {"expressions": 2667, "lists": 2667, "graphs": 2666},
        "dev_per_family": 200,
        "test_per_family": 800,
    },
    "expressions": {
        "leaves_by_split": {"train": [3, 4], "dev": [5], "test": [6]},
        "shapes": ["chain", "balanced"],
        "shape_schedule": "accepted_row_index_alternating_chain_first",
        "leaf_values": [0, 9],
        "operators": ["+", "-", "*"],
        "maximum_abs_answer": 9999,
    },
    "lists": {
        "input_lengths_by_split": {"train": [3, 5], "dev": [6], "test": [7]},
        "program_lengths_by_split": {"train": [1, 2], "dev": [3], "test": [4]},
        "value_range": [0, 9],
        "operations": ["ROTL1", "REVERSE", "ADD1"],
        "reject_adjacent": ["REVERSE", "REVERSE"],
        "reject_identity": True,
    },
    "graphs": {
        "vertices_by_split": {"train": [5, 6], "dev": [7], "test": [8]},
        "vertex_names": "A onward",
        "ordered_distinct_edge_probability": 0.20,
        "edges_sorted_lexicographically": True,
        "query_vertices": "distinct_uniform",
        "label_schedule": "accepted_row_index_alternating_yes_first",
        "empty_edges_allowed": True,
    },
}


@dataclass(frozen=True, slots=True)
class Example:
    id: str
    family: str
    split: str
    structure_id: str
    prompt: str
    answer: str
    latent_spec: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "latent_spec", _freeze_json(self.latent_spec))


@dataclass(frozen=True, slots=True)
class RenderedExample:
    example: Example
    prompt_input_ids: tuple[int, ...]
    train_input_ids: tuple[int, ...]
    labels: tuple[int, ...]


_RETENTION_TARGET_TOKENS = 20_000
_RETENTION_CHUNK_TOKENS = 512
_RETENTION_MANIFEST_NAME = "retention-manifest.json"
_RETENTION_CHUNKS_NAME = "retention-chunks.jsonl"
_RETENTION_FULL_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


class RetentionSourceError(RuntimeError):
    """The historical FineWeb documents cannot be recovered exactly."""


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """One raw historical FineWeb document verified against its old manifest."""

    id: str
    text: str
    text_sha256: str
    row_index: int


@dataclass(frozen=True, slots=True)
class RetentionChunk:
    """One independently bootstrapable causal-loss unit from the retention stream."""

    index: int
    input_ids: tuple[int, ...]
    source_document_ids: tuple[str, ...]
    scored_tokens: int

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("retention chunk index must be a non-negative integer")
        if not isinstance(self.input_ids, tuple) or not (2 <= len(self.input_ids) <= _RETENTION_CHUNK_TOKENS):
            raise ValueError("retention chunk must contain two to 512 input tokens")
        if any(isinstance(token, bool) or not isinstance(token, int) for token in self.input_ids):
            raise ValueError("retention chunk input IDs must be integers")
        if (
            not isinstance(self.source_document_ids, tuple)
            or not self.source_document_ids
            or any(not isinstance(identifier, str) or not identifier for identifier in self.source_document_ids)
            or len(set(self.source_document_ids)) != len(self.source_document_ids)
        ):
            raise ValueError("retention chunk requires unique contributing document IDs")
        if self.scored_tokens != len(self.input_ids) - 1 or self.scored_tokens <= 0:
            raise ValueError("retention chunk scored-token count is invalid")


class _GeneratedExamples(dict[str, list[Example]]):
    """The public split mapping plus deterministic generation accounting."""

    def __init__(
        self,
        rows: dict[str, list[Example]],
        rejections: Mapping[str, Mapping[str, int]],
        order_evidence: Mapping[str, Sequence[str]],
    ) -> None:
        super().__init__(rows)
        self.rejections = _freeze_json(rejections)
        self.order_evidence = _freeze_json(order_evidence)


def make_examples(config: RecurrenceConfig) -> dict[str, list[Example]]:
    """Generate every registered split with independent family/split PRNG streams."""
    counts = {
        "train": {
            "expressions": config.data.train_expression_examples,
            "lists": config.data.train_list_examples,
            "graphs": config.data.train_graph_examples,
        },
        "dev": {family: config.data.dev_examples_per_family for family in FAMILIES},
        "test": {family: config.data.test_examples_per_family for family in FAMILIES},
    }
    seeds = _config_split_seeds(config)
    generated: dict[str, list[Example]] = {split: [] for split in SPLITS}
    rejections = {split: {family: 0 for family in FAMILIES} for split in SPLITS}
    seen_ids: set[str] = set()
    for split in SPLITS:
        for family in FAMILIES:
            rows, rejected = _generate_family(
                family, split, counts[split][family], seeds[split][family], seen_ids
            )
            generated[split].extend(rows)
            rejections[split][family] = rejected
    order_evidence = {
        split: tuple(_example_integrity_hash(row) for row in generated[split]) for split in SPLITS
    }
    return _GeneratedExamples(generated, rejections, order_evidence)


def interpret(latent_spec: Mapping[str, object], family: str) -> object:
    """Evaluate only the three registered latent languages, never generated Python."""
    if family == "expressions":
        leaves = _integer_list(latent_spec.get("leaves"), "expression leaves")
        operators = _string_list(latent_spec.get("operators"), "expression operators")
        shape = latent_spec.get("shape")
        if shape not in {"chain", "balanced"} or len(leaves) < 2 or len(operators) != len(leaves) - 1:
            raise ValueError("invalid expression latent specification")
        return _expression_tree(leaves, operators, str(shape))[0]
    if family == "lists":
        values = _integer_list(latent_spec.get("input"), "list input")
        ops = _string_list(latent_spec.get("ops"), "list operations")
        if any(value < 0 or value > 9 for value in values) or any(op not in {"ROTL1", "REVERSE", "ADD1"} for op in ops):
            raise ValueError("invalid list latent specification")
        result = list(values)
        for op in ops:
            if op == "ROTL1":
                result = result[1:] + result[:1]
            elif op == "REVERSE":
                result.reverse()
            else:
                result = [(value + 1) % 10 for value in result]
        return result
    if family == "graphs":
        n = latent_spec.get("n")
        edges = latent_spec.get("edges")
        start = latent_spec.get("start")
        end = latent_spec.get("end")
        if isinstance(n, bool) or not isinstance(n, int) or n < 2 or not isinstance(start, str) or not isinstance(end, str):
            raise ValueError("invalid graph latent specification")
        vertices = [chr(ord("A") + index) for index in range(n)]
        if start not in vertices or end not in vertices or start == end or not isinstance(edges, (list, tuple)):
            raise ValueError("invalid graph latent specification")
        adjacency: dict[str, list[str]] = {vertex: [] for vertex in vertices}
        for edge in edges:
            if not isinstance(edge, str) or edge.count("->") != 1:
                raise ValueError("invalid graph edge")
            source, target = edge.split("->")
            if source not in adjacency or target not in adjacency or source == target:
                raise ValueError("invalid graph edge")
            adjacency[source].append(target)
        return "yes" if _reachable(adjacency, start, end) else "no"
    raise ValueError(f"unknown exact-answer family {family!r}")


def parse_answer(text: str, example: Example) -> object:
    """Parse a model completion as a complete canonical answer, without repair."""
    if not isinstance(text, str):
        raise ValueError("completion must be text")
    value = text.strip()
    if example.family == "expressions":
        if _INTEGER.fullmatch(value) is None:
            raise ValueError("invalid integer answer")
        return int(value)
    if example.family == "lists":
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("invalid list answer") from error
        expected = interpret(example.latent_spec, "lists")
        if (
            not isinstance(parsed, list)
            or len(parsed) != len(expected)
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 or item > 9 for item in parsed)
        ):
            raise ValueError("invalid list answer")
        return parsed
    if example.family == "graphs":
        normalized = value.casefold()
        if normalized not in {"yes", "no"}:
            raise ValueError("invalid graph answer")
        return normalized
    raise ValueError(f"unknown exact-answer family {example.family!r}")


def render_example(tokenizer: object, example: Example) -> RenderedExample:
    """Render a single direct-answer chat turn and align answer-only causal labels."""
    renderer = getattr(tokenizer, "apply_chat_template", None)
    if not callable(renderer):
        raise TypeError("tokenizer must provide apply_chat_template")
    rendered_prompt = renderer(
        [{"role": "user", "content": example.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # Transformers tokenizers may return a BatchEncoding mapping for chat
    # templates, while the focused fixtures return the bare ID sequence.
    prompt_ids = tuple(
        _token_ids(rendered_prompt.get("input_ids") if isinstance(rendered_prompt, Mapping) else rendered_prompt)
    )
    if len(prompt_ids) > 480:
        raise ValueError("rendered prompt exceeds 480-token generation budget")
    tokenized = tokenizer(example.answer, add_special_tokens=False)
    if not isinstance(tokenized, Mapping) or "input_ids" not in tokenized:
        raise TypeError("tokenizer completion encoding must contain input_ids")
    answer_ids = tuple(_token_ids(tokenized["input_ids"]))
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise ValueError("tokenizer must provide an integer EOS token id")
    completion_ids = answer_ids + (eos_token_id,)
    if len(completion_ids) > 32:
        raise ValueError("answer plus EOS exceeds 32-token generation budget")
    train_ids = prompt_ids + completion_ids
    if len(train_ids) > 512:
        raise ValueError("rendered training sequence exceeds 512 tokens")
    return RenderedExample(
        example=example,
        prompt_input_ids=prompt_ids,
        train_input_ids=train_ids,
        labels=(-100,) * len(prompt_ids) + completion_ids,
    )


def freeze_examples(
    root: Path,
    examples: Mapping[str, Sequence[Example]],
    *,
    tokenizer: object | None = None,
    tokenizer_revision: str | None = None,
    tokenizer_sha256: str | None = None,
) -> dict[str, object]:
    """Atomically materialize deterministic JSONL evidence and its immutable manifest."""
    _validate_tokenizer_provenance(tokenizer, tokenizer_revision, tokenizer_sha256)
    if set(examples) != set(SPLITS):
        raise ValueError("exact-answer freeze requires train, dev, and test splits")
    order_evidence = getattr(examples, "order_evidence", None)
    if not isinstance(order_evidence, Mapping) or set(order_evidence) != set(SPLITS):
        raise ValueError("freeze requires registered generation order evidence")

    # Audit all source rows before creating even the first output directory.
    data_root = root / "data"
    split_hashes: dict[str, str] = {}
    split_id_hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    global_ids: set[str] = set()
    split_payloads: dict[str, bytes] = {}
    maximum_prompt = 0
    maximum_answer_eos = 0
    maximum_sequence = 0
    for split in SPLITS:
        rows = examples[split]
        actual_order = tuple(_example_integrity_hash(row) for row in rows)
        expected_order = tuple(order_evidence[split])
        if actual_order != expected_order:
            raise ValueError(f"{split} rows do not match registered generation order")
        for row in rows:
            _validate_example_identity(row, split)
            rendered = render_example(tokenizer, row)
            maximum_prompt = max(maximum_prompt, len(rendered.prompt_input_ids))
            maximum_answer_eos = max(
                maximum_answer_eos, len(rendered.train_input_ids) - len(rendered.prompt_input_ids)
            )
            maximum_sequence = max(maximum_sequence, len(rendered.train_input_ids))
        payload = b"".join(canonical_json(_example_payload(row)) for row in rows)
        ids = [row.id for row in rows]
        if len(ids) != len(set(ids)) or global_ids.intersection(ids):
            raise ValueError("exact-answer examples contain duplicate IDs")
        global_ids.update(ids)
        split_payloads[split] = payload
        split_hashes[split] = sha256(payload).hexdigest()
        split_id_hashes[split] = sha256_payload({"ids": sorted(ids)})
        counts[split] = len(rows)
    rejections = getattr(examples, "rejections", None)
    if not _valid_rejections(rejections):
        raise ValueError("freeze requires immutable split-by-family rejection accounting")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "counts": counts,
        "split_sha256": split_hashes,
        "split_id_sha256": split_id_hashes,
        "generator": _GENERATOR_CONTRACT,
        "generator_sha256": _canonical_payload_sha256(_GENERATOR_CONTRACT),
        "rejections": _thaw_json(rejections),
        "maximum_token_lengths": {
            "prompt": maximum_prompt,
            "answer_plus_eos": maximum_answer_eos,
            "sequence": maximum_sequence,
        },
        "tokenizer": {"revision": tokenizer_revision.lower(), "sha256": tokenizer_sha256.lower()},
    }
    # Only the fully audited payload may now reach the filesystem.
    for split in SPLITS:
        _write_immutable_bytes(data_root / f"{split}.jsonl", split_payloads[split])
    manifest_bytes = canonical_json(manifest)
    _write_immutable_bytes(data_root / "manifest.json", manifest_bytes)
    return manifest


def _generate_family(
    family: str, split: str, requested_count: int, seed: int, seen_ids: set[str]
) -> tuple[list[Example], int]:
    rng = random.Random(seed)
    rows: list[Example] = []
    attempts = 0
    while len(rows) < requested_count:
        attempts += 1
        if attempts > 100 * requested_count:
            raise RuntimeError(f"{family}/{split} exceeded {100 * requested_count} generation attempts")
        candidate = _candidate(family, split, len(rows), rng)
        if candidate is None or candidate.id in seen_ids:
            continue
        seen_ids.add(candidate.id)
        rows.append(candidate)
    return rows, attempts - requested_count


def _config_split_seeds(config: RecurrenceConfig) -> dict[str, dict[str, int]]:
    configured = {
        "train": config.data.train_seeds,
        "dev": config.data.dev_seeds,
        "test": config.data.test_seeds,
    }
    output: dict[str, dict[str, int]] = {}
    for split, seeds in configured.items():
        if len(seeds) != len(FAMILIES):
            raise ValueError(f"{split} must define one exact-answer seed per family")
        output[split] = dict(zip(FAMILIES, seeds, strict=True))
    if output != _SPLIT_SEEDS:
        raise ValueError("locked recurrence-v0 protocol rejects exact-answer generation seeds")
    return output


def _candidate(family: str, split: str, accepted_index: int, rng: random.Random) -> Example | None:
    if family == "expressions":
        leaves_count = rng.choice((3, 4)) if split == "train" else 5 if split == "dev" else 6
        shape = "chain" if accepted_index % 2 == 0 else "balanced"
        latent_spec: dict[str, object] = {
            "shape": shape,
            "leaves": [rng.randrange(10) for _ in range(leaves_count)],
            "operators": [rng.choice(("+", "-", "*")) for _ in range(leaves_count - 1)],
        }
        answer = interpret(latent_spec, family)
        if abs(answer) > 9999:
            return None
        expression = _expression_tree(
            _integer_list(latent_spec["leaves"], "expression leaves"),
            _string_list(latent_spec["operators"], "expression operators"),
            shape,
        )[1]
        structure_id = f"{shape}{leaves_count}"
        prompt = f"Compute {expression}. Return only the integer answer."
        answer_text = str(answer)
    elif family == "lists":
        input_length = rng.choice((3, 4, 5)) if split == "train" else 6 if split == "dev" else 7
        program_length = rng.choice((1, 2)) if split == "train" else 3 if split == "dev" else 4
        latent_spec = {
            "input": [rng.randrange(10) for _ in range(input_length)],
            "ops": [rng.choice(("ROTL1", "REVERSE", "ADD1")) for _ in range(program_length)],
        }
        ops = _string_list(latent_spec["ops"], "list operations")
        if any(pair == ("REVERSE", "REVERSE") for pair in zip(ops, ops[1:])):
            return None
        answer = interpret(latent_spec, family)
        if answer == latent_spec["input"]:
            return None
        input_text = _compact_json(latent_spec["input"])
        ops_text = ",".join(ops)
        structure_id = f"n{input_length}:{ops_text}"
        prompt = (
            f"Start with {input_text}. Apply {ops_text} from left to right. ROTL1 moves the first element "
            "to the end; REVERSE reverses the list; ADD1 adds 1 modulo 10 to each element. "
            "Return only the final list."
        )
        answer_text = _compact_json(answer)
    elif family == "graphs":
        n = rng.choice((5, 6)) if split == "train" else 7 if split == "dev" else 8
        vertices = [chr(ord("A") + index) for index in range(n)]
        edges = [
            f"{source}->{target}"
            for source in vertices
            for target in vertices
            if source != target and rng.random() < 0.20
        ]
        edges.sort()
        start, end = rng.sample(vertices, 2)
        latent_spec = {"n": n, "edges": edges, "start": start, "end": end}
        answer = interpret(latent_spec, family)
        wanted = "yes" if accepted_index % 2 == 0 else "no"
        if answer != wanted:
            return None
        path_length = _shortest_path_length(n, edges, start, end)
        structure_id = f"n{n}:d{path_length if path_length is not None else 'unreachable'}"
        edges_text = ", ".join(edges) if edges else "none"
        prompt = (
            f"Directed edges: {edges_text}. Is there a directed path from {start} to {end}? "
            "Return only yes or no."
        )
        answer_text = answer
    else:
        raise ValueError(f"unknown exact-answer family {family!r}")
    identifier = sha256_payload({"family": family, "latent_spec": latent_spec})
    return Example(identifier, family, split, structure_id, prompt, answer_text, latent_spec)


def _expression_tree(leaves: list[int], operators: list[str], shape: str) -> tuple[int, str]:
    if shape == "chain":
        result = leaves[0]
        rendered = str(result)
        for operator, leaf in zip(operators, leaves[1:]):
            result = _apply_operator(result, leaf, operator)
            rendered = f"({rendered} {operator} {leaf})"
        return result, rendered
    leaf_position = 0
    operator_position = 0

    def build(count: int) -> tuple[int, str]:
        nonlocal leaf_position, operator_position
        if count == 1:
            value = leaves[leaf_position]
            leaf_position += 1
            return value, str(value)
        left_count = count // 2
        left_value, left_text = build(left_count)
        operator = operators[operator_position]
        operator_position += 1
        right_value, right_text = build(count - left_count)
        return _apply_operator(left_value, right_value, operator), f"({left_text} {operator} {right_text})"

    return build(len(leaves))


def _apply_operator(left: int, right: int, operator: str) -> int:
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    raise ValueError("invalid expression operator")


def _reachable(adjacency: Mapping[str, Sequence[str]], start: str, end: str) -> bool:
    pending: deque[str] = deque([start])
    seen = {start}
    while pending:
        current = pending.popleft()
        if current == end:
            return True
        for target in adjacency[current]:
            if target not in seen:
                seen.add(target)
                pending.append(target)
    return False


def _shortest_path_length(n: int, edges: list[str], start: str, end: str) -> int | None:
    adjacency: dict[str, list[str]] = {chr(ord("A") + index): [] for index in range(n)}
    for edge in edges:
        source, target = edge.split("->")
        adjacency[source].append(target)
    pending: deque[tuple[str, int]] = deque([(start, 0)])
    seen = {start}
    while pending:
        current, distance = pending.popleft()
        if current == end:
            return distance
        for target in adjacency[current]:
            if target not in seen:
                seen.add(target)
                pending.append((target, distance + 1))
    return None


def _integer_list(value: object, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{name} must be an integer list")
    return list(value)


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a string list")
    return list(value)


def _compact_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _token_ids(value: object) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError("tokenizer must return one sequence of integer token IDs")
    return tuple(value)


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"immutable evidence already exists: {path.resolve()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _freeze_json(value: object) -> object:
    """Recursively freeze the JSON-only data allowed inside immutable evidence records."""
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("latent JSON object keys must be strings")
            frozen[key] = _freeze_json(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("latent specifications must contain only JSON-compatible values")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _example_payload(row: Example) -> dict[str, object]:
    return {
        "id": row.id,
        "family": row.family,
        "split": row.split,
        "structure_id": row.structure_id,
        "prompt": row.prompt,
        "answer": row.answer,
        "latent_spec": _thaw_json(row.latent_spec),
    }


def _example_integrity_hash(row: Example) -> str:
    payload = _example_payload(row)
    # Stream-order evidence protects all generated content apart from the identity
    # field, whose independent canonical recomputation is checked at freeze time.
    payload.pop("id")
    return sha256_payload(payload)


def _validate_example_identity(row: Example, split: str) -> None:
    if not isinstance(row, Example) or row.split != split or row.family not in FAMILIES:
        raise ValueError("frozen row has an invalid split or family")
    expected_id = sha256_payload({"family": row.family, "latent_spec": _thaw_json(row.latent_spec)})
    if row.id != expected_id:
        raise ValueError("frozen row does not match its canonical ID")
    evaluated = interpret(row.latent_spec, row.family)
    expected_answer = evaluated if row.family == "graphs" else _compact_json(evaluated)
    if row.answer != expected_answer:
        raise ValueError("frozen row answer does not match its latent specification")


def _validate_tokenizer_provenance(
    tokenizer: object | None, tokenizer_revision: str | None, tokenizer_sha256: str | None
) -> None:
    try:
        validate_revision(tokenizer_revision)  # type: ignore[arg-type]
    except ValueError as error:
        raise ValueError("final freeze requires a full immutable tokenizer revision") from error
    if not isinstance(tokenizer_sha256, str) or _FULL_SHA256.fullmatch(tokenizer_sha256) is None:
        raise ValueError("final freeze requires a full tokenizer SHA-256")
    if tokenizer is None:
        raise ValueError("final freeze requires a tokenizer")


def _valid_rejections(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != set(SPLITS):
        return False
    for split in SPLITS:
        family_counts = value[split]
        if not isinstance(family_counts, Mapping) or set(family_counts) != set(FAMILIES):
            return False
        if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in family_counts.values()):
            return False
    return True


def _canonical_payload_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def recover_retention_documents(
    old_root: Path, dataset_loader: Callable[..., Iterable[Mapping[str, object]]]
) -> tuple[DocumentRecord, ...]:
    """Recover raw historical FineWeb documents in manifest order.

    Historical token IDs are deliberately never accepted as a substitute for raw
    text. A cache is useful only when its text hashes prove it is the same source.
    """
    manifest_path = _retention_old_manifest_path(old_root)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RetentionSourceError("historical retention manifest is unreadable") from error
    if not isinstance(manifest, Mapping):
        raise RetentionSourceError("historical retention manifest must be an object")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise RetentionSourceError("historical retention manifest is missing dataset provenance")
    name, repo_id, revision, split, streaming = (
        dataset.get("name"), dataset.get("repo_id"), dataset.get("revision"), dataset.get("split"), dataset.get("streaming")
    )
    if repo_id != "HuggingFaceFW/fineweb-edu":
        raise RetentionSourceError("historical retention source must be HuggingFaceFW/fineweb-edu")
    if name != "sample-10BT":
        raise RetentionSourceError("historical retention source must use the sample-10BT configuration")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise RetentionSourceError("historical retention dataset revision must be a full immutable commit SHA")
    if not isinstance(split, str) or not split or streaming is not True:
        raise RetentionSourceError("historical retention dataset provenance is incomplete")
    if manifest.get("ordering") != "repository_order":
        raise RetentionSourceError("historical retention source must preserve repository order")
    if manifest.get("shuffle") is not False:
        raise RetentionSourceError("historical retention source must set shuffle to false")
    expected = _retention_expected_documents(manifest.get("documents"))
    recovered: dict[str, str] = {}
    for identifier, text in _retention_cached_text(manifest_path.parent).items():
        if identifier not in expected:
            raise RetentionSourceError(
                f"cached raw retention text contains an unregistered document id: {identifier}"
            )
        _validate_retention_text(identifier, text, expected[identifier]["text_sha256"])
        recovered[identifier] = text
    if len(recovered) != len(expected):
        try:
            rows = dataset_loader(repo_id, name=name, split=split, revision=revision.lower(), streaming=True)
            seen_source_ids: set[str] = set(recovered)
            for row in rows:
                if not isinstance(row, Mapping):
                    raise RetentionSourceError("pinned FineWeb loader yielded a non-object row")
                identifier = row.get("id")
                if identifier not in expected:
                    continue
                if not isinstance(identifier, str):
                    raise RetentionSourceError("pinned FineWeb row requires a document id")
                if identifier in seen_source_ids:
                    raise RetentionSourceError(f"duplicate document id in pinned FineWeb source: {identifier}")
                seen_source_ids.add(identifier)
                text = row.get("text")
                if not isinstance(text, str):
                    raise RetentionSourceError(
                        f"pinned FineWeb source for {identifier} lacks raw text; token IDs are not valid retention input"
                    )
                _validate_retention_text(identifier, text, expected[identifier]["text_sha256"])
                recovered[identifier] = text
                if len(recovered) == len(expected):
                    break
        except RetentionSourceError:
            raise
        except Exception as error:
            raise RetentionSourceError("could not stream the pinned FineWeb retention source") from error
    missing = [identifier for identifier in expected if identifier not in recovered]
    if missing:
        raise RetentionSourceError(f"missing raw text for historical retention documents: {missing[:3]}")
    return tuple(
        DocumentRecord(identifier, recovered[identifier], expected[identifier]["text_sha256"], expected[identifier]["row_index"])
        for identifier in expected
    )


def materialize_retention(
    tokenizer: object, documents: Sequence[DocumentRecord], output_dir: Path
) -> dict[str, object]:
    """Retokenize verified text and persist immutable, independently scored chunks."""
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise ValueError("retention tokenizer must provide an integer EOS token id")
    if not documents:
        raise RetentionSourceError("retention materialization requires recovered documents")
    stream: list[tuple[int, str]] = []
    seen_ids: set[str] = set()
    used_ids: list[str] = []
    source_hashes: dict[str, str] = {}
    used_row_indices: list[int] = []
    previous_id: str | None = None
    previous_row_index: int | None = None
    final_document: DocumentRecord | None = None
    final_document_token_count = 0
    final_document_consumed = 0
    for document in documents:
        _validate_document_record(document, seen_ids)
        if previous_row_index is not None and document.row_index <= previous_row_index:
            raise RetentionSourceError("retention documents must have strictly increasing historical row indices")
        previous_row_index = document.row_index
        token_ids = _retention_token_ids(tokenizer, document.text)
        if not token_ids:
            continue
        if len(stream) >= _RETENTION_TARGET_TOKENS:
            break
        remaining = _RETENTION_TARGET_TOKENS - len(stream)
        if previous_id is not None:
            # A boundary only exists when at least one token of its right document fits.
            if remaining < 2:
                break
            stream.append((eos_token_id, previous_id))
            remaining -= 1
        consumed = min(len(token_ids), remaining)
        stream.extend((token, document.id) for token in token_ids[:consumed])
        used_ids.append(document.id)
        source_hashes[document.id] = document.text_sha256.lower()
        used_row_indices.append(document.row_index)
        previous_id = document.id
        final_document = document
        final_document_token_count = len(token_ids)
        final_document_consumed = consumed
        if len(stream) == _RETENTION_TARGET_TOKENS:
            break
    if len(stream) != _RETENTION_TARGET_TOKENS:
        raise RetentionSourceError(
            f"recovered FineWeb text provides {len(stream)} valid source tokens, not {_RETENTION_TARGET_TOKENS}"
        )
    chunks = _retention_chunks(stream)
    chunks_payload = b"".join(canonical_json(_retention_chunk_payload(chunk)) for chunk in chunks)
    if final_document is None or final_document_consumed <= 0:
        raise RetentionSourceError("retention materialization could not identify its final source document")
    manifest: dict[str, object] = {
        "schema_version": 2,
        "valid_source_tokens": _RETENTION_TARGET_TOKENS,
        "chunk_tokens": _RETENTION_CHUNK_TOKENS,
        "chunk_count": len(chunks),
        "scored_tokens": sum(chunk.scored_tokens for chunk in chunks),
        "eos_token_id": eos_token_id,
        "source_document_ids": used_ids,
        "source_document_row_indices": used_row_indices,
        "source_document_order_sha256": sha256_payload(
            {"source_document_ids": used_ids, "source_document_row_indices": used_row_indices}
        ),
        "source_document_sha256": source_hashes,
        "final_document_id": final_document.id,
        "final_document_full_token_count": final_document_token_count,
        "final_document_consumed_token_count": final_document_consumed,
        "final_document_truncated": final_document_consumed < final_document_token_count,
        "chunks_sha256": sha256(chunks_payload).hexdigest(),
    }
    manifest_payload = canonical_json(manifest)
    chunks_path = output_dir / _RETENTION_CHUNKS_NAME
    manifest_path = output_dir / _RETENTION_MANIFEST_NAME
    _verify_or_absent_immutable(chunks_path, chunks_payload)
    _verify_or_absent_immutable(manifest_path, manifest_payload)
    _write_immutable_bytes(chunks_path, chunks_payload)
    _write_immutable_bytes(manifest_path, manifest_payload)
    return manifest


def load_retention_chunks(output_dir: Path) -> tuple[RetentionChunk, ...]:
    """Load only the exact immutable bootstrap units recorded by materialization."""
    chunks_path = output_dir / _RETENTION_CHUNKS_NAME
    manifest_path = output_dir / _RETENTION_MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = chunks_path.read_bytes()
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("retention artifact is missing or unreadable") from error
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 2:
        raise ValueError("retention manifest schema version is invalid")
    if manifest.get("valid_source_tokens") != _RETENTION_TARGET_TOKENS:
        raise ValueError("retention manifest token count is invalid")
    if manifest.get("chunk_tokens") != _RETENTION_CHUNK_TOKENS:
        raise ValueError("retention manifest chunk size is invalid")
    if manifest.get("chunks_sha256") != sha256(payload).hexdigest():
        raise ValueError("retention chunk payload hash does not match manifest")
    try:
        chunks = tuple(_retention_chunk_from_payload(json.loads(line)) for line in payload.decode("utf-8").splitlines())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("retention chunks are not valid JSONL") from error
    if not chunks or tuple(chunk.index for chunk in chunks) != tuple(range(len(chunks))):
        raise ValueError("retention chunks do not have stable consecutive indices")
    if manifest.get("chunk_count") != len(chunks):
        raise ValueError("retention manifest chunk count is invalid")
    if manifest.get("scored_tokens") != sum(chunk.scored_tokens for chunk in chunks):
        raise ValueError("retention manifest scored-token count is invalid")
    if sum(len(chunk.input_ids) for chunk in chunks) != _RETENTION_TARGET_TOKENS:
        raise ValueError("retention chunks do not cover the registered token budget")
    source_ids = manifest.get("source_document_ids")
    source_row_indices = manifest.get("source_document_row_indices")
    source_order_sha256 = manifest.get("source_document_order_sha256")
    source_hashes = manifest.get("source_document_sha256")
    if (
        not isinstance(source_ids, list)
        or not source_ids
        or any(not isinstance(identifier, str) or not identifier for identifier in source_ids)
        or len(set(source_ids)) != len(source_ids)
        or not isinstance(source_hashes, Mapping)
        or set(source_hashes) != set(source_ids)
        or any(not isinstance(value, str) or _RETENTION_FULL_SHA256.fullmatch(value) is None for value in source_hashes.values())
        or not isinstance(source_row_indices, list)
        or len(source_row_indices) != len(source_ids)
        or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in source_row_indices)
        or any(right <= left for left, right in zip(source_row_indices, source_row_indices[1:]))
        or not isinstance(source_order_sha256, str)
        or _RETENTION_FULL_SHA256.fullmatch(source_order_sha256) is None
        or source_order_sha256
        != sha256_payload({"source_document_ids": source_ids, "source_document_row_indices": source_row_indices})
    ):
        raise ValueError("retention source-document provenance is invalid")
    if any(set(chunk.source_document_ids).difference(source_ids) for chunk in chunks):
        raise ValueError("retention chunk references an unregistered source document")
    _validate_retention_final_document(manifest, source_ids)
    return chunks


def _retention_old_manifest_path(old_root: Path) -> Path:
    candidates = (
        old_root if old_root.is_file() else old_root / "tokens" / "fineweb-heldout" / "manifest.json",
        old_root / "fineweb-heldout" / "manifest.json",
        old_root / "manifest.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RetentionSourceError("historical FineWeb retention manifest is missing")


def _retention_expected_documents(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise RetentionSourceError("historical retention manifest requires document records")
    expected: dict[str, dict[str, object]] = {}
    previous_row_index: int | None = None
    for default_index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise RetentionSourceError("historical retention document record must be an object")
        identifier, text_hash = row.get("id"), row.get("text_sha256")
        row_index = row.get("row_index", default_index)
        if not isinstance(identifier, str) or not identifier:
            raise RetentionSourceError("historical retention document requires a non-empty id")
        if identifier in expected:
            raise RetentionSourceError(f"duplicate document id in historical retention manifest: {identifier}")
        if not isinstance(text_hash, str) or _RETENTION_FULL_SHA256.fullmatch(text_hash) is None:
            raise RetentionSourceError(f"historical retention document {identifier} requires a text SHA-256")
        if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
            raise RetentionSourceError(f"historical retention document {identifier} has invalid row index")
        if previous_row_index is not None and row_index <= previous_row_index:
            raise RetentionSourceError("historical retention document row indices must be strictly increasing")
        expected[identifier] = {"text_sha256": text_hash.lower(), "row_index": row_index}
        previous_row_index = row_index
    return expected


def _validate_retention_final_document(manifest: Mapping[str, object], source_ids: Sequence[str]) -> None:
    """Validate the explicit, auditable boundary that ends the 20k-token stream."""
    identifier = manifest.get("final_document_id")
    full_count = manifest.get("final_document_full_token_count")
    consumed_count = manifest.get("final_document_consumed_token_count")
    truncated = manifest.get("final_document_truncated")
    if (
        not isinstance(identifier, str)
        or identifier != source_ids[-1]
        or isinstance(full_count, bool)
        or not isinstance(full_count, int)
        or full_count <= 0
        or isinstance(consumed_count, bool)
        or not isinstance(consumed_count, int)
        or not 0 < consumed_count <= full_count
        or not isinstance(truncated, bool)
        or truncated != (consumed_count < full_count)
    ):
        raise ValueError("retention manifest final-document cutoff is invalid")


def _retention_cached_text(manifest_dir: Path) -> dict[str, str]:
    path = manifest_dir / "raw_documents.jsonl"
    if not path.exists():
        return {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RetentionSourceError("cached raw retention text is unreadable") from error
    cached: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RetentionSourceError(f"cached raw retention text has invalid JSON at line {line_number}") from error
        if not isinstance(row, Mapping) or not isinstance(row.get("id"), str) or not row["id"]:
            raise RetentionSourceError(f"cached raw retention text requires id at line {line_number}")
        if not isinstance(row.get("text"), str):
            raise RetentionSourceError(f"cached raw retention text requires text at line {line_number}")
        identifier = row["id"]
        if identifier in cached:
            raise RetentionSourceError(f"duplicate document id in cached raw retention text: {identifier}")
        cached[identifier] = row["text"]
    return cached


def _validate_retention_text(identifier: str, text: str, expected_hash: object) -> None:
    if sha256(text.encode("utf-8")).hexdigest() != expected_hash:
        raise RetentionSourceError(f"retention source text hash mismatch for document {identifier}")


def _validate_document_record(document: DocumentRecord, seen_ids: set[str]) -> None:
    if not isinstance(document, DocumentRecord):
        raise TypeError("retention materialization requires DocumentRecord values")
    if not document.id or document.id in seen_ids:
        raise RetentionSourceError(f"duplicate document id in recovered retention source: {document.id}")
    if isinstance(document.row_index, bool) or not isinstance(document.row_index, int) or document.row_index < 0:
        raise RetentionSourceError(f"retention document {document.id} has invalid row index")
    if not isinstance(document.text, str):
        raise RetentionSourceError(f"retention document {document.id} lacks raw text")
    if _RETENTION_FULL_SHA256.fullmatch(document.text_sha256) is None:
        raise RetentionSourceError(f"retention document {document.id} has invalid text SHA-256")
    _validate_retention_text(document.id, document.text, document.text_sha256.lower())
    seen_ids.add(document.id)


def _retention_token_ids(tokenizer: object, text: str) -> tuple[int, ...]:
    if callable(tokenizer):
        encoded = tokenizer(text, add_special_tokens=False)
    else:
        encode = getattr(tokenizer, "encode", None)
        if not callable(encode):
            raise TypeError("retention tokenizer must be callable or provide encode")
        encoded = encode(text, add_special_tokens=False)
    return _token_ids(encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded)


def _retention_chunks(stream: Sequence[tuple[int, str]]) -> tuple[RetentionChunk, ...]:
    chunks: list[RetentionChunk] = []
    for index, start in enumerate(range(0, len(stream), _RETENTION_CHUNK_TOKENS)):
        window = stream[start : start + _RETENTION_CHUNK_TOKENS]
        chunks.append(
            RetentionChunk(
                index=index,
                input_ids=tuple(token for token, _ in window),
                source_document_ids=tuple(dict.fromkeys(identifier for _, identifier in window)),
                scored_tokens=len(window) - 1,
            )
        )
    return tuple(chunks)


def _retention_chunk_payload(chunk: RetentionChunk) -> dict[str, object]:
    return {
        "index": chunk.index,
        "input_ids": list(chunk.input_ids),
        "source_document_ids": list(chunk.source_document_ids),
        "scored_tokens": chunk.scored_tokens,
    }


def _retention_chunk_from_payload(value: object) -> RetentionChunk:
    fields = {"index", "input_ids", "source_document_ids", "scored_tokens"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("retention chunk has an invalid schema")
    input_ids, source_document_ids = value["input_ids"], value["source_document_ids"]
    if not isinstance(input_ids, list) or not isinstance(source_document_ids, list):
        raise ValueError("retention chunk has invalid sequences")
    return RetentionChunk(
        index=value["index"],  # type: ignore[arg-type]
        input_ids=tuple(input_ids),  # type: ignore[arg-type]
        source_document_ids=tuple(source_document_ids),  # type: ignore[arg-type]
        scored_tokens=value["scored_tokens"],  # type: ignore[arg-type]
    )


def _verify_or_absent_immutable(path: Path, payload: bytes) -> None:
    if path.exists() and path.read_bytes() != payload:
        raise FileExistsError(f"immutable retention artifact already exists: {path}")
