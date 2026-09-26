"""V2 list prompt rendering and deterministic training-view transformation."""

from __future__ import annotations

import json
from typing import Mapping, Sequence

from .data import Example, _thaw_json, interpret
from .provenance import sha256_payload


_OPERATION_DEFINITIONS = {
    "ROTL1": "ROTL1 moves the first element to the end",
    "REVERSE": "REVERSE reverses the list",
    "ADD1": "ADD1 adds 1 modulo 10 to each element",
}
_FAMILIES = {"expressions", "lists", "graphs"}


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
