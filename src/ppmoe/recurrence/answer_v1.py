"""Frozen, conservative final-answer scorer for recurrence V1."""

from __future__ import annotations

import re

from .data import Example, interpret


_HEADING = r"(?:#{1,6}[ \t]*)?"
_EXPRESSION_CUE = re.compile(
    rf"^[ \t]*{_HEADING}(?:Final[ \t]+answer\b|So[ \t]+the[ \t]+final[ \t]+answer[ \t]+is\b)",
    re.IGNORECASE | re.MULTILINE,
)
_LIST_CUE = re.compile(
    rf"^[ \t]*{_HEADING}(Final[ \t]+answer|Final[ \t]+result|Answer)\b[ \t]*:?[ \t]*",
    re.IGNORECASE | re.MULTILINE,
)
_BOXED_INTEGER = re.compile(r"\\boxed\s*\{\s*([+-]?\d+)\s*\}")
_VECTOR = re.compile(r"\[\s*[+-]?\d+(?:\s*,\s*[+-]?\d+)*\s*\]")
_GRAPH_LEADING = re.compile(r"^\s*(Yes|No)\s*\.", re.IGNORECASE)
_GRAPH_TERMINAL = re.compile(r"^[ \t]*Answer[ \t]*:[ \t]*(yes|no)[.!]?[ \t]*$", re.IGNORECASE | re.MULTILINE)
_GRAPH_CONCLUSION = re.compile(
    r"\bthere\s+is\s+(?P<negative>no\s+)?(?:(?:direct(?:\s+or\s+indirect)?|indirect)\s+)?path\s+"
    r"from\s+(?P<start>[A-Z])\s+to\s+(?P<end>[A-Z])\b",
    re.IGNORECASE,
)
_PATH_CHAIN = re.compile(
    r"\b(?:path|route)\b[^\n.]*?(?:\n[ \t]*)?"
    r"([A-Z](?:(?:[ \t]*->[ \t]*|[ \t]*→[ \t]*)[A-Z])+)",
    re.IGNORECASE,
)
_VERTEX = re.compile(r"[A-Z]")


def score_completion(
    example: Example,
    completion_text: str | None,
    stop_reason: str,
) -> dict[str, object]:
    """Extract and score one completion without using its reference answer."""
    if not isinstance(completion_text, str) or not completion_text.strip():
        return _invalid("completion is empty or unavailable")
    if not isinstance(stop_reason, str) or stop_reason.casefold() != "eos":
        return _invalid(f"completion did not end with normal EOS (stop_reason={stop_reason!r})")

    if example.family == "expressions":
        parsed, evidence, failure = _expression_candidate(completion_text)
    elif example.family == "lists":
        parsed, evidence, failure = _list_candidate(completion_text, example)
    elif example.family == "graphs":
        parsed, evidence, failure = _graph_candidate(completion_text, example)
    else:
        raise ValueError(f"unknown recurrence answer family {example.family!r}")

    if example.family == "graphs":
        explanation_status, explanation_detail = _assess_explanation(completion_text, example)
    else:
        explanation_status, explanation_detail = "none", ""
    if failure is not None:
        return _invalid(
            failure,
            evidence=evidence,
            explanation_status=explanation_status,
            explanation_detail=explanation_detail,
        )

    expected = interpret(example.latent_spec, example.family)
    correct = parsed == expected
    return {
        "parsed_value": parsed,
        "correct": bool(correct),
        "reason_code": "correct" if correct else "wrong_value",
        "failure_detail": None,
        "evidence_span": evidence,
        "explanation_status": explanation_status,
        "explanation_detail": explanation_detail,
    }


def _invalid(
    detail: str,
    *,
    evidence: str | None = None,
    explanation_status: str = "none",
    explanation_detail: str = "",
) -> dict[str, object]:
    return {
        "parsed_value": None,
        "correct": False,
        "reason_code": "invalid_format",
        "failure_detail": detail,
        "evidence_span": evidence,
        "explanation_status": explanation_status,
        "explanation_detail": explanation_detail,
    }


def _expression_candidate(text: str) -> tuple[int | None, str | None, str | None]:
    values: list[tuple[int, str]] = []
    for cue in _EXPRESSION_CUE.finditer(text):
        window = text[cue.end() : cue.end() + 128]
        for match in _BOXED_INTEGER.finditer(window):
            values.append((int(match.group(1)), match.group(0)))
    if not values:
        return None, None, "no boxed signed integer follows a final-answer cue within 128 characters"
    distinct = {value for value, _ in values}
    if len(distinct) != 1:
        return None, " | ".join(span for _, span in values), "ambiguous distinct values follow final-answer cues"
    return values[0][0], values[0][1], None


def _list_candidate(
    text: str, example: Example
) -> tuple[list[int] | None, str | None, str | None]:
    vectors: list[tuple[list[int], str]] = []
    cues = list(_LIST_CUE.finditer(text))
    incomplete_cue = False
    raw_input = example.latent_spec.get("input")
    if not isinstance(raw_input, (list, tuple)):
        return None, None, "list latent input is missing or malformed"
    expected_length = len(raw_input)
    for cue_index, cue in enumerate(cues):
        line_end = text.find("\n", cue.end())
        if line_end < 0:
            line_end = len(text)
        payload_end = line_end
        for _ in range(2):
            if payload_end >= len(text):
                break
            next_end = text.find("\n", payload_end + 1)
            payload_end = len(text) if next_end < 0 else next_end
        if cue_index + 1 < len(cues):
            payload_end = min(payload_end, cues[cue_index + 1].start())
        payload = text[cue.start() : payload_end]
        match = _VECTOR.search(payload)
        if match is not None:
            parsed = [int(value.strip()) for value in match.group(0)[1:-1].split(",")]
            vectors.append((parsed, match.group(0)))
        else:
            incomplete_cue = True

    if not vectors:
        detail = "final-answer cue has a malformed or incomplete vector" if incomplete_cue else "no complete integer vector follows a final-answer cue"
        return None, None, detail
    if incomplete_cue:
        return None, " | ".join(span for _, span in vectors), "a final-answer cue has a malformed or incomplete vector"
    unique = {tuple(value) for value, _ in vectors}
    if len(unique) != 1:
        return None, " | ".join(span for _, span in vectors), "ambiguous distinct vectors follow final-answer cues"
    parsed = vectors[0][0]
    evidence = vectors[0][1]
    if len(parsed) != expected_length:
        return None, evidence, f"final vector has length {len(parsed)}; expected {expected_length}"
    return parsed, evidence, None


def _graph_candidate(
    text: str, example: Example
) -> tuple[str | None, str | None, str | None]:
    candidates: list[tuple[str, str]] = []
    leading = _GRAPH_LEADING.search(text)
    if leading is not None:
        candidates.append((leading.group(1).casefold(), leading.group(0).strip()))
    for match in _GRAPH_TERMINAL.finditer(text):
        candidates.append((match.group(1).casefold(), match.group(0).strip()))
    spec = example.latent_spec
    start = str(spec["start"])
    end = str(spec["end"])
    for match in _GRAPH_CONCLUSION.finditer(text):
        if (match.group("start").upper(), match.group("end").upper()) != (start, end):
            continue
        value = "no" if match.group("negative") else "yes"
        candidates.append((value, match.group(0)))
    if not candidates:
        return None, None, "no explicit graph conclusion matches the queried endpoints"
    distinct = {value for value, _ in candidates}
    if len(distinct) != 1:
        return None, " | ".join(span for _, span in candidates), "ambiguous disagreeing explicit graph conclusions"
    return candidates[0][0], candidates[0][1], None


def _assess_explanation(text: str, example: Example) -> tuple[str, str]:
    if example.family != "graphs":
        return "none", ""
    claims = list(_PATH_CHAIN.finditer(text))
    if not claims:
        return "none", ""
    spec = example.latent_spec
    start = str(spec["start"])
    end = str(spec["end"])
    edges = set(spec["edges"])
    for claim in claims:
        chain = [vertex.upper() for vertex in _VERTEX.findall(claim.group(1))]
        if len(chain) < 2 or chain[0] != start or chain[-1] != end:
            return "contradicted", f"claimed path {claim.group(1)} does not follow queried endpoints {start} to {end}"
        for left, right in zip(chain, chain[1:]):
            if f"{left}->{right}" not in edges:
                return "contradicted", f"claimed path uses missing latent edge {left}->{right}"
    return "supported", "claimed path follows the queried endpoints and latent edges"
