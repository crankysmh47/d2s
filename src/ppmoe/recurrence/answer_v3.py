"""Frozen V3 final-answer grammar with the V2 graph scorer retained."""

from __future__ import annotations

import re
from . import answer_v2
from .data import Example, interpret


_EXPRESSION_PHRASES = (
    "So the final answer is",
    "So the final result is",
    "Final answer",
    "Final result",
    "Answer",
)
_LIST_PHRASES = (
    "So the final list after applying ADD1 once is",
    "So the final list is",
    "Final answer",
    "Final result",
    "Final list",
    "Answer",
)
_CUE_CACHE: dict[tuple[str, ...], re.Pattern[str]] = {}
_INTEGER = re.compile(r"[+-]?\d+\Z")
_VECTOR = re.compile(r"\[\s*[+-]?\d+(?:\s*,\s*[+-]?\d+)*\s*\]\Z")
_WRAPPERS = ("$$", "$", "**", "__")


def score_completion_v3(
    example: Example,
    completion_text: str | None,
    stop_reason: str,
) -> dict[str, object]:
    """Extract from response text, then compare with the objective interpreter."""
    if example.family == "graphs":
        return answer_v2.score_completion_v2(example, completion_text, stop_reason)
    if not isinstance(completion_text, str) or not completion_text.strip():
        return _invalid("completion is empty or unavailable")
    if not isinstance(stop_reason, str) or stop_reason.casefold() != "eos":
        return _invalid(f"completion did not end with normal EOS (stop_reason={stop_reason!r})")

    if example.family == "expressions":
        parsed, evidence, failure = _expression_candidate(completion_text)
    elif example.family == "lists":
        parsed, evidence, failure = _list_candidate(completion_text, example)
    else:
        raise ValueError(f"unknown recurrence answer family {example.family!r}")
    if failure is not None:
        return _invalid(failure, evidence=evidence)

    expected = interpret(example.latent_spec, example.family)
    correct = parsed == expected
    return {
        "parsed_value": parsed,
        "correct": bool(correct),
        "reason_code": "correct" if correct else "wrong_value",
        "failure_detail": None,
        "evidence_span": evidence,
        "explanation_status": "none",
        "explanation_detail": "",
    }


def _cue_pattern(phrases: tuple[str, ...]) -> re.Pattern[str]:
    cached = _CUE_CACHE.get(phrases)
    if cached is not None:
        return cached
    phrase_pattern = "|".join(re.escape(phrase) for phrase in phrases)
    decorated = rf"(?:\*\*(?:{phrase_pattern})\*\*|__(?:{phrase_pattern})__)"
    cue = re.compile(
        rf"^[ \t]*(?:\#{{1,6}}[ \t]+)?(?:{decorated}|(?:{phrase_pattern}))(?!\w)",
        re.IGNORECASE,
    )
    _CUE_CACHE[phrases] = cue
    return cue


def _visible_cues(text: str, phrases: tuple[str, ...]) -> list[tuple[int, int]]:
    """Return line-anchored cues outside quotations and fenced code."""
    cue_pattern = _cue_pattern(phrases)
    found: list[tuple[int, int]] = []
    offset = 0
    fence: tuple[str, int] | None = None
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        stripped = content.lstrip(" \t")
        # CommonMark fences allow at most three literal spaces of indentation;
        # tabs do not count as indentation here.
        fence_match = re.match(r"^( {0,3})(`{3,}|~{3,})(.*)\Z", content)
        if fence_match is not None:
            run = fence_match.group(2)
            marker = run[0]
            if fence is None:
                fence = (marker, len(run))
            elif (
                marker == fence[0]
                and len(run) >= fence[1]
                and not fence_match.group(3).strip(" \t")
            ):
                fence = None
            offset += len(line)
            continue
        if fence is None and not stripped.startswith(">"):
            match = cue_pattern.match(content)
            if match is not None:
                found.append((offset + match.start(), offset + match.end()))
        offset += len(line)
    return found


def _candidate_payloads(
    text: str, phrases: tuple[str, ...]
) -> list[tuple[str, str]]:
    cues = _visible_cues(text, phrases)
    payloads: list[tuple[str, str]] = []
    for index, (_start, end) in enumerate(cues):
        next_start = cues[index + 1][0] if index + 1 < len(cues) else len(text)
        payload_start = end
        if payload_start < next_start and text[payload_start : payload_start + 1] == ":":
            payload_start += 1
        while payload_start < next_start and text[payload_start] in " \t":
            payload_start += 1
        window_end = min(end + 256, next_start)
        payloads.append((text[payload_start:window_end], text[payload_start:next_start]))
    return payloads


def _expression_candidate(text: str) -> tuple[int | None, str | None, str | None]:
    payloads = _candidate_payloads(text, _EXPRESSION_PHRASES)
    if not payloads:
        return None, None, "no line-anchored final-answer cue"
    values: list[tuple[int, str]] = []
    for window, _unbounded in payloads:
        scalar = _parse_scalar(window)
        if scalar is None:
            return None, " | ".join(raw.strip() for raw, _ in payloads if raw.strip()) or None, "a final-answer cue has an incomplete or malformed scalar"
        value, evidence = scalar
        values.append((value, evidence))
    distinct = {value for value, _ in values}
    evidence = " | ".join(span for _, span in values)
    if len(distinct) != 1:
        return None, evidence, "ambiguous distinct values follow final-answer cues"
    return values[0][0], evidence, None


def _list_candidate(
    text: str, example: Example
) -> tuple[list[int] | None, str | None, str | None]:
    payloads = _candidate_payloads(text, _LIST_PHRASES)
    if not payloads:
        return None, None, "no line-anchored final-answer cue"
    vectors: list[tuple[list[int], str]] = []
    for window, _unbounded in payloads:
        vector = _parse_vector(window)
        if vector is None:
            return None, " | ".join(raw.strip() for raw, _ in payloads if raw.strip()) or None, "a final-answer cue has a malformed or incomplete vector"
        values, evidence = vector
        vectors.append((values, evidence))
    distinct = {tuple(values) for values, _ in vectors}
    evidence = " | ".join(span for _, span in vectors)
    if len(distinct) != 1:
        return None, evidence, "ambiguous distinct vectors follow final-answer cues"
    parsed = vectors[0][0]
    raw_input = example.latent_spec.get("input")
    if not isinstance(raw_input, (tuple, list)):
        return None, evidence, "list latent input is missing or malformed"
    expected_length = len(raw_input)
    if len(parsed) != expected_length:
        return None, evidence, f"final vector has length {len(parsed)}; expected {expected_length}"
    return parsed, evidence, None


def _parse_scalar(payload: str) -> tuple[int, str] | None:
    candidate = payload.strip()
    if candidate.endswith("."):
        candidate = candidate[:-1].rstrip()
    unwrapped = _unwrap_complete(candidate)
    if unwrapped is None or _INTEGER.fullmatch(unwrapped[0]) is None:
        return None
    try:
        value = int(unwrapped[0])
    except ValueError:
        return None
    return value, candidate


def _parse_vector(payload: str) -> tuple[list[int], str] | None:
    candidate = payload.strip()
    unwrapped = _unwrap_complete(candidate)
    if unwrapped is None or _VECTOR.fullmatch(unwrapped[0]) is None:
        return None
    values = [int(part.strip()) for part in unwrapped[0][1:-1].split(",")]
    return values, candidate


def _unwrap_complete(candidate: str) -> tuple[str, int] | None:
    """Remove at most two complete outer math, emphasis, or boxed wrappers."""
    value = candidate.strip()
    layers = 0
    while layers < 2:
        value = value.strip()
        if value.startswith(r"\boxed"):
            match = re.fullmatch(r"\\boxed\s*\{(.*)\}", value, re.DOTALL)
            if match is None:
                return None
            value = match.group(1)
            layers += 1
            continue
        wrapper = next(
            (marker for marker in _WRAPPERS if value.startswith(marker) and value.endswith(marker) and len(value) > 2 * len(marker)),
            None,
        )
        if wrapper is None:
            break
        value = value[len(wrapper) : -len(wrapper)]
        layers += 1
    value = value.strip()
    if not value or any(value.startswith(marker) or value.startswith(r"\boxed") for marker in _WRAPPERS):
        return None
    if value.endswith(tuple(marker for marker in _WRAPPERS)) or value.endswith("}"):
        return None
    return value, layers


def _invalid(detail: str, *, evidence: str | None = None) -> dict[str, object]:
    return {
        "parsed_value": None,
        "correct": False,
        "reason_code": "invalid_format",
        "failure_detail": detail,
        "evidence_span": evidence,
        "explanation_status": "none",
        "explanation_detail": "",
    }
