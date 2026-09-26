"""Frozen V2 answer scorer with conservative graph conclusion extraction."""

from __future__ import annotations

import re

from . import answer_v1
from .data import Example, interpret


_MARKDOWN = ("**", "__", "*", "_")
_STANDALONE = re.compile(r"^(yes|no)[.!]?$", re.IGNORECASE)
_ANSWER_CUE = re.compile(
    r"^(?:Answer|Final[ \t]+answer|So,[ \t]+the[ \t]+answer[ \t]+is):[ \t]*",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"^(yes|no)$", re.IGNORECASE)
_MARKED_VERTEX = r"(?:\*\*[A-Z]\*\*|__[A-Z]__|\*[A-Z]\*|_[A-Z]_|[A-Z])"
_GRAPH_CONCLUSION = re.compile(
    rf"\bthere\s+is\s+(?P<claim>[^.!?\n]+?)\s+from\s+"
    rf"(?P<start>{_MARKED_VERTEX})\s+to\s+(?P<end>{_MARKED_VERTEX})(?![A-Z])",
    re.IGNORECASE,
)
_GRAPH_CLAIM = re.compile(
    r"(?:(?P<negative>no)\s+)?(?P<article>a\s+)?"
    r"(?:(?:direct(?:ed)?(?:\s+or\s+indirect)?|indirect)\s+)?path\Z",
    re.IGNORECASE,
)


def score_completion_v2(
    example: Example,
    completion_text: str | None,
    stop_reason: str,
) -> dict[str, object]:
    """Extract and score one V2 completion without consulting its gold answer."""
    if example.family != "graphs":
        return answer_v1.score_completion(example, completion_text, stop_reason)
    if not isinstance(completion_text, str) or not completion_text.strip():
        return _invalid("completion is empty or unavailable")
    if not isinstance(stop_reason, str) or stop_reason.casefold() != "eos":
        return _invalid(f"completion did not end with normal EOS (stop_reason={stop_reason!r})")

    candidates = _graph_candidates(completion_text, example)
    parsed: str | None = None
    candidate_failure: str | None = None
    evidence: str | None = None
    if not candidates:
        candidate_failure = "no explicit graph conclusion matches the queried endpoints"
    else:
        values = {value for value, _span in candidates}
        evidence = " | ".join(span for _value, span in candidates)
        if len(values) != 1:
            candidate_failure = "ambiguous disagreeing explicit graph conclusions"
        else:
            parsed = candidates[0][0]

    explanation_status, explanation_detail = answer_v1._assess_explanation(completion_text, example)
    if candidate_failure is not None:
        return _invalid(
            candidate_failure,
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


def _graph_candidates(text: str, example: Example) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    lines = text.splitlines()
    first = next((line for line in lines if line.strip()), None)
    if first is not None:
        value = _standalone_value(first)
        if value is not None:
            candidates.append((value, first.strip()))

    for line in lines:
        value = _answer_cue_value(line)
        if value is not None:
            candidates.append((value, line.strip()))

    spec = example.latent_spec
    queried = str(spec["start"]).upper(), str(spec["end"]).upper()
    for match in _GRAPH_CONCLUSION.finditer(text):
        preceding_boundaries = [text.rfind(mark, 0, match.start()) for mark in ".!?\n"]
        sentence_start = max(preceding_boundaries) + 1
        if text[sentence_start : match.start()].strip():
            continue
        following_boundaries = [
            position
            for mark in ".!?\n"
            if (position := text.find(mark, match.end())) >= 0
        ]
        sentence_end = min(following_boundaries, default=len(text))
        if text[sentence_end : sentence_end + 1] == "?":
            continue
        if text[match.end() : sentence_end].strip():
            continue
        endpoints = _unmark(match.group("start")).upper(), _unmark(match.group("end")).upper()
        if endpoints != queried:
            continue
        claim = _unmark(match.group("claim").strip())
        claim_match = _GRAPH_CLAIM.fullmatch(claim)
        if claim_match is None:
            continue
        negative = claim_match.group("negative") is not None
        if negative and claim_match.group("article") is not None:
            continue
        candidates.append(("no" if negative else "yes", match.group(0)))
    return candidates


def _standalone_value(line: str) -> str | None:
    normalized = _remove_heading_and_outer_emphasis(line.strip())
    match = _STANDALONE.fullmatch(normalized)
    return match.group(1).casefold() if match is not None else None


def _answer_cue_value(line: str) -> str | None:
    normalized = _remove_heading_and_outer_emphasis(line.strip())
    match = _ANSWER_CUE.match(normalized)
    if match is None:
        return None
    payload = normalized[match.end() :].strip()
    payload = re.sub(r"[.!][ \t]*$", "", payload)
    payload = _unmark(payload)
    token = _TOKEN.fullmatch(payload)
    return token.group(1).casefold() if token is not None else None


def _remove_heading_and_outer_emphasis(line: str) -> str:
    line = re.sub(r"^#{1,6}[ \t]*", "", line)
    for marker in ("**", "__", "*", "_"):
        if line.startswith(marker) and line.endswith(marker) and len(line) > 2 * len(marker):
            return line[len(marker) : -len(marker)]
    return line


def _unmark(text: str) -> str:
    for marker in _MARKDOWN:
        emphasis = re.escape(marker)
        text = re.sub(
            rf"{emphasis}([A-Za-z]+(?:\s+[A-Za-z]+)*){emphasis}",
            r"\1",
            text,
        )
    return text


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
