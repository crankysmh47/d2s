"""Gold-independent numeric answer extraction and separately invoked scoring."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re


@dataclass(frozen=True, slots=True)
class Extraction:
    status: str
    value: str | None
    evidence: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class Score:
    extraction: Extraction
    correct: bool


_CUE = re.compile(r"(?i)(?:\*\*\s*final\s+answer\s*:\s*\*\*|\*\*\s*final\s+answer\s*:|final\s+answer\s*:)")
_NUMBER = re.compile(
    r"(?P<currency>\$)?\s*(?P<number>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[+-]?\.\d+|[+-]?\d+\s*/\s*[+-]?\d+)"
    r"(?:\s*(?P<unit>%|[A-Za-z][A-Za-z0-9%°-]*(?:\s+[A-Za-z][A-Za-z0-9%°-]*)*))?"
)
_BOX = re.compile(r"^\\boxed\{(?P<inside>.*)\}$", re.DOTALL)
_TRAILING_PUNCTUATION = " .,!;:"
_UNITS = {
    "usd", "dollar", "dollars", "cent", "cents", "eur", "euro", "euros", "gbp", "pound", "pounds",
    "kg", "kgs", "kilogram", "kilograms", "g", "gram", "grams", "lb", "lbs", "pound", "pounds",
    "m", "meter", "meters", "metre", "metres", "km", "kilometer", "kilometers", "mile", "miles",
    "hour", "hours", "minute", "minutes", "second", "seconds", "day", "days", "year", "years",
    "person", "people", "item", "items", "unit", "units", "percent", "percentage", "percentages",
    "litre", "litres", "liter", "liters", "gallon", "gallons", "square feet", "square meters",
    "%",
    "per hour", "per day", "per month", "per year", "usd per hour", "dollars per hour",
}


def _invalid(reason: str, evidence: str | None = None) -> Extraction:
    return Extraction("invalid", None, evidence, reason)


def _fraction(number: str) -> Fraction | None:
    compact = number.replace(",", "").replace(" ", "")
    try:
        if "/" in compact:
            numerator, denominator = compact.split("/", 1)
            if int(denominator) == 0:
                return None
            return Fraction(int(numerator), int(denominator))
        return Fraction(Decimal(compact))
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return None


def _normalize(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _parse_payload(payload: str) -> tuple[str, Fraction] | None:
    candidate = payload.strip()
    if candidate.startswith("**") and candidate.endswith("**") and len(candidate) >= 4:
        candidate = candidate[2:-2].strip()
    elif candidate.endswith("**"):
        candidate = candidate[:-2].strip()
    if "\\boxed" in candidate:
        match = _BOX.fullmatch(candidate)
        if match is None or "{" in match.group("inside") or "}" in match.group("inside"):
            return None
        candidate = match.group("inside").strip()
    elif candidate.startswith("\\boxed{") or candidate.endswith("}"):
        return None
    candidate = candidate.rstrip(_TRAILING_PUNCTUATION).strip()
    match = _NUMBER.fullmatch(candidate)
    if match is None:
        return None
    if re.search(r"(?i)\b(?:actually|correction|instead|rather|the answer is|i meant)\b", match.group("unit") or ""):
        return None
    unit = " ".join((match.group("unit") or "").lower().split())
    if unit and unit not in _UNITS:
        return None
    number = match.group("number")
    parsed = _fraction(number)
    if parsed is None:
        return None
    # The regex only admits one scalar. Reject malformed numeric punctuation that
    # could otherwise be interpreted as an unknown unit token.
    if re.search(r"\d", match.group("unit") or ""):
        return None
    return _normalize(parsed), parsed


def extract_final_numeric(text: str, stop_reason: str) -> Extraction:
    """Extract a single final scalar using response text and stop status only."""
    if stop_reason.lower() not in {"eos", "eos_token"}:
        return _invalid("not_eos")
    cues = list(_CUE.finditer(text))
    if not cues:
        return _invalid("missing_final_cue")
    parsed: list[tuple[str, str, Fraction]] = []
    for index, cue in enumerate(cues):
        end = cues[index + 1].start() if index + 1 < len(cues) else len(text)
        evidence = text[cue.start():end].strip()
        payload = text[cue.end():end].strip()
        result = _parse_payload(payload)
        if result is None:
            return _invalid("invalid_final_value", evidence)
        normalized, exact = result
        parsed.append((normalized, evidence, exact))
    if any(item[2] != parsed[0][2] for item in parsed[1:]):
        return _invalid("conflicting_final_cues", " ".join(item[1] for item in parsed))
    return Extraction("valid", parsed[0][0], parsed[0][1], "ok")


def score_answer(extraction: Extraction, gold: Fraction) -> Score:
    """Compare an already extracted value with gold; never feeds gold to parsing."""
    if extraction.status != "valid" or extraction.value is None:
        return Score(extraction, False)
    try:
        exact = Fraction(extraction.value)
    except (ValueError, ZeroDivisionError):
        return Score(extraction, False)
    return Score(extraction, exact == gold)


__all__ = ["Extraction", "Score", "extract_final_numeric", "score_answer"]
