"""Canonical identities for durable Task-9 failure events."""

from collections.abc import Mapping

from .provenance import sha256_payload

_TASK9_FAILURE_ID_FIELDS = (
    "kind",
    "stage",
    "arm",
    "seed",
    "attempt",
    "update",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_manifest_digest",
)


def task9_failure_event_id(event: Mapping[str, object]) -> str:
    """Return the immutable Task-9 identity digest for one failure event."""
    return sha256_payload({field: event.get(field) for field in _TASK9_FAILURE_ID_FIELDS})
