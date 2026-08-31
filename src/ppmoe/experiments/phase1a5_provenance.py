"""Immutable Phase 1A source verification for Phase 1A.5."""

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Callable, Mapping

from ppmoe.experiments.phase1a5_config import Phase1A5Config
from ppmoe.provenance import atomic_write_json, sha256_file


PINNED_STAGE_SHA256 = {
    "local": "90382EA91C74B1850FB7AEF81C6F4705DD625A8559EF0FA725BD4511FB2D8341",
    "rollout": "0F415AF16EBFFB6231041605C0A72C4A3505C6DB1F5FF832D6126B8B339E334E",
    "m0": "B9BE2ECFE2B73A906682B652D65CE1C9DB094E0D23ECC825AB00E677BD3A40D0",
}


class SourceVerificationError(RuntimeError):
    """Raised when the immutable Phase 1A source cannot be trusted."""


CodeRevisionProvider = Callable[[Path], Mapping[str, object]]


def _current_code_revision(_: Path) -> Mapping[str, object]:
    repository_root = Path(__file__).parents[3]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _required_hashes(source_root: Path, relative_paths: Mapping[str, Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, relative_path in relative_paths.items():
        path = source_root / relative_path
        if not path.is_file():
            raise SourceVerificationError(f"missing required Phase 1A source: {path}")
        result[name] = sha256_file(path)
    return result


def verify_phase1a_sources(
    source_root: Path,
    output_root: Path,
    config: Phase1A5Config,
    *,
    code_revision_provider: CodeRevisionProvider | None = None,
) -> Mapping[str, object]:
    """Verify immutable Phase 1A evidence and atomically persist its manifest."""
    resolved_source_root = source_root.resolve()
    stage_hashes = _required_hashes(
        resolved_source_root,
        {name: Path("stages") / f"{name}.json" for name in PINNED_STAGE_SHA256},
    )
    for stage, expected in PINNED_STAGE_SHA256.items():
        actual = stage_hashes[stage].upper()
        if actual != expected:
            raise SourceVerificationError(
                f"{stage} SHA-256 does not match locked Phase 1A source"
            )

    dataset_hashes = _required_hashes(
        resolved_source_root,
        {
            "calibration": Path("tokens/calibration/manifest.json"),
            "fineweb-heldout": Path("tokens/fineweb-heldout/manifest.json"),
            "wikitext": Path("tokens/wikitext/manifest.json"),
        },
    )
    model_hashes = _required_hashes(
        resolved_source_root,
        {
            "conversion": Path("manifests/conversion.json"),
            "model": Path("manifests/model.json"),
        },
    )
    revision = dict((code_revision_provider or _current_code_revision)(resolved_source_root))
    if not isinstance(revision.get("commit"), str) or not revision["commit"]:
        raise SourceVerificationError("code revision provider did not supply a commit")
    if revision.get("dirty") is not False:
        raise SourceVerificationError("dirty source tree is not permitted for production stages")

    manifest: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(resolved_source_root),
        "stage_sha256": {name: stage_hashes[name].upper() for name in PINNED_STAGE_SHA256},
        "dataset_manifest_sha256": dataset_hashes,
        "model_manifest_sha256": model_hashes,
        "code_revision": revision,
        "solver_version": config.solver_version,
    }
    atomic_write_json(output_root / "source-manifest.json", manifest)
    return manifest


def phase1a5_signature(
    config: Phase1A5Config,
    source_manifest: Mapping[str, object],
    solver_version: str,
    code_revision: str,
    microbatch_size: int,
) -> str:
    """Return the canonical signature that isolates non-interchangeable runs."""
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")
    payload = {
        "config": asdict(config),
        "source_manifest": source_manifest,
        "solver_version": solver_version,
        "code_revision": code_revision,
        "microbatch_size": microbatch_size,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: value.as_posix() if isinstance(value, Path) else None,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()
