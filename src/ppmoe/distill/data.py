"""Pinned source authentication, deterministic partitions, and sealed JSONL data."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from hashlib import sha256
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping
import unicodedata
import urllib.request

from .config import DistillConfig, PartitionConfig


_SOURCE_HASHES = {
    "gsm8k_train": ("ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d", 2306545, 7473),
    "gsm8k_test": ("ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59", 419088, 1319),
    "svamp": ("5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e", 339970, 1000),
}
_OLD_DATA_MANIFEST_PATH = Path("artifacts/recurrence-v0/129d39646933/data/manifest.json")
_OLD_DATA_STAGE_PATH = Path("artifacts/recurrence-v0/129d39646933/stages/data.json")
_OLD_SOURCE_MANIFEST_PATH = Path("artifacts/recurrence-v0/129d39646933/source.json")
_STUDENT_TOKENIZER_PATH = Path(
    "artifacts/recurrence-v0/129d39646933/source/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/tokenizer.json"
)
_OLD_DATA_MANIFEST_SHA256 = "aafff1f7acffd594c5629fb2e2573a57c7ed3ae776aae61758a77a746a8c291d"
_OLD_DATA_STAGE_SHA256 = "8750d557bf3175d8893cad6758a134558defc5be1311cbb68d33940f4c56bfc4"
_OLD_SOURCE_MANIFEST_SHA256 = "5c595758fb082136841498eaad8c6fee5761c5ca5927a0975f1c1c775053f0e2"
_STUDENT_TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
_GSM_ANSWER = re.compile(r"####\s*([-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))\s*\Z", re.MULTILINE)
_ANNOTATION = re.compile(r"<<.*?>>")
_SEALED = {"sealed_test", "sealed_transfer"}
_SPLIT_NAMES = (
    "calibration", "pilot", "development", "pool", "sealed_test",
    "transfer_development", "sealed_transfer",
)


@dataclass(frozen=True, slots=True)
class Example:
    id: str
    dataset: str
    source_split: str
    question: str
    gold: str
    solution: str


@dataclass(frozen=True, slots=True)
class Exclusion:
    id: str
    dataset: str
    reason: str


@dataclass(frozen=True, slots=True)
class PartitionCounts:
    calibration: int = 60
    pilot: int = 120
    development: int = 200
    pool: int = 1200
    sealed_test: int = 400
    transfer_development: int = 200
    sealed_transfer: int = 500


@dataclass(frozen=True, slots=True)
class SplitManifest:
    splits: dict[str, tuple[Example, ...]]
    exclusions: tuple[Exclusion, ...]
    source_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    path: Path
    dataset: str
    revision: str
    license: str
    split: str | None
    sha256: str
    size_bytes: int
    row_count: int


def canonical_id(dataset: str, source_split: str, question: str) -> str:
    payload = dataset.encode() + b"\0" + source_split.encode() + b"\0" + question.encode("utf-8")
    return sha256(payload).hexdigest()


def normalized_question(question: str) -> str:
    lowered = question.lower()
    no_punctuation = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in lowered
    )
    return " ".join(no_punctuation.split())


def _order_key(record: Example) -> tuple[str, str]:
    return sha256(f"d2s-distill-v1|47|{record.id}".encode()).hexdigest(), record.id


def _fivegrams(value: str) -> set[str]:
    normalized = normalized_question(value)
    return {normalized[i : i + 5] for i in range(max(0, len(normalized) - 4))}


def _near(left: set[str], right: set[str]) -> bool:
    union = left | right
    return bool(union) and len(left & right) / len(union) >= 0.85


def fivegram_jaccard(left: str, right: str) -> float:
    a, b = _fivegrams(left), _fivegrams(right)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _as_counts(counts: PartitionCounts | PartitionConfig | None) -> PartitionCounts:
    if counts is None:
        return PartitionCounts()
    return PartitionCounts(
        calibration=counts.calibration if isinstance(counts, PartitionCounts) else counts.gsm_train.calibration,
        pilot=counts.pilot if isinstance(counts, PartitionCounts) else counts.gsm_train.pilot,
        development=counts.development if isinstance(counts, PartitionCounts) else counts.gsm_train.development,
        pool=counts.pool if isinstance(counts, PartitionCounts) else counts.gsm_train.pool,
        sealed_test=counts.sealed_test if isinstance(counts, PartitionCounts) else counts.gsm_test_sealed,
        transfer_development=counts.transfer_development if isinstance(counts, PartitionCounts) else counts.svamp_transfer_development,
        sealed_transfer=counts.sealed_transfer if isinstance(counts, PartitionCounts) else counts.svamp_sealed_transfer,
    )


def _deduplicate(records: Iterable[Example], *, cross_source: bool) -> tuple[list[Example], list[Exclusion]]:
    ordered = sorted(records, key=_order_key)
    seen_ids: dict[str, Example] = {}
    exact: dict[tuple[str, str], Example] = {}
    retained: list[Example] = []
    excluded: list[Exclusion] = []
    for record in ordered:
        if record.id != canonical_id(record.dataset, record.source_split, record.question):
            raise ValueError("record ID does not match its canonical question")
        if record.id in seen_ids:
            if seen_ids[record.id] != record:
                raise ValueError("duplicate canonical ID with conflicting row data")
            excluded.append(Exclusion(record.id, record.dataset, "same_source_exact_duplicate"))
            continue
        seen_ids[record.id] = record
        key = (record.dataset, normalized_question(record.question))
        if key in exact:
            excluded.append(Exclusion(record.id, record.dataset, "same_source_exact_duplicate"))
            continue
        exact[key] = record
        retained.append(record)
    return retained, excluded


def build_splits(
    gsm_train: Iterable[Example],
    gsm_test: Iterable[Example],
    svamp: Iterable[Example],
    *,
    counts: PartitionCounts | PartitionConfig | None = None,
) -> SplitManifest:
    """Deduplicate by priority and select deterministic prefixes.

    The near-duplicate candidate filter is exact-safe. For ngram-set sizes A and B,
    Jaccard >= 17/20 requires 20*min(A,B) >= 17*max(A,B). Also, if I is the
    intersection size, 37*I >= 17*(A+B); therefore a match can miss at most
    A-ceil(17*(A+B)/37) grams from the query. Selecting one more rare anchor than
    that maximum miss count guarantees every qualifying pair reaches the exact
    Jaccard check below. No approximate hashing or threshold relaxation is used.
    """
    sizes = _as_counts(counts)
    source_rows = {
        "gsm8k_train": list(gsm_train),
        "gsm8k_test": list(gsm_test),
        "svamp": list(svamp),
    }
    for name, rows in source_rows.items():
        if any(not isinstance(item, Example) for item in rows):
            raise TypeError(f"{name} contains an invalid record")
        expected_dataset, expected_split = {
            "gsm8k_train": ("gsm8k", "train"),
            "gsm8k_test": ("gsm8k", "test"),
            "svamp": ("svamp", "all"),
        }[name]
        if any(item.dataset != expected_dataset or item.source_split != expected_split for item in rows):
            raise ValueError(f"{name} has unexpected source metadata")

    exclusions: list[Exclusion] = []
    priority = ("gsm8k_test", "svamp", "gsm8k_train")
    cross_seen: list[tuple[Example, str, set[str]]] = []
    gram_index: dict[str, dict[int, list[int]]] = {}
    length_index: dict[int, list[int]] = {}
    normalized_index: dict[str, int] = {}
    clean: dict[str, list[Example]] = {}
    for source in priority:
        local, local_exclusions = _deduplicate(source_rows[source], cross_source=False)
        exclusions.extend(local_exclusions)
        accepted: list[Example] = []
        current_source: list[tuple[Example, str, set[str]]] = []
        for record in local:
            normalized = normalized_question(record.question)
            grams = _fivegrams(record.question)
            collision: str | None = None
            if normalized in normalized_index:
                collision = "cross_source_exact_duplicate"
            candidate_lengths = sorted(
                length for length in length_index
                if grams and 20 * min(len(grams), length) >= 17 * max(len(grams), length)
            )
            if collision is None and candidate_lengths:
                min_shared = min((17 * (len(grams) + length) + 36) // 37 for length in candidate_lengths)
                anchor_count = len(grams) - min_shared + 1
                rarest = sorted(
                    grams,
                    key=lambda gram: (
                        sum(len(gram_index.get(gram, {}).get(length, ())) for length in candidate_lengths),
                        gram,
                    ),
                )[:anchor_count]
                candidate_indexes: set[int] = set()
                for gram in rarest:
                    per_length = gram_index.get(gram, {})
                    for length in candidate_lengths:
                        candidate_indexes.update(per_length.get(length, ()))
                for candidate_index in sorted(candidate_indexes):
                    _, _, previous_grams = cross_seen[candidate_index]
                    if _near(grams, previous_grams):
                        collision = "cross_source_near_duplicate"
                        break
            if collision is not None:
                exclusions.append(Exclusion(record.id, record.dataset, collision))
            else:
                accepted.append(record)
                current_source.append((record, normalized, grams))
        # Cross-source near-duplicate rejection never compares examples within one source.
        for record, normalized, grams in current_source:
            previous_index = len(cross_seen)
            cross_seen.append((record, normalized, grams))
            normalized_index[normalized] = previous_index
            length_index.setdefault(len(grams), []).append(previous_index)
            for gram in grams:
                gram_index.setdefault(gram, {}).setdefault(len(grams), []).append(previous_index)
        clean[source] = accepted

    train_counts = (sizes.calibration, sizes.pilot, sizes.development, sizes.pool)
    required_train = sum(train_counts)
    if len(clean["gsm8k_train"]) < required_train:
        raise ValueError("insufficient deduplicated GSM8K train pool")
    if len(clean["gsm8k_test"]) < sizes.sealed_test:
        raise ValueError("insufficient deduplicated GSM8K test pool")
    if len(clean["svamp"]) < sizes.transfer_development + sizes.sealed_transfer:
        raise ValueError("insufficient deduplicated SVAMP pool")

    train = clean["gsm8k_train"]
    test = clean["gsm8k_test"]
    transfer = clean["svamp"]
    i = 0
    slices: dict[str, tuple[Example, ...]] = {}
    for name, count in zip(("calibration", "pilot", "development", "pool"), train_counts, strict=True):
        slices[name] = tuple(train[i : i + count])
        i += count
    slices["sealed_test"] = tuple(test[: sizes.sealed_test])
    slices["transfer_development"] = tuple(transfer[: sizes.transfer_development])
    slices["sealed_transfer"] = tuple(transfer[sizes.transfer_development : sizes.transfer_development + sizes.sealed_transfer])
    ids = [item.id for rows in slices.values() for item in rows]
    normalized = [normalized_question(item.question) for rows in slices.values() for item in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate record ID across selected splits")
    if len(set(normalized)) != len(normalized):
        raise ValueError("normalized exact question overlap across selected splits")
    return SplitManifest(
        splits=slices,
        exclusions=tuple(exclusions),
        source_counts={key: len(rows) for key, rows in source_rows.items()},
    )


def validate_source_bytes(path: Path, pin: Mapping[str, Any]) -> dict[str, Any]:
    actual = sha256_file(path)
    size = Path(path).stat().st_size
    if actual != pin["sha256"]:
        raise ValueError(f"source SHA-256 mismatch for {Path(path).name}")
    if size != pin["size_bytes"]:
        raise ValueError(f"source size mismatch for {Path(path).name}")
    return {"sha256": actual, "size_bytes": size}


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cached_or_download_gsm(config: DistillConfig, repo_root: Path) -> tuple[Path, Path]:
    cache = repo_root / config.data_cache
    revision = config.gsm8k.revision
    snapshot = cache / "datasets--openai--gsm8k" / "snapshots" / revision / "main"
    train_path = snapshot / Path(config.gsm8k.train_file or "").name
    test_path = snapshot / Path(config.gsm8k.test_file or "").name
    if train_path.is_file() and test_path.is_file():
        return train_path, test_path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to fetch the pinned GSM8K parquet files") from exc
    cache.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename in (config.gsm8k.train_file, config.gsm8k.test_file):
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                result = hf_hub_download(
                    repo_id=config.gsm8k.repo_id,
                    repo_type="dataset",
                    filename=filename,
                    revision=revision,
                    cache_dir=str(cache / "huggingface"),
                )
                paths.append(Path(result))
                break
            except Exception as exc:  # bounded retries for transient source retrieval only
                last_error = exc
                if attempt == 2:
                    raise RuntimeError("pinned GSM8K download failed after two retries") from last_error
                time.sleep(1 + attempt)
    return paths[0], paths[1]


def _download_svamp(config: DistillConfig, repo_root: Path) -> Path:
    cache = repo_root / config.data_cache
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "SVAMP.json"
    pin = _SOURCE_HASHES["svamp"]
    if path.exists():
        validate_source_bytes(path, {"sha256": pin[0], "size_bytes": pin[1]})
        return path
    last_error: Exception | None = None
    for attempt in range(3):
        temporary = path.with_suffix(".json.part")
        try:
            request = urllib.request.Request(config.svamp.url or "", headers={"User-Agent": "ppmoe-distillation-v1"})
            with urllib.request.urlopen(request, timeout=30) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            validate_source_bytes(temporary, {"sha256": pin[0], "size_bytes": pin[1]})
            temporary.replace(path)
            return path
        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise RuntimeError("pinned SVAMP download failed after two retries") from last_error
            time.sleep(1 + attempt)
    raise RuntimeError("unreachable source download state")


def _fraction_string(value: str) -> str:
    value = value.replace(",", "").strip()
    try:
        decimal = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("answer is not a decimal numeric value") from exc
    if not decimal.is_finite():
        raise ValueError("answer must be finite")
    fraction = Fraction(decimal)
    return str(fraction.numerator) if fraction.denominator == 1 else f"{fraction.numerator}/{fraction.denominator}"


def _gsm_records(path: Path, split: str) -> list[Example]:
    import pyarrow.parquet as parquet

    table = parquet.read_table(path)
    if set(table.column_names) != {"question", "answer"}:
        raise ValueError("GSM8K schema drift")
    records: list[Example] = []
    for row in table.to_pylist():
        question, answer = row["question"], row["answer"]
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError("GSM8K value schema drift")
        found = list(_GSM_ANSWER.finditer(answer))
        if not found:
            raise ValueError("GSM8K answer has no final #### numeric value")
        gold = _fraction_string(found[-1].group(1))
        solution = _ANNOTATION.sub("", answer).strip()
        solution = re.sub(r"####\s*[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)\s*\Z", f"Final answer: {gold}", solution)
        records.append(Example(canonical_id("gsm8k", split, question), "gsm8k", split, question, gold, solution))
    return records


def _svamp_records(path: Path) -> list[Example]:
    payload = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
    if not isinstance(payload, list) or (payload and set(payload[0]) != {"Body", "Question", "Answer", "Equation", "Type", "ID"}):
        raise ValueError("SVAMP schema drift")
    records: list[Example] = []
    for row in payload:
        if set(row) != {"Body", "Question", "Answer", "Equation", "Type", "ID"}:
            raise ValueError("SVAMP row schema drift")
        if not isinstance(row["Body"], str) or not isinstance(row["Question"], str):
            raise ValueError("SVAMP question schema drift")
        question = f"{row['Body']} {row['Question']}"
        answer = row["Answer"]
        if isinstance(answer, bool) or not isinstance(answer, (str, int, float, Decimal)):
            raise ValueError("SVAMP answer schema drift")
        gold = _fraction_string(str(answer))
        if not isinstance(row["Equation"], str):
            raise ValueError("SVAMP equation schema drift")
        records.append(Example(canonical_id("svamp", "all", question), "svamp", "all", question, gold, row["Equation"]))
    return records


def authenticate_sources(config: DistillConfig, repo_root: Path) -> tuple[dict[str, SourceArtifact], list[Example], list[Example], list[Example], dict[str, Any]]:
    """Fetch only the pinned source bytes and authenticate schemas/counts and retention provenance."""
    train_path, test_path = _cached_or_download_gsm(config, repo_root)
    svamp_path = _download_svamp(config, repo_root)
    artifacts: dict[str, SourceArtifact] = {}
    parsed: dict[str, list[Example]] = {}
    for key, path, dataset, revision, license_name, split in (
        ("gsm8k_train", train_path, config.gsm8k.repo_id, config.gsm8k.revision, config.gsm8k.license, "train"),
        ("gsm8k_test", test_path, config.gsm8k.repo_id, config.gsm8k.revision, config.gsm8k.license, "test"),
        ("svamp", svamp_path, config.svamp.repo_id, config.svamp.revision, config.svamp.license, None),
    ):
        sha, size, expected_rows = _SOURCE_HASHES[key]
        validate_source_bytes(path, {"sha256": sha, "size_bytes": size})
        rows = _gsm_records(path, split) if key.startswith("gsm8k") else _svamp_records(path)
        if len(rows) != expected_rows:
            raise ValueError(f"{key} count drift: expected {expected_rows}")
        artifacts[key] = SourceArtifact(path, dataset, revision, license_name, split, sha, size, len(rows))
        parsed[key] = rows
    retention = authenticate_retention(config, repo_root)
    return artifacts, parsed["gsm8k_train"], parsed["gsm8k_test"], parsed["svamp"], retention


def authenticate_retention(config: DistillConfig, repo_root: Path) -> dict[str, Any]:
    manifest_path = repo_root / config.retention_manifest
    chunks_path = repo_root / config.retention_chunks
    validate_source_bytes(
        manifest_path,
        {"sha256": config.retention_manifest_sha256, "size_bytes": manifest_path.stat().st_size},
    )
    validate_source_bytes(
        chunks_path,
        {"sha256": config.retention_chunks_sha256, "size_bytes": chunks_path.stat().st_size},
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("chunk_count") != 40 or manifest.get("scored_tokens") != 19960:
        raise ValueError("retention count drift")
    if manifest.get("chunks_sha256") != config.retention_chunks_sha256:
        raise ValueError("retention chunk hash drift")
    old_data_manifest_path = repo_root / _OLD_DATA_MANIFEST_PATH
    stage_path = repo_root / _OLD_DATA_STAGE_PATH
    source_manifest_path = repo_root / _OLD_SOURCE_MANIFEST_PATH
    tokenizer_path = repo_root / _STUDENT_TOKENIZER_PATH
    provenance_pins = (
        ("prior data manifest", old_data_manifest_path, _OLD_DATA_MANIFEST_SHA256),
        ("prior data stage", stage_path, _OLD_DATA_STAGE_SHA256),
        ("prior source manifest", source_manifest_path, _OLD_SOURCE_MANIFEST_SHA256),
        ("student tokenizer", tokenizer_path, _STUDENT_TOKENIZER_SHA256),
    )
    for name, path, expected_sha in provenance_pins:
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"pinned retention provenance hash mismatch: {name}")
    old_data_manifest = json.loads(old_data_manifest_path.read_text(encoding="utf-8"))
    if old_data_manifest.get("tokenizer", {}).get("revision") != config.student.revision:
        raise ValueError("retention tokenizer revision mismatch")
    old_data_stage = json.loads(stage_path.read_text(encoding="utf-8"))
    if old_data_stage.get("status") != "passed" or not old_data_stage.get("signatures"):
        raise ValueError("retention data-stage provenance is incomplete")
    stage_payload = old_data_stage.get("payload", {})
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    tokenizer_files = {
        entry["relative_path"]: entry["sha256"]
        for entry in source_manifest.get("files", [])
        if isinstance(entry, dict) and "relative_path" in entry and "sha256" in entry
    }
    tokenizer_sha = old_data_manifest["tokenizer"].get("sha256")
    if (
        tokenizer_sha != _STUDENT_TOKENIZER_SHA256
        or tokenizer_files.get("tokenizer.json") != _STUDENT_TOKENIZER_SHA256
    ):
        raise ValueError("retention tokenizer SHA mismatch")
    data_manifest_sha = sha256_file(old_data_manifest_path)
    if stage_payload.get("manifest_sha256") != data_manifest_sha:
        raise ValueError("retention data-stage manifest hash mismatch")
    if stage_payload.get("retention_manifest_sha256") != config.retention_manifest_sha256:
        raise ValueError("retention data-stage manifest signature mismatch")
    if stage_payload.get("retention_chunks_sha256") != config.retention_chunks_sha256:
        raise ValueError("retention data-stage chunk signature mismatch")
    data_hash = stage_payload.get("data_hash")
    if not isinstance(data_hash, str) or old_data_stage["signatures"].get("data") != data_hash:
        raise ValueError("retention data-stage hash signature mismatch")
    # Authenticate chunk syntax/count without exposing or decoding held-out benchmark data.
    chunk_lines = chunks_path.read_text(encoding="utf-8").splitlines()
    if len(chunk_lines) != 40 or any(not line for line in chunk_lines):
        raise ValueError("retention chunk count drift")
    total_tokens = 0
    scored_tokens = 0
    for index, line in enumerate(chunk_lines):
        chunk = json.loads(line)
        ids = chunk.get("input_ids")
        expected_length = 512 if index < 39 else 32
        if not isinstance(ids, list) or len(ids) != expected_length or any(not isinstance(token, int) for token in ids):
            raise ValueError("retention token chunk schema drift")
        total_tokens += len(ids)
        if not isinstance(chunk.get("scored_tokens"), int) or isinstance(chunk["scored_tokens"], bool):
            raise ValueError("retention scored-token schema drift")
        scored_tokens += chunk["scored_tokens"]
    if total_tokens != manifest.get("valid_source_tokens") or scored_tokens != manifest["scored_tokens"]:
        raise ValueError("retention token totals drift")
    return {
        "manifest_sha256": config.retention_manifest_sha256,
        "chunks_sha256": config.retention_chunks_sha256,
        "chunk_count": manifest["chunk_count"],
        "valid_source_tokens": manifest["valid_source_tokens"],
        "scored_tokens": manifest["scored_tokens"],
        "tokenizer_revision": old_data_manifest["tokenizer"]["revision"],
        "tokenizer_sha256": old_data_manifest["tokenizer"]["sha256"],
        "data_manifest_sha256": data_manifest_sha,
        "data_stage_sha256": sha256_file(stage_path),
        "data_stage_signatures": old_data_stage["signatures"],
    }


def _record_json(record: Example) -> dict[str, str]:
    return {
        "id": record.id, "dataset": record.dataset, "source_split": record.source_split,
        "question": record.question, "gold": record.gold, "solution": record.solution,
    }


def _manifest_jsonl(run_root: Path, name: str, rows: tuple[Example, ...]) -> dict[str, Any]:
    relative = Path("sealed" if name in _SEALED else "data") / f"{name}.jsonl"
    path = run_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        for row in rows:
            stream.write(json.dumps(_record_json(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
    return {
        "path": relative.as_posix(),
        "sha256": sha256_file(path),
        "count": len(rows),
        "ordered_ids": [row.id for row in rows],
    }


def load_split(run_root: Path, split: str, allow_sealed: bool = False) -> list[dict[str, str]]:
    """Load questions only; sealed split access is denied without stage authorization."""
    if split not in _SPLIT_NAMES:
        raise ValueError(f"unknown split {split}")
    if split in _SEALED and not allow_sealed:
        raise PermissionError(f"sealed split {split} requires authorized stage")
    relative = Path("sealed" if split in _SEALED else "data") / f"{split}.jsonl"
    path = Path(run_root) / relative
    return [
        {key: value for key, value in item.items() if key in {"id", "dataset", "source_split", "question"}}
        for item in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    ]


def prepare_splits(config: DistillConfig, repo_root: Path) -> tuple[dict[str, Any], SplitManifest]:
    artifacts, train, test, svamp, retention = authenticate_sources(config, repo_root)
    manifest = build_splits(train, test, svamp, counts=config.partition)
    exclusions = [{"id": item.id, "dataset": item.dataset, "reason": item.reason} for item in manifest.exclusions]
    exclusion_counts: dict[str, int] = {}
    for item in manifest.exclusions:
        exclusion_counts[item.reason] = exclusion_counts.get(item.reason, 0) + 1
    return {
        "sources": {key: {
            "repo_id": item.dataset, "revision": item.revision, "license": item.license,
            "split": item.split, "path": str(item.path.relative_to(repo_root)),
            "sha256": item.sha256, "size_bytes": item.size_bytes, "row_count": item.row_count,
        } for key, item in artifacts.items()},
        "retention": retention,
        "source_counts": manifest.source_counts,
        "excluded_count": len(exclusions),
        "exclusion_counts": exclusion_counts,
        "exclusions": exclusions,
    }, manifest


__all__ = [
    "Example", "Exclusion", "PartitionCounts", "SplitManifest", "SourceArtifact",
    "authenticate_retention", "authenticate_sources", "build_splits", "canonical_id",
    "fivegram_jaccard", "load_split", "normalized_question", "prepare_splits", "sha256_file", "validate_source_bytes",
]
