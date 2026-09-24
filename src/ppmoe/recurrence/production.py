"""Production workers for the registered recurrence experiment.

The module is deliberately import-light.  Model, tokenizer, and CUDA imports
remain inside stage entry points so CLI parsing and evidence inspection do not
load weights or initialize a device.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import TYPE_CHECKING

from .provenance import sha256_file, write_immutable_json

if TYPE_CHECKING:
    from .experiment import StageContext


_ARMS = ("a1", "a2", "a3")
_SEEDS = (47, 48)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


_PAYLOAD_KEYS = {
    "pilot": {
        "schema_version", "n3", "training_samples", "seconds_per_training_update",
        "seconds_per_generation_example", "seconds_per_retention_chunk",
        "longest_prompt_tokens", "generation_record_count", "retention_measurement_count",
        "disposable_relative_path", "initialization_hash", "initialization_sha256",
        "initialization_hashes", "forced_generation_seconds",
        "peak_allocated_bytes", "peak_reserved_bytes", "total_vram_bytes",
    },
    "frozen": {
        "schema_version", "decision", "summaries", "diagnostics", "peak_allocated_bytes",
        "peak_reserved_bytes", "total_vram_bytes",
    },
    "discover": {
        "schema_version", "training", "summaries", "accuracy", "retention", "timing",
        "decision", "selection_sha256", "data_manifest_sha256", "peak_allocated_bytes", "peak_reserved_bytes", "total_vram_bytes",
    },
    "confirm": {
        "schema_version", "training", "accuracy", "retention", "timing", "decision",
        "accuracy_bootstrap", "retention_bootstrap", "retention_summaries", "peak_allocated_bytes",
        "peak_reserved_bytes", "total_vram_bytes",
    },
}


def validate_stage_payload(
    stage: str, payload: Mapping[str, object], context: "StageContext" | None = None
) -> None:
    """Reject shape-valid-looking but incomplete production stage evidence."""
    expected = _PAYLOAD_KEYS.get(stage)
    if expected is None or set(payload) != expected or payload.get("schema_version") != 1:
        raise ValueError(f"{stage} production payload schema is invalid")
    for name in ("peak_allocated_bytes", "peak_reserved_bytes", "total_vram_bytes"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{stage} production payload {name} is invalid")
    if int(payload["total_vram_bytes"]) <= 0:
        raise ValueError(f"{stage} production payload total VRAM is invalid")
    if stage == "pilot":
        samples = payload.get("training_samples")
        rates = payload.get("seconds_per_training_update")
        identities = payload.get("initialization_hashes")
        file_hashes = payload.get("initialization_sha256")
        if (
            payload.get("generation_record_count") != 36
            or payload.get("retention_measurement_count") != 4
            or not isinstance(payload.get("n3"), int)
            or isinstance(payload.get("n3"), bool)
            or int(payload["n3"]) < 1000
            or not isinstance(samples, Mapping)
            or set(samples) != {"a1", "a2"}
            or any(not isinstance(values, list) or len(values) != 20 for values in samples.values())
            or not isinstance(rates, Mapping)
            or set(rates) != {"a1", "a2"}
            or any(_finite_nonnegative(value, "pilot training rate") <= 0 for value in rates.values())
            or not isinstance(identities, Mapping)
            or set(identities) != {"47", "48"}
            or not isinstance(file_hashes, Mapping)
            or set(file_hashes) != {"47", "48"}
            or any(_SHA256.fullmatch(str(value)) is None for value in [*identities.values(), *file_hashes.values()])
            or _finite_nonnegative(payload.get("seconds_per_generation_example"), "generation rate") <= 0
            or _finite_nonnegative(payload.get("forced_generation_seconds"), "forced generation") <= 0
            or _finite_nonnegative(payload.get("seconds_per_retention_chunk"), "retention rate") <= 0
        ):
            raise ValueError("pilot production payload timing schema is invalid")
        return
    decision = payload.get("decision")
    if (
        not isinstance(decision, Mapping)
        or decision.get("status") not in {"passed", "stopped_scientific", "retention_uncertain", "timing_inconclusive"}
        or not isinstance(decision.get("passed"), bool)
        or decision["passed"] is not (decision["status"] == "passed")
    ):
        raise ValueError(f"{stage} production payload decision is invalid")
    if stage == "frozen":
        summaries = payload.get("summaries")
        diagnostics = payload.get("diagnostics")
        if (
            not isinstance(summaries, Mapping)
            or set(summaries) != {"f0", "f1", "f2"}
            or any(not isinstance(value, Mapping) or not value for value in summaries.values())
            or not isinstance(diagnostics, Mapping)
            or set(diagnostics) != {"f1", "f2"}
            or any(not isinstance(value, Mapping) or not value for value in diagnostics.values())
        ):
            raise ValueError("frozen production payload summaries are invalid")
        for arm, value in summaries.items():
            exact_value = value.get("exact")
            retention_value = value.get("retention")
            if (
                not isinstance(exact_value, Mapping)
                or exact_value.get("record_count") != 600
                or not isinstance(exact_value.get("families"), Mapping)
                or set(exact_value["families"]) != {"expressions", "lists", "graphs"}
                or not isinstance(retention_value, Mapping)
                or retention_value.get("complete") is not True
                or isinstance(retention_value.get("record_count"), bool)
                or not isinstance(retention_value.get("record_count"), int)
                or int(retention_value["record_count"]) <= 0
                or not isinstance(retention_value.get("records_sha256"), str)
                or _SHA256.fullmatch(str(retention_value["records_sha256"])) is None
            ):
                raise ValueError(f"frozen production payload {arm} evaluation is incomplete")
        from .gates import classify_informative_families
        f0 = summaries["f0"]
        exact = f0.get("exact") if isinstance(f0, Mapping) else None
        if not isinstance(exact, Mapping) or exact.get("record_count") != 600:
            raise ValueError("frozen F0 summary is incomplete")
        expected_gates = classify_informative_families(exact)
        informative = expected_gates["passes"] is True
        expected_decision = {
            "status": "passed" if informative else "stopped_scientific",
            "reason": "informative" if informative else "uninformative_benchmark",
            "passed": informative,
            "gates": expected_gates,
        }
        if decision != expected_decision:
            raise ValueError("frozen production payload decision does not recompute")
    else:
        training = payload.get("training")
        if (
            not isinstance(training, Mapping)
            or set(training) != set(_ARMS)
            or any(not isinstance(value, Mapping) or not value for value in training.values())
        ):
            raise ValueError(f"{stage} production payload training matrix is invalid")
        expected_seed = 47 if stage == "discover" else 48
        for arm, value in training.items():
            expected_updates = (
                int(load_passed_pilot(context)["n3"])
                if arm == "a3" and context is not None
                else payload.get("training", {}).get("a3", {}).get("requested_updates")
                if arm == "a3" else 1000
            )
            if (
                isinstance(expected_updates, bool)
                or not isinstance(expected_updates, int)
                or expected_updates < 1000
                or value.get("arm") != arm
                or value.get("seed") != expected_seed
                or value.get("status") != "complete"
                or value.get("completed_updates") != expected_updates
                or value.get("requested_updates") != expected_updates
                or value.get("total_example_presentations") != int(expected_updates) * 8
                or value.get("unique_example_count") != 8000
                or value.get("signatures") is None
                or not isinstance(value.get("checkpoint_sha256"), str)
                or _SHA256.fullmatch(str(value["checkpoint_sha256"])) is None
                or not isinstance(value.get("initialization_hash"), str)
                or _SHA256.fullmatch(str(value["initialization_hash"])) is None
            ):
                raise ValueError(f"{stage} production payload {arm} training evidence is invalid")
        if stage == "discover":
            summaries = payload.get("summaries")
            if not isinstance(summaries, Mapping) or set(summaries) != set(_ARMS):
                raise ValueError("discover production payload summaries are incomplete")
            for arm, summary in summaries.items():
                families = summary.get("families") if isinstance(summary, Mapping) else None
                if (
                    not isinstance(summary, Mapping)
                    or summary.get("record_count") != 600
                    or not isinstance(families, Mapping)
                    or set(families) != {"expressions", "lists", "graphs"}
                ):
                    raise ValueError(f"discover production payload {arm} summary is incomplete")
        from .gates import confirmation_decision, discovery_decision
        recomputed = (
            discovery_decision(payload["accuracy"], payload["retention"], payload["timing"])
            if stage == "discover"
            else confirmation_decision(payload["accuracy"], payload["retention"], payload["timing"])
        )
        if decision != recomputed:
            raise ValueError(f"{stage} production payload decision does not recompute")
    if stage == "confirm":
        summaries = payload.get("retention_summaries")
        if not isinstance(summaries, Mapping) or set(summaries) != {"A1@48", "A2@48", "A3@48"}:
            raise ValueError("confirm production payload retention matrix is incomplete")
        for name, value in summaries.items():
            if (
                not isinstance(value, Mapping)
                or value.get("complete") is not True
                or value.get("failures") != 0
                or isinstance(value.get("record_count"), bool)
                or not isinstance(value.get("record_count"), int)
                or value.get("record_count") != value.get("expected_count")
                or int(value["record_count"]) <= 0
                or not isinstance(value.get("records_sha256"), str)
                or _SHA256.fullmatch(str(value["records_sha256"])) is None
                or value.get("arm") != name.split("@", 1)[0]
                or value.get("seed") != 48
                or value.get("split") != "test"
            ):
                raise ValueError(f"confirm retention summary {name} is invalid")
        for name, seed in (("accuracy_bootstrap", 9047), ("retention_bootstrap", 9157)):
            value = payload.get(name)
            if (
                not isinstance(value, Mapping)
                or value.get("seed") != seed
                or value.get("replicates") != 10000
                or not isinstance(value.get("comparisons"), Mapping)
            ):
                raise ValueError(f"confirm production payload {name} is invalid")
    if context is not None:
        _validate_persisted_stage_evidence(stage, payload, context)


def _validate_persisted_stage_evidence(
    stage: str, payload: Mapping[str, object], context: "StageContext"
) -> None:
    """Reconcile parent-visible payload claims to immutable raw producers."""
    from .experiment import _read_test_bytes_once

    def summary(
        relative: str, *, arm: str, seed: int, split: str, count: int
    ) -> tuple[Mapping[str, object], list[dict[str, object]]]:
        path = context.run_root / relative
        summary_path = path.with_name(f"{path.stem}.summary.json")
        value = json.loads(_read_test_bytes_once(summary_path, trusted_root=context.run_root))
        raw = _read_test_bytes_once(path, trusted_root=context.run_root)
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        if (
            not isinstance(value, Mapping)
            or value.get("complete") is not True
            or value.get("arm") != arm
            or value.get("seed") != seed
            or value.get("split") != split
            or value.get("record_count") != count
            or value.get("expected_count") != count
            or value.get("failures") != 0
            or value.get("signatures") != dict(context.signatures)
            or value.get("records_sha256") != hashlib.sha256(raw).hexdigest()
            or len(rows) != count
            or any(
                row.get("arm") != arm
                or row.get("seed") != seed
                or row.get("signatures") != dict(context.signatures)
                or row.get("split") != split
                for row in rows
            )
        ):
            raise ValueError(f"{relative} does not reconcile with raw evidence")
        key = "example_id" if relative.startswith("scores/") else "chunk_index"
        identifiers = [row.get(key) for row in rows]
        if len(set(identifiers)) != len(identifiers) or any(
            not isinstance(identifier, str if key == "example_id" else int)
            or isinstance(identifier, bool)
            for identifier in identifiers
        ):
            raise ValueError(f"{relative} has duplicate or invalid raw identifiers")
        return value, rows

    data_stage = json.loads(_read_test_bytes_once(
        context.run_root / "stages" / "data.json", trusted_root=context.run_root
    ))
    chunks = int(data_stage["payload"]["retention_chunk_count"])
    from .gates import paired_accuracy_bootstrap, paired_retention_bootstrap, summarize_exact_scores
    from .runtime_engine import _frozen_diagnostic, _slim_bootstrap
    if stage == "frozen":
        exact_rows: dict[str, list[dict[str, object]]] = {}
        retention_rows: dict[str, list[dict[str, object]]] = {}
        for arm in ("F0", "F1", "F2"):
            _, exact_rows[arm] = summary(
                f"scores/frozen-{arm.lower()}.jsonl", arm=arm, seed=47, split="dev", count=600
            )
            retention_value, retention_rows[arm] = summary(
                f"retention-scores/frozen-{arm.lower()}.jsonl", arm=arm, seed=47, split="dev", count=chunks
            )
            declared = payload["summaries"][arm.lower()]["retention"]
            if (
                not isinstance(declared, Mapping)
                or declared.get("records_sha256") != retention_value.get("records_sha256")
                or declared.get("record_count") != len(retention_rows[arm])
                or declared.get("nll_sum") != retention_value.get("nll_sum")
                or declared.get("scored_tokens") != retention_value.get("scored_tokens")
            ):
                raise ValueError(f"frozen {arm} retention summary contradicts its raw producer")
            if payload["summaries"][arm.lower()]["exact"] != summarize_exact_scores(exact_rows[arm]):
                raise ValueError(f"frozen {arm} exact summary contradicts its raw producer")
        for arm in ("F1", "F2"):
            declared_diagnostic = payload["diagnostics"][arm.lower()]
            recomputed = _frozen_diagnostic(
                exact_rows[arm], exact_rows["F0"], retention_rows[arm], retention_rows["F0"],
                declared_diagnostic.get("memory", {}),
            )
            if recomputed != declared_diagnostic:
                raise ValueError(f"frozen {arm} diagnostic contradicts raw evidence")
    elif stage == "discover":
        exact_rows = {
            "F0": summary(
                "scores/frozen-f0.jsonl", arm="F0", seed=47, split="dev", count=600
            )[1]
        }
        retention_f0 = summary(
            "retention-scores/frozen-f0.jsonl", arm="F0", seed=47, split="dev", count=chunks
        )[1]
        retention_a2: list[dict[str, object]] = []
        for arm in ("A1", "A2", "A3"):
            _, exact_rows[arm] = summary(
                f"scores/{arm.lower()}-47-dev.jsonl", arm=arm, seed=47, split="dev", count=600
            )
            _, rows = summary(
                f"retention-scores/{arm.lower()}-47-dev.jsonl", arm=arm, seed=47, split="dev", count=chunks
            )
            if arm == "A2":
                retention_a2 = rows
            if payload["summaries"][arm.lower()] != summarize_exact_scores(exact_rows[arm]):
                raise ValueError(f"discover {arm} summary contradicts its raw producer")
        records = {("F0", 47): exact_rows["F0"], **{(arm, 47): exact_rows[arm] for arm in ("A1", "A2", "A3")}}
        expected_accuracy = paired_accuracy_bootstrap(records)
        if payload["accuracy"] != _slim_bootstrap(expected_accuracy):
            raise ValueError("discover accuracy decision operands do not reconcile with raw scores")
        expected_retention = paired_retention_bootstrap(retention_a2, retention_f0)
        if payload["retention"] != _slim_bootstrap(expected_retention):
            raise ValueError("discover retention decision operands do not reconcile with raw chunks")
        if payload.get("decision", {}).get("status") == "passed":
            selection = _read_test_bytes_once(context.run_root / "selection.json", trusted_root=context.run_root)
            if payload.get("selection_sha256") != hashlib.sha256(selection).hexdigest():
                raise ValueError("discover selection digest does not match persisted selection")
    elif stage == "confirm":
        exact_rows = {
            ("F0", 47): summary(
                "scores/f0-test.jsonl", arm="F0", seed=47, split="test", count=2400
            )[1]
        }
        for seed in (47, 48):
            for arm in ("A1", "A2", "A3"):
                exact_rows[(arm, seed)] = summary(
                    f"scores/{arm.lower()}-{seed}-test.jsonl",
                    arm=arm, seed=seed, split="test", count=2400,
                )[1]
        retention_f0 = summary(
            "retention-scores/frozen-f0.jsonl", arm="F0", seed=47, split="dev", count=chunks
        )[1]
        retention_seed47_a2 = summary(
            "retention-scores/a2-47-dev.jsonl", arm="A2", seed=47, split="dev", count=chunks
        )[1]
        retention_by_seed: dict[int, list[dict[str, object]]] = {}
        for arm in ("A1", "A2", "A3"):
            value, rows = summary(
                f"retention-scores/{arm.lower()}-48-confirm.jsonl",
                arm=arm, seed=48, split="test", count=chunks,
            )
            declared = payload["retention_summaries"][f"{arm}@48"]
            if (
                declared.get("records_sha256") != value.get("records_sha256")
                or declared.get("record_count") != value.get("record_count")
                or declared.get("nll_sum") != value.get("nll_sum")
                or declared.get("scored_tokens") != value.get("scored_tokens")
                or declared.get("complete") is not True
            ):
                raise ValueError(f"confirm {arm} retention summary contradicts its raw producer")
            if arm == "A2":
                retention_by_seed[48] = rows
        accuracy_result = paired_accuracy_bootstrap(exact_rows)
        per_seed: dict[int, object] = {}
        for seed in (47, 48):
            seed_bootstrap = paired_accuracy_bootstrap({
                key: rows for key, rows in exact_rows.items()
                if key[1] == seed or key[0] == "F0"
            })
            per_seed[seed] = {
                **seed_bootstrap,
                **{
                    f"a2_family_deltas_vs_{reference}": seed_bootstrap["comparisons"][f"A2-{reference.upper()}"]["family_deltas"]
                    for reference in ("f0", "a1")
                },
            }
        accuracy_result["per_seed"] = per_seed
        accuracy_result["accuracy_ci_lows"] = {
            name: value["ci_low"] for name, value in accuracy_result["comparisons"].items()
        }
        if payload["accuracy"] != _slim_bootstrap(accuracy_result):
            raise ValueError("confirm accuracy decision operands do not reconcile with raw scores")
        retention_result = {
            47: paired_retention_bootstrap(retention_seed47_a2, retention_f0),
            48: paired_retention_bootstrap(retention_by_seed[48], retention_f0),
        }
        if payload["retention"] != _slim_bootstrap(retention_result):
            raise ValueError("confirm retention decision operands do not reconcile with raw chunks")
        for bootstrap_name in ("accuracy_bootstrap", "retention_bootstrap"):
            bootstrap = payload[bootstrap_name]
            matrix_path = context.run_root / str(bootstrap["paired_sample_indices_path"])
            matrix_bytes = _read_test_bytes_once(matrix_path, trusted_root=context.run_root)
            if hashlib.sha256(matrix_bytes).hexdigest() != bootstrap["paired_sample_indices_sha256"]:
                raise ValueError("bootstrap matrix digest mismatch")
            matrix = json.loads(matrix_bytes.decode("utf-8"))
            if bootstrap_name == "accuracy_bootstrap":
                reference = accuracy_result["comparisons"]["A2-F0"]
                first_rows = next(iter(exact_rows.values()))
                family_ids: dict[str, list[str]] = {}
                for row in first_rows:
                    family_ids.setdefault(str(row["family"]), []).append(str(row["example_id"]))
                expected_matrix = {"families": {
                    family: {"ids": sorted(ids), "indices": reference["sample_indices"][family]}
                    for family, ids in sorted(family_ids.items())
                }}
                expected_comparisons = accuracy_result["comparisons"]
            else:
                reference = retention_result[47]
                expected_matrix = {"families": {"retention": {
                    "ids": [str(row["chunk_index"]) for row in retention_seed47_a2],
                    "indices": reference["sample_indices"]["retention"],
                }}}
                expected_comparisons = {
                    f"A2-F0@{seed}": result for seed, result in retention_result.items()
                }
            if matrix != expected_matrix:
                raise ValueError("persisted paired bootstrap matrix contradicts raw evidence")
            for name, item in bootstrap["comparisons"].items():
                expected_result = expected_comparisons.get(name)
                if not isinstance(expected_result, Mapping):
                    raise ValueError("persisted bootstrap comparison is not registered")
                vector_bytes = _read_test_bytes_once(
                    context.run_root / str(item["vector_path"]), trusted_root=context.run_root
                )
                expected_vector = {
                    "comparison": name,
                    **({"seed": expected_result["seed"]} if bootstrap_name == "retention_bootstrap" else {}),
                    "values": expected_result["bootstrap"],
                }
                if (
                    hashlib.sha256(vector_bytes).hexdigest() != item["vector_sha256"]
                    or json.loads(vector_bytes.decode("utf-8")) != expected_vector
                    or item.get("point") != expected_result["point"]
                    or item.get("ci_low") != expected_result["ci_low"]
                    or item.get("ci_high") != expected_result["ci_high"]
                ):
                    raise ValueError("bootstrap vector digest mismatch")


def _adapter_shapes() -> dict[str, dict[str, object]]:
    """Return the frozen Qwen3-1.7B rank-8 adapter schema."""
    dimensions = {
        "self_attn.q_proj": (2048, 2048),
        "self_attn.v_proj": (1024, 2048),
        "self_attn.o_proj": (2048, 2048),
        "mlp.gate_proj": (6144, 2048),
        "mlp.up_proj": (6144, 2048),
        "mlp.down_proj": (2048, 6144),
    }
    parameters: dict[str, dict[str, object]] = {}
    for layer in range(12, 16):
        for projection, (output_features, input_features) in dimensions.items():
            prefix = f"stock_model.model.layers.{layer}.{projection}"
            parameters[f"{prefix}.a"] = {
                "shape": [8, input_features], "dtype": "torch.float32"
            }
            parameters[f"{prefix}.b"] = {
                "shape": [output_features, 8], "dtype": "torch.float32"
            }
    return parameters


def production_checkpoint_schema(
    run_root: Path,
    arm: str,
    seed: int,
    signatures: Mapping[str, str],
) -> Mapping[str, object]:
    """Bind retry validation to the registered shapes and immutable seed state."""
    from .experiment import StagePrerequisiteError, _read_test_bytes_once

    if arm not in _ARMS or seed not in _SEEDS:
        raise StagePrerequisiteError("checkpoint arm/seed is outside the registered experiment")
    if set(signatures) != {"source", "data", "config", "code"}:
        raise StagePrerequisiteError("checkpoint control signatures are incomplete")
    path = run_root / "initializations" / f"{seed}.pt"
    try:
        import io
        import torch

        payload = torch.load(
            io.BytesIO(_read_test_bytes_once(path, trusted_root=run_root)),
            map_location="cpu",
            weights_only=False,
        )
    except Exception as error:
        raise StagePrerequisiteError(f"seed-{seed} initialization is missing or unreadable") from error
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("seed") != seed
        or not isinstance(payload.get("initialization_hash"), str)
        or _SHA256.fullmatch(str(payload["initialization_hash"])) is None
    ):
        raise StagePrerequisiteError(f"seed-{seed} initialization identity is invalid")
    return {
        "parameters": _adapter_shapes(),
        "initialization_hash": str(payload["initialization_hash"]),
    }


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _load_pilot_projection(root: Path) -> Mapping[str, object] | None:
    path = root / "pilot.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise ValueError("pilot projection evidence is malformed")
    return value


def _load_current_pilot_for_postflight(context: "StageContext") -> Mapping[str, object]:
    """Authenticate G2 output before its own stage record can exist."""
    from .experiment import StagePrerequisiteError, _read_test_bytes_once

    try:
        value = json.loads(_read_test_bytes_once(
            context.run_root / "pilot.json", trusted_root=context.run_root
        ).decode("utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("pilot is not a mapping")
        validate_stage_payload("pilot", value)
        return dict(value)
    except Exception as error:
        raise StagePrerequisiteError("current pilot evidence is invalid") from error


def load_passed_pilot(context: "StageContext") -> Mapping[str, object]:
    """Load G2 evidence only when it exactly matches its passed stage record."""
    from .experiment import StagePrerequisiteError, _read_test_bytes_once

    try:
        pilot_bytes = _read_test_bytes_once(
            context.run_root / "pilot.json", trusted_root=context.run_root
        )
        stage_bytes = _read_test_bytes_once(
            context.run_root / "stages" / "pilot.json", trusted_root=context.run_root
        )
        pilot = json.loads(pilot_bytes.decode("utf-8"))
        stage = json.loads(stage_bytes.decode("utf-8"))
    except Exception as error:
        raise StagePrerequisiteError("passed pilot evidence is missing or unreadable") from error
    stage_payload = stage.get("payload") if isinstance(stage, Mapping) else None
    if (
        not isinstance(pilot, Mapping)
        or pilot.get("schema_version") != 1
        or not isinstance(stage, Mapping)
        or stage.get("stage") != "pilot"
        or stage.get("status") != "passed"
        or stage.get("signatures") != dict(context.signatures)
        or not isinstance(stage_payload, Mapping)
        or any(stage_payload.get(key) != value for key, value in pilot.items())
    ):
        raise StagePrerequisiteError("pilot evidence is not bound to its passed stage")
    n3 = pilot.get("n3")
    hashes = pilot.get("initialization_sha256")
    identities = pilot.get("initialization_hashes")
    if (
        isinstance(n3, bool) or not isinstance(n3, int) or n3 < 1000
        or not isinstance(hashes, Mapping) or set(hashes) != {"47", "48"}
        or not isinstance(identities, Mapping) or set(identities) != {"47", "48"}
        or any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in identities.values())
    ):
        raise StagePrerequisiteError("pilot n3/initialization evidence is invalid")
    for seed in (47, 48):
        expected = hashes.get(str(seed))
        path = context.run_root / "initializations" / f"{seed}.pt"
        if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
            raise StagePrerequisiteError("pilot initialization hash is invalid")
        try:
            actual = hashlib.sha256(
                _read_test_bytes_once(path, trusted_root=context.run_root)
            ).hexdigest()
        except Exception as error:
            raise StagePrerequisiteError(f"seed-{seed} initialization is missing") from error
        if actual != expected:
            raise StagePrerequisiteError(
                f"seed-{seed} initialization differs from passed pilot evidence"
            )
    return dict(pilot)


def production_remaining_work_projection(context: "StageContext") -> Mapping[str, float]:
    """Project every unfinished registered CUDA unit from measured pilot rates.

    Before G2 has measurements, the pilot itself is projected using a bounded
    conservative seed value rather than the unsafe historical all-zero value.
    After G2, every number comes from immutable ``pilot.json`` evidence.
    """
    pilot = None
    if (context.run_root / "pilot.json").is_file():
        pilot = (
            _load_current_pilot_for_postflight(context)
            if context.stage == "pilot" and not (context.run_root / "stages" / "pilot.json").is_file()
            else load_passed_pilot(context)
        )
    if pilot is None:
        if context.stage in {"source", "data", "smoke"}:
            # These prerequisite stages cannot start any G2-G5 work.  Their
            # own measured interval is charged independently; a production
            # projection becomes meaningful only once G1 has passed.
            return {
                "production_arms": 0.0,
                "longest_prompt_generation": 0.0,
                "retention_chunks": 0.0,
                "retry_allowance": 0.0,
                "report_cuda": 0.0,
            }
        smoke_path = context.run_root / "stages" / "smoke.json"
        try:
            smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
            smoke_payload = smoke.get("payload") if isinstance(smoke, Mapping) else None
            measured = _finite_nonnegative(
                smoke_payload.get("cuda_seconds") if isinstance(smoke_payload, Mapping) else None,
                "passed smoke CUDA seconds",
            )
            if smoke.get("status") != "passed" or smoke.get("signatures") != dict(context.signatures) or measured <= 0:
                raise ValueError("passed smoke evidence is invalid")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                "pre-pilot projection requires passed measured smoke evidence"
            ) from error
        # Treat the complete measured smoke interval as the upper bound for
        # every pilot unit.  This is deliberately pessimistic but empirical:
        # 50 timed optimizer steps (warmups included), 36 sampled generations
        # plus one forced 32-token longest-prompt generation, and four chunks.
        training = 50 * measured
        generation = 37 * measured
        retention = 4 * measured
        remaining = training + generation + retention
        return {
            "production_arms": training,
            "longest_prompt_generation": generation,
            "retention_chunks": retention,
            "retry_allowance": remaining,
            "report_cuda": 0.0,
        }
    rates = pilot.get("seconds_per_training_update")
    if not isinstance(rates, Mapping) or set(rates) != {"a1", "a2"}:
        raise ValueError("pilot training-rate evidence is malformed")
    a1 = _finite_nonnegative(rates["a1"], "A1 update seconds")
    a2 = _finite_nonnegative(rates["a2"], "A2 update seconds")
    generation = _finite_nonnegative(
        pilot.get("seconds_per_generation_example"), "generation example seconds"
    )
    retention = _finite_nonnegative(
        pilot.get("seconds_per_retention_chunk"), "retention chunk seconds"
    )
    n3 = pilot.get("n3")
    if isinstance(n3, bool) or not isinstance(n3, int) or n3 < 1000:
        raise ValueError("pilot n3 is invalid")

    production_seconds = 0.0
    unfinished = 0
    identities = pilot["initialization_hashes"]
    for seed in _SEEDS:
        for arm in _ARMS:
            manifest = context.run_root / "training" / f"{arm}-{seed}.json"
            if manifest.is_file():
                value = _load_training_manifest(context.run_root, arm, seed, int(n3))
                checkpoint = context.run_root / "checkpoints" / arm / str(seed) / "latest.pt"
                if (
                    value.get("signatures") != dict(context.signatures)
                    or value.get("initialization_hash") != identities[str(seed)]
                    or not checkpoint.is_file()
                    or sha256_file(checkpoint) != value.get("checkpoint_sha256")
                ):
                    raise ValueError("completed training evidence is not bound to pilot/checkpoint state")
                continue
            unfinished += 1
            updates = n3 if arm == "a3" else 1000
            production_seconds += updates * (a2 if arm == "a2" else a1)

    stage_records = context.run_root / "stages"
    frozen_done = (stage_records / "frozen.json").is_file()
    discover_done = (stage_records / "discover.json").is_file()
    confirm_done = (stage_records / "confirm.json").is_file()
    generation_examples = (0 if frozen_done else 1800) + (0 if discover_done else 1800) + (0 if confirm_done else 16800)
    retention_chunks = (0 if frozen_done else 120) + (0 if discover_done else 120) + (0 if confirm_done else 120)
    measured_remaining = production_seconds + generation_examples * generation + retention_chunks * retention
    return {
        "production_arms": production_seconds,
        "longest_prompt_generation": generation_examples * generation,
        "retention_chunks": retention_chunks * retention,
        "retry_allowance": measured_remaining if unfinished or generation_examples or retention_chunks else 0.0,
        "report_cuda": 0.0,
    }


def _load_training_manifest(root: Path, arm: str, seed: int, n3: int) -> Mapping[str, object]:
    from .experiment import StagePrerequisiteError, _read_test_bytes_once

    path = root / "training" / f"{arm}-{seed}.json"
    try:
        raw = _read_test_bytes_once(path, trusted_root=root)
        value = json.loads(raw.decode("utf-8"))
    except Exception as error:
        raise StagePrerequisiteError(
            f"all six final production checkpoints are required; missing {arm.upper()} seed {seed}"
        ) from error
    expected_updates = n3 if arm == "a3" else 1000
    checkpoint_hashes = value.get("checkpoint_hashes") if isinstance(value, Mapping) else None
    attempts = value.get("attempts") if isinstance(value, Mapping) else None
    failures = value.get("failures") if isinstance(value, Mapping) else None
    event_indices = value.get("event_indices") if isinstance(value, Mapping) else None
    failure_indices = value.get("failure_event_indices") if isinstance(value, Mapping) else None
    saved_signatures = value.get("signatures") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != 1
        or value.get("status") != "complete"
        or value.get("arm") != arm
        or value.get("seed") != seed
        or not isinstance(saved_signatures, Mapping)
        or set(saved_signatures) != {"source", "data", "config", "code"}
        or any(not isinstance(item, str) or _SHA256.fullmatch(item) is None for item in saved_signatures.values())
        or value.get("completed_updates") != expected_updates
        or value.get("requested_updates") != expected_updates
        or value.get("total_example_presentations") != expected_updates * 8
        or value.get("cursor") != expected_updates * 8
        or value.get("unique_example_count") != 8000
        or not isinstance(value.get("ordered_id_sha256"), str)
        or _SHA256.fullmatch(str(value["ordered_id_sha256"])) is None
        or not isinstance(value.get("initialization_hash"), str)
        or _SHA256.fullmatch(str(value["initialization_hash"])) is None
        or not isinstance(value.get("final_adapter_hash"), str)
        or _SHA256.fullmatch(str(value["final_adapter_hash"])) is None
        or value.get("trainable_parameter_count") != 1_146_880
        or value.get("completion_status") != "completed"
        or value.get("determinism_notes") != []
        or not isinstance(checkpoint_hashes, list)
        or not checkpoint_hashes
        or any(not isinstance(item, str) or _SHA256.fullmatch(item) is None for item in checkpoint_hashes)
        or checkpoint_hashes[-1] != value.get("checkpoint_sha256")
        or not isinstance(attempts, list)
        or not attempts
        or any(isinstance(item, bool) or not isinstance(item, int) or item not in {1, 2} for item in attempts)
        or attempts != sorted(set(attempts))
        or not isinstance(failures, list)
        or not isinstance(event_indices, list)
        or not event_indices
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in event_indices)
        or not isinstance(failure_indices, list)
        or any(item not in event_indices for item in failure_indices)
        or len(failure_indices) != len(failures)
        or isinstance(value.get("peak_allocated_bytes"), bool)
        or not isinstance(value.get("peak_allocated_bytes"), int)
        or int(value["peak_allocated_bytes"]) < 0
        or isinstance(value.get("peak_reserved_bytes"), bool)
        or not isinstance(value.get("peak_reserved_bytes"), int)
        or int(value["peak_reserved_bytes"]) < int(value["peak_allocated_bytes"])
        or isinstance(value.get("synchronized_cuda_seconds"), bool)
        or not isinstance(value.get("synchronized_cuda_seconds"), (int, float))
        or not math.isfinite(float(value["synchronized_cuda_seconds"]))
        or float(value["synchronized_cuda_seconds"]) <= 0
        or not isinstance(value.get("checkpoint_sha256"), str)
        or _SHA256.fullmatch(str(value["checkpoint_sha256"])) is None
    ):
        raise StagePrerequisiteError(
            f"all six final production checkpoints are required; invalid {arm.upper()} seed {seed}"
        )
    return value


def _validate_training_events(root: Path, manifest: Mapping[str, object]) -> None:
    from .experiment import StagePrerequisiteError, _read_test_bytes_once
    from .event_identity import task9_failure_event_id

    try:
        events = [json.loads(line) for line in _read_test_bytes_once(
            root / "events.jsonl", trusted_root=root
        ).decode("utf-8").splitlines()]
    except Exception as error:
        raise StagePrerequisiteError("training event history is missing") from error
    by_index = {
        event.get("index"): event for event in events
        if isinstance(event, Mapping) and isinstance(event.get("index"), int)
    }
    indices = manifest["event_indices"]
    failure_indices = manifest["failure_event_indices"]
    if (
        len(by_index) != len(events)
        or len(indices) != len(set(indices))
        or any(index not in by_index for index in indices)
        or any(
            by_index[index].get("arm") != manifest["arm"]
            or by_index[index].get("seed") != manifest["seed"]
            for index in indices
        )
    ):
        raise StagePrerequisiteError("training manifest event indices do not reconcile")
    linked = [by_index[index] for index in indices]
    linked_failures = [event for event in linked if event.get("kind") == "training_failure"]
    manifest_failures = manifest.get("failures")
    if (
        not isinstance(manifest_failures, list)
        or [event.get("index") for event in linked_failures] != failure_indices
        or manifest_failures != linked_failures
    ):
        raise StagePrerequisiteError("training manifest failures do not reconcile with events")
    completed = [event for event in linked if event.get("kind") == "training_completed"]
    expected_attempt = max(manifest["attempts"])
    if (
        len(completed) != 1
        or completed[0].get("attempt") != expected_attempt
        or completed[0].get("update") != manifest.get("completed_updates")
        or completed[0].get("cursor") != manifest.get("cursor")
    ):
        raise StagePrerequisiteError("training completion event does not reconcile")
    expected_kinds = {"training_checkpoint", "training_progress", "training_completed", "training_failure"}
    if any(event.get("kind") not in expected_kinds for event in linked):
        raise StagePrerequisiteError("training manifest links an unrelated event")
    for index in failure_indices:
        event = by_index.get(index)
        if (
            not isinstance(event, Mapping)
            or event.get("kind") != "training_failure"
            or event.get("arm") != manifest["arm"]
            or event.get("seed") != manifest["seed"]
            or event.get("event_id") != task9_failure_event_id(event)
        ):
            raise StagePrerequisiteError("training failure event identity does not reconcile")


def authorize_confirmation_test_read(context: "StageContext") -> bytes:
    """Open sealed rows only after immutable selection and all six checkpoints."""
    from .experiment import StagePrerequisiteError, _read_test_bytes_once

    selection_path = context.run_root / "selection.json"
    try:
        selection_bytes = _read_test_bytes_once(selection_path, trusted_root=context.run_root)
        selection = json.loads(selection_bytes.decode("utf-8"))
    except Exception as error:
        raise StagePrerequisiteError("confirmation requires immutable passing selection") from error
    expected_keys = {
        "schema_version", "status", "selected_condition", "signatures", "test_sha256",
        "n3", "dev_score_hashes", "data_manifest_sha256",
    }
    if (
        not isinstance(selection, Mapping)
        or set(selection) != expected_keys
        or selection.get("schema_version") != 1
        or selection.get("status") != "passed"
        or selection.get("selected_condition") != "A2"
        or selection.get("signatures") != dict(context.signatures)
        or not isinstance(selection.get("test_sha256"), str)
        or _SHA256.fullmatch(str(selection["test_sha256"])) is None
        or isinstance(selection.get("n3"), bool)
        or not isinstance(selection.get("n3"), int)
        or not isinstance(selection.get("dev_score_hashes"), Mapping)
        or set(selection["dev_score_hashes"]) != {"F0", "A1", "A2", "A3"}
        or any(_SHA256.fullmatch(str(value)) is None for value in selection["dev_score_hashes"].values())
    ):
        raise StagePrerequisiteError("confirmation requires canonical passing A2 selection")
    pilot = load_passed_pilot(context)
    n3 = int(selection["n3"])
    if n3 != pilot["n3"]:
        raise StagePrerequisiteError("selection n3 differs from passed pilot evidence")
    identities = pilot["initialization_hashes"]
    try:
        discover = json.loads(_read_test_bytes_once(
            context.run_root / "stages" / "discover.json", trusted_root=context.run_root
        ).decode("utf-8"))
        data_stage = json.loads(_read_test_bytes_once(
            context.run_root / "stages" / "data.json", trusted_root=context.run_root
        ).decode("utf-8"))
        data_manifest_bytes = _read_test_bytes_once(
            context.run_root / "data" / "manifest.json", trusted_root=context.run_root
        )
    except Exception as error:
        raise StagePrerequisiteError("discover/data authorization evidence is missing") from error
    discover_payload = discover.get("payload") if isinstance(discover, Mapping) else None
    data_payload = data_stage.get("payload") if isinstance(data_stage, Mapping) else None
    selection_digest = hashlib.sha256(selection_bytes).hexdigest()
    manifest_digest = hashlib.sha256(data_manifest_bytes).hexdigest()
    try:
        data_manifest = json.loads(data_manifest_bytes.decode("utf-8"))
        split_hashes = data_manifest.get("split_sha256") if isinstance(data_manifest, Mapping) else None
        test_manifest_hash = split_hashes.get("test") if isinstance(split_hashes, Mapping) else None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StagePrerequisiteError("G0 data manifest is unreadable") from error
    if (
        not isinstance(discover_payload, Mapping)
        or discover.get("status") != "passed"
        or discover.get("signatures") != dict(context.signatures)
        or discover_payload.get("selection_sha256") != selection_digest
        or discover_payload.get("data_manifest_sha256") != manifest_digest
        or not isinstance(data_payload, Mapping)
        or data_stage.get("status") != "passed"
        or data_stage.get("signatures") != dict(context.signatures)
        or data_payload.get("manifest_sha256") != manifest_digest
        or selection.get("data_manifest_sha256") != manifest_digest
        or test_manifest_hash != selection.get("test_sha256")
    ):
        raise StagePrerequisiteError("selection is not anchored to passed discover/data evidence")
    dev_paths = {
        "F0": context.run_root / "scores" / "frozen-f0.jsonl",
        "A1": context.run_root / "scores" / "a1-47-dev.jsonl",
        "A2": context.run_root / "scores" / "a2-47-dev.jsonl",
        "A3": context.run_root / "scores" / "a3-47-dev.jsonl",
    }
    for arm, path in dev_paths.items():
        try:
            rows = _read_test_bytes_once(path, trusted_root=context.run_root)
        except Exception as error:
            raise StagePrerequisiteError(
                f"selection dev evidence is missing for {arm}"
            ) from error
        if hashlib.sha256(rows).hexdigest() != selection["dev_score_hashes"][arm]:
            raise StagePrerequisiteError(
                f"selection dev score hash mismatch for {arm}"
            )
    checkpoint_evidence: dict[
        tuple[str, int], tuple[Mapping[str, object], bytes, Path]
    ] = {}
    for seed in _SEEDS:
        for arm in _ARMS:
            manifest = _load_training_manifest(context.run_root, arm, seed, n3)
            if manifest.get("signatures") != dict(context.signatures):
                raise StagePrerequisiteError(
                    f"checkpoint signatures do not match for {arm.upper()} seed {seed}"
                )
            if manifest.get("initialization_hash") != identities[str(seed)]:
                raise StagePrerequisiteError(
                    f"checkpoint initialization differs from pilot for {arm.upper()} seed {seed}"
                )
            checkpoint = context.run_root / "checkpoints" / arm / str(seed) / "latest.pt"
            try:
                checkpoint_bytes = _read_test_bytes_once(
                    checkpoint, trusted_root=context.run_root
                )
            except Exception as error:
                raise StagePrerequisiteError(
                    f"final checkpoint is missing for {arm.upper()} seed {seed}"
                ) from error
            if hashlib.sha256(checkpoint_bytes).hexdigest() != manifest["checkpoint_sha256"]:
                raise StagePrerequisiteError(
                    f"final checkpoint hash mismatch for {arm.upper()} seed {seed}"
                )
            checkpoint_evidence[(arm, seed)] = (manifest, checkpoint_bytes, checkpoint)
        a2 = _load_training_manifest(context.run_root, "a2", seed, n3)
        a3 = _load_training_manifest(context.run_root, "a3", seed, n3)
        ratio = float(a3["synchronized_cuda_seconds"]) / float(a2["synchronized_cuda_seconds"])
        if not 0.90 <= ratio <= 1.10:
            raise StagePrerequisiteError(
                f"seed-{seed} A3/A2 timing ratio is outside the registered interval"
            )
    for (arm, seed), (manifest, checkpoint_bytes, checkpoint) in checkpoint_evidence.items():
        try:
            from .train import validate_checkpoint_artifact
            artifact = validate_checkpoint_artifact(
                checkpoint.parent,
                context.signatures,
                expected_attempt=max(manifest["attempts"]),
                expected_update=int(manifest["completed_updates"]),
                expected_arm=arm,
                expected_seed=seed,
                expected_adapter_schema=production_checkpoint_schema(
                    context.run_root, arm, seed, context.signatures
                ),
                checkpoint_bytes=checkpoint_bytes,
            )
        except Exception as error:
            raise StagePrerequisiteError(
                f"Task-9 checkpoint artifact is invalid for {arm.upper()} seed {seed}"
            ) from error
        if artifact.get("checkpoint_sha256") != manifest["checkpoint_sha256"]:
            raise StagePrerequisiteError("Task-9 checkpoint identity does not reconcile")
        _validate_training_events(context.run_root, manifest)
    preexisting = [
        path for folder in ("scores", "retention-scores")
        for path in (context.run_root / folder).glob("*")
        if path.is_file() and ("-test" in path.name or "-confirm" in path.name)
    ]
    authorization = {
        "schema_version": 1,
        "selection_sha256": selection_digest,
        "test_sha256": selection["test_sha256"],
        "signatures": dict(context.signatures),
        "checkpoint_sha256": {
            f"{arm}@{seed}": _load_training_manifest(context.run_root, arm, seed, n3)["checkpoint_sha256"]
            for seed in _SEEDS for arm in _ARMS
        },
    }
    authorization_path = context.run_root / "confirmation-authorization.json"
    if context.attempt == 1:
        if authorization_path.exists() or preexisting:
            raise StagePrerequisiteError("pre-existing test score files block first sealed-test access")
        write_immutable_json(authorization_path, authorization)
    elif context.attempt == 2:
        try:
            persisted = json.loads(_read_test_bytes_once(
                authorization_path, trusted_root=context.run_root
            ).decode("utf-8"))
        except Exception as error:
            raise StagePrerequisiteError("sealed retry lacks authenticated authorization") from error
        if persisted != authorization:
            raise StagePrerequisiteError("sealed retry authorization does not reconcile")
        test_path = context.run_root / "data" / "test.jsonl"
        retry_test_bytes = _read_test_bytes_once(test_path, trusted_root=context.run_root)
        if hashlib.sha256(retry_test_bytes).hexdigest() != selection["test_sha256"]:
            raise StagePrerequisiteError("sealed retry test rows do not match authorized selection")
        _validate_partial_confirmation_scores(context, preexisting, retry_test_bytes)
    else:
        raise StagePrerequisiteError("sealed evaluation permits only one authenticated retry")
    test_path = context.run_root / "data" / "test.jsonl"
    test_bytes = _read_test_bytes_once(test_path, trusted_root=context.run_root)
    if hashlib.sha256(test_bytes).hexdigest() != selection["test_sha256"]:
        raise StagePrerequisiteError("sealed test rows do not match authorized selection")
    if _read_test_bytes_once(selection_path, trusted_root=context.run_root) != selection_bytes:
        raise StagePrerequisiteError("selection changed while sealed test rows were opened")
    return test_bytes


def _validate_partial_confirmation_scores(
    context: "StageContext", paths: list[Path], test_bytes: bytes
) -> None:
    """Accept only evaluator-shaped prefixes on the authenticated retry."""
    from .experiment import StagePrerequisiteError, _read_test_bytes_once

    test_rows = [json.loads(line) for line in test_bytes.decode("utf-8").splitlines()]
    expected_ids = [row.get("id") for row in test_rows]
    if not expected_ids or any(not isinstance(item, str) or not item for item in expected_ids) or len(set(expected_ids)) != len(expected_ids):
        raise StagePrerequisiteError("authorized retry test ID universe is invalid")
    exact_names = {"f0-test", *{
        f"{arm}-{seed}-test" for arm in _ARMS for seed in _SEEDS
    }}
    retention_names = {f"{arm}-48-confirm" for arm in _ARMS}
    for path in paths:
        stem = path.name.removesuffix(".summary.json").removesuffix(".jsonl")
        allowed = exact_names if path.parent.name == "scores" else retention_names
        if stem not in allowed:
            raise StagePrerequisiteError("sealed retry contains an unauthorized score artifact")
        if path.name.endswith(".summary.json"):
            value = json.loads(_read_test_bytes_once(path, trusted_root=context.run_root))
            raw_path = path.with_name(path.name.removesuffix(".summary.json") + ".jsonl")
            raw = _read_test_bytes_once(raw_path, trusted_root=context.run_root)
            rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
            if path.parent.name == "scores":
                expected_count = len(expected_ids)
                if stem == "f0-test":
                    expected_arm, expected_seed = "F0", 47
                else:
                    arm_value, seed_value, _ = stem.split("-", 2)
                    expected_arm, expected_seed = arm_value.upper(), int(seed_value)
            else:
                manifest = json.loads(_read_test_bytes_once(
                    context.run_root / "data" / "manifest.json", trusted_root=context.run_root
                ).decode("utf-8"))
                expected_count = manifest.get("retention_chunk_count") if isinstance(manifest, Mapping) else None
                expected_arm, expected_seed = stem.split("-", 1)[0].upper(), 48
            if (
                not isinstance(value, Mapping)
                or value.get("complete") is not True
                or isinstance(expected_count, bool)
                or not isinstance(expected_count, int)
                or value.get("record_count") != expected_count
                or value.get("expected_count") != expected_count
                or len(rows) != expected_count
                or value.get("split") != "test"
                or value.get("arm") != expected_arm
                or value.get("seed") != expected_seed
                or value.get("signatures") != dict(context.signatures)
            ):
                raise StagePrerequisiteError("sealed retry summary is incomplete or malformed")
            if value.get("records_sha256") != hashlib.sha256(raw).hexdigest():
                raise StagePrerequisiteError("sealed retry summary does not bind its raw prefix")
            continue
        rows = [json.loads(line) for line in _read_test_bytes_once(
            path, trusted_root=context.run_root
        ).decode("utf-8").splitlines()]
        key = "example_id" if path.parent.name == "scores" else "chunk_index"
        identifiers = [row.get(key) for row in rows if isinstance(row, Mapping)]
        if (
            len(rows) != len(identifiers)
            or len(set(identifiers)) != len(identifiers)
            or any(row.get("signatures") != dict(context.signatures) for row in rows)
            or any(row.get("split") != "test" for row in rows)
        ):
            raise StagePrerequisiteError("sealed retry score prefix is incompatible")
        if path.parent.name == "scores":
            if identifiers != expected_ids[:len(identifiers)]:
                raise StagePrerequisiteError("sealed retry exact score is not a registered ID prefix")
            stem = path.stem
            if stem == "f0-test":
                arm, seed = "F0", 47
            else:
                arm_value, seed_value, _ = stem.split("-", 2)
                arm, seed = arm_value.upper(), int(seed_value)
            if any(row.get("arm") != arm or row.get("seed") != seed or not isinstance(row.get("correct"), bool) for row in rows):
                raise StagePrerequisiteError("sealed retry exact records have the wrong producer identity")
        else:
            manifest = json.loads(_read_test_bytes_once(
                context.run_root / "data" / "manifest.json", trusted_root=context.run_root
            ).decode("utf-8"))
            expected_chunks = int(manifest["retention_chunk_count"])
            if identifiers != list(range(expected_chunks))[:len(identifiers)]:
                raise StagePrerequisiteError("sealed retry retention score is not a registered chunk prefix")
            if any(row.get("arm") != path.stem.split("-", 1)[0].upper() or row.get("seed") != 48 for row in rows):
                raise StagePrerequisiteError("sealed retry retention records have the wrong producer identity")
            if any(
                row.get("failure") is not None
                or isinstance(row.get("nll_sum"), bool)
                or not isinstance(row.get("nll_sum"), (int, float))
                or not math.isfinite(float(row["nll_sum"]))
                or int(row.get("scored_tokens", 0)) <= 0
                for row in rows
            ):
                raise StagePrerequisiteError("sealed retry retention records contain invalid evaluations")


def dispose_pilot_state(context: "StageContext", payload: Mapping[str, object]) -> None:
    """Remove only the stage-declared disposable subtree below the run root."""
    relative = payload.get("disposable_relative_path")
    if not isinstance(relative, str) or Path(relative).as_posix() != relative:
        raise ValueError("pilot payload has no canonical disposable path")
    target = (context.run_root / relative).resolve()
    if target.parent != context.run_root.resolve() or target.name != "pilot-disposable":
        raise ValueError("pilot disposable path escapes its registered location")
    if target.exists():
        shutil.rmtree(target)


def restore_production_initializations(
    context: "StageContext", payload: Mapping[str, object]
) -> None:
    """Prove pilot cleanup did not change either immutable production seed."""
    expected = payload.get("initialization_sha256")
    if not isinstance(expected, Mapping) or set(expected) != {"47", "48"}:
        raise ValueError("pilot payload has incomplete initialization evidence")
    for seed in _SEEDS:
        path = context.run_root / "initializations" / f"{seed}.pt"
        digest = sha256_file(path)
        if expected[str(seed)] != digest:
            raise ValueError(f"seed-{seed} production initialization changed during pilot")


def disposable_retry_artifact(run_root: Path, stage: str) -> Mapping[str, object]:
    """Authenticate a checkpoint-free G2/G3 retry from its durable event."""
    from .experiment import EnvironmentStop, _read_test_bytes_once
    from .provenance import sha256_payload

    if stage not in {"pilot", "frozen"}:
        raise EnvironmentStop("stage does not support a disposable retry")
    try:
        events = [
            json.loads(line)
            for line in _read_test_bytes_once(
                run_root / "events.jsonl", trusted_root=run_root
            ).decode("utf-8").splitlines()
        ]
    except Exception as error:
        raise EnvironmentStop("disposable retry has no durable failure event") from error
    matches = [
        event for event in events
        if isinstance(event, Mapping)
        and event.get("kind") == "stage_attempt"
        and event.get("stage") == stage
        and event.get("attempt") == 1
        and event.get("outcome") == "fresh_process_required"
    ]
    if not matches:
        raise EnvironmentStop("disposable retry has no durable failure event")
    event = matches[-1]
    index = event.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise EnvironmentStop("disposable retry event index is invalid")
    event_id = sha256_payload({
        "stage": stage, "attempt": 1, "index": index,
        "outcome": "fresh_process_required", "payload": event.get("payload"),
    })
    return {
        "path": "", "checkpoint_sha256": "", "manifest_digest": "",
        "attempt": 1, "update": 0, "event_id": event_id,
        "event_index": index, "arm": stage, "seed": 0,
        "relative_path": "", "adapter_schema": None,
    }


def run_production_pilot(context: "StageContext") -> Mapping[str, object]:
    return _run_stage_engine(context, "pilot")


def run_production_frozen(context: "StageContext") -> Mapping[str, object]:
    return _run_stage_engine(context, "frozen")


def run_production_discover(context: "StageContext") -> Mapping[str, object]:
    return _run_stage_engine(context, "discover")


def run_production_confirm(context: "StageContext") -> Mapping[str, object]:
    # The engine trains/resumes all seed-48 arms first.  Its sole sealed-read
    # call site then verifies selection plus all six final checkpoints.
    return _run_stage_engine(context, "confirm")


def _run_stage_engine(
    context: "StageContext", stage: str, *, sealed_test_bytes: bytes | None = None
) -> Mapping[str, object]:
    """Dispatch into the concrete heavyweight engine lazily.

    Keeping this boundary small makes the security-critical orchestration
    importable on CPU-only hosts while the engine owns actual model/training
    operations.
    """
    from .runtime_engine import execute_stage

    payload = dict(execute_stage(context, stage, sealed_test_bytes=sealed_test_bytes))
    validate_stage_payload(stage, payload, context)
    return payload


def _child_worker(request: Mapping[str, object], stage: str) -> Mapping[str, object]:
    from .runtime_engine import execute_child_stage

    return execute_child_stage(request, stage)


def run_child_pilot(request: Mapping[str, object]) -> Mapping[str, object]:
    return _child_worker(request, "pilot")


def run_child_frozen(request: Mapping[str, object]) -> Mapping[str, object]:
    return _child_worker(request, "frozen")


def run_child_discover(request: Mapping[str, object]) -> Mapping[str, object]:
    return _child_worker(request, "discover")


def run_child_confirm(request: Mapping[str, object]) -> Mapping[str, object]:
    return _child_worker(request, "confirm")
