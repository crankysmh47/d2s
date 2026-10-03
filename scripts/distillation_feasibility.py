"""Run a pinned QLoRA compatibility smoke with isolated GPU phase workers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import tomllib
import traceback
from typing import Any, Callable

from ppmoe.distill.feasibility import (
    STUDENT_MODEL_ID,
    STUDENT_REVISION,
    TEACHER_MODEL_ID,
    TEACHER_REVISION,
    assert_only_adapters_trainable,
    compute_memory_cap,
    validate_result,
    validate_source_inventory,
    validate_student_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "distillation-v1" / "feasibility-20261003"
STUDENT_SOURCE_ROOT = (
    PROJECT_ROOT / "artifacts" / "recurrence-v0" / "129d39646933" / "source" / STUDENT_REVISION
)
TEACHER_CACHE_ROOT = PROJECT_ROOT / ".cache" / "huggingface" / "hub"
MAX_NEW_DOWNLOAD_BYTES = 20 * 1024**3
MODEL_FILES = (
    "config.json", "generation_config.json", "model.safetensors.index.json", "model-*.safetensors",
    "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json",
    "special_tokens_map.json", "added_tokens.json", "LICENSE",
)
PHASES = ("student_update", "teacher_generation")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


def _torch_environment() -> dict[str, Any]:
    try:
        import torch
    except BaseException as error:
        return {"imported": False, "error": f"{type(error).__name__}: {error}"}
    result: dict[str, Any] = {
        "imported": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if result["cuda_available"]:
        try:
            properties = torch.cuda.get_device_properties(0)
            result.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                "physical_capacity_bytes": int(properties.total_memory),
                "capability": list(torch.cuda.get_device_capability(0)),
            })
        except BaseException as error:
            result["device_error"] = f"{type(error).__name__}: {error}"
    return result


def _device_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=8, check=False)
    except BaseException as error:
        return {"status": "unavailable", "error": f"{type(error).__name__}: {error}"}
    return {
        "status": "available" if completed.returncode == 0 else "error",
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _device_used_mib(snapshot: dict[str, Any]) -> int | None:
    if snapshot.get("status") != "available":
        return None
    rows = str(snapshot.get("stdout", "")).splitlines()
    if not rows:
        return None
    columns = rows[0].split(",")
    if len(columns) < 3:
        return None
    try:
        return int(columns[2].strip())
    except ValueError:
        return None


def _device_used_delta(before: dict[str, Any], after: dict[str, Any]) -> int | None:
    before_used = _device_used_mib(before)
    after_used = _device_used_mib(after)
    return None if before_used is None or after_used is None else after_used - before_used


def _distribution_versions() -> dict[str, str | None]:
    names = {
        "torch": "torch",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "bitsandbytes": "bitsandbytes",
        "peft": "peft",
        "huggingface_hub": "huggingface-hub",
    }
    versions: dict[str, str | None] = {}
    for key, package in names.items():
        try:
            versions[key] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def _environment_record(
    *,
    project_root: Path = PROJECT_ROOT,
    teacher_path: Path | None = None,
    student_files: dict[str, str] | None = None,
    downloaded_bytes: int = 0,
    prior_downloaded_bytes: int = 0,
    torch_environment: dict[str, Any] | None = None,
    device_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build useful environment evidence even without CUDA or model config files."""
    root = Path(project_root)
    lock_path = root / "uv.lock"
    locked: dict[str, Any] = {}
    if lock_path.is_file():
        try:
            lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
            wanted = {"accelerate", "bitsandbytes", "peft"}
            for package in lock.get("package", []):
                if package.get("name") in wanted:
                    locked[package["name"]] = {
                        "version": package.get("version"),
                        "wheels": [
                            {"url": wheel.get("url"), "hash": wheel.get("hash")}
                            for wheel in package.get("wheels", [])
                            if "win_amd64" in wheel.get("url", "") or "py3-none-any" in wheel.get("url", "")
                        ],
                    }
        except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
            locked = {"error": f"{type(error).__name__}: {error}"}
    hashes: dict[str, Any] = {}
    for label, path in (
        ("pyproject.toml", root / "pyproject.toml"),
        ("uv.lock", lock_path),
        ("feasibility.py", root / "src" / "ppmoe" / "distill" / "feasibility.py"),
        ("distillation_feasibility.py", Path(__file__).resolve()),
    ):
        hashes[label] = _sha256(path) if path.is_file() else None
    teacher_config = teacher_path / "config.json" if teacher_path else None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _distribution_versions(),
        "torch_environment": torch_environment if torch_environment is not None else _torch_environment(),
        "device_snapshot": device_snapshot if device_snapshot is not None else _device_snapshot(),
        "dependencies_lock": locked,
        "hashes": hashes,
        "student_files": student_files or {},
        "model_hashes": {
            "teacher_config": _sha256(teacher_config) if teacher_config is not None and teacher_config.is_file() else None,
        },
        "new_model_bytes": downloaded_bytes,
        "prior_downloaded_bytes": prior_downloaded_bytes,
    }


def _write_failure_artifacts(
    artifact_root: Path,
    error: BaseException,
    environment: dict[str, Any],
    *,
    state: dict[str, Any] | None = None,
) -> None:
    """Preserve environment, traceback, and failed status as first-write-only JSON."""
    root = Path(artifact_root)
    try:
        _write_once(root / "environment.json", environment)
    except FileExistsError:
        pass
    diagnostic = {
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "python": sys.version,
        "platform": platform.platform(),
        "state": state or {},
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
    }
    try:
        _write_once(root / "diagnostic.json", diagnostic)
    except FileExistsError:
        pass
    try:
        _write_once(root / "result.json", {
            "schema_version": 1,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "state": state or {},
        })
    except FileExistsError:
        pass


def _source_inventory() -> dict[str, str]:
    source_manifest = PROJECT_ROOT / "artifacts" / "recurrence-v0" / "129d39646933" / "source.json"
    record = json.loads(source_manifest.read_text(encoding="utf-8"))
    return validate_source_inventory(STUDENT_SOURCE_ROOT, record)


def _hub_metadata(repo_id: str, revision: str) -> tuple[Any, int]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError(f"Hugging Face resolved {repo_id}@{revision} to unexpected SHA {info.sha}")
    card_data = info.card_data or {}
    license_value = card_data.get("license", "") if isinstance(card_data, dict) else getattr(card_data, "license", "")
    if str(license_value).lower() != "apache-2.0":
        raise ValueError(f"{repo_id}@{revision} license changed or is unknown: {license_value!r}")
    siblings = {file.rfilename: file for file in info.siblings}
    missing = {"model.safetensors.index.json", "config.json", "tokenizer.json"}.difference(siblings)
    if missing:
        raise ValueError(f"pinned model snapshot is missing required files: {sorted(missing)}")
    total = 0
    for item in info.siblings:
        if item.rfilename.endswith(".safetensors"):
            size = getattr(item, "size", None)
            if size is None:
                raise ValueError(f"Hugging Face metadata omitted size for {item.rfilename}")
            total += int(size)
    return info, total


def _snapshot_path(repo_id: str, revision: str) -> Path:
    repo_dir = TEACHER_CACHE_ROOT / ("models--" + repo_id.replace("/", "--"))
    return repo_dir / "snapshots" / revision


def _download_teacher(max_download_bytes: int = MAX_NEW_DOWNLOAD_BYTES) -> tuple[Path, int]:
    info, _ = _hub_metadata(TEACHER_MODEL_ID, TEACHER_REVISION)
    snapshot = _snapshot_path(TEACHER_MODEL_ID, TEACHER_REVISION)
    sizes = {
        item.rfilename: int(item.size)
        for item in info.siblings
        if getattr(item, "size", None) is not None
    }
    required = [
        name for name in sizes
        if name in MODEL_FILES or (name.startswith("model-") and name.endswith(".safetensors"))
    ]
    estimate = sum(size for name, size in sizes.items() if name in required and not (snapshot / name).is_file())
    if estimate > max_download_bytes:
        raise ValueError("teacher download estimate exceeds the 20 GiB feasibility cap")
    if snapshot.is_dir() and all((snapshot / name).is_file() for name in required):
        return snapshot, 0

    from huggingface_hub import snapshot_download

    last_error: BaseException | None = None
    for attempt in range(3):
        try:
            returned = snapshot_download(
                repo_id=TEACHER_MODEL_ID,
                revision=TEACHER_REVISION,
                cache_dir=str(TEACHER_CACHE_ROOT),
                allow_patterns=list(MODEL_FILES),
                max_workers=4,
            )
            resolved = Path(returned).resolve()
            if resolved != snapshot.resolve():
                raise ValueError(f"teacher snapshot resolved to unexpected path: {resolved}")
            downloaded = sum(
                path.stat().st_size for path in snapshot.iterdir()
                if path.is_file() and path.name in required
            )
            if downloaded > max_download_bytes:
                raise ValueError("teacher snapshot exceeds the 20 GiB feasibility cap")
            return snapshot, min(downloaded, estimate) if estimate else downloaded
        except (OSError, TimeoutError, ConnectionError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(2**attempt)
    assert last_error is not None
    raise RuntimeError(f"teacher snapshot download failed after 2 retries: {last_error}") from last_error


def run_workers_in_order(artifact_root: Path, run_worker: Callable[[str, Path], int]) -> list[str]:
    """Run isolated phase workers sequentially; do not start teacher after student failure."""
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    for phase in PHASES:
        result_path = root / f"{phase}.json"
        exit_code = run_worker(phase, result_path)
        if exit_code != 0:
            raise RuntimeError(f"{phase} worker exited with status {exit_code}")
        if not result_path.is_file():
            raise RuntimeError(f"{phase} worker exited successfully without durable result")
        record = json.loads(result_path.read_text(encoding="utf-8"))
        if record.get("status") != "passed":
            raise RuntimeError(f"{phase} worker result status is not passed")
        completed.append(phase)
    return completed


def _capture_memory_peaks(torch: Any) -> dict[str, int]:
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _student_worker(artifact_root: Path) -> int:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from ppmoe.distill.feasibility import validate_memory_peaks

    phase_path = artifact_root / "student_update.json"
    error_path = artifact_root / "student_update_error.json"
    device_before = _device_snapshot()
    model: Any = None
    optimizer: Any = None
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("student worker requires CUDA with BF16 support")
        capacity = int(torch.cuda.get_device_properties(0).total_memory)
        memory_cap = compute_memory_cap(capacity)
        torch.manual_seed(20261003)
        torch.cuda.manual_seed_all(20261003)
        tokenizer = AutoTokenizer.from_pretrained(str(STUDENT_SOURCE_ROOT), local_files_only=True)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("student tokenizer has neither pad nor EOS token")
        token_ids = tokenizer.encode(
            "Synthetic compatibility smoke. This ordinary text is locally generated and not a benchmark item.",
            add_special_tokens=False,
        )
        if not token_ids:
            raise ValueError("student tokenizer returned no IDs for synthetic text")
        active_tokens = 128
        active_ids = (token_ids * math.ceil(active_tokens / len(token_ids)))[:active_tokens]
        full_ids = active_ids + [int(pad_id)] * (1536 - active_tokens)
        input_ids = torch.tensor([full_ids], dtype=torch.long, device="cuda")
        attention_mask = torch.tensor(
            [[1] * active_tokens + [0] * (1536 - active_tokens)], dtype=torch.long, device="cuda"
        )
        labels = input_ids.clone()
        labels.masked_fill_(attention_mask == 0, -100)
        if torch.count_nonzero(labels != -100).item() != active_tokens:
            raise ValueError("synthetic labels do not match the active attention mask")

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        load_start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            str(STUDENT_SOURCE_ROOT),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization,
            device_map={"": 0},
            attn_implementation="sdpa",
            use_safetensors=True,
        )
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_start
        load_peaks = _capture_memory_peaks(torch)
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.gradient_checkpointing_enable()
        model = get_peft_model(
            model,
            LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.0,
                target_modules="all-linear", task_type="CAUSAL_LM",
            ),
        )
        trainable_names = assert_only_adapters_trainable(model)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=1e-4)
        before = [parameter.detach().clone() for parameter in parameters]
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        update_start = time.perf_counter()
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        loss = output.loss
        if loss is None or not torch.isfinite(loss):
            raise ValueError("student synthetic loss is missing or nonfinite")
        loss_value = float(loss.detach().float().cpu().item())
        loss.backward()
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if not gradients:
            raise ValueError("student backward produced no adapter gradients")
        grad_norm = float(torch.sqrt(sum(torch.sum(grad.detach().float() ** 2) for grad in gradients)).item())
        if not math.isfinite(grad_norm) or grad_norm <= 0:
            raise ValueError("student adapter gradient is zero or nonfinite")
        optimizer.step()
        torch.cuda.synchronize()
        update_seconds = time.perf_counter() - update_start
        changed = sum(not torch.equal(old, parameter.detach()) for old, parameter in zip(before, parameters))
        if changed <= 0:
            raise ValueError("AdamW update did not change an adapter parameter")
        update_peaks = _capture_memory_peaks(torch)
        peaks = {
            "peak_allocated_bytes": max(load_peaks["peak_allocated_bytes"], update_peaks["peak_allocated_bytes"]),
            "peak_reserved_bytes": max(load_peaks["peak_reserved_bytes"], update_peaks["peak_reserved_bytes"]),
        }
        record: dict[str, Any] = {
            "schema_version": 1,
            "status": "passed",
            "loss": loss_value,
            "adapter_gradient_norm": grad_norm,
            "changed_parameter_count": int(changed),
            "trainable_adapter_parameter_count": len(parameters),
            "trainable_parameter_name_sample": trainable_names[:8],
            "model_load_seconds": load_seconds,
            "duration_seconds": update_seconds,
            "update_seconds": update_seconds,
            **peaks,
            "load_peak_allocated_bytes": load_peaks["peak_allocated_bytes"],
            "load_peak_reserved_bytes": load_peaks["peak_reserved_bytes"],
            "update_peak_allocated_bytes": update_peaks["peak_allocated_bytes"],
            "update_peak_reserved_bytes": update_peaks["peak_reserved_bytes"],
            "memory_cap_bytes": memory_cap,
            "physical_capacity_bytes": capacity,
            "device_snapshot_before": device_before,
            "sequence_tokens": 1536,
            "active_tokens": active_tokens,
            "masked_label_count": 1536 - active_tokens,
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "quantization": "NF4 double quantization, BF16 compute",
            "lora": {"target_modules": "all-linear", "rank": 16, "alpha": 32, "dropout": 0.0},
            "gradient_checkpointing": True,
            "attention_implementation": "sdpa",
            "use_cache": False,
        }
        try:
            validate_memory_peaks(peaks["peak_allocated_bytes"], peaks["peak_reserved_bytes"], memory_cap, phase="student")
        except ValueError as error:
            record["status"] = "failed"
            record["memory_error"] = str(error)
            record["device_snapshot_after"] = _device_snapshot()
            record["device_used_delta_mib"] = _device_used_delta(device_before, record["device_snapshot_after"])
            _write_once(phase_path, record)
            print(f"PHASE_FAILED=student_update {error}", file=sys.stderr, flush=True)
            return 1
        record["device_snapshot_after"] = _device_snapshot()
        record["device_used_delta_mib"] = _device_used_delta(device_before, record["device_snapshot_after"])
        _write_once(phase_path, record)
        print(f"PHASE_RESULT={phase_path}", flush=True)
        return 0
    except BaseException as error:
        try:
            device_after = _device_snapshot()
            _write_once(error_path, {
                "status": "failed", "error_type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc(), "device_snapshot_before": device_before,
                "device_snapshot_after": device_after,
                "device_used_delta_mib": _device_used_delta(device_before, device_after),
            })
        except FileExistsError:
            pass
        print(f"PHASE_FAILED=student_update {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


def _teacher_worker(artifact_root: Path, teacher_path: Path) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ppmoe.distill.feasibility import validate_memory_peaks

    phase_path = artifact_root / "teacher_generation.json"
    error_path = artifact_root / "teacher_generation_error.json"
    device_before = _device_snapshot()
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("teacher worker requires CUDA with BF16 support")
        capacity = int(torch.cuda.get_device_properties(0).total_memory)
        memory_cap = compute_memory_cap(capacity)
        torch.manual_seed(20261003)
        torch.cuda.manual_seed_all(20261003)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        load_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(str(teacher_path), local_files_only=True)
        messages = [{"role": "user", "content": "In one short sentence, describe why shadows change length during the day."}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        teacher = AutoModelForCausalLM.from_pretrained(
            str(teacher_path),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
            use_safetensors=True,
            use_cache=True,
        ).eval()
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_start
        load_peaks = _capture_memory_peaks(torch)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        generation_start = time.perf_counter()
        with torch.inference_mode():
            generated = teacher.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=16,
                use_cache=True,
                pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
            )
        torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - generation_start
        new_ids = generated[0, encoded["input_ids"].shape[-1]:].tolist()
        if not 1 <= len(new_ids) <= 16:
            raise ValueError(f"teacher generated invalid token count {len(new_ids)}")
        eos_id = tokenizer.eos_token_id
        eos_ids = set(eos_id if isinstance(eos_id, (list, tuple)) else [eos_id]) - {None}
        stop_reason = "eos" if any(token_id in eos_ids for token_id in new_ids) else "max_new_tokens"
        generation_peaks = _capture_memory_peaks(torch)
        peaks = {
            "peak_allocated_bytes": max(load_peaks["peak_allocated_bytes"], generation_peaks["peak_allocated_bytes"]),
            "peak_reserved_bytes": max(load_peaks["peak_reserved_bytes"], generation_peaks["peak_reserved_bytes"]),
        }
        device_after = _device_snapshot()
        record: dict[str, Any] = {
            "schema_version": 1,
            "status": "passed",
            "generated_tokens": len(new_ids),
            "stop_reason": stop_reason,
            "duration_seconds": generation_seconds,
            "model_load_seconds": load_seconds,
            **peaks,
            "load_peak_allocated_bytes": load_peaks["peak_allocated_bytes"],
            "load_peak_reserved_bytes": load_peaks["peak_reserved_bytes"],
            "generation_peak_allocated_bytes": generation_peaks["peak_allocated_bytes"],
            "generation_peak_reserved_bytes": generation_peaks["peak_reserved_bytes"],
            "memory_cap_bytes": memory_cap,
            "physical_capacity_bytes": capacity,
            "device_snapshot_before": device_before,
            "device_snapshot_after": device_after,
            "device_used_delta_mib": _device_used_delta(device_before, device_after),
            "prompt_type": "synthetic ordinary prose; no benchmark item",
            "thinking_enabled": False,
            "attention_implementation": "sdpa",
            "use_cache": True,
            "do_sample": False,
            "max_new_tokens": 16,
        }
        try:
            validate_memory_peaks(peaks["peak_allocated_bytes"], peaks["peak_reserved_bytes"], memory_cap, phase="teacher")
        except ValueError as error:
            record["status"] = "failed"
            record["memory_error"] = str(error)
            _write_once(phase_path, record)
            print(f"PHASE_FAILED=teacher_generation {error}", file=sys.stderr, flush=True)
            return 1
        _write_once(phase_path, record)
        print(f"PHASE_RESULT={phase_path}", flush=True)
        return 0
    except BaseException as error:
        try:
            device_after = _device_snapshot()
            _write_once(error_path, {
                "status": "failed", "error_type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc(), "device_snapshot_before": device_before,
                "device_snapshot_after": device_after,
                "device_used_delta_mib": _device_used_delta(device_before, device_after),
            })
        except FileExistsError:
            pass
        print(f"PHASE_FAILED=teacher_generation {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


def _worker_main(phase: str, artifact_root: Path, teacher_path: Path | None) -> int:
    if phase == "student_update":
        return _student_worker(artifact_root)
    if teacher_path is None:
        print("teacher worker requires --teacher-path", file=sys.stderr)
        return 2
    return _teacher_worker(artifact_root, teacher_path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one pinned synthetic QLoRA compatibility smoke.")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--prior-download-bytes", type=int, default=0,
                        help="bytes already downloaded for this study before this attempt")
    parser.add_argument("--worker-phase", choices=PHASES)
    parser.add_argument("--teacher-path", type=Path)
    return parser.parse_args(argv)


def _parent_main(artifact_root: Path, prior_download_bytes: int = 0) -> int:
    if prior_download_bytes < 0:
        raise ValueError("prior download bytes cannot be negative")
    if artifact_root.exists() and any(artifact_root.iterdir()):
        print(f"Refusing to overwrite nonempty artifact directory: {artifact_root}", file=sys.stderr)
        return 1
    artifact_root.mkdir(parents=True, exist_ok=True)
    _write_once(artifact_root / "process.json", {
        "pid": os.getpid(),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": sys.argv,
        "artifact_root": str(artifact_root),
    })
    state: dict[str, Any] = {
        "prior_downloaded_bytes": prior_download_bytes,
        "downloaded_bytes": 0,
        "completed_workers": [],
        "student_revision": STUDENT_REVISION,
        "teacher_revision": TEACHER_REVISION,
    }
    environment = _environment_record(prior_downloaded_bytes=prior_download_bytes)
    try:
        _write_once(artifact_root / "environment.json", environment)
        if not environment["torch_environment"].get("cuda_available"):
            raise RuntimeError("CUDA is unavailable")
        if not environment["torch_environment"].get("bf16_supported"):
            raise RuntimeError("GPU does not support BF16")
        print("STAGE=authenticate-existing-student", flush=True)
        student_files = _source_inventory()
        state["student_files"] = student_files
        print("STAGE=authenticate-and-materialize-teacher", flush=True)
        remaining_download_cap = MAX_NEW_DOWNLOAD_BYTES - prior_download_bytes
        if remaining_download_cap < 0:
            raise ValueError("prior study downloads exceed the 20 GiB download cap")
        teacher_path, downloaded_bytes = _download_teacher(remaining_download_cap)
        state["downloaded_bytes"] = downloaded_bytes
        cumulative_downloaded_bytes = prior_download_bytes + downloaded_bytes
        if cumulative_downloaded_bytes > MAX_NEW_DOWNLOAD_BYTES:
            raise ValueError("actual new model downloads exceeded the 20 GiB cap")
        _, teacher_weight_bytes = _hub_metadata(TEACHER_MODEL_ID, TEACHER_REVISION)
        _, student_weight_bytes = _hub_metadata(STUDENT_MODEL_ID, STUDENT_REVISION)
        from transformers import AutoConfig

        student_config = AutoConfig.from_pretrained(str(STUDENT_SOURCE_ROOT), local_files_only=True)
        validate_student_config(student_config)
        teacher_config = AutoConfig.from_pretrained(str(teacher_path), local_files_only=True)
        if "Qwen3ForCausalLM" not in (teacher_config.architectures or []):
            raise ValueError("pinned teacher does not expose Qwen3ForCausalLM")
        cap = compute_memory_cap(int(environment["torch_environment"]["physical_capacity_bytes"]))
        _write_once(artifact_root / "preflight.json", {
            "student": {"repo_id": STUDENT_MODEL_ID, "revision": STUDENT_REVISION,
                        "license": "apache-2.0", "weight_bytes": student_weight_bytes,
                        "files": student_files},
            "teacher": {"repo_id": TEACHER_MODEL_ID, "revision": TEACHER_REVISION,
                        "license": "apache-2.0", "weight_bytes": teacher_weight_bytes,
                        "config_sha256": _sha256(teacher_path / "config.json")},
            "downloaded_bytes": downloaded_bytes,
            "prior_downloaded_bytes": prior_download_bytes,
            "cumulative_downloaded_bytes": cumulative_downloaded_bytes,
            "memory_cap_bytes": cap,
            "device_snapshot": environment["device_snapshot"],
        })
        print(f"STAGE=isolated_workers memory_cap_bytes={cap} downloaded_bytes={downloaded_bytes}", flush=True)

        def launch(phase: str, _result_path: Path) -> int:
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--worker-phase", phase,
                "--artifact-root", str(artifact_root),
                "--teacher-path", str(teacher_path),
            ]
            completed = subprocess.run(command, check=False)
            return int(completed.returncode)

        def launch_and_record(phase: str, result_path: Path) -> int:
            code = launch(phase, result_path)
            if code == 0:
                state["completed_workers"].append(phase)
            return code

        run_workers_in_order(artifact_root, launch_and_record)
        student_record = json.loads((artifact_root / "student_update.json").read_text(encoding="utf-8"))
        teacher_record = json.loads((artifact_root / "teacher_generation.json").read_text(encoding="utf-8"))
        result = {
            "schema_version": 1,
            "status": "passed",
            "student": {"repo_id": STUDENT_MODEL_ID, "revision": STUDENT_REVISION,
                        "license": "apache-2.0", "weight_bytes": student_weight_bytes,
                        "license_sha256": student_files["LICENSE"]},
            "teacher": {"repo_id": TEACHER_MODEL_ID, "revision": TEACHER_REVISION,
                        "license": "apache-2.0", "weight_bytes": teacher_weight_bytes},
            "student_update": student_record,
            "teacher_generation": teacher_record,
            "downloads": {"new_model_bytes": downloaded_bytes,
                          "prior_downloaded_bytes": prior_download_bytes,
                          "cumulative_downloaded_bytes": cumulative_downloaded_bytes,
                          "cap_bytes": MAX_NEW_DOWNLOAD_BYTES,
                          "student_reused_existing_snapshot": True},
            "environment_sha256": _sha256(artifact_root / "environment.json"),
            "preflight_sha256": _sha256(artifact_root / "preflight.json"),
        }
        validate_result(result)
        _write_once(artifact_root / "result.json", result)
        print("STATUS=passed", flush=True)
        print(f"RESULT={artifact_root / 'result.json'}", flush=True)
        return 0
    except BaseException as error:
        state["environment"] = environment
        _write_failure_artifacts(artifact_root, error, environment, state=state)
        print(f"FAILED={type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    artifact_root = args.artifact_root.resolve()
    if args.worker_phase:
        return _worker_main(args.worker_phase, artifact_root, args.teacher_path)
    return _parent_main(artifact_root, args.prior_download_bytes)


if __name__ == "__main__":
    raise SystemExit(main())
