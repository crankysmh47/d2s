"""Run one pinned compatibility smoke for the verified-distillation pilot.

This is deliberately not a training or evaluation entry point. It performs one
synthetic QLoRA update and one synthetic teacher generation, then writes a
single durable result under the dated feasibility artifact directory.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import tomllib
from typing import Any

from ppmoe.distill.feasibility import (
    STUDENT_MODEL_ID,
    STUDENT_REVISION,
    TEACHER_MODEL_ID,
    TEACHER_REVISION,
    assert_only_adapters_trainable,
    validate_result,
    validate_student_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "distillation-v1" / "feasibility-20261003"
STUDENT_SOURCE_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "recurrence-v0"
    / "129d39646933"
    / "source"
    / STUDENT_REVISION
)
TEACHER_CACHE_ROOT = PROJECT_ROOT / ".cache" / "huggingface" / "hub"
MAX_NEW_DOWNLOAD_BYTES = 20 * 1024**3
MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "LICENSE",
)


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


def _source_inventory() -> dict[str, str]:
    source_manifest = PROJECT_ROOT / "artifacts" / "recurrence-v0" / "129d39646933" / "source.json"
    record = json.loads(source_manifest.read_text(encoding="utf-8"))
    if record.get("repo_id") != STUDENT_MODEL_ID or record.get("revision") != STUDENT_REVISION:
        raise ValueError("existing student source manifest does not match the fixed repository revision")
    inventory: dict[str, str] = {}
    for item in record.get("files", []):
        relative = item.get("relative_path")
        expected = item.get("sha256")
        if isinstance(relative, str) and isinstance(expected, str):
            path = STUDENT_SOURCE_ROOT / relative
            if path.is_file():
                actual = _sha256(path)
                if actual != expected:
                    raise ValueError(f"student source hash mismatch: {relative}")
                inventory[relative] = actual
    expected_shards = {
        "model-00001-of-00002.safetensors": "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
        "model-00002-of-00002.safetensors": "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9",
    }
    for relative, expected in expected_shards.items():
        if inventory.get(relative) != expected:
            raise ValueError(f"student source manifest is missing authenticated shard {relative}")
    if not (STUDENT_SOURCE_ROOT / "tokenizer.json").is_file():
        raise FileNotFoundError(f"student tokenizer missing from existing snapshot: {STUDENT_SOURCE_ROOT}")
    return inventory


def _hub_metadata(repo_id: str, revision: str) -> tuple[Any, int]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError(f"Hugging Face resolved {repo_id}@{revision} to unexpected SHA {info.sha}")
    card_data = info.card_data or {}
    license_value = card_data.get("license", "") if isinstance(card_data, dict) else getattr(card_data, "license", "")
    license_name = str(license_value).lower()
    if license_name != "apache-2.0":
        raise ValueError(f"{repo_id}@{revision} license changed or is unknown: {license_name!r}")
    required = {"model.safetensors.index.json", "config.json", "tokenizer.json"}
    siblings = {file.rfilename: file for file in info.siblings}
    missing = required.difference(siblings)
    if missing:
        raise ValueError(f"pinned model snapshot is missing required files: {sorted(missing)}")
    total = 0
    for file in info.siblings:
        if file.rfilename.endswith(".safetensors"):
            size = getattr(file, "size", None)
            if size is None:
                raise ValueError(f"Hugging Face metadata omitted size for {file.rfilename}")
            total += int(size)
    return info, total


def _snapshot_path(repo_id: str, revision: str) -> Path:
    repo_dir = TEACHER_CACHE_ROOT / ("models--" + repo_id.replace("/", "--"))
    return repo_dir / "snapshots" / revision


def _download_teacher() -> tuple[Path, int]:
    info, expected_weight_bytes = _hub_metadata(TEACHER_MODEL_ID, TEACHER_REVISION)
    snapshot = _snapshot_path(TEACHER_MODEL_ID, TEACHER_REVISION)
    required_sizes = {
        file.rfilename: int(file.size)
        for file in info.siblings
        if getattr(file, "size", None) is not None
    }
    required = [
        filename
        for filename in required_sizes
        if filename in MODEL_FILES or filename.startswith("model-") and filename.endswith(".safetensors")
    ]
    expected_download = sum(
        required_sizes[name]
        for name in required
        if not (snapshot / name).is_file()
    )
    # Student files are already present in the authoritative recurrence snapshot;
    # only genuinely absent teacher files count toward this pre-download gate.
    if expected_download + sum(path.stat().st_size for path in STUDENT_SOURCE_ROOT.glob("*.safetensors")) > MAX_NEW_DOWNLOAD_BYTES:
        raise ValueError("model download estimate exceeds the 20 GiB feasibility cap")
    if snapshot.is_dir() and all((snapshot / name).is_file() for name in required):
        return snapshot, 0

    from huggingface_hub import snapshot_download

    last_error: BaseException | None = None
    for attempt in range(3):
        try:
            snapshot_text = snapshot_download(
                repo_id=TEACHER_MODEL_ID,
                revision=TEACHER_REVISION,
                cache_dir=str(TEACHER_CACHE_ROOT),
                allow_patterns=list(MODEL_FILES),
                max_workers=4,
            )
            resolved = Path(snapshot_text).resolve()
            if resolved != snapshot.resolve():
                raise ValueError(f"teacher snapshot resolved to unexpected path: {resolved}")
            downloaded = sum(
                path.stat().st_size
                for path in snapshot.iterdir()
                if path.is_file() and path.name in required
            )
            if downloaded > MAX_NEW_DOWNLOAD_BYTES:
                raise ValueError("actual teacher snapshot exceeds the 20 GiB feasibility cap")
            return snapshot, min(downloaded, expected_download) if expected_download else downloaded
        except (OSError, TimeoutError, ConnectionError) as error:
            last_error = error
            if attempt == 2:
                break
            time.sleep(2**attempt)
    assert last_error is not None
    raise RuntimeError(f"teacher snapshot download failed after 2 retries: {last_error}") from last_error


def _environment_record(downloaded_bytes: int, student_files: dict[str, str], teacher_path: Path) -> dict[str, Any]:
    import torch
    import transformers
    import bitsandbytes
    import accelerate
    import peft

    lock_path = PROJECT_ROOT / "uv.lock"
    with lock_path.open("rb") as stream:
        lock = tomllib.load(stream)
    wanted = {"accelerate", "bitsandbytes", "peft"}
    locked: dict[str, Any] = {}
    for package in lock.get("package", []):
        if package.get("name") in wanted:
            locked[package["name"]] = {
                "version": package.get("version"),
                "wheels": [
                    {"url": wheel["url"], "hash": wheel.get("hash")}
                    for wheel in package.get("wheels", [])
                    if "win_amd64" in wheel.get("url", "") or "py3-none-any" in wheel.get("url", "")
                ],
            }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "peft": peft.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "dependencies_lock": locked,
        "hashes": {
            "pyproject.toml": _sha256(PROJECT_ROOT / "pyproject.toml"),
            "uv.lock": _sha256(lock_path),
            "feasibility.py": _sha256(PROJECT_ROOT / "src" / "ppmoe" / "distill" / "feasibility.py"),
            "distillation_feasibility.py": _sha256(Path(__file__).resolve()),
            "student_files": student_files,
            "teacher_config": _sha256(teacher_path / "config.json"),
        },
        "new_model_bytes": downloaded_bytes,
    }


def _run_smoke(teacher_path: Path, downloaded_bytes: int) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    student_inventory = _source_inventory()
    student_info, student_weight_bytes = _hub_metadata(STUDENT_MODEL_ID, STUDENT_REVISION)
    student_license_sha = hashlib.sha256(
        (STUDENT_SOURCE_ROOT / "LICENSE").read_bytes()
    ).hexdigest()
    student_snapshot_size = sum(path.stat().st_size for path in STUDENT_SOURCE_ROOT.glob("*.safetensors"))
    if student_snapshot_size != student_weight_bytes:
        raise ValueError("student source snapshot weight size disagrees with authenticated Hub metadata")
    if downloaded_bytes > MAX_NEW_DOWNLOAD_BYTES:
        raise ValueError("new model downloads exceeded the 20 GiB cap")
    teacher_info, teacher_weight_bytes = _hub_metadata(TEACHER_MODEL_ID, TEACHER_REVISION)

    student_config = AutoConfig.from_pretrained(str(STUDENT_SOURCE_ROOT), local_files_only=True)
    validate_student_config(student_config)
    teacher_config = AutoConfig.from_pretrained(str(teacher_path), local_files_only=True)
    if "Qwen3ForCausalLM" not in (teacher_config.architectures or []):
        raise ValueError("pinned teacher does not expose Qwen3ForCausalLM")

    tokenizer = AutoTokenizer.from_pretrained(str(STUDENT_SOURCE_ROOT), local_files_only=True)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("student tokenizer has no pad or EOS token")
    base_ids = tokenizer.encode(
        "Synthetic compatibility smoke. This ordinary text is generated locally and is not a benchmark item.",
        add_special_tokens=False,
    )
    if not base_ids:
        raise ValueError("student tokenizer returned no IDs for the synthetic prompt")
    active_tokens = 128
    synthetic_ids = (base_ids * math.ceil(active_tokens / len(base_ids)))[:active_tokens]
    padded_ids = synthetic_ids + [int(pad_token_id)] * (1536 - active_tokens)
    input_ids = torch.tensor([padded_ids], dtype=torch.long, device="cuda")
    attention_mask = torch.tensor([[1] * active_tokens + [0] * (1536 - active_tokens)], dtype=torch.long, device="cuda")
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
    student_load_start = time.perf_counter()
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
    student_load_seconds = time.perf_counter() - student_load_start
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    trainable_names = assert_only_adapters_trainable(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    before = [parameter.detach().clone() for parameter in parameters]
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    update_start = time.perf_counter()
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
    loss = output.loss
    if loss is None or not torch.isfinite(loss):
        raise ValueError("student synthetic loss is missing or nonfinite")
    loss_value = float(loss.detach().float().cpu().item())
    loss.backward()
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        raise ValueError("student adapter backward pass produced no gradients")
    grad_norm = float(torch.sqrt(sum(torch.sum(gradient.detach().float() ** 2) for gradient in gradients)).item())
    if not math.isfinite(grad_norm) or grad_norm <= 0:
        raise ValueError("student adapter gradient is zero or nonfinite")
    optimizer.step()
    torch.cuda.synchronize()
    update_seconds = time.perf_counter() - update_start
    changed = sum(not torch.equal(initial, parameter.detach()) for initial, parameter in zip(before, parameters))
    student_peak = int(torch.cuda.max_memory_allocated())
    if changed <= 0:
        raise ValueError("AdamW update did not change any adapter parameter")
    student_result = {
        "loss": loss_value,
        "adapter_gradient_norm": grad_norm,
        "changed_parameter_count": int(changed),
        "trainable_adapter_parameter_count": len(parameters),
        "trainable_parameter_name_sample": trainable_names[:8],
        "duration_seconds": update_seconds,
        "model_load_seconds": student_load_seconds,
        "peak_memory_bytes": student_peak,
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

    # Release the quantized student and optimizer before the BF16 teacher load.
    del output, loss, gradients, before, parameters, optimizer, model
    del input_ids, attention_mask, labels, tokenizer
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if torch.cuda.memory_allocated() != 0:
        raise RuntimeError(f"student GPU allocations remain after unload: {torch.cuda.memory_allocated()} bytes")

    teacher_tokenizer = AutoTokenizer.from_pretrained(str(teacher_path), local_files_only=True)
    messages = [{"role": "user", "content": "In one short sentence, describe why shadows change length during the day."}]
    prompt = teacher_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = teacher_tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    teacher_load_start = time.perf_counter()
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
    teacher_load_seconds = time.perf_counter() - teacher_load_start
    teacher_load_peak = int(torch.cuda.max_memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    generation_start = time.perf_counter()
    with torch.inference_mode():
        generated = teacher.generate(
            **encoded,
            do_sample=False,
            num_beams=1,
            max_new_tokens=16,
            use_cache=True,
            pad_token_id=teacher_tokenizer.pad_token_id or teacher_tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_start
    generated_tokens = int(generated.shape[-1] - encoded["input_ids"].shape[-1])
    if generated_tokens != 16:
        raise ValueError(f"teacher generated {generated_tokens} tokens; exactly 16 required")
    teacher_peak = int(torch.cuda.max_memory_allocated())
    teacher_result = {
        "generated_tokens": generated_tokens,
        "duration_seconds": generation_seconds,
        "model_load_seconds": teacher_load_seconds,
        "model_load_peak_memory_bytes": teacher_load_peak,
        "peak_memory_bytes": teacher_peak,
        "prompt_type": "synthetic ordinary prose; no benchmark item",
        "thinking_enabled": False,
        "attention_implementation": "sdpa",
        "use_cache": True,
        "do_sample": False,
    }
    del generated, encoded, teacher, teacher_tokenizer
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    return {
        "schema_version": 1,
        "status": "passed",
        "student": {
            "repo_id": STUDENT_MODEL_ID,
            "revision": STUDENT_REVISION,
            "license": str((student_info.card_data or {}).get("license")),
            "license_sha256": student_license_sha,
            "weight_bytes": student_weight_bytes,
        },
        "teacher": {
            "repo_id": TEACHER_MODEL_ID,
            "revision": TEACHER_REVISION,
            "license": str((teacher_info.card_data or {}).get("license")),
            "weight_bytes": teacher_weight_bytes,
        },
        "student_update": student_result,
        "teacher_generation": teacher_result,
        "downloads": {
            "new_model_bytes": downloaded_bytes,
            "cap_bytes": MAX_NEW_DOWNLOAD_BYTES,
            "student_reused_existing_snapshot": True,
        },
    }


def main() -> int:
    if ARTIFACT_ROOT.exists() and any(ARTIFACT_ROOT.iterdir()):
        print(f"Refusing to overwrite nonempty artifact directory: {ARTIFACT_ROOT}", file=sys.stderr)
        return 1
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"ARTIFACT_ROOT={ARTIFACT_ROOT}", flush=True)
    _write_once(ARTIFACT_ROOT / "process.json", {
        "pid": os.getpid(),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": [sys.executable, str(Path(__file__).resolve())],
        "artifact_root": str(ARTIFACT_ROOT),
    })
    result: dict[str, Any]
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU does not support BF16")
        print("STAGE=authenticate-existing-student", flush=True)
        student_inventory = _source_inventory()
        print("STAGE=authenticate-and-materialize-teacher", flush=True)
        teacher_path, downloaded_bytes = _download_teacher()
        print(f"STAGE=smoke-models downloaded_teacher_bytes={downloaded_bytes}", flush=True)
        # _run_smoke performs metadata verification too; retain its measured bytes.
        result = _run_smoke(teacher_path, downloaded_bytes)
        downloaded_bytes = int(result["downloads"]["new_model_bytes"])
        environment = _environment_record(downloaded_bytes, student_inventory, teacher_path)
        validate_result(result)
        _write_once(ARTIFACT_ROOT / "environment.json", environment)
        _write_once(ARTIFACT_ROOT / "result.json", result)
        print(f"STATUS={result['status']}", flush=True)
        print(f"RESULT={ARTIFACT_ROOT / 'result.json'}", flush=True)
        return 0
    except BaseException as error:
        diagnostic = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "python": sys.version,
            "platform": platform.platform(),
            "traceback": __import__("traceback").format_exc(),
        }
        try:
            try:
                _write_once(ARTIFACT_ROOT / "environment.json", _environment_record(0, {}, TEACHER_CACHE_ROOT))
            except BaseException:
                pass
            _write_once(ARTIFACT_ROOT / "diagnostic.json", diagnostic)
            failed_result = {
                "schema_version": 1,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "student_revision": STUDENT_REVISION,
                "teacher_revision": TEACHER_REVISION,
            }
            _write_once(ARTIFACT_ROOT / "result.json", failed_result)
        except FileExistsError:
            print("Immutable artifact already exists; preserved it.", file=sys.stderr)
        print(f"FAILED={type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
