"""Pinned generation primitives shared by blind calibration and later inference."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import multiprocessing
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence


class GenerationBudgetError(ValueError):
    """The registered prompt/output budgets cannot hold a rendered prompt."""


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    stop_reason: str
    generated_tokens: int
    wall_seconds: float
    cuda_seconds: float | None = None
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    free_device_bytes: int | None = None


def build_prompt_ids(tokenizer: Any, prompt_config: Any, question: str) -> list[int]:
    """Apply the pinned user-only chat template once, tokenizing without another BOS."""
    if getattr(prompt_config, "system_message", "") or getattr(prompt_config, "few_shot_examples", 0):
        raise ValueError("calibration prompt must have no system message or few-shot examples")
    if getattr(prompt_config, "thinking", False):
        raise ValueError("calibration prompt must disable thinking")
    prompt = prompt_config.template.format(question=question)
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("chat template returned a non-singleton batch")
        encoded = encoded[0]
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes)):
        raise ValueError("chat template did not return token IDs")
    return [int(token) for token in encoded]


def generate_one(
    model: Any,
    prompt_ids: Sequence[int],
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
    candidate_total: int,
) -> GenerationResult:
    """Run one greedy request only after validating all length budgets."""
    prompt_size = len(prompt_ids)
    if prompt_size > max_prompt_tokens:
        raise GenerationBudgetError(
            f"prompt has {prompt_size} tokens, exceeding prompt budget {max_prompt_tokens}"
        )
    if max_prompt_tokens + max_new_tokens > candidate_total:
        raise GenerationBudgetError("prompt plus generation budget exceeds candidate total")
    started = time.perf_counter()
    output = model.generate(
        input_ids=list(prompt_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        return_dict_in_generate=True,
    )
    elapsed = time.perf_counter() - started
    if isinstance(output, Mapping):
        text = str(output.get("text", ""))
        stop_reason = str(output.get("stop_reason", "error"))
        generated = int(output.get("generated_tokens", 0))
        return GenerationResult(
            text, stop_reason, generated,
            float(output.get("wall_seconds", elapsed)),
            output.get("cuda_seconds"), output.get("peak_allocated_bytes"),
            output.get("peak_reserved_bytes"), output.get("free_device_bytes"),
        )
    raise TypeError("injected model must return a mapping with text and stop metadata")


def verify_student_is_frozen(model: Any) -> None:
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable:
        raise ValueError(f"student inference model has trainable parameters: {trainable[:3]}")
    if getattr(model, "training", False):
        raise ValueError("student inference model is in training mode")


def generation_stop_reason(generated_ids: Sequence[int], eos_token_ids: int | Sequence[int] | None, *, max_new_tokens: int) -> str:
    """Classify an effective EOS (including alternate EOS IDs) before the length cap."""
    if isinstance(eos_token_ids, int):
        eos_ids = {eos_token_ids}
    else:
        eos_ids = set(eos_token_ids or ())
    if generated_ids and eos_ids and int(generated_ids[-1]) in eos_ids:
        return "eos"
    if len(generated_ids) >= max_new_tokens:
        return "length"
    return "error"


def resolve_pinned_snapshot(
    repo_id: str, revision: str, *, snapshot_download: Callable[..., str] | None = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve once through the hub API, requiring the resulting directory to match the pin."""
    if snapshot_download is None:
        from huggingface_hub import snapshot_download as hf_snapshot_download

        snapshot_download = hf_snapshot_download
    path = Path(snapshot_download(repo_id=repo_id, revision=revision, local_files_only=local_files_only))
    if path.name != revision:
        raise ValueError(f"resolved model snapshot does not match pinned revision {revision}")
    for required in ("config.json", "tokenizer_config.json"):
        if not (path / required).is_file():
            raise ValueError(f"pinned model snapshot lacks {required}")
    return path


def authenticate_model_snapshot(
    repo_id: str, revision: str, snapshot: Path, inventory_path: Path | None = None, *,
    model_info: Callable[..., Any] | None = None, metadata: Any | None = None,
) -> dict[str, Any]:
    """Authenticate local model bytes against the pinned Hub tree and persist inventory."""
    if model_info is None:
        from huggingface_hub import HfApi

        model_info = HfApi().model_info
    if metadata is None:
        metadata = model_info(repo_id, revision=revision, files_metadata=True)
    resolved = getattr(metadata, "sha", None)
    if resolved != revision:
        raise ValueError(f"Hub resolved {repo_id} to {resolved!r}, expected pinned revision {revision}")
    siblings = getattr(metadata, "siblings", None)
    if not isinstance(siblings, list) or not siblings:
        raise ValueError("Hub pin returned no model file inventory")
    expected = {item.rfilename: item for item in siblings}
    local = Path(snapshot)
    if local.name != revision or not local.is_dir():
        raise ValueError("local model snapshot path does not match its pin")
    actual_files = {path.relative_to(local).as_posix() for path in local.rglob("*") if path.is_file()}
    if actual_files != set(expected):
        raise ValueError("local model file inventory differs from pinned Hub metadata")
    files: list[dict[str, Any]] = []
    for name in sorted(expected):
        path = local / Path(name)
        metadata_item = expected[name]
        size = path.stat().st_size
        expected_size = getattr(metadata_item, "size", None)
        if expected_size is not None and size != expected_size:
            raise ValueError(f"pinned model file size mismatch: {name}")
        digest = _sha256_file(path)
        lfs = getattr(metadata_item, "lfs", None)
        lfs_sha = getattr(lfs, "sha256", None) if lfs else None
        if lfs_sha and digest != lfs_sha:
            raise ValueError(f"pinned Hub LFS content hash mismatch: {name}")
        files.append({"path": name, "size_bytes": size, "sha256": digest, "hub_lfs_sha256": lfs_sha})
    inventory = {"repo_id": repo_id, "revision": revision, "resolved_sha": resolved, "files": files}
    if inventory_path is not None:
        persist_model_inventory(inventory, inventory_path)
    return inventory


def pinned_eos_token_ids(snapshot_path: Path) -> tuple[int, ...]:
    """Read the generation EOS IDs from authenticated pinned model metadata."""
    root = Path(snapshot_path)
    for name in ("generation_config.json", "config.json"):
        path = root / name
        if not path.is_file():
            continue
        config = json.loads(path.read_text(encoding="utf-8"))
        value = config.get("eos_token_id")
        if isinstance(value, int) and not isinstance(value, bool):
            return (value,)
        if isinstance(value, list) and value and all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            return tuple(sorted(set(value)))
    raise ValueError("pinned model metadata does not declare effective EOS token IDs")


def persist_model_inventory(inventory: Mapping[str, Any], inventory_path: Path) -> None:
    """Persist the verified Hub/local inventory without allowing replacement."""
    import json
    import os

    target = Path(inventory_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        stored = json.loads(target.read_text(encoding="utf-8"))
        if stored != dict(inventory):
            raise ValueError("persisted pinned model inventory differs from current Hub/local bytes")
        return
    with target.open("xb") as stream:
        stream.write(json.dumps(dict(inventory), sort_keys=True, separators=(",", ":")).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sha256_file(path: Path) -> str:
    from hashlib import sha256

    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hf_worker(
    role: str, config: Any, *, repo_root: Path, snapshot_path: Path | None = None,
    tokenizer_snapshot_path: Path | None = None, eos_token_ids: Sequence[int] | None = None,
) -> "HuggingFaceWorker":
    """Load one pinned model in the declared precision; optional ML imports stay lazy."""
    if role not in {"student", "teacher"}:
        raise ValueError("role must be student or teacher")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    pin = config.student if role == "student" else config.teacher
    physical = int(torch.cuda.get_device_properties(0).total_memory)
    memory_cap = min(16 * 1024**3, physical) - config.resources.vram_reserve_bytes
    if memory_cap <= 0:
        raise ValueError("registered VRAM reserve leaves no model memory")
    if role == "student":
        snapshot = Path(snapshot_path) if snapshot_path else resolve_pinned_snapshot(
            pin.repo_id, pin.revision, local_files_only=True,
        )
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,
            quantization_config=quantization,
            attn_implementation="sdpa",
            device_map="auto",
            max_memory={0: memory_cap},
        )
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    else:
        teacher_snapshot = (
            Path(repo_root)
            / ".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots"
            / pin.revision
        )
        if not teacher_snapshot.is_dir() or teacher_snapshot.name != pin.revision:
            raise ValueError("pinned teacher snapshot is unavailable")
        snapshot = teacher_snapshot
        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
            max_memory={0: memory_cap},
        )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_snapshot_path or snapshot, local_files_only=True,
    )
    model.eval()
    if role == "student":
        verify_student_is_frozen(model)
    return HuggingFaceWorker(role, model, tokenizer, config.prompt, torch, eos_token_ids=eos_token_ids)


class HuggingFaceWorker:
    """One-model-at-a-time generation worker with row-level CUDA/resource snapshots."""

    def __init__(
        self, role: str, model: Any, tokenizer: Any, prompt_config: Any, torch: Any,
        *, eos_token_ids: Sequence[int] | None = None,
    ) -> None:
        self.role = role
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_config = prompt_config
        self.torch = torch
        self.eos_token_ids = tuple(eos_token_ids) if eos_token_ids is not None else None

    def generate(self, row: Mapping[str, str], candidate: Sequence[int]) -> dict[str, Any]:
        max_prompt, max_new = map(int, candidate)
        prompt_ids = build_prompt_ids(self.tokenizer, self.prompt_config, row["question"])
        if len(prompt_ids) > max_prompt or max_prompt + max_new > 4096:
            return {
                "text": "", "stop_reason": "budget_overflow", "generated_tokens": 0,
                "prompt_tokens": len(prompt_ids), "wall_seconds": 0.0, "cuda_seconds": 0.0,
            }
        torch = self.torch
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            free_before, _ = torch.cuda.mem_get_info()
        started = time.perf_counter()
        output = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            return_dict_in_generate=True,
            **({"eos_token_id": list(self.eos_token_ids)} if self.eos_token_ids is not None else {}),
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            cuda_seconds: float | None = time.perf_counter() - started
            allocated = int(torch.cuda.max_memory_allocated())
            reserved = int(torch.cuda.max_memory_reserved())
            free_after, _ = torch.cuda.mem_get_info()
        else:
            cuda_seconds, allocated, reserved, free_after = None, None, None, None
        wall_seconds = time.perf_counter() - started
        sequence = output.sequences[0]
        generated_ids = sequence[input_ids.shape[1]:]
        effective_eos = getattr(self.model.generation_config, "eos_token_id", None)
        if effective_eos is None:
            effective_eos = self.tokenizer.eos_token_id
        if self.eos_token_ids is not None:
            effective_eos = self.eos_token_ids
        stop_reason = generation_stop_reason(
            [int(token) for token in generated_ids.tolist()],
            effective_eos,
            max_new_tokens=max_new,
        )
        return {
            "text": self.tokenizer.decode(generated_ids, skip_special_tokens=True),
            "stop_reason": stop_reason,
            "generated_tokens": int(generated_ids.numel()),
            "prompt_tokens": len(prompt_ids),
            "wall_seconds": wall_seconds,
            "cuda_seconds": cuda_seconds,
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved,
            "free_device_bytes": min(int(free_before), int(free_after)) if torch.cuda.is_available() else None,
        }

    def close(self) -> None:
        """Release this phase's references and clear allocator cache before the next role."""
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _isolated_model_worker(
    queue_in: Any, queue_out: Any, role: str, config: Any,
    repo_root: str, snapshot_path: str, tokenizer_snapshot_path: str, eos_token_ids: Sequence[int],
) -> None:
    started = time.perf_counter()
    worker = None
    try:
        worker = load_hf_worker(
            role, config, repo_root=Path(repo_root), snapshot_path=Path(snapshot_path),
            tokenizer_snapshot_path=Path(tokenizer_snapshot_path),
            eos_token_ids=eos_token_ids,
        )
        queue_out.put({"ok": True, "event": "ready", "startup_wall_seconds": time.perf_counter() - started})
        while True:
            request = queue_in.get()
            if request.get("op") == "close":
                worker.close()
                worker = None
                queue_out.put({"ok": True, "event": "closed"})
                return
            try:
                queue_out.put({"ok": True, "result": worker.generate(request["row"], request["candidate"])})
            except Exception as exc:
                queue_out.put({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
                return
    except BaseException as exc:
        queue_out.put({
            "ok": False, "event": "load_failed", "error_type": type(exc).__name__,
            "error": str(exc), "startup_wall_seconds": time.perf_counter() - started,
        })
    finally:
        if worker is not None:
            worker.close()


class IsolatedModelWorker:
    """Spawn one role worker process; process exit is the fresh-process release boundary."""

    def __init__(
        self, role: str, config: Any, *, repo_root: Path, snapshot_path: Path,
        tokenizer_snapshot_path: Path, eos_token_ids: Sequence[int], timeout: float = 900,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        self._requests = context.Queue()
        self._responses = context.Queue()
        self._process = context.Process(
            target=_isolated_model_worker,
            args=(self._requests, self._responses, role, config, str(repo_root), str(snapshot_path), str(tokenizer_snapshot_path), tuple(eos_token_ids)),
            daemon=False,
        )
        self._process.start()
        try:
            message = self._responses.get(timeout=timeout)
        except Exception:
            self._process.join(timeout=1)
            raise RuntimeError("isolated model worker startup timed out") from None
        self.startup_wall_seconds = float(message.get("startup_wall_seconds", 0.0))
        if not message.get("ok"):
            self._process.join(timeout=1)
            raise RuntimeError(f"isolated model worker failed: {message.get('error_type')}: {message.get('error')}")

    def generate(self, row: Mapping[str, str], candidate: Sequence[int]) -> dict[str, Any]:
        self._requests.put({"op": "generate", "row": dict(row), "candidate": list(candidate)})
        try:
            message = self._responses.get(timeout=3600)
        except Exception:
            raise RuntimeError("isolated model worker response timed out") from None
        if not message.get("ok"):
            raise RuntimeError(f"isolated model worker failed: {message.get('error_type')}: {message.get('error')}")
        return message["result"]

    def close(self) -> None:
        if self._process.is_alive():
            self._requests.put({"op": "close"})
            self._process.join(timeout=120)
        if self._process.is_alive():
            raise RuntimeError("isolated worker did not exit; fresh-process release is unverified")
        self._requests.close()
        self._responses.close()


def isolated_worker_factory(
    role: str, config: Any, *, repo_root: Path, snapshot_path: Path,
    tokenizer_snapshot_path: Path, eos_token_ids: Sequence[int],
) -> IsolatedModelWorker:
    return IsolatedModelWorker(
        role, config, repo_root=repo_root, snapshot_path=snapshot_path,
        tokenizer_snapshot_path=tokenizer_snapshot_path, eos_token_ids=eos_token_ids,
    )


__all__ = [
    "GenerationBudgetError", "GenerationResult", "HuggingFaceWorker", "build_prompt_ids",
    "IsolatedModelWorker", "authenticate_model_snapshot", "generate_one", "generation_stop_reason", "isolated_worker_factory",
    "load_hf_worker", "pinned_eos_token_ids", "resolve_pinned_snapshot", "verify_student_is_frozen",
    "persist_model_inventory",
]
