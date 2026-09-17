# Pretrained Recurrence V0 Sequential Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Do not use subagents unless the user explicitly requests delegation. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run one bounded experiment that determines whether one additional pass through Qwen3-1.7B layers `[12,16)`, adapted with shared rank-8 LoRA, improves sealed exact-answer accuracy over the unadapted model, ordinary LoRA, and a training-time-matched ordinary-LoRA control without exceeding the registered retention or resource limits.

**Architecture:** Add an isolated `ppmoe.recurrence` subsystem around the stock `Qwen3ForCausalLM`. Separate modules own immutable configuration/provenance, deterministic task and retention data, explicit prefix/block/suffix execution, LoRA and training recovery, generation/scoring/statistics, stage orchestration, and evidence-derived reporting. Production work advances through G0-G6 only when the persisted predecessor stage authorizes it.

**Tech Stack:** Python 3.11, uv, PyTorch 2.13/CUDA 13, Transformers 5.16, datasets 4, pytest 8, TOML configuration, JSON/JSONL evidence, PowerShell execution.

**Spec:** `docs/superpowers/specs/2026-09-08-pretrained-recurrence-design.md`

**Protocol:** `plan.md`

**Operator checklist:** `todo.md`

## Global constraints

- Work only on branch `exp`; do not switch or modify `main`.
- Preserve all existing Phase 0, Phase 1A, and Phase 1A.5 source files and artifacts.
- Use `Qwen/Qwen3-1.7B`, post-trained, resolved to a full immutable Hugging Face commit SHA before downloading weights.
- Assert `Qwen3ForCausalLM`, 28 layers, hidden size 2048, intermediate size 6144, 16 query heads, 8 KV heads, head dimension 128, and no sliding attention.
- Repeat the fixed half-open block `[12,16)` only. Do not search layers, rank, damping, prompts, dataset sizes, seeds, or optimizer settings.
- Keep frozen weights and activations BF16; keep LoRA matrices and AdamW state FP32.
- Use eager attention, `use_cache=False`, `past_key_values=None`, one sequence per forward, and at most 512 total tokens.
- Use rank 8, alpha 16, dropout 0 LoRA on `q_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` in layers 12-15.
- Use AdamW `lr=1e-4`, betas `(0.9,0.95)`, epsilon `1e-8`, weight decay 0, gradient clipping 1, and 20-update linear warmup.
- Train A1 and A2 for exactly 1,000 optimizer updates with eight unique examples per update. Derive A3 steps once from the registered pilot rule.
- Use seed 47 for discovery and seed 48 only after a passing discovery manifest.
- Count every synchronized CUDA-job interval, including failed attempts, against the 24-hour cap. Stop before the next stage if its pessimistic projection cannot fit.
- Keep downloads at or below 8 GiB and new recurrence artifacts at or below 20 GiB.
- Never inspect or score `data/test.jsonl` before a signed passing `selection.json` exists.
- Never omit invalid model outputs. Invalid outputs are scored wrong; infrastructure failures invalidate the complete stage.
- Never weaken a gate, expand a sample, retry a scientific failure, or start a different research direction automatically.
- Continue automatically through passing gates. Pause for the user only when credentials, license acceptance, physical intervention, an unavailable external resource, ambiguous ownership of conflicting edits, or another genuinely human-only action blocks progress.
- A scientific stop, budget stop, or repeated infrastructure fault is a terminal outcome for this study, not a request for human permission to retune it.

## Evidence status vocabulary

Every stage record must use exactly one terminal status:

```python
TERMINAL_STATUSES = {
    "passed",
    "stopped_scientific",
    "stopped_budget",
    "failed_environment",
}
```

CLI exit codes are fixed:

```text
0 = completed stage or passing gate
1 = environment or implementation failure
2 = scientific stop, including benchmark uninformative or retention uncertain
3 = budget stop
```

Before returning any nonzero exit, persist the stage record and failure/event record when the artifact filesystem remains writable. `report` may run after any terminal status and must never launch missing work.

## Execution and commit policy

Execute every task in numeric order. A task may start only when all earlier task verification commands pass or the task is the terminal-report path authorized by a stop.

Commits occur at coherent milestones rather than after every small task:

1. **Protocol freeze:** this plan, the recurrence spec, root protocol, todo, archive, and project log.
2. **Foundation implementation:** configuration, immutable provenance, source resolution, synthetic data, and retention materialization.
3. **Model and training implementation:** LoRA, recurrent wrapper, causal loss, event ledger, checkpoints, and trainer.
4. **Evaluation and orchestration implementation:** generation, metrics, bootstraps, gates, timing, CLI, and report builder.
5. **Observed evidence:** project-log/todo/report-source updates after the experiment reaches success or a terminal stop.

Do not commit ignored model weights, generated data, checkpoints, scores, caches, or recurrence artifacts. At each commit boundary, inspect `git diff --cached` and ensure only intended source, tests, configuration, and documentation are staged.

## File responsibility map

| File | Single responsibility |
|---|---|
| `configs/recurrence-v0.toml` | Serialize every frozen scientific and resource constant. |
| `src/ppmoe/recurrence/config.py` | Parse, type, and validate the immutable configuration. |
| `src/ppmoe/recurrence/provenance.py` | Canonical hashing, immutable writes, source resolution, run signatures, and artifact-size checks. |
| `src/ppmoe/recurrence/data.py` | Generate, render, parse, freeze, and load the three exact-answer families and the FineWeb retention stream. |
| `src/ppmoe/recurrence/lora.py` | FP32 LoRA primitive, deterministic installation, and adapter state hashing. |
| `src/ppmoe/recurrence/model.py` | Exact stock/single/repeat/damped Qwen3 execution and causal answer loss. |
| `src/ppmoe/recurrence/train.py` | Deterministic optimizer loop, checkpoints, resume, timing, and training manifests. |
| `src/ppmoe/recurrence/evaluate.py` | Greedy full-prefix generation, exact-answer scoring, retention NLL collection, and raw score persistence. |
| `src/ppmoe/recurrence/gates.py` | Informativeness checks, paired bootstraps, discovery/confirmation decisions, timing validity, and cost ledger. |
| `src/ppmoe/recurrence/experiment.py` | G0-G6 dependency graph and stage orchestration. |
| `src/ppmoe/recurrence/report.py` | Validate persisted evidence and render honest JSON/Markdown reports. |
| `scripts/run_recurrence_v0.py` | Thin argument parser, dependency construction, exit-code mapping, and resolved-path output. |
| `tests/test_recurrence_*.py` | Focused unit, integration, recovery, gate, and report contracts. |

---

## Phase A — freeze the control plane

### Task 0: Reconcile and freeze the planning baseline

**Files:**

- Verify: `plan.md`
- Verify: `todo.md`
- Verify: `PROJECT_LOG.md`
- Verify: `docs/archive/2026-08-30-dense-to-sparse-plan.md`
- Verify: `docs/superpowers/specs/2026-09-08-pretrained-recurrence-design.md`
- Verify: `docs/superpowers/plans/2026-09-11-pretrained-recurrence-execution.md`

**Produces:** One committed, reviewable protocol baseline from which implementation and run signatures can be derived.

- [ ] **Step 0.1: Confirm repository identity and branch**

Run:

```powershell
git rev-parse --show-toplevel
git branch --show-current
git status --short --branch
git rev-parse HEAD
```

Expected: top level is `C:\sem4\d2s`, branch is exactly `exp`, and HEAD is recorded. Stop for human intervention if another branch owns overlapping uncommitted work.

- [ ] **Step 0.2: Read the three authoritative documents in precedence order**

Read the recurrence spec, then `plan.md`, then this execution plan. Treat `todo.md` only as a progress projection. When wording conflicts, stop implementation and repair the contradiction in all three authoritative documents before proceeding.

- [ ] **Step 0.3: Verify the archived roadmap is historically exact**

Run:

```powershell
$headPlan = Join-Path $env:TEMP 'd2s-head-plan.md'
git show HEAD:plan.md | Set-Content -LiteralPath $headPlan
git diff --no-index --ignore-cr-at-eol -- $headPlan docs/archive/2026-08-30-dense-to-sparse-plan.md
```

Expected: the only semantic addition is the historical-status banner. Exit code 1 from `git diff --no-index` is expected when that banner is present; inspect the diff rather than treating it as failure.

- [ ] **Step 0.4: Recompute the four historical hashes**

Run:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath `
  artifacts/phase1-v0/stages/m0.json, `
  artifacts/phase1-v0/stages/local.json, `
  artifacts/phase1-v0/stages/rollout.json, `
  artifacts/phase1-v0_5/runs/a8ab03b/stages/local.json
```

Expected hashes, compared case-insensitively:

```text
B9BE2ECFE2B73A906682B652D65CE1C9DB094E0D23ECC825AB00E677BD3A40D0
90382EA91C74B1850FB7AEF81C6F4705DD625A8559EF0FA725BD4511FB2D8341
0F415AF16EBFFB6231041605C0A72C4A3505C6DB1F5FF832D6126B8B339E334E
5C48266C4CB5429E9B8AA3544B7AFE112EF3B198C6084CC02921AC0C7DA472B1
```

- [ ] **Step 0.5: Capture the pre-implementation verification baseline**

Run:

```powershell
uv run pytest -q
uv lock --check
git diff --check
```

Expected: the existing suite passes, the lock is current, and whitespace checking passes. A pre-existing test failure must be recorded separately and resolved or explicitly quarantined before recurrence code is added.

- [ ] **Step 0.6: Validate local Markdown references**

Check every local link in the spec, `plan.md`, `todo.md`, and this file resolves relative to its containing file. Check fenced-code markers are balanced. Do not attempt to validate external URLs as local files.

- [ ] **Step 0.7: Stage only planning artifacts**

Run:

```powershell
git add plan.md todo.md PROJECT_LOG.md docs/archive/2026-08-30-dense-to-sparse-plan.md docs/superpowers/specs/2026-09-08-pretrained-recurrence-design.md docs/superpowers/plans/2026-09-11-pretrained-recurrence-execution.md
git diff --cached --stat
git diff --cached --check
```

Expected: no source, tests, artifacts, weights, or caches are staged.

- [ ] **Step 0.8: Create the protocol-freeze commit**

Run:

```powershell
git commit -m "docs: freeze pretrained recurrence protocol"
```

Record the resulting commit SHA as `protocol_revision`; production manifests must cite it.

---

## Phase B — foundation implementation

### Task 1: Add the typed configuration and immutable provenance primitives

**Files:**

- Create: `configs/recurrence-v0.toml`
- Create: `src/ppmoe/recurrence/__init__.py`
- Create: `src/ppmoe/recurrence/config.py`
- Create: `src/ppmoe/recurrence/provenance.py`
- Create: `tests/test_recurrence_config.py`
- Create: `tests/test_recurrence_provenance.py`

**Interfaces:**

- Produces: `RecurrenceConfig.load(path: Path) -> RecurrenceConfig`
- Produces: `RecurrenceConfig.scientific_payload() -> dict[str, object]`
- Produces: `canonical_json(payload: Mapping[str, object]) -> bytes`
- Produces: `sha256_payload(payload: Mapping[str, object]) -> str`
- Produces: `write_immutable_json(path: Path, payload: Mapping[str, object]) -> str`
- Produces: `append_event(path: Path, event: Mapping[str, object]) -> int`
- Produces: `directory_size(path: Path) -> int`

- [ ] **Step 1.1: Write a complete frozen TOML fixture**

Create `configs/recurrence-v0.toml` with explicit sections for model, loop, precision, LoRA, optimizer, data, decoding, retention, statistics, resources, and stages. Store all fixed values from `plan.md`, including bootstrap seeds `9047` and `9157`, 10,000 replicates, family informativeness bounds `0.05/0.95`, maximum family invalid rate `0.10`, and minimum informative families `2`.

- [ ] **Step 1.2: Write failing exact-value and unknown-key tests**

Add tests equivalent to:

```python
def test_registered_config_loads(tmp_path: Path) -> None:
    config = RecurrenceConfig.load(Path("configs/recurrence-v0.toml"))
    assert config.model.repo_id == "Qwen/Qwen3-1.7B"
    assert config.loop.layer_start == 12
    assert config.loop.layer_stop == 16
    assert config.lora.rank == 8
    assert config.statistics.retention_bootstrap_seed == 9157
    assert config.statistics.minimum_informative_families == 2

def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    path = write_toml_copy(tmp_path, {"loop": {"layer_search": True}})
    with pytest.raises(ValueError, match="unknown.*layer_search"):
        RecurrenceConfig.load(path)

def test_registered_constant_cannot_be_overridden(tmp_path: Path) -> None:
    path = write_toml_copy(tmp_path, {"lora": {"rank": 16}})
    with pytest.raises(ValueError, match="lora.rank"):
        RecurrenceConfig.load(path)
```

`write_toml_copy` in the test must copy the registered file and replace only the requested key; it must not duplicate a second incomplete configuration schema.

- [ ] **Step 1.3: Run the configuration tests and observe failure**

Run:

```powershell
uv run pytest -q tests/test_recurrence_config.py
```

Expected: collection/import fails because the recurrence package and config types do not yet exist.

- [ ] **Step 1.4: Implement frozen dataclasses and strict parsing**

Use `@dataclass(frozen=True, slots=True)` for each section. Load with `tomllib`, reject missing and unknown keys recursively, reject JSON/TOML booleans where integers are required, check exact registered values, and expose a canonical scientific payload that excludes paths, timestamps, hostname, and runtime measurements.

The validation entry point must follow this shape:

```python
@classmethod
def load(cls, path: Path) -> "RecurrenceConfig":
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    validate_exact_keys(raw, REGISTERED_SCHEMA)
    config = cls.from_mapping(raw)
    config.validate_registered_values()
    return config
```

- [ ] **Step 1.5: Write failing canonical-hash and immutable-write tests**

```python
def test_payload_hash_ignores_mapping_order() -> None:
    assert sha256_payload({"b": 2, "a": 1}) == sha256_payload({"a": 1, "b": 2})

def test_immutable_writer_accepts_identical_replay(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    first = write_immutable_json(path, {"value": 1})
    second = write_immutable_json(path, {"value": 1})
    assert first == second == sha256_file(path)

def test_immutable_writer_rejects_different_payload(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_immutable_json(path, {"value": 1})
    with pytest.raises(FileExistsError, match="immutable"):
        write_immutable_json(path, {"value": 2})
```

- [ ] **Step 1.6: Implement canonical serialization and immutable writes**

Serialize UTF-8 JSON with sorted keys, compact separators, `allow_nan=False`, and one trailing newline. Write to a sibling temporary file, flush, call `os.fsync`, and atomically replace only when the destination does not exist. If it exists, compare bytes: return its hash if identical and raise `FileExistsError` otherwise.

- [ ] **Step 1.7: Implement monotonic append-only events**

Open `events.jsonl` under an exclusive process-local file lock, read the last valid index, assign `index+1`, add an RFC3339 UTC timestamp, append one line, flush, and fsync. Reject a malformed existing final line rather than truncating history.

- [ ] **Step 1.8: Run focused tests**

Run:

```powershell
uv run pytest -q tests/test_recurrence_config.py tests/test_recurrence_provenance.py
```

Expected: all pass.

- [ ] **Step 1.9: Do not commit yet**

Task 4 owns the foundation milestone commit after source and both data paths are independently testable.

### Task 2: Resolve and verify the immutable model source

**Files:**

- Modify: `src/ppmoe/recurrence/provenance.py`
- Modify: `tests/test_recurrence_provenance.py`
- Create: `tests/test_recurrence_source.py`

**Interfaces:**

- Consumes: `RecurrenceConfig`, `write_immutable_json`, `sha256_file`
- Produces: `resolve_model_revision(repo_id: str, requested_revision: str) -> str`
- Produces: `validate_revision(value: str) -> None`
- Produces: `build_source_manifest(config: RecurrenceConfig, snapshot_dir: Path, output_path: Path) -> dict[str, object]`
- Produces: `validate_qwen3_config(config_dict: Mapping[str, object]) -> None`
- Produces: `compute_run_id(source_hash: str, data_hash: str, config_hash: str, code_hash: str) -> str`

- [ ] **Step 2.1: Write failing immutable-revision tests with fake metadata**

```python
def test_resolver_returns_full_commit_sha() -> None:
    assert resolve_model_revision(
        "Qwen/Qwen3-1.7B", "main", api=FakeApi(sha="a" * 40)
    ) == "a" * 40

@pytest.mark.parametrize("value", ["main", "refs/pr/1", "abc123", "g" * 40])
def test_source_manifest_rejects_mutable_or_invalid_revision(value: str) -> None:
    with pytest.raises(ValueError, match="full immutable"):
        validate_revision(value)
```

- [ ] **Step 2.2: Write failing architecture-contract tests**

Use the official expected configuration as the passing fixture. Parameterize one-field mutations for architecture, layer count, hidden/intermediate sizes, attention heads, KV heads, head dimension, sliding-window activation, and vocabulary mismatch between config and embeddings. Every mutation must identify its field in the exception.

- [ ] **Step 2.3: Run source tests and observe failure**

Run:

```powershell
uv run pytest -q tests/test_recurrence_source.py
```

- [ ] **Step 2.4: Implement revision resolution before loading weights**

The source stage must query metadata, validate a lowercase or uppercase 40-hex SHA, call tokenizer/config/model snapshot APIs only with that SHA, and reject any resulting file whose resolved cache path does not identify the same revision. It must never pass mutable `main` to `from_pretrained` after resolution.

- [ ] **Step 2.5: Implement the source manifest**

Define frozen `SourceFile(relative_path: str, bytes: int, sha256: str)` and `SourceManifest` dataclasses. `SourceManifest` contains schema version literal 1; repo ID literal `Qwen/Qwen3-1.7B`; validated 40-hex revision; model class literal `Qwen3ForCausalLM`; a tuple of `SourceFile`; license relative path and SHA; concrete Python/torch/Transformers versions; concrete GPU name/VRAM/driver; protocol Git revision; and config/code SHA256 values. Serialize with `dataclasses.asdict` through the canonical writer. Sort file entries by relative path. Hash config, tokenizer files, weight shards, shard index, generation config, Python source files, lockfile, and license snapshot.

- [ ] **Step 2.6: Test run-id stability and signature sensitivity**

Assert the 12-hex run ID is unchanged by timestamps or mapping order and changes when source, data, config, or runtime-code hashes change. Assert a recovery-only revision must be supplied as an explicit secondary provenance record and cannot impersonate the original code hash.

- [ ] **Step 2.7: Run focused tests**

Run:

```powershell
uv run pytest -q tests/test_recurrence_config.py tests/test_recurrence_provenance.py tests/test_recurrence_source.py
```

- [ ] **Step 2.8: Do not contact the network in tests**

All unit tests use fake metadata and a temporary local snapshot. Real source resolution occurs only in production Task 15.

### Task 3: Generate and freeze the exact-answer datasets

**Files:**

- Create: `src/ppmoe/recurrence/data.py`
- Create: `tests/test_recurrence_data.py`

**Interfaces:**

- Produces: immutable `Example(id, family, split, structure_id, prompt, answer, latent_spec)`
- Produces: immutable `RenderedExample(example: Example, prompt_input_ids: tuple[int, ...], train_input_ids: tuple[int, ...], labels: tuple[int, ...])`
- Produces: `make_examples(config: RecurrenceConfig) -> dict[str, list[Example]]`
- Produces: `interpret(latent_spec: Mapping[str, object], family: str) -> object`
- Produces: `parse_answer(text: str, example: Example) -> object`
- Produces: `render_example(tokenizer: object, example: Example) -> RenderedExample`
- Produces: `freeze_examples(root: Path, examples: Mapping[str, Sequence[Example]]) -> dict[str, object]`

- [ ] **Step 3.1: Write deterministic count and disjointness tests**

```python
def test_registered_generation_is_deterministic() -> None:
    first = make_examples(registered_config())
    second = make_examples(registered_config())
    assert first == second
    assert Counter(x.family for x in first["train"]) == {
        "expressions": 2667,
        "lists": 2667,
        "graphs": 2666,
    }
    assert Counter(x.family for x in first["dev"]) == {
        "expressions": 200,
        "lists": 200,
        "graphs": 200,
    }
    assert Counter(x.family for x in first["test"]) == {
        "expressions": 800,
        "lists": 800,
        "graphs": 800,
    }
    ids = [x.id for rows in first.values() for x in rows]
    assert len(ids) == len(set(ids))
```

- [ ] **Step 3.2: Write structure-shift tests**

Assert expression leaves are train `{3,4}`, dev `{5}`, test `{6}`; list input/program lengths are train `3..5/1..2`, dev `6/3`, test `7/4`; graph vertex counts are train `{5,6}`, dev `{7}`, test `{8}`. Assert expression shapes alternate, graph labels remain balanced within one row, and list programs exclude adjacent double reverse and identity results.

- [ ] **Step 3.3: Write strict parser tests**

```python
def test_expression_parser_requires_whole_string() -> None:
    example = example_for("expressions", answer="12")
    assert parse_answer(" 12\n", example) == 12
    for bad in ["+12", "012", "The answer is 12", "12.", "12 13"]:
        with pytest.raises(ValueError):
            parse_answer(bad, example)

def test_list_parser_rejects_bool_and_wrong_length() -> None:
    example = example_for("lists", answer="[1,2,3]")
    for bad in ["[1, true, 3]", "[1,2]", "[1, 11, 3]", "result=[1,2,3]"]:
        with pytest.raises(ValueError):
            parse_answer(bad, example)

def test_graph_parser_is_case_insensitive_but_not_extractive() -> None:
    example = example_for("graphs", answer="yes")
    assert parse_answer(" YES ", example) == "yes"
    with pytest.raises(ValueError):
        parse_answer("the answer is yes", example)
```

- [ ] **Step 3.4: Run data tests and observe failure**

Run:

```powershell
uv run pytest -q tests/test_recurrence_data.py
```

- [ ] **Step 3.5: Implement expression generation and interpreter**

Use independent `random.Random(101/201/301)` streams. Generate the registered fully parenthesized shapes, sample leaves and operators exactly, evaluate with recursive integer functions, reject `abs(answer)>9999`, and never call `eval`.

- [ ] **Step 3.6: Implement list generation and interpreter**

Represent operations as exact strings in `{"ROTL1", "REVERSE", "ADD1"}`. Interpret a copied list step by step, with `ADD1` modulo 10. Reject registered invalid structures before ID generation.

- [ ] **Step 3.7: Implement directed-graph generation and BFS**

Generate each ordered distinct edge independently at probability 0.20, sort edges, sample distinct query vertices, compute reachability with a queue, and accept rows according to alternating desired labels without exposing that target in the prompt.

- [ ] **Step 3.8: Implement canonical IDs and attempt caps**

Set `id=sha256_payload({"family": family, "latent_spec": latent_spec})`. Track IDs globally. Abort a family/split after `100 * requested_count` attempts, with the exact family and split in the error.

- [ ] **Step 3.9: Write rendering and mask-alignment tests**

Use a fake tokenizer to assert one user turn, `enable_thinking=False`, generation prefix followed by answer tokens and one EOS, `-100` prompt labels, shifted prediction alignment, answer+EOS length at most 32, and total length at most 512. Assert no truncation path exists.

- [ ] **Step 3.10: Implement rendering and immutable JSONL freezing**

Write rows in deterministic family/order sequence. JSONL uses canonical compact JSON and one newline per row. Write to a staging sibling, fsync, atomically rename, then hash. `data/manifest.json` records counts, hashes, generator constants, tokenizer revision/hash, rejection counts, maximum token lengths, and split-ID set hashes.

- [ ] **Step 3.11: Run focused tests**

Run:

```powershell
uv run pytest -q tests/test_recurrence_data.py tests/test_recurrence_config.py tests/test_recurrence_provenance.py
```

- [ ] **Step 3.12: Do not commit yet**

Task 4 completes the foundation slice and owns its milestone commit.

### Task 4: Recover and materialize the FineWeb retention stream

**Files:**

- Modify: `src/ppmoe/recurrence/data.py`
- Modify: `tests/test_recurrence_data.py`
- Create: `tests/test_recurrence_retention.py`

**Interfaces:**

- Consumes: existing Phase 1 FineWeb document manifests and `ppmoe.data.token_stream`
- Produces: `recover_retention_documents(old_root: Path, dataset_loader: Callable[..., Iterable[Mapping[str, object]]]) -> tuple[DocumentRecord, ...]`
- Produces: `materialize_retention(tokenizer: object, documents: Sequence[DocumentRecord], output_dir: Path) -> dict[str, object]`
- Produces: `load_retention_chunks(output_dir: Path) -> tuple[RetentionChunk, ...]`
- Produces: immutable `RetentionChunk(index, input_ids, source_document_ids, scored_tokens)`
- Produces: `RetentionSourceError(RuntimeError)` for missing, mutable, or hash-mismatched source text

- [ ] **Step 4.1: Write failing provenance-recovery tests**

Construct a temporary old manifest with document IDs and text hashes. Assert recovery accepts cached raw text only when every ID and text hash matches. Assert missing text, changed text, duplicate IDs, token-ID-only input, and a mutable dataset revision each raise a retention-source error.

- [ ] **Step 4.2: Write failing 20,000-token materialization tests**

Use a deterministic fake tokenizer and enough fake documents. Assert:

```python
manifest = materialize_retention(tokenizer, documents, tmp_path)
assert manifest["valid_source_tokens"] == 20_000
assert sum(chunk.scored_tokens for chunk in load_retention_chunks(tmp_path)) == (
    20_000 - manifest["chunk_count"]
)
assert max(len(chunk.input_ids) for chunk in load_retention_chunks(tmp_path)) <= 512
```

Also assert one EOS is inserted between documents, the final partial chunk is retained, the first token of each chunk is excluded from scoring, and a second materialization produces byte-identical files.

- [ ] **Step 4.3: Run retention tests and observe failure**

Run:

```powershell
uv run pytest -q tests/test_recurrence_retention.py
```

- [ ] **Step 4.4: Implement exact document recovery**

Reuse old document identifiers, not old Qwen3.5 token IDs. Prefer locally cached raw text when its hash matches; otherwise stream the pinned original dataset revision until all registered IDs are found. Preserve original ID order. If any document remains missing, write no completed retention manifest and raise `RetentionSourceError`.

- [ ] **Step 4.5: Implement Qwen3 re-tokenization and chunk provenance**

Tokenize text without silently truncating individual documents, insert EOS between documents, stop exactly after 20,000 valid source tokens, and chunk into consecutive windows of at most 512. Persist token IDs plus the contributing document IDs for each chunk so the report can expose provenance.

- [ ] **Step 4.6: Test bootstrap-unit readiness**

Assert every chunk has a stable index, at least one scored token, and an unambiguous count. The later paired retention bootstrap resamples these chunk indices; do not aggregate away chunk boundaries.

- [ ] **Step 4.7: Run the complete foundation tests**

Run:

```powershell
uv run pytest -q tests/test_recurrence_config.py tests/test_recurrence_provenance.py tests/test_recurrence_source.py tests/test_recurrence_data.py tests/test_recurrence_retention.py
uv lock --check
git diff --check
```

- [ ] **Step 4.8: Review and commit the foundation milestone**

Run:

```powershell
git status --short
git diff --stat
git add configs/recurrence-v0.toml src/ppmoe/recurrence/__init__.py src/ppmoe/recurrence/config.py src/ppmoe/recurrence/provenance.py src/ppmoe/recurrence/data.py tests/test_recurrence_config.py tests/test_recurrence_provenance.py tests/test_recurrence_source.py tests/test_recurrence_data.py tests/test_recurrence_retention.py
git diff --cached --check
git commit -m "feat: add recurrence inputs and provenance"
```

Do not stage generated datasets or source snapshots.

---

## Phase C — model and training implementation

### Task 5: Implement the FP32 LoRA primitive and deterministic installation

**Files:**

- Create: `src/ppmoe/recurrence/lora.py`
- Create: `tests/test_recurrence_lora.py`

**Interfaces:**

- Produces: `LoRALinear(base: nn.Linear, rank: int = 8, alpha: float = 16.0)`
- Produces: `install_adapters(model: nn.Module, layer_start: int, layer_stop: int, seed: int) -> tuple[nn.Parameter, ...]`
- Produces: `adapter_state_dict(model: nn.Module) -> dict[str, Tensor]`
- Produces: `adapter_sha256(model: nn.Module) -> str`
- Produces: `set_adapters_enabled(model: nn.Module, enabled: bool) -> None`

- [ ] **Step 5.1: Write the zero-B identity test**

```python
def test_zero_b_adapter_matches_base_exactly() -> None:
    base = nn.Linear(3, 2, bias=False).to(dtype=torch.bfloat16)
    layer = LoRALinear(base, rank=8, alpha=16)
    x = torch.randn(4, 3, dtype=torch.bfloat16)
    assert torch.equal(layer(x), base(x))
    assert layer.a.dtype == torch.float32
    assert layer.b.dtype == torch.float32
```

- [ ] **Step 5.2: Write an analytical nonzero-adapter test**

Set a small base matrix, A, B, and input explicitly. Compare the module output with `base(x) + 2 * ((x.float() @ A.T) @ B.T)` cast to the base output dtype. Use exact representable values and `torch.testing.assert_close` with zero relative tolerance.

- [ ] **Step 5.3: Write freezing and target-selection tests**

Build a six-layer tiny Qwen3 model. Install adapters in `[2,4)`. Assert exactly 12 target projections are wrapped, no `k_proj` is wrapped, every base parameter has `requires_grad=False`, every A/B parameter is FP32 and trainable, and optimizer candidates contain A/B only.

- [ ] **Step 5.4: Run tests and observe failure**

Run:

```powershell
uv run pytest -q tests/test_recurrence_lora.py
```

- [ ] **Step 5.5: Implement `LoRALinear` without registering the base twice**

Store the original linear module as the single `base` child. Allocate A with shape `[rank,in_features]` and B `[out_features,rank]`. Use a seed-controlled `torch.Generator` for Kaiming-uniform A and zero B. The forward path is:

```python
base_output = self.base(x)
with torch.autocast(device_type=x.device.type, enabled=False):
    hidden = F.linear(x.float(), self.a)
    delta = F.linear(hidden, self.b) * self.scale
return base_output + delta.to(dtype=base_output.dtype)
```

For CPU, enter `nullcontext()` rather than constructing unsupported autocast settings.

- [ ] **Step 5.6: Implement deterministic installation**

Iterate layer indices and projection names in fixed numeric/lexical order. Replace each projection once, reject an already wrapped projection, and assert exactly 24 production projections for `[12,16)`. Preserve module device and base dtype; create A/B on the same device in FP32.

- [ ] **Step 5.7: Implement stable adapter hashing**

Hash ordered names, dtype, shape, and contiguous CPU bytes. Assert identical seeded installations hash equally and changing one element changes the hash.

- [ ] **Step 5.8: Run LoRA tests**

Run:

```powershell
uv run pytest -q tests/test_recurrence_lora.py
```

- [ ] **Step 5.9: Do not commit yet**

Task 9 owns the model/training milestone commit.

### Task 6: Implement the exact recurrent Qwen3 wrapper

**Files:**

- Create: `src/ppmoe/recurrence/model.py`
- Create: `tests/test_recurrence_model.py`

**Interfaces:**

- Produces: `apply_block(block: Callable[[Tensor], Tensor], hidden: Tensor, mode: str) -> Tensor`
- Produces: `RecurrentQwen3(stock_model: Qwen3ForCausalLM, layer_start: int = 12, layer_stop: int = 16)`
- Produces: `RecurrentQwen3.hidden(input_ids: Tensor, mode: str) -> Tensor`
- Produces: `RecurrentQwen3.forward(input_ids: Tensor, mode: str) -> Tensor`
- Consumes later: `answer_loss` added in Task 7

- [ ] **Step 6.1: Write the algebra test first**

```python
def test_recurrence_algebra() -> None:
    h = torch.tensor([2.0])
    block = lambda x: 2 * x
    assert torch.equal(apply_block(block, h, "single"), torch.tensor([4.0]))
    assert torch.equal(apply_block(block, h, "repeat"), torch.tensor([8.0]))
    assert torch.equal(apply_block(block, h, "damped"), torch.tensor([4.5]))
    with pytest.raises(ValueError, match="unsupported mode"):
        apply_block(block, h, "other")
```

- [ ] **Step 6.2: Build the local tiny-model fixture**

Construct `Qwen3Config` with six layers, hidden size 32, intermediate size 64, four query heads, two KV heads, head dimension 8, vocabulary 97, no sliding attention, eager attention, and caches disabled. Use test block `[2,4)` and FP32 evaluation. No network call is allowed.

- [ ] **Step 6.3: Write stock-equivalence and ownership tests**

For fixed token IDs, compare stock logits with wrapper `mode="single"` at `rtol=1e-5, atol=1e-6`. Assert the stock model is registered once, wrapper parameter IDs equal stock parameter IDs, and `repeat` does not create parameters or copy decoder layers.

- [ ] **Step 6.4: Write layer-call and terminal-module tests**

Instrument all layers, final norm, and LM head. In single mode expect each decoder layer once; in repeat/damped expect block layers twice and all others once. Expect final norm and LM head once in every mode.

- [ ] **Step 6.5: Write the causality test**

```python
@pytest.mark.parametrize("mode", ["single", "repeat", "damped"])
def test_suffix_tokens_cannot_change_prefix_logits(mode: str) -> None:
    left = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    right = torch.tensor([[1, 2, 3, 4, 5, 8, 9]])
    torch.testing.assert_close(
        wrapper(left, mode)[:, :5],
        wrapper(right, mode)[:, :5],
        rtol=1e-5,
        atol=1e-6,
    )
```

- [ ] **Step 6.6: Run model tests and observe failure**

Run:

```powershell
uv run pytest -q tests/test_recurrence_model.py
```

- [ ] **Step 6.7: Implement explicit prefix/block/suffix execution**

Validate shape `[1,T]`, `1<=T<=512`, no padding, and supported mode. Embed tokens, construct zero-based position IDs, call installed Transformers `create_causal_mask`, compute rotary embeddings once, execute layers `0:12`, apply the block closure over `12:16`, execute `16:28`, final norm, then LM head. Pass identical mask, positions, and rotary embeddings to both block calls. Always pass `past_key_values=None` and `use_cache=False`.

- [ ] **Step 6.8: Reject unsupported state explicitly**

Tests must prove that batch size greater than one, padding/attention-mask input, external cache input, sliding attention, invalid prefix length, invalid loop range, and unsupported modes raise before partial execution.

- [ ] **Step 6.9: Run model and LoRA tests together**

Run:

```powershell
uv run pytest -q tests/test_recurrence_model.py tests/test_recurrence_lora.py
```

- [ ] **Step 6.10: Do not commit yet**

Task 9 owns the model/training milestone commit.

### Task 7: Implement causal answer-only loss and gradient verification

**Files:**

- Modify: `src/ppmoe/recurrence/model.py`
- Modify: `tests/test_recurrence_model.py`
- Create: `tests/test_recurrence_loss.py`

**Interfaces:**

- Produces: `RecurrentQwen3.answer_loss(input_ids: Tensor, labels: Tensor, mode: str) -> Tensor`
- Produces: `chunked_causal_loss(hidden: Tensor, labels: Tensor, lm_head: nn.Module, chunk_size: int = 16) -> Tensor`

- [ ] **Step 7.1: Write explicit causal-shift tests**

Create hidden states and a scripted head where each position has a known correct next token. Assert supervised label at index `j` uses hidden index `j-1`; replacing hidden `j` alone cannot change that label's loss. Reject a supervised label at index 0 because it has no causal predecessor.

- [ ] **Step 7.2: Compare chunked and full supervised CE**

```python
def test_chunked_loss_matches_full_shifted_cross_entropy() -> None:
    full = F.cross_entropy(
        head(hidden[:, :-1]).float().reshape(-1, vocab),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    actual = chunked_causal_loss(hidden, labels, head, chunk_size=2)
    torch.testing.assert_close(actual, full, rtol=1e-6, atol=1e-7)
```

Use one example so PyTorch's supervised-token mean equals the required per-example mean.

- [ ] **Step 7.3: Test prompt and EOS masking**

Assert changing prompt labels left at `-100` has no effect, answer tokens contribute, and the single EOS label contributes. Assert zero supervised positions, multiple EOS labels, shape mismatch, or nonfinite loss raises.

- [ ] **Step 7.4: Write shared-pass gradient tests**

Compare adapter gradients from wrapper repeat mode with a direct call of the same tiny block twice. Add a tiny FP64 finite-difference check for one adapter element. Assert suffix-layer execution is differentiable with respect to its input while suffix parameters remain frozen.

- [ ] **Step 7.5: Run loss tests and observe failure**

Run:

```powershell
uv run pytest -q tests/test_recurrence_loss.py
```

- [ ] **Step 7.6: Implement chunked projection and per-example mean**

Select indices where `labels[:,1:] != -100`, gather `hidden[:, :-1]` at those indices, project groups of at most 16 positions through the BF16 head, compute FP32 `log_softmax`/NLL sums, and divide by the exact supervised-position count. Do not allocate `[1,T,V]` during training.

- [ ] **Step 7.7: Preserve gradients through the suffix**

Execute the frozen prefix under `torch.no_grad`, then detach and re-enable gradient on the prefix output before the trainable block. Execute both block calls and all suffix layers with autograd enabled. Freeze their parameters rather than enclosing their forward paths in `no_grad`.

- [ ] **Step 7.8: Run all core math tests**

Run:

```powershell
uv run pytest -q tests/test_recurrence_lora.py tests/test_recurrence_model.py tests/test_recurrence_loss.py
```

- [ ] **Step 7.9: Do not commit yet**

Task 9 owns the model/training milestone commit.

### Task 8: Implement append-only events and atomic recoverable checkpoints

**Files:**

- Create: `src/ppmoe/recurrence/train.py`
- Create: `tests/test_recurrence_train.py`

**Interfaces:**

- Produces: `TrainingState(update: int, cursor: int, elapsed_cuda_seconds: float, attempt: int)`
- Produces: `save_checkpoint(directory: Path, model: nn.Module, optimizer: Optimizer, state: TrainingState, signatures: Mapping[str, str]) -> Path`
- Produces: `load_checkpoint(directory: Path, model: nn.Module, optimizer: Optimizer, signatures: Mapping[str, str]) -> TrainingState`
- Produces later: `train_arm(model: RecurrentQwen3, examples: Sequence[RenderedExample], arm: str, seed: int, steps: int, checkpoint_dir: Path, event_path: Path, signatures: Mapping[str, str]) -> dict[str, object]`, completed in Task 9

- [ ] **Step 8.1: Write atomic checkpoint tests**

Create a valid `latest.pt`, inject a failure before temporary-file rename, and assert the previous `latest.pt` still loads. Assert `previous.pt` remains available after a successful replacement. Assert checkpoint payload includes adapter/optimizer states, update, cursor, scheduler state, Python/NumPy/CPU/CUDA RNG, elapsed cost, attempt, initialization hash, and all signatures.

- [ ] **Step 8.2: Write signature mismatch tests**

Parameterize source, data, config, code, arm, and seed changes. Each mismatch must prevent model or optimizer mutation and name the mismatched signature.

- [ ] **Step 8.3: Write append-only failure-history tests**

Record attempt 1 failure, then attempt 2 success. Assert both attempts remain in `events.jsonl`, the failure JSON remains, and the completed training manifest references both. A successful retry may resolve a failure but never erase it.

- [ ] **Step 8.4: Run checkpoint tests and observe failure**

Run:

```powershell
uv run pytest -q tests/test_recurrence_train.py -k "checkpoint or signature or failure"
```

- [ ] **Step 8.5: Implement trusted local checkpoint serialization**

Write with `torch.save` to a sibling temporary file, flush and fsync its file handle, move current `latest.pt` to `previous.pt`, then atomically replace `latest.pt`. Write a small immutable JSON sidecar with the checkpoint SHA256 and signatures. Load only from the run's own trusted checkpoint directory.

- [ ] **Step 8.6: Implement complete-step recovery semantics**

Increment update/cursor only after optimizer step, scheduler update, and zeroing gradients complete. Write progress every 10 updates and checkpoints every 50 updates plus orderly stop. After interruption, replay the last uncommitted update; never infer progress from a partially written event.

- [ ] **Step 8.7: Run checkpoint tests**

Run:

```powershell
uv run pytest -q tests/test_recurrence_train.py -k "checkpoint or signature or failure"
```

- [ ] **Step 8.8: Do not commit yet**

Task 9 owns the model/training milestone commit.

### Task 9: Implement the deterministic training loop and resume equivalence

**Files:**

- Modify: `src/ppmoe/recurrence/train.py`
- Modify: `tests/test_recurrence_train.py`

**Interfaces:**

- Consumes: `RecurrentQwen3.answer_loss`, adapter helpers, checkpoints, frozen examples
- Produces: `train_arm(model: RecurrentQwen3, examples: Sequence[RenderedExample], arm: str, seed: int, steps: int, checkpoint_dir: Path, event_path: Path, signatures: Mapping[str, str]) -> dict[str, object]`

- [ ] **Step 9.1: Write optimizer and schedule tests**

Assert optimizer contains only adapter A/B parameters; betas, epsilon, weight decay, and base LR exactly match protocol. For updates 1, 10, 20, and 21, assert learning rates are `5e-6`, `5e-5`, `1e-4`, and `1e-4` respectively.

- [ ] **Step 9.2: Write accumulation-semantics tests**

For eight one-example losses `l0..l7`, assert the accumulated gradient equals backpropagation through `(l0+...+l7)/8`. Assert an optimizer step cannot occur with fewer or more than eight examples and that each A1/A2 seed consumes exactly one seeded permutation of all 8,000 unique IDs.

- [ ] **Step 9.3: Write A3 cycling tests**

For `n3>1000`, assert A3 uses the same seed-specific 8,000-ID permutation, then cycles from its first ID without reshuffling. Assert no unseen example or alternate seed enters the stream.

- [ ] **Step 9.4: Write uninterrupted-versus-resumed equivalence**

Train a deterministic CPU fixture for four updates. Separately train two updates, save, reconstruct the model/optimizer, resume for two. Assert exact update/cursor equality and bitwise equality of adapter tensors, optimizer tensors, and saved RNG states.

- [ ] **Step 9.5: Write shared-initialization tests**

For each seed, save one initialized adapter state and load it independently into A1, A2, and A3. Assert identical initialization hashes before training. Assert different seeds have different hashes. Assert A1/A2 production begins only after disposable pilot adapters are destroyed and initial state is reloaded.

- [ ] **Step 9.6: Run trainer tests and observe failure**

Run:

```powershell
uv run pytest -q tests/test_recurrence_train.py
```

- [ ] **Step 9.7: Implement deterministic seeding and data order**

Seed Python, NumPy, torch CPU, and every CUDA device. Set deterministic algorithms where supported; record any operation that cannot be forced deterministic. Construct the data permutation once per seed and hash the ordered ID list.

- [ ] **Step 9.8: Implement the optimizer loop**

For each update, run eight sequential examples, divide each per-example mean loss by eight before backward, use non-reentrant activation checkpointing for each trainable block call and suffix layer, clip the global adapter norm to 1.0, step optimizer, apply the registered LR, zero gradients, synchronize CUDA, update the cumulative ledger, and persist progress/checkpoints on schedule.

- [ ] **Step 9.9: Implement fault boundaries**

On CUDA fault, persist failure details and cost if possible, exit the process, and permit at most one fresh-process retry for that stage. A second fault sets `failed_environment`. Do not catch a CUDA fault and continue using the same process.

- [ ] **Step 9.10: Return a complete training manifest**

Include arm, seed, requested/completed updates, unique example count, total example presentations, ordered-ID hash, initialization/final adapter hashes, trainable parameter count, peak allocated/reserved memory, synchronized CUDA seconds, checkpoint hashes, attempts, failures, and completion status.

- [ ] **Step 9.11: Run the full model/training slice**

Run:

```powershell
uv run pytest -q tests/test_recurrence_lora.py tests/test_recurrence_model.py tests/test_recurrence_loss.py tests/test_recurrence_train.py
uv lock --check
git diff --check
```

- [ ] **Step 9.12: Review and commit the model/training milestone**

Run:

```powershell
git status --short
git diff --stat
git add src/ppmoe/recurrence/lora.py src/ppmoe/recurrence/model.py src/ppmoe/recurrence/train.py tests/test_recurrence_lora.py tests/test_recurrence_model.py tests/test_recurrence_loss.py tests/test_recurrence_train.py
git diff --cached --check
git commit -m "feat: add recurrent model and recoverable training"
```

---

## Phase D — evaluation, gates, orchestration, and reporting

### Task 10: Implement greedy generation and exact-answer evaluation

**Files:**

- Create: `src/ppmoe/recurrence/evaluate.py`
- Create: `tests/test_recurrence_evaluate.py`

**Interfaces:**

- Produces: `generate_answer(model: RecurrentQwen3, input_ids: Tensor, mode: str, eos_token_id: int, max_new_tokens: int = 32) -> dict[str, object]`
- Produces: `evaluate_examples(model: RecurrentQwen3, examples: Sequence[RenderedExample], mode: str, output_path: Path) -> dict[str, object]`
- Produces: `evaluate_retention(model: RecurrentQwen3, chunks: Sequence[RetentionChunk], mode: str, output_path: Path) -> dict[str, object]`

- [ ] **Step 10.1: Write scripted greedy-generation tests**

Use a fake model returning predetermined logits. Assert argmax decoding, EOS termination, exact generated-token count, full-prefix recomputation on every token, no gold answer in any model call, and no KV-cache argument.

```python
def test_generation_recomputes_full_prefix() -> None:
    model = ScriptedModel(tokens=[7, 8, EOS])
    result = generate_answer(model, torch.tensor([[1, 2]]), "repeat", EOS, 32)
    assert model.seen_inputs == [[1, 2], [1, 2, 7], [1, 2, 7, 8]]
    assert result["token_ids"] == [7, 8, EOS]
    assert result["stop_reason"] == "eos"
```

- [ ] **Step 10.2: Test cap and invalid-output handling**

If EOS is not emitted in 32 tokens, return `stop_reason="length"`; parsing must classify it as `missing_eos_at_cap` and score it wrong. Test separate reason codes for invalid format, wrong value, empty completion, decode failure, and correct answer.

- [ ] **Step 10.3: Test score-file reconciliation**

Write one record per expected ID with prompt hash, completion token IDs/text, parsed value or parse error, correctness, reason code, latency, arm, seed, split, mode, and signatures. Assert duplicate, missing, unexpected, or out-of-order IDs prevent a complete aggregate.

- [ ] **Step 10.4: Run evaluator tests and observe failure**

Run:

```powershell
uv run pytest -q tests/test_recurrence_evaluate.py -k "generation or example"
```

- [ ] **Step 10.5: Implement full-prefix greedy generation**

At each step call the wrapper with the entire current token sequence, select `argmax` from the final-position logits, append the token, and stop at EOS or 32. Synchronize CUDA around end-to-end per-example timing. Decode completion tokens excluding EOS; never use Transformers `generate` because cache semantics and hidden wrapper behavior must remain explicit.

- [ ] **Step 10.6: Implement exact-answer evaluation**

Load only the authorized split, invoke generation without answer fields, parse the whole completion, persist every record immediately to append-only JSONL, and reconcile the exact expected set before writing the immutable summary. An execution exception records a failure and prevents `complete=true`; it is never converted into an incorrect prediction.

- [ ] **Step 10.7: Write retention NLL tests**

For two hand-computed chunks, compare FP64 NLL sums, scored-token counts, total mean NLL, and `exp(mean_nll)`. Assert the first token of each chunk is excluded and no boundary loss is invented between chunks.

- [ ] **Step 10.8: Implement per-chunk retention scoring**

For each frozen chunk, run full logits under the requested mode, apply explicit next-token shift, sum FP64 NLL, and persist `chunk_index`, token count, NLL sum, mean NLL, document IDs, arm, seed, and signatures. Aggregate only after every expected chunk exists once.

- [ ] **Step 10.9: Run evaluator tests**

Run:

```powershell
uv run pytest -q tests/test_recurrence_evaluate.py
```

- [ ] **Step 10.10: Do not commit yet**

Task 14 owns the evaluation/orchestration milestone commit.

### Task 11: Implement accuracy, informativeness, paired bootstrap, and decision gates

**Files:**

- Create: `src/ppmoe/recurrence/gates.py`
- Create: `tests/test_recurrence_gates.py`

**Interfaces:**

- Produces: `summarize_exact_scores(records: Sequence[Mapping[str, object]]) -> dict[str, object]`
- Produces: `is_informative_family(accuracy: float, invalid_rate: float) -> bool`
- Produces: `classify_informative_families(f0_summary: Mapping[str, object]) -> dict[str, object]`
- Produces: `paired_accuracy_bootstrap(records_by_arm_seed: Mapping[tuple[str, int], Sequence[Mapping[str, object]]], replicates: int = 10_000, seed: int = 9047) -> dict[str, object]`
- Produces: `paired_retention_bootstrap(candidate_chunks: Sequence[Mapping[str, object]], f0_chunks: Sequence[Mapping[str, object]], replicates: int = 10_000, seed: int = 9157) -> dict[str, object]`
- Produces: `discovery_decision(scores: Mapping[str, object], retention: Mapping[str, object], timing: Mapping[str, object]) -> dict[str, object]`
- Produces: `confirmation_decision(scores: Mapping[str, object], retention: Mapping[str, object], timing: Mapping[str, object]) -> dict[str, object]`

- [ ] **Step 11.1: Write exact macro and transition tests**

Use unequal synthetic family counts to prove macro accuracy is the unweighted mean of family accuracies, not pooled accuracy. Pair by example ID and test wrong-to-right/right-to-wrong counts. Reject mismatched ID sets.

- [ ] **Step 11.2: Write every family-informativeness boundary test**

```python
@pytest.mark.parametrize(
    ("accuracy", "invalid_rate", "informative"),
    [(0.05, 0.00, False), (0.05001, 0.10, True),
     (0.94999, 0.10, True), (0.95, 0.00, False),
     (0.50, 0.10001, False)],
)
def test_family_informativeness_boundaries(accuracy, invalid_rate, informative):
    assert is_informative_family(accuracy, invalid_rate) is informative
```

Assert G3 passes only when global macro is strictly between 5% and 95%, global invalid rate is at most 10%, and at least two families are informative. All families must remain present in summaries regardless of classification.

- [ ] **Step 11.3: Write deterministic accuracy-bootstrap tests**

Construct paired correctness vectors with known positive, zero, and negative differences. Assert exactly 10,000 replicates, seed 9047 reproducibility, within-family resampling, identical sampled IDs across arms/seeds, seed-averaged per-item differences, percentile ordering, and strict lower endpoint `>0` behavior.

- [ ] **Step 11.4: Write deterministic retention-bootstrap tests**

For candidate and F0 chunks with hand-selected NLL sums/counts, resample identical chunk indices, recompute total token-weighted mean NLL per arm, exponentiate their difference to a PPL ratio, and assert seed 9157 reproducibility. Reject mismatched chunk indices/counts, nonfinite NLLs, zero scored tokens, and replicate counts other than 10,000 in a production config.

- [ ] **Step 11.5: Run gate tests and observe failure**

Run:

```powershell
uv run pytest -q tests/test_recurrence_gates.py
```

- [ ] **Step 11.6: Implement summary and bootstrap functions**

Sort families and IDs before creating NumPy arrays. Use `np.random.default_rng(seed)`. For accuracy, sample indices independently within each family and reuse each sample across all arm/seed vectors. For retention, sample chunk indices and recompute `exp(sum_nll_candidate/sum_tokens - sum_nll_f0/sum_tokens)` for each replicate. Use the 2.5th and 97.5th percentiles without rounding before decisions.

- [ ] **Step 11.7: Encode discovery gates as named predicates**

Return one boolean and measured value for each requirement:

```text
a2_minus_f0_macro >= 0.02
a2_minus_a1_macro >= 0.02
a2_minus_a3_macro >= 0.02
every A2-F0 family delta >= -0.02
every A2-A1 family delta >= -0.02
a2_f0_ppl_ratio_point <= 1.02
a2_f0_ppl_ratio_ci_high <= 1.02
0.90 <= a3_a2_realized_training_time_ratio <= 1.10
```

Distinguish `stopped_scientific`, `retention_uncertain`, and `timing_inconclusive` reasons in the returned decision even though the CLI maps scientific/retention stops to exit 2.

- [ ] **Step 11.8: Encode confirmation gates as named predicates**

Require each seed's A2 macro gain to be positive versus F0/A1/A3; two-seed mean gain at least 0.02 versus each; every two-seed mean family delta at least -0.02 versus F0/A1; every relevant accuracy-bootstrap lower endpoint strictly positive; both A2/F0 PPL ratio points and CI upper endpoints at most 1.02; and timing validity for both seed runs.

- [ ] **Step 11.9: Test exact threshold directions**

Parameterize equality and one-unit-around-boundary cases for every predicate. Equality passes for `>=0.02`, `>=-0.02`, `<=1.02`, and inclusive timing bounds; equality at zero fails where the lower CI endpoint must be strictly positive.

- [ ] **Step 11.10: Run gate tests**

Run:

```powershell
uv run pytest -q tests/test_recurrence_gates.py
```

- [ ] **Step 11.11: Do not commit yet**

Task 14 owns the evaluation/orchestration milestone commit.

### Task 12: Implement timing, memory, disk, and cumulative CUDA-budget accounting

**Files:**

- Modify: `src/ppmoe/recurrence/gates.py`
- Modify: `tests/test_recurrence_gates.py`

**Interfaces:**

- Produces: `CudaCostLedger.load(path: Path, cap_seconds: float) -> CudaCostLedger`
- Produces: `CudaCostLedger.record(attempt_id: str, stage: str, elapsed_seconds: float, outcome: str) -> None`
- Produces: `measure_training_step(model, batch, optimizer, warmups: int = 5, samples: int = 20) -> tuple[float, ...]`
- Produces: `derive_a3_steps(t1: Sequence[float], t2: Sequence[float]) -> int`
- Produces: `resource_decision(measurements: Mapping[str, object], ledger: CudaCostLedger, config: RecurrenceConfig) -> dict[str, object]`

- [ ] **Step 12.1: Write cumulative-ledger tests**

Assert successful, failed, and retried CUDA intervals all increase cumulative cost exactly once; idle time and download time do not. Reject negative/nonfinite intervals, duplicate attempt IDs, decreasing totals, and mutation of a finalized ledger entry.

- [ ] **Step 12.2: Write pilot timing tests**

Use a fake synchronizer/clock. Assert five warmups are excluded, exactly 20 samples are returned per arm, synchronization brackets complete forward/backward/clip/step work, and arm measurement order alternates A1/A2.

- [ ] **Step 12.3: Write A3 derivation tests**

```python
def test_a3_steps_uses_registered_ceiling_rule() -> None:
    assert derive_a3_steps([2.0] * 20, [2.5] * 20) == 1250
    assert derive_a3_steps([3.0] * 20, [2.0] * 20) == 1000
```

Reject nonpositive/nonfinite samples and sample counts other than 20.

- [ ] **Step 12.4: Write resource-gate tests**

Pass only when peak reserved VRAM is at most total VRAM minus 2 GiB, cumulative plus `1.5 * projected_remaining` is at most 24 hours, source downloads are at most 8 GiB, and recurrence artifacts are at most 20 GiB. Test equality and one-byte/one-second failures.

- [ ] **Step 12.5: Run resource tests and observe failure**

Run:

```powershell
uv run pytest -q tests/test_recurrence_gates.py -k "ledger or timing or resource or a3"
```

- [ ] **Step 12.6: Implement synchronized timing**

Use CUDA events or `torch.cuda.synchronize()` plus `time.perf_counter`; record the selected mechanism. Include forward, backward, gradient clipping, optimizer step, and synchronization. Exclude data loading by preloading pilot batches. Destroy pilot adapters and reload the saved initial state before production.

- [ ] **Step 12.7: Implement pessimistic cost projection**

Project every not-yet-completed production arm, generation example at longest observed prompt plus 32 tokens, retention chunk, retry allowance, and report-stage CUDA work. Add already spent time, multiply the remaining projection by 1.5, and stop before starting the next stage if the total exceeds 86,400 seconds.

- [ ] **Step 12.8: Run all gate tests**

Run:

```powershell
uv run pytest -q tests/test_recurrence_gates.py
```

- [ ] **Step 12.9: Do not commit yet**

Task 14 owns the evaluation/orchestration milestone commit.

### Task 13: Implement the dependency-checked stage orchestrator and CLI

**Files:**

- Create: `src/ppmoe/recurrence/experiment.py`
- Create: `scripts/run_recurrence_v0.py`
- Create: `tests/test_recurrence_experiment.py`
- Create: `tests/test_recurrence_cli.py`

**Interfaces:**

- Produces: `run_stage(stage: str, config_path: Path, artifact_root: Path, run_id: str | None, dependencies: RecurrenceDependencies | None = None) -> dict[str, object]`
- Produces: `StageError`, `StagePrerequisiteError`, `ScientificStop`, `BudgetStop`, and `EnvironmentStop`
- CLI: `--stage {source,data,smoke,pilot,frozen,discover,confirm,report}`
- CLI: `--config configs/recurrence-v0.toml`
- CLI: `--artifact-root artifacts/recurrence-v0`
- CLI: optional `--run-id` required after data finalization

- [ ] **Step 13.1: Write the exact stage-DAG tests**

Encode dependencies:

```python
STAGE_DEPENDENCIES = {
    "source": (),
    "data": ("source",),
    "smoke": ("data",),
    "pilot": ("smoke",),
    "frozen": ("pilot",),
    "discover": ("frozen",),
    "confirm": ("discover",),
    "report": (),
}
```

`report` has no passing prerequisite but must find a finalized run root or staging root. Test every missing predecessor, a stopped predecessor, incompatible signatures, and a passing predecessor.

- [ ] **Step 13.2: Test stage idempotency and immutability**

A completed compatible stage replays as read-only and returns the persisted record without reopening the model or rewriting timestamps. An incompatible or partial completed record fails. A terminal scientific/budget state blocks every downstream non-report stage.

- [ ] **Step 13.3: Test test-set sealing**

Use a fake filesystem guard to assert only `confirm` can open `data/test.jsonl`, and only after it verifies a passing immutable `selection.json`. Source, data, smoke, pilot, frozen, discover, and report must not read test rows or test score paths.

- [ ] **Step 13.4: Write CLI isolation tests**

Assert `--help` exits 0 without importing/loading torch model weights, contacting the network, creating artifacts, or requiring CUDA. Assert unknown stages, missing post-data `--run-id`, malformed run IDs, and scientific-constant override flags fail before work. There is no `all` stage.

- [ ] **Step 13.5: Run orchestration tests and observe failure**

Run:

```powershell
uv run pytest -q tests/test_recurrence_experiment.py tests/test_recurrence_cli.py
```

- [ ] **Step 13.6: Implement dependency injection**

Define a frozen `RecurrenceDependencies` dataclass containing metadata resolver, snapshot loader, tokenizer/model loader, dataset loader, clock/synchronizer, disk/VRAM probes, and stage functions. Defaults call production libraries; tests supply fakes and make no network/GPU calls.

- [ ] **Step 13.7: Implement staging-to-signed-run finalization**

Before network access, `source` derives a deterministic 12-hex staging ID from repo ID plus protocol/config/code hashes, creates `artifacts/recurrence-v0/staging/<staging-id>/`, and prints `STAGING_ID=<value>` and its absolute path. A compatible replay resumes that staging root. `data` freezes exact-answer and retention data, calculates source/data/config/code signatures, derives the final 12-hex run ID, and atomically moves/finalizes into `artifacts/recurrence-v0/<run-id>`. Reject an existing incompatible root; accept an identical finalized root read-only.

- [ ] **Step 13.8: Implement stage functions and stop propagation**

Map stages to G0-G6 responsibilities exactly. Before starting, verify predecessors, signatures, available budget, artifact size, and sealed-data authorization. After work, reconcile counts and persist terminal stage status. Never call the next stage internally.

- [ ] **Step 13.9: Implement CLI path and exit behavior**

Print the absolute resolved config and artifact root before substantive work. For source/data, also print the deterministic staging ID/path; after data finalization and for later stages, print exactly one `RUN_ID=<12-hex>` line and the final run path. Map `passed` to 0, scientific/retention stop to 2, budget stop to 3, and environment/implementation failure to 1. Catch only typed stage exceptions; unexpected exceptions still attempt a failure record and then propagate as exit 1.

- [ ] **Step 13.10: Run orchestration tests**

Run:

```powershell
uv run pytest -q tests/test_recurrence_experiment.py tests/test_recurrence_cli.py
```

- [ ] **Step 13.11: Do not commit yet**

Task 14 adds reporting and owns the milestone commit.

### Task 14: Implement evidence-derived reporting and commit the complete implementation

**Files:**

- Create: `src/ppmoe/recurrence/report.py`
- Create: `tests/test_recurrence_report.py`
- Modify: `src/ppmoe/recurrence/experiment.py`

**Interfaces:**

- Produces: `build_report(run_root: Path) -> dict[str, object]`
- Produces: immutable `report.json`, `report.md`, and a report-source manifest

- [ ] **Step 14.1: Write complete-negative and positive report fixtures**

Create compact but internally complete artifact trees for: G3 uninformative stop, G4 accuracy no-go, G4 retention-uncertain stop, G4 timing-inconclusive stop, G2 budget stop, repeated infrastructure failure, G5 confirmation failure, and G5 success.

- [ ] **Step 14.2: Write anti-fabrication tests**

Assert missing evaluation is rendered as `not_evaluated`, never 0 accuracy. Reject aggregates without raw score/chunk hashes, confidence intervals without bootstrap metadata, success missing any arm/seed/control, mismatched counts, unreported failures, or claims broader than the decision state.

- [ ] **Step 14.3: Write required-content tests**

Every report must include source revision/files/hashes, protocol/code/data signatures, branch and mixed revision notes, exact stage statuses, examples scored/failed, family scores/informativeness, transitions, accuracy CIs, PPL points and chunk-bootstrap intervals, NLL/token counts, training steps/examples/times, A3 validity, parameters, layer evaluations, memory/disk/cost, failure history, protocol deviations, supported conclusion, and the next allowed command or terminal state.

- [ ] **Step 14.4: Run report tests and observe failure**

Run:

```powershell
uv run pytest -q tests/test_recurrence_report.py
```

- [ ] **Step 14.5: Implement strict artifact loading**

Read and hash every input file before aggregation. Validate schemas, signatures, ID sets, counts, finite numbers, and stage authorization. Build `report.json` entirely from validated inputs. Render Markdown from that object; do not independently recompute or hand-enter reported numbers.

- [ ] **Step 14.6: Implement narrow conclusion templates**

Use status-specific language:

```text
passed: recurrent LoRA passed this fixed two-seed, three-family protocol.
stopped_scientific: recurrent LoRA did not pass the registered quality gate.
retention_uncertain: task quality may have passed, but the registered retention bound was not established.
stopped_budget: no scientific conclusion; the registered study did not fit the resource cap.
failed_environment: no scientific conclusion; execution evidence is incomplete because of an environment failure.
```

Append explicit limitations: one checkpoint, one loop window, synthetic exact-answer tasks, two fixed seeds, uncached inference, and no sparse-efficiency/general-intelligence claim.

- [ ] **Step 14.7: Run all new tests**

Run:

```powershell
uv run pytest -q tests/test_recurrence_config.py tests/test_recurrence_provenance.py tests/test_recurrence_source.py tests/test_recurrence_data.py tests/test_recurrence_retention.py tests/test_recurrence_lora.py tests/test_recurrence_model.py tests/test_recurrence_loss.py tests/test_recurrence_train.py tests/test_recurrence_evaluate.py tests/test_recurrence_gates.py tests/test_recurrence_experiment.py tests/test_recurrence_cli.py tests/test_recurrence_report.py
```

- [ ] **Step 14.8: Run repository-wide verification**

Run:

```powershell
uv run pytest -q
uv lock --check
git diff --check
```

Expected: all tests pass, including historical suites. Do not change historical tests merely to conceal a regression.

- [ ] **Step 14.9: Audit source boundaries and artifacts**

Run `git status --short --ignored`. Confirm no model weights, generated data, checkpoints, scores, artifacts, `.venv`, or caches will be committed. Confirm no site-packages file changed. Confirm historical artifact hashes still match Task 0.

- [ ] **Step 14.10: Review and commit the evaluation/orchestration milestone**

Run:

```powershell
git add src/ppmoe/recurrence/evaluate.py src/ppmoe/recurrence/gates.py src/ppmoe/recurrence/experiment.py src/ppmoe/recurrence/report.py scripts/run_recurrence_v0.py tests/test_recurrence_evaluate.py tests/test_recurrence_gates.py tests/test_recurrence_experiment.py tests/test_recurrence_cli.py tests/test_recurrence_report.py
git diff --cached --check
git commit -m "feat: add recurrence evaluation and experiment gates"
```

- [ ] **Step 14.11: Verify a clean implementation boundary**

Run:

```powershell
git status --short
git log -5 --oneline --decorate
uv run pytest -q
```

Only ignored/generated files may remain. If tracked changes remain, identify their owner before production execution.

---

## Phase E — production execution through registered gates

The following tasks run the committed implementation. They do not authorize code or protocol changes after scores are seen. A bug fix that changes runtime code creates a new code signature; preserve existing attempts and record mixed provenance. A change to scientific constants requires a new protocol version and is outside this run.

Use one literal run ID throughout. Never select a run by modification time or a `latest` directory.

### Task 15: Execute G0 source and data finalization

**Files produced under ignored artifact storage:**

- Staging `source.json`
- Pinned model/tokenizer/config/license snapshot references
- `data/train.jsonl`, `data/dev.jsonl`, sealed `data/test.jsonl`
- Qwen3-tokenized retention chunks and manifest
- Final signed run root `artifacts/recurrence-v0/<run-id>/`
- `stages/source.json`, `stages/data.json`, and append-only `events.jsonl`

**Prerequisites:** Task 14 clean implementation boundary; network access and sufficient disk; no other process writing to the recurrence artifact root.

- [ ] **Step 15.1: Reconfirm branch, commits, GPU, and free disk**

Run:

```powershell
git branch --show-current
git status --short
git log -4 --oneline --decorate
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
Get-PSDrive -Name C | Select-Object Name,Free,Used
```

Expected: branch `exp`, intended milestone commits present, no unexplained tracked changes, approximately 16 GiB GPU visible, and enough disk for the 8 GiB download plus 20 GiB artifact caps and temporary files.

- [ ] **Step 15.2: Run source resolution**

Run:

```powershell
uv run python scripts/run_recurrence_v0.py --stage source --config configs/recurrence-v0.toml --artifact-root artifacts/recurrence-v0
$sourceExit = $LASTEXITCODE
```

Interpret `$sourceExit` using the registered exit-code table. If authentication or license acceptance requires a human action, pause and report the exact URL/error. For transient network or disk errors, perform safe diagnostics and the one permitted fresh-process retry before declaring `failed_environment`.

- [ ] **Step 15.3: Validate the source record before data generation**

Confirm the resolved revision is 40 hex characters, every downloaded file is at that revision, architecture assertions passed, download bytes are within 8 GiB, license/source hashes exist, dependency versions match `uv.lock`, and the source stage status is `passed`.

- [ ] **Step 15.4: Run deterministic data finalization and capture its run ID**

Capture the emitted 12-character run ID directly from the command's explicit output rather than listing directories:

```powershell
$dataOutput = & uv run python scripts/run_recurrence_v0.py --stage data --config configs/recurrence-v0.toml --artifact-root artifacts/recurrence-v0 2>&1
$dataExit = $LASTEXITCODE
$dataOutput | ForEach-Object { Write-Output $_ }
$runIdMatches = @($dataOutput | Select-String -Pattern '^RUN_ID=([0-9a-f]{12})$')
if ($runIdMatches.Count -ne 1) { throw 'data stage did not emit exactly one RUN_ID line' }
$recurrenceRunId = $runIdMatches[0].Matches[0].Groups[1].Value
```

Immediately validate it:

```powershell
if ($recurrenceRunId -notmatch '^[0-9a-f]{12}$') { throw 'invalid recurrence run id' }
```

- [ ] **Step 15.5: Reconcile exact-answer datasets without opening test content**

Read `data/manifest.json`, file sizes, hashes, and counts. Do not display or parse `data/test.jsonl`. Confirm train `2667/2667/2666`, dev `200/200/200`, test manifest counts `800/800/800`, global ID uniqueness, registered structures/seeds, prompt/answer length bounds, and tokenizer/model revision hashes.

- [ ] **Step 15.6: Reconcile retention data**

Confirm exactly 20,000 valid source tokens, positive scored tokens per chunk, maximum chunk length 512, document ID/text provenance, tokenizer SHA, chunk index/hash list, and no use of old Qwen3.5 token IDs.

- [ ] **Step 15.7: Verify G0 and artifact bounds**

Confirm `stages/source.json` and `stages/data.json` are immutable and passing, signatures reproduce `$recurrenceRunId`, old historical hashes remain unchanged, download/artifact sizes are within caps, and `events.jsonl` indices are strictly increasing.

- [ ] **Step 15.8: Branch on G0 outcome**

- Passed: proceed automatically to Task 16.
- Human-only access/license blocker: pause with exact required action and preserved staging path.
- Environment failure after permitted retry: run Task 22 report and then Task 23 closeout; do not select another model or dataset.
- Budget stop: run Task 22 report and then Task 23 closeout.

### Task 16: Execute G1 real-model correctness smoke

**Artifacts:** `stages/smoke.json`, disposable adapter hashes/results, real-model identity/causality/gradient measurements, failure history.

- [ ] **Step 16.1: Verify the exact run and predecessor**

Run:

```powershell
Test-Path "artifacts/recurrence-v0/$recurrenceRunId/stages/data.json"
Get-Content -Raw -LiteralPath "artifacts/recurrence-v0/$recurrenceRunId/stages/data.json"
```

Confirm it is passing and its signatures match source/data/config/code.

- [ ] **Step 16.2: Run smoke in a fresh process**

```powershell
uv run python scripts/run_recurrence_v0.py --stage smoke --config configs/recurrence-v0.toml --artifact-root artifacts/recurrence-v0 --run-id $recurrenceRunId
$smokeExit = $LASTEXITCODE
```

- [ ] **Step 16.3: Validate stock-wrapper identity**

For all eight fixed development prompts, require finite logits, BF16 maximum absolute difference at most `1e-3`, and cosine at least `0.999999` between stock execution and wrapper single mode. Record exact per-prompt and maximum differences; do not round before gating.

- [ ] **Step 16.4: Validate model semantics and gradients**

Confirm adapter-off A1 matches F0, adapter-off A2 matches F1, parameter/object identities prove sharing, layer-call counts are `28/32`, cache is absent, output causality passes, short adapter gradients are finite/nonzero, suffix input gradients exist, and disposable pilot/smoke adapters are not production checkpoints.

- [ ] **Step 16.5: Validate memory and cost accounting**

Confirm peak allocated/reserved memory and synchronized CUDA time, including any failed attempt, were appended to the ledger. Confirm artifacts remain within 20 GiB.

- [ ] **Step 16.6: Branch on G1 outcome**

- Passed: proceed automatically to Task 17.
- First CUDA fault: preserve failure, exit, start one fresh-process retry with the same command.
- Repeated fault or correctness failure: mark `failed_environment`, run Task 22, then close out. Do not relax tolerances or edit the wrapper in place after production scoring begins; a genuine implementation defect requires a new code signature and explicit recovery revision.

### Task 17: Execute G2 timing and resource pilot

**Artifacts:** `pilot.json`, `stages/pilot.json`, disposable A1/A2 timing attempts, 36 frozen generation records, four retention-chunk measurements, updated cost ledger.

- [ ] **Step 17.1: Start the pilot with the exact run ID**

```powershell
uv run python scripts/run_recurrence_v0.py --stage pilot --config configs/recurrence-v0.toml --artifact-root artifacts/recurrence-v0 --run-id $recurrenceRunId
$pilotExit = $LASTEXITCODE
```

- [ ] **Step 17.2: Verify training timing samples**

Confirm A1 and A2 each ran five excluded warmups plus 20 measured complete optimizer steps on identical preloaded batches, measurement order alternated, CUDA was synchronized, and each sample includes forward/backward/clip/step but excludes data loading.

- [ ] **Step 17.3: Verify A3 step derivation**

Recompute independently:

```text
n3 = max(1000, ceil(1000 * median(A2 measured seconds) / median(A1 measured seconds)))
```

Confirm `pilot.json` freezes the same integer and raw samples. No later stage may change `n3`.

- [ ] **Step 17.4: Verify evaluation timing bounds**

Confirm four examples per family per F0/F1/F2 condition were generated, totaling 36; four retention chunks were measured; the longest observed prompt with a forced 32-token completion was included in pessimistic projection; and all attempts were charged to the ledger.

- [ ] **Step 17.5: Verify resource inequalities**

Require:

```text
peak_reserved_bytes <= total_vram_bytes - 2 GiB
already_spent_seconds + 1.5 * projected_remaining_seconds <= 86,400
download_bytes <= 8 GiB
artifact_bytes <= 20 GiB
```

- [ ] **Step 17.6: Destroy disposable state and confirm production initialization**

Confirm no pilot adapter/checkpoint can be mistaken for production. Verify seed-47 and seed-48 initialization files remain unchanged and their hashes match source manifests.

- [ ] **Step 17.7: Branch on G2 outcome**

- Passed: proceed automatically to Task 18.
- Budget/resource stop: persist `stopped_budget`, run Task 22, then close out. Do not shrink data, reduce output length, quantize, switch the model, or increase the cap.
- Environment failure: use the single fresh-process retry rule; after repetition, run Task 22 and close out.

### Task 18: Execute G3 frozen recurrence diagnostics and benchmark-informativeness gate

**Artifacts:** complete F0/F1/F2 development scores, retention scores, family-informativeness classification, transitions, `stages/frozen.json`.

- [ ] **Step 18.1: Run the frozen stage**

```powershell
uv run python scripts/run_recurrence_v0.py --stage frozen --config configs/recurrence-v0.toml --artifact-root artifacts/recurrence-v0 --run-id $recurrenceRunId
$frozenExit = $LASTEXITCODE
```

- [ ] **Step 18.2: Reconcile all frozen score IDs**

Require exactly 600 development records for each F0, F1, and F2, with 200 per family, no duplicates/missing/unexpected IDs, complete reason codes, and valid raw-file hashes. Require all registered retention chunks once per condition.

- [ ] **Step 18.3: Apply the global informativeness gate**

F0 macro exact accuracy must be strictly greater than 5% and strictly less than 95%; F0 global invalid-format rate must be at most 10%.

- [ ] **Step 18.4: Apply the per-family informativeness gate**

For expressions, lists, and graphs separately, classify informative only when F0 exact accuracy is strictly between 5% and 95% and invalid-format rate is at most 10%. Require at least two informative families. Preserve and report all families even when one is non-informative.

- [ ] **Step 18.5: Record diagnostic changes without using them for model selection**

For F1 and F2 versus F0, record macro/family deltas, invalid-output deltas, wrong-to-right and right-to-wrong transitions, PPL points and intervals, latency, and memory. Do not choose the better frozen method, move the loop block, or use frozen quality as a reason to skip A2 when the benchmark remains informative.

- [ ] **Step 18.6: Branch on G3 outcome**

- Informative: proceed automatically to Task 19 even if F1/F2 quality is poor.
- Uninformative: persist `stopped_scientific` with global/family values, run Task 22, then close out. Do not redesign prompts or replace families in this run.
- Environment failure: apply the one fresh-process retry, then report/close on repetition.

### Task 19: Execute G4 discovery training and development selection for seed 47

**Artifacts:** A1/A2/A3 seed-47 checkpoints/manifests, dev and retention scores, paired transitions, PPL bootstrap, timing validity, immutable `selection.json`, `stages/discover.json`.

- [ ] **Step 19.1: Verify budget before each discovery arm**

Before A1, A2, and A3 separately, recompute cumulative spent time plus pessimistic remaining cost. If the next arm cannot fit, stop before starting it and preserve prior arms as partial evidence.

- [ ] **Step 19.2: Start or resume discovery**

```powershell
uv run python scripts/run_recurrence_v0.py --stage discover --config configs/recurrence-v0.toml --artifact-root artifacts/recurrence-v0 --run-id $recurrenceRunId
$discoverExit = $LASTEXITCODE
```

The stage internally executes A1, A2, then A3. Reissuing the same command after interruption must resume the first incomplete compatible arm, not repeat a completed arm.

- [ ] **Step 19.3: Monitor durable training progress**

For each arm, require `PROGRESS`/event records every 10 updates and atomic checkpoints every 50. Confirm exactly one writer owns each checkpoint tree. After wrapper/process timeout, inspect live process, GPU, event, and checkpoint state before deciding whether to resume; never launch a concurrent writer.

- [ ] **Step 19.4: Reconcile A1 seed-47 training**

Require 1,000 completed updates, 8,000 unique examples presented once in the seeded order, correct initialization hash, only adapter parameters trained, finite loss/gradients, exact checkpoint sequence, and cumulative time/memory recorded.

- [ ] **Step 19.5: Reconcile A2 seed-47 training**

Apply the same checks, plus shared adapter object IDs and gradients from both passes. Require exactly 1,000 updates and 8,000 unique examples.

- [ ] **Step 19.6: Reconcile A3 seed-47 training**

Require exactly frozen `n3` updates. Confirm the same seed-47 permutation cycles without reshuffling after 8,000 presentations and no new examples appear. Record total presentations and unique examples separately.

- [ ] **Step 19.7: Evaluate final checkpoints on development and retention**

Use only final checkpoints—no best-checkpoint selection. Require 600 dev records and all retention chunks for A1/A2/A3. Reuse the deterministic F0 vectors and chunks from G3 after signature/hash validation.

- [ ] **Step 19.8: Validate the training-time control**

Compute realized A3/A2 production training-time ratio. It must lie in inclusive `[0.90,1.10]`; otherwise the budget-controlled comparison is inconclusive and no corrective training is permitted.

- [ ] **Step 19.9: Validate retention with paired chunk bootstrap**

For A2 versus F0, require point PPL ratio at most 1.02 and the 97.5th percentile of 10,000 paired chunk-bootstrap ratios, seed 9157, at most 1.02. Persist the full bootstrap summary and implementation/version metadata.

- [ ] **Step 19.10: Apply every discovery quality predicate**

Require A2 macro improvement at least 2 percentage points over F0, A1, and A3; no A2 family decrease worse than 2 points versus F0 or A1; valid retention; and valid A3 timing. Record each predicate separately with raw value, comparator, threshold, and boolean.

- [ ] **Step 19.11: Freeze selection or stop**

- All predicates pass: write immutable `selection.json` authorizing only A2 with all fixed conditions, then proceed to Task 20.
- Accuracy/family predicate fails: `stopped_scientific`, run Task 22, close out.
- Retention point or CI fails: `retention_uncertain`, exit 2, run Task 22, close out.
- Timing fails: `timing_inconclusive`, exit 2, run Task 22, close out.
- Budget/environment stops: persist correct status, then report/close.

Do not open the test JSONL or run seed 48 unless `selection.json` is passing and immutable.

### Task 20: Execute G5 confirmation training for seed 48

**Artifacts:** A1/A2/A3 seed-48 final checkpoints/manifests and pre-test authorization records inside the confirm stage.

- [ ] **Step 20.1: Verify sealed-test authorization without opening test rows**

Validate `selection.json`, its dev score hashes, all source/data/config/code signatures, selected condition A2, frozen `n3`, and remaining budget. Confirm no test score file exists.

- [ ] **Step 20.2: Start or resume confirmation**

```powershell
uv run python scripts/run_recurrence_v0.py --stage confirm --config configs/recurrence-v0.toml --artifact-root artifacts/recurrence-v0 --run-id $recurrenceRunId
$confirmExit = $LASTEXITCODE
```

The confirm stage first trains all seed-48 arms and only then opens sealed tests. If interrupted during training, resuming must not open test data early.

- [ ] **Step 20.3: Reconcile seed-48 A1, A2, and A3 sequentially**

For each arm apply Task 19's update, example, initialization, parameter, checkpoint, cost, and failure-history checks. A1/A2 each use 1,000 updates; A3 uses the same frozen `n3` from seed 47's pilot. Seed 48 uses the same dataset pool and its registered seed-48 permutation, not regenerated examples.

- [ ] **Step 20.4: Require all six final production checkpoints before test access**

Validate final A1/A2/A3 checkpoints for seeds 47 and 48, hashes, complete training manifests, and valid realized A3/A2 ratios. If any is incomplete or timing-invalid, do not open final tests; persist the appropriate terminal state and go to Task 22.

- [ ] **Step 20.5: Allow the same command to advance into sealed evaluation**

After Step 20.4 passes, the confirm stage may continue automatically to Task 21's internal sequence. No user approval is needed because the signed selection manifest is the authorization gate.

### Task 21: Complete sealed tests, confirmation statistics, and G5 decision

**Artifacts:** F0 and six adapted test score sets, seed-48 retention plus reused validated seed-47 retention, accuracy/retention bootstrap summaries, `stages/confirm.json`.

- [ ] **Step 21.1: Score the fixed final-test matrix**

Score exactly 2,400 examples for F0 and for each A1/A2/A3 checkpoint at seeds 47 and 48. F0 is deterministic and stored once; reuse its correctness vector for both seed comparisons. Reconcile 800 IDs per family and all expected arm/seed combinations.

- [ ] **Step 21.2: Score seed-48 retention and validate both seeds**

Score every registered chunk for A1/A2/A3 seed 48; validate seed-47 retention hashes from discovery. Compute A2/F0 point and paired chunk-bootstrap interval for each seed with 10,000 replicates and seed 9157.

- [ ] **Step 21.3: Compute paired task-stratified accuracy intervals**

For A2 versus F0, A1, and A3, average the two seed-specific paired correctness differences per example, then resample IDs within each family using the same sampled indices across arms/seeds. Run exactly 10,000 replicates with seed 9047 and store the 2.5/97.5 percentiles without rounding before decisions.

- [ ] **Step 21.4: Apply every confirmation predicate**

Require:

```text
for each seed: A2-F0 macro > 0
for each seed: A2-A1 macro > 0
for each seed: A2-A3 macro > 0
two-seed mean A2-F0 macro >= 0.02
two-seed mean A2-A1 macro >= 0.02
two-seed mean A2-A3 macro >= 0.02
each two-seed mean family delta versus F0 >= -0.02
each two-seed mean family delta versus A1 >= -0.02
each of three accuracy CI lower endpoints > 0
each seed A2/F0 PPL point <= 1.02
each seed A2/F0 PPL CI high <= 1.02
each seed A3/A2 realized time ratio in [0.90,1.10]
```

- [ ] **Step 21.5: Persist the G5 decision without reinterpretation**

Write each predicate and raw value. If all pass, status is `passed`. If quality fails, status is `stopped_scientific`. If retention uncertainty fails, reason is `retention_uncertain`. If timing fails, reason is `timing_inconclusive`. Do not describe a near miss as success or start additional seeds.

- [ ] **Step 21.6: Verify cost and artifact caps after evaluation**

Reconcile all attempts and synchronized times. If the cap was exceeded because a stage failed to stop prospectively, record a protocol deviation; do not hide it. Confirm disk and download totals.

- [ ] **Step 21.7: Proceed to reporting for every outcome**

Task 22 runs whether G5 passed, failed, or stopped. No additional experiment follows automatically.

### Task 22: Execute G6 report and independently validate it

**Artifacts:** immutable `report.json`, `report.md`, report-source manifest/hashes, `stages/report.json`.

- [ ] **Step 22.1: Run report in a fresh non-training process**

If data finalization emitted a final run ID, run:

```powershell
uv run python scripts/run_recurrence_v0.py --stage report --config configs/recurrence-v0.toml --artifact-root artifacts/recurrence-v0 --run-id $recurrenceRunId
$reportExit = $LASTEXITCODE
```

If source or data stopped before a final run ID existed, omit `--run-id`; the CLI must derive the same deterministic staging ID from the still-current protocol/config/code signatures and report that exact staging root:

```powershell
uv run python scripts/run_recurrence_v0.py --stage report --config configs/recurrence-v0.toml --artifact-root artifacts/recurrence-v0
$reportExit = $LASTEXITCODE
```

If the code signature changed after the early failure, use the original committed implementation to render the staging report; do not scan for or guess a latest staging directory.

Report must run after success or any terminal stop and must not load model weights unless required solely to validate an already persisted static identity field.

- [ ] **Step 22.2: Reconcile report sources**

Verify every source path exists, every recorded SHA matches, raw score/chunk ID counts match summaries, terminal stages are represented, failure history is complete, and missing downstream stages are `not_evaluated`.

- [ ] **Step 22.3: Independently recompute decisive values**

From raw JSONL, independently recompute macro/family accuracy, transitions, PPL points, bootstrap seeds/replicate counts/intervals, realized timing ratios, cumulative CUDA time, peak memory, parameter counts, and layer evaluations. Compare to `report.json` before accepting Markdown.

- [ ] **Step 22.4: Inspect conclusion scope**

The report may conclude only one of the predefined status-specific statements. It must explicitly state that synthetic-task success is not general intelligence, recurrence is not sparse efficiency, uncached latency is not serving performance, and two fixed seeds do not establish population-wide seed robustness.

- [ ] **Step 22.5: Run report and repository verification**

Run:

```powershell
uv run pytest -q tests/test_recurrence_report.py tests/test_recurrence_gates.py tests/test_recurrence_experiment.py
uv run pytest -q
uv lock --check
git diff --check
```

- [ ] **Step 22.6: Record exact completion state**

Capture report exit, report/status hashes, exact scored versus failed counts, CUDA/disk totals, decision, and next allowed action. A successful report build returns 0 even when the scientific outcome is negative; the report content carries the experiment decision.

### Task 23: Update durable project state and create the evidence milestone commit

**Files:**

- Modify: `PROJECT_LOG.md`
- Modify: `todo.md`
- Do not commit: `artifacts/recurrence-v0/**`

- [ ] **Step 23.1: Append an evidence-only project-log entry**

Record date, branch, protocol/code/model/data revisions, commands, stage exit statuses, exact completed/scored/failed counts, attempts and failures, resource use, all decisive gate values, report hashes, supported conclusion, and terminal next action. Distinguish verified facts from unexecuted stages.

- [ ] **Step 23.2: Update todo checkboxes only from artifacts**

Mark an item complete only when its deliverable and verification evidence exist. Leave skipped downstream experimental items unchecked and annotate `not_evaluated` after an earlier stop. Never mark a scientific gate complete merely because the report was generated.

- [ ] **Step 23.3: Reverify historical immutability**

Recompute the four historical hashes from Task 0 and compare case-insensitively. Any mismatch blocks the evidence commit until ownership and cause are resolved.

- [ ] **Step 23.4: Verify final repository state**

Run:

```powershell
git status --short --ignored
git diff -- PROJECT_LOG.md todo.md
git diff --check
uv run pytest -q
uv lock --check
```

- [ ] **Step 23.5: Stage only durable documentation**

```powershell
git add PROJECT_LOG.md todo.md
git diff --cached --stat
git diff --cached --check
```

Inspect the staged diff. It must contain no generated metrics copied without provenance and no ignored artifacts.

- [ ] **Step 23.6: Create the evidence milestone commit**

Choose the exact message from observed outcome:

```powershell
git commit -m "docs: record recurrence v0 result"
```

For an early environment-only stop with no scientific result, use:

```powershell
git commit -m "docs: record recurrence v0 environment stop"
```

- [ ] **Step 23.7: Final handoff**

Report the final commit SHA, run ID, report paths/hashes, exact outcome, checks run, unresolved human-only blockers if any, and whether the repository has remaining unrelated changes. Do not push, merge to `main`, start another experiment, or modify ExpertFlow without a new explicit request.

---

## Automatic-stop and human-pause matrix

| Condition | Persist | Exit | Next action |
|---|---|---:|---|
| Passing stage | `passed` stage record | 0 | Continue to next numbered task automatically. |
| Accuracy/family gate fails | `stopped_scientific` | 2 | Build report, update durable docs, end study. |
| Benchmark has fewer than two informative families | `stopped_scientific` with family table | 2 | Build report, update durable docs, end study. |
| Retention point or CI exceeds 1.02 | `retention_uncertain` reason under scientific stop | 2 | Build report, update durable docs, end study. |
| A3 realized timing outside `[0.90,1.10]` | `timing_inconclusive` reason | 2 | Build report, update durable docs, end study. |
| Projected or actual registered resource bound fails | `stopped_budget` | 3 | Build report, update durable docs, end study. |
| First CUDA/process fault | failure attempt and charged cost | 1 for attempt | Exit process; one fresh-process compatible retry. |
| Repeated CUDA/process fault | `failed_environment` plus both attempts | 1 | Build report, update durable docs, end study. |
| Missing credentials/license acceptance | staging failure record | 1 | Pause for the user with exact required action. |
| Conflicting user-owned tracked edits | no mutation of conflicting files | n/a | Pause for ownership/direction. |
| Need for scientific constant/model/data change | protocol deviation proposal only | n/a | Stop; require a separately approved V1 protocol. |

## Completion definition

The plan is complete only when one of these is true:

1. All implementation milestones pass, G0-G6 complete, the final result is reported, durable documentation is committed, and no required work remains.
2. A registered scientific, retention, timing, or budget stop is fully reported and committed, with downstream work correctly marked `not_evaluated`.
3. A repeated environment failure is fully reported and committed, with exact recovery evidence and no unsupported scientific conclusion.
4. A genuine human-only blocker is presented with its exact required action, preserved state, safe resume command, and no concurrent writer.

Running out of time in an interactive session, receiving a wrapper timeout while a child process continues, or encountering a difficult implementation problem is not completion and is not by itself a human blocker. Use persisted state, inspect live ownership, and continue safely.
