"""CPU-only regression tests for recurrence runtime adapter wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from ppmoe.recurrence import runtime_engine, train
from ppmoe.recurrence.lora import adapter_state_dict, install_adapters
from ppmoe.recurrence.model import RecurrentQwen3


def _tiny_stock_model() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        sliding_window=None,
        _attn_implementation="eager",
        use_cache=False,
        pad_token_id=0,
    )
    return Qwen3ForCausalLM(config).eval()


def _context(tmp_path: Path, stage: str = "pilot") -> SimpleNamespace:
    return SimpleNamespace(
        stage=stage,
        run_root=tmp_path,
        config=SimpleNamespace(loop=SimpleNamespace(layer_start=1, layer_stop=3)),
    )


def test_initialization_installs_adapters_on_stock_model_under_wrapper(
    tmp_path: Path, monkeypatch
) -> None:
    stock = _tiny_stock_model()
    monkeypatch.setattr(runtime_engine, "_load_stock", lambda context: (stock, None))

    initialization, digest = runtime_engine._ensure_initialization(_context(tmp_path), 47)

    snapshot = torch.load(tmp_path / "initializations" / "47.pt", map_location="cpu", weights_only=False)
    assert initialization == snapshot["initialization_hash"]
    assert digest
    assert len(snapshot["adapter_state"]) == 24
    assert len(adapter_state_dict(stock)) == 24


def test_adapted_path_installs_then_loads_initialization_on_stock_model(
    tmp_path: Path, monkeypatch
) -> None:
    init_stock = _tiny_stock_model()
    install_adapters(init_stock, 1, 3, seed=47)
    init_wrapper = RecurrentQwen3(init_stock, 1, 3)
    init_path = tmp_path / "initializations" / "47.pt"
    train.capture_initialization(init_path, init_wrapper, 47)

    target_stock = _tiny_stock_model()
    monkeypatch.setattr(runtime_engine, "_load_stock", lambda context: (target_stock, object()))

    wrapper, _, initialization = runtime_engine._adapted(_context(tmp_path), "a1", 47)

    assert wrapper.stock_model is target_stock
    assert len(adapter_state_dict(wrapper)) == 24
    assert initialization == train.adapter_sha256(init_wrapper)
