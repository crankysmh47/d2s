"""Explicit, cacheless recurrence over one shared contiguous Qwen3 block."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, overload

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3ForCausalLM
from transformers.masking_utils import create_causal_mask


RecurrenceMode = Literal["single", "repeat", "damped"]
_SUPPORTED_MODES = frozenset(("single", "repeat", "damped"))


def apply_block(block: Callable[[Tensor], Tensor], hidden: Tensor, mode: str) -> Tensor:
    """Apply one shared block according to the frozen recurrence algebra."""
    if mode == "single":
        return block(hidden)
    if mode == "repeat":
        return block(block(hidden))
    if mode == "damped":
        intermediate = hidden + 0.5 * (block(hidden) - hidden)
        return intermediate + 0.5 * (block(intermediate) - intermediate)
    raise ValueError(f"unsupported mode: {mode}")


def chunked_causal_loss(
    hidden: Tensor,
    labels: Tensor,
    lm_head: nn.Module,
    chunk_size: int = 16,
) -> Tensor:
    """Return FP32 answer-only CE without materializing full-sequence logits.

    Each supervised target at position ``j`` is predicted by the normalized
    hidden state at ``j - 1``.  V0 has a batch size of one, so this is also
    the registered per-example mean rather than a token-weighted batch mean.
    """
    if not isinstance(hidden, Tensor) or hidden.ndim != 3:
        raise ValueError("hidden must have shape [1, T, D]")
    if not isinstance(labels, Tensor) or labels.ndim != 2:
        raise ValueError("labels must have shape [1, T]")
    if labels.dtype != torch.long:
        raise ValueError("labels must use torch.long dtype")
    if hidden.shape[:2] != labels.shape:
        raise ValueError("hidden and labels shape mismatch")
    if hidden.shape[0] != 1:
        raise ValueError("batch size must be exactly one")
    if hidden.shape[1] < 2:
        raise ValueError("loss requires at least two positions")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer")
    if not 1 <= chunk_size <= 16:
        raise ValueError("chunk_size must be between 1 and 16")
    if not isinstance(lm_head, nn.Module):
        raise TypeError("lm_head must be an nn.Module")
    if bool(labels[:, 0].ne(-100).any()):
        raise ValueError("a supervised label at index 0 has no causal predecessor")

    supervised = labels[:, 1:].ne(-100)
    positions = supervised.nonzero(as_tuple=False)
    if positions.numel() == 0:
        raise ValueError("at least one supervised answer/EOS label is required")

    predictors = hidden[:, :-1, :]
    selected_hidden = predictors[positions[:, 0], positions[:, 1]]
    targets = labels[:, 1:][positions[:, 0], positions[:, 1]]
    total_nll = torch.zeros((), device=hidden.device, dtype=torch.float32)
    for start in range(0, selected_hidden.shape[0], chunk_size):
        stop = min(start + chunk_size, selected_hidden.shape[0])
        projected = lm_head(selected_hidden[start:stop])
        if not isinstance(projected, Tensor) or projected.ndim != 2:
            raise ValueError("lm_head must return logits with shape [positions, vocabulary]")
        if projected.shape[0] != stop - start:
            raise ValueError("lm_head changed the supervised-position dimension")
        chunk_targets = targets[start:stop]
        if bool(chunk_targets.lt(0).any()) or bool(chunk_targets.ge(projected.shape[1]).any()):
            raise ValueError("supervised label is outside the lm_head vocabulary")
        log_probabilities = F.log_softmax(projected.float(), dim=-1)
        nll = -log_probabilities.gather(1, chunk_targets.unsqueeze(1)).squeeze(1)
        if not bool(torch.isfinite(nll).all()):
            raise ValueError("nonfinite answer loss")
        total_nll = total_nll + nll.sum()
    loss = total_nll / selected_hidden.shape[0]
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite answer loss")
    return loss


class RecurrentQwen3(nn.Module):
    """Run Qwen3 with an exact shared block recurrence and no KV cache.

    The supplied model is kept as this module's sole child.  The selected
    decoder layers are called more than once, never copied, and receive the
    exact same mask, position IDs, and rotary embeddings on every pass.
    """

    def __init__(
        self,
        stock_model: Qwen3ForCausalLM,
        layer_start: int = 12,
        layer_stop: int = 16,
    ) -> None:
        super().__init__()
        if not isinstance(stock_model, Qwen3ForCausalLM):
            raise TypeError("stock_model must be a Qwen3ForCausalLM")
        backbone = stock_model.model
        layers = getattr(backbone, "layers", None)
        if not isinstance(layers, nn.ModuleList):
            raise ValueError("stock model must expose model.layers as a ModuleList")
        if isinstance(layer_start, bool) or isinstance(layer_stop, bool) or not (
            isinstance(layer_start, int) and isinstance(layer_stop, int)
        ):
            raise ValueError("layer range bounds must be integers")
        if not 0 <= layer_start < layer_stop <= len(layers):
            raise ValueError("layer range is outside stock model.layers")
        if _has_sliding_attention(stock_model):
            raise ValueError("sliding attention is unsupported by recurrent Qwen3")
        if not isinstance(getattr(backbone, "embed_tokens", None), nn.Module):
            raise ValueError("stock model is missing model.embed_tokens")
        if not isinstance(getattr(backbone, "rotary_emb", None), nn.Module):
            raise ValueError("stock model is missing model.rotary_emb")
        if not isinstance(getattr(backbone, "norm", None), nn.Module):
            raise ValueError("stock model is missing model.norm")
        if not isinstance(getattr(stock_model, "lm_head", None), nn.Module):
            raise ValueError("stock model is missing lm_head")

        # This is intentionally the only module registration in this adapter.
        self.stock_model = stock_model
        self.layer_start = layer_start
        self.layer_stop = layer_stop

    @overload
    def hidden(
        self,
        input_ids: Tensor,
        mode: str,
        *,
        return_pass_hidden_states: Literal[False] = False,
        attention_mask: Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool | None = None,
    ) -> Tensor: ...

    @overload
    def hidden(
        self,
        input_ids: Tensor,
        mode: str,
        *,
        return_pass_hidden_states: Literal[True],
        attention_mask: Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool | None = None,
    ) -> tuple[Tensor, tuple[Tensor, ...]]: ...

    def hidden(
        self,
        input_ids: Tensor,
        mode: str,
        *,
        return_pass_hidden_states: bool = False,
        attention_mask: Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool | None = None,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
        """Return final-normalized hidden states, optionally with block-pass states."""
        self._validate_inputs(input_ids, mode, attention_mask, past_key_values, use_cache)
        backbone = self.stock_model.model
        inputs_embeds = backbone.embed_tokens(input_ids)
        sequence_length = input_ids.shape[1]
        position_ids = torch.arange(
            sequence_length, device=input_ids.device, dtype=torch.long
        ).unsqueeze(0)
        causal_mask = create_causal_mask(
            config=self.stock_model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
        )
        position_embeddings = backbone.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        if self.training:
            # Prefix weights are frozen by adapter installation.  Re-enable an
            # input gradient at the recurrence boundary so checkpointed shared
            # passes and frozen suffix layers retain their adapter-gradient path.
            with torch.no_grad():
                for layer_index in range(self.layer_start):
                    hidden_states = self._run_layer(
                        backbone.layers[layer_index],
                        hidden_states,
                        causal_mask,
                        position_ids,
                        position_embeddings,
                    )
            if torch.is_grad_enabled():
                hidden_states = hidden_states.detach().requires_grad_(True)
            else:
                hidden_states = hidden_states.detach()
        else:
            for layer_index in range(self.layer_start):
                hidden_states = self._run_layer(
                    backbone.layers[layer_index],
                    hidden_states,
                    causal_mask,
                    position_ids,
                    position_embeddings,
                )

        pass_hidden_states: list[Tensor] = []

        def block(current: Tensor) -> Tensor:
            def run_shared_block(block_input: Tensor) -> Tensor:
                block_output = block_input
                for layer_index in range(self.layer_start, self.layer_stop):
                    block_output = self._run_layer(
                        backbone.layers[layer_index],
                        block_output,
                        causal_mask,
                        position_ids,
                        position_embeddings,
                    )
                return block_output

            output = self._checkpoint_if_training(run_shared_block, current)
            pass_hidden_states.append(output)
            return output

        hidden_states = apply_block(block, hidden_states, mode)
        for layer_index in range(self.layer_stop, len(backbone.layers)):
            layer = backbone.layers[layer_index]

            def run_suffix_layer(suffix_input: Tensor, *, layer: nn.Module = layer) -> Tensor:
                return self._run_layer(
                    layer,
                    suffix_input,
                    causal_mask,
                    position_ids,
                    position_embeddings,
                )

            hidden_states = self._checkpoint_if_training(run_suffix_layer, hidden_states)
        normalized = backbone.norm(hidden_states)
        if not isinstance(normalized, Tensor):
            raise TypeError("Qwen3 final norm must return a Tensor")
        if return_pass_hidden_states:
            return normalized, tuple(pass_hidden_states)
        return normalized

    def forward(
        self,
        input_ids: Tensor,
        mode: str,
        *,
        attention_mask: Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool | None = None,
    ) -> Tensor:
        """Return full logits for a single unpadded sequence."""
        hidden_states = self.hidden(
            input_ids,
            mode,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if not isinstance(hidden_states, Tensor):  # Satisfy the overload contract.
            raise AssertionError("hidden() unexpectedly returned pass states")
        logits = self.stock_model.lm_head(hidden_states)
        if not isinstance(logits, Tensor):
            raise TypeError("Qwen3 lm_head must return a Tensor")
        return logits

    def answer_loss(self, input_ids: Tensor, labels: Tensor, mode: str) -> Tensor:
        """Compute shifted answer/EOS loss from one hidden-state execution."""
        self._validate_answer_labels(input_ids, labels)
        hidden_states = self.hidden(input_ids, mode)
        if not isinstance(hidden_states, Tensor):
            raise AssertionError("hidden() unexpectedly returned pass states")
        return chunked_causal_loss(hidden_states, labels, self.stock_model.lm_head)

    def _validate_answer_labels(self, input_ids: Tensor, labels: Tensor) -> None:
        if not isinstance(labels, Tensor) or labels.ndim != 2:
            raise ValueError("labels must have shape [1, T]")
        if labels.dtype != torch.long:
            raise ValueError("labels must use torch.long dtype")
        if labels.shape != input_ids.shape:
            raise ValueError("input_ids and labels shape mismatch")
        eos_token_id = getattr(self.stock_model.config, "eos_token_id", None)
        if isinstance(eos_token_id, int) and not isinstance(eos_token_id, bool):
            eos_count = int(labels.eq(eos_token_id).sum())
            if eos_count > 1:
                raise ValueError("multiple EOS labels are unsupported")

    def _checkpoint_if_training(
        self, operation: Callable[[Tensor], Tensor], hidden_states: Tensor
    ) -> Tensor:
        if self.training and torch.is_grad_enabled():
            return checkpoint(operation, hidden_states, use_reentrant=False)
        return operation(hidden_states)

    def _validate_inputs(
        self,
        input_ids: Tensor,
        mode: str,
        attention_mask: Tensor | None,
        past_key_values: object | None,
        use_cache: bool | None,
    ) -> None:
        if mode not in _SUPPORTED_MODES:
            raise ValueError(f"unsupported mode: {mode}")
        if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [1, T]")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must use torch.long dtype")
        batch_size, sequence_length = input_ids.shape
        if batch_size != 1:
            raise ValueError("batch size must be exactly one; padded batches are unsupported")
        if not 1 <= sequence_length <= 512:
            raise ValueError("input length must satisfy 1 <= T <= 512")
        if attention_mask is not None:
            raise ValueError("attention_mask is unsupported; padding is forbidden")
        if past_key_values is not None:
            raise ValueError("past_key_values is unsupported")
        if use_cache not in (None, False):
            raise ValueError("use_cache must be False")
        pad_token_id = getattr(self.stock_model.config, "pad_token_id", None)
        eos_token_id = getattr(self.stock_model.config, "eos_token_id", None)
        if (
            pad_token_id is not None
            and pad_token_id != eos_token_id
            and bool(input_ids.eq(pad_token_id).any())
        ):
            raise ValueError("padding token IDs are unsupported")

    @staticmethod
    def _run_layer(
        layer: nn.Module,
        hidden_states: Tensor,
        causal_mask: Tensor | object | None,
        position_ids: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> Tensor:
        output = layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
        if not isinstance(output, Tensor):
            raise TypeError("Qwen3 decoder layers must return a Tensor")
        return output


def _has_sliding_attention(stock_model: Qwen3ForCausalLM) -> bool:
    """Reject every configuration that could select a non-full causal mask."""
    backbone = stock_model.model
    config = stock_model.config
    return bool(
        getattr(backbone, "has_sliding_layers", False)
        or getattr(config, "use_sliding_window", False)
        or getattr(config, "sliding_window", None) is not None
    )
