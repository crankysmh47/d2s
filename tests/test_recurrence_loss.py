"""CPU contracts for the answer-only, explicitly shifted causal loss."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ppmoe.recurrence.model import chunked_causal_loss


class _ScriptedHead(nn.Module):
    """Project one-hot hidden states directly to known vocabulary logits."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(vocab_size), requires_grad=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden @ self.weight


def _position_logits(*tokens: int, vocab_size: int = 5) -> Tensor:
    logits = torch.full((1, len(tokens), vocab_size), -4.0)
    for index, token in enumerate(tokens):
        logits[0, index, token] = 4.0
    return logits


def test_shifted_loss_uses_the_preceding_hidden_position() -> None:
    """Catches predicting label j from hidden j rather than hidden j - 1."""
    head = _ScriptedHead(vocab_size=5)
    hidden = _position_logits(3, 1, 4)
    labels = torch.tensor([[-100, 3, -100]])

    reference = chunked_causal_loss(hidden, labels, head, chunk_size=1)
    changed_current_position = hidden.clone()
    changed_current_position[0, 1] = _position_logits(0)[0, 0]

    torch.testing.assert_close(
        chunked_causal_loss(changed_current_position, labels, head, chunk_size=1),
        reference,
        rtol=0,
        atol=0,
    )
    assert reference < 0.01


def test_chunked_loss_matches_full_shifted_cross_entropy() -> None:
    """Catches chunking that changes the supervised-token CE reduction."""
    torch.manual_seed(13)
    hidden = torch.randn(1, 5, 3)
    labels = torch.tensor([[-100, 1, -100, 2, 0]])
    head = nn.Linear(3, 4, bias=False)

    full = F.cross_entropy(
        head(hidden[:, :-1]).float().reshape(-1, 4),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )

    actual = chunked_causal_loss(hidden, labels, head, chunk_size=2)

    torch.testing.assert_close(actual, full, rtol=1e-6, atol=1e-7)


def test_prompt_masking_and_one_eos_label_select_only_answer_tokens() -> None:
    """Catches prompt labels leaking into answer/EOS loss or EOS being omitted."""
    head = _ScriptedHead(vocab_size=5)
    hidden = _position_logits(1, 2, 3, 4)
    labels = torch.tensor([[-100, -100, 2, 3]])

    actual = chunked_causal_loss(hidden, labels, head)
    expected = F.cross_entropy(
        torch.stack((hidden[0, 1], hidden[0, 2])).float(),
        torch.tensor([2, 3]),
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ("hidden", "labels", "error"),
    (
        (torch.zeros(1, 3, 2), torch.tensor([[-100, -100, -100]]), "supervised"),
        (torch.zeros(1, 3, 2), torch.tensor([[1, -100, -100]]), "index 0"),
        (torch.zeros(1, 3, 2), torch.tensor([[-100, 1]]), "shape"),
    ),
)
def test_chunked_loss_rejects_invalid_supervision(
    hidden: Tensor, labels: Tensor, error: str
) -> None:
    """Catches silently training on causal-impossible or malformed label tensors."""
    with pytest.raises(ValueError, match=error):
        chunked_causal_loss(hidden, labels, _ScriptedHead(vocab_size=2))


def test_chunked_loss_rejects_nonfinite_projection() -> None:
    """Catches NaN/Inf projection values becoming a silently recorded loss."""
    head = _ScriptedHead(vocab_size=2)
    hidden = torch.tensor([[[float("nan"), 0.0], [0.0, 1.0]]])
    labels = torch.tensor([[-100, 1]])

    with pytest.raises(ValueError, match="nonfinite"):
        chunked_causal_loss(hidden, labels, head)


@pytest.mark.parametrize(
    ("chunk_size", "error_type", "error"),
    (
        (0, ValueError, "between 1 and 16"),
        (17, ValueError, "between 1 and 16"),
        (True, TypeError, "integer"),
        (1.5, TypeError, "integer"),
    ),
)
def test_chunked_loss_rejects_invalid_chunk_sizes(
    chunk_size: object, error_type: type[Exception], error: str
) -> None:
    """Catches callers bypassing the registered 16-position memory bound."""
    hidden = torch.zeros(1, 2, 2)
    head = _ScriptedHead(vocab_size=2)

    with pytest.raises(error_type, match=error):
        chunked_causal_loss(hidden, torch.tensor([[-100, 1]]), head, chunk_size=chunk_size)  # type: ignore[arg-type]


def test_chunked_loss_rejects_label_outside_vocabulary() -> None:
    """Catches an invalid target gather after a bounded projection."""
    hidden = torch.zeros(1, 2, 2)
    head = _ScriptedHead(vocab_size=2)

    with pytest.raises(ValueError, match="vocabulary"):
        chunked_causal_loss(hidden, torch.tensor([[-100, 2]]), head)
