from copy import deepcopy

import torch

from ppmoe.evaluation.perplexity import EvaluationBatch, evaluate_m0
from ppmoe.moefication.qwen35 import convert_qwen35_text_ffns
from tests.helpers import tiny_qwen35


def test_miniature_real_qwen_dense_and_all_expert_pass_m0(tmp_path) -> None:
    torch.manual_seed(47)
    teacher = tiny_qwen35()
    student = deepcopy(teacher)
    convert_qwen35_text_ffns(student, num_experts=8, seed_base=47)
    batch = EvaluationBatch(
        index=0,
        input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
        attention_mask=torch.ones(1, 5, dtype=torch.long),
    )

    result = evaluate_m0(teacher, student, (batch,), tmp_path)

    assert result.passed is True
    assert 0.999 <= result.ppl_ratio <= 1.001
    assert result.logits_cosine >= 0.99999
