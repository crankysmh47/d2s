import torch

from tests.helpers import TinySwiGLUTransformer, partitioned_copy


def test_two_layer_all_expert_model_matches_dense_fp32():
    torch.manual_seed(19)
    dense = TinySwiGLUTransformer()
    converted = partitioned_copy(dense, num_experts=4)
    token_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
        ]
    )

    expected = dense(token_ids)
    actual = converted(token_ids)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
