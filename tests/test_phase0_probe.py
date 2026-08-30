import json

from ppmoe.phase0 import run_phase0_probe


def test_phase0_probe_is_deterministic_and_json_safe():
    arguments = {
        "device": "cpu",
        "dtype": "float32",
        "seed": 31,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_experts": 4,
    }

    first = run_phase0_probe(**arguments)
    second = run_phase0_probe(**arguments)

    assert first == second
    assert first["schema_version"] == 1
    assert first["all_expert"]["relative_error"] <= 1e-5
    assert first["oracle"]["direct_gram_subsets_equal"] is True
    assert first["parameter_counts"]["dense"] == first["parameter_counts"]["partitioned"]
    json.dumps(first, allow_nan=False)


def test_phase0_probe_reports_requested_dimensions_and_k_values():
    report = run_phase0_probe(
        device="cpu",
        dtype="float32",
        seed=37,
        hidden_size=8,
        intermediate_size=16,
        num_experts=4,
    )

    assert report["dimensions"] == {
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_experts": 4,
        "expert_width": 4,
    }
    assert report["oracle"]["k_values"] == [1, 2, 3, 4]

