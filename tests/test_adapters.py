from lighteval.tasks.requests import Doc

from llmevals.web.adapters import adapt_sample_result


def test_gsm8k_adapter_extracts_final_number_and_success():
    doc = Doc(
        query="Question",
        choices=["Work\n#### 42"],
        gold_index=0,
        task_name="gsm8k|5",
    )
    parsed, expected, success = adapt_sample_result(
        benchmark_name="gsm8k",
        doc=doc,
        raw_response="Reasoning\n#### 42",
        metrics={"extractive_match": 1.0},
    )

    assert parsed == "42"
    assert expected == "42"
    assert success is True


def test_generic_adapter_uses_raw_values():
    doc = Doc(
        query="Question",
        choices=["expected text"],
        gold_index=0,
        task_name="custom|0",
    )
    parsed, expected, success = adapt_sample_result(
        benchmark_name="custom",
        doc=doc,
        raw_response="model output",
        metrics={"accuracy": 0.0},
    )

    assert parsed == "model output"
    assert expected == "expected text"
    assert success is False

