from __future__ import annotations

from typing import Any

from lighteval.tasks.requests import Doc

from llmevals.parsing import extract_final_number


def _generic_success(metrics: dict[str, Any]) -> bool | None:
    if not metrics:
        return None
    first_value = next(iter(metrics.values()))
    if isinstance(first_value, bool):
        return first_value
    if isinstance(first_value, (int, float)):
        return first_value > 0
    return None


def adapt_sample_result(
    *,
    benchmark_name: str,
    doc: Doc,
    raw_response: str,
    metrics: dict[str, Any],
) -> tuple[str | None, str | None, bool | None]:
    golds = doc.get_golds()
    expected_raw = golds[0] if golds else ""

    if benchmark_name.startswith("gsm8k"):
        parsed_response = extract_final_number(raw_response)
        expected_response = extract_final_number(expected_raw) or expected_raw
        success = parsed_response is not None and parsed_response == expected_response
        return parsed_response, expected_response, success

    return raw_response, expected_raw, _generic_success(metrics)

