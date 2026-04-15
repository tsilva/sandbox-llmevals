from llmevals.config import resolve_settings
from llmevals.runtime import GSM8K_STRICT_SYSTEM_PROMPT, build_litellm_model_parameters


def test_gsm8k_injects_strict_output_format_prompt():
    settings = resolve_settings(model="default", benchmark="gsm8k")

    model_parameters = build_litellm_model_parameters(settings, api_key="test-key")

    assert model_parameters["system_prompt"] == GSM8K_STRICT_SYSTEM_PROMPT
    assert "The final line must be exactly: #### <answer>" in model_parameters["system_prompt"]


def test_gsm8k_reasoning_keeps_existing_prompt_and_appends_strict_rules():
    settings = resolve_settings(model="default", benchmark="gsm8k_reasoning")

    model_parameters = build_litellm_model_parameters(settings, api_key="test-key")
    system_prompt = model_parameters["system_prompt"]

    assert "Solve the math problem carefully." in system_prompt
    assert "Show the intermediate reasoning steps." in system_prompt
    assert "The final line must be exactly: #### <answer>" in system_prompt
    assert system_prompt.endswith("Do not write anything after the final #### <answer> line.")


def test_non_gsm8k_task_does_not_inject_strict_prompt():
    settings = resolve_settings(
        model="default",
        benchmark="gsm8k",
        overrides={"benchmark": {"task_name": "custom|0"}},
    )

    model_parameters = build_litellm_model_parameters(settings, api_key="test-key")

    assert "system_prompt" not in model_parameters
