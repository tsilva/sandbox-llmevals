from pathlib import Path

from llmevals.web.schemas import RunStartRequest, SampleResult
from llmevals.web.store import RunStore


def test_run_store_persists_run_and_samples(tmp_path: Path):
    store = RunStore(base_dir=tmp_path)
    request = RunStartRequest(model_config_name="default", benchmark_config_name="gsm8k", limit=3)
    settings = {
        "model": {
            "id": "model-a",
            "identifier": "model-a",
            "litellm_model_name": "openai/model-a",
        },
        "benchmark": {"limit": 3},
    }

    run = store.create_run(request, settings)
    store.update_run(run.run_id, status="running", total_samples=3)
    store.append_sample(
        run.run_id,
        SampleResult(
            sample_index=0,
            sample_id="0",
            task_name="gsm8k|5",
            raw_prompt="prompt",
            raw_response="response",
            parsed_response="1",
            expected_response="1",
            success=True,
            metrics={"extractive_match": 1.0},
            completed_at="2026-04-15T00:00:00Z",
        ),
    )
    store.write_results(run.run_id, {"results": {"all": {"extractive_match": 1.0}}})

    reloaded = store.get_run(run.run_id)
    samples = store.list_samples(run.run_id)

    assert reloaded.run_id == run.run_id
    assert reloaded.total_samples == 3
    assert len(samples) == 1
    assert samples[0].success is True

