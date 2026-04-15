from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


RunStatus = Literal["queued", "running", "finished", "failed"]
RunEventType = Literal["run_started", "progress", "sample_completed", "run_finished", "run_failed"]


class ConfigOption(BaseModel):
    name: str
    path: str
    model_id: str | None = None
    identifier: str | None = None
    litellm_model_name: str | None = None
    context_length: int | None = None
    parallel: int | None = None
    task_name: str | None = None
    limit: int | None = None
    system_prompt: str | None = None
    concurrent_requests: int | None = None


class ConfigsResponse(BaseModel):
    model_configs: list[ConfigOption]
    benchmark_configs: list[ConfigOption]


class DiscoveredModelOption(BaseModel):
    value: str
    label: str
    source: str
    model_key: str | None = None
    path: str | None = None
    context_length: int | None = None


class RunStartRequest(BaseModel):
    model_config_name: str
    benchmark_config_name: str
    model_id: str | None = None
    identifier: str | None = None
    litellm_model_name: str | None = None
    limit: int | None = Field(default=None, ge=1)


class SampleResult(BaseModel):
    sample_index: int
    sample_id: str
    task_name: str
    raw_prompt: str
    raw_response: str
    parsed_response: str | None = None
    expected_response: str | None = None
    success: bool | None = None
    metrics: dict[str, Any]
    completed_at: str


class RunSummary(BaseModel):
    run_id: str
    status: RunStatus
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None
    model_config_name: str
    benchmark_config_name: str
    model_id: str
    identifier: str
    litellm_model_name: str
    limit: int | None = None
    total_samples: int | None = None
    completed_samples: int = 0
    progress: float = 0.0
    score: float | None = None
    score_metric: str | None = None
    error_message: str | None = None


class RunDetail(RunSummary):
    run_dir: str
    samples_path: str
    stdout_path: str
    stderr_path: str
    results_path: str | None = None
    official_results_path: str | None = None


class RunEvent(BaseModel):
    type: RunEventType
    run_id: str
    data: dict[str, Any]
