from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from llmevals.web.app import create_app
from llmevals.web.manager import RunManager
from llmevals.web.schemas import RunEvent, RunStartRequest, SampleResult
from llmevals.web.store import RunStore, utc_now


class StreamingFakeRunner:
    def __init__(self, sample_count: int = 3, sample_delay: float = 0.01):
        self.sample_count = sample_count
        self.sample_delay = sample_delay

    def run(self, run_id, request, settings, store, publish):
        detail = store.update_run(run_id, total_samples=self.sample_count, score_metric="accuracy")
        publish(RunEvent(type="progress", run_id=run_id, data=detail.model_dump(mode="json")))

        for index in range(self.sample_count):
            sample = SampleResult(
                sample_index=index,
                sample_id=str(index),
                task_name="gsm8k|5",
                raw_prompt=f"prompt {index}",
                raw_response=f"response {index}",
                parsed_response=str(index),
                expected_response=str(index),
                success=True,
                metrics={"accuracy": 1.0},
                completed_at=utc_now(),
            )
            store.append_sample(run_id, sample)
            publish(RunEvent(type="sample_completed", run_id=run_id, data=sample.model_dump(mode="json")))
            detail = store.update_run(
                run_id,
                completed_samples=index + 1,
                score=(index + 1) / self.sample_count,
                score_metric="accuracy",
            )
            publish(RunEvent(type="progress", run_id=run_id, data=detail.model_dump(mode="json")))
            time.sleep(self.sample_delay)

        results_path = store.write_results(run_id, {"results": {"all": {"accuracy": 1.0}}})
        detail = store.update_run(
            run_id,
            status="finished",
            completed_at=utc_now(),
            score=1.0,
            score_metric="accuracy",
            results_path=results_path,
        )
        publish(RunEvent(type="run_finished", run_id=run_id, data=detail.model_dump(mode="json")))


class BlockingFakeRunner:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def run(self, run_id, request, settings, store, publish):
        detail = store.update_run(run_id, total_samples=1, score_metric="accuracy")
        publish(RunEvent(type="progress", run_id=run_id, data=detail.model_dump(mode="json")))
        self.started.set()
        self.release.wait(timeout=5)
        sample = SampleResult(
            sample_index=0,
            sample_id="0",
            task_name="gsm8k|5",
            raw_prompt="prompt",
            raw_response="response",
            parsed_response="0",
            expected_response="0",
            success=True,
            metrics={"accuracy": 1.0},
            completed_at=utc_now(),
        )
        store.append_sample(run_id, sample)
        publish(RunEvent(type="sample_completed", run_id=run_id, data=sample.model_dump(mode="json")))
        results_path = store.write_results(run_id, {"results": {"all": {"accuracy": 1.0}}})
        detail = store.update_run(
            run_id,
            status="finished",
            completed_at=utc_now(),
            completed_samples=1,
            score=1.0,
            score_metric="accuracy",
            results_path=results_path,
        )
        publish(RunEvent(type="run_finished", run_id=run_id, data=detail.model_dump(mode="json")))


def make_client(tmp_path: Path, runner) -> tuple[TestClient, RunManager]:
    manager = RunManager(store=RunStore(base_dir=tmp_path / "web_runs"), runner=runner)
    app = create_app(manager)
    return TestClient(app), manager


def start_payload() -> dict[str, object]:
    return {
        "model_config_name": "default",
        "benchmark_config_name": "gsm8k",
        "limit": 3,
    }


def test_start_run_and_fetch_history(tmp_path: Path):
    client, manager = make_client(tmp_path, StreamingFakeRunner())
    response = client.post("/api/runs", json=start_payload())
    assert response.status_code == 200

    manager.wait_for_active_run(timeout=5)

    runs_response = client.get("/api/runs")
    assert runs_response.status_code == 200
    runs = runs_response.json()
    assert len(runs) == 1
    assert runs[0]["benchmark_config_name"] == "gsm8k"


def test_index_renders_primary_navigation(tmp_path: Path):
    client, _ = make_client(tmp_path, StreamingFakeRunner())

    response = client.get("/")

    assert response.status_code == 200
    assert "Dashboard" in response.text
    assert "Launchpad" in response.text
    assert "Monitoring" in response.text
    assert "Results" in response.text
    assert "Select runtime template..." in response.text


def test_lists_local_models(tmp_path: Path):
    client, manager = make_client(tmp_path, StreamingFakeRunner())
    manager.local_models = lambda model_config_name=None: [
        {
            "value": "qwen/qwen3.5-9b",
            "label": "Qwen3.5 9B · qwen · 9B",
            "source": "lmstudio_cli",
            "model_key": "qwen/qwen3.5-9b",
            "path": "qwen/qwen3.5-9b",
            "context_length": 262144,
        }
    ]

    response = client.get("/api/providers/local/models")

    assert response.status_code == 200
    assert response.json()[0]["value"] == "qwen/qwen3.5-9b"


def test_rejects_second_active_run(tmp_path: Path):
    runner = BlockingFakeRunner()
    client, manager = make_client(tmp_path, runner)

    first = client.post("/api/runs", json=start_payload())
    assert first.status_code == 200
    runner.started.wait(timeout=5)

    second = client.post("/api/runs", json=start_payload())
    assert second.status_code == 409

    runner.release.set()
    manager.wait_for_active_run(timeout=5)


def test_fetch_run_and_samples(tmp_path: Path):
    client, manager = make_client(tmp_path, StreamingFakeRunner(sample_count=3))
    create_response = client.post("/api/runs", json=start_payload())
    run_id = create_response.json()["run_id"]

    manager.wait_for_active_run(timeout=5)

    run_response = client.get(f"/api/runs/{run_id}")
    samples_response = client.get(f"/api/runs/{run_id}/samples")

    assert run_response.status_code == 200
    assert run_response.json()["status"] == "finished"
    assert samples_response.status_code == 200
    assert len(samples_response.json()) == 3


def test_sse_event_sequence(tmp_path: Path):
    client, manager = make_client(tmp_path, StreamingFakeRunner(sample_count=2, sample_delay=0.02))
    create_response = client.post("/api/runs", json=start_payload())
    run_id = create_response.json()["run_id"]

    events: list[str] = []
    with client.stream("GET", f"/api/runs/{run_id}/events") as response:
        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode() if isinstance(line, bytes) else line
            if decoded.startswith("event:"):
                events.append(decoded.split(":", 1)[1].strip())
            if events and events[-1] == "run_finished":
                break

    manager.wait_for_active_run(timeout=5)
    assert events[0] == "run_started"
    assert "sample_completed" in events
    assert events[-1] == "run_finished"


def test_refresh_can_recover_in_progress_run(tmp_path: Path):
    runner = BlockingFakeRunner()
    client, manager = make_client(tmp_path, runner)
    create_response = client.post("/api/runs", json=start_payload())
    run_id = create_response.json()["run_id"]
    runner.started.wait(timeout=5)

    run_response = client.get(f"/api/runs/{run_id}")
    samples_response = client.get(f"/api/runs/{run_id}/samples")

    assert run_response.status_code == 200
    assert run_response.json()["status"] == "running"
    assert samples_response.status_code == 200
    assert samples_response.json() == []

    runner.release.set()
    manager.wait_for_active_run(timeout=5)
