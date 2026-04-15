from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llmevals.config import WEB_RUNS_DIR

from .schemas import RunDetail, RunStartRequest, RunSummary, SampleResult


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class RunStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = (base_dir or WEB_RUNS_DIR).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _run_dir(self, run_id: str) -> Path:
        return self.base_dir / run_id

    def _run_json_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "run.json"

    def _samples_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "samples.jsonl"

    def _stdout_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "stdout.log"

    def _stderr_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "stderr.log"

    def _results_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "results.json"

    def _lighteval_dir(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "lighteval"

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        tmp_path.replace(path)

    def _build_run_id(self) -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}-{uuid.uuid4().hex[:8]}"

    def create_run(self, request: RunStartRequest, settings: dict[str, Any]) -> RunDetail:
        with self._lock:
            run_id = self._build_run_id()
            run_dir = self._run_dir(run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            self._samples_path(run_id).touch()
            self._stdout_path(run_id).touch()
            self._stderr_path(run_id).touch()

            detail = RunDetail(
                run_id=run_id,
                status="queued",
                created_at=utc_now(),
                updated_at=utc_now(),
                model_config_name=request.model_config_name,
                benchmark_config_name=request.benchmark_config_name,
                model_id=settings["model"]["id"],
                identifier=settings["model"]["identifier"],
                litellm_model_name=settings["model"]["litellm_model_name"],
                limit=settings["benchmark"].get("limit"),
                completed_samples=0,
                progress=0.0,
                run_dir=str(run_dir),
                samples_path=str(self._samples_path(run_id)),
                stdout_path=str(self._stdout_path(run_id)),
                stderr_path=str(self._stderr_path(run_id)),
            )
            self._write_json_atomic(self._run_json_path(run_id), detail.model_dump(mode="json"))
            return detail

    def get_run(self, run_id: str) -> RunDetail:
        path = self._run_json_path(run_id)
        if not path.is_file():
            raise FileNotFoundError(run_id)
        return RunDetail.model_validate_json(path.read_text(encoding="utf-8"))

    def save_run(self, detail: RunDetail) -> RunDetail:
        with self._lock:
            detail.updated_at = utc_now()
            self._write_json_atomic(self._run_json_path(detail.run_id), detail.model_dump(mode="json"))
            return detail

    def update_run(self, run_id: str, **updates: Any) -> RunDetail:
        detail = self.get_run(run_id)
        for key, value in updates.items():
            setattr(detail, key, value)
        if detail.total_samples:
            detail.progress = detail.completed_samples / detail.total_samples
        elif detail.completed_samples:
            detail.progress = 1.0
        else:
            detail.progress = 0.0
        return self.save_run(detail)

    def append_sample(self, run_id: str, sample: SampleResult) -> None:
        with self._lock:
            with self._samples_path(run_id).open("a", encoding="utf-8") as handle:
                handle.write(sample.model_dump_json())
                handle.write("\n")

    def list_samples(self, run_id: str) -> list[SampleResult]:
        path = self._samples_path(run_id)
        if not path.is_file():
            return []
        items: list[SampleResult] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(SampleResult.model_validate_json(line))
        return items

    def write_log(self, run_id: str, stream: str, message: str) -> None:
        path = self._stdout_path(run_id) if stream == "stdout" else self._stderr_path(run_id)
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(message.rstrip())
                handle.write("\n")

    def write_results(self, run_id: str, payload: dict[str, Any]) -> str:
        path = self._results_path(run_id)
        with self._lock:
            self._write_json_atomic(path, payload)
        return str(path)

    def lighteval_dir(self, run_id: str) -> Path:
        path = self._lighteval_dir(run_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def list_runs(self) -> list[RunSummary]:
        runs: list[RunSummary] = []
        for path in sorted(self.base_dir.glob("*/run.json"), reverse=True):
            detail = RunDetail.model_validate_json(path.read_text(encoding="utf-8"))
            runs.append(RunSummary.model_validate(detail.model_dump()))
        runs.sort(key=lambda item: item.created_at, reverse=True)
        return runs

    def mark_incomplete_runs_failed(self, message: str) -> None:
        for run in self.list_runs():
            if run.status in {"queued", "running"}:
                self.update_run(
                    run.run_id,
                    status="failed",
                    completed_at=utc_now(),
                    error_message=message,
                )
