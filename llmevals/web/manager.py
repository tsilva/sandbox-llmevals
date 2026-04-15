from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

from llmevals.config import resolve_settings
from llmevals.runtime import list_available_benchmark_configs, list_available_model_configs, list_discoverable_local_models

from .runner import LightevalWebRunner
from .schemas import ConfigOption, ConfigsResponse, DiscoveredModelOption, RunDetail, RunEvent, RunStartRequest, RunSummary, SampleResult
from .store import RunStore, utc_now


class ActiveRunError(RuntimeError):
    pass


def _process_entrypoint(
    runner_cls: type,
    run_id: str,
    request_payload: dict[str, Any],
    settings: dict[str, Any],
    store_dir: str,
    event_queue: mp.Queue,
) -> None:
    request = RunStartRequest.model_validate(request_payload)
    store = RunStore(base_dir=Path(store_dir))
    runner = runner_cls()

    def publish(event: RunEvent) -> None:
        event_queue.put(event.model_dump(mode="json"))

    runner.run(run_id, request, settings, store, publish)


def _sanitize_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "benchmark_model"


class RunManager:
    def __init__(self, store: RunStore | None = None, runner: Any | None = None) -> None:
        self.store = store or RunStore()
        self.runner = runner or LightevalWebRunner()
        self.store.mark_incomplete_runs_failed("Server restarted while benchmark was active.")
        self._lock = threading.Lock()
        self._active_thread: threading.Thread | None = None
        self._active_process: mp.Process | None = None
        self._active_run_id: str | None = None
        self._subscribers: dict[str, list[queue.Queue[RunEvent]]] = defaultdict(list)

    def configs(self) -> ConfigsResponse:
        return ConfigsResponse(
            model_configs=[ConfigOption.model_validate(item) for item in list_available_model_configs()],
            benchmark_configs=[ConfigOption.model_validate(item) for item in list_available_benchmark_configs()],
        )

    def local_models(self, model_config_name: str | None = None) -> list[DiscoveredModelOption]:
        return [
            DiscoveredModelOption.model_validate(item)
            for item in list_discoverable_local_models(model=model_config_name)
        ]

    def list_runs(self) -> list[RunSummary]:
        return self.store.list_runs()

    def get_run(self, run_id: str) -> RunDetail:
        return self.store.get_run(run_id)

    def get_samples(self, run_id: str) -> list[SampleResult]:
        return self.store.list_samples(run_id)

    def _has_active_run(self) -> bool:
        thread_alive = self._active_thread is not None and self._active_thread.is_alive()
        process_alive = self._active_process is not None and self._active_process.is_alive()
        return thread_alive or process_alive

    def _request_overrides(self, request: RunStartRequest, settings: dict[str, Any]) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        if request.model_id:
            overrides.setdefault("model", {})["id"] = request.model_id

        identifier = request.identifier
        if request.model_id and not identifier:
            identifier = _sanitize_identifier(Path(request.model_id).stem)
        if identifier:
            overrides.setdefault("model", {})["identifier"] = identifier

        litellm_model_name = request.litellm_model_name
        if identifier and not litellm_model_name:
            litellm_model_name = f"openai/{identifier}"
        if litellm_model_name:
            overrides.setdefault("model", {})["litellm_model_name"] = litellm_model_name

        if request.limit is not None:
            overrides.setdefault("benchmark", {})["limit"] = request.limit

        return overrides

    def _resolve_settings(self, request: RunStartRequest) -> dict[str, Any]:
        base_settings = resolve_settings(model=request.model_config_name, benchmark=request.benchmark_config_name)
        overrides = self._request_overrides(request, base_settings)
        if not overrides:
            return base_settings
        return resolve_settings(
            model=request.model_config_name,
            benchmark=request.benchmark_config_name,
            overrides=overrides,
        )

    def _publish(self, event: RunEvent) -> None:
        for subscriber in list(self._subscribers.get(event.run_id, [])):
            subscriber.put(event)

    def _clear_active(self, run_id: str) -> None:
        with self._lock:
            if self._active_run_id == run_id:
                self._active_run_id = None
                self._active_thread = None
                self._active_process = None

    def _watch_process(self, run_id: str, process: mp.Process, event_queue: mp.Queue) -> None:
        saw_terminal_event = False
        while True:
            try:
                payload = event_queue.get(timeout=0.2)
            except queue.Empty:
                if not process.is_alive():
                    break
                continue

            event = RunEvent.model_validate(payload)
            self._publish(event)
            if event.type in {"run_finished", "run_failed"}:
                saw_terminal_event = True
                break

        process.join(timeout=1)

        if not saw_terminal_event:
            try:
                run = self.store.get_run(run_id)
            except FileNotFoundError:
                run = None
            if run is not None and run.status not in {"finished", "failed"}:
                detail = self.store.update_run(
                    run_id,
                    status="failed",
                    completed_at=utc_now(),
                    error_message="Benchmark process exited unexpectedly.",
                )
                self._publish(RunEvent(type="run_failed", run_id=run_id, data=detail.model_dump(mode="json")))

        self._clear_active(run_id)

    def start_run(self, request: RunStartRequest) -> RunDetail:
        with self._lock:
            if self._has_active_run():
                raise ActiveRunError("Another benchmark run is already active.")

            settings = self._resolve_settings(request)
            detail = self.store.create_run(request, settings)

            started = self.store.update_run(detail.run_id, status="running", started_at=utc_now())
            self._publish(RunEvent(type="run_started", run_id=detail.run_id, data=started.model_dump(mode="json")))

            self._active_run_id = detail.run_id
            execution_mode = getattr(self.runner, "execution_mode", "thread")

            if execution_mode == "process":
                event_queue: mp.Queue = mp.Queue()
                process = mp.Process(
                    target=_process_entrypoint,
                    args=(
                        self.runner.__class__,
                        detail.run_id,
                        request.model_dump(mode="json"),
                        settings,
                        str(self.store.base_dir),
                        event_queue,
                    ),
                    name=f"benchmark-run-{detail.run_id}",
                    daemon=True,
                )
                watcher = threading.Thread(
                    target=self._watch_process,
                    args=(detail.run_id, process, event_queue),
                    name=f"benchmark-watch-{detail.run_id}",
                    daemon=True,
                )
                self._active_process = process
                self._active_thread = watcher
                process.start()
                watcher.start()
                return started

            def target() -> None:
                try:
                    self.runner.run(detail.run_id, request, settings, self.store, self._publish)
                except Exception:
                    pass
                finally:
                    self._clear_active(detail.run_id)

            thread = threading.Thread(target=target, name=f"benchmark-run-{detail.run_id}", daemon=True)
            self._active_thread = thread
            thread.start()
            return started

    def subscribe(self, run_id: str) -> queue.Queue[RunEvent]:
        channel: queue.Queue[RunEvent] = queue.Queue()
        self._subscribers[run_id].append(channel)
        return channel

    def unsubscribe(self, run_id: str, channel: queue.Queue[RunEvent]) -> None:
        subscribers = self._subscribers.get(run_id, [])
        if channel in subscribers:
            subscribers.remove(channel)
        if not subscribers and run_id in self._subscribers:
            del self._subscribers[run_id]

    async def stream_events(self, run_id: str):
        run = self.get_run(run_id)
        if run.status in {"finished", "failed"}:
            final_type = "run_finished" if run.status == "finished" else "run_failed"
            event = RunEvent(type=final_type, run_id=run_id, data=run.model_dump(mode="json"))
            yield self._format_sse(event)
            return

        snapshot = RunEvent(type="run_started", run_id=run_id, data=run.model_dump(mode="json"))
        yield self._format_sse(snapshot)

        channel = self.subscribe(run_id)
        try:
            while True:
                event = await asyncio.to_thread(channel.get)
                yield self._format_sse(event)
                if event.type in {"run_finished", "run_failed"}:
                    break
        finally:
            self.unsubscribe(run_id, channel)

    def _format_sse(self, event: RunEvent) -> str:
        return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"

    def wait_for_active_run(self, timeout: float | None = None) -> None:
        thread = self._active_thread
        process = self._active_process
        if process is not None:
            process.join(timeout)
        if thread is not None:
            thread.join(timeout)
