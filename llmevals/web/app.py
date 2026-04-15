from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from .manager import ActiveRunError, RunManager
from .schemas import ConfigsResponse, DiscoveredModelOption, RunDetail, RunStartRequest, RunSummary, SampleResult


INDEX_HTML = Path(__file__).with_name("index.html")


def create_app(run_manager: RunManager | None = None) -> FastAPI:
    manager = run_manager or RunManager()
    app = FastAPI(title="LLM Benchmarks UI")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return INDEX_HTML.read_text(encoding="utf-8")

    @app.get("/api/configs", response_model=ConfigsResponse)
    async def get_configs() -> ConfigsResponse:
        return manager.configs()

    @app.get("/api/providers/local/models", response_model=list[DiscoveredModelOption])
    async def get_local_models(model_config_name: str | None = None) -> list[DiscoveredModelOption]:
        return manager.local_models(model_config_name)

    @app.get("/api/runs", response_model=list[RunSummary])
    async def list_runs() -> list[RunSummary]:
        return manager.list_runs()

    @app.post("/api/runs", response_model=RunDetail)
    async def start_run(request: RunStartRequest) -> RunDetail:
        try:
            return manager.start_run(request)
        except ActiveRunError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}", response_model=RunDetail)
    async def get_run(run_id: str) -> RunDetail:
        try:
            return manager.get_run(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc

    @app.get("/api/runs/{run_id}/samples", response_model=list[SampleResult])
    async def get_samples(run_id: str) -> list[SampleResult]:
        try:
            manager.get_run(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        return manager.get_samples(run_id)

    @app.get("/api/runs/{run_id}/events")
    async def get_run_events(run_id: str) -> StreamingResponse:
        try:
            manager.get_run(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        return StreamingResponse(manager.stream_events(run_id), media_type="text/event-stream")

    return app


app = create_app()
