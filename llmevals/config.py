from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT_DIR / "configs"

DEFAULT_MODEL = "default"
DEFAULT_BENCHMARK = "gsm8k"


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at {path}, got {type(data).__name__}")
    return data


def _candidate_paths(name_or_path: str, kind: str) -> list[Path]:
    raw = Path(name_or_path).expanduser()
    candidates = [raw]
    if not raw.suffix:
        candidates.append(raw.with_suffix(".yaml"))

    config_dir = CONFIGS_DIR / kind
    candidates.append(config_dir / name_or_path)
    if not raw.suffix:
        candidates.append(config_dir / f"{name_or_path}.yaml")

    return candidates


def resolve_config_path(name_or_path: str | None, kind: str) -> Path:
    default_name = DEFAULT_MODEL if kind == "models" else DEFAULT_BENCHMARK
    target = name_or_path or default_name

    for candidate in _candidate_paths(target, kind):
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(f"Could not resolve {kind[:-1]} config: {target}")


def _sanitize_task_name(task_name: str) -> str:
    return task_name.replace("/", "_").replace("|", "_")


def _normalize_model_config(config: dict[str, Any], source_path: Path) -> dict[str, Any]:
    model = deepcopy(config.get("model", {}))
    if not isinstance(model, dict):
        raise ValueError(f"Invalid model config in {source_path}")

    lm_studio = deepcopy(model.get("lm_studio", {}))
    if not isinstance(lm_studio, dict):
        raise ValueError(f"Invalid model.lm_studio config in {source_path}")

    model_id = model.get("id")
    if not model_id:
        raise ValueError(f"Model config {source_path} is missing model.id")

    lm_studio.setdefault("bind_address", "127.0.0.1")
    lm_studio.setdefault("port", 1234)
    lm_studio.setdefault("parallel", 1)
    lm_studio.setdefault("context_length", 8192)
    lm_studio.setdefault("gpu", "max")
    lm_studio.setdefault(
        "base_url",
        f"http://{lm_studio['bind_address']}:{lm_studio['port']}/v1",
    )

    model.setdefault("name", source_path.stem)
    model.setdefault("identifier", model_id)
    model.setdefault("litellm_model_name", f"openai/{model_id}")
    model.setdefault("extra_lighteval_model_parameters", {})
    model["lm_studio"] = lm_studio

    return {"model": model}


def _normalize_benchmark_config(config: dict[str, Any], source_path: Path) -> dict[str, Any]:
    benchmark = deepcopy(config.get("benchmark", {}))
    if not isinstance(benchmark, dict):
        raise ValueError(f"Invalid benchmark config in {source_path}")

    task_name = benchmark.get("task_name")
    if not task_name:
        raise ValueError(f"Benchmark config {source_path} is missing benchmark.task_name")

    benchmark.setdefault("name", source_path.stem)
    benchmark.setdefault("limit", None)
    benchmark.setdefault(
        "output_dir",
        str((ROOT_DIR / "results-lighteval" / benchmark["name"]).resolve()),
    )
    benchmark.setdefault(
        "cache_dir",
        str((ROOT_DIR / ".cache" / "lighteval" / _sanitize_task_name(task_name)).resolve()),
    )
    benchmark.setdefault(
        "run_id_template",
        "lighteval_{benchmark}_p{parallel}_c{concurrent_requests}_ctx{context_length}_n{limit}",
    )

    lighteval_cfg = deepcopy(benchmark.get("lighteval", {}))
    if not isinstance(lighteval_cfg, dict):
        raise ValueError(f"Invalid benchmark.lighteval config in {source_path}")

    lighteval_cfg.setdefault("concurrent_requests", 8)
    lighteval_cfg.setdefault("max_gen_toks", 256)
    lighteval_cfg.setdefault("max_length", None)
    lighteval_cfg.setdefault("system_prompt", "")
    lighteval_cfg.setdefault("model_parameters", {})
    lighteval_cfg.setdefault("generation_parameters", {})
    lighteval_cfg.setdefault("extra_args", [])
    benchmark["lighteval"] = lighteval_cfg

    throughput_cfg = deepcopy(benchmark.get("throughput", {}))
    if not isinstance(throughput_cfg, dict):
        raise ValueError(f"Invalid benchmark.throughput config in {source_path}")

    throughput_cfg.setdefault("parallel", None)
    throughput_cfg.setdefault("auto_retry_failed_run", True)
    throughput_cfg.setdefault("retry_parallel", 1)
    throughput_cfg.setdefault("retry_concurrent_requests", 1)
    benchmark["throughput"] = throughput_cfg

    return {"benchmark": benchmark}


def set_nested(mapping: dict[str, Any], path: str, value: Any) -> None:
    cursor = mapping
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def resolve_settings(
    *,
    model: str | None,
    benchmark: str | None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_path = resolve_config_path(model, "models")
    benchmark_path = resolve_config_path(benchmark, "benchmarks")

    merged = _deep_merge(
        _normalize_model_config(_load_yaml(model_path), model_path),
        _normalize_benchmark_config(_load_yaml(benchmark_path), benchmark_path),
    )
    merged["paths"] = {
        "root_dir": str(ROOT_DIR),
        "model_config": str(model_path),
        "benchmark_config": str(benchmark_path),
    }

    if overrides:
        merged = _deep_merge(merged, overrides)

    lm_overrides = (overrides or {}).get("model", {}).get("lm_studio", {})
    if ("port" in lm_overrides or "bind_address" in lm_overrides) and "base_url" not in lm_overrides:
        lm_studio = merged["model"]["lm_studio"]
        lm_studio["base_url"] = f"http://{lm_studio['bind_address']}:{lm_studio['port']}/v1"

    lighteval_cfg = merged["benchmark"]["lighteval"]
    if lighteval_cfg.get("max_length") is None:
        lighteval_cfg["max_length"] = merged["model"]["lm_studio"]["context_length"]

    throughput_cfg = merged["benchmark"]["throughput"]
    if throughput_cfg.get("parallel") is None:
        throughput_cfg["parallel"] = merged["model"]["lm_studio"]["parallel"]

    return merged
