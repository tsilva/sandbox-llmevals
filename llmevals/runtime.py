from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any

from lighteval.models.endpoints.litellm_model import LiteLLMModelConfig
from lighteval.models.model_input import GenerationParameters

from .config import list_config_paths, resolve_settings

GSM8K_STRICT_SYSTEM_PROMPT = """Output format rules:
- Solve the problem and make sure the final answer is correct.
- The final line must be exactly: #### <answer>
- Replace <answer> with only the numeric answer, using digits and an optional leading minus sign or decimal point.
- Do not include units, words, commas, LaTeX, or any extra text on the final line.
- Do not write anything after the final #### <answer> line."""


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def ensure_lm_studio(base_url: str, api_key: str | None) -> None:
    url = f"{base_url.rstrip('/')}/models"
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=5):
            return
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LM Studio is not reachable at {url}: {exc}") from exc


def _fetch_lm_studio_models(base_url: str, api_key: str | None) -> list[dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/models"
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")

    with urllib.request.urlopen(request, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))

    items = payload.get("data", payload)
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def run_command(cmd: list[str], *, cwd: Path, check: bool, quiet: bool = False) -> int:
    stdout = subprocess.DEVNULL if quiet else None
    stderr = subprocess.DEVNULL if quiet else None
    completed = subprocess.run(cmd, cwd=cwd, check=False, stdout=stdout, stderr=stderr)
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, cmd)
    return completed.returncode


def stop_server(root_dir: Path, identifier: str) -> None:
    run_command(["lms", "server", "stop"], cwd=root_dir, check=False, quiet=True)
    run_command(["lms", "unload", identifier], cwd=root_dir, check=False, quiet=True)
    run_command(["lms", "unload", "-a"], cwd=root_dir, check=False, quiet=True)


def start_server(
    root_dir: Path,
    *,
    model_id: str,
    identifier: str,
    port: int,
    bind_address: str,
    parallel: int,
    context_length: int,
    gpu: str,
) -> None:
    stop_server(root_dir, identifier)
    run_command(["lms", "server", "start", "--port", str(port), "--bind", bind_address], cwd=root_dir, check=True, quiet=True)
    run_command(
        [
            "lms",
            "load",
            model_id,
            "--yes",
            "--identifier",
            identifier,
            "--parallel",
            str(parallel),
            "--context-length",
            str(context_length),
            "--gpu",
            gpu,
        ],
        cwd=root_dir,
        check=True,
    )


def build_generation_parameters(settings: dict[str, Any]) -> GenerationParameters:
    lighteval_cfg = settings["benchmark"]["lighteval"]
    params = deepcopy(lighteval_cfg.get("generation_parameters", {}))
    params.setdefault("max_new_tokens", int(lighteval_cfg["max_gen_toks"]))
    return GenerationParameters(**params)


def is_gsm8k_task(settings: dict[str, Any]) -> bool:
    task_name = str(settings["benchmark"].get("task_name", ""))
    return task_name.split("|", 1)[0] == "gsm8k"


def build_system_prompt(settings: dict[str, Any]) -> str:
    lighteval_cfg = settings["benchmark"]["lighteval"]
    base_prompt = str(lighteval_cfg.get("system_prompt", "") or "").strip()

    if not is_gsm8k_task(settings):
        return base_prompt

    if GSM8K_STRICT_SYSTEM_PROMPT in base_prompt:
        return base_prompt
    if not base_prompt:
        return GSM8K_STRICT_SYSTEM_PROMPT
    return f"{base_prompt}\n\n{GSM8K_STRICT_SYSTEM_PROMPT}"


def build_litellm_model_parameters(settings: dict[str, Any], api_key: str | None) -> dict[str, Any]:
    model_cfg = settings["model"]
    benchmark_cfg = settings["benchmark"]
    lighteval_cfg = benchmark_cfg["lighteval"]

    model_parameters = {
        "model_name": model_cfg["litellm_model_name"],
        "base_url": model_cfg["lm_studio"]["base_url"].rstrip("/"),
        "api_key": api_key or "dummy",
        "concurrent_requests": int(lighteval_cfg["concurrent_requests"]),
        "max_model_length": int(lighteval_cfg["max_length"]),
        "cache_dir": benchmark_cfg["cache_dir"],
        "generation_parameters": build_generation_parameters(settings),
    }
    model_parameters = deep_merge(model_parameters, model_cfg.get("extra_lighteval_model_parameters", {}))
    model_parameters = deep_merge(model_parameters, lighteval_cfg.get("model_parameters", {}))

    system_prompt = build_system_prompt(settings)
    if system_prompt:
        model_parameters["system_prompt"] = system_prompt

    return model_parameters


def build_litellm_model_config(settings: dict[str, Any], api_key: str | None) -> LiteLLMModelConfig:
    return LiteLLMModelConfig(**build_litellm_model_parameters(settings, api_key))


def list_discoverable_local_models(model: str | None = None) -> list[dict[str, Any]]:
    discovered: list[dict[str, Any]] = []
    seen: set[str] = set()

    try:
        completed = subprocess.run(
            ["lms", "ls", "--json", "--llm"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            payload = json.loads(completed.stdout)
            if isinstance(payload, list):
                for item in payload:
                    if not isinstance(item, dict):
                        continue
                    value = str(item.get("modelKey") or item.get("indexedModelIdentifier") or item.get("path") or "").strip()
                    if not value or value in seen:
                        continue
                    seen.add(value)
                    label_parts = [str(item.get("displayName") or value)]
                    publisher = str(item.get("publisher") or "").strip()
                    params = str(item.get("paramsString") or "").strip()
                    if publisher:
                        label_parts.append(publisher)
                    if params:
                        label_parts.append(params)
                    discovered.append(
                        {
                            "value": value,
                            "label": " · ".join(label_parts),
                            "source": "lmstudio_cli",
                            "model_key": item.get("modelKey"),
                            "path": item.get("path"),
                            "context_length": item.get("maxContextLength"),
                        }
                    )
    except (OSError, json.JSONDecodeError):
        pass

    if discovered:
        return discovered

    try:
        settings = resolve_settings(model=model, benchmark=None)
        model_cfg = settings["model"]
        api_key = api_key_from_env()
        items = _fetch_lm_studio_models(model_cfg["lm_studio"]["base_url"], api_key)
        for item in items:
            value = str(item.get("id") or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            discovered.append(
                {
                    "value": value,
                    "label": value,
                    "source": "lmstudio_http",
                    "model_key": value,
                    "path": None,
                    "context_length": None,
                }
            )
    except (FileNotFoundError, RuntimeError, urllib.error.URLError, json.JSONDecodeError):
        return []

    return discovered


def list_available_model_configs() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in list_config_paths("models"):
        settings = resolve_settings(model=path.as_posix(), benchmark=None)
        model_cfg = settings["model"]
        items.append(
            {
                "name": path.stem,
                "path": str(path),
                "model_id": model_cfg["id"],
                "identifier": model_cfg["identifier"],
                "litellm_model_name": model_cfg["litellm_model_name"],
                "context_length": model_cfg["lm_studio"]["context_length"],
                "parallel": model_cfg["lm_studio"]["parallel"],
            }
        )
    return items


def list_available_benchmark_configs() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in list_config_paths("benchmarks"):
        settings = resolve_settings(model=None, benchmark=path.as_posix())
        benchmark_cfg = settings["benchmark"]
        items.append(
            {
                "name": path.stem,
                "path": str(path),
                "task_name": benchmark_cfg["task_name"],
                "limit": benchmark_cfg.get("limit"),
                "system_prompt": benchmark_cfg["lighteval"].get("system_prompt", ""),
                "concurrent_requests": benchmark_cfg["lighteval"]["concurrent_requests"],
            }
        )
    return items


def pretty_prompt(value: str | list | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("role") or "unknown"
                content = item.get("content") or ""
                chunks.append(f"[{role}] {content}".strip())
            else:
                chunks.append(str(item))
        return "\n\n".join(chunks)
    return json.dumps(value, ensure_ascii=False, indent=2)


def api_key_from_env() -> str | None:
    return os.environ.get("OPENAI_API_KEY")
