from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .config import ROOT_DIR, resolve_settings, set_nested


def env_str(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else None


def env_int(name: str) -> int | None:
    value = env_str(name)
    return int(value) if value is not None else None


def env_bool(name: str) -> bool | None:
    value = env_str(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {value}")


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    mappings = {
        "model_id": "model.id",
        "identifier": "model.identifier",
        "litellm_model_name": "model.litellm_model_name",
        "lm_studio_base_url": "model.lm_studio.base_url",
        "port": "model.lm_studio.port",
        "bind_address": "model.lm_studio.bind_address",
        "parallel": "model.lm_studio.parallel",
        "context_length": "model.lm_studio.context_length",
        "gpu": "model.lm_studio.gpu",
        "task_name": "benchmark.task_name",
        "limit": "benchmark.limit",
        "output_dir": "benchmark.output_dir",
        "cache_dir": "benchmark.cache_dir",
        "run_id": "benchmark.run_id",
        "run_id_template": "benchmark.run_id_template",
        "concurrent_requests": "benchmark.lighteval.concurrent_requests",
        "max_gen_toks": "benchmark.lighteval.max_gen_toks",
        "max_length": "benchmark.lighteval.max_length",
        "system_prompt": "benchmark.lighteval.system_prompt",
        "retry_parallel": "benchmark.throughput.retry_parallel",
        "retry_concurrent_requests": "benchmark.throughput.retry_concurrent_requests",
        "auto_retry_failed_run": "benchmark.throughput.auto_retry_failed_run",
    }

    for attr, path in mappings.items():
        value = getattr(args, attr, None)
        if value is not None:
            set_nested(overrides, path, value)

    return overrides


def resolve_runtime_settings(args: argparse.Namespace) -> dict[str, Any]:
    return resolve_settings(
        model=args.model,
        benchmark=args.benchmark,
        overrides=build_overrides(args),
    )


def ensure_lm_studio(base_url: str, api_key: str | None) -> None:
    url = f"{base_url.rstrip('/')}/models"
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=5):
            return
    except urllib.error.URLError as exc:
        raise SystemExit(f"LM Studio is not reachable at {url}: {exc}") from exc


def build_lighteval_model_config(settings: dict[str, Any], api_key: str | None) -> dict[str, Any]:
    model_cfg = settings["model"]
    benchmark_cfg = settings["benchmark"]
    lighteval_cfg = benchmark_cfg["lighteval"]

    generation_parameters = deepcopy(lighteval_cfg.get("generation_parameters", {}))
    generation_parameters.setdefault("max_new_tokens", int(lighteval_cfg["max_gen_toks"]))

    model_parameters = {
        "model_name": model_cfg["litellm_model_name"],
        "base_url": model_cfg["lm_studio"]["base_url"].rstrip("/"),
        "api_key": api_key or "dummy",
        "concurrent_requests": int(lighteval_cfg["concurrent_requests"]),
        "max_model_length": int(lighteval_cfg["max_length"]),
        "cache_dir": benchmark_cfg["cache_dir"],
        "generation_parameters": generation_parameters,
    }

    model_parameters = deep_merge(
        model_parameters,
        model_cfg.get("extra_lighteval_model_parameters", {}),
    )
    model_parameters = deep_merge(
        model_parameters,
        lighteval_cfg.get("model_parameters", {}),
    )

    system_prompt = lighteval_cfg.get("system_prompt", "")
    if system_prompt:
        model_parameters["system_prompt"] = system_prompt

    return {"model_parameters": model_parameters}


def write_model_config_file(settings: dict[str, Any], api_key: str | None) -> Path:
    output_dir = Path(settings["benchmark"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_lighteval_model_config(settings, api_key)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="lighteval-model-config.",
        suffix=".yaml",
        dir=output_dir,
        delete=False,
    ) as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
        return Path(handle.name)


def build_lighteval_command(
    settings: dict[str, Any],
    model_config_path: Path,
    extra_args: list[str],
) -> list[str]:
    benchmark_cfg = settings["benchmark"]
    lighteval_cfg = benchmark_cfg["lighteval"]

    cmd = [
        "lighteval",
        "endpoint",
        "litellm",
        str(model_config_path),
        benchmark_cfg["task_name"],
        "--output-dir",
        str(benchmark_cfg["output_dir"]),
    ]

    if benchmark_cfg.get("limit") is not None:
        cmd += ["--max-samples", str(benchmark_cfg["limit"])]

    cmd += [str(arg) for arg in lighteval_cfg.get("extra_args", [])]
    cmd += extra_args
    return cmd


def execute_lighteval(
    settings: dict[str, Any],
    extra_args: list[str],
    *,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    ensure_lm_studio(settings["model"]["lm_studio"]["base_url"], api_key)

    cache_dir = Path(settings["benchmark"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_config_path = write_model_config_file(settings, api_key)
    cmd = build_lighteval_command(settings, model_config_path, extra_args)

    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            check=False,
            stdout=stdout,
            stderr=stderr,
        )
        return completed.returncode
    finally:
        model_config_path.unlink(missing_ok=True)


def run_command(cmd: list[str], *, check: bool, quiet: bool = False) -> int:
    stdout = subprocess.DEVNULL if quiet else None
    stderr = subprocess.DEVNULL if quiet else None
    completed = subprocess.run(cmd, cwd=ROOT_DIR, check=False, stdout=stdout, stderr=stderr)
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, cmd)
    return completed.returncode


def stop_server(identifier: str) -> None:
    run_command(["lms", "server", "stop"], check=False, quiet=True)
    run_command(["lms", "unload", identifier], check=False, quiet=True)
    run_command(["lms", "unload", "-a"], check=False, quiet=True)


def start_server(
    *,
    model_id: str,
    identifier: str,
    port: int,
    bind_address: str,
    parallel: int,
    context_length: int,
    gpu: str,
) -> None:
    stop_server(identifier)
    run_command(["lms", "server", "start", "--port", str(port), "--bind", bind_address], check=True, quiet=True)
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
        check=True,
    )


def latest_results_json(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.rglob("results_*.json"))
    return candidates[-1] if candidates else None


def json_metric(path: Path, task_name: str) -> Any:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["results"][task_name]["extractive_match"]


def count_api_failures(stderr_path: Path) -> int:
    if not stderr_path.is_file():
        return 0
    return stderr_path.read_text(encoding="utf-8").count("API call failed after 8 attempts")


def count_context_errors(stderr_path: Path) -> int:
    if not stderr_path.is_file():
        return 0
    return stderr_path.read_text(encoding="utf-8").count("Context size has been exceeded")


def count_empty_text_responses(cache_dir: Path) -> int:
    import pyarrow.parquet as pq

    parquet_files = sorted(cache_dir.rglob("GENERATIVE.parquet"))
    if not parquet_files:
        return 0

    rows = pq.read_table(parquet_files[-1]).to_pylist()
    empty = 0
    for row in rows:
        sample = row["sample"]
        texts = sample.get("text") or []
        first = texts[0] if texts and texts[0] is not None else ""
        if first == "":
            empty += 1
    return empty


def render_run_id(settings: dict[str, Any]) -> str:
    benchmark_cfg = settings["benchmark"]
    model_cfg = settings["model"]
    throughput_cfg = benchmark_cfg["throughput"]
    template = benchmark_cfg.get("run_id_template", "{benchmark}")
    limit = benchmark_cfg.get("limit")

    try:
        return template.format(
            benchmark=benchmark_cfg["name"],
            model=model_cfg["name"],
            model_id=model_cfg["id"],
            parallel=throughput_cfg["parallel"],
            concurrent_requests=benchmark_cfg["lighteval"]["concurrent_requests"],
            context_length=model_cfg["lm_studio"]["context_length"],
            limit="all" if limit is None else limit,
        )
    except KeyError as exc:
        raise SystemExit(f"Unknown placeholder in run_id_template: {exc}") from exc


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [f"{key}={value}" for key, value in summary.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def command_run_eval(args: argparse.Namespace, extra_args: list[str]) -> int:
    settings = resolve_runtime_settings(args)
    return execute_lighteval(settings, extra_args)


def command_start_server(args: argparse.Namespace) -> int:
    settings = resolve_runtime_settings(args)
    model_cfg = settings["model"]
    lm_studio = model_cfg["lm_studio"]
    start_server(
        model_id=model_cfg["id"],
        identifier=model_cfg["identifier"],
        port=int(lm_studio["port"]),
        bind_address=lm_studio["bind_address"],
        parallel=int(lm_studio["parallel"]),
        context_length=int(lm_studio["context_length"]),
        gpu=lm_studio["gpu"],
    )
    return 0


def command_stop_server(args: argparse.Namespace) -> int:
    settings = resolve_runtime_settings(args)
    stop_server(settings["model"]["identifier"])
    return 0


def command_print_config(args: argparse.Namespace) -> int:
    settings = resolve_runtime_settings(args)
    if args.format == "json":
        print(json.dumps(settings, indent=2, sort_keys=True))
    else:
        print(yaml.safe_dump(settings, sort_keys=False).rstrip())
    return 0


def command_benchmark_throughput(args: argparse.Namespace, extra_args: list[str]) -> int:
    settings = resolve_runtime_settings(args)
    model_cfg = settings["model"]
    benchmark_cfg = settings["benchmark"]
    throughput_cfg = benchmark_cfg["throughput"]
    lm_studio = model_cfg["lm_studio"]

    run_id = benchmark_cfg.get("run_id") or render_run_id(settings)
    run_dir = Path(args.bench_dir or ROOT_DIR / "benchmarks") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    def run_attempt(attempt: int, attempt_parallel: int, attempt_concurrent: int) -> dict[str, Any]:
        attempt_dir = run_dir / f"attempt-{attempt}"
        output_dir = attempt_dir / "results"
        cache_dir = attempt_dir / "cache"
        stdout_log = attempt_dir / "stdout.log"
        stderr_log = attempt_dir / "stderr.log"
        summary_path = attempt_dir / "summary.txt"

        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        start_server(
            model_id=model_cfg["id"],
            identifier=model_cfg["identifier"],
            port=int(lm_studio["port"]),
            bind_address=lm_studio["bind_address"],
            parallel=attempt_parallel,
            context_length=int(lm_studio["context_length"]),
            gpu=lm_studio["gpu"],
        )

        shutil.rmtree(ROOT_DIR / ".litellm_cache", ignore_errors=True)

        attempt_settings = deepcopy(settings)
        attempt_settings["benchmark"]["output_dir"] = str(output_dir)
        attempt_settings["benchmark"]["cache_dir"] = str(cache_dir)
        attempt_settings["benchmark"]["lighteval"]["concurrent_requests"] = attempt_concurrent

        with stdout_log.open("w", encoding="utf-8") as stdout_handle, stderr_log.open(
            "w",
            encoding="utf-8",
        ) as stderr_handle:
            started_at = time.monotonic()
            exit_code = execute_lighteval(
                attempt_settings,
                extra_args,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
            real_seconds = time.monotonic() - started_at

        result_json = latest_results_json(output_dir)
        api_failures = count_api_failures(stderr_log)
        context_errors = count_context_errors(stderr_log)
        empty_text_responses = count_empty_text_responses(cache_dir)

        status = "ok"
        extractive_match = ""
        samples_per_second = ""

        if result_json is None:
            status = "failed"
        else:
            extractive_match = json_metric(result_json, benchmark_cfg["task_name"])
            sample_count = benchmark_cfg["limit"]
            if sample_count is not None and real_seconds > 0:
                samples_per_second = f"{float(sample_count) / real_seconds:.6f}"

        if exit_code != 0 or api_failures > 0 or context_errors > 0 or empty_text_responses > 0:
            status = "failed"

        summary = {
            "attempt": attempt,
            "model_id": model_cfg["id"],
            "identifier": model_cfg["identifier"],
            "model_config": settings["paths"]["model_config"],
            "benchmark_config": settings["paths"]["benchmark_config"],
            "task_name": benchmark_cfg["task_name"],
            "parallel": attempt_parallel,
            "concurrent_requests": attempt_concurrent,
            "context_length": lm_studio["context_length"],
            "gpu": lm_studio["gpu"],
            "limit": benchmark_cfg["limit"],
            "exit_code": exit_code,
            "status": status,
            "result_json": result_json or "",
            "real_seconds": f"{real_seconds:.6f}",
            "samples_per_second": samples_per_second,
            "extractive_match": extractive_match,
            "api_failures": api_failures,
            "context_errors": context_errors,
            "empty_text_responses": empty_text_responses,
            "stdout_log": stdout_log,
            "stderr_log": stderr_log,
        }
        write_summary(summary_path, summary)
        return summary

    first = run_attempt(
        attempt=1,
        attempt_parallel=int(throughput_cfg["parallel"]),
        attempt_concurrent=int(benchmark_cfg["lighteval"]["concurrent_requests"]),
    )

    final_summary = first
    if throughput_cfg.get("auto_retry_failed_run") and first["status"] != "ok":
        final_summary = run_attempt(
            attempt=2,
            attempt_parallel=int(throughput_cfg["retry_parallel"]),
            attempt_concurrent=int(throughput_cfg["retry_concurrent_requests"]),
        )

    write_summary(run_dir / "summary.txt", final_summary)
    for key, value in final_summary.items():
        print(f"{key}={value}")

    return 0 if final_summary["status"] == "ok" else 1


def add_shared_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=env_str("MODEL") or env_str("MODEL_CONFIG") or "default")
    parser.add_argument("--benchmark", default=env_str("BENCHMARK") or env_str("BENCHMARK_CONFIG") or "gsm8k")
    parser.add_argument("--model-id", default=env_str("MODEL_ID"))
    parser.add_argument("--identifier", default=env_str("IDENTIFIER"))
    parser.add_argument("--litellm-model-name", default=env_str("LITELLM_MODEL_NAME"))
    parser.add_argument("--lm-studio-base-url", default=env_str("LM_STUDIO_BASE_URL"))
    parser.add_argument("--task-name", default=env_str("TASK_NAME"))
    parser.add_argument("--limit", type=int, default=env_int("LIMIT"))
    parser.add_argument("--output-dir", default=env_str("OUTPUT_DIR"))
    parser.add_argument("--cache-dir", default=env_str("CACHE_DIR"))
    parser.add_argument("--system-prompt", default=env_str("SYSTEM_PROMPT"))
    parser.add_argument("--concurrent-requests", type=int, default=env_int("CONCURRENT_REQUESTS"))
    parser.add_argument("--max-gen-toks", type=int, default=env_int("MAX_GEN_TOKS"))
    parser.add_argument("--max-length", type=int, default=env_int("MAX_LENGTH"))
    parser.add_argument("--parallel", type=int, default=env_int("PARALLEL"))
    parser.add_argument("--context-length", type=int, default=env_int("CONTEXT_LENGTH"))
    parser.add_argument("--gpu", default=env_str("GPU"))
    parser.add_argument("--port", type=int, default=env_int("PORT"))
    parser.add_argument("--bind-address", default=env_str("BIND_ADDRESS"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Configurable helpers for LM Studio + lighteval workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_eval = subparsers.add_parser("run-eval", help="Run a single lighteval benchmark.")
    add_shared_config_args(run_eval)

    benchmark = subparsers.add_parser("benchmark-throughput", help="Run a throughput benchmark with retry support.")
    add_shared_config_args(benchmark)
    benchmark.add_argument("--run-id", default=env_str("RUN_ID"))
    benchmark.add_argument("--run-id-template", default=env_str("RUN_ID_TEMPLATE"))
    benchmark.add_argument("--bench-dir", default=env_str("BENCH_DIR"))
    benchmark.add_argument("--retry-parallel", type=int, default=env_int("RETRY_PARALLEL"))
    benchmark.add_argument(
        "--retry-concurrent-requests",
        type=int,
        default=env_int("RETRY_CONCURRENT_REQUESTS"),
    )
    benchmark.set_defaults(auto_retry_failed_run=env_bool("AUTO_RETRY_FAILED_RUN"))
    auto_retry_group = benchmark.add_mutually_exclusive_group()
    auto_retry_group.add_argument(
        "--auto-retry-failed-run",
        dest="auto_retry_failed_run",
        action="store_true",
    )
    auto_retry_group.add_argument(
        "--no-auto-retry-failed-run",
        dest="auto_retry_failed_run",
        action="store_false",
    )

    start_server_parser = subparsers.add_parser("start-server", help="Start LM Studio and load a model.")
    add_shared_config_args(start_server_parser)

    stop_server_parser = subparsers.add_parser("stop-server", help="Unload the current model and stop LM Studio.")
    add_shared_config_args(stop_server_parser)

    print_config = subparsers.add_parser("print-config", help="Resolve and print the active runtime config.")
    add_shared_config_args(print_config)
    print_config.add_argument("--format", choices=["yaml", "json"], default="yaml")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra_args = parser.parse_known_args(argv)

    if args.command == "run-eval":
        return command_run_eval(args, extra_args)
    if args.command == "benchmark-throughput":
        return command_benchmark_throughput(args, extra_args)
    if extra_args:
        parser.error(f"Unexpected extra arguments for {args.command}: {' '.join(extra_args)}")
    if args.command == "start-server":
        return command_start_server(args)
    if args.command == "stop-server":
        return command_stop_server(args)
    if args.command == "print-config":
        return command_print_config(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
