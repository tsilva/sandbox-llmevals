from __future__ import annotations

import json
import subprocess

from llmevals import runtime
from llmevals.runtime import list_available_benchmark_configs, list_available_model_configs, list_discoverable_local_models


def test_lists_model_configs():
    names = {item["name"] for item in list_available_model_configs()}
    assert "default" in names


def test_lists_benchmark_configs():
    names = {item["name"] for item in list_available_benchmark_configs()}
    assert "gsm8k" in names
    assert "gsm8k_reasoning" in names


def test_lists_discoverable_local_models_from_lm_studio_cli(monkeypatch):
    payload = [
        {
            "modelKey": "qwen/qwen3.5-9b",
            "displayName": "Qwen3.5 9B",
            "publisher": "qwen",
            "paramsString": "9B",
            "path": "qwen/qwen3.5-9b",
            "maxContextLength": 262144,
        }
    ]

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)

    models = list_discoverable_local_models()

    assert len(models) == 1
    assert models[0]["value"] == "qwen/qwen3.5-9b"
    assert models[0]["source"] == "lmstudio_cli"
