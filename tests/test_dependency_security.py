from importlib.metadata import version

import litellm
import pytest
from litellm.types.caching import LiteLLMCacheType

from llmevals.config import resolve_settings
from llmevals.runtime import build_litellm_model_parameters


def _numeric_version(distribution: str) -> tuple[int, ...]:
    release = version(distribution).split("+", maxsplit=1)[0]
    return tuple(int(part) for part in release.split(".") if part.isdigit())


@pytest.mark.parametrize(
    ("distribution", "minimum"),
    [
        ("aiohttp", (3, 14, 3)),
        ("gitpython", (3, 1, 58)),
        ("idna", (3, 15)),
        ("litellm", (1, 84, 0)),
        ("nltk", (3, 9, 3)),
        ("python-dotenv", (1, 2, 2)),
        ("setuptools", (83, 0, 0)),
        ("soupsieve", (2, 8, 4)),
        ("starlette", (1, 3, 1)),
        ("torch", (2, 13, 0)),
        ("urllib3", (2, 7, 0)),
    ],
)
def test_audited_dependency_floors(distribution: str, minimum: tuple[int, ...]) -> None:
    assert _numeric_version(distribution) >= minimum


def test_no_fix_diskcache_pickle_backend_is_replaced_with_memory_cache() -> None:
    settings = resolve_settings(model="default", benchmark="gsm8k")
    parameters = build_litellm_model_parameters(settings, api_key="test-key")

    assert parameters["model_name"].startswith("openai/")
    assert litellm.cache is not None
    assert litellm.cache.type == LiteLLMCacheType.LOCAL
    assert type(litellm.cache.cache).__name__ == "InMemoryCache"
    assert not hasattr(litellm.cache.cache, "disk_cache")
