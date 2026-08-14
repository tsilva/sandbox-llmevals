from __future__ import annotations

import litellm
from litellm.types.caching import LiteLLMCacheType


def ensure_safe_litellm_cache() -> None:
    """Replace Lighteval's pickle-backed disk cache with process-local memory."""
    if litellm.cache is not None and litellm.cache.type == LiteLLMCacheType.LOCAL:
        return

    litellm.disable_cache()
    litellm.enable_cache(type=LiteLLMCacheType.LOCAL)
