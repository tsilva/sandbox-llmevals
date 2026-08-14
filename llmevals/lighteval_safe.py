from __future__ import annotations

import runpy

from .lighteval_cache import ensure_safe_litellm_cache


def main() -> None:
    ensure_safe_litellm_cache()
    runpy.run_module("lighteval", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
