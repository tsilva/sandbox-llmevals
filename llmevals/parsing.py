from __future__ import annotations

import re


def extract_final_number(text: str) -> str | None:
    hash_match = re.findall(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    if hash_match:
        return hash_match[-1]

    number_match = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if number_match:
        return number_match[-1]

    return None

