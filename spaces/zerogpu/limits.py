"""Shared, explicit deployment limits for the form, controller and GPU worker."""

from __future__ import annotations

import os

DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_MAX_SEQ_LENGTH = 1024
DEFAULT_MAX_RELATIVE_NORM = 0.1
CORE_MAX_NEW_TOKENS = 32768
CORE_MAX_SEQ_LENGTH = 131072


def space_limits():
    """Operator overrides opt into larger requests, not guaranteed GPU capacity."""
    result = {}
    for field, default, minimum, maximum in (
        ("max_new_tokens", DEFAULT_MAX_NEW_TOKENS, 1, CORE_MAX_NEW_TOKENS),
        ("max_seq_length", DEFAULT_MAX_SEQ_LENGTH, 32, CORE_MAX_SEQ_LENGTH),
    ):
        name = "VESPASIAN_SPACE_" + field.upper()
        raw = os.environ.get(name, str(default))
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a whole number between {minimum} and {maximum}") from None
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")
        result[field] = value
    return result
