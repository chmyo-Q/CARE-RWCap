"""Conditional Endpoint Re-estimation (CER) on a valid logical-conductor row."""
import math
from .parser import geometry, logical, parse_output


def cer_value(values: dict[str, float], expected: set[str], master: str) -> tuple[float, str]:
    observed = set(values)
    missing, extra = expected - observed, observed - expected
    off = [value for name, value in values.items() if name != master]
    if not missing and not extra and off and all(value <= 0 for value in off):
        estimate = math.fsum(abs(value) for value in off)
        if estimate > 0:
            return estimate, "abs_coupling_sum"
    if missing or extra:
        return values[master], "identity_incomplete_or_extra_columns"
    return values[master], "identity_sign_or_degenerate"

# Compatibility with archived readers.
s24_value = cer_value
