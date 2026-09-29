#!/usr/bin/env python3
"""Parse one public-case run and apply the frozen strict logical-net S24 rule."""
from __future__ import annotations

import math
import re
from pathlib import Path


def logical(name: str) -> str:
    match = re.match(r"^\d+__(.+)$", name)
    return match.group(1) if match else name


def geometry(path: Path) -> tuple[str, set[str]]:
    text = path.read_text(encoding="utf-8")
    names = []
    for block in re.findall(r"<conductor>(.*?)</conductor>", text, re.S):
        match = re.search(r"^\s*name\s+(\S+)", block, re.M)
        if not match:
            raise ValueError("unnamed conductor")
        names.append(logical(match.group(1)))
    tasks = " ".join(re.findall(r"<capacitance>(.*?)</capacitance>", text, re.S)).split()
    if len(tasks) != 1 or not names:
        raise ValueError("expected one capacitance task and nonempty conductor set")
    return tasks[0], set(names)


def parse_output(path: Path) -> tuple[dict[str, dict[str, float]], int, float, float, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    matrix: dict[str, dict[str, float]] = {}
    master = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("Master "):
            master = line.split()[1]
            if master in matrix:
                raise ValueError(f"duplicate master block: {master}")
            matrix[master] = {}
        elif line.startswith("Capacitance on "):
            match = re.match(r"Capacitance on\s+(\S+)\s*=\s*([-+0-9.eE]+)", line)
            if master is None or match is None:
                raise ValueError("bad capacitance row")
            name, value = match.groups()
            net = logical(name)
            if net in matrix[master]:
                raise ValueError(f"duplicate logical column {net!r} for master {master!r}; "
                                 "expected one already-aggregated output per logical net")
            matrix[master][net] = float(value)
    walks = re.search(r"RWCap has run\s+(\d+)\s+walks", text)
    hops = re.search(r"\(([0-9.]+)\s+hops/walk\)", text)
    timing = re.search(r"Elapsed time:\s*([0-9.]+)sec,\s*CPU time:\s*([0-9.]+)sec", text)
    if not matrix or not walks or not hops or not timing:
        raise ValueError("missing matrix, walks, hops, or timing")
    return matrix, int(walks.group(1)), float(hops.group(1)), float(timing.group(1)), float(timing.group(2))


def s24_value(values: dict[str, float], expected: set[str], master: str) -> tuple[float, str]:
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
