#!/usr/bin/env python3
"""Parse a solver output and compute raw/strict-S24 capacitance and errors."""
import argparse
import math
from pathlib import Path
from common import dump
from readout.frozen import geometry, parse_output, s24_value


def evaluate(geom, output, reference=None, master=None):
    layout_master, expected = geometry(Path(geom))
    if master is not None and master != layout_master:
        raise ValueError('Configured master disagrees with layout capacitance task')
    master = layout_master
    matrix, walks, hops, elapsed, cpu = parse_output(Path(output))
    if not all(math.isfinite(v) for row in matrix.values() for v in row.values()):
        raise ValueError('Nonfinite capacitance matrix')
    values = matrix.get(master, {})
    raw = values.get(master)
    if raw is None or raw <= 0:
        raise ValueError('Positive master self capacitance missing')
    adjusted, action = s24_value(values, expected, master)
    row = dict(master=master, raw_capacitance_F=raw, s24_capacitance_F=adjusted,
        s24_action=action, s24_applied=action=='abs_coupling_sum',
        logical_missing=sorted(expected-set(values)), logical_extra=sorted(set(values)-expected),
        matrix_F=matrix, walks=walks, hops_per_walk=hops,
        total_steps_approx=round(walks*hops), elapsed_seconds=elapsed, cpu_seconds=cpu)
    if reference is not None:
        if not math.isfinite(reference) or reference <= 0:
            raise ValueError('Reference capacitance must be finite and positive')
        row.update(reference_F=reference, raw_self_error_percent=100*abs(raw/reference-1),
                   s24_self_error_percent=100*abs(adjusted/reference-1))
    return row


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--geometry', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--reference', type=float)
    ap.add_argument('--json', type=Path, required=True)
    a = ap.parse_args()
    result = evaluate(a.geometry, a.output, a.reference)
    dump(a.json, result)
    print(a.json.read_text())
