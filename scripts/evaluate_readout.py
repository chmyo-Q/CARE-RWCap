#!/usr/bin/env python3
"""Parse a solver output and compute raw/CER capacitance and errors."""
import argparse
import math
from pathlib import Path
from common import dump
from readout.parser import geometry, logical, parse_output
from readout.cer import cer_value


def coupling_error(values, master, reference_source):
    """Reference-covered off-diagonal magnitude NL1, in percent (manuscript IV-D).

    DSPF/SPICE loading follows DeepRWCap's public run_script.py conventions,
    including ground naming and averaging a repeated reciprocal entry.
    """
    path = Path(reference_source)
    if path.suffix.lower() not in ('.dspf', '.spice'):
        raise ValueError('Coupling reference must be a DSPF or SPICE file')
    lines = path.read_text(encoding='utf-8').splitlines()

    def capacitance(text):
        scale = 1e-12 if text.endswith('pF') else 1e-15 if text.endswith('fF') else 1.
        value = float(text[:-2] if scale != 1. else text) * scale
        if not math.isfinite(value):
            raise ValueError('Nonfinite reference capacitance')
        return value

    matrix = {}
    for line in lines:
        tokens = line.split()
        if line.startswith('*|NET ') or (line.startswith('** ') and '(total)' in line):
            matrix[tokens[1]] = {tokens[1]: capacitance(tokens[2])}
    for line in lines:
        if not (line.startswith('Cp') if path.suffix.lower() == '.spice' else line.startswith('C')):
            continue
        tokens = line.split()
        a, b = tokens[1], 'GROUND' if tokens[2] == '0' else tokens[2]
        value = capacitance(tokens[3])
        if a not in matrix:
            raise ValueError('Reference coupling has no declared conductor: ' + a)
        if b in matrix[a]:
            value = (matrix[a][b] + value) / 2
        matrix[a][b] = value
        if b in matrix:
            matrix[b][a] = value
    matches = [row for name, row in matrix.items() if logical(name) == master]
    if len(matches) != 1:
        raise ValueError('Reference master missing or ambiguous')
    reference = {logical(name): value for name, value in matches[0].items()}
    if len(reference) != len(matches[0]):
        raise ValueError('Ambiguous logical reference columns')
    shared = (set(values) & set(reference)) - {master}
    denominator = math.fsum(abs(reference[name]) for name in shared)
    if not shared or denominator <= 0:
        raise ValueError('No positive reference mass on shared coupling entries')
    if not all(math.isfinite(values[name]) for name in shared):
        raise ValueError('Nonfinite output coupling')
    numerator = math.fsum(abs(abs(values[name]) - abs(reference[name])) for name in shared)
    return {'normalized_l1_percent': 100 * numerator / denominator,
            'shared_offdiagonal_entries': len(shared),
            'reference_mass_F': denominator,
            'scope': 'Off-diagonal entries present in both reference and output; magnitudes, self excluded.'}


def evaluate(geom, output, reference=None, master=None, coupling_reference=None):
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
    adjusted, action = cer_value(values, expected, master)
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
    # Preserve historical JSON fields while exposing the paper terminology.
    row.update({key.replace('s24_', 'cer_', 1): value
                for key, value in list(row.items()) if key.startswith('s24_')})
    if coupling_reference is not None:
        row['coupling_error'] = coupling_error(values, master, coupling_reference)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--geometry', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--reference', type=float)
    ap.add_argument('--coupling-reference', type=Path,
                    help='Optional DSPF/SPICE reference for coupling-magnitude NL1; does not affect CER')
    ap.add_argument('--json', type=Path, required=True)
    a = ap.parse_args()
    result = evaluate(a.geometry, a.output, a.reference, coupling_reference=a.coupling_reference)
    dump(a.json, result)
    import json
    print(json.dumps({k:v for k,v in result.items() if not k.startswith('s24_')}, indent=2))


if __name__ == '__main__':
    main()
