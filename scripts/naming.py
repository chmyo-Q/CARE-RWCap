"""Paper names at the user interface; stable IDs in archived run records."""
import re

ARM_LABELS = {
    'p0': 'DeepRWCap',
    'bpr': 'DeepRWCap + CAPR',
    'full': 'CARE-RWCap',
}
CLI_ARMS = '{deeprwcap,capr,care-rwcap}'
ALIASES = {'deeprwcap': 'p0', 'capr': 'bpr', 'care-rwcap': 'full', **{k: k for k in ARM_LABELS}}


def normalize_arm(value):
    if value not in ALIASES:
        raise ValueError('Choose deeprwcap, capr, or care-rwcap')
    return ALIASES[value]


def comparison_label(value):
    # Render after statistics are computed: archived keys also select RNG streams.
    return re.sub(r'\b(p0|bpr|full)\b', lambda m: ARM_LABELS[m.group()], value)
