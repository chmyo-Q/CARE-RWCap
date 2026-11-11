"""Method identifiers and readable table labels."""

ARM_LABELS = {"deeprwcap": "DeepRWCap", "capr": "DeepRWCap + CAPR", "care-rwcap": "CARE-RWCap"}
CLI_ARMS = "{deeprwcap,capr,care-rwcap}"


def normalize_arm(value):
    if value not in ARM_LABELS:
        raise ValueError("Choose deeprwcap, capr, or care-rwcap")
    return value


def comparison_label(value):
    import re

    return re.sub(
        r"(?<![\w-])(care-rwcap|deeprwcap|capr)(?![\w-])",
        lambda match: ARM_LABELS[match.group()],
        value,
    )
