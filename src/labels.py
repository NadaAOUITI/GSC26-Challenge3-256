import pandas as pd

POSITIVE_STATES = {"FAILED", "TIMEOUT", "NODE_FAIL"}
NEGATIVE_STATES = {"COMPLETED"}


def normalize_exitcode(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().upper()


def exitcode_to_label(exitcode) -> int | None:
    state = normalize_exitcode(exitcode)
    if state in POSITIVE_STATES:
        return 1
    if state in NEGATIVE_STATES:
        return 0
    return None


def job_label_from_rows(exitcodes: pd.Series) -> int | None:
    labels = [exitcode_to_label(value) for value in exitcodes.dropna().unique()]
    labels = [label for label in labels if label is not None]
    if not labels:
        return None
    return max(labels)
