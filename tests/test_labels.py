import pandas as pd

from src.labels import exitcode_to_label, job_label_from_rows


def test_exitcode_to_label_mapping() -> None:
    assert exitcode_to_label("FAILED") == 1
    assert exitcode_to_label("TIMEOUT") == 1
    assert exitcode_to_label("COMPLETED") == 0
    assert exitcode_to_label("CANCELLED") is None


def test_job_label_from_rows_prefers_failure() -> None:
    series = pd.Series(["COMPLETED", "FAILED"])
    assert job_label_from_rows(series) == 1
