import pytest

from src.data_loader import load_job_table_from_training, read_training_parquet
from src.data_loader import iter_training_parquet_paths


@pytest.mark.usefixtures("require_bundles")
def test_read_one_training_parquet_file() -> None:
    month, inner_name = next(iter_training_parquet_paths(1))
    frame = read_training_parquet(month, inner_name)
    assert "jid" in frame.columns
    assert len(frame) > 0


@pytest.mark.usefixtures("require_bundles")
def test_load_limited_training_jobs_have_labels() -> None:
    jobs = load_job_table_from_training(limit_files=2)
    assert not jobs.empty
    assert "label" in jobs.columns
    assert jobs["label"].notna().any()
