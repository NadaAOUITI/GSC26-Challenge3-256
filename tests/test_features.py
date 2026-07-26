import pandas as pd

from src.data_loader import infer_row_id
from src.features import aggregate_jobs


def test_infer_row_id_patterns() -> None:
    assert infer_row_id("JOB8357302_S", "2017-08") == "S_JOB8357302_S"
    assert infer_row_id("JOB2657834", "2017-04") == "C_JOB2657834"
    assert infer_row_id("JOB1176183", "2023-04") == "Anvil_JOB1176183"


def test_aggregate_jobs_builds_row_level_features() -> None:
    frame = pd.DataFrame(
        {
            "jid": ["JOB1", "JOB1"],
            "timelimit": [3600, 3600],
            "nhosts": [1, 1],
            "ncores": [32, 32],
            "value_cpuuser": [1.0, 3.0],
            "value_gpu": [0.0, 0.0],
            "value_memused": [10.0, 20.0],
            "value_memused_minus_diskcache": [8.0, 16.0],
            "value_nfs": [0.1, 0.2],
            "value_block": [0.0, 0.0],
            "queue": ["gpu", "gpu"],
            "account": ["acct", "acct"],
            "start_time": ["2023-04-01 00:00:00+00:00", "2023-04-01 00:05:00+00:00"],
            "end_time": ["2023-04-01 01:00:00+00:00", "2023-04-01 01:05:00+00:00"],
        }
    )
    row_ids = pd.Series(["Anvil_JOB1", "Anvil_JOB1"])
    aggregated = aggregate_jobs(frame, row_ids)
    assert len(aggregated) == 1
    assert aggregated.loc[0, "row_id"] == "Anvil_JOB1"
    assert aggregated.loc[0, "measurement_count"] == 2.0
