import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from src.model import TrainedModel, predict_proba
from src.predict import build_submission


def test_submission_schema(tmp_path) -> None:
    template = tmp_path / "sample.csv"
    template.write_text("row_id,failure_probability\nAnvil_JOB1,0.5\nAnvil_JOB2,0.5\n", encoding="utf-8")
    jobs = pd.DataFrame(
        {
            "row_id": ["Anvil_JOB1"],
            "timelimit": [100.0],
            "nhosts": [1.0],
            "ncores": [8.0],
            "walltime_seconds": [50.0],
            "measurement_count": [1.0],
            "queue": ["batch"],
            "account": ["a"],
        }
    )
    model = TrainedModel(
        classifier=HistGradientBoostingClassifier(max_iter=10, random_state=0),
        feature_cols=["timelimit", "nhosts", "ncores", "walltime_seconds", "measurement_count", "queue", "account"],
        encoded_columns=["timelimit", "nhosts", "ncores", "walltime_seconds", "measurement_count", "queue_batch", "account_a"],
        medians={"timelimit": 100.0},
        positive_rate=0.2,
    )
    model.classifier.fit(
        pd.DataFrame(
            {
                "timelimit": [100, 200],
                "nhosts": [1, 2],
                "ncores": [8, 16],
                "walltime_seconds": [50, 60],
                "measurement_count": [1, 2],
                "queue_batch": [1, 0],
                "account_a": [1, 1],
            }
        ),
        [0, 1],
    )
    output = tmp_path / "submission.csv"
    submission = build_submission(
        model,
        output,
        submission_template=template,
        judge_jobs=jobs,
    )
    assert list(submission.columns) == ["row_id", "failure_probability"]
    assert len(submission) == 2
