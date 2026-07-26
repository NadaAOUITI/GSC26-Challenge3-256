from pathlib import Path

import pandas as pd

from src.data_loader import load_job_table_from_judge
from src.model import TrainedModel, load_model, predict_proba
from src.paths import SAMPLE_SUBMISSION_CSV


def build_submission(
    model: TrainedModel,
    output_path: Path,
    judge_limit_files: int | None = None,
    submission_template: Path = SAMPLE_SUBMISSION_CSV,
    judge_jobs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    template = pd.read_csv(submission_template)[["row_id"]]
    if judge_jobs is None:
        judge_jobs = load_job_table_from_judge(judge_limit_files)
    if not judge_jobs.empty:
        probabilities = predict_proba(model, judge_jobs)
        judge_jobs = judge_jobs.assign(failure_probability=probabilities)
        merged = template.merge(judge_jobs[["row_id", "failure_probability"]], on="row_id", how="left")
    else:
        merged = template.copy()
        merged["failure_probability"] = pd.NA

    fallback = float(model.positive_rate) if model.positive_rate > 0 else 0.2
    merged["failure_probability"] = merged["failure_probability"].fillna(fallback).clip(0.0, 1.0)
    submission = merged[["row_id", "failure_probability"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def predict_to_file(
    output_path: Path,
    judge_limit_files: int | None = None,
) -> pd.DataFrame:
    model = load_model()
    return build_submission(model, output_path, judge_limit_files=judge_limit_files)
