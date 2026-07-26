from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.model import TrainedModel, predict_proba, train_classifier


@dataclass
class EvalMetrics:
    samples: int
    positives: int
    auc: float


def evaluate_jobs(labeled_jobs: pd.DataFrame, test_size: float = 0.2) -> tuple[EvalMetrics, TrainedModel]:
    labeled = labeled_jobs.dropna(subset=["label"]).copy()
    if labeled.empty:
        raise ValueError("No labeled jobs to evaluate.")

    train_frame, test_frame = train_test_split(
        labeled,
        test_size=test_size,
        random_state=42,
        stratify=labeled["label"],
    )
    model = train_classifier(train_frame)
    probabilities = predict_proba(model, test_frame)
    y_true = test_frame["label"].astype(int)
    auc = roc_auc_score(y_true, probabilities) if y_true.nunique() > 1 else 0.0
    metrics = EvalMetrics(
        samples=len(test_frame),
        positives=int(y_true.sum()),
        auc=float(auc),
    )
    return metrics, model
