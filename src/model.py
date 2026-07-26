import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.features import encode_features, feature_columns
from src.paths import FEATURE_MEDIANS_PATH, MODEL_PATH


@dataclass
class TrainedModel:
    classifier: HistGradientBoostingClassifier | LogisticRegression
    feature_cols: list[str]
    encoded_columns: list[str]
    medians: dict[str, float]
    positive_rate: float


def _prepare_encoded(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    encoded = encode_features(frame, feature_cols)
    encoded = encoded.replace([np.inf, -np.inf], np.nan)
    medians = encoded.median(numeric_only=True)
    encoded = encoded.fillna(medians).fillna(0.0)
    unique_counts = encoded.nunique(dropna=False)
    keep_cols = unique_counts[unique_counts > 1].index.tolist()
    if not keep_cols:
        keep_cols = list(encoded.columns)
    return encoded[keep_cols], {key: float(medians.get(key, 0.0)) for key in keep_cols}


def train_classifier(labeled_jobs: pd.DataFrame) -> TrainedModel:
    labeled = labeled_jobs.dropna(subset=["label"]).copy()
    if labeled.empty:
        raise ValueError("No labeled jobs available for training.")

    feature_cols = feature_columns(labeled)
    encoded, medians = _prepare_encoded(labeled, feature_cols)
    y_train = labeled["label"].astype(int)

    if len(encoded.columns) == 0 or y_train.nunique() < 2:
        classifier: HistGradientBoostingClassifier | LogisticRegression = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )
        classifier.fit(encoded, y_train)
    else:
        try:
            classifier = HistGradientBoostingClassifier(
                max_depth=8,
                learning_rate=0.08,
                max_iter=200,
                random_state=42,
            )
            classifier.fit(encoded, y_train)
        except ValueError:
            classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
            classifier.fit(encoded, y_train)

    return TrainedModel(
        classifier=classifier,
        feature_cols=feature_cols,
        encoded_columns=list(encoded.columns),
        medians=medians,
        positive_rate=float(y_train.mean()),
    )


def save_model(model: TrainedModel, model_path: Path = MODEL_PATH, medians_path: Path = FEATURE_MEDIANS_PATH) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "classifier": model.classifier,
            "feature_cols": model.feature_cols,
            "encoded_columns": model.encoded_columns,
            "positive_rate": model.positive_rate,
        },
        model_path,
    )
    medians_path.write_text(json.dumps(model.medians, indent=2), encoding="utf-8")


def load_model(model_path: Path = MODEL_PATH, medians_path: Path = FEATURE_MEDIANS_PATH) -> TrainedModel:
    payload = joblib.load(model_path)
    medians = json.loads(medians_path.read_text(encoding="utf-8"))
    return TrainedModel(
        classifier=payload["classifier"],
        feature_cols=payload["feature_cols"],
        encoded_columns=payload["encoded_columns"],
        medians=medians,
        positive_rate=float(payload.get("positive_rate", 0.2)),
    )


def predict_proba(model: TrainedModel, jobs: pd.DataFrame) -> np.ndarray:
    if jobs.empty:
        return np.array([])

    working = jobs.copy()
    encoded = encode_features(working, model.feature_cols)
    encoded = encoded.replace([np.inf, -np.inf], np.nan)
    aligned = encoded.reindex(columns=model.encoded_columns, fill_value=0.0)
    for column, value in model.medians.items():
        if column in aligned.columns:
            aligned[column] = aligned[column].fillna(value)
    aligned = aligned.fillna(0.0)
    return model.classifier.predict_proba(aligned)[:, 1]
