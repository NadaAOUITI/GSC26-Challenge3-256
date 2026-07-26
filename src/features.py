import numpy as np
import pandas as pd

ACCOUNTING_COLS = [
    "timelimit",
    "nhosts",
    "ncores",
]
METRIC_COLS = [
    "value_cpuuser",
    "value_gpu",
    "value_memused",
    "value_memused_minus_diskcache",
    "value_nfs",
    "value_block",
]
CATEGORICAL_COLS = ["queue", "account"]


def _duration_seconds(start, end) -> float:
    start_ts = pd.to_datetime(start, errors="coerce", utc=True)
    end_ts = pd.to_datetime(end, errors="coerce", utc=True)
    if pd.isna(start_ts) or pd.isna(end_ts):
        return np.nan
    return max((end_ts - start_ts).total_seconds(), 0.0)


def aggregate_jobs(frame: pd.DataFrame, row_id_series: pd.Series) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    working = frame.copy()
    working["row_id"] = pd.Series(row_id_series, index=working.index).astype(str)
    working = working.dropna(subset=["row_id"])
    if working.empty:
        return pd.DataFrame()

    if "start_time" in working.columns and "end_time" in working.columns:
        working["walltime_seconds"] = [
            _duration_seconds(start, end)
            for start, end in zip(working["start_time"], working["end_time"], strict=False)
        ]
    else:
        working["walltime_seconds"] = np.nan

    grouped_parts: list[pd.DataFrame] = []
    for row_id, group in working.groupby("row_id", sort=False):
        features: dict[str, float | str] = {"row_id": row_id, "measurement_count": float(len(group))}

        for column in ACCOUNTING_COLS:
            if column in group.columns:
                series = pd.to_numeric(group[column], errors="coerce")
                features[column] = float(series.iloc[0]) if series.notna().any() else np.nan

        for column in METRIC_COLS:
            if column in group.columns:
                series = pd.to_numeric(group[column], errors="coerce")
                features[f"{column}_mean"] = float(series.mean()) if series.notna().any() else np.nan
                features[f"{column}_max"] = float(series.max()) if series.notna().any() else np.nan

        if "walltime_seconds" in group.columns:
            walltime = pd.to_numeric(group["walltime_seconds"], errors="coerce")
            features["walltime_seconds"] = float(walltime.max()) if walltime.notna().any() else np.nan

        for column in CATEGORICAL_COLS:
            if column in group.columns:
                features[column] = str(group[column].dropna().iloc[0]) if group[column].notna().any() else "missing"

        grouped_parts.append(pd.DataFrame([features]))

    if not grouped_parts:
        return pd.DataFrame()
    return pd.concat(grouped_parts, ignore_index=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in ACCOUNTING_COLS + ["walltime_seconds", "measurement_count"]:
        if column in frame.columns:
            columns.append(column)
    for metric in METRIC_COLS:
        for suffix in ("_mean", "_max"):
            name = f"{metric}{suffix}"
            if name in frame.columns:
                columns.append(name)
    for column in CATEGORICAL_COLS:
        if column in frame.columns:
            columns.append(column)
    return columns


def encode_features(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    encoded = pd.get_dummies(frame[feature_cols], columns=[c for c in CATEGORICAL_COLS if c in feature_cols])
    return encoded.astype(float)
