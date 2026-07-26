import io
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from src.features import aggregate_jobs
from src.labels import job_label_from_rows
from src.paths import JUDGE_BUNDLE_ZIP, TRAINING_BUNDLE_ZIP


def _month_from_archive(name: str) -> str:
    base = Path(name).name.replace(".zip", "")
    return base


def infer_row_id(jid: str, month: str) -> str:
    if jid.endswith("_S"):
        return f"S_{jid}"
    if month.startswith("2023") or month >= "2022-07":
        return f"Anvil_{jid}"
    if month >= "2017-07":
        return f"S_{jid}"
    if month >= "2017-04":
        return f"C_{jid}"
    if month >= "2015-05":
        return f"C_{jid}"
    return f"S_{jid}"


def _iter_nested_parquet_paths(outer_zip: Path, prefix: str) -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    with zipfile.ZipFile(outer_zip) as outer:
        for outer_name in sorted(outer.namelist()):
            if not outer_name.startswith(prefix) or not outer_name.endswith(".zip"):
                continue
            month = _month_from_archive(outer_name)
            with zipfile.ZipFile(io.BytesIO(outer.read(outer_name))) as inner:
                for inner_name in sorted(inner.namelist()):
                    if inner_name.endswith(".parquet"):
                        paths.append((month, inner_name))
    return paths


def iter_training_parquet_paths(limit: int | None = None) -> Iterator[tuple[str, str]]:
    count = 0
    for month, inner_name in _iter_nested_parquet_paths(TRAINING_BUNDLE_ZIP, "training_data/"):
        yield month, inner_name
        count += 1
        if limit is not None and count >= limit:
            break


def iter_judge_parquet_paths(limit: int | None = None) -> Iterator[tuple[str, str]]:
    count = 0
    for month, inner_name in _iter_nested_parquet_paths(JUDGE_BUNDLE_ZIP, "unlabeled_judge_data/"):
        yield month, inner_name
        count += 1
        if limit is not None and count >= limit:
            break


def read_training_parquet(month: str, inner_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(TRAINING_BUNDLE_ZIP) as outer:
        outer_name = f"training_data/{month}.zip"
        with zipfile.ZipFile(io.BytesIO(outer.read(outer_name))) as inner:
            return pd.read_parquet(io.BytesIO(inner.read(inner_name)))


def read_judge_parquet(month: str, inner_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(JUDGE_BUNDLE_ZIP) as outer:
        outer_name = f"unlabeled_judge_data/{month}.zip"
        with zipfile.ZipFile(io.BytesIO(outer.read(outer_name))) as inner:
            return pd.read_parquet(io.BytesIO(inner.read(inner_name)))


def load_job_table_from_training(limit_files: int | None = None) -> pd.DataFrame:
    job_frames: list[pd.DataFrame] = []
    for month, inner_name in iter_training_parquet_paths(limit_files):
        frame = read_training_parquet(month, inner_name)
        if "jid" not in frame.columns:
            continue
        row_ids = frame["jid"].astype(str).map(lambda jid: infer_row_id(jid, month))
        aggregated = aggregate_jobs(frame, row_ids)
        if aggregated.empty:
            continue
        if "exitcode" in frame.columns:
            label_map = (
                frame.assign(row_id=row_ids.astype(str))
                .groupby("row_id", sort=False)["exitcode"]
                .apply(job_label_from_rows)
                .reset_index(name="label")
            )
            aggregated = aggregated.merge(label_map, on="row_id", how="left")
        job_frames.append(aggregated)

    if not job_frames:
        return pd.DataFrame()
    combined = pd.concat(job_frames, ignore_index=True)
    return combined.groupby("row_id", sort=False).last().reset_index(drop=True)


def load_job_table_from_judge(limit_files: int | None = None) -> pd.DataFrame:
    job_frames: list[pd.DataFrame] = []
    for month, inner_name in iter_judge_parquet_paths(limit_files):
        frame = read_judge_parquet(month, inner_name)
        if "jid" not in frame.columns:
            continue
        row_ids = frame["jid"].astype(str).map(lambda jid: infer_row_id(jid, month))
        aggregated = aggregate_jobs(frame, row_ids)
        if not aggregated.empty:
            job_frames.append(aggregated)

    if not job_frames:
        return pd.DataFrame()
    combined = pd.concat(job_frames, ignore_index=True)
    return combined.groupby("row_id", sort=False).last().reset_index(drop=True)
