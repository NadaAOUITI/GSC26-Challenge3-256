# ADR 0001: Project Architecture and Delivery Strategy

**Status:** Accepted  
**Date:** 2026-07-26  
**Challenge:** GSC 2026 — Challenge 03 (FRESCO early job failure prediction)

## Context

Predict `failure_probability` per HPC job (`row_id` = `{cluster}_{jid}`) from FRESCO Parquet telemetry. Training labels come from `exitcode` in participant data; judge data drops labels for Kaggle scoring.

Constraints:

- Data ships as nested ZIP bundles (~9 GB); full extraction is optional
- One prediction per job after aggregating multiple hourly Parquet rows
- Phase I requires README, ~100 LOC, and basic data parsing

## Decision

Adopt a **pandas + sklearn baseline pipeline** reading Parquet directly from nested ZIP archives.

```text
ZIP bundles → data_loader → job features → HistGradientBoosting → submission.csv
```

| Module | Role |
|--------|------|
| `data_loader` | Nested ZIP iteration, `row_id` inference, job aggregation |
| `features` | Numeric + categorical encoding |
| `labels` | Map exitcode → binary failure label |
| `model` | Train/save/load classifier |
| `predict` | Merge judge features with submission template |
| `main` | `inspect`, `train`, `eval`, `predict` CLI |

## Alternatives rejected

- **Full 9 GB extract to disk** — unnecessary for MVP; nested ZIP reads suffice
- **Deep learning on raw timeseries** — too heavy for Phase I deadline
- **DuckDB-only pipeline** — added dependency without clear win for Feature 1

## Success criteria

- [ ] 100+ LOC in `src/`
- [ ] `inspect` prints real Parquet columns
- [ ] `train` + `predict` produce valid submission CSV
- [ ] Kaggle score improves over constant 0.5 baseline (0.24050 public score)
