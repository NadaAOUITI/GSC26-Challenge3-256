import argparse
import sys
from pathlib import Path

from src.data_loader import (
    iter_judge_parquet_paths,
    iter_training_parquet_paths,
    load_job_table_from_judge,
    load_job_table_from_training,
    read_training_parquet,
)
from src.evaluate import evaluate_jobs
from src.model import save_model, train_classifier
from src.paths import OUTPUT_DIR, TRAINING_BUNDLE_ZIP, JUDGE_BUNDLE_ZIP
from src.predict import predict_to_file


def cmd_inspect(args: argparse.Namespace) -> None:
    print(f"Training bundle: {TRAINING_BUNDLE_ZIP} ({TRAINING_BUNDLE_ZIP.exists()})")
    print(f"Judge bundle: {JUDGE_BUNDLE_ZIP} ({JUDGE_BUNDLE_ZIP.exists()})")
    for index, (month, inner_name) in enumerate(iter_training_parquet_paths(args.limit)):
        frame = read_training_parquet(month, inner_name)
        print(f"[train {index + 1}] {month}/{inner_name} rows={len(frame)} cols={list(frame.columns)}")


def cmd_train(args: argparse.Namespace) -> None:
    jobs = load_job_table_from_training(args.limit)
    print(f"Loaded {len(jobs)} labeled jobs from training bundle")
    model = train_classifier(jobs)
    save_model(model)
    pos_rate = jobs["label"].mean()
    print(f"Saved model to {OUTPUT_DIR / 'model.joblib'} (train positive rate={pos_rate:.3f})")


def cmd_eval(args: argparse.Namespace) -> None:
    jobs = load_job_table_from_training(args.limit)
    metrics, model = evaluate_jobs(jobs)
    save_model(model)
    print(f"Hold-out samples={metrics.samples} positives={metrics.positives} AUC={metrics.auc:.4f}")


def cmd_predict(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    submission = predict_to_file(output_path, judge_limit_files=args.judge_limit)
    print(f"Wrote {len(submission)} rows to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FRESCO job failure prediction toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Print parquet schema samples")
    inspect_parser.add_argument("--limit", type=int, default=3, help="Number of parquet files to inspect")
    inspect_parser.set_defaults(func=cmd_inspect)

    train_parser = subparsers.add_parser("train", help="Train baseline model")
    train_parser.add_argument("--limit", type=int, default=200, help="Max training parquet files")
    train_parser.set_defaults(func=cmd_train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate with hold-out split")
    eval_parser.add_argument("--limit", type=int, default=200, help="Max training parquet files")
    eval_parser.set_defaults(func=cmd_eval)

    predict_parser = subparsers.add_parser("predict", help="Write Kaggle submission CSV")
    predict_parser.add_argument("--output", default=str(OUTPUT_DIR / "submission.csv"))
    predict_parser.add_argument(
        "--judge-limit",
        type=int,
        default=None,
        help="Limit judge parquet files (default: all)",
    )
    predict_parser.set_defaults(func=cmd_predict)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
