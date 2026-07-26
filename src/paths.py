from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "output"

TRAINING_BUNDLE_ZIP = ROOT / "training_data_bundle.zip"
JUDGE_BUNDLE_ZIP = ROOT / "unlabeled_judge_data_bundle.zip"

SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"
DATA_CHUNK_INFO = DATA_DIR / "data-chunk-info.txt"

MODEL_PATH = OUTPUT_DIR / "model.joblib"
FEATURE_MEDIANS_PATH = OUTPUT_DIR / "feature_medians.json"
