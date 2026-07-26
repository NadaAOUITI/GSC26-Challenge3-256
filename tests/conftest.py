import pytest


@pytest.fixture
def require_bundles() -> None:
    from src.paths import JUDGE_BUNDLE_ZIP, TRAINING_BUNDLE_ZIP

    if not TRAINING_BUNDLE_ZIP.exists() or not JUDGE_BUNDLE_ZIP.exists():
        pytest.skip("FRESCO ZIP bundles not present at project root")
