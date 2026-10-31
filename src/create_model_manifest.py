
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.feature_contract import A0_FEATURES, assert_deployable_feature_set
from src.inference import (
    CATEGORICAL_FEATURES,
    DEFAULT_MODEL_PATH,
    MODEL_FEATURES,
    PaySimA0Scorer,
)


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "paysim_a0_features.parquet"
LOCK_PATH = ROOT / "reports" / "test_evaluation_lock.json"
EVALUATION_PATH = ROOT / "reports" / "a0_final_test_evaluation.json"
TUNING_REPORT_PATH = ROOT / "reports" / "lightgbm_a0_tuning_report.json"
OUTPUT_PATH = ROOT / "reports" / "model_manifest.json"

PACKAGE_NAMES = [
    "lightgbm",
    "numpy",
    "pandas",
    "scikit-learn",
    "joblib",
    "pytest",
    "pyarrow",
]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def to_jsonable(value):
    """Convert NumPy and other common values to JSON-compatible types."""
    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]

    if hasattr(value, "tolist"):
        return value.tolist()

    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required report not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def file_metadata(path: Path, include_hash: bool = False) -> dict:
    """Record an artifact's location, availability, size, and optional hash."""
    result = {
        "path": path.relative_to(ROOT).as_posix(),
        "exists": path.is_file(),
    }

    if not path.is_file():
        return result

    result["size_bytes"] = path.stat().st_size

    if include_hash:
        result["sha256"] = sha256_file(path)

    return result


def git_metadata() -> dict:
    """Capture the Git revision from which the manifest was generated."""
    result = {
        "commit_sha": None,
        "branch": None,
        "working_tree_dirty": None,
    }

    try:
        commands = {
            "commit_sha": ["git", "rev-parse", "HEAD"],
            "branch": ["git", "branch", "--show-current"],
        }

        for key, command in commands.items():
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            result[key] = completed.stdout.strip()

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        result["working_tree_dirty"] = bool(status.stdout.strip())

    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return result


def normalize_repo_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the PaySim final model reproducibility manifest."
    )
    parser.add_argument(
        "--hash-data",
        action="store_true",
        help="Also calculate SHA-256 for the processed feature parquet.",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------
    # 1. Validate the approved feature set and load the model
    # ---------------------------------------------------------

    assert_deployable_feature_set("A0")

    if not DEFAULT_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Final model not found: {DEFAULT_MODEL_PATH}"
        )

    scorer = PaySimA0Scorer(DEFAULT_MODEL_PATH)
    model = scorer.model

    actual_features = list(model.booster_.feature_name())

    if actual_features != MODEL_FEATURES:
        raise ValueError(
            "Loaded model does not match the inference feature contract.\n"
            f"Expected: {MODEL_FEATURES}\n"
            f"Actual: {actual_features}"
        )

    # ---------------------------------------------------------
    # 2. Load the existing locked evaluation evidence
    # ---------------------------------------------------------

    lock = read_json(LOCK_PATH)
    evaluation = read_json(EVALUATION_PATH)

    if lock.get("status") != "completed_once":
        raise ValueError("The locked test is not marked completed_once.")

    expected_model_path = DEFAULT_MODEL_PATH.relative_to(ROOT).as_posix()

    if normalize_repo_path(lock["model"]) != expected_model_path:
        raise ValueError("The test lock references another model.")

    if normalize_repo_path(evaluation["model_path"]) != expected_model_path:
        raise ValueError("The evaluation report references another model.")

    if lock["test_steps"] != evaluation["test_steps"]:
        raise ValueError("Test-period mismatch between lock and report.")

    # Confirm actual estimator parameters agree with the frozen report.
    model_parameters = model.get_params()

    for name, reported_value in evaluation["model_parameters"].items():
        actual_value = to_jsonable(model_parameters.get(name))

        if actual_value != reported_value:
            raise ValueError(
                f"Parameter mismatch for {name}: "
                f"model={actual_value!r}, report={reported_value!r}"
            )

    # ---------------------------------------------------------
    # 3. Capture environment information
    # ---------------------------------------------------------

    package_versions = {}

    for package in PACKAGE_NAMES:
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None

    # ---------------------------------------------------------
    # 4. Build the manifest
    # ---------------------------------------------------------

    manifest = {
        "manifest_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),

        "project": {
            "name": "PaySim Fraud Detection",
            "dataset": "PaySim synthetic mobile-money transactions",
            "scoring_moment": "Before transaction approval",
            "deployment_status": "Research challenger; not production approved",
        },

        "model": {
            "name": "LightGBM A0 final test model",
            "class": type(model).__name__,
            "framework": "LightGBM",
            "artifact": file_metadata(
                DEFAULT_MODEL_PATH,
                include_hash=True,
            ),
            "feature_set": "A0",
            "features": actual_features,
            "categorical_features": CATEGORICAL_FEATURES,
            "categorical_levels": to_jsonable(scorer.category_levels),
            "parameters": to_jsonable(model_parameters),
            "priority_formula": "fraud_score * transaction_amount",
        },

        "training_provenance": {
            "validation_selection": {
                "method": "Bounded hyperparameter search",
                "trial_count": 12,
                "selected_trial": 10,
                "selected_trial_name": "lower_feature_sampling",
                "training_steps": [1, 421],
                "purged_steps": [422, 445],
                "validation_steps": [446, 594],
                "selection_metric": "Fraud value caught at fixed alert budget",
            },
            "final_refit": {
                "feature_set": "A0",
                "training_label_cutoff_step":
                    evaluation["training_label_cutoff_step"],
                "purged_steps": evaluation["purged_steps"],
                "input_artifact": file_metadata(DATA_PATH),
            },
        },

        "operating_policy": {
            "policy": "A0 fraud score x amount",
            "alerts_per_24_steps": evaluation["alerts_per_24_steps"],
            "comparator": "TRANSFER/CASH_OUT by amount",
            "success_hurdle": "At least 5% more fraud value than the rule",
        },

        "locked_test": {
            "status": lock["status"],
            "scored_at_utc": lock["scored_at_utc"],
            "test_steps": evaluation["test_steps"],
            "test_rows": evaluation["test_rows"],
            "model_fraud_value_caught":
                evaluation["model_fraud_value_caught"],
            "rule_fraud_value_caught":
                evaluation["best_rule_fraud_value_caught"],
            "value_uplift_vs_rule_pct":
                evaluation["value_uplift_vs_rule_pct"],
            "required_uplift_pct": 5.0,
            "meets_success_hurdle":
                evaluation["meets_5pct_value_hurdle"],
            "decision": (
                "Rule remains selected because the model did not meet "
                "the predeclared 5% improvement hurdle."
            ),
        },

        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "packages": package_versions,
        },

        "git": git_metadata(),

        "source_artifacts": {
            "feature_contract": file_metadata(
                ROOT / "src" / "feature_contract.py",
                include_hash=True,
            ),
            "inference_module": file_metadata(
                ROOT / "src" / "inference.py",
                include_hash=True,
            ),
            "final_test_script": file_metadata(
                ROOT / "src" / "evaluate_a0_final_test.py",
                include_hash=True,
            ),
            "tuning_script": file_metadata(
                ROOT / "src" / "tune_lightgbm_a0.py",
                include_hash=True,
            ),
            "requirements": file_metadata(
                ROOT / "requirements.txt",
                include_hash=True,
            ),
            "test_evaluation": file_metadata(
                EVALUATION_PATH,
                include_hash=True,
            ),
            "test_lock": file_metadata(
                LOCK_PATH,
                include_hash=True,
            ),
            "tuning_report": file_metadata(
                TUNING_REPORT_PATH,
                include_hash=True,
            ),
        },

        "reproducibility_notes": [
            "Model and processed data binaries are excluded from Git.",
            "The model SHA-256 identifies the exact local model artifact.",
            "The locked test was completed once and must not be rerun.",
            "The model did not meet the predeclared 5% success hurdle.",
            "Transaction value is a proxy, not verified financial loss.",
            "The test holdout was exposed to earlier full-dataset EDA.",
        ],
    }

    if args.hash_data:
        manifest["training_provenance"]["final_refit"][
            "input_artifact"
        ] = file_metadata(DATA_PATH, include_hash=True)

    # ---------------------------------------------------------
    # 5. Save the manifest
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nModel manifest created successfully.")
    print(f"Output: {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Model SHA-256: {manifest['model']['artifact']['sha256']}")
    print(f"Model features: {len(actual_features)}")
    print(f"Test status: {lock['status']}")
    print(
        f"Test uplift: "
        f"{evaluation['value_uplift_vs_rule_pct']:.4f}%"
    )
    print(
        f"5% hurdle met: "
        f"{evaluation['meets_5pct_value_hurdle']}"
    )


if __name__ == "__main__":
    main()
