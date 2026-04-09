
import importlib.util
import logging
import subprocess
import sys
from pathlib import Path


def find_project_root(start_path: Path) -> Path:
    current = start_path.resolve()

    for path in [current, *current.parents]:
        if (path / "data").exists() and (path / "notebooks").exists():
            return path

    raise FileNotFoundError("Project root not found.")


BASE_DIR = find_project_root(Path(__file__).resolve())
NOTEBOOK_DIR = BASE_DIR / "notebooks"
TRAINING_DIR = NOTEBOOK_DIR / "training_models"
TEST_DIR = BASE_DIR / "tests"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "pipeline.log"

REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
    "imblearn": "imbalanced-learn",
    "xgboost": "xgboost",
    "catboost": "catboost",
    "optuna": "optuna",
    "jupyter": "jupyter",
}


PIPELINE_STEPS = [
    NOTEBOOK_DIR / "preprocessing.ipynb",
    NOTEBOOK_DIR / "EDA.ipynb",
    TRAINING_DIR / "Logistic Regression.ipynb",
    TRAINING_DIR / "Random Forest.ipynb",
    TRAINING_DIR / "XGBoost.ipynb",
    TRAINING_DIR / "catboost.ipynb",
    TEST_DIR / "test_model.py",
]


def configure_logging() -> None:
    # Write pipeline progress both to the console and to a persistent log file.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def check_required_packages() -> None:
    # Fail early with a clear message if the current environment is missing packages.
    missing_packages = []

    for module_name, package_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(module_name) is None:
            missing_packages.append(package_name)

    if missing_packages:
        unique_missing = sorted(set(missing_packages))
        install_command = f'"{sys.executable}" -m pip install ' + " ".join(unique_missing)
        raise ModuleNotFoundError(
            "Missing required packages for the pipeline: "
            f"{', '.join(unique_missing)}. "
            f"Install them with: {install_command}"
        )


def run_notebook(notebook_path: Path) -> None:
    # Execute one notebook in place so outputs are stored back into the same file.
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        str(notebook_path),
    ]

    logging.info("Running notebook: %s", notebook_path.name)
    subprocess.run(command, check=True, cwd=BASE_DIR)
    logging.info("Completed notebook: %s", notebook_path.name)


def run_python_file(file_path: Path) -> None:
    # Execute the Python test file after all notebooks finish successfully.
    command = [sys.executable, str(file_path)]

    logging.info("Running python file: %s", file_path.name)
    subprocess.run(command, check=True, cwd=BASE_DIR)
    logging.info("Completed python file: %s", file_path.name)


def run_pipeline() -> None:
    # Run every step in the required order and stop immediately on failure.
    for step in PIPELINE_STEPS:
        if not step.exists():
            raise FileNotFoundError(f"Missing pipeline step: {step}")

        if step.suffix == ".ipynb":
            run_notebook(step)
        elif step.suffix == ".py":
            run_python_file(step)
        else:
            raise ValueError(f"Unsupported pipeline file type: {step}")


def main() -> None:
    configure_logging()
    logging.info("Pipeline started.")
    logging.info("Log file: %s", LOG_FILE)

    try:
        check_required_packages()
        run_pipeline()
    except Exception as error:
        logging.exception("Pipeline failed: %s", error)
        raise

    logging.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
