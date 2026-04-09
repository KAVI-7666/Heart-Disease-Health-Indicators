from pathlib import Path
import pickle

import pandas as pd


def find_project_root(start_path: Path) -> Path:
    current = start_path.resolve()

    for path in [current, *current.parents]:
        if (path / "data").exists() and (path / "models").exists():
            return path

    raise FileNotFoundError("Project root not found.")


BASE_DIR = find_project_root(Path(__file__).resolve())
DATA_PATH = BASE_DIR / "data" / "processed" / "heart_disease_processed.csv"
MODEL_PATH = BASE_DIR / "models" / "catboost_model.pkl"
COLUMNS_PATH = BASE_DIR / "models" / "catboost_columns.pkl"
TARGET_COLUMN = "HeartDiseaseorAttack"


with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(COLUMNS_PATH, "rb") as columns_file:
    columns = pickle.load(columns_file)

data = pd.read_csv(DATA_PATH)
df = data.copy()

X_sample = df[columns].sample(5, random_state=42)
print(X_sample)

predictions = model.predict(X_sample)
probabilities = model.predict_proba(X_sample)


print("Prediction Results:")
for i, pred in enumerate(predictions):
    label = "Heart Disease" if int(pred) == 1 else "No Heart Disease"
    prob = probabilities[i][1]
    print(f"Row {i+1}: {label} (probability: {prob:.4f})")
