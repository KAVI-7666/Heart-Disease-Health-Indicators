

import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request


def find_project_root(start_path: Path) -> Path:
    current = start_path.resolve()

    for path in [current, *current.parents]:
        if (path / "models").exists() and (path / "data").exists():
            return path

    raise FileNotFoundError("Project root not found.")


BASE_DIR = find_project_root(Path(__file__).resolve())
MODEL_PATH = BASE_DIR / "models" / "catboost_model.pkl"
COLUMNS_PATH = BASE_DIR / "models" / "catboost_columns.pkl"


with open(MODEL_PATH, "rb") as model_file:
    catboost_model = pickle.load(model_file)

with open(COLUMNS_PATH, "rb") as columns_file:
    MODEL_COLUMNS = pickle.load(columns_file)


app = Flask(__name__)


FIELD_CONFIG = {
    "HighBP": {"label": "High Blood Pressure", "type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
    "HighChol": {"label": "High Cholesterol", "type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
    "CholCheck": {"label": "Cholesterol Check in Last 5 Years", "type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
    "BMI": {"label": "Body Mass Index", "type": "number", "step": "0.1", "min": 10, "max": 100},
    "Smoker": {"label": "Smoker", "type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
    "Stroke": {"label": "History of Stroke", "type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
    "Diabetes": {
        "label": "Diabetes Level",
        "type": "select",
        "options": [
            {"value": 0, "label": "No Diabetes"},
            {"value": 1, "label": "Pre-Diabetes or During Pregnancy"},
            {"value": 2, "label": "Diabetes"},
        ],
    },
    "PhysActivity": {"label": "Physical Activity", "type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
    "Fruits": {"label": "Eats Fruits Regularly", "type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
    "Veggies": {"label": "Eats Vegetables Regularly", "type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
    "HvyAlcoholConsump": {"label": "Heavy Alcohol Consumption", "type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
    "AnyHealthcare": {"label": "Has Healthcare Coverage", "type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
    "NoDocbcCost": {"label": "Could Not See Doctor Due to Cost", "type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
    "GenHlth": {
        "label": "General Health",
        "type": "select",
        "options": [
            {"value": 1, "label": "1 - Excellent"},
            {"value": 2, "label": "2 - Very Good"},
            {"value": 3, "label": "3 - Good"},
            {"value": 4, "label": "4 - Fair"},
            {"value": 5, "label": "5 - Poor"},
        ],
    },
    "MentHlth": {"label": "Poor Mental Health Days", "type": "number", "step": "1", "min": 0, "max": 30},
    "PhysHlth": {"label": "Poor Physical Health Days", "type": "number", "step": "1", "min": 0, "max": 30},
    "DiffWalk": {"label": "Difficulty Walking", "type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
    "Sex": {"label": "Sex", "type": "select", "options": [{"value": 0, "label": "Female"}, {"value": 1, "label": "Male"}]},
    "Age": {
        "label": "Age Group",
        "type": "select",
        "options": [
            {"value": 1, "label": "1 - 18 to 24"},
            {"value": 2, "label": "2 - 25 to 29"},
            {"value": 3, "label": "3 - 30 to 34"},
            {"value": 4, "label": "4 - 35 to 39"},
            {"value": 5, "label": "5 - 40 to 44"},
            {"value": 6, "label": "6 - 45 to 49"},
            {"value": 7, "label": "7 - 50 to 54"},
            {"value": 8, "label": "8 - 55 to 59"},
            {"value": 9, "label": "9 - 60 to 64"},
            {"value": 10, "label": "10 - 65 to 69"},
            {"value": 11, "label": "11 - 70 to 74"},
            {"value": 12, "label": "12 - 75 to 79"},
            {"value": 13, "label": "13 - 80 or older"},
        ],
    },
    "Education": {
        "label": "Education Level",
        "type": "select",
        "options": [
            {"value": 1, "label": "1 - Never attended school or kindergarten only"},
            {"value": 2, "label": "2 - Grades 1 through 8"},
            {"value": 3, "label": "3 - Grades 9 through 11"},
            {"value": 4, "label": "4 - High school graduate"},
            {"value": 5, "label": "5 - College 1 year to 3 years"},
            {"value": 6, "label": "6 - College 4 years or more"},
        ],
    },
    "Income": {
        "label": "Income Level",
        "type": "select",
        "options": [
            {"value": 1, "label": "1 - Less than $10,000"},
            {"value": 2, "label": "2 - $10,000 to less than $15,000"},
            {"value": 3, "label": "3 - $15,000 to less than $20,000"},
            {"value": 4, "label": "4 - $20,000 to less than $25,000"},
            {"value": 5, "label": "5 - $25,000 to less than $35,000"},
            {"value": 6, "label": "6 - $35,000 to less than $50,000"},
            {"value": 7, "label": "7 - $50,000 to less than $75,000"},
            {"value": 8, "label": "8 - $75,000 or more"},
        ],
    },
}


def build_input_frame(form_data: dict[str, str]) -> pd.DataFrame:
    # Create the input row in the exact order expected by the saved CatBoost model.
    values: dict[str, float] = {}

    for column in MODEL_COLUMNS:
        raw_value = form_data.get(column, "0").strip()
        values[column] = float(raw_value)

    return pd.DataFrame([values], columns=MODEL_COLUMNS)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    risk_probability = None
    submitted_values = {column: "0" for column in MODEL_COLUMNS}

    if request.method == "POST":
        submitted_values = {column: request.form.get(column, "0") for column in MODEL_COLUMNS}
        input_frame = build_input_frame(submitted_values)

        prediction = int(catboost_model.predict(input_frame)[0])
        probability = float(catboost_model.predict_proba(input_frame)[0][1])

        prediction_text = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"
        risk_probability = round(probability * 100, 2)

    return render_template(
        "index.html",
        fields=MODEL_COLUMNS,
        field_config=FIELD_CONFIG,
        prediction_text=prediction_text,
        risk_probability=risk_probability,
        submitted_values=submitted_values,
    )


if __name__ == "__main__":
    app.run(debug=True)
