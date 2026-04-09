# Heart Disease Health Indicators

This project predicts heart disease risk using health indicator data. It includes preprocessing, exploratory analysis, multiple machine learning models, model comparison, testing, and a Flask-based interface powered by a saved CatBoost model.

## Project Overview

The workflow of this project is:

1. Preprocess the raw dataset
2. Perform exploratory data analysis
3. Train multiple machine learning models
4. Compare model performance
5. Save the best model artifacts
6. Use the trained CatBoost model in a web interface

## Models Used

- Logistic Regression
- Random Forest
- XGBoost
- CatBoost

## Imbalance Handling

Different imbalance techniques were explored in notebooks, including:

- Cost-sensitive learning
- Random Oversampling
- Random Undersampling
- SMOTE
- Borderline-SMOTE
- ADASYN
- Tomek Links
- SMOTE + Tomek Links
- SMOTE + ENN

## Project Structure

```text
Heart Disease Health Indicators/
├── app/
│   ├── app.py
│   ├── static/
│   │   ├── script.js
│   │   └── style.css
│   └── templates/
│       └── index.html
├── config/
│   └── config.yaml
├── data/
│   ├── external/
│   ├── processed/
│   └── raw/
├── logs/
│   └── pipeline.log
├── models/
├── notebooks/
│   ├── EDA.ipynb
│   ├── preprocessing.ipynb
│   ├── eda_images/
│   └── training_models/
│       ├── Algorithm Comparison.ipynb
│       ├── Imbalance Techniques.ipynb
│       ├── Logistic Regression.ipynb
│       ├── Random Forest.ipynb
│       ├── XGBoost.ipynb
│       └── catboost.ipynb
├── tests/
│   └── test_model.py
├── main.py
├── README.md
└── requirements.txt
```

## Requirements

Install dependencies using:

```powershell
python -m pip install -r requirements.txt
```

Main packages used in this project:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- xgboost
- catboost
- optuna
- flask
- jupyter

## How to Run the Full Pipeline

Run the project pipeline in sequence with:

```powershell
python main.py
```

The pipeline executes:

1. `preprocessing.ipynb`
2. `EDA.ipynb`
3. `Logistic Regression.ipynb`
4. `Random Forest.ipynb`
5. `XGBoost.ipynb`
6. `catboost.ipynb`
7. `tests/test_model.py`

Logs are written to:

```text
logs/pipeline.log
```

## How to Run the Interface

The Flask app uses the saved CatBoost model from the `models` folder.

Run the app with:

```powershell
python app/app.py
```

Then open the local Flask URL shown in the terminal.

## Saved Model Files

The project saves model artifacts in the `models` folder, including:

- `logistic_regression_model.pkl`
- `logistic_regression_scaler.pkl`
- `logistic_regression_columns.pkl`
- `random_forest_model.pkl`
- `random_forest_columns.pkl`
- `xgboost_model.pkl`
- `xgboost_columns.pkl`
- `catboost_model.pkl`
- `catboost_columns.pkl`

## Interface Features

The web interface:

- uses the CatBoost model for prediction
- accepts all required health indicator inputs
- shows human-readable labels for encoded values
- displays the predicted heart disease risk
- displays the predicted probability

## Notes

- The model expects encoded values that match the training dataset
- Notebooks are executed in place using Jupyter `nbconvert`
- The interface is designed for local prediction and demonstration


