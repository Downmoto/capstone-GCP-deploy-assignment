from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/heart.csv")
ARTIFACTS_DIR = Path("artifacts")
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "ChestPainType": "chest pain type",
        "RestingBP": "resting bp s",
        "FastingBS": "fasting bs",
        "RestingECG": "resting ecg",
        "MaxHR": "max hr",
        "ExerciseAngina": "exercise angina",
        "ST_Slope": "st slope",
        "HeartDisease": "heartdisease",
        "Age": "age",
        "Sex": "sex",
        "Cholesterol": "cholesterol",
        "Oldpeak": "oldpeak",
    }
    return df.rename(columns=rename_map)


def _build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    categorical_features = frame.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = [col for col in frame.columns if col not in categorical_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def _evaluate(y_true: pd.Series, y_pred: list[int], y_prob: list[float]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "expected dataset at data/heart.csv. download from kaggle and place it there."
        )

    data = _normalize_columns(pd.read_csv(DATA_PATH))
    target_column = "heartdisease"

    if target_column not in data.columns:
        raise ValueError("dataset must include 'HeartDisease' (or normalized 'heartdisease')")

    y = data[target_column].astype(int)
    X = data.drop(columns=[target_column])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = _build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    candidate_models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=400, random_state=42),
    }

    results: dict[str, dict[str, float]] = {}
    best_name = ""
    best_f1 = -1.0
    best_model = None

    for model_name, model in candidate_models.items():
        model.fit(X_train_processed, y_train)
        preds = model.predict(X_test_processed)
        probs = model.predict_proba(X_test_processed)[:, 1]
        metrics = _evaluate(y_test, preds, probs)
        results[model_name] = metrics

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = model_name
            best_model = model

    if best_model is None:
        raise RuntimeError("no model was trained")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as fp:
        json.dump({"selected_model": best_name, "metrics": results}, fp, indent=2)

    print(f"saved model: {best_name}")
    print(f"artifacts: {MODEL_PATH}, {PREPROCESSOR_PATH}")
    print(f"metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
