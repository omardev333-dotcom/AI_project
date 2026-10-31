"""Train and evaluate the student grade prediction model."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "student_performance_demo.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "student_grade_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

FEATURES = [
    "study_hours", "attendance", "previous_grade",
    "assignment_score", "sleep_hours", "past_failures"
]
TARGET = "final_grade"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Demo dataset not found. Run `python generate_data.py` first."
        )

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "model": "RandomForestRegressor",
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
        "rmse": round(float(mean_squared_error(y_test, predictions) ** 0.5), 2),
        "r2": round(float(r2_score(y_test, predictions)), 3),
        "features": FEATURES,
        "data_note": "Metrics are from synthetic demo data only."
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
