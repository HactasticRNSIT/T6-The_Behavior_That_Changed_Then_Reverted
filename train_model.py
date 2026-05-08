from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

CATEGORICAL = ["program", "age_group", "community"]
NUMERIC = [
    "early_engagement",
    "mid_engagement",
    "engagement_slope",
    "engagement_volatility",
    "baseline_habit_strength",
    "convenience",
    "social_norm",
    "environmental_fit",
    "motivation",
    "competing_demands",
    "habit_persistence",
    "feedback_visibility",
    "incentive_strength",
    "support_quality",
    "life_event_week",
]


def load_summary() -> pd.DataFrame:
    path = DATA_DIR / "participant_summary.csv"
    if not path.exists():
        raise FileNotFoundError("Run `python src/generate_dataset.py` before training.")
    return pd.read_csv(path)


def build_reversal_model() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=350,
        min_samples_leaf=6,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def feature_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL)
    feature_names = list(cat_names) + NUMERIC
    return (
        pd.DataFrame({"feature": feature_names, "importance": classifier.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(15)
    )


def train() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_summary()
    x = df[CATEGORICAL + NUMERIC]
    y = df["reversal_risk_label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.22, random_state=42, stratify=y
    )

    reversal_model = build_reversal_model()
    reversal_model.fit(x_train, y_train)
    predictions = reversal_model.predict(x_test)
    probabilities = reversal_model.predict_proba(x_test)[:, 1]

    auc = roc_auc_score(y_test, probabilities)
    report = classification_report(y_test, predictions)

    trajectory_features = df[
        ["early_engagement", "mid_engagement", "late_engagement", "engagement_slope", "engagement_volatility"]
    ]
    cluster_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("cluster", KMeans(n_clusters=4, n_init=20, random_state=42)),
        ]
    )
    df["trajectory_cluster"] = cluster_model.fit_predict(trajectory_features)

    joblib.dump(reversal_model, MODEL_DIR / "reversal_model.pkl")
    joblib.dump(cluster_model, MODEL_DIR / "trajectory_cluster_model.pkl")
    feature_importance(reversal_model).to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    df.to_csv(DATA_DIR / "participant_summary_with_clusters.csv", index=False)

    print("Reversal model trained")
    print(f"ROC AUC: {auc:.3f}")
    print(report)
    print(f"Saved models to {MODEL_DIR}")


if __name__ == "__main__":
    train()
