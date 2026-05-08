from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.insights import recommended_action, risk_band, top_context_flags
from src.train_model import CATEGORICAL, NUMERIC


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "participant_summary.csv"
MODEL_PATH = ROOT / "models" / "reversal_model.pkl"


def predict_for_participant(participant_id: int) -> None:
    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Missing dataset/model. Run `python generate_dataset.py` and `python train_model.py` first."
        )

    data = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    match = data[data["participant_id"] == participant_id]
    if match.empty:
        raise ValueError(f"Participant {participant_id} not found.")

    row = match.iloc[0].copy()
    probability = float(model.predict_proba(match[CATEGORICAL + NUMERIC])[:, 1][0])

    row["reversal_probability"] = probability
    row["risk_band"] = risk_band(probability)

    print(f"Participant ID: {participant_id}")
    print(f"Program: {row['program']}")
    print(f"Trajectory label: {row['trajectory_label']}")
    print(f"Reversal probability: {probability:.1%}")
    print(f"Risk band: {row['risk_band']}")
    print(f"Recommended action: {recommended_action(row)}")
    print("Warning flags:")
    for flag in top_context_flags(row):
        print(f"- {flag}")


if __name__ == "__main__":
    predict_for_participant(1)
