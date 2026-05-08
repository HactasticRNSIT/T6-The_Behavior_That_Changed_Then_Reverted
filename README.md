# Habit Pattern Detection

AI-powered hackathon prototype for detecting behavior persistence, early reversal risk, and long-term engagement trajectories after an intervention.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/generate_dataset.py
python src/train_model.py
streamlit run app.py
```

Generated files:

- `data/behavior_change_dataset.csv`
- `data/participant_summary.csv`
- `models/reversal_model.pkl`
- `models/trajectory_cluster_model.pkl`

## What It Does

- Creates a realistic longitudinal dataset for behavior-change programs.
- Detects early warning signs of behavioral reversal.
- Predicts whether a participant will sustain, fade, or revert after initial adoption.
- Clusters long-term engagement trajectories.
- Shows interpretable risk drivers and recommended intervention actions.

## Dataset Theme

The synthetic data models intervention programs such as fitness, recycling, digital wellbeing, public transport, and healthy eating. Each participant has weekly engagement, motivation, environmental context, social support, incentives, feedback visibility, competing demands, and activity logs over 24 weeks.
