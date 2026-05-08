from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RNG = np.random.default_rng(42)

PROGRAMS = [
    "fitness",
    "recycling",
    "digital_wellbeing",
    "public_transport",
    "healthy_eating",
]


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1 / (1 + np.exp(-x))


def label_trajectory(early: float, late: float, slope: float) -> str:
    if early >= 0.58 and late >= 0.55 and slope > -0.015:
        return "sustained"
    if early >= 0.55 and late < 0.42:
        return "reverted"
    if early >= 0.50 and late < early - 0.18:
        return "faded"
    if late >= early + 0.15:
        return "late_adopter"
    return "inconsistent"


def generate(n_participants: int = 900, weeks: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summaries = []

    for participant_id in range(1, n_participants + 1):
        program = RNG.choice(PROGRAMS)
        age_group = RNG.choice(["18-25", "26-40", "41-60", "60+"], p=[0.24, 0.38, 0.27, 0.11])
        community = RNG.choice(["urban", "suburban", "rural"], p=[0.48, 0.34, 0.18])
        baseline_habit = RNG.beta(2.4, 2.1)
        convenience = RNG.beta(2.8, 1.9)
        social_norm = RNG.beta(2.2, 2.0)
        environmental_fit = RNG.beta(2.5, 2.0)
        motivation = RNG.beta(2.4, 1.8)
        competing_demands = RNG.beta(2.1, 2.3)
        habit_persistence = RNG.beta(2.5, 1.8)
        feedback_visibility = RNG.beta(2.3, 2.0)
        incentive_strength = RNG.beta(2.6, 2.1)
        support_quality = RNG.beta(2.4, 1.9)

        reinforcement_decay = RNG.uniform(0.015, 0.09)
        life_event_week = int(RNG.choice([0, *range(7, 22)], p=[0.72, *([0.28 / 15] * 15)]))
        personal_bias = RNG.normal(0, 0.18)

        weekly_engagement = []
        prior = baseline_habit

        for week in range(1, weeks + 1):
            incentive_active = 1 if week <= 8 else 0
            structured_support = max(0, support_quality - 0.018 * max(week - 10, 0))
            feedback = max(0, feedback_visibility - 0.012 * max(week - 12, 0))
            life_shock = 0.18 if life_event_week and week >= life_event_week else 0.0
            novelty = 0.25 * np.exp(-0.10 * week)
            fatigue = reinforcement_decay * max(week - 8, 0)

            latent = (
                -0.55
                + 1.15 * motivation
                + 0.85 * convenience
                + 0.65 * social_norm
                + 0.70 * environmental_fit
                + 0.55 * structured_support
                + 0.50 * feedback
                + 0.60 * incentive_strength * incentive_active
                + novelty
                - 0.95 * competing_demands
                - 0.72 * habit_persistence
                - fatigue
                - life_shock
                + personal_bias
                + RNG.normal(0, 0.17)
            )

            engagement = float(np.clip(0.62 * prior + 0.38 * sigmoid(latent), 0, 1))
            prior = engagement
            weekly_engagement.append(engagement)

            rows.append(
                {
                    "participant_id": participant_id,
                    "week": week,
                    "program": program,
                    "age_group": age_group,
                    "community": community,
                    "baseline_habit_strength": round(baseline_habit, 3),
                    "convenience": round(convenience, 3),
                    "social_norm": round(social_norm, 3),
                    "environmental_fit": round(environmental_fit, 3),
                    "motivation": round(max(0, motivation - 0.008 * week + RNG.normal(0, 0.025)), 3),
                    "competing_demands": round(min(1, competing_demands + life_shock + RNG.normal(0, 0.03)), 3),
                    "habit_persistence": round(habit_persistence, 3),
                    "feedback_visibility": round(feedback, 3),
                    "incentive_active": incentive_active,
                    "support_quality": round(structured_support, 3),
                    "life_event": int(life_event_week == week),
                    "engagement_score": round(engagement, 3),
                    "activity_count": int(max(0, RNG.poisson(1 + engagement * 8))),
                    "self_report_adoption": int(engagement > RNG.uniform(0.42, 0.62)),
                }
            )

        early = float(np.mean(weekly_engagement[:6]))
        mid = float(np.mean(weekly_engagement[6:12]))
        late = float(np.mean(weekly_engagement[-6:]))
        slope = float(np.polyfit(np.arange(weeks), weekly_engagement, 1)[0])
        label = label_trajectory(early, late, slope)
        reversal = int(label in {"reverted", "faded"})
        early_warning = int(early >= 0.55 and np.mean(weekly_engagement[8:12]) < early - 0.10)

        summaries.append(
            {
                "participant_id": participant_id,
                "program": program,
                "age_group": age_group,
                "community": community,
                "early_engagement": round(early, 3),
                "mid_engagement": round(mid, 3),
                "late_engagement": round(late, 3),
                "engagement_slope": round(slope, 4),
                "engagement_volatility": round(float(np.std(weekly_engagement)), 3),
                "trajectory_label": label,
                "reversal_risk_label": reversal,
                "early_warning_label": early_warning,
                "baseline_habit_strength": round(baseline_habit, 3),
                "convenience": round(convenience, 3),
                "social_norm": round(social_norm, 3),
                "environmental_fit": round(environmental_fit, 3),
                "motivation": round(motivation, 3),
                "competing_demands": round(competing_demands, 3),
                "habit_persistence": round(habit_persistence, 3),
                "feedback_visibility": round(feedback_visibility, 3),
                "incentive_strength": round(incentive_strength, 3),
                "support_quality": round(support_quality, 3),
                "life_event_week": life_event_week,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(summaries)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    longitudinal, summary = generate()
    longitudinal.to_csv(DATA_DIR / "behavior_change_dataset.csv", index=False)
    summary.to_csv(DATA_DIR / "participant_summary.csv", index=False)
    print(f"Wrote {len(longitudinal):,} weekly records and {len(summary):,} participants to {DATA_DIR}")


if __name__ == "__main__":
    main()
