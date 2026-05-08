from __future__ import annotations

import pandas as pd


def risk_band(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    if probability >= 0.42:
        return "Medium"
    return "Low"


def recommended_action(row: pd.Series) -> str:
    if row["competing_demands"] > 0.62:
        return "Reduce friction and offer flexible, low-effort alternatives."
    if row["feedback_visibility"] < 0.45:
        return "Increase progress feedback and make benefits visible weekly."
    if row["support_quality"] < 0.45:
        return "Add peer/accountability support after the incentive period."
    if row["convenience"] < 0.45:
        return "Improve access, reminders, defaults, or nearby infrastructure."
    if row["habit_persistence"] > 0.65:
        return "Use relapse planning and replacement routines for the old habit."
    return "Maintain reinforcement with light-touch check-ins."


def top_context_flags(row: pd.Series) -> list[str]:
    flags = []
    if row["early_engagement"] >= 0.55 and row["mid_engagement"] < row["early_engagement"] - 0.10:
        flags.append("Early drop after initial adoption")
    if row["life_event_week"] > 0:
        flags.append(f"Life/context disruption around week {int(row['life_event_week'])}")
    if row["competing_demands"] > 0.62:
        flags.append("High competing demands")
    if row["feedback_visibility"] < 0.45:
        flags.append("Low feedback visibility")
    if row["support_quality"] < 0.45:
        flags.append("Weak structured support")
    return flags or ["No major warning flag"]
