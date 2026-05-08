from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.insights import recommended_action, risk_band, top_context_flags
from src.train_model import CATEGORICAL, NUMERIC


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


st.set_page_config(
    page_title="Behavior Reversal Detector",
    page_icon="",
    layout="wide",
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weekly = pd.read_csv(DATA_DIR / "behavior_change_dataset.csv")
    summary_path = DATA_DIR / "participant_summary_with_clusters.csv"
    if not summary_path.exists():
        summary_path = DATA_DIR / "participant_summary.csv"
    summary = pd.read_csv(summary_path)
    importance = pd.read_csv(MODEL_DIR / "feature_importance.csv")
    return weekly, summary, importance


@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "reversal_model.pkl")


def require_assets() -> None:
    missing = [
        str(path.relative_to(ROOT))
        for path in [
            DATA_DIR / "behavior_change_dataset.csv",
            DATA_DIR / "participant_summary.csv",
            MODEL_DIR / "reversal_model.pkl",
            MODEL_DIR / "feature_importance.csv",
        ]
        if not path.exists()
    ]
    if missing:
        st.error("Project assets are missing. Run these commands first:")
        st.code("python src/generate_dataset.py\npython src/train_model.py", language="powershell")
        st.write("Missing:", ", ".join(missing))
        st.stop()


def metric(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


require_assets()
weekly, summary, importance = load_data()
model = load_model()

summary = summary.copy()
summary["reversal_probability"] = model.predict_proba(summary[CATEGORICAL + NUMERIC])[:, 1]
summary["risk_band"] = summary["reversal_probability"].map(risk_band)
summary["recommended_action"] = summary.apply(recommended_action, axis=1)

st.title("Behavior Reversal Detector")
st.caption("AI-powered analysis of behavior change after an intervention: persistence, fade-out, and early reversal signals.")

with st.sidebar:
    st.header("Filters")
    program = st.multiselect("Program", sorted(summary["program"].unique()), default=sorted(summary["program"].unique()))
    community = st.multiselect(
        "Community", sorted(summary["community"].unique()), default=sorted(summary["community"].unique())
    )
    risk = st.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    min_prob = st.slider("Minimum reversal probability", 0.0, 1.0, 0.0, 0.05)

filtered = summary[
    summary["program"].isin(program)
    & summary["community"].isin(community)
    & summary["risk_band"].isin(risk)
    & (summary["reversal_probability"] >= min_prob)
]

col1, col2, col3, col4 = st.columns(4)
with col1:
    metric("Participants", f"{len(filtered):,}")
with col2:
    metric("Avg reversal probability", f"{filtered['reversal_probability'].mean():.1%}")
with col3:
    metric("Observed reversal/fade", f"{filtered['reversal_risk_label'].mean():.1%}")
with col4:
    sustained = (filtered["trajectory_label"] == "sustained").mean()
    metric("Sustained adoption", f"{sustained:.1%}")

tab_overview, tab_trajectories, tab_participants, tab_scenario, tab_dataset = st.tabs(
    ["Overview", "Trajectories", "Participant Risk", "Try Scenario", "Dataset"]
)

with tab_overview:
    left, right = st.columns([1.35, 1])
    with left:
        fig = px.histogram(
            filtered,
            x="reversal_probability",
            color="risk_band",
            nbins=24,
            category_orders={"risk_band": ["High", "Medium", "Low"]},
            color_discrete_map={"High": "#d63f3f", "Medium": "#d99b2b", "Low": "#2b8f68"},
            title="Predicted Reversal Risk Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(
            importance.head(10).sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="Top Model Drivers",
            color="importance",
            color_continuous_scale="Tealrose",
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = px.sunburst(
        filtered,
        path=["program", "trajectory_label", "risk_band"],
        values=None,
        title="Program, Outcome, and Risk Mix",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_trajectories:
    sample_ids = filtered.sort_values("reversal_probability", ascending=False).head(24)["participant_id"]
    trajectory_weekly = weekly[weekly["participant_id"].isin(sample_ids)]
    fig = px.line(
        trajectory_weekly,
        x="week",
        y="engagement_score",
        color="participant_id",
        line_group="participant_id",
        title="Highest-Risk Engagement Trajectories",
        labels={"engagement_score": "Engagement score"},
    )
    fig.add_vrect(x0=8, x1=9, fillcolor="#f4c542", opacity=0.18, line_width=0)
    fig.add_annotation(x=8.5, y=1.02, text="Incentives removed", showarrow=False)
    st.plotly_chart(fig, use_container_width=True)

    grouped = weekly.merge(filtered[["participant_id", "trajectory_label"]], on="participant_id")
    avg = grouped.groupby(["week", "trajectory_label"], as_index=False)["engagement_score"].mean()
    fig = px.line(
        avg,
        x="week",
        y="engagement_score",
        color="trajectory_label",
        title="Average Long-Term Behavior by Outcome",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_participants:
    ordered = filtered.sort_values("reversal_probability", ascending=False)
    selected_id = st.selectbox("Participant", ordered["participant_id"].tolist())
    person = ordered[ordered["participant_id"] == selected_id].iloc[0]
    person_weekly = weekly[weekly["participant_id"] == selected_id]

    a, b, c = st.columns([1, 1, 1.2])
    with a:
        metric("Risk band", person["risk_band"])
        metric("Predicted reversal", f"{person['reversal_probability']:.1%}")
    with b:
        metric("Trajectory", str(person["trajectory_label"]).replace("_", " ").title())
        metric("Late engagement", f"{person['late_engagement']:.1%}")
    with c:
        st.subheader("Recommended Action")
        st.write(person["recommended_action"])

    st.subheader("Warning Flags")
    for flag in top_context_flags(person):
        st.write(f"- {flag}")

    fig = px.line(
        person_weekly,
        x="week",
        y="engagement_score",
        markers=True,
        title=f"Participant {selected_id} Weekly Engagement",
    )
    fig.add_vrect(x0=8, x1=9, fillcolor="#f4c542", opacity=0.18, line_width=0)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        person[
            [
                "program",
                "age_group",
                "community",
                "early_engagement",
                "mid_engagement",
                "late_engagement",
                "convenience",
                "social_norm",
                "environmental_fit",
                "motivation",
                "competing_demands",
                "feedback_visibility",
                "support_quality",
            ]
        ].to_frame("value"),
        use_container_width=True,
    )

with tab_scenario:
    st.subheader("Predict Risk for a New Participant")
    st.write("Enter current program and behavior context to estimate whether the early behavior change may fade or revert.")

    with st.form("scenario_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            input_program = st.selectbox("Program", sorted(summary["program"].unique()))
            input_age = st.selectbox("Age group", sorted(summary["age_group"].unique()))
            input_community = st.selectbox("Community", sorted(summary["community"].unique()))
            early_engagement = st.slider("Early engagement", 0.0, 1.0, 0.68, 0.01)
            mid_engagement = st.slider("Mid engagement", 0.0, 1.0, 0.48, 0.01)
        with c2:
            baseline_habit_strength = st.slider("Old habit strength", 0.0, 1.0, 0.62, 0.01)
            convenience = st.slider("Convenience", 0.0, 1.0, 0.52, 0.01)
            social_norm = st.slider("Social support/norm", 0.0, 1.0, 0.50, 0.01)
            environmental_fit = st.slider("Environmental fit", 0.0, 1.0, 0.55, 0.01)
            motivation = st.slider("Motivation", 0.0, 1.0, 0.58, 0.01)
        with c3:
            competing_demands = st.slider("Competing demands", 0.0, 1.0, 0.66, 0.01)
            habit_persistence = st.slider("Habit persistence", 0.0, 1.0, 0.64, 0.01)
            feedback_visibility = st.slider("Feedback visibility", 0.0, 1.0, 0.42, 0.01)
            incentive_strength = st.slider("Incentive strength", 0.0, 1.0, 0.70, 0.01)
            support_quality = st.slider("Support quality", 0.0, 1.0, 0.45, 0.01)

        life_event_week = st.slider("Life event week, 0 means none", 0, 24, 0)
        submitted = st.form_submit_button("Predict Reversal Risk")

    scenario = pd.DataFrame(
        [
            {
                "program": input_program,
                "age_group": input_age,
                "community": input_community,
                "early_engagement": early_engagement,
                "mid_engagement": mid_engagement,
                "engagement_slope": (mid_engagement - early_engagement) / 12,
                "engagement_volatility": abs(early_engagement - mid_engagement) / 2,
                "baseline_habit_strength": baseline_habit_strength,
                "convenience": convenience,
                "social_norm": social_norm,
                "environmental_fit": environmental_fit,
                "motivation": motivation,
                "competing_demands": competing_demands,
                "habit_persistence": habit_persistence,
                "feedback_visibility": feedback_visibility,
                "incentive_strength": incentive_strength,
                "support_quality": support_quality,
                "life_event_week": life_event_week,
            }
        ]
    )

    if submitted:
        probability = float(model.predict_proba(scenario[CATEGORICAL + NUMERIC])[:, 1][0])
        scenario_row = scenario.iloc[0].copy()
        scenario_row["reversal_probability"] = probability
        scenario_row["risk_band"] = risk_band(probability)

        r1, r2, r3 = st.columns(3)
        with r1:
            metric("Predicted reversal", f"{probability:.1%}")
        with r2:
            metric("Risk band", scenario_row["risk_band"])
        with r3:
            metric("Early-to-mid change", f"{mid_engagement - early_engagement:+.1%}")

        st.subheader("Recommended Action")
        st.write(recommended_action(scenario_row))

        st.subheader("Warning Flags")
        for flag in top_context_flags(scenario_row):
            st.write(f"- {flag}")

with tab_dataset:
    st.subheader("Participant Summary")
    st.dataframe(filtered.sort_values("reversal_probability", ascending=False), use_container_width=True)
    st.subheader("Weekly Longitudinal Dataset")
    st.dataframe(weekly[weekly["participant_id"].isin(filtered["participant_id"])], use_container_width=True)
