from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

from advisor import generate_recommendations, get_performance_level

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "student_grade_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
DATA_PATH = ROOT / "data" / "student_performance_demo.csv"

st.set_page_config(
    page_title="AI Smart Study Advisor",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 AI Smart Study Advisor")
st.caption("A learning-support demo that predicts an estimated grade and suggests study improvements.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

with st.sidebar:
    st.header("About this project")
    st.write(
        "This educational prototype uses a Random Forest regression model "
        "and a transparent recommendation layer."
    )
    st.info(
        "Demo limitation: the included training records are synthetic. "
        "Predictions are for software demonstration only, not real academic decisions."
    )

if model is None:
    st.error("The model file is missing. Run `python generate_data.py` and `python train_model.py`.")
    st.stop()

left, right = st.columns([1, 1.15], gap="large")

with left:
    st.subheader("Student information")
    with st.form("student_form"):
        study_hours = st.slider("Daily study hours", 0.0, 12.0, 3.0, 0.5)
        attendance = st.slider("Attendance (%)", 0, 100, 85)
        previous_grade = st.slider("Previous grade (%)", 0, 100, 70)
        assignment_score = st.slider("Assignment average (%)", 0, 100, 75)
        sleep_hours = st.slider("Sleep per night (hours)", 0.0, 12.0, 8.0, 0.5)
        past_failures = st.selectbox("Number of previous failed subjects", [0, 1, 2, 3, 4, 5])
        submitted = st.form_submit_button("Analyze my study profile", type="primary", use_container_width=True)

student = {
    "study_hours": float(study_hours),
    "attendance": float(attendance),
    "previous_grade": float(previous_grade),
    "assignment_score": float(assignment_score),
    "sleep_hours": float(sleep_hours),
    "past_failures": int(past_failures),
}

with right:
    st.subheader("AI analysis")
    if submitted:
        input_df = pd.DataFrame([student])
        prediction = float(model.predict(input_df)[0])
        prediction = max(0.0, min(100.0, prediction))
        level = get_performance_level(prediction)

        c1, c2 = st.columns(2)
        c1.metric("Estimated final grade", f"{prediction:.1f}%")
        c2.metric("Performance category", level)

        st.progress(int(round(prediction)))
        st.markdown("#### Personalized study suggestions")
        for tip in generate_recommendations(student):
            st.markdown(f"- {tip}")

        st.warning(
            "This is an approximate model output based on synthetic demonstration data. "
            "It is not a guaranteed result or a judgment of a student's ability."
        )
    else:
        st.info("Adjust the student information, then select **Analyze my study profile**.")
        st.markdown(
            """
            **The system will:**
            - Estimate a possible final grade.
            - Categorize the model's estimate.
            - Suggest practical study improvements.
            """
        )

st.divider()
st.subheader("Model evaluation")
if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", metrics["mae"])
    m2.metric("RMSE", metrics["rmse"])
    m3.metric("R²", metrics["r2"])
    st.caption(
        f"Evaluated on {metrics['test_rows']} held-out synthetic demo records. "
        "These metrics do not establish real-world accuracy."
    )
else:
    st.info("Evaluation metrics will appear after training the model.")

if DATA_PATH.exists():
    with st.expander("Preview the demo dataset"):
        data = pd.read_csv(DATA_PATH)
        st.dataframe(data.head(10), use_container_width=True)
        st.caption(f"{len(data)} synthetic demonstration rows.")
