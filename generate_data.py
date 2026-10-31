"""Generate a reproducible DEMO dataset for the AI Smart Study Advisor.

Important: These are synthetic records created for learning and demonstration.
They are not real student records and should not be used to make real academic
decisions. Replace them with an appropriately licensed, anonymized dataset
before making real-world performance claims.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 1200
OUT = Path(__file__).parent / "data" / "student_performance_demo.csv"

def main():
    rng = np.random.default_rng(SEED)
    study_hours = rng.uniform(0.2, 7.0, N_ROWS)
    attendance = rng.uniform(45, 100, N_ROWS)
    previous_grade = rng.uniform(25, 95, N_ROWS)
    assignment_score = rng.uniform(30, 100, N_ROWS)
    sleep_hours = rng.uniform(4.0, 10.0, N_ROWS)
    past_failures = rng.integers(0, 4, N_ROWS)

    # Transparent simulated relationship with noise; not a scientific model.
    sleep_factor = 1 - np.abs(sleep_hours - 8.0) / 12.0
    final_grade = (
        0.24 * previous_grade
        + 3.2 * study_hours
        + 0.16 * attendance
        + 0.18 * assignment_score
        + 5.0 * sleep_factor
        - 3.0 * past_failures
        + rng.normal(0, 5.5, N_ROWS)
    )
    final_grade = np.clip(final_grade, 0, 100)

    df = pd.DataFrame({
        "study_hours": study_hours.round(1),
        "attendance": attendance.round(1),
        "previous_grade": previous_grade.round(1),
        "assignment_score": assignment_score.round(1),
        "sleep_hours": sleep_hours.round(1),
        "past_failures": past_failures,
        "final_grade": final_grade.round(1),
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Created {len(df)} synthetic demo records at: {OUT}")

if __name__ == "__main__":
    main()
