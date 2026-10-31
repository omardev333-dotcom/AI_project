"""Interpretable study recommendation rules."""
def get_performance_level(predicted_grade: float) -> str:
    if predicted_grade >= 85:
        return "Excellent"
    if predicted_grade >= 70:
        return "Good"
    if predicted_grade >= 50:
        return "Needs improvement"
    return "At risk"

def generate_recommendations(student: dict) -> list[str]:
    tips = []

    if student["study_hours"] < 2:
        tips.append("Try building a consistent study routine, starting with two focused sessions.")
    elif student["study_hours"] < 4:
        tips.append("Your study time is a useful start. Add short revision sessions for difficult topics.")

    if student["attendance"] < 80:
        tips.append("Improve class attendance where possible and review notes from missed lessons.")

    if student["previous_grade"] < 60:
        tips.append("Review the foundations of earlier topics before moving to advanced material.")

    if student["assignment_score"] < 65:
        tips.append("Review assignment feedback and practise similar questions.")

    if student["sleep_hours"] < 7:
        tips.append("A regular, sufficient sleep schedule can support concentration and learning.")

    if student["past_failures"] >= 2:
        tips.append("Break challenging subjects into smaller goals and ask a teacher for targeted support.")

    if not tips:
        tips.append("Keep your current habits, review mistakes regularly, and maintain a balanced routine.")

    return tips
