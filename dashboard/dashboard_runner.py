import json
import os
from dotenv import load_dotenv

# --- CRITICAL FIX: Load Environment BEFORE importing src.meta_engine ---
load_dotenv() 

from src.meta_engine import generate_weekly_report

def get_mock_database_history():
    """
    Simulates fetching sessions from the database.
    """
    return [
        {
            "session_id": "session_001",
            "created_at": "2024-01-15T10:00:00",
            "form_score": 65,
            "primary_fault": "Hip Sag",
            # We add a mock 'analysis' block so the AI has something to read
            "analysis": {
                "strengths": ["Chest Depth"],
                "weaknesses": ["Hip Sag", "Core instability"]
            }
        },
        {
            "session_id": "session_002",
            "created_at": "2024-01-16T10:00:00",
            "form_score": 82,
            "primary_fault": "Minor Elbow Flare",
            "analysis": {
                "strengths": ["Core Stability", "Tempo"],
                "weaknesses": ["Minor Elbow Flare"]
            }
        }
    ]

def main():
    # 1. Load Data
    print("Fetching user history from database...")
    history_data = get_mock_database_history()

    # 2. Run Meta-Analysis
    report = generate_weekly_report(history_data)

    # 3. Print The Dashboard
    print("\n" + "="*60)
    print(f"📊 WEEKLY PROGRESS REPORT")
    print("="*60)
    
    print(f"🔥 Current Streak: {report.current_streak_days} Days")
    print(f"🏋️ Total Workouts: {report.total_exercises_completed}")
    print(f"🏆 Best Workout: {report.best_workout_id}")
    print(f"   Reason: {report.best_workout_reason}\n")
    
    print("--- BODY PART BREAKDOWN ---")
    print(f"💪 ARMS: {report.body_part_breakdown.arms_analysis}")
    print(f"🦵 LEGS: {report.body_part_breakdown.legs_analysis}")
    print(f"🧩 CORE: {report.body_part_breakdown.core_analysis}\n")
    
    print("--- FUTURE PLAN (AI RECOMMENDATIONS) ---")
    for rec in report.recommended_plan:
        print(f"📌 {rec.exercise_name}")
        print(f"   Why: {rec.reasoning}")
        print(f"   Benefit: {rec.expected_benefit}\n")

    # 4. Save to JSON for Frontend
    with open("data/weekly_dashboard.json", "w") as f:
        f.write(report.model_dump_json(indent=2))
    print("Dashboard JSON saved to data/weekly_dashboard.json")

if __name__ == "__main__":
    main()