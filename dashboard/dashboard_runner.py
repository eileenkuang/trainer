import json 
import os 
from dotenv import load_dotenv
from src.meta_engine import generate_weekly_report 
from src.db_connector import fetch_user_history 

load_dotenv() 

def main (): 
    TARGET_USER_ID = "76f963f6-4a5b-4248-bbe2-69122852a87b"
    print(f"--- STARTING AI ANALYSIS FOR USER: {TARGET_USER_ID} ---")

    history_data = fetch_user_history(TARGET_USER_ID)

    if not history_data: 
        print("Stoping analysis due to missing data") 
        return 
    
    report = generate_weekly_report(history_data) 

    # 4. Output the Results
    print("\n" + "="*60)
    print(f"📊 LIVE DASHBOARD REPORT")
    print("="*60)
    
    print(f"🔥 Streak: {report.current_streak_days} Days")
    print(f"🏆 Best Workout: {report.best_workout_id}")
    print(f"📝 AI Summary: {report.body_part_breakdown.core_analysis}")

    # 5. Save for Frontend
    output_path = "data/weekly_dashboard.json"
    with open(output_path, "w") as f:
        f.write(report.model_dump_json(indent=2))
    print(f"\n✅ Dashboard JSON updated at: {output_path}")

if __name__ == "__main__":
    main()