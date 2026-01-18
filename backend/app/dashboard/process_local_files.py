import json 
import os 
from dotenv import load_dotenv 
from src.csv_parser import parse_rep_data
from src.llm_engine import generate_rep_analysis_json

load_dotenv() 

CSV_PATH = r"C:\Users\Happy\Desktop\uthacks\trainer\backend\app\pose_outputs\user_comparison.csv"
TXT_PATH = r"C:\Users\Happy\Desktop\uthacks\trainer\backend\app\pose_outputs\final_summary.txt"
OUTPUT_JSON_PATH = "data/final_rep_analysis.json"

def main(): 
    print("Starting rep-by-rep analysis...")

    if not os.path.exists(CSV_PATH):
        print(f"Error: File not found at {CSV_PATH}")
        return False
    
    rep_summaries = parse_rep_data(CSV_PATH, TXT_PATH)
    print(f"DEBUG: Data being sent to AI: {json.dumps(rep_summaries, indent=2)}") 
    # ----------------------------
    if not rep_summaries: 
        print("No rep data found in CSV, exiting")
        return False
    
    print(f"Parsed {len(rep_summaries)} repetitions. ")

    try: 
        final_analysis = generate_rep_analysis_json(rep_summaries)
        
        # Inject standard metadata
        final_analysis.video_id = "user_test_session_01"
        final_analysis.fps = 30
        
        # 4. Save to File
        with open(OUTPUT_JSON_PATH, "w") as f:
            f.write(final_analysis.model_dump_json(indent=2))
            
        print(f"\n SUCCESS! JSON saved to: {OUTPUT_JSON_PATH}")
        print("   (You can now load this file in your Frontend timeline)")
        
    except Exception as e:
        print(f"\n AI Generation Failed: {e}")
        return False 
    
    return True 

if __name__ == "__main__":
    main() 