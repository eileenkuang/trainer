import json 
import os 
from dotenv import load_dotenv 
# Ensure these imports match your folder structure (dashboard/src)
from dashboard.src.csv_parser import parse_rep_data
from dashboard.src.llm_engine_process import generate_rep_analysis_json

load_dotenv() 

# --- PATHS ---
# Adjust these absolute paths if you move folders
CSV_PATH = "pose_outputs/user_comparison.csv"
TXT_PATH = "pose_outputs/final_summary.txt"
OUTPUT_JSON_PATH = "dashboard/data/final_rep_analysis.json"
CUE_BANK_PATH = "dashboard/config/cue_bank.json"


def main(): 
    print("Starting rep-by-rep analysis...")

    # 1. Load CSV Data
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV File not found at {CSV_PATH}")
        return False
    
    rep_summaries = parse_rep_data(CSV_PATH, TXT_PATH)
    
    if not rep_summaries: 
        print("No rep data found in CSV, exiting")
        return False
    
    print(f"Parsed {len(rep_summaries)} repetitions.")
    # print(f"DEBUG: Data being sent to AI: {json.dumps(rep_summaries, indent=2)}") 

    # 2. Load Cue Bank
    cue_bank = []
    if os.path.exists(CUE_BANK_PATH):
        try:
            with open(CUE_BANK_PATH, "r") as f:
                cue_bank = json.load(f)
            print(f"Loaded {len(cue_bank)} cues from bank.")
        except Exception as e:
            print(f"Warning: Could not load cue bank ({e}). AI will have to improvise.")
    else:
        print(f"Warning: Cue Bank file not found at {CUE_BANK_PATH}")

    # 3. Generate Analysis with AI
    try: 
        # --- UPDATED CALL: Passing both data AND cue_bank ---
        final_analysis = generate_rep_analysis_json(rep_summaries, cue_bank)
        
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