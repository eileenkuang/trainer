import json 
from dotenv import load_dotenv
from src.llm_engine import generate_analysis_script 
from src.video_overlay import render_video 
from src.database_handler import save_session_to_db
from src.schemas import AnalysisResult
load_dotenv() 

def main(): 
    # Loading Data 
    print("Loading metrics...") 
    with open ("data/final_rep_analysis.json", "r") as f: 
        metrics_data = f.read() 

    with open("config/cue_bank.json", "r") as f: 
        cue_bank = f.read() 

    # Get LLM to fill out output
    print("Analyzing form and generating script...") 
    analysis_result = generate_analysis_script(metrics_data, cue_bank)  

    # Save the result as the final JSON Script 
    output_filename = "data/final_analysis.json" 
    with open(output_filename, "w") as f: 
        f.write(analysis_result.model_dump_json(indent=2))
    
    print(f"Analysis Complete, saved to {output_filename}")
    print(f"Diagnosed Skill Level: {analysis_result.user_skill_level}")
    
    # --- NEW: RUN VIDEO RENDERER ---
    from src.video_overlay import render_video
    
    # Define paths
    # Note: If 'input_video.mp4' doesn't exist, it will create a black video test.
    raw_video = "data/input_video.mp4" 
    json_script = "data/final_analysis.json"
    final_video = "data/annotated_output.mp4"

    render_video(raw_video, json_script, final_video)
    
    with open("data/final_analysis.json", "r") as f: 
        data = json.load(f)

    print("\n--- Saving to Cloud ---")
    try: 
        save_session_to_db(AnalysisResult(**data), video_name="demo_pushup.mp4")
    except Exception as e: 
        print(f" Database Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()