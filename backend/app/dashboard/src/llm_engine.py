import os
from dotenv import load_dotenv
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

# 1. SETUP CLIENT
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY is missing from .env file")

client = instructor.from_openai(
    OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.getenv("GEMINI_API_KEY")
    ),
    mode=instructor.Mode.JSON
)

# 2. DATA SCHEMAS
class RepetitionAnalysis(BaseModel):
    timestamp_start: float
    timestamp_end: float
    rep_id: int
    deviation_score: float
    primary_metric: str = Field(..., description="Clean name of the issue e.g. 'Knee Stability'")
    metric_value: str = Field(..., description="Specific numbers e.g. '145 deg' or 'N/A'")
    description: str = Field(..., description="A 1-sentence critique.")

class VideoAnalysis(BaseModel):
    video_id: str
    fps: int = 30
    comparison_metrics: List[RepetitionAnalysis]

# 3. GENERATION LOGIC
def generate_rep_analysis_json(rep_summaries: list) -> VideoAnalysis:
    
    # --- FIXED PROMPT TO MATCH YOUR DEBUG DATA ---
    prompt = f"""
    You are an expert Biomechanics AI Coach.
    I will provide raw data from a user's workout.
    
    INPUT DATA KEYS:
    - "primary_metric_raw": The body part involved.
    - "metric_value_raw" or "all_errors": The technical details.

    YOUR GOAL:
    Convert the raw data into a clean JSON timeline.

    RULES:
    1. **Description:** - If `deviation_score` is 1.0 (like in the input), be constructive but firm: "Significant deviation detected."
       - Use the `all_errors` list to describe *what* happened (e.g. "Knee angle was 145 degrees, expected 162").
    
    2. **Metric Extraction:**
       - Use the numbers from `metric_value_raw` or `all_errors`. 
       - Example: "145.35 < GT 162.12" -> metric_value: "145° vs 162°"

    INPUT DATA:
    {rep_summaries}
    """

    print("   ... 🧠 Gemini is analyzing rep data...")

    try:
        response = client.chat.completions.create(
            model="gemini-2.5-flash", 
            response_model=VideoAnalysis,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return response

    except Exception as e:
        print(f"❌ LLM Generation Error: {e}")
        return VideoAnalysis(video_id="error", comparison_metrics=[])