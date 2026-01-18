import os 
from dotenv import load_dotenv
import instructor 
from openai import OpenAI 
from .schemas import AnalysisResult 

load_dotenv()

# Initialize the client with Instructor patches
client = instructor.from_openai(
    OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.getenv("GEMINI_API_KEY") 
    ), 
    mode=instructor.Mode.JSON
)
def generate_analysis_script(metrics_json: str, cue_bank_text: str) -> AnalysisResult: 
    """
    Converts raw metrics to structured Analysis output for the website to use.
    """

    response = client.chat.completions.create( 
        model="gemini-2.5-flash", 
        response_model=AnalysisResult, 
        messages=[
            {
                "role": "system", 
                "content": (
                    "You are an expert biomechanics and fitness coach. " 
                    "1. Analyze the input metrics and timestamps. " 
                    "2. Determine the user's skill level (Beginner=High deviation, Advanced=Low deviation). "
                    "3. Create a timeline of feedback events. " 
                    "4. For each event, write a SHORT text for the video overlay and DETAILED text for the website."
                    "5. TAILOR THE TONE: Encouraging/Simple for Beginners; Technical/Direct for Advanced."
                )
            }, 
            {
                "role": "user", 
                "content": f"""
                METRICS DATA: 
                {metrics_json}

                GUIDELINES / CUE BANK: 
                {cue_bank_text}
                """
            }
        ]
    )
    return response