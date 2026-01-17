from pydantic import BaseModel, Field 
from typing import List, Literal

# Event with Timestamp in the video 
class VideoEvent(BaseModel): 
    start_time: float = Field(..., description="Start time of Event")
    end_time: float = Field(..., description="End time of Event") 
    overlay_text : str=Field(..., max_length=50, description="Text to annotate over video")
    detailed_explanation: str  = Field(..., description="Full sentence explanation")
    correction_cue: str = Field(..., description="Specific cue to fix the problem") 
    status_color: Literal["green", "red", "yellow"]

class AnalysisResult(BaseModel): 
    user_skill_level: Literal["Beginner", "Intermediate", "Advanced"] = Field(..., description="Infered skill level")
    general_summary: str = Field(..., description = "A holistic view across the whole video")
    timeline_events: List[VideoEvent] = Field(..., description="List of events and descriptions")

class BodyPartSummary(BaseModel): 
    arms_analysis: str = Field(..., description = "Summary of arm stability, strength, and form trends. ") 
    legs_analysis: str = Field(..., description = "Summary of leg positioning, balance, and alignment trends. ")
    core_analysis: str = Field(..., description = "Summary of core engagement, posture, and stability trends. ")
    overall_summary: str = Field(..., description = "Holistic summary of body part performances during exercise. ")

class FutureRecommendations(BaseModel): 
    exercise_name: str
    reasoning: str = Field(..., description="Rationale for recommending this exercise.")
    expected_benefit: str

class WeeklyReport(BaseModel): 
    body_part_summary: BodyPartSummary = Field(..., description="Detailed analysis of different body parts.")
    recommendations: List[FutureRecommendations] = Field(..., description="List of recommended exercises for improvement.")

    current_streak_days: int
    total_exercises_completed: int 
    best_workout_id: str = Field(..., description="Identifier for the best workout session." )
    best_workout_reason: str = Field(..., description="Reason why this workout was the best.")