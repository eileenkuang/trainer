from pydantic import BaseModel, Field
from typing import List, Literal

# --- 1. Basic Analysis Schemas ---

class RemedialExercise(BaseModel):
    name: str = Field(..., description="Name of the exercise.")
    target_fault: str = Field(..., description="The error this exercise fixes.")
    prescription: str = Field(..., description="Sets and Reps.")
    adaptive_reasoning: str = Field(..., description="Why this fits the user's profile.")

class GamificationStats(BaseModel):
    form_score: int = Field(..., description="0-100 score based on biomechanics.")
    xp_earned: int = Field(..., description="Points for consistency and improvement.")
    current_level_title: str = Field(..., description="E.g., 'Stability Rookie'.")
    streak_bonus: bool = Field(..., description="True if they got extra XP.")

class VideoEvent(BaseModel):
    start_time: float
    end_time: float
    overlay_text: str = Field(..., max_length=25)
    detailed_explanation: str
    type: Literal["strength", "weakness"]
    severity: Literal["low", "medium", "high"]

class AnalysisResult(BaseModel):
    user_skill_level: Literal["Beginner", "Intermediate", "Advanced"]
    personalized_summary: str
    gamification: GamificationStats
    strengths: List[str]
    weaknesses: List[str]
    remedial_plan: List[RemedialExercise]
    timeline_events: List[VideoEvent]

# --- 2. Meta-Analysis (Weekly Report) Schemas ---

class BodyPartSummary(BaseModel):
    arms_analysis: str = Field(..., description="Summary of arm stability, strength, and form trends.")
    legs_analysis: str = Field(..., description="Summary of leg drive, stability, and stance trends.")
    core_analysis: str = Field(..., description="Summary of midline stability and bracing trends.")

class FutureRecommendation(BaseModel):
    exercise_name: str
    reasoning: str = Field(..., description="Why chosen: 'High Frequency', 'Fixes Weakness', etc.")
    expected_benefit: str

class WeeklyReport(BaseModel):
    # This is the field your runner was missing!
    body_part_breakdown: BodyPartSummary
    
    recommended_plan: List[FutureRecommendation]
    
    # Hard Stats
    current_streak_days: int
    total_exercises_completed: int
    best_workout_id: str
    best_workout_reason: str