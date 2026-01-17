from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    confidence: float
    note: str
