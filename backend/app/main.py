from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
import sys
import json

app = FastAPI()

# 1. Setup Upload Folder
UPLOAD_FOLDER = Path(__file__).parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Add the app directory to the path so we can import test_analysis
sys.path.insert(0, str(Path(__file__).parent))

@app.get("/api/ping")
def ping():
    return {"message": "backend is alive"}

@app.post("/api/save-video") 
async def save_video(file: UploadFile = File(...)):
    try:
        file_location = UPLOAD_FOLDER / file.filename
        counter = 1
        original_name = file_location.stem
        extension = file_location.suffix
        while file_location.exists():
            file_location = UPLOAD_FOLDER / f"{original_name}_{counter}{extension}"
            counter += 1
        
        with open(file_location, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return {
            "status": "saved",
            "filename": file_location.name,
            "path": str(file_location)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/analyze-video")
async def analyze_video(file: UploadFile = File(...), exercise: str = Form(...)):
    try:
        # Save the uploaded video file
        file_location = UPLOAD_FOLDER / file.filename
        counter = 1
        original_name = file_location.stem
        extension = file_location.suffix
        while file_location.exists():
            file_location = UPLOAD_FOLDER / f"{original_name}_{counter}{extension}"
            counter += 1
        
        with open(file_location, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Import and call the analyze function from test_analysis
        from test_analysis import analyze
        
        # Call analyze with the video path
        video_path = str(file_location)
        result = analyze(video_path)
        
        # Read final_analysis.json and extract general_summary
        final_analysis_path = Path(__file__).parent / "dashboard" / "data" / "final_analysis.json"
        general_summary = ""
        
        if final_analysis_path.exists():
            try:
                with open(final_analysis_path, "r") as f:
                    analysis_data = json.load(f)
                    general_summary = analysis_data.get("general_summary", "")
            except Exception as e:
                general_summary = f"Error reading final_analysis.json: {str(e)}"
        else:
            general_summary = "final_analysis.json not found"
        
        # Check if annotated_output.mp4 exists
        annotated_video_path = Path(__file__).parent / "dashboard" / "data" / "annotated_output.mp4"
        annotated_video_url = ""
        if annotated_video_path.exists():
            annotated_video_url = "/api/videos/annotated_output.mp4"
        
        return {
            "status": "success",
            "message": "Video analyzed successfully",
            "exercise": exercise,
            "video_path": video_path,
            "result": result,
            "general_summary": general_summary,
            "annotated_video_url": annotated_video_url
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }

# Mount static files for dashboard data (videos, etc.)
dashboard_data_path = Path(__file__).parent / "dashboard" / "data"
app.mount("/api/videos", StaticFiles(directory=str(dashboard_data_path)), name="videos")

app.mount("/", StaticFiles(directory="static", html=True), name="static")
