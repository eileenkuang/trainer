from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
import sys

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
        
        return {
            "status": "success",
            "message": "Video analyzed successfully",
            "exercise": exercise,
            "video_path": video_path,
            "result": result
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }

app.mount("/", StaticFiles(directory="static", html=True), name="static")
