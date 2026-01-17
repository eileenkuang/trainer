from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

# Mount static folder
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Upload folder - relative to this file's directory
UPLOAD_FOLDER = Path(__file__).parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

@app.get("/api/ping")
def ping():
    return {"message": "backend is alive"}

@app.post("/api/save-video")
async def save_video(file: UploadFile = File(...)):
    try:
        # Generate unique filename if file already exists
        file_location = UPLOAD_FOLDER / file.filename
        counter = 1
        original_name = file_location.stem
        extension = file_location.suffix
        while file_location.exists():
            file_location = UPLOAD_FOLDER / f"{original_name}_{counter}{extension}"
            counter += 1
        
        # Save the file
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