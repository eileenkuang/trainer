from fastapi import FastAPI, UploadFile, File
import os
from backend.app.video_processing.video_handler import extract_frames
from backend.app.utils.pose_extractor import load_yolov8_pose_model, extract_pose_yolov8
from loguru import logger

app = FastAPI()

# Mount static folder
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Initialize YOLOv8 model once at the start
model = load_yolov8_pose_model()

@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    # Save the uploaded video file
    video_path = os.path.join("/tmp", file.filename)
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())

    # Step 1: Extract frames from the video
    frames = extract_frames(video_path)
    
    logger.info(f"Starting pose extraction for {len(frames)} frames.")

    # Step 2: Extract pose keypoints from each frame
    all_keypoints = []
    for idx, frame in enumerate(frames):
        keypoints = extract_pose_yolov8(frame, model)
        logger.info(f"Frame {idx + 1}: {len(keypoints)} keypoints detected")
        all_keypoints.append(keypoints)
    
    # Return keypoints data (just for testing purposes)
    return {"frame_count": len(frames), "keypoints_per_frame": all_keypoints}
