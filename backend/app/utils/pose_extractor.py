import torch
import cv2
from loguru import logger
import time

def load_yolov8_pose_model():
    model = YOLO("yolo26n-pose.pt")  # load an official model


def extract_pose_yolov8(frame, model):
    """Run YOLOv8 Pose on a single frame."""
    start_time = time.time()
    results = model(frame)  # Run inference on the frame
    
    # Extract keypoints from the YOLOv8 results
    keypoints = results.pandas().xywh[0][['keypoints']].values.tolist()
    end_time = time.time()
    
    logger.info(f"Pose extraction time: {end_time - start_time:.4f}s")
    return keypoints
