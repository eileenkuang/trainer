import cv2
import mediapipe as mp
import numpy as np
import os
import csv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Configuration ---
VIDEO_PATH = './data_temp/IMG_2222.MOV'
OUTPUT_FOLDER = './pose_outputs'
# NOTE: Ensure you have the 'pose_landmarker_full.task' (Level 1) in this path
MODEL_PATH = './mediapipe/pose_landmarker_lite.task' 
CSV_OUTPUT = os.path.join(OUTPUT_FOLDER, 'biomechanical_data.csv')

# TwelveLabs Specs: Complexity Level 1 is the "Full" model
MIN_CONFIDENCE = 0.5

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", 
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left", 
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", 
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip", 
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", 
    "right_heel", "left_foot_index", "right_foot_index"
]

def get_confidence_color(confidence):
    """Maps 0.0-1.0 confidence to BGR (Red -> Green)."""
    green = int(255 * confidence)
    red = int(255 * (1 - confidence))
    return (0, green, red)

def run_analysis():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # NEW: Initialize MediaPipe Tasks API
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO, # optimized for sequential frames
        min_pose_detection_confidence=MIN_CONFIDENCE,
        min_pose_presence_confidence=MIN_CONFIDENCE,
        min_tracking_confidence=MIN_CONFIDENCE
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with open(CSV_OUTPUT, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['frame', 'timestamp_ms']
        for name in LANDMARK_NAMES:
            header.extend([f'{name}_x', f'{name}_y', f'{name}_z', f'{name}_vis'])
        writer.writerow(header)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            timestamp_ms = int(1000 * frame_idx / fps)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            # Detect landmarks for current video frame
            result = detector.detect_for_video(mp_image, timestamp_ms)

            row = [frame_idx, timestamp_ms]
            
            if result.pose_landmarks:
                # We take the first detected person (TwelveLabs logic)
                landmarks = result.pose_landmarks[0]
                
                for i, lm in enumerate(landmarks):
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])

                    # Visualization: Colored by Confidence
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    color = get_confidence_color(lm.visibility)
                    cv2.circle(frame, (cx, cy), 5, color, -1)
                    cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1)
            else:
                row.extend([np.nan] * (len(LANDMARK_NAMES) * 4))

            # HUD Display
            cv2.putText(frame, f"MODE: TASKS_V1 (Video)", (20, 40), 1, 1.5, (0, 255, 0), 2)
            
            writer.writerow(row)
            cv2.imshow("Multi-Sport Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            frame_idx += 1

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_analysis()