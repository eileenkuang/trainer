import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import re
import subprocess
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from twelvelabs import TwelveLabs
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
VIDEO_PATH = './data_temp/IMG_2221.MOV'
OUTPUT_FOLDER = './pose_outputs'
MODEL_PATH = './mediapipe/pose_landmarker_lite.task'
CSV_OUTPUT = os.path.join(OUTPUT_FOLDER, 'biomechanical_data.csv')

TWELVELABS_API_KEY = os.getenv("TWELVELABS_API_KEY")

# MediaPipe Configuration
MIN_CONFIDENCE = 0.4
SMOOTHING_WINDOW_SIZE = 15 

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", 
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left", 
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", 
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip", 
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", 
    "right_heel", "left_foot_index", "right_foot_index"
]

class PushUpTimestampDetector:
    """Uses TwelveLabs Pegasus to find rep boundaries."""
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("TWELVELABS_API_KEY not found in environment.")
        self.client = TwelveLabs(api_key=api_key)
        self.index_id = None
    
    def setup_and_get_timestamps(self, video_path):
        # Create Index
        print("Creating TwelveLabs index...")
        index = self.client.indexes.create(
            index_name="Pushup-Analysis-Task",
            models=[{"model_name": "pegasus1.2", "model_options": ["visual"]}]
        )
        self.index_id = index.id

        # Upload
        print(f"Uploading video: {video_path}")
        task = self.client.tasks.create(index_id=self.index_id, video_file=video_path)
        task.wait_for_done(sleep_interval=5)
        
        # Pegasus Analysis
        print("TwelveLabs Pegasus is identifying repetitions...")
        prompt = """Identify every single push-up repetition in this video. 
        For each rep, provide the exact start (descending) and end (fully returned) timestamps.
        Format your response exactly like this:
        [start_time] - [end_time] rep
        Example: [0:02.1] - [0:04.5] rep"""

        result = self.client.generate.text(video_id=task.video_id, prompt=prompt)
        return self._parse_timestamps(result.data)

    def _parse_timestamps(self, text):
        # Pattern to extract [M:SS.S] or [SS.S]
        pattern = r'\[(\d+:)?(\d+\.?\d*)\]'
        matches = re.finditer(pattern, text)
        timestamps = []
        for m in matches:
            mins = int(m.group(1).replace(':', '')) if m.group(1) else 0
            secs = float(m.group(2))
            timestamps.append(mins * 60 + secs)
        
        # Group timestamps into pairs (Start, End)
        pairs = []
        for i in range(0, len(timestamps) - 1, 2):
            pairs.append({'start': timestamps[i], 'end': timestamps[i+1]})
        
        print(f"Detected {len(pairs)} push-up repetitions.")
        return pairs

def smooth_landmarks(history):
    """Simple moving average smoothing logic."""
    if not history: return None
    smoothed = []
    num_landmarks = len(history[0])
    for i in range(num_landmarks):
        x = np.mean([frame[i][0] for frame in history])
        y = np.mean([frame[i][1] for frame in history])
        z = np.mean([frame[i][2] for frame in history])
        v = np.mean([frame[i][3] for frame in history])
        smoothed.append((x, y, z, v))
    return smoothed

def get_confidence_color(confidence):
    """Green for high confidence, Red for low."""
    return (0, int(255 * confidence), int(255 * (1 - confidence)))

def extract_segment(video_path, start, end, out_path):
    """Cuts the video using FFmpeg for targeted MediaPipe analysis."""
    duration = end - start
    cmd = [
        "ffmpeg", "-ss", str(max(0, start - 0.2)), # Small buffer
        "-i", video_path, 
        "-t", str(duration + 0.4), 
        "-c:v", "libx264", "-avoid_negative_ts", "make_zero", "-y", out_path
    ]
    subprocess.run(cmd, capture_output=True)

def analyze_segment(segment_path, rep_idx, absolute_start_sec):
    """Processes a single rep segment with MediaPipe Tasks."""
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=MIN_CONFIDENCE,
        min_pose_presence_confidence=MIN_CONFIDENCE,
        min_tracking_confidence=MIN_CONFIDENCE
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(segment_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    segment_data = []
    history = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        timestamp_ms = int(1000 * frame_idx / fps)
        # Calculate the actual timestamp relative to the original full video
        abs_timestamp_ms = int((absolute_start_sec * 1000) + timestamp_ms)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        row = [frame_idx, abs_timestamp_ms]

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            current_frame_pts = [(lm.x, lm.y, lm.z, lm.visibility) for lm in landmarks]
            
            # Smoothing Logic
            history.append(current_frame_pts)
            if len(history) > SMOOTHING_WINDOW_SIZE:
                history.pop(0)
            
            smoothed = smooth_landmarks(history)
            
            for pt in smoothed:
                row.extend(pt)
                # Visualization: Circle colored by confidence
                cx, cy = int(pt[0] * width), int(pt[1] * height)
                cv2.circle(frame, (cx, cy), 5, get_confidence_color(pt[3]), -1)
                cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1)
        else:
            # If no landmarks found, fill with NaNs
            row.extend([np.nan] * (len(LANDMARK_NAMES) * 4))

        segment_data.append(row)
        
        # HUD
        cv2.putText(frame, f"REPETITION #{rep_idx}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("AI Trainer Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        frame_idx += 1

    detector.close()
    cap.release()
    return segment_data

def run_analysis():
    """Main pipeline: TwelveLabs Detection -> Segment Extraction -> MediaPipe Processing."""
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 1. TwelveLabs Step
    tl_detector = PushUpTimestampDetector(TWELVELABS_API_KEY)
    rep_times = tl_detector.setup_and_get_timestamps(VIDEO_PATH)
    
    if not rep_times:
        print("Pegasus could not identify any push-up repetitions.")
        return

    # 2. MediaPipe Step
    all_rows = []
    print(f"Beginning biomechanical extraction for {len(rep_times)} reps...")

    for i, ts in enumerate(rep_times, 1):
        seg_file = os.path.join(OUTPUT_FOLDER, f"rep_segment_{i}.mp4")
        extract_segment(VIDEO_PATH, ts['start'], ts['end'], seg_file)
        
        # Analyze the cut segment
        rep_rows = analyze_segment(seg_file, i, ts['start'])
        all_rows.extend(rep_rows)
        
        # Clean up temporary segments if desired
        if os.path.exists(seg_file):
            os.remove(seg_file)

    # 3. CSV Saving
    with open(CSV_OUTPUT, 'w', newline='') as f:
        writer = csv.writer(f)
        # Create Header
        header = ['frame', 'timestamp_ms']
        for name in LANDMARK_NAMES:
            header.extend([f'{name}_x', f'{name}_y', f'{name}_z', f'{name}_vis'])
        writer.writerow(header)
        # Write Data
        writer.writerows(all_rows)

    cv2.destroyAllWindows()
    print(f"\nAnalysis complete. Biomechanical data saved to: {CSV_OUTPUT}")
    print("You can now run 'analysis.py' to generate advanced metrics and charts.")

if __name__ == "__main__":
    run_analysis()
