import numpy as np

def calculate_angle(a, b, c):
    """Calculate the angle between three points (joint angle)."""
    ab = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

def extract_squat_features(keypoints):
    """Extract features for squat movement."""
    # Example: keypoints indices for squat
    hip = keypoints[23]
    knee = keypoints[25]
    ankle = keypoints[27]
    
    # Calculate knee angle
    knee_angle = calculate_angle(hip, knee, ankle)
    
    return {"knee_angle": knee_angle}

def extract_pushup_features(keypoints):
    """Extract features for push-up movement."""
    shoulder = keypoints[11]
    elbow = keypoints[13]
    wrist = keypoints[15]
    
    # Calculate elbow angle
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    
    return {"elbow_angle": elbow_angle}

def calculate_speed(keypoints, frame_rate, rep_start_frame, rep_end_frame):
    """Calculate speed of movement (rep speed) between two keyframes."""
    start_point = keypoints[0]  # This would be a specific body joint like the hip
    end_point = keypoints[-1]   # Example: another point for end position
    distance = np.linalg.norm(np.array(end_point) - np.array(start_point))
    time_taken = (rep_end_frame - rep_start_frame) / frame_rate
    speed = distance / time_taken
    return speed
