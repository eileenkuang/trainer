import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Load the data
file_path = 'pose_outputs/biomechanical_data.csv'
data = pd.read_csv(file_path)

CONFIDENCE_THRESHOLD = 0.7  # Adjust as needed

# Define smoothing function to apply to x, y, z coordinates
def smooth_data(data, sigma=2):
    return gaussian_filter1d(data, sigma=sigma, axis=0)

# Normalize function: using distance between shoulders and feet to estimate height
def estimate_height(data):
    # Use mean positions for height estimation
    left_shoulder = np.array([data['left_shoulder_x'].mean(), data['left_shoulder_y'].mean(), data['left_shoulder_z'].mean()])
    right_shoulder = np.array([data['right_shoulder_x'].mean(), data['right_shoulder_y'].mean(), data['right_shoulder_z'].mean()])
    left_foot = np.array([data['left_foot_index_x'].mean(), data['left_foot_index_y'].mean(), data['left_foot_index_z'].mean()])
    right_foot = np.array([data['right_foot_index_x'].mean(), data['right_foot_index_y'].mean(), data['right_foot_index_z'].mean()])

    # Distance between shoulders and feet to estimate height
    shoulder_to_foot = np.linalg.norm(left_shoulder - left_foot) + np.linalg.norm(right_shoulder - right_foot)
    height = shoulder_to_foot / 2  # Rough average height estimate
    return height

# Normalize coordinates based on estimated height
def normalize_coordinates(data, height):
    for column in data.columns:
        if 'x' in column or 'y' in column or 'z' in column:
            data[column] = data[column] / height
    return data

# Calculate angle between three points
def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Extract joint angles: Example for elbow (shoulder, elbow, wrist)
def extract_joint_angles(data):
    angles = {
        'left_elbow': [],
        'right_elbow': [],
        'left_knee': [],
        'right_knee': [],
        'left_hip': [],
        'right_hip': [],
        'left_ankle': [],
        'right_ankle': []
    }
    
    # Loop over each frame in the dataset
    for i in range(len(data)):
        # Elbow angles
        left_shoulder = np.array([data.loc[i, 'left_shoulder_x'], data.loc[i, 'left_shoulder_y'], data.loc[i, 'left_shoulder_z']])
        left_elbow = np.array([data.loc[i, 'left_elbow_x'], data.loc[i, 'left_elbow_y'], data.loc[i, 'left_elbow_z']])
        left_wrist = np.array([data.loc[i, 'left_wrist_x'], data.loc[i, 'left_wrist_y'], data.loc[i, 'left_wrist_z']])
        angles['left_elbow'].append(calculate_angle(left_shoulder, left_elbow, left_wrist))
        
        right_shoulder = np.array([data.loc[i, 'right_shoulder_x'], data.loc[i, 'right_shoulder_y'], data.loc[i, 'right_shoulder_z']])
        right_elbow = np.array([data.loc[i, 'right_elbow_x'], data.loc[i, 'right_elbow_y'], data.loc[i, 'right_elbow_z']])
        right_wrist = np.array([data.loc[i, 'right_wrist_x'], data.loc[i, 'right_wrist_y'], data.loc[i, 'right_wrist_z']])
        angles['right_elbow'].append(calculate_angle(right_shoulder, right_elbow, right_wrist))
        
        # Knee angles
        left_hip = np.array([data.loc[i, 'left_hip_x'], data.loc[i, 'left_hip_y'], data.loc[i, 'left_hip_z']])
        left_knee = np.array([data.loc[i, 'left_knee_x'], data.loc[i, 'left_knee_y'], data.loc[i, 'left_knee_z']])
        left_ankle = np.array([data.loc[i, 'left_ankle_x'], data.loc[i, 'left_ankle_y'], data.loc[i, 'left_ankle_z']])
        angles['left_knee'].append(calculate_angle(left_hip, left_knee, left_ankle))
        
        right_hip = np.array([data.loc[i, 'right_hip_x'], data.loc[i, 'right_hip_y'], data.loc[i, 'right_hip_z']])
        right_knee = np.array([data.loc[i, 'right_knee_x'], data.loc[i, 'right_knee_y'], data.loc[i, 'right_knee_z']])
        right_ankle = np.array([data.loc[i, 'right_ankle_x'], data.loc[i, 'right_ankle_y'], data.loc[i, 'right_ankle_z']])
        angles['right_knee'].append(calculate_angle(right_hip, right_knee, right_ankle))
        
        # Hip angles (angle at hip: shoulder, hip, knee)
        angles['left_hip'].append(calculate_angle(left_shoulder, left_hip, left_knee))
        angles['right_hip'].append(calculate_angle(right_shoulder, right_hip, right_knee))
        
        # Ankle angles (angle at ankle: knee, ankle, foot_index)
        left_foot = np.array([data.loc[i, 'left_foot_index_x'], data.loc[i, 'left_foot_index_y'], data.loc[i, 'left_foot_index_z']])
        angles['left_ankle'].append(calculate_angle(left_knee, left_ankle, left_foot))
        
        right_foot = np.array([data.loc[i, 'right_foot_index_x'], data.loc[i, 'right_foot_index_y'], data.loc[i, 'right_foot_index_z']])
        angles['right_ankle'].append(calculate_angle(right_knee, right_ankle, right_foot))

    return angles

# Calculate speed of keypoints as 3D Euclidean distance
def calculate_speed(data):
    speeds = {}
    for column in data.columns:
        if '_x' in column:  # Only process x columns to avoid duplicates
            keypoint = column.replace('_x', '')  # Extract keypoint name
            
            if keypoint not in speeds:
                speeds[keypoint] = []
            
            # Get x, y, z for current and next frame
            for i in range(len(data) - 1):
                pos1 = np.array([data.loc[i, f'{keypoint}_x'], data.loc[i, f'{keypoint}_y'], data.loc[i, f'{keypoint}_z']])
                pos2 = np.array([data.loc[i+1, f'{keypoint}_x'], data.loc[i+1, f'{keypoint}_y'], data.loc[i+1, f'{keypoint}_z']])
                distance = np.linalg.norm(pos2 - pos1)
                time_diff = (data.loc[i+1, 'timestamp_ms'] - data.loc[i, 'timestamp_ms']) / 1000  # in seconds
                speed = distance / time_diff if time_diff > 0 else 0
                speeds[keypoint].append(speed)
            # For the last frame, append 0
            speeds[keypoint].append(0)
    
    return speeds

# Calculate symmetry score (based on distance between left and right body parts)
def calculate_symmetry(data):
    symmetry_scores = {}
    
    # Loop over each frame
    for i in range(len(data)):
        # Example: Comparing left and right shoulders
        left_shoulder = np.array([data.loc[i, 'left_shoulder_x'], data.loc[i, 'left_shoulder_y'], data.loc[i, 'left_shoulder_z']])
        right_shoulder = np.array([data.loc[i, 'right_shoulder_x'], data.loc[i, 'right_shoulder_y'], data.loc[i, 'right_shoulder_z']])
        
        symmetry_scores.setdefault('shoulder_symmetry', []).append(np.linalg.norm(left_shoulder - right_shoulder))
    
    return symmetry_scores

# Visualization of progress (e.g., angles over time)
def plot_progress(data, angles, speeds, symmetry_scores):
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    # Plot joint angles over time
    for angle_name in angles:
        ax[0,0].plot(data['timestamp_ms'], angles[angle_name], label=angle_name.replace('_', ' ').title())
    ax[0,0].set_title("Joint Angles Over Time")
    ax[0,0].set_xlabel("Time (ms)")
    ax[0,0].set_ylabel("Angle (degrees)")
    ax[0,0].legend()

    # Plot speeds over time for key keypoints
    key_speeds = ['left_wrist', 'right_wrist', 'left_ankle', 'right_ankle']
    for key in key_speeds:
        if key in speeds:
            ax[0,1].plot(data['timestamp_ms'], speeds[key], label=key.replace('_', ' ').title())
    ax[0,1].set_title("Speeds Over Time")
    ax[0,1].set_xlabel("Time (ms)")
    ax[0,1].set_ylabel("Speed (units/s)")
    ax[0,1].legend()

    # Plot symmetry score over time
    ax[1,0].plot(data['timestamp_ms'], symmetry_scores['shoulder_symmetry'], label='Shoulder Symmetry')
    ax[1,0].set_title("Symmetry Over Time")
    ax[1,0].set_xlabel("Time (ms)")
    ax[1,0].set_ylabel("Symmetry Distance (units)")
    ax[1,0].legend()

    # Plot average confidence over time
    vis_columns = [col for col in data.columns if 'vis' in col]
    if vis_columns:
        avg_confidence = data[vis_columns].mean(axis=1)
        ax[1,1].plot(data['timestamp_ms'], avg_confidence, label='Average Confidence')
        ax[1,1].set_title("Average Keypoint Confidence Over Time")
        ax[1,1].set_xlabel("Time (ms)")
        ax[1,1].set_ylabel("Confidence")
        ax[1,1].legend()

    plt.tight_layout()
    plt.show()

# Main analysis function
def analyze(data):
    # Smooth data
    for col in ['x', 'y', 'z']:
        for keypoint in [k for k in data.columns if col in k]:
            data[keypoint] = smooth_data(data[keypoint])
    
    # Estimate height dynamically based on shoulder-to-feet distance
    height = estimate_height(data)
    
    # Normalize coordinates based on height
    data = normalize_coordinates(data, height)
    
    # Extract joint angles
    angles = extract_joint_angles(data)
    
    # Calculate speed
    speeds = calculate_speed(data)
    
    # Calculate symmetry
    symmetry_scores = calculate_symmetry(data)
    
    # Plot the progress (angles, speed, symmetry)
    plot_progress(data, angles, speeds, symmetry_scores)
    
    return angles, speeds, symmetry_scores

# Run the analysis on the data
angles, speeds, symmetry_scores = analyze(data)

# Save results to CSV
results = pd.DataFrame({
    'frame': data['frame'],
    'timestamp_ms': data['timestamp_ms'],
    **angles,
    **{f'{k}_speed': v for k, v in speeds.items()},
    **symmetry_scores
})
results.to_csv('pose_outputs/rep_advanced_metrics.csv', index=False)
