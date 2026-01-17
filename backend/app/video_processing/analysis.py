import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from filterpy.kalman import KalmanFilter

# Load the data
file_path = './pose_outputs/biomechanical_data.csv'
data = pd.read_csv(file_path)

# Function to apply Kalman filtering to smooth the data
def apply_kalman_filter(data, sigma=1):
    kf = KalmanFilter(dim_x=3, dim_z=1)
    smoothed_data = np.zeros_like(data)
    for i in range(len(data)):
        kf.predict()
        kf.update(data[i])
        smoothed_data[i] = kf.x[0]
    return smoothed_data

# Dynamic height estimation using shoulders to feet distance
def estimate_height(data):
    left_shoulder = np.array([data['left_shoulder_x'], data['left_shoulder_y'], data['left_shoulder_z']])
    right_shoulder = np.array([data['right_shoulder_x'], data['right_shoulder_y'], data['right_shoulder_z']])
    left_foot = np.array([data['left_foot_index_x'], data['left_foot_index_y'], data['left_foot_index_z']])
    right_foot = np.array([data['right_foot_index_x'], data['right_foot_index_y'], data['right_foot_index_z']])

    # Shoulder-to-foot distance
    shoulder_to_foot = np.linalg.norm(left_shoulder - left_foot) + np.linalg.norm(right_shoulder - right_foot)
    height = shoulder_to_foot / 2  # Averaged height
    return height

# Normalize coordinates by height (dynamic scaling)
def normalize_coordinates(data, height):
    for column in data.columns:
        if 'x' in column or 'y' in column or 'z' in column:
            data[column] = data[column] / height
    return data

# Function to calculate the 3D Euclidean distance between two points
def calculate_3d_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

# Calculate the angle between three points
def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Extract joint angles for each frame, e.g., elbow, knee, wrist, etc.
def extract_joint_angles(data):
    angles = {}
    
    # Loop through each frame to calculate joint angles
    for i in range(len(data)):
        # Left elbow angle
        left_shoulder = np.array([data.loc[i, 'left_shoulder_x'], data.loc[i, 'left_shoulder_y'], data.loc[i, 'left_shoulder_z']])
        left_elbow = np.array([data.loc[i, 'left_elbow_x'], data.loc[i, 'left_elbow_y'], data.loc[i, 'left_elbow_z']])
        left_wrist = np.array([data.loc[i, 'left_wrist_x'], data.loc[i, 'left_wrist_y'], data.loc[i, 'left_wrist_z']])
        angles['left_elbow'] = calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        # Left knee angle
        left_hip = np.array([data.loc[i, 'left_hip_x'], data.loc[i, 'left_hip_y'], data.loc[i, 'left_hip_z']])
        left_knee = np.array([data.loc[i, 'left_knee_x'], data.loc[i, 'left_knee_y'], data.loc[i, 'left_knee_z']])
        left_ankle = np.array([data.loc[i, 'left_ankle_x'], data.loc[i, 'left_ankle_y'], data.loc[i, 'left_ankle_z']])
        angles['left_knee'] = calculate_angle(left_hip, left_knee, left_ankle)

    return angles

# Calculate movement speed and acceleration for each keypoint (x, y, z)
def calculate_speed(data):
    speeds = {}
    for column in data.columns:
        if 'x' in column or 'y' in column or 'z' in column:
            keypoint = column.split('_')[0]
            if keypoint not in speeds:
                speeds[keypoint] = {'x': [], 'y': [], 'z': []}
            
            # Calculate speed (delta position / delta time)
            speeds[keypoint]['x'].append(data[column].diff() / (data['timestamp_ms'].diff() / 1000))
            speeds[keypoint]['y'].append(data[column.replace('x', 'y')].diff() / (data['timestamp_ms'].diff() / 1000))
            speeds[keypoint]['z'].append(data[column.replace('x', 'z')].diff() / (data['timestamp_ms'].diff() / 1000))

    return speeds

# Calculate symmetry between left and right body parts
def calculate_symmetry(data):
    symmetry_scores = {}
    # Example: Comparing shoulders
    left_shoulder = np.array([data['left_shoulder_x'], data['left_shoulder_y'], data['left_shoulder_z']])
    right_shoulder = np.array([data['right_shoulder_x'], data['right_shoulder_y'], data['right_shoulder_z']])
    symmetry_scores['shoulder_symmetry'] = np.linalg.norm(left_shoulder - right_shoulder)

    # Add more body parts comparisons as needed (e.g., elbows, knees)
    return symmetry_scores

# Plot joint angles, speed, and symmetry
def plot_progress(data, angles, speeds, symmetry_scores):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Plot joint angles
    ax[0].plot(data['timestamp_ms'], angles['left_elbow'], label='Left Elbow Angle')
    ax[0].plot(data['timestamp_ms'], angles['left_knee'], label='Left Knee Angle')
    ax[0].set_title("Joint Angles Over Time")
    ax[0].set_xlabel("Time (ms)")
    ax[0].set_ylabel("Angle (degrees)")
    ax[0].legend()

    # Plot speed over time for keypoints
    ax[1].plot(data['timestamp_ms'], speeds['left_wrist']['x'], label='Left Wrist Speed X')
    ax[1].set_title("Speed Over Time (X axis)")
    ax[1].set_xlabel("Time (ms)")
    ax[1].set_ylabel("Speed (m/s)")

    # Plot symmetry score over time
    ax[2].plot(data['timestamp_ms'], [symmetry_scores['shoulder_symmetry']] * len(data), label='Shoulder Symmetry')
    ax[2].set_title("Symmetry Over Time")
    ax[2].set_xlabel("Time (ms)")
    ax[2].set_ylabel("Symmetry Distance (m)")

    plt.tight_layout()
    plt.show()

# Main analysis function
def analyze(data):
    # Apply Kalman filter for smoothing
    for col in ['x', 'y', 'z']:
        for keypoint in [k for k in data.columns if col in k]:
            data[keypoint] = apply_kalman_filter(data[keypoint].values)
    
    # Estimate height dynamically based on keypoints
    height = estimate_height(data)
    
    # Normalize coordinates by height
    data = normalize_coordinates(data, height)
    
    # Extract joint angles
    angles = extract_joint_angles(data)
    
    # Calculate speed
    speeds = calculate_speed(data)
    
    # Calculate symmetry
    symmetry_scores = calculate_symmetry(data)
    
    # Plot the progress
    plot_progress(data, angles, speeds, symmetry_scores)

    return angles, speeds, symmetry_scores

# Run the analysis
angles, speeds, symmetry_scores = analyze(data)
