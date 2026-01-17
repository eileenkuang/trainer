import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

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

# Calculate speed and acceleration of keypoints as 3D Euclidean distance
def calculate_speed(data):
    speeds = {}
    accelerations = {}
    signed_speeds_y = {}  # Signed y-velocity
    for column in data.columns:
        if '_x' in column:  # Only process x columns to avoid duplicates
            keypoint = column.replace('_x', '')  # Extract keypoint name
            
            if keypoint not in speeds:
                speeds[keypoint] = []
                accelerations[keypoint] = []
                signed_speeds_y[keypoint] = []
            
            # Get x, y, z for current and next frame
            for i in range(len(data) - 1):
                pos1 = np.array([data.loc[i, f'{keypoint}_x'], data.loc[i, f'{keypoint}_y'], data.loc[i, f'{keypoint}_z']])
                pos2 = np.array([data.loc[i+1, f'{keypoint}_x'], data.loc[i+1, f'{keypoint}_y'], data.loc[i+1, f'{keypoint}_z']])
                distance = np.linalg.norm(pos2 - pos1)
                time_diff = (data.loc[i+1, 'timestamp_ms'] - data.loc[i, 'timestamp_ms']) / 1000  # in seconds
                speed = distance / time_diff if time_diff > 0 else 0
                speeds[keypoint].append(speed)
                
                # Signed y-velocity (negative if moving down, positive up)
                vel_y = (pos2[1] - pos1[1]) / time_diff if time_diff > 0 else 0
                signed_speeds_y[keypoint].append(vel_y)
            # For the last frame, append 0
            speeds[keypoint].append(0)
            signed_speeds_y[keypoint].append(0)
            
            # Calculate acceleration as change in speed / time
            for i in range(len(speeds[keypoint]) - 1):
                acc = (speeds[keypoint][i+1] - speeds[keypoint][i]) / time_diff if time_diff > 0 else 0
                accelerations[keypoint].append(acc)
            accelerations[keypoint].append(0)  # last
    
    # Smooth speeds
    for key in speeds:
        speeds[key] = smooth_data(np.array(speeds[key]), sigma=2).tolist()
        signed_speeds_y[key] = smooth_data(np.array(signed_speeds_y[key]), sigma=2).tolist()
    
    return speeds, accelerations, signed_speeds_y

# Detect rep start, end, and max depth
def detect_rep_points(displacement):
    tolerance = 0.01  # Tolerance for "near 0" displacement; adjust based on data scale
    
    # Max depth: frame with maximum displacement (deepest point)
    max_depth_frame = np.argmax(displacement)
    
    # Start frame: search backwards from max_depth for the last frame where displacement is near 0
    start_frame = None
    for i in range(max_depth_frame - 1, -1, -1):
        if abs(displacement.iloc[i]) < tolerance:
            start_frame = i
            break
    if start_frame is None:
        start_frame = 0  # Fallback to start if no near-0 found
    
    # End frame: search forwards from max_depth for the first frame where displacement is near 0
    end_frame = None
    for i in range(max_depth_frame, len(displacement)):
        if abs(displacement.iloc[i]) < tolerance:
            end_frame = i
            break
    if end_frame is None:
        end_frame = len(displacement) - 1  # Fallback to end if no return to 0
    
    return start_frame, end_frame, max_depth_frame

# Calculate symmetry score (based on distance between left and right body parts and angle differences)
def calculate_symmetry(data, angles):
    symmetry_scores = {}
    
    # Loop over each frame
    for i in range(len(data)):
        # Shoulder position symmetry
        left_shoulder = np.array([data.loc[i, 'left_shoulder_x'], data.loc[i, 'left_shoulder_y'], data.loc[i, 'left_shoulder_z']])
        right_shoulder = np.array([data.loc[i, 'right_shoulder_x'], data.loc[i, 'right_shoulder_y'], data.loc[i, 'right_shoulder_z']])
        
        symmetry_scores.setdefault('shoulder_position_symmetry', []).append(np.linalg.norm(left_shoulder - right_shoulder))
        
        # Angle symmetries
        symmetry_scores.setdefault('elbow_angle_symmetry', []).append(abs(angles['left_elbow'][i] - angles['right_elbow'][i]))
        symmetry_scores.setdefault('knee_angle_symmetry', []).append(abs(angles['left_knee'][i] - angles['right_knee'][i]))
        symmetry_scores.setdefault('hip_angle_symmetry', []).append(abs(angles['left_hip'][i] - angles['right_hip'][i]))
        symmetry_scores.setdefault('ankle_angle_symmetry', []).append(abs(angles['left_ankle'][i] - angles['right_ankle'][i]))
    
    return symmetry_scores

# Visualization of progress (e.g., angles over time)
def plot_progress(data, angles, speeds, symmetry_scores, accelerations, angular_speeds, signed_speeds_y, start_frame, end_frame, max_depth_frame):
    if not HAS_MATPLOTLIB:
        return
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    # Plot joint angles over time
    for angle_name in angles:
        ax[0,0].plot(data['timestamp_ms'], angles[angle_name], label=angle_name.replace('_', ' ').title())
    ax[0,0].axvline(data.loc[start_frame, 'timestamp_ms'], color='green', linestyle='--', label='Rep Start')
    ax[0,0].axvline(data.loc[end_frame, 'timestamp_ms'], color='red', linestyle='--', label='Rep End')
    ax[0,0].axvline(data.loc[max_depth_frame, 'timestamp_ms'], color='blue', linestyle='--', label='Max Depth')
    ax[0,0].set_title("Joint Angles Over Time")
    ax[0,0].set_xlabel("Time (ms)")
    ax[0,0].set_ylabel("Angle (degrees)")
    ax[0,0].legend()

    # Plot signed y-speeds over time for elbows
    if 'left_elbow' in signed_speeds_y:
        ax[0,1].plot(data['timestamp_ms'], signed_speeds_y['left_elbow'], label='Left Elbow Signed Y Speed')
    ax[0,1].axvline(data.loc[start_frame, 'timestamp_ms'], color='green', linestyle='--')
    ax[0,1].axvline(data.loc[end_frame, 'timestamp_ms'], color='red', linestyle='--')
    ax[0,1].axvline(data.loc[max_depth_frame, 'timestamp_ms'], color='blue', linestyle='--')
    ax[0,1].set_title("Signed Y-Velocity Over Time")
    ax[0,1].set_xlabel("Time (ms)")
    ax[0,1].set_ylabel("Velocity (units/s)")
    ax[0,1].legend()

    # Plot accelerations over time for key keypoints
    key_acc = ['left_hip', 'right_hip', 'left_knee', 'right_knee']
    for key in key_acc:
        if key in accelerations:
            ax[1,0].plot(data['timestamp_ms'], accelerations[key], label=key.replace('_', ' ').title() + ' Acc')
    ax[1,0].axvline(data.loc[start_frame, 'timestamp_ms'], color='green', linestyle='--')
    ax[1,0].axvline(data.loc[end_frame, 'timestamp_ms'], color='red', linestyle='--')
    ax[1,0].axvline(data.loc[max_depth_frame, 'timestamp_ms'], color='blue', linestyle='--')
    ax[1,0].set_title("Accelerations Over Time")
    ax[1,0].set_xlabel("Time (ms)")
    ax[1,0].set_ylabel("Acceleration (units/s²)")
    ax[1,0].legend()

    # Plot symmetry scores over time
    for sym_name in symmetry_scores:
        ax[1,1].plot(data['timestamp_ms'], symmetry_scores[sym_name], label=sym_name.replace('_', ' ').title())
    ax[1,1].axvline(data.loc[start_frame, 'timestamp_ms'], color='green', linestyle='--')
    ax[1,1].axvline(data.loc[end_frame, 'timestamp_ms'], color='red', linestyle='--')
    ax[1,1].axvline(data.loc[max_depth_frame, 'timestamp_ms'], color='blue', linestyle='--')
    ax[1,1].set_title("Symmetry Over Time")
    ax[1,1].set_xlabel("Time (ms)")
    ax[1,1].set_ylabel("Symmetry (units or degrees)")
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
    
    # Calculate angular speeds
    angular_speeds = {}
    for angle_name in angles:
        angular_speeds[angle_name] = []
        for i in range(len(angles[angle_name]) - 1):
            time_diff = (data.loc[i+1, 'timestamp_ms'] - data.loc[i, 'timestamp_ms']) / 1000
            ang_speed = (angles[angle_name][i+1] - angles[angle_name][i]) / time_diff if time_diff > 0 else 0
            angular_speeds[angle_name].append(ang_speed)
        angular_speeds[angle_name].append(0)
    
    # Calculate speed and acceleration
    speeds, accelerations, signed_speeds_y = calculate_speed(data)
    
    # Calculate symmetry
    symmetry_scores = calculate_symmetry(data, angles)
    
    # Calculate displacement for rep detection
    left_elbow_y = data['left_elbow_y']
    right_elbow_y = data['right_elbow_y']
    elbow_y = (left_elbow_y + right_elbow_y) / 2
    displacement = elbow_y - elbow_y.iloc[0]
    
    # Detect rep points
    start_frame, end_frame, max_depth_frame = detect_rep_points(displacement)
    
    # Plot the progress (angles, speed, symmetry, accelerations)
    plot_progress(data, angles, speeds, symmetry_scores, accelerations, angular_speeds, signed_speeds_y, start_frame, end_frame, max_depth_frame)
    
    return angles, speeds, accelerations, symmetry_scores, angular_speeds, signed_speeds_y, start_frame, end_frame, max_depth_frame

# Run the analysis on the data
angles, speeds, accelerations, symmetry_scores, angular_speeds, signed_speeds_y, start_frame, end_frame, max_depth_frame = analyze(data)

# Save results to CSV
results = pd.DataFrame({
    'frame': data['frame'],
    'timestamp_ms': data['timestamp_ms'],
    **angles,
    **{f'{k}_speed': v for k, v in speeds.items()},
    **{f'{k}_acceleration': v for k, v in accelerations.items()},
    **{f'{k}_angular_speed': v for k, v in angular_speeds.items()},
    **{f'{k}_signed_speed_y': v for k, v in signed_speeds_y.items()},
    **symmetry_scores,
    'rep_start_frame': start_frame,
    'rep_end_frame': end_frame,
    'max_depth_frame': max_depth_frame
})
results.to_csv('pose_outputs/rep_advanced_metrics.csv', index=False)
