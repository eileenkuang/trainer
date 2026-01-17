import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

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
        # Check confidence for keypoints and set to NaN if below threshold
        def get_point(keypoint):
            conf = data.loc[i, f'{keypoint}_conf'] if f'{keypoint}_conf' in data.columns else 1.0
            if conf < CONFIDENCE_THRESHOLD:
                return np.array([np.nan, np.nan, np.nan])
            return np.array([data.loc[i, f'{keypoint}_x'], data.loc[i, f'{keypoint}_y'], data.loc[i, f'{keypoint}_z']])
        
        # Elbow angles
        left_shoulder = get_point('left_shoulder')
        left_elbow = get_point('left_elbow')
        left_wrist = get_point('left_wrist')
        angles['left_elbow'].append(calculate_angle(left_shoulder, left_elbow, left_wrist) if not np.isnan(left_elbow).any() else np.nan)
        
        right_shoulder = get_point('right_shoulder')
        right_elbow = get_point('right_elbow')
        right_wrist = get_point('right_wrist')
        angles['right_elbow'].append(calculate_angle(right_shoulder, right_elbow, right_wrist) if not np.isnan(right_elbow).any() else np.nan)
        
        # Knee angles
        left_hip = get_point('left_hip')
        left_knee = get_point('left_knee')
        left_ankle = get_point('left_ankle')
        angles['left_knee'].append(calculate_angle(left_hip, left_knee, left_ankle) if not np.isnan(left_knee).any() else np.nan)
        
        right_hip = get_point('right_hip')
        right_knee = get_point('right_knee')
        right_ankle = get_point('right_ankle')
        angles['right_knee'].append(calculate_angle(right_hip, right_knee, right_ankle) if not np.isnan(right_knee).any() else np.nan)
        
        # Hip angles (angle at hip: shoulder, hip, knee)
        angles['left_hip'].append(calculate_angle(left_shoulder, left_hip, left_knee) if not np.isnan(left_hip).any() else np.nan)
        angles['right_hip'].append(calculate_angle(right_shoulder, right_hip, right_knee) if not np.isnan(right_hip).any() else np.nan)
        
        # Ankle angles (angle at ankle: knee, ankle, foot_index)
        left_foot = get_point('left_foot_index')
        angles['left_ankle'].append(calculate_angle(left_knee, left_ankle, left_foot) if not np.isnan(left_ankle).any() else np.nan)
        
        right_foot = get_point('right_foot_index')
        angles['right_ankle'].append(calculate_angle(right_knee, right_ankle, right_foot) if not np.isnan(right_ankle).any() else np.nan)

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
            
            # Get x, y, z for current and next frame, check confidence
            for i in range(len(data) - 1):
                conf1 = data.loc[i, f'{keypoint}_conf'] if f'{keypoint}_conf' in data.columns else 1.0
                conf2 = data.loc[i+1, f'{keypoint}_conf'] if f'{keypoint}_conf' in data.columns else 1.0
                if conf1 < CONFIDENCE_THRESHOLD or conf2 < CONFIDENCE_THRESHOLD:
                    speeds[keypoint].append(np.nan)
                    signed_speeds_y[keypoint].append(np.nan)
                    continue
                pos1 = np.array([data.loc[i, f'{keypoint}_x'], data.loc[i, f'{keypoint}_y'], data.loc[i, f'{keypoint}_z']])
                pos2 = np.array([data.loc[i+1, f'{keypoint}_x'], data.loc[i+1, f'{keypoint}_y'], data.loc[i+1, f'{keypoint}_z']])
                distance = np.linalg.norm(pos2 - pos1)
                time_diff = (data.loc[i+1, 'timestamp_ms'] - data.loc[i, 'timestamp_ms']) / 1000  # in seconds
                speed = distance / time_diff if time_diff > 0 else 0
                speeds[keypoint].append(speed)
                
                # Signed y-velocity (negative if moving down, positive up)
                vel_y = (pos2[1] - pos1[1]) / time_diff if time_diff > 0 else 0
                signed_speeds_y[keypoint].append(vel_y)
            # For the last frame, append NaN if low confidence
            conf_last = data.loc[len(data)-1, f'{keypoint}_conf'] if f'{keypoint}_conf' in data.columns else 1.0
            speeds[keypoint].append(0 if conf_last >= CONFIDENCE_THRESHOLD else np.nan)
            signed_speeds_y[keypoint].append(0 if conf_last >= CONFIDENCE_THRESHOLD else np.nan)
            
            # Calculate acceleration as change in speed / time, skip if NaN
            for i in range(len(speeds[keypoint]) - 1):
                if np.isnan(speeds[keypoint][i]) or np.isnan(speeds[keypoint][i+1]):
                    accelerations[keypoint].append(np.nan)
                    continue
                time_diff = (data.loc[i+1, 'timestamp_ms'] - data.loc[i, 'timestamp_ms']) / 1000
                acc = (speeds[keypoint][i+1] - speeds[keypoint][i]) / time_diff if time_diff > 0 else 0
                accelerations[keypoint].append(acc)
            accelerations[keypoint].append(np.nan)  # last
    
    # Smooth speeds, skip NaN
    for key in speeds:
        valid_mask = ~np.isnan(speeds[key])
        if np.any(valid_mask):
            smoothed = smooth_data(np.array(speeds[key])[valid_mask], sigma=2)
            speeds[key] = np.full(len(speeds[key]), np.nan)
            speeds[key][valid_mask] = smoothed
        signed_speeds_y[key] = np.array(signed_speeds_y[key])  # No smoothing for signed, but could add if needed
    
    return speeds, accelerations, signed_speeds_y

# Detect rep start, end, and max depth
def detect_rep_points(displacement):
    tolerance = 0.01  # Tolerance for "near 0" displacement; adjust based on data scale
    
    # Find peaks in displacement (local maxima)
    peaks, _ = find_peaks(displacement, height=tolerance)
    
    if len(peaks) == 0:
        # Fallback to single rep
        max_depth_frame = np.argmax(displacement)
        peaks = [max_depth_frame]
    
    reps = []
    prev_end = -1
    for peak in peaks:
        max_depth_frame = peak
        
        # Start frame: search backwards from max_depth for the last frame where displacement is near 0
        start_frame = None
        for i in range(max_depth_frame - 1, prev_end, -1):  # Start from prev_end +1 to avoid overlap
            if abs(displacement.iloc[i]) < tolerance:
                start_frame = i
                break
        if start_frame is None:
            start_frame = prev_end + 1 if prev_end + 1 < max_depth_frame else max_depth_frame
        
        # End frame: search forwards from max_depth for the first frame where displacement is near 0
        end_frame = None
        for i in range(max_depth_frame, len(displacement)):
            if abs(displacement.iloc[i]) < tolerance:
                end_frame = i
                break
        if end_frame is None:
            end_frame = len(displacement) - 1
        
        reps.append({
            'start_frame': start_frame,
            'end_frame': end_frame,
            'max_depth_frame': max_depth_frame
        })
        prev_end = end_frame
    
    return reps

# Calculate symmetry score (based on distance between left and right body parts and angle differences)
def calculate_symmetry(data, angles):
    symmetry_scores = {}
    
    # Loop over each frame
    for i in range(len(data)):
        # Shoulder position symmetry, check confidence
        conf_ls = data.loc[i, 'left_shoulder_conf'] if 'left_shoulder_conf' in data.columns else 1.0
        conf_rs = data.loc[i, 'right_shoulder_conf'] if 'right_shoulder_conf' in data.columns else 1.0
        if conf_ls >= CONFIDENCE_THRESHOLD and conf_rs >= CONFIDENCE_THRESHOLD:
            left_shoulder = np.array([data.loc[i, 'left_shoulder_x'], data.loc[i, 'left_shoulder_y'], data.loc[i, 'left_shoulder_z']])
            right_shoulder = np.array([data.loc[i, 'right_shoulder_x'], data.loc[i, 'right_shoulder_y'], data.loc[i, 'right_shoulder_z']])
            symmetry_scores.setdefault('shoulder_position_symmetry', []).append(np.linalg.norm(left_shoulder - right_shoulder))
        else:
            symmetry_scores.setdefault('shoulder_position_symmetry', []).append(np.nan)
        
        # Angle symmetries, check if angles are NaN
        symmetry_scores.setdefault('elbow_angle_symmetry', []).append(abs(angles['left_elbow'][i] - angles['right_elbow'][i]) if not (np.isnan(angles['left_elbow'][i]) or np.isnan(angles['right_elbow'][i])) else np.nan)
        symmetry_scores.setdefault('knee_angle_symmetry', []).append(abs(angles['left_knee'][i] - angles['right_knee'][i]) if not (np.isnan(angles['left_knee'][i]) or np.isnan(angles['right_knee'][i])) else np.nan)
        symmetry_scores.setdefault('hip_angle_symmetry', []).append(abs(angles['left_hip'][i] - angles['right_hip'][i]) if not (np.isnan(angles['left_hip'][i]) or np.isnan(angles['right_hip'][i])) else np.nan)
        symmetry_scores.setdefault('ankle_angle_symmetry', []).append(abs(angles['left_ankle'][i] - angles['right_ankle'][i]) if not (np.isnan(angles['left_ankle'][i]) or np.isnan(angles['right_ankle'][i])) else np.nan)
    
    return symmetry_scores

# Visualization of progress (e.g., angles over time)
def plot_progress(data, angles, speeds, symmetry_scores, accelerations, angular_speeds, signed_speeds_y, reps):
    if not HAS_MATPLOTLIB:
        return
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    # Plot joint angles over time
    for angle_name in angles:
        ax[0,0].plot(data['timestamp_ms'], angles[angle_name], label=angle_name.replace('_', ' ').title())
    for i, rep in enumerate(reps):
        label_start = 'Rep Start' if i == 0 else ""
        label_end = 'Rep End' if i == 0 else ""
        label_max = 'Max Depth' if i == 0 else ""
        ax[0,0].axvline(data.loc[rep['start_frame'], 'timestamp_ms'], color='green', linestyle='--', label=label_start)
        ax[0,0].axvline(data.loc[rep['end_frame'], 'timestamp_ms'], color='red', linestyle='--', label=label_end)
        ax[0,0].axvline(data.loc[rep['max_depth_frame'], 'timestamp_ms'], color='blue', linestyle='--', label=label_max)
    ax[0,0].set_title("Joint Angles Over Time")
    ax[0,0].set_xlabel("Time (ms)")
    ax[0,0].set_ylabel("Angle (degrees)")
    ax[0,0].legend()

    # Plot signed y-speeds over time for elbows
    if 'left_elbow' in signed_speeds_y:
        ax[0,1].plot(data['timestamp_ms'], signed_speeds_y['left_elbow'], label='Left Elbow Signed Y Speed')
    for rep in reps:
        ax[0,1].axvline(data.loc[rep['start_frame'], 'timestamp_ms'], color='green', linestyle='--')
        ax[0,1].axvline(data.loc[rep['end_frame'], 'timestamp_ms'], color='red', linestyle='--')
        ax[0,1].axvline(data.loc[rep['max_depth_frame'], 'timestamp_ms'], color='blue', linestyle='--')
    ax[0,1].set_title("Signed Y-Velocity Over Time")
    ax[0,1].set_xlabel("Time (ms)")
    ax[0,1].set_ylabel("Velocity (units/s)")
    ax[0,1].legend()

    # Plot accelerations over time for key keypoints
    key_acc = ['left_hip', 'right_hip', 'left_knee', 'right_knee']
    for key in key_acc:
        if key in accelerations:
            ax[1,0].plot(data['timestamp_ms'], accelerations[key], label=key.replace('_', ' ').title() + ' Acc')
    for rep in reps:
        ax[1,0].axvline(data.loc[rep['start_frame'], 'timestamp_ms'], color='green', linestyle='--')
        ax[1,0].axvline(data.loc[rep['end_frame'], 'timestamp_ms'], color='red', linestyle='--')
        ax[1,0].axvline(data.loc[rep['max_depth_frame'], 'timestamp_ms'], color='blue', linestyle='--')
    ax[1,0].set_title("Accelerations Over Time")
    ax[1,0].set_xlabel("Time (ms)")
    ax[1,0].set_ylabel("Acceleration (units/s²)")
    ax[1,0].legend()

    # Plot symmetry scores over time
    for sym_name in symmetry_scores:
        ax[1,1].plot(data['timestamp_ms'], symmetry_scores[sym_name], label=sym_name.replace('_', ' ').title())
    for rep in reps:
        ax[1,1].axvline(data.loc[rep['start_frame'], 'timestamp_ms'], color='green', linestyle='--')
        ax[1,1].axvline(data.loc[rep['end_frame'], 'timestamp_ms'], color='red', linestyle='--')
        ax[1,1].axvline(data.loc[rep['max_depth_frame'], 'timestamp_ms'], color='blue', linestyle='--')
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
    
    # Extract joint angles (now with confidence filtering)
    angles = extract_joint_angles(data)
    
    # Calculate angular speeds (with NaN handling)
    angular_speeds = {}
    for angle_name in angles:
        angular_speeds[angle_name] = []
        for i in range(len(angles[angle_name]) - 1):
            if np.isnan(angles[angle_name][i]) or np.isnan(angles[angle_name][i+1]):
                angular_speeds[angle_name].append(np.nan)
                continue
            time_diff = (data.loc[i+1, 'timestamp_ms'] - data.loc[i, 'timestamp_ms']) / 1000
            ang_speed = (angles[angle_name][i+1] - angles[angle_name][i]) / time_diff if time_diff > 0 else 0
            angular_speeds[angle_name].append(ang_speed)
        angular_speeds[angle_name].append(np.nan)
    
    # Calculate speed and acceleration (now with confidence filtering)
    speeds, accelerations, signed_speeds_y = calculate_speed(data)
    
    # Calculate symmetry (now with confidence filtering)
    symmetry_scores = calculate_symmetry(data, angles)
    
    # Calculate displacement for rep detection
    left_elbow_y = data['left_elbow_y']
    right_elbow_y = data['right_elbow_y']
    elbow_y = (left_elbow_y + right_elbow_y) / 2
    displacement = elbow_y - elbow_y.iloc[0]
    
    # Detect rep points
    reps = detect_rep_points(displacement)
    
    # Add per-rep metrics
    for rep in reps:
        rep['max_depth_value'] = displacement.iloc[rep['max_depth_frame']]
        rep['avg_speed_left_elbow'] = np.mean(signed_speeds_y['left_elbow'][rep['start_frame']:rep['end_frame']])
        rep['avg_speed_right_elbow'] = np.mean(signed_speeds_y['right_elbow'][rep['start_frame']:rep['end_frame']])
        rep['rep_duration'] = (rep['end_frame'] - rep['start_frame']) * (data.loc[1, 'timestamp_ms'] - data.loc[0, 'timestamp_ms']) / 1000 if len(data) > 1 else 0
    
    # Plot the progress (angles, speed, symmetry, accelerations)
    plot_progress(data, angles, speeds, symmetry_scores, accelerations, angular_speeds, signed_speeds_y, reps)
    
    return angles, speeds, accelerations, symmetry_scores, angular_speeds, signed_speeds_y, reps

# New function to compare two output CSVs
def compare_csvs(csv1_path, csv2_path, output_path='pose_outputs/comparison_results.csv'):
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    
    # Assume same length; trim to min if needed
    min_len = min(len(df1), len(df2))
    df1 = df1.head(min_len)
    df2 = df2.head(min_len)
    
    comparison = {}
    angle_cols = [col for col in df1.columns if col in ['left_elbow', 'right_elbow', 'left_knee', 'right_knee', 'left_hip', 'right_hip', 'left_ankle', 'right_ankle']]
    for col in angle_cols + [col for col in df1.columns if '_speed' in col or '_acceleration' in col or '_angular_speed' in col]:
        if col in df1.columns and col in df2.columns:
            diff = np.abs(df1[col] - df2[col])
            comparison[f'{col}_mean_diff'] = np.nanmean(diff)
            comparison[f'{col}_max_diff'] = np.nanmax(diff)
    
    # Per-rep comparison if reps exist
    if 'rep_start_frame' in df1.columns and 'rep_end_frame' in df1.columns:
        rep_diffs = []
        for i in range(min_len):
            if not pd.isna(df1.loc[i, 'rep_start_frame']):
                start = int(df1.loc[i, 'rep_start_frame'])
                end = int(df1.loc[i, 'rep_end_frame'])
                if start < min_len and end < min_len:
                    rep_diff = np.nanmean([np.abs(df1.loc[j, 'left_elbow'] - df2.loc[j, 'left_elbow']) for j in range(start, end+1) if not pd.isna(df1.loc[j, 'left_elbow'])])
                    rep_diffs.append(rep_diff)
        comparison['mean_rep_angle_diff'] = np.nanmean(rep_diffs) if rep_diffs else np.nan
    
    comp_df = pd.DataFrame([comparison])
    comp_df.to_csv(output_path, index=False)
    
    # Visualize comparison: overlay key angles (e.g., left_elbow) from both CSVs
    if HAS_MATPLOTLIB and 'timestamp_ms' in df1.columns and 'left_elbow' in df1.columns and 'left_elbow' in df2.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df1['timestamp_ms'], df1['left_elbow'], label='CSV1 Left Elbow Angle', color='blue')
        plt.plot(df2['timestamp_ms'], df2['left_elbow'], label='CSV2 Left Elbow Angle', color='red', linestyle='--')
        plt.title("Comparison of Left Elbow Angles Over Time")
        plt.xlabel("Time (ms)")
        plt.ylabel("Angle (degrees)")
        plt.legend()
        # Add rep boundaries if available
        if 'rep_start_frame' in df1.columns:
            for i in range(min_len):
                if not pd.isna(df1.loc[i, 'rep_start_frame']):
                    start_time = df1.loc[int(df1.loc[i, 'rep_start_frame']), 'timestamp_ms']
                    end_time = df1.loc[int(df1.loc[i, 'rep_end_frame']), 'timestamp_ms']
                    max_time = df1.loc[int(df1.loc[i, 'max_depth_frame']), 'timestamp_ms']
                    plt.axvline(start_time, color='green', linestyle='--', alpha=0.7, label='Rep Start' if i == 0 else "")
                    plt.axvline(end_time, color='orange', linestyle='--', alpha=0.7, label='Rep End' if i == 0 else "")
                    plt.axvline(max_time, color='purple', linestyle='-', alpha=0.7, label='Max Depth' if i == 0 else "")
        plt.tight_layout()
        plt.show()  # Or save with plt.savefig('pose_outputs/comparison_plot.png')
    
    return comp_df

# Run the analysis on the data
angles, speeds, accelerations, symmetry_scores, angular_speeds, signed_speeds_y, reps = analyze(data)

# For CSV, use first rep if available
if reps:
    start_frame = reps[0]['start_frame']
    end_frame = reps[0]['end_frame']
    max_depth_frame = reps[0]['max_depth_frame']
else:
    start_frame = end_frame = max_depth_frame = 0

# Save results to CSV (now includes all angles and filtered values)
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

# Example usage of comparison (uncomment to run)
# compare_csvs('pose_outputs/rep_advanced_metrics.csv', 'pose_outputs/another_metrics.csv')
