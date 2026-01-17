import pandas as pd
import numpy as np
from analysis import analyze  # Assuming analysis.py is in the same directory

# Load ground truth and user data
gt_file = 'pose_outputs/biomechanical_data.csv'  # Ground truth CSV
user_file = 'pose_outputs/user_biomechanical_data.csv'  # User CSV

gt_data = pd.read_csv(gt_file)
user_data = pd.read_csv(user_file)

# Analyze both
print("Analyzing ground truth...")
gt_reps = analyze(gt_data)[6]  # reps is the 7th return value

print("Analyzing user data...")
user_reps = analyze(user_data)[6]

# Compute ground truth averages
if gt_reps:
    gt_avg_depth = np.mean([rep['max_depth_value'] for rep in gt_reps])
    gt_avg_speed_left = np.mean([rep['avg_speed_left_elbow'] for rep in gt_reps])
    gt_avg_speed_right = np.mean([rep['avg_speed_right_elbow'] for rep in gt_reps])
    gt_avg_duration = np.mean([rep['rep_duration'] for rep in gt_reps])
    gt_std_depth = np.std([rep['max_depth_value'] for rep in gt_reps])
    gt_std_speed_left = np.std([rep['avg_speed_left_elbow'] for rep in gt_reps])
    gt_std_speed_right = np.std([rep['avg_speed_right_elbow'] for rep in gt_reps])
    gt_std_duration = np.std([rep['rep_duration'] for rep in gt_reps])
else:
    print("No reps detected in ground truth.")
    exit()

# Compare user reps to ground truth averages
feedback = []
for i, rep in enumerate(user_reps):
    diff_depth = rep['max_depth_value'] - gt_avg_depth
    diff_speed_left = rep['avg_speed_left_elbow'] - gt_avg_speed_left
    diff_speed_right = rep['avg_speed_right_elbow'] - gt_avg_speed_right
    diff_duration = rep['rep_duration'] - gt_avg_duration
    
    rep_feedback = f"Rep {i+1}: "
    if abs(diff_depth) > gt_std_depth:
        rep_feedback += f"Depth {'too deep' if diff_depth > 0 else 'not deep enough'} (diff: {diff_depth:.3f}). "
    if abs(diff_speed_left) > gt_std_speed_left:
        rep_feedback += f"Left elbow speed {'too fast' if diff_speed_left > 0 else 'too slow'} (diff: {diff_speed_left:.3f}). "
    if abs(diff_speed_right) > gt_std_speed_right:
        rep_feedback += f"Right elbow speed {'too fast' if diff_speed_right > 0 else 'too slow'} (diff: {diff_speed_right:.3f}). "
    if abs(diff_duration) > gt_std_duration:
        rep_feedback += f"Duration {'too long' if diff_duration > 0 else 'too short'} (diff: {diff_duration:.3f}). "
    
    if not rep_feedback.endswith(": "):
        feedback.append(rep_feedback)
    else:
        feedback.append(f"Rep {i+1}: Good form.")

# Overall summary
overall_depth_diff = np.mean([rep['max_depth_value'] for rep in user_reps]) - gt_avg_depth
overall_speed_diff = np.mean([rep['avg_speed_left_elbow'] for rep in user_reps]) - gt_avg_speed_left
overall_duration_diff = np.mean([rep['rep_duration'] for rep in user_reps]) - gt_avg_duration

print("Ground Truth Averages:")
print(f"Avg Depth: {gt_avg_depth:.3f}, Std: {gt_std_depth:.3f}")
print(f"Avg Left Speed: {gt_avg_speed_left:.3f}, Std: {gt_std_speed_left:.3f}")
print(f"Avg Right Speed: {gt_avg_speed_right:.3f}, Std: {gt_std_speed_right:.3f}")
print(f"Avg Duration: {gt_avg_duration:.3f}, Std: {gt_std_duration:.3f}")

print("\nUser Rep Feedback:")
for fb in feedback:
    print(fb)

print(f"\nOverall: Depth diff {overall_depth_diff:.3f}, Speed diff {overall_speed_diff:.3f}, Duration diff {overall_duration_diff:.3f}")