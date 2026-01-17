def calculate_score(user_features, ground_truth_features):
    """Compare user features with ground truth and calculate a score."""
    score = 0
    if abs(user_features['knee_angle'] - ground_truth_features['knee_angle']) <= 5:
        score += 10  # Perfect score for knee angle
    
    return score

def generate_feedback(score):
    """Generate actionable feedback based on the score."""
    if score == 10:
        return "Perfect squat form!"
    elif score >= 7:
        return "Good job, but try to deepen your squat slightly."
    else:
        return "Focus on reaching at least a 90-degree knee angle for optimal form."
