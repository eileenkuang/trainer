from loguru import logger

def log_video_processing(video_path, total_frames):
    """Log telemetry for video processing."""
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Total frames to process: {total_frames}")

def log_pose_detection(frame_idx, keypoints_detected):
    """Log telemetry for pose detection."""
    logger.info(f"Frame {frame_idx}: Detected {len(keypoints_detected)} keypoints")

def log_feature_extraction(features):
    """Log telemetry for feature extraction."""
    logger.info(f"Extracted features: {features}")
