import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

# Angle names to compute
ANGLE_NAMES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle"
]

# Mapping body landmarks to Mediapipe PoseLandmark
LANDMARK_MAP = {
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
    "left_knee": mp_pose.PoseLandmark.LEFT_KNEE,
    "right_knee": mp_pose.PoseLandmark.RIGHT_KNEE,
    "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE
}

def calculate_angle(pA, pB, pC):
    """
    Calculate the angle (in degrees) between three points pA, pB, pC.
    The angle is between vectors BA and BC, with B as the central point.
    """
    a = np.array(pA)
    b = np.array(pB)
    c = np.array(pC)

    ba = a - b
    bc = c - b

    # Compute angle in degrees
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Prevent numerical errors
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def safe_calculate_angle(pA, pB, pC):
    """
    Safe wrapper for calculate_angle.
    Returns None if data is invalid (e.g., missing point).
    """
    try:
        return calculate_angle(pA, pB, pC)
    except Exception:
        return None

def compute_all_angles(results):
    """
    Given Mediapipe results, return a dictionary:
    {
        "left_elbow_angle": 145.2,
        "right_elbow_angle": 167.3,
        ...
    }
    """
    if results is None or not hasattr(results, "pose_landmarks") or results.pose_landmarks is None:
        return {name: None for name in ANGLE_NAMES}

    lm = results.pose_landmarks.landmark

    def get_point(name):
        if name not in LANDMARK_MAP:
            return None
        idx = LANDMARK_MAP[name].value
        landmark = lm[idx]
        return (landmark.x, landmark.y)

    angles = {}

    # Elbows
    angles["left_elbow_angle"] = safe_calculate_angle(
        get_point("left_shoulder"), get_point("left_elbow"), get_point("left_wrist")
    )
    angles["right_elbow_angle"] = safe_calculate_angle(
        get_point("right_shoulder"), get_point("right_elbow"), get_point("right_wrist")
    )

    # Shoulders
    angles["left_shoulder_angle"] = safe_calculate_angle(
        get_point("left_elbow"), get_point("left_shoulder"), get_point("left_hip")
    )
    angles["right_shoulder_angle"] = safe_calculate_angle(
        get_point("right_elbow"), get_point("right_shoulder"), get_point("right_hip")
    )

    # Knees
    angles["left_knee_angle"] = safe_calculate_angle(
        get_point("left_hip"), get_point("left_knee"), get_point("left_ankle")
    )
    angles["right_knee_angle"] = safe_calculate_angle(
        get_point("right_hip"), get_point("right_knee"), get_point("right_ankle")
    )

    # Hips
    angles["left_hip_angle"] = safe_calculate_angle(
        get_point("left_shoulder"), get_point("left_hip"), get_point("left_knee")
    )
    angles["right_hip_angle"] = safe_calculate_angle(
        get_point("right_shoulder"), get_point("right_hip"), get_point("right_knee")
    )

    return angles
