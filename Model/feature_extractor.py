# feature_extractor.py
import numpy as np

class FeatureExtractor:
    def __init__(self):
        pass

    def extract_features(self, keypoints, angles):
        """
        Generate a simple feature vector from keypoints and angles.
        - keypoints: numpy array (33,4) [x,y,z,visibility]
        - angles: dictionary with angle names and values

        Returns: 1D numpy array
        """
        features = []

        # Flatten keypoints (replace None with 0 just in case)
        if keypoints is not None:
            flat_keypoints = keypoints.flatten()
            features.extend(flat_keypoints)
        else:
            features.extend([0.0]*33*4)

        # Append angles in the same order as ANGLE_NAMES
        if angles is not None:
            for angle_name in [
                "left_elbow_angle", "right_elbow_angle",
                "left_shoulder_angle", "right_shoulder_angle",
                "left_knee_angle", "right_knee_angle",
                "left_hip_angle", "right_hip_angle"
            ]:
                val = angles.get(angle_name)
                features.append(val if val is not None else 0.0)
        else:
            features.extend([0.0]*8)

        return np.array(features, dtype=np.float32)
