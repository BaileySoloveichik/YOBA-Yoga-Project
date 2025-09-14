# main.py
import json
from pose_detector import PoseDetector
from pose_utils import compute_all_angles, compute_all_angle_directions, compare_poses
from feature_extractor import FeatureExtractor

# Initialize pose detector and feature extractor
detector = PoseDetector()
extractor = FeatureExtractor()

# Choose an image filename from IMAGES folder
image_filename = "1.png"  # Replace with your image file

# Detect pose
results, keypoints, confidence = detector.detect_pose(image_filename)

# Compute angles and directions
angles = compute_all_angles(results)
directions = compute_all_angle_directions(results)

# Extract features
features = extractor.extract_features(keypoints, angles)

pose_name = "downward_dog"  # 👈 Replace manually before each run

# Load reference pose (angles + directions) from JSON
with open(f"Model/json_reference/{pose_name}_reference.json", "r") as f:
    ref_data = json.load(f)
angles_ref = ref_data["angles"]
directions_ref = ref_data["directions"]

# Print current detection
print("Detection confidence:", confidence)
print("Computed angles:", angles)
print("Computed angle directions:", directions)

# Compare with reference
print("\n### The pose compared to JSON reference ###")
fixes = compare_poses(angles, angles_ref, directions, directions_ref)
print(f"{len(fixes) == 0}\n")
print(fixes)

# Print features
print("Feature vector length:", len(features))
print("First 10 features:", features[:10])
