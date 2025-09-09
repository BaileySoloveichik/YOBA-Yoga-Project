# main.py
from pose_detector import PoseDetector
from pose_utils import compute_all_angles, compute_all_angle_directions, compare_poses
from feature_extractor import FeatureExtractor

# Initialize pose detector and feature extractor
detector = PoseDetector()
extractor = FeatureExtractor()

# Choose an image filename from IMAGES folder
image_filename = "1.png"  # Replace with your image file
image_filename2 = "2.png"  # Replace with your image file 2

# Detect pose
results, keypoints, confidence = detector.detect_pose(image_filename)
results2, keypoints2, confidence2 = detector.detect_pose(image_filename2)

# Compute angles
angles = compute_all_angles(results)
directions = compute_all_angle_directions(results)

angles2 = compute_all_angles(results2)
directions2 = compute_all_angle_directions(results2)

# Extract features
features = extractor.extract_features(keypoints, angles)
features2 = extractor.extract_features(keypoints2, angles2)

# Print results
print("Detection confidence:", confidence)
print("Computed angles:", angles)
print("Computed angle directions:", directions)

print("Computed angles pose 2:", angles2)
print("Computed angle directions pose 2:", directions2)

print("\n### The two poses are equal? ###")
fixes = compare_poses(angles, angles2, directions, directions2)
print(f"{len(fixes)==0}\n")
print(fixes)

print("Feature vector length:", len(features))
print("First 10 features:", features[:10])