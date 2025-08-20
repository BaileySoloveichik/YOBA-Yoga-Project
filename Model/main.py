# main.py
from pose_detector import PoseDetector
from pose_utils import compute_all_angles
from feature_extractor import FeatureExtractor

# Initialize pose detector and feature extractor
detector = PoseDetector()
extractor = FeatureExtractor()

# Choose an image filename from IMAGES folder
image_filename = "1.jpg"  # Replace with your image file

# Detect pose
results, keypoints, confidence = detector.detect_pose(image_filename)

# Compute angles
angles = compute_all_angles(results)

# Extract features
features = extractor.extract_features(keypoints, angles)

# Print results
print("Detection confidence:", confidence)
print("Computed angles:", angles)
print("Feature vector length:", len(features))
print("First 10 features:", features[:10])
