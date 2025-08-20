# Yoga Pose JSON Files

This folder contains predefined yoga pose constraints in JSON format.  
Each file describes the rules and angle ranges for a specific pose.

## File Structure

Each pose JSON file follows this schema:

```json
{
  "pose_name": "string",              // The name of the pose
  "side": "left|right|front|back",    // Orientation of the body
  "angles": [                         // List of angle constraints
    {
      "joints": ["joint1", "joint2", "joint3"], // The 3 keypoints used
      "range": [min_angle, max_angle]           // Allowed range in degrees
    }
  ],
  "relations": [                      // (Optional) Relations between body parts
    {
      "from": "jointA",
      "to": "jointB",
      "relation": "above|below|aligned|left_of|right_of"
    }
  ]
}
```

Example
{
"pose_name": "Warrior",
"side": "left",
"angles": [
{ "joints": ["hip", "knee", "ankle"], "range": [160, 180] },
{ "joints": ["shoulder", "elbow", "wrist"], "range": [150, 180] }
],
"relations": [
{ "from": "left_hand", "to": "left_shoulder", "relation": "above" }
]
}

Files in this folder

chair.json

tree.json

triangle.json

warrior.json

index.json (metadata of all poses)
