# VSLAM

### High-Level Architecture

A multi-camera SLAM system will have four major parts:
1. Calibration (extrinsic + intrinsic for all 4 cameras)
2. Time synchronization + frame synchronization
3. Front-end (feature extraction + matching/tracking)
4. Back-end (global optimization, loop closure, pose graph or factor graph)
5. Optional fusion with IMU / wheel odometry / LiDAR (if available)
