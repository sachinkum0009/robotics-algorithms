from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
from base_vslam import BaseVSLAM
from cv2 import BFMatcher, UMat
from cv2.typing import MatLike
from jax import numpy as jnp
from tqdm import tqdm
from utils import CameraIntrinsics, ImageDataloader, load_intrinsics


class VSLAM(BaseVSLAM):
    """
    Visual Simultaneous Localization and Mapping (VSLAM) class.
    Inherits from BaseVSLAM and implements various VSLAM functionalities."""

    _intrinsics: CameraIntrinsics
    _img: MatLike
    _bf: BFMatcher
    _dataloader: ImageDataloader
    _orb: cv2.ORB
    _visualize: bool
    _use_rerun: bool
    _rerun_output_path: str | None

    def __init__(self, n_features: int = 1000, visualize: bool = False, use_rerun: bool = True, rerun_output_path: str | None = "vslam_output.rrd") -> None:
        super().__init__()
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._orb = cv2.ORB()
        self._orb = self._orb.create(nfeatures=n_features)
        self._visualize = visualize
        self._use_rerun = use_rerun
        self._rerun_output_path = rerun_output_path
        
        # Initialize rerun if enabled
        if self._use_rerun:
            # If output path is provided, save to file; otherwise spawn viewer
            spawn_viewer = rerun_output_path is None
            rr.init("vslam_pipeline", spawn=spawn_viewer)
            # Set up world coordinate system
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    def _load_intrinsics(self, path: Path) -> None:
        """
        Load camera intrinsics from a text file.
        """
        if not path.exists():
            raise FileNotFoundError(f"Intrinsics file not found at {path}")

        self._intrinsics = load_intrinsics(path)
        
        # Log camera intrinsics to rerun
        if self._use_rerun:
            rr.log(
                "world/camera/image",
                rr.Pinhole(
                    resolution=[int(self._intrinsics.cx * 2), int(self._intrinsics.cy * 2)],
                    focal_length=[self._intrinsics.fx, self._intrinsics.fy],
                    principal_point=[self._intrinsics.cx, self._intrinsics.cy],
                ),
                static=True,
            )

    def bundle_adjustment(self) -> Any:
        pass

    def feature_extraction(self, img: MatLike) -> tuple[Any, UMat]:
        """
        Docstring for feature_extraction

        :param self: Description
        :return: Description
        :rtype: Any
        """
        keypoints, descriptors = self._orb.detectAndCompute(img, None)
        return keypoints, descriptors

    def feature_matching(self, descriptors1: UMat, descriptors2: UMat) -> list:
        matches: Sequence = self._bf.match(descriptors1, descriptors2)
        sorted_matches = sorted(matches, key=lambda x: x.distance)
        return sorted_matches

    def feature_tracking(
        self, image1: MatLike, image2: MatLike, points1: UMat
    ) -> Any:
        # Parameters for lucas kanade optical flow
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03,
            ),
        )

        # Calculate optical flow
        points2, status, err = cv2.calcOpticalFlowPyrLK(
            image1, image2, points1, None, **lk_params
        )

        # Select good points
        good_new = points2[status.ravel() == 1]
        good_old = points1[status.ravel() == 1]

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        # Implementation of the fit method for VSLAM
        pass

    def load_dataset(self, dataset_path: str) -> None:
        """
        Docstring for load_dataset

        :param self: Description
        :param dataset_path: Description
        :type dataset_path: str
        """
        self._dataloader = ImageDataloader(Path(dataset_path))

    # @staticmethod
    def load_image(self, file_path: Path) -> MatLike | None:
        if not file_path.exists():
            return None
        else:
            # Load an image using OpenCV
            image = cv2.imread(file_path.as_posix())
            self._img = image
            return image

    def loop_closure(self) -> Any:
        pass

    def map_management(self) -> Any:
        pass

    def optimization(self) -> Any:
        pass

    def pipeline(self) -> None:
        """
        Complete VSLAM pipeline processing
        
        Front-end:
        1. Feature extraction (ORB)
        2. Feature matching
        3. Pose estimation (using epipolar geometry)
        4. Triangulation (3D point reconstruction)
        5. Local mapping
        
        This implements a simplified monocular SLAM following ORB-SLAM principles.
        """
        # Initialize variables for tracking consecutive frames
        img1: MatLike | None = None
        img2: MatLike | None = None
        descriptor1: UMat | None = None
        keypoints1: tuple[Any, UMat] | None = None
        pose_prev = np.eye(4)  # Previous pose (4x4 homogeneous transformation)
        
        # Store trajectory and 3D map points
        trajectory = []
        map_points_3d = []
        
        # Get total number of images for progress bar
        total_frames = len(self._dataloader)
        
        # Wrap dataloader with tqdm for progress tracking with visual bar
        for idx, img in enumerate(tqdm(self._dataloader, 
                                       total=total_frames,
                                       desc="Processing frames", 
                                       unit="frame",
                                       bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}',
                                       ncols=100)):
            # === STEP 1: Feature Extraction ===
            # For the first frame, initialize
            if img1 is None:
                img1 = img
                keypoints1, descriptor1 = self.feature_extraction(img1)
                trajectory.append(pose_prev.copy())
                
                # Log to rerun
                if self._use_rerun:
                    rr.set_time_sequence("frame", 0)
                    rr.log("world/camera/image", rr.Image(img1))
                    # Log initial camera pose
                    rr.log("world/camera", rr.Transform3D(translation=[0, 0, 0]))
                
                # Display first frame with keypoints (if visualization enabled)
                if self._visualize:
                    vis_img = cv2.drawKeypoints(img1, keypoints1, None, color=(0, 255, 0), 
                                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    cv2.putText(vis_img, f"Frame 0 - Initialization", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('VSLAM Pipeline', vis_img)
                    cv2.waitKey(1)
                continue
            
            # Process subsequent frames
            img2 = img
            keypoints2, descriptor2 = self.feature_extraction(img2)
            
            # === STEP 2: Feature Matching ===
            # Match features between consecutive frames
            matches = self.feature_matching(descriptor1, descriptor2)
            
            # Need at least 8 points for fundamental matrix estimation
            if len(matches) < 8:
                print(f"Frame {idx}: Not enough matches ({len(matches)} < 8), skipping")
                img1 = img2
                keypoints1 = keypoints2
                descriptor1 = descriptor2
                continue
            
            # Extract matched keypoint coordinates
            pts1 = np.array([keypoints1[m.queryIdx].pt for m in matches[:50]], dtype=np.float32)
            pts2 = np.array([keypoints2[m.trainIdx].pt for m in matches[:50]], dtype=np.float32)
            
            # === STEP 3: Pose Estimation ===
            R, t, inliers = self.pose_estimation(pts1, pts2)
            
            if R is None or t is None:
                print(f"Frame {idx}: Failed to estimate pose")
                img1 = img2
                keypoints1 = keypoints2
                descriptor1 = descriptor2
                continue
            
            # Filter inlier points for triangulation
            pts1_inliers = pts1[inliers.ravel() == 1]
            pts2_inliers = pts2[inliers.ravel() == 1]
            
            # Build 4x4 homogeneous transformation matrix
            Rt = np.eye(4)
            Rt[:3, :3] = R
            Rt[:3, 3] = t.ravel()
            
            # Update current pose (relative to previous)
            pose_current = pose_prev @ Rt
            trajectory.append(pose_current.copy())
            
            # === STEP 4: Triangulation ===
            points_3d_filtered = self.triangulation(pts1_inliers, pts2_inliers, Rt)
            
            # === STEP 5: Local Mapping ===
            # Add validated 3D points to the map
            if points_3d_filtered.shape[1] > 0:
                map_points_3d.append(points_3d_filtered)
            
            # === Log to Rerun ===
            if self._use_rerun:
                rr.set_time_sequence("frame", idx)
                
                # Log current image
                rr.log("world/camera/image", rr.Image(img2))
                
                # Log camera pose (transform from world to camera)
                position = pose_current[:3, 3]
                rotation = pose_current[:3, :3]
                rr.log(
                    "world/camera",
                    rr.Transform3D(
                        translation=position,
                        mat3x3=rotation,
                    ),
                )
                
                # Log 3D points in world coordinates
                if points_3d_filtered.shape[1] > 0:
                    # Transform points to world coordinates
                    points_3d_world = pose_current[:3, :3] @ points_3d_filtered + pose_current[:3, 3:4]
                    rr.log(
                        "world/points",
                        rr.Points3D(
                            positions=points_3d_world.T,
                            colors=[255, 255, 255],
                            radii=0.01,
                        ),
                    )
                
                # Log trajectory line
                if len(trajectory) > 1:
                    positions = np.array([pose[:3, 3] for pose in trajectory])
                    rr.log(
                        "world/trajectory",
                        rr.LineStrips3D(
                            strips=[positions],
                            colors=[0, 255, 0],
                        ),
                    )
            
            # === Visualize current frame with matches (if visualization enabled) ===
            if self._visualize and len(matches) > 0:
                # Draw matches between frames (use stored previous keypoints)
                vis_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, 
                                         matches[:20], None, 
                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                
                # Add info overlay
                info_text = [
                    f"Frame: {idx}/{total_frames}",
                    f"Matches: {len(matches)}",
                    f"Inliers: {np.sum(inliers)}",
                    f"3D Points: {points_3d_filtered.shape[1]}",
                    f"Total Map Points: {sum(pts.shape[1] for pts in map_points_3d)}"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(vis_img, text, (10, 30 + i*30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('VSLAM Pipeline', vis_img)
                cv2.waitKey(1)  # 1ms delay to allow display update
            
            # Update for next iteration
            img1 = img2
            keypoints1 = keypoints2
            descriptor1 = descriptor2
            pose_prev = pose_current
        
        # Close visualization window (if visualization was enabled)
        if self._visualize:
            cv2.destroyAllWindows()
        
        # Store results
        self._trajectory = trajectory
        self._map_points = map_points_3d
        
        print("\nPipeline complete!")
        print(f"Total frames processed: {len(trajectory)}")
        print(f"Total 3D map points: {sum(pts.shape[1] for pts in map_points_3d)}")
        
        # Log final accumulated 3D map to rerun
        if self._use_rerun and len(map_points_3d) > 0:
            # Combine all 3D points
            all_points = np.hstack(map_points_3d)
            rr.log(
                "world/map_points",
                rr.Points3D(
                    positions=all_points.T,
                    colors=[255, 0, 0],
                    radii=0.02,
                ),
            )
            
            # Save to file if output path is specified
            if self._rerun_output_path:
                rr.save(self._rerun_output_path)
                print(f"\nRerun recording saved to: {self._rerun_output_path}")
                print(f"To view: rerun {self._rerun_output_path}")
            else:
                print("\nRerun visualization active. Check the Rerun viewer window.")
        
        # Visualize the trajectory
        if not self._use_rerun:
            print("\nVisualizing camera trajectory...")
            self.visualize_trajectory()
        else:
            print("\nTrajectory visualization available in Rerun viewer.")
        
        pass

    def pose_estimation(self, pts1: jnp.ndarray, pts2: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | tuple[None, None, None]:
        """
        Estimate camera pose between two frames using epipolar geometry.
        
        Args:
            pts1: Matched keypoints from frame 1 (Nx2)
            pts2: Matched keypoints from frame 2 (Nx2)
        
        Returns:
            Tuple of (R, t, inliers) - Rotation matrix, translation vector, and inlier mask
            Returns (None, None, None) if pose estimation fails
        """
        # Convert JAX arrays to numpy arrays for OpenCV
        pts1_np = np.array(pts1, dtype=np.float32)
        pts2_np = np.array(pts2, dtype=np.float32)
        
        # Build camera intrinsic matrix
        K = np.array([
            [self._intrinsics.fx, 0, self._intrinsics.cx],
            [0, self._intrinsics.fy, self._intrinsics.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Compute the Fundamental matrix using RANSAC
        F, inliers = cv2.findFundamentalMat(pts1_np, pts2_np, cv2.FM_RANSAC, 1.0, 0.99)
        
        if F is None or inliers is None:
            return None, None, None
        
        # Filter inlier points
        pts1_inliers = pts1_np[inliers.ravel() == 1]
        pts2_inliers = pts2_np[inliers.ravel() == 1]

        # Compute the Essential matrix using the camera's intrinsic parameters
        # E = K^T @ F @ K
        E = K.T @ F @ K

        # Decompose the Essential matrix to get R and t
        _, R, t, mask = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)
        
        return R, t, inliers

    def tracking(self) -> Any:
        pass

    def visualize_trajectory(self, save_path: str | None = None) -> None:
        """
        Visualize the camera trajectory in 3D.
        
        Args:
            save_path: Optional path to save the figure. If None, displays interactively.
        """
        if not hasattr(self, '_trajectory') or len(self._trajectory) == 0:
            print("No trajectory data available. Run pipeline() first.")
            return
        
        # Extract camera positions from trajectory
        positions = np.array([pose[:3, 3] for pose in self._trajectory])
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'b-', linewidth=2, label='Camera Trajectory')
        
        # Mark start and end points
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                  c='green', s=100, marker='o', label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                  c='red', s=100, marker='x', label='End')
        
        # Set labels and title
        ax.set_xlabel('X (meters)', fontsize=10)
        ax.set_ylabel('Y (meters)', fontsize=10)
        ax.set_zlabel('Z (meters)', fontsize=10)
        ax.set_title('VSLAM Camera Trajectory', fontsize=14, fontweight='bold')
        ax.legend()
        
        # Set equal aspect ratio for better visualization
        ax.set_box_aspect([1,1,1])
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trajectory saved to {save_path}")
        else:
            plt.show()

    def triangulation(self, pts1: jnp.ndarray, pts2: jnp.ndarray, Rt: jnp.ndarray) -> jnp.ndarray:
        """
        Reconstruct 3D points from matched 2D correspondences.
        
        Args:
            pts1: Inlier points from frame 1 (Nx2)
            pts2: Inlier points from frame 2 (Nx2)
            Rt: Relative transformation matrix (4x4) from frame 1 to frame 2
        
        Returns:
            3D points in Euclidean coordinates (3xN) - filtered for valid depth
        """
        # Convert to numpy arrays for OpenCV
        pts1_np = np.array(pts1, dtype=np.float32)
        pts2_np = np.array(pts2, dtype=np.float32)
        Rt_np = np.array(Rt, dtype=np.float32)
        
        # Build camera intrinsic matrix
        K = np.array([
            [self._intrinsics.fx, 0, self._intrinsics.cx],
            [0, self._intrinsics.fy, self._intrinsics.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Create projection matrices
        P1 = K @ np.eye(3, 4)  # Previous camera (identity)
        P2 = K @ Rt_np[:3, :]  # Current camera
        
        # Triangulate points using OpenCV
        points_4d = cv2.triangulatePoints(P1, P2, pts1_np.T, pts2_np.T)
        points_3d = points_4d[:3, :] / points_4d[3, :]  # Convert to Euclidean coords
        
        # Filter out points with negative depth or too far
        valid_depth = (points_3d[2, :] > 0) & (points_3d[2, :] < 50)
        points_3d_filtered = points_3d[:, valid_depth]
        
        return points_3d_filtered
