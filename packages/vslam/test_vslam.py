from pathlib import Path

from vslam import VSLAM


def main(args=None):
    # Setup paths
    camera_param_path = Path("/workspace/datasets/sequence_14/camera.txt")
    image_path = Path("/workspace/datasets/sequence_14/images")

    print("=" * 60)
    print("Visual SLAM Pipeline Test")
    print("=" * 60)

    # Initialize VSLAM with 1000 features
    print("\n1. Initializing VSLAM...")
    vslam = VSLAM(n_features=1000)

    # Load camera intrinsics
    print(f"2. Loading camera intrinsics from {camera_param_path}...")
    vslam._load_intrinsics(camera_param_path)
    print("   Camera parameters loaded:")
    print(f"   - fx: {vslam._intrinsics.fx}")
    print(f"   - fy: {vslam._intrinsics.fy}")
    print(f"   - cx: {vslam._intrinsics.cx}")
    print(f"   - cy: {vslam._intrinsics.cy}")

    # Load dataset
    print(f"\n3. Loading dataset from {image_path}...")
    vslam.load_dataset(str(image_path))
    print("   Dataset loaded successfully")

    # Run the VSLAM pipeline
    print("\n4. Running VSLAM pipeline...")
    print("-" * 60)
    vslam.pipeline()
    print("-" * 60)

    print("\n" + "=" * 60)
    print("VSLAM Pipeline Test Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
