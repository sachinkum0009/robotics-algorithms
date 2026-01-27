from dataclasses import dataclass
from pathlib import Path

import cv2
from cv2.typing import MatLike


class ImageDataloader:
    def __init__(self, image_path: Path):
        """
        Initialize the image dataloader.

        :param image_path: Path to directory containing images or
         single image file
        :type image_path: Path
        """
        self.image_path = image_path
        if image_path.is_dir():
            # Get all image files with common extensions
            self.image_files = sorted(
                [f for f in image_path.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}]
            )
        else:
            self.image_files = [image_path]
        self.index = 0

    def __iter__(self):
        """Return the iterator object."""
        self.index = 0
        return self

    def __len__(self) -> int:
        """Return the number of images in the dataloader."""
        return len(self.image_files)

    def __next__(self) -> MatLike | None:
        """Return the next image in the sequence."""
        if self.index >= len(self.image_files):
            raise StopIteration

        image = cv2.imread(str(self.image_files[self.index]))
        self.index += 1
        return image

    def get(self, index: int) -> MatLike | None:
        """
        Get image at specific index.

        :param index: Index of the image
        :type index: int
        :return: Loaded image
        :rtype: MatLike
        """
        if index < 0 or index >= len(self.image_files):
            raise IndexError(f"Index {index} out of range")
        return cv2.imread(str(self.image_files[index]))


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@staticmethod
def load_intrinsics(camera_param_path: Path) -> CameraIntrinsics:
    """
    Load camera intrinsics from a file.

    Expected format:
    Line 0: distortion/other parameters (tab-separated)
    Line 1: max width and height (space-separated)
    Line 2: fx_norm fy_norm cx_norm cy_norm skew (space-separated, normalized)
    Line 3: actual width height (space-separated)

    :param camera_param_path: Path to camera parameters file
    :type camera_param_path: Path
    """
    with open(camera_param_path) as f:
        lines = f.readlines()

        # Parse line 2: normalized focal lengths and principal point
        params = list(map(float, lines[2].strip().split()))
        fx_norm, fy_norm, cx_norm, cy_norm = params[:4]

        # Parse line 3: actual image dimensions
        width, height = map(int, lines[3].strip().split())

    return CameraIntrinsics(
        fx=fx_norm,
        fy=fy_norm,
        cx=cx_norm,
        cy=cy_norm,
        width=width,
        height=height,
    )


def show_image(window_name: str, image: MatLike) -> None:
    """
    Display an image in a window.

    :param window_name: Name of the window
    :type window_name: str
    :param image: Image to display
    :type image: MatLike
    """
    import cv2

    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
