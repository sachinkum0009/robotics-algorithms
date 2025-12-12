from .base_vslam import BaseVSLAM
from typing import Any
import cv2
from cv2.typing import MatLike


class VSLAM(BaseVSLAM):
    def __init__(self):
        super().__init__()

    def load_image(self, file_path: str) -> MatLike | None:
        # Load an image using OpenCV
        image = cv2.imread(file_path)
        return image

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        # Implementation of the fit method for VSLAM
        pass
