from .vslam import VSLAM


def main(args=None):
    vslam = VSLAM()
    image = vslam.load_image("test_image.jpg")
    print("Image loaded:", image is not None)