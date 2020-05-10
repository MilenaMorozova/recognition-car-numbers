import copy

import cv2


class Image:

    def __init__(self, image):
        self.image = image

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def height(self):
        return self.image.shape[0]

    def grayscale(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def __deepcopy__(self, memodict={}):
        return Image(copy.deepcopy(self.image))

    def crop(self, x1, y1, x2, y2):
        cropped_image = copy.deepcopy(self)
        cropped_image = cropped_image.image[y1: y2, x1: x2]
        return Image(cropped_image)

    def canny(self, threshold1: float, threshold2: float, aperture_size: int = 3):
        return cv2.Canny(self.grayscale(), threshold1, threshold2, apertureSize=aperture_size)
