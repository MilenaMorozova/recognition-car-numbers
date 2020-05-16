import copy

import cv2
import numpy as np


class Image:

    def __init__(self, image):
        self.image = image
        self.bounds = None
        self.brightness = self.calc_brightness()

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

    def set_bounds(self, bounds):
        self.bounds = bounds

    def set_image(self, image):
        self.image = image
        self.brightness = self.calc_brightness()

    def binarize(self):
        gray_image = self.grayscale()

        dark_pix = int(np.min(gray_image))
        bright_pix = int(np.max(gray_image))

        threshold = (dark_pix + bright_pix) / 2
        _, binarized_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        self.image = binarized_image

    def calc_brightness(self):
        if len(self.image.shape) == 3:
            gray_image = self.grayscale()
            return [np.mean(gray_image[:, i]) for i in range(self.width)]
        else:
            return [np.mean(self.image[:, i]) for i in range(self.width)]

    def is_black_stick(self):
        if self.width < int(0.1 * self.height) \
                and np.sum(self.image == 0) >= 0.2*self.width*self.height:
            return True
        for i in range(self.width):
            if np.sum(self.image[:, i] == 0) < int(0.8*self.height):
                return False
        return True
