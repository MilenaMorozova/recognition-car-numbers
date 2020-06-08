import copy

import cv2
import numpy as np
from matplotlib import pyplot as plt


class Image:

    def __init__(self, image):
        self.brightness = None
        self.__image = None

        self.image = image

    @property
    def width(self):
        return self.__image.shape[1]

    @property
    def height(self):
        return self.__image.shape[0]

    def show(self, title: str):
        cv2.imshow(title, self.image)
        cv2.waitKey(0)

    def grayscale(self):
        return cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY)

    def __deepcopy__(self, memodict={}):
        return Image(copy.deepcopy(self.__image))

    def crop(self, x1, y1, x2, y2):
        cropped_image = copy.deepcopy(self)
        cropped_image = cropped_image.__image[y1: y2, x1: x2]
        return Image(cropped_image)

    def canny(self, threshold1: float, threshold2: float, aperture_size: int = 3):
        return cv2.Canny(self.grayscale(), threshold1, threshold2, apertureSize=aperture_size)

    @property
    def image(self):
        return self.__image

    @image.setter
    def image(self, value):
        self.__image = value
        self.brightness = self.calc_brightness()

    def rotate(self, angle):
        image_center = (self.width/2, self.height/2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
        result = cv2.warpAffine(self.image, rot_mat, (self.width, self.height), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255))
        return Image(result)

    def binarize(self):
        gray_image = self.grayscale()

        dark_pix = int(np.min(gray_image))
        bright_pix = int(np.max(gray_image))

        threshold = (dark_pix + bright_pix) / 2
        _, binarized_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        self.__image = binarized_image  #

    def calc_brightness(self):
        if len(self.__image.shape) == 3:
            gray_image = self.grayscale()
            return [np.mean(gray_image[:, i]) for i in range(self.width)]
        else:
            return [np.mean(self.__image[:, i]) for i in range(self.width)]

    def is_black_stick(self):
        if self.width < int(0.1 * self.height) \
                and np.sum(self.__image == 0) >= 0.2*self.width*self.height:
            return True
        for i in range(self.width):
            if np.sum(self.__image[:, i] == 0) < int(0.8 * self.height):
                return False
        return True

    def is_empty(self):
        return self.width == 0 or self.height == 0

    def flip_vertical(self):
        return Image(cv2.flip(self.image, 1))

    def flip_horizontal(self):
        return Image(cv2.flip(self.image, 0))

    def hist(self):
        plt.plot(self.brightness)
        plt.show()