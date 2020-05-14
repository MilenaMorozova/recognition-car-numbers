import copy

import cv2


class Image:

    def __init__(self, image):
        self.image = image
        self.bounds = None
        self.brightness = None
        self.characters_on_image = None

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

    def set_brightness(self, brightness):
        self.brightness = brightness

    def binarize(self):
        gray_image = self.grayscale()
        center_column = gray_image[:, int(self.width / 2)]

        min_pix = int(min(center_column))
        max_pix = int(max(center_column))
        threshold = (min_pix + max_pix) / 2.

        _, thresh1 = cv2.threshold(self.image, threshold, 255, cv2.THRESH_BINARY)
        self.set_image(thresh1)
