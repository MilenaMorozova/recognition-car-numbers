import math

import cv2
import numpy as np
from matplotlib import pyplot as plt

from Image import Image


class RecognitionCarPlate:
    def __init__(self):
        self.origin = None
        self.cropped_images = None

    def load_image(self, file_image_name):
        image = cv2.imread(file_image_name)
        self.origin = Image(image)

    def find_number_plates_on_origin_image(self):
        # TODO показать нарисованный прямоугольнк
        russian_number_cascade = cv2.CascadeClassifier('xml-car-numbers\\haarcascade_russian_plate_number.xml')
        russian_number_plate_rect = russian_number_cascade.detectMultiScale(self.origin.grayscale(), scaleFactor=1.2,
                                                                            minNeighbors=2)
        self.cropped_images = []
        if len(russian_number_plate_rect):
            for (x, y, w, h) in russian_number_plate_rect:
                self.cropped_images.append(self.origin.crop(x, y, x + w, y + h))
                cv2.rectangle(self.origin.image, (x, y), (x + w, y + h), (0, 255, 0), 10)

    @staticmethod
    def image_show(title: str, image):
        cv2.imshow(title, image)
        cv2.waitKey(0)

    def normalizing_image_of_number_plate_contours(self, image: Image):
        edges = image.canny(30, 150)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        image_copy = image.image.copy()

        for cnt in contours:
            rect = cv2.minAreaRect(cnt)  # try to fit a rectangle
            box = cv2.boxPoints(rect)  # search for four vertices of a rectangle

            if abs(box[0][0] - box[2][0]) < image.width / 20 or abs(box[0][1] - box[1][1]) < image.height / 4:
                continue

            box = np.int0(box)  # round coordinates
            cv2.drawContours(image_copy, [box], 0, (255, 0, 0), 2)

        self.image_show("Contours", image_copy)

    def normalizing_image_of_number_plate_hough_lines(self, image: Image):
        image_copy = image.image.copy()

        edges = image.canny(50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
        if lines is not None:
            for line in lines:
                for rho, theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * a)
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * a)

                    cv2.line(image_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # self.image_show("Hough lines", image_copy)

    def normalizing_image_of_number_plate_hough_lines_p(self, image: Image):
        edges = image.canny(50, 150)

        min_line_length = 150
        max_line_gap = 30

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, min_line_length, max_line_gap)

        if lines is not None:
            self.find_two_main_lines(lines, image)
        else:
            print('Увы')

    def find_two_main_lines(self, lines: list, image: Image):
        center_of_image_height = int(image.height / 2)

        lines_above = [line for line in lines for _, y1, _, y2 in line
                       if y1 < center_of_image_height and y2 < center_of_image_height]
        lines_below = [line for line in lines for _, y1, _, y2 in line
                       if y1 > center_of_image_height and y2 > center_of_image_height]

        bounds = []
        image_copy = image.image.copy()

        for part_of_lines in [lines_above, lines_below]:
            if part_of_lines:
                tangent_of_lines = [(y2 - y1) / (x2 - x1) for line in part_of_lines for x1, y1, x2, y2 in line]

                free_members_of_lines = [y1 - tangent_of_lines[i] * x1
                                         for i, line in enumerate(part_of_lines) for x1, y1, _, _ in line]

                average_line = [np.mean(tangent_of_lines), np.mean(free_members_of_lines)]  # find average line
                bounds.append(average_line)

                cv2.line(image_copy, (0, int(average_line[1])),
                         (image.width, int(average_line[0] * image.width + average_line[1])), (0, 255, 0))

        image.set_bounds(bounds)
        # self.image_show("TWO MAIN LINES", image_copy)

    @staticmethod
    def rotate_image(image: Image):
        if image.bounds is None:
            return

        max_k = np.mean([line[0] for line in image.bounds])
        angle = (math.atan(max_k) * 180) / np.pi

        image_center = tuple(np.array(image.image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
        result = cv2.warpAffine(image.image, rot_mat, image.image.shape[1::-1], flags=cv2.INTER_LINEAR)

        image.set_image(result)

    @staticmethod
    def crop_image_by_bounds(image: Image) -> Image:
        if image.bounds is None:
            return image

        center_height_of_image = int(image.height / 2)

        if len(image.bounds) == 2:
            y1 = int(image.bounds[0][1])
            y2 = int(image.bounds[1][0] * image.width + image.bounds[1][1])
            return image.crop(0, y1, image.width, y2)

        elif len(image.bounds) == 1:
            if int(image.bounds[0][0] * image.width / 2 + image.bounds[0][1]) < center_height_of_image:
                return image.crop(0, int(image.bounds[0][1]), image.width, image.height)
            else:
                return image.crop(0, 0, image.width, int(image.bounds[0][0] * image.width + image.bounds[0][1]))

    @staticmethod
    def increase_image_contrast(image: Image):
        # Converting image to LAB Color model
        lab = cv2.cvtColor(image.image, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        # Applying CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        limg = cv2.merge((cl, a, b))

        # Converting image from LAB Color model to RGB model
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        image.set_image(final)

    @staticmethod
    def hist(brightness: list):
        plt.plot(brightness)
        plt.show()

    @staticmethod
    def crop_side_edges_of_the_image(image: Image) -> Image:
        minimum = min(image.brightness)
        maximum = max(image.brightness)
        point = minimum + (maximum - minimum)*0.6
        borders = []

        # left border
        found_the_first_occurrence = False
        if image.brightness[0] < point:
            for i, value in enumerate(image.brightness):
                if found_the_first_occurrence and image.brightness[i-1] >= value < image.brightness[i+1]:
                    borders.append(i)
                    break

                if value > point:
                    found_the_first_occurrence = True
        else:
            for i, value in enumerate(image.brightness):
                if found_the_first_occurrence and image.brightness[i - 1] >= value < image.brightness[i + 1]:
                    borders.append(i)
                    break

                if value < point:
                    found_the_first_occurrence = True

        # right border
        found_the_first_occurrence = False
        if image.brightness[-1] < point:
            for i, value in enumerate(list(reversed(image.brightness))):
                if found_the_first_occurrence and image.brightness[i - 1] >= value < image.brightness[i + 1]:
                    borders.append(len(image.brightness) - (1 + i))
                    break

                if value > point:
                    found_the_first_occurrence = True
        else:
            for i, value in enumerate(list(reversed(image.brightness))):
                if found_the_first_occurrence and image.brightness[i - 1] >= value < image.brightness[i + 1]:
                    borders.append(len(image.brightness) - (1 + i))
                    break

                if value < point:
                    found_the_first_occurrence = True

        if borders[0] > borders[1]:
            return image.crop(borders[1], 0, borders[0], image.height)
        return image.crop(borders[0], 0, borders[1], image.height)

    def crop_side_edges_of_the_image_2(self, image: Image, index: int):
        # minimum_left = np.argmin(image.brightness[:int(len(image.brightness)*0.25)])
        # minimum_right = np.argmin(list(reversed(image.brightness))[:int(len(image.brightness)*0.25)])
        # img = image.crop(minimum_left, 0, len(image.brightness) - minimum_right-1, image.height)
        # self.image_show('CROPPED BY THE EDGES', img.image)
        quarter = int(len(image.brightness)*0.25)

        minimum_left = np.argmin(image.brightness[:quarter])
        minimum_right = len(image.brightness) - 1 - np.argmin(list(reversed(image.brightness))[:quarter])
        minimum_center = quarter + np.argmin(image.brightness[quarter:-quarter])

        img = None
        if image.brightness[minimum_right] < image.brightness[minimum_center]:
            img = image.crop(0, 0, minimum_right, image.height)
            # self.image_show('CROPPED right', img.image)

        if image.brightness[minimum_left] < image.brightness[minimum_center]:
            if img is None:
                img = image.crop(minimum_left, 0, image.width, image.height)
            else:
                img = img.crop(minimum_left, 0, img.width, img.height)
            # self.image_show('CROPPED left', img.image)

        if img is not None:
            self.cropped_images[index] = img
            self.binarize_number_plate(img)
            # self.split_number_plate_into_characters(self.cropped_images[index])
        else:
            self.binarize_number_plate(image)

    def split_number_plate_into_characters(self, image: Image):
        wdth = image.width / 520  # width of the russian number plate is 520mm
        hght = image.height / 112  # height of the russian number plate is 112mm
        image.characters_on_image = []

        first_symbol = image.crop(int(wdth*35), int(hght*30), int(wdth*100), int(hght*105))
        second_symbol = image.crop(int(wdth*100), int(hght*20), int(wdth*155), int(hght*105))
        third_symbol = image.crop(int(wdth*155), int(hght*20), int(wdth*210), int(hght*105))
        fourth_symbol = image.crop(int(wdth*210), int(hght*20), int(wdth*265), int(hght*105))
        fifth_symbol = image.crop(int(wdth*265), int(hght*30), int(wdth*330), int(hght*105))
        sixth_symbol = image.crop(int(wdth*330), int(hght*30), int(wdth*385), int(hght*105))
        region = image.crop(int(wdth*395), int(hght*10), int(wdth*495), int(hght*90))

        for symbol in [first_symbol, second_symbol, third_symbol, fourth_symbol, fifth_symbol, sixth_symbol]:
            image.characters_on_image.append(symbol.brightness)
            # self.hist(symbol.brightness)
            # self.image_show(str(len(image.characters_on_image))+'SYMBOL', symbol.image)

        self.image_show('REGION', region.image)
        self.hist(region.brightness)

    def binarize_number_plate(self, image: Image):
        gray_image = image.grayscale()
        dark_pix = int(np.min(gray_image))
        bright_pix = int(np.max(gray_image))

        threshold = (dark_pix+bright_pix)/2
        _, binarized_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

        image.set_image(binarized_image)
        self.image_show("BINARIZED IMAGE", image.image)
        self.splitting_binarized_image_into_numbers(image)

    def splitting_binarized_image_into_numbers(self, image: Image) -> list:
        start = 0
        for i in range(int(image.height/2), -1, -1):
            if np.sum(image.image[i] == 0) > 0.8*image.width:
                start = i
                break

        end = image.height
        for i in range(int(image.height/2), image.height):
            if np.sum(image.image[i] == 0) > 0.8*image.width:
                end = i
                break

        image = image.crop(0, start, image.width, end)
        left_edge = None
        numbers = []

        for i in range(image.width):
            column = image.image[int(0.25 * image.height):int(-0.1 * image.height), i]
            if 0 in column:
                if left_edge:
                    continue
                else:
                    left_edge = i
            else:
                if not left_edge:
                    continue
                else:
                    numbers.append(image.crop(left_edge, 0, i, image.height))
                    left_edge = None
        if left_edge:
            numbers.append(image.crop(left_edge, 0, image.width, image.height))

        for number in numbers:
            self.image_show("NUMBERS", number.image)

    def normalize_finded_symbols(self, image: Image):
        if image.characters_on_image is None:
            return

        for i, img in enumerate(image.characters_on_image):
            self.image_show('SYMBOL BEFORE BINARIZE', img.image)
            img.binarize()
            self.image_show('AFTER BINARIZE', img.image)
            image.characters_on_image[i] = self.crop_symbol_by_edges(img)
            self.image_show('CROPPED BINARIZED IMAGE', image.characters_on_image[i].image)

    @staticmethod
    def crop_symbol_by_edges(image: Image):
        result = image
        # up
        for i in range(int(result.height / 2) - 1, -1, -1):
            if np.mean(result.image[i][:, 0]) == 255.:
                result = result.crop(0, i, result.width, result.height)
                break
        # down
        for i in range(int(result.height / 2), result.height):
            if np.mean(result.image[i][:, 0]) == 255.:
                result = result.crop(0, 0, result.width, i)
                break

        # left
        for i in range(int(result.width / 2) - 1, -1, -1):
            if np.mean(result.image[:, i][:, 0]) == 255.:
                result = result.crop(i, 0, result.width, result.height)
                break

        # right
        for i in range(int(result.width / 2), result.width):
            if np.mean(result.image[:, i][:, 0]) == 255.:
                result = result.crop(0, 0, i, result.height)
                break
        return result

    def run(self):
        # self.image_show("Origin", self.origin.image)
        self.find_number_plates_on_origin_image()

        for i in range(len(self.cropped_images)):

            self.normalizing_image_of_number_plate_hough_lines_p(self.cropped_images[i])
            self.rotate_image(self.cropped_images[i])
            self.normalizing_image_of_number_plate_hough_lines_p(self.cropped_images[i])
            self.cropped_images[i] = self.crop_image_by_bounds(self.cropped_images[i])

            self.increase_image_contrast(self.cropped_images[i])

            self.crop_side_edges_of_the_image_2(self.cropped_images[i], i)
            # self.normalize_finded_symbols(self.cropped_images[i])