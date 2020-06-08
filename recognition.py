import copy
import math

import cv2
import numpy as np

from CarNumber import CarNumber
from Image import Image
from TestDataCreator import TestDataCreator


class RecognitionCarPlate:
    def __init__(self):
        self.origin = None
        self.car_numbers = []

    def load_image(self, file_image_name):
        image = cv2.imread(file_image_name)
        self.origin = Image(image)

    def __find_number_plates_on_origin_image(self):
        russian_number_cascade = cv2.CascadeClassifier('xml-car-numbers\\haarcascade_russian_plate_number.xml')
        russian_number_plate_rect = russian_number_cascade.detectMultiScale(self.origin.grayscale(), scaleFactor=1.2,
                                                                            minNeighbors=2)

        if len(russian_number_plate_rect):
            self.origin: Image
            copy_origin = copy.deepcopy(self.origin)

            for (x, y, w, h) in russian_number_plate_rect:
                cropped_image = self.origin.crop(x, y, x + w, y + h)
                self.car_numbers.append(CarNumber(cropped_image))
                cv2.rectangle(copy_origin.image, (x, y), (x + w, y + h), (0, 255, 0), 10)
            # copy_origin.show("Number Plates")

    def __normalizing_image_of_number_plate_contours(self, image: Image):
        edges = image.canny(30, 150)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        image_copy = copy.deepcopy(image)

        for cnt in contours:
            rect = cv2.minAreaRect(cnt)  # try to fit a rectangle
            box = cv2.boxPoints(rect)  # search for four vertices of a rectangle

            if abs(box[0][0] - box[2][0]) < image.width / 20 or abs(box[0][1] - box[1][1]) < image.height / 4:
                continue

            box = np.int0(box)  # round coordinates
            cv2.drawContours(image_copy.image, [box], 0, (255, 0, 0), 2)

        # image_copy.show("Contours")

    def __normalizing_image_of_number_plate_hough_lines(self, image: Image):
        image_copy = copy.deepcopy(image)

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

                    cv2.line(image_copy.image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # image_copy.show("Hough lines")

    def __find_lines_with_hough_lines_p(self, image: Image) -> list:
        edges = image.canny(50, 150)

        min_line_length = 150
        max_line_gap = 30

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, min_line_length, max_line_gap)
        bounds = []

        if lines is not None:
            center_of_image_height = int(image.height / 2)

            lines_above = [line for line in lines for _, y1, _, y2 in line
                           if y1 < center_of_image_height and y2 < center_of_image_height]
            lines_below = [line for line in lines for _, y1, _, y2 in line
                           if y1 > center_of_image_height and y2 > center_of_image_height]

            image_copy = copy.deepcopy(image)

            for part_of_lines in [lines_above, lines_below]:
                if part_of_lines:
                    tangent_of_lines = [(y2 - y1) / (x2 - x1) for line in part_of_lines for x1, y1, x2, y2 in line]

                    free_members_of_lines = [y1 - tangent_of_lines[i] * x1
                                             for i, line in enumerate(part_of_lines) for x1, y1, _, _ in line]

                    average_line = [np.mean(tangent_of_lines), np.mean(free_members_of_lines)]  # find average line
                    bounds.append(average_line)

                    cv2.line(image_copy.image, (0, int(average_line[1])),
                             (image.width, int(average_line[0] * image.width + average_line[1])), (0, 255, 0))
            # image_copy.show("TWO MAIN LINES")

        return bounds

    def __normalize_image(self, image: Image) -> Image:
        # ---------- rotate image -----------------
        bounds = self.__find_lines_with_hough_lines_p(image)
        if not bounds:
            return image

        max_k = np.mean([line[0] for line in bounds])
        angle = (math.atan(max_k) * 180) / np.pi
        image.rotate(angle)
        # image.show("ROTATED")
        # --------- crop rotated images ------------
        bounds = self.__find_lines_with_hough_lines_p(image)
        if not bounds:
            return image

        return self.__crop_image_by_bounds(image, bounds)

    def __crop_image_by_bounds(self, image: Image, bounds: list) -> Image:
        center_height_of_image = int(image.height / 2)

        if len(bounds) == 2:
            y1 = int(bounds[0][1])
            y2 = int(bounds[1][0] * image.width + bounds[1][1])
            return image.crop(0, y1, image.width, y2)

        elif len(bounds) == 1:
            if int(bounds[0][0] * image.width / 2 + bounds[0][1]) < center_height_of_image:
                return image.crop(0, int(bounds[0][1]), image.width, image.height)
            else:
                return image.crop(0, 0, image.width, int(bounds[0][0] * image.width + bounds[0][1]))

    def __increase_image_contrast(self, image: Image):
        # Converting image to LAB Color model
        lab = cv2.cvtColor(image.image, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        # Applying CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        limg = cv2.merge((cl, a, b))

        # Converting image from LAB Color model to RGB model
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        image.image = final

    def __crop_side_edges_of_the_image(self, image: Image) -> Image:
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

    def __crop_side_edges_of_the_image_2(self, image: Image):
        # minimum_left = np.argmin(image.brightness[:int(len(image.brightness)*0.25)])
        # minimum_right = np.argmin(list(reversed(image.brightness))[:int(len(image.brightness)*0.25)])
        # img = image.crop(minimum_left, 0, len(image.brightness) - minimum_right-1, image.height)
        # img.show('CROPPED BY THE EDGES')
        quarter = int(len(image.brightness)*0.25)

        i_minimum_left = np.argmin(image.brightness[:quarter])
        i_minimum_right = len(image.brightness) - 1 - np.argmin(list(reversed(image.brightness))[:quarter])
        i_minimum_center = quarter + np.argmin(image.brightness[quarter:-quarter])

        result_image = None
        # crop right edge
        if image.brightness[i_minimum_right] < image.brightness[i_minimum_center]:
            result_image = image.crop(0, 0, i_minimum_right, image.height)
            # result_image.show('CROPPED right')

        # crop left edge
        if image.brightness[i_minimum_left] < image.brightness[i_minimum_center]:
            if result_image is None:
                result_image = image.crop(i_minimum_left, 0, image.width, image.height)
            else:
                result_image = result_image.crop(i_minimum_left, 0, result_image.width, result_image.height)
            # result_image.show('CROPPED left')

        if result_image is not None:
            return result_image
        else:
            return image

    def __split_number_plate_into_characters_by_certain_dist(self, image: Image):
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
            symbol.show(str(len(image.characters_on_image))+'SYMBOL')

        # region.show('REGION')
        self.hist(region.brightness)

    def __splitting_binarized_image_into_numbers(self, image: Image) -> tuple:
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

        left_edge_of_char = None
        series_and_reg_num = []

        # TODO сделать обрезку белому месту вместо чёрной палки
        left_edge_of_region = None
        region = None

        for i in range(image.width):
            column = image.image[int(0.25 * image.height):int(-0.1 * image.height), i]
            if 0 in column:
                if left_edge_of_char:
                    continue
                else:
                    left_edge_of_char = i
            else:
                if not left_edge_of_char:
                    continue
                else:
                    char_image = image.crop(left_edge_of_char, 0, i, image.height)

                    if left_edge_of_region is None:
                        series_and_reg_num.append(char_image)

                    if char_image.is_black_stick() and len(series_and_reg_num) > 1:
                        if left_edge_of_region is None:
                            left_edge_of_region = i
                        else:
                            region = image.crop(left_edge_of_region, 0, left_edge_of_char, image.height)
                            left_edge_of_region = None
                            break

                    left_edge_of_char = None

        if left_edge_of_region:
            region = image.crop(left_edge_of_region, 0, left_edge_of_char, image.height)

        # for number in series_and_reg_num:
        #     number.show("NUMBERS")

        return series_and_reg_num, region

    def __process_region(self, region: Image) -> list:
        # crop rus on region image
        for i in range(int(region.height/2), region.height):
            if np.mean(region.image[i]) == 255.:
                region = region.crop(0, 0, region.width, i)
                break

        region_characters, _ = self.__splitting_binarized_image_into_numbers(region)
        for i, char in enumerate(region_characters):
            if not char.is_empty():
                region_characters[i] = char.crop_binarized_char_by_edges()  # self.crop_binarized_char_by_edges(char)

        for i in range(len(region_characters)-1, -1, -1):
            if region_characters[i].is_empty():
                del region_characters[i]

        return region_characters

    # @staticmethod
    # def crop_binarized_char_by_edges(image: Image):
    #     up, down, left, right = 0, image.height, 0, image.width
    #     # up
    #     for i in range(int(image.height / 2) - 1, -1, -1):
    #         if np.mean(image.image[i]) == 255.:
    #             up = i
    #             break
    #     # down
    #     for i in range(int(image.height / 2), image.height):
    #         if np.mean(image.image[i]) == 255.:
    #             down = i
    #             break
    #
    #     # left
    #     for i in range(int(image.width / 2) - 1, -1, -1):
    #         if np.mean(image.image[:, i]) == 255.:
    #             left = i
    #             break
    #
    #     # right
    #     for i in range(int(image.width / 2), image.width):
    #         if np.mean(image.image[:, i]) == 255.:
    #             right = i
    #             break
    #
    #     return image.crop(left, up, right, down)

    def __prepare_symbols_for_recognition(self, car_number: CarNumber):
        car_number.region = self.__process_region(car_number.region_image)
        test_data_creator = TestDataCreator()

        for i in range(len(car_number.series_and_registration_num)-1, -1, -1):
            char = car_number.series_and_registration_num[i]
            char: Image
            car_number.series_and_registration_num[i] = char.crop_binarized_char_by_edges()  # self.crop_binarized_char_by_edges(char)

            if car_number.series_and_registration_num[i].is_empty():
                del car_number.series_and_registration_num[i]
                continue
            car_number.series_and_registration_num[i].rotate(5)
            # car_number.series_and_registration_num[i].show("CROPPED SYMBOLS SERIES AND REGISTRATION NUMBER")
            test_data_creator.run(car_number.series_and_registration_num[i])

        for char in car_number.region:
            # char.show("CROPPED SYMBOLS REGION")
            test_data_creator.run(car_number.region[i])

    def run(self):
        # self.origin.show("Origin")
        self.__find_number_plates_on_origin_image()
        for i in range(len(self.car_numbers)-1, -1, -1):
            car_number = self.car_numbers[i]
            print(i)

            car_number.image = self.__normalize_image(car_number.image)

            self.__increase_image_contrast(car_number.image)

            car_number.image = self.__crop_side_edges_of_the_image_2(car_number.image)

            car_number.image.binarize()
            car_number.image.show("BINARIZED NUMBER")

            car_number.series_and_registration_num, car_number.region_image \
                = self.__splitting_binarized_image_into_numbers(car_number.image)
            car_number.remove_all_empty_images()

            if not car_number.is_valid():
                del self.car_numbers[i]
                continue

            self.__prepare_symbols_for_recognition(car_number)
