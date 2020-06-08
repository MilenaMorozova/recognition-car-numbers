import os
import copy

import cv2

from Image import Image

# TODO увеличить объём выборки(переворот 0, 6, 9), попробовать повернуть на немного картинки


class TestDataCreator:

    def create_image(self, image_file_name, image):
        resized = cv2.resize(image, (27, 36), interpolation=cv2.INTER_AREA)
        if image_file_name.isalpha():
            files = os.listdir(path=".\\test_data\\letters")
            image_file_name = ".\\test_data\\letters\\" + image_file_name
        else:
            image_file_name = ".\\test_data\\digits\\" + image_file_name
            files = os.listdir(path=".\\test_data\\digits")
        cv2.imwrite(image_file_name + '__' + str(len(files)) + '.jpg', resized)

    def multiply_image(self, image: Image, answer):
        self.create_image(answer, image.image)
        image_copy = copy.deepcopy(image)
        image_copy.rotate(5)
        self.create_image(answer, image_copy.crop_binarized_char_by_edges())

        image_copy = copy.deepcopy(image)
        image_copy.rotate(-5)
        self.create_image(answer, image_copy.crop_binarized_char_by_edges())

        if answer in ['B', 'C', 'D', 'E', 'H', 'K', 'O', 'X', '0']:
            self.create_image(answer, image.flip_horizontal())
        elif answer == '6':
            self.create_image('9', image.flip_horizontal())
        elif answer == '9':
            self.create_image('6', image.flip_horizontal())

        if answer in ['H', 'M', 'O', 'X', '8', '0']:
            self.create_image(answer, image.flip_vertical())

    def run(self, image: Image):
        print('What is it?(image_name/-)   ')
        image.show('What is it?')
        answer = input()
        if answer == '-':
            return
        else:
            self.multiply_image(image, answer)
            # self.create_image(answer, image.image)

