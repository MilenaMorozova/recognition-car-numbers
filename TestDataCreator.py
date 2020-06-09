import os

import cv2

import recognition as r
from Image import Image


class TestDataCreator:

    def create_image(self, image_file_name, image, start: int = 30):
        resized = cv2.resize(image, (27, 36), interpolation=cv2.INTER_AREA)
        if image_file_name.isalpha():
            files = os.listdir(path=".\\test_data\\letters")
            image_file_name = ".\\test_data\\letters\\" + image_file_name
        else:
            image_file_name = ".\\test_data\\digits\\" + image_file_name
            files = os.listdir(path=".\\test_data\\digits")
        cv2.imwrite(image_file_name + '__' + str(start + len(files)) + '.jpg', resized)

    def multiply_image(self, image: Image, answer):
        self.create_image(answer, image.image)
        self.create_image(answer,
                          r.RecognitionCarPlate.crop_binarized_char_by_edges(image.rotate(5, (255, 255, 255))).image)

        self.create_image(answer,
                          r.RecognitionCarPlate.crop_binarized_char_by_edges(image.rotate(-5, (255, 255, 255))).image)

        if answer in ['B', 'C', 'D', 'E', 'H', 'K', 'O', 'X', '0']:
            self.create_image(answer, image.flip_horizontal().image)
        elif answer == '6':
            self.create_image('9', image.rotate(180, (255, 255, 255)).image)
        elif answer == '9':
            self.create_image('6', image.rotate(180, (255, 255, 255)).image)

        if answer in ['H', 'M', 'O', 'X', '8', '0']:
            self.create_image(answer, image.flip_vertical().image)

    def run(self, image: Image):
        print('What is it?(image_name/-)   ')
        image.show('What is it?')
        answer = input()
        if answer == '-':
            return
        else:
            self.multiply_image(image, answer)
            # self.create_image(answer, image.image)
