import os

import cv2

from Image import Image

# TODO увеличить объём выборки(переворот 0, 6, 9), попробовать повернуть на немного картинки


class TestDataCreator:

    @staticmethod
    def create_image(image_file_name, image):
        # TODO size of image
        resized = cv2.resize(image, (27, 36), interpolation=cv2.INTER_AREA)
        if image_file_name.isalpha():
            files = os.listdir(path=".\\test_data\\letters")
            image_file_name = ".\\test_data\\letters\\" + image_file_name
        else:
            image_file_name = ".\\test_data\\digits\\" + image_file_name
            files = os.listdir(path=".\\test_data\\digits")
        cv2.imwrite(image_file_name + '__' + str(len(files)) + '.jpg', resized)

    def run(self, image: Image):
        print('What is it?(image_name/-)   ')
        image.show('What is it?')
        answer = input()
        # cv2.destroyWindow('What is it?')
        if answer == '-':
            return
        else:
            self.create_image(answer, image.image)
