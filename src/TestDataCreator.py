import os

import cv2
from src.recognition import RecognitionCarPlate
from src.MyImage import MyImage


class TestDataCreator:
    def __init__(self):
        self.directory = 'testing'
        self.recognitor = RecognitionCarPlate()
        self.create_directoties()

    def start(self, file):
        self.recognitor.run(file)
        characters = []
        for car_number in self.recognitor.car_numbers:
            characters.extend(car_number.series_and_registration_num + car_number.region)
        return characters

    def create_directoties(self):
        try:
            os.mkdir(self.directory)
            os.mkdir(os.path.join(self.directory, 'digits'))
            os.mkdir(os.path.join(self.directory, 'letters'))
        except FileExistsError:
            pass

    def create_image(self, image_file_name, image, start: int = 0):
        resized = cv2.resize(image, (27, 36), interpolation=cv2.INTER_AREA)
        if image_file_name.isalpha():
            files = os.listdir(path=os.path.join(self.directory, 'letters'))
            image_file_name = os.path.join(self.directory, 'letters', image_file_name)
        else:
            image_file_name = os.path.join(self.directory, 'digits', image_file_name)
            files = os.listdir(path=os.path.join(self.directory, 'digits'))
        cv2.imwrite(image_file_name + '__' + str(start + len(files)) + '.jpg', resized)

    def multiply_image(self, image: MyImage, answer):
        self.create_image(answer, image.image)
        self.create_image(answer,
                          self.recognitor.crop_binarized_char_by_edges(image.rotate(5, (255, 255, 255))).image)

        self.create_image(answer,
                          self.recognitor.crop_binarized_char_by_edges(image.rotate(-5, (255, 255, 255))).image)

        if answer in ['B', 'C', 'D', 'E', 'H', 'K', 'O', 'X', '0']:
            self.create_image(answer, image.flip_horizontal().image)
        elif answer == '6':
            self.create_image('9', image.rotate(180, (255, 255, 255)).image)
        elif answer == '9':
            self.create_image('6', image.rotate(180, (255, 255, 255)).image)

        if answer in ['H', 'M', 'O', 'X', '8', '0']:
            self.create_image(answer, image.flip_vertical().image)

    def create_data(self, image: MyImage):
        print('What is it?(image_name/-)   ')
        image.show('What is it?')
        answer = input()
        if answer == '-':
            return
        else:
            self.multiply_image(image, answer)
