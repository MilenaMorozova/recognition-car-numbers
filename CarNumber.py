from Image import Image


class CarNumber:
    def __init__(self, image: Image):
        self.image = image
        self.series_and_registration_num = []
        self.region = []
