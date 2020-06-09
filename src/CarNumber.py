from src.Image import Image


class CarNumber:
    def __init__(self, image: Image):
        self.image = image
        self.region_image = None

        self.series_and_registration_num = []
        self.region = []

    def is_empty(self):
        return not self.series_and_registration_num and not self.region

    def is_valid(self):
        return self.region_image and len(self.series_and_registration_num) >= 6

    def show_region(self):
        for image in self.region:
            image.show('REGION CHARACTERS')

    def show_series_and_registration_num(self):
        for image in self.series_and_registration_num:
            image.show('SERIES AND REGISTRATION NUMBER')

    def remove_empty_images_from_region(self):
        for i in range(len(self.region)-1, -1, -1):
            if self.region[i].is_empty():
                del self.region[i]
                print("delete from region")

    def remove_empty_images_from_series_reg_num(self):
        for i in range(len(self.series_and_registration_num)-1, -1, -1):
            if self.series_and_registration_num[i].is_empty():
                del self.series_and_registration_num[i]
                print("delete from series")

    def remove_all_empty_images(self):
        self.remove_empty_images_from_region()
        self.remove_empty_images_from_series_reg_num()
