from Image import Image


class CarNumber:
    def __init__(self, image: Image):
        self.image = image
        self.series_and_registration_num = []
        self.region = []

    def remove_nan_from_series_reg_num(self):
        for i in range(len(self.series_and_registration_num)-1, -1, -1):
            if not self.series_and_registration_num[i].image:
                del self.series_and_registration_num[i]

    def remove_nan_from_region(self):
        for i in range(len(self.region)-1, -1, -1):
            if not self.region[i].image:
                del self.region[i]

    def remove_all_nan(self):
        self.remove_nan_from_region()
        self.remove_nan_from_series_reg_num()

    def is_empty(self):
        return self.series_and_registration_num and self.region

    def show_region(self):
        for image in self.region:
            image.show('REGION CHARACTERS')

    def show_series_and_registration_num(self):
        for image in self.series_and_registration_num:
            image.show('SERIES AND REGISTRATION NUMBER')
