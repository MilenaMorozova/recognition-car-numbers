import os

base_path = os.path.dirname(__file__)  # .../recognition-car-number

path_to_network_weights = os.path.join(base_path,
                                       'src',
                                       'network_parameters.json')

path_to_haarcascade = os.path.join(base_path,
                                   'xml-car-numbers',
                                   'haarcascade_russian_plate_number.xml')
