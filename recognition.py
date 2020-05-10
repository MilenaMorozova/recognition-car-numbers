import copy

import cv2
import numpy as np

from Image import Image


def find_location_of_number_plate(file_image_name) -> tuple:
    russian_number_cascade = cv2.CascadeClassifier('xml-car-numbers\\haarcascade_russian_plate_number.xml')

    image = cv2.imread(file_image_name)  # 5
    origin = Image(image)

    result = []

    # for scale in np.arange(1.1, 20.1, 0.1):
    #     for neighbors in range(2, 15):
    #         russian_number_plate_rect = russian_number_cascade.detectMultiScale(image_copy, scaleFactor=scale, minNeighbors=neighbors)
    #         if len(russian_number_plate_rect):
    #             result.append([(scale, neighbors), russian_number_plate_rect])
    cropped_images = []
    russian_number_plate_rect = russian_number_cascade.detectMultiScale(origin.grayscale(), scaleFactor=1.2, minNeighbors=2)

    if len(russian_number_plate_rect):
        for (x, y, w, h) in russian_number_plate_rect:
            cropped_images.append(origin.crop(x, y, x+w, y+h))
            cv2.rectangle(origin.image, (x, y), (x + w, y + h), (0, 255, 0), 10)

    cv2.imshow("Original image", origin.image)
    cv2.waitKey(0)
    return cropped_images, result


def normalizing_image_of_number_plate_contours(cropped_images: list):
    cropped_number_plates = []

    for i in cropped_images:
        # height, width, _ = i.shape
        height = i.height
        width = i.width

        # gray_image = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        # edges = cv2.Canny(image, 50, 150, apertureSize=3)
        edges = i.canny(30, 150)
        contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            rect = cv2.minAreaRect(cnt)  # пытаемся вписать прямоугольник
            box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника

            if abs(box[0][0] - box[2][0]) < width/2 or abs(box[0][1] - box[1][1]) < height/2:
                continue

            box = np.int0(box)  # округление координат
            cv2.drawContours(i.image, [box], 0, (255, 0, 0), 2)  # рисуем прямоугольник

        cv2.imshow("Contours", i.image)
        cv2.waitKey(0)


def normalizing_image_of_number_plate_hough_lines(cropped_images: list):

    for i in cropped_images:
        edges = i.canny(50, 150)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
        if lines is not None:
            for line in lines:
                for rho, theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))

                    cv2.line(i.image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("Hough lines", i.image)
        cv2.waitKey(0)


def normalizing_image_of_number_plate_hough_lines_p(cropped_images: list):
    for i in cropped_images:
        edges = i.canny(50, 150)

        min_line_length = 150
        max_line_gap = 30

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,  min_line_length, max_line_gap)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(i.image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Hough lines P", i.image)
        cv2.waitKey(0)


if __name__ == '__main__':
    cropped_images, result = find_location_of_number_plate('images\\car (8).jpg')

    normalizing_image_of_number_plate_contours(copy.deepcopy(cropped_images))
    normalizing_image_of_number_plate_hough_lines(copy.deepcopy(cropped_images))
    normalizing_image_of_number_plate_hough_lines_p(copy.deepcopy(cropped_images))