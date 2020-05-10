import copy
import math

import cv2
import numpy as np

from Image import Image


def find_location_of_number_plate(file_image_name) -> tuple:
    russian_number_cascade = cv2.CascadeClassifier('xml-car-numbers\\haarcascade_russian_plate_number.xml')

    image = cv2.imread(file_image_name)  # 5
    origin = Image(image)

    result = []

    cropped_images = []
    russian_number_plate_rect = russian_number_cascade.detectMultiScale(origin.grayscale(), scaleFactor=1.2, minNeighbors=2)

    if len(russian_number_plate_rect):
        for (x, y, w, h) in russian_number_plate_rect:
            cropped_images.append(origin.crop(x, y, x+w, y+h))
            cv2.rectangle(origin.image, (x, y), (x + w, y + h), (0, 255, 0), 10)

    # cv2.imshow("Original image", origin.image)
    # cv2.waitKey(0)
    return cropped_images, result


def normalizing_image_of_number_plate_contours(cropped_images: list):

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

    for i in range(len(cropped_images)-1, -1, -1):
        edges = cropped_images[i].canny(50, 150)

        min_line_length = 150
        max_line_gap = 30

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,  min_line_length, max_line_gap)

        if lines is not None:
            cropped_images[i].set_bounds(find_two_main_lines(lines, copy.deepcopy(cropped_images[i])))
        else:
            # del cropped_images[i]
            print('Увы')


def find_two_main_lines(lines: list, image: Image):
    result = []

    center_of_image_height = int(image.height/2)

    lines_part_up = [line for line in lines for _, y1, _, y2 in line if y1 < center_of_image_height and y2 < center_of_image_height]
    lines_part_down = [line for line in lines for _, y1, _, y2 in line if y1 > center_of_image_height and y2 > center_of_image_height]

    for part_of_lines in [lines_part_up, lines_part_down]:
        if part_of_lines:
            tangent_of_lines = [(y2 - y1)/(x2 - x1) for line in part_of_lines for x1, y1, x2, y2 in line]

            free_members_of_lines = [y1 - tangent_of_lines[i] * x1
                                           for i, line in enumerate(part_of_lines) for x1, y1, _, _ in line]

            average_line = [np.mean(tangent_of_lines), np.mean(free_members_of_lines)]  # find average line for lines
            result.append(average_line)

            cv2.line(image.image, (0, int(average_line[1])),
                     (image.width, int(average_line[0] * image.width + average_line[1])), (0, 255, 0))

    # cv2.imshow("TWO MAIN LINES", image.image)
    # cv2.waitKey(0)
    return result


def b(lines):
    tangent_of_lines = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in
                        line]
    print('tangets', tangent_of_lines)

    for i, line in enumerate(lines):
        for x1, y1, _, _ in line:
            free_members_of_lines = [y1 - tangent_of_lines[i] * x1]

    print('free members', free_members_of_lines)


def rotate_image(cropped_images: list):
    for i, image in enumerate(cropped_images):
        max_k = np.mean([abs(j[0]) for j in image.bounds])
        angle = (math.atan(max_k)*180) / np.pi

        image_center = tuple(np.array(image.image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
        result = cv2.warpAffine(image.image, rot_mat, image.image.shape[1::-1], flags=cv2.INTER_LINEAR)

        cropped_images[i] = Image(result)
        # cv2.imshow("Rotated image", cropped_images[i].image)
        # cv2.waitKey(0)


def crop_image_by_bounds(image: Image):
    center_height_of_image = int(image.height/2)
    image_bounds = image.bounds

    if len(image_bounds) == 2:
        y1 = int(image_bounds[0][1])
        y2 = int(image_bounds[1][0]*image.width + image_bounds[1][1])
        image = image.crop(0, y1, image.width, y2)

    elif len(image_bounds) == 1:
        if int(image_bounds[0][0]*image.width/2 + image_bounds[0][1]) > center_height_of_image:
            image = image.crop(0, int(image_bounds[0][1]), image.width, image.height)
        else:
            image = image.crop(0, 0, image.width, int(image_bounds[0][0]*image.width + image_bounds[0][1]))

    # cv2.imshow("Cropped image", image.image)
    # cv2.waitKey(0)


if __name__ == '__main__':
    cropped_images, result = find_location_of_number_plate('images\\car (6).jpg')

    # normalizing_image_of_number_plate_contours(copy.deepcopy(cropped_images))
    # normalizing_image_of_number_plate_hough_lines(copy.deepcopy(cropped_images))
    normalizing_image_of_number_plate_hough_lines_p(cropped_images)

    rotate_image(cropped_images)
    normalizing_image_of_number_plate_hough_lines_p(cropped_images)
    for i in cropped_images:
        crop_image_by_bounds(i)