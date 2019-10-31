import cv2 as cv
import numpy as np


def cannize(frame, blur):
    gray_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    blur_frame = cv.GaussianBlur(gray_frame, (blur, blur), 0)
    cannized_frame = cv.Canny(blur_frame, 50, 150)
    return cannized_frame


def cut_the_horizon(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    polygons = np.array([
        [(0, height), (width, height),
         (width, int(height/2)), (0, int(height/2))]
    ])
    mask = np.zeros_like(frame)
    cv.fillPoly(mask, polygons, 255)
    rectangle = cv.bitwise_and(frame, mask)
    return rectangle


def draw_lines_in_frame(frame, lines):
    lines_visualize = np.zeros_like(frame)
    for line in lines:
        if len(line) != 0:
            x_1, y_1, x_2, y_2 = line
            cv.line(lines_visualize, (x_1, y_1), (x_2, y_2), (0, 255, 0), 5)
    return lines_visualize
