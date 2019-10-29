import cv2 as cv
import numpy as np


def cannize(frame):
    gray_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    blur_frame = cv.GaussianBlur(gray_frame, (15, 15), 0)
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


def merge_lines(frame, lines, inCoordinates):
    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))
    if len(left) != 0:
        left = np.average(left, axis=0)
        if inCoordinates is True:
            left = calculate_coordinates(frame, left)
    if len(right) != 0:
        right = np.average(right, axis=0)
        if inCoordinates is True:
            right = calculate_coordinates(frame, right)
    return np.array([left, right])


def calculate_coordinates(frame, parameters):
    height = frame.shape[0]
    slope, intercept = parameters
    y1 = height
    y2 = int(height/2)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def symmetrize(line):
    if(len(line) == 0):
        return np.array([])
    slope, intercept = line
    slope = -slope
    return [slope, intercept]


def reward(frame):
    cannized_frame = cannize(frame)
    segment = cut_the_horizon(cannized_frame)
    hough = cv.HoughLinesP(segment, 1, np.pi / 90, 50,
                           np.array([]), minLineLength=100, maxLineGap=100)
    lines = merge_lines(frame, hough, False)
    left, right = lines
    return np.array([left, symmetrize(right)])
