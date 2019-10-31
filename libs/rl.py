import cv2 as cv
import math
import numpy as np
from libs import line, visualization


def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    x_1, y_1, x_2, y_2 = v1
    a_1, b_1, a_2, b_2 = v2
    v_1 = [x_1-x_2, y_1-y_2]
    v_2 = [a_1-a_2, b_1-b_2]
    return math.acos(dotproduct(v_1, v_2) / (length(v_1) * length(v_2)))


def reward(frame):
    canny = visualization.cannize(frame, 15)
    segment = visualization.cut_the_horizon(canny)
    hough = cv.HoughLinesP(segment, 1, np.pi / 180, 50,
                           np.array([]), minLineLength=100, maxLineGap=100)
    lines = line.merge_by_kmeans(hough)
    lines = line.slopes_to_points(frame, lines)
    lines = line.colapse_neighbours(500, lines)
    lines = line.points_to_slopes(lines)
    left, right = lines
    vectors = None
    if len(left) == 0:
        vectors = [right, line.symmetrize(right)]
    elif len(right) == 0:
        vectors = [left, line.symmetrize(left)]
    else:
        vectors = [left, line.symmetrize(right)]
    v1, v2 = line.slopes_to_points(frame, vectors)
    return math.pi-angle(v1, v2)
