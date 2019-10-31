import cv2 as cv
import numpy as np
from libs import line
from libs import visualization


def reward(frame):
    cannized_frame = visualization.cannize(frame)
    segment = visualization.cut_the_horizon(cannized_frame)
    hough = cv.HoughLinesP(segment, 1, np.pi / 90, 50,
                           np.array([]), minLineLength=100, maxLineGap=100)
    lines = line.merge_by_slopes(hough)
    left, right = lines
    return np.array([left, line.symmetrize(right)])
