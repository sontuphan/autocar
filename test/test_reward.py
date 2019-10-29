from libs import reward
import cv2 as cv
import numpy as np


def visualize_lines(frame, lines):
    lines_visualize = np.zeros_like(frame)
    for line in lines:
        if len(line) != 0:
            x1, y1, x2, y2 = line
            cv.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return lines_visualize


def test():
    cap = cv.VideoCapture("data/autocar.mp4")
    while (cap.isOpened()):
        ret, frame = cap.read()
        lines = reward.reward(frame)
        print(lines)

    cap.release()
    cv.destroyAllWindows()
