import cv2 as cv
import numpy as np
from libs import line, visualization


def test_by_slope_sign():
    cap = cv.VideoCapture("data/autocar.mp4")
    while cap.isOpened():
        frame = cap.read()[1]
        canny = visualization.cannize(frame, 15)
        segment = visualization.cut_the_horizon(canny)
        hough = cv.HoughLinesP(segment, 1, np.pi / 180, 50,
                               np.array([]), minLineLength=100, maxLineGap=100)
        lines = line.merge_by_slope_sign(hough)
        lines = line.slopes_to_points(frame, lines)
        cv_lines = visualization.draw_lines_in_frame(frame, lines)

        output = cv.addWeighted(frame, 0.9, cv_lines, 1, 1)
        cv.imshow("Slope Sign", output)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def test_by_kmeans():
    cap = cv.VideoCapture("data/autocar.mp4")
    while cap.isOpened():
        frame = cap.read()[1]
        canny = visualization.cannize(frame, 15)
        segment = visualization.cut_the_horizon(canny)
        hough = cv.HoughLinesP(segment, 1, np.pi / 180, 50,
                               np.array([]), minLineLength=100, maxLineGap=100)
        lines = line.merge_by_kmeans(hough)
        lines = line.slopes_to_points(frame, lines)
        lines = line.colapse_neighbours(500, lines)
        cv_lines = visualization.draw_lines_in_frame(frame, lines)

        output = cv.addWeighted(frame, 0.9, cv_lines, 1, 1)
        cv.imshow("Kmeans", output)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
