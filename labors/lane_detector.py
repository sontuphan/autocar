import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def do_canny(frame):
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (15, 15), 0)
    canny = cv.Canny(blur, 50, 150)
    return canny


def do_rectangle(frame):
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


def calculate_lines(lines):
    re = np.array([])
    for line in lines:
        x_1, y_1, x_2, y_2 = line.reshape(4)
        parameters = np.array([np.polyfit((x_1, x_2), (y_1, y_2), 1)])
        if(len(re) == 0):
            re = parameters
        else:
            re = np.append(re, parameters, axis=0)
    re = calculate_by_kmeans(re)
    return re


def calculate_by_kmeans(points):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(points)
    return kmeans.cluster_centers_


def calculate_coordinates(frame, lines):

    def sub_calculate_coordinates(height, line):
        if(len(line) == 0):
            return np.array([0, 0, 0, 0])
        slope, intercept = line
        y1 = height
        y2 = int(height/2)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])

    height = frame.shape[0]
    lines_in_coordinate = np.array([])
    for line in lines:
        temp = np.array([sub_calculate_coordinates(height, line)])
        if(len(lines_in_coordinate) == 0):
            lines_in_coordinate = temp
        else:
            lines_in_coordinate = np.append(lines_in_coordinate, temp, axis=0)
    return lines_in_coordinate


def visualize_lines(frame, lines):
    lines_visualize = np.zeros_like(frame)
    for line in lines:
        if len(line) != 0:
            x1, y1, x2, y2 = line
            cv.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return lines_visualize


def main():
    cap = cv.VideoCapture("data/autocar.mp4")
    while (cap.isOpened()):
        ret, frame = cap.read()
        canny = do_canny(frame)
        # plt.imshow(canny)
        # plt.show()

        segment = do_rectangle(canny)
        hough = cv.HoughLinesP(segment, 1, np.pi / 180, 50,
                               np.array([]), minLineLength=100, maxLineGap=100)
        # Test run
        # test = np.array([])
        # if(hough is not None):
        #     for line in hough:
        #         if(len(test) == 0):
        #             test = line
        #         test = np.append(test, line, axis=0)
        # cvlines = visualize_lines(frame, test)

        # Real run
        lines = calculate_lines(hough)
        lines_in_coordinate = calculate_coordinates(frame, lines)
        cvlines = visualize_lines(frame, lines_in_coordinate)

        # Visualize
        output = cv.addWeighted(frame, 0.9, cvlines, 1, 1)
        cv.imshow("Realtime", output)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
