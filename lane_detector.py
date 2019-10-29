import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


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


def calculate_lines(frame, lines):
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
    if(len(left) != 0):
        left = np.average(left, axis=0)
        left = calculate_coordinates(frame, left)
    if(len(right) != 0):
        right = np.average(right, axis=0)
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
        hough = cv.HoughLinesP(segment, 1, np.pi / 90, 50,
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
        lines = calculate_lines(frame, hough)
        cvlines = visualize_lines(frame, lines)

        # Visualize
        output = cv.addWeighted(frame, 0.9, cvlines, 1, 1)
        cv.imshow("Realtime", output)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
