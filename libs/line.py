import numpy as np
from numpy import linalg as la
from sklearn.cluster import KMeans


def slope_to_point(height, line):
    if len(line) == 0:
        return np.array([])
    slope, intercept = line
    y_1 = height
    y_2 = int(height/2)
    x_1 = int((y_1 - intercept) / slope)
    x_2 = int((y_2 - intercept) / slope)
    return np.array([x_1, y_1, x_2, y_2])


def slopes_to_points(frame, lines):
    height = frame.shape[0]
    results = []
    for line in lines:
        temp = slope_to_point(height, line)
        results.append(temp)
    return results


def point_to_slope(line):
    x_1, y_1, x_2, y_2 = line.reshape(4)
    return np.polyfit((x_1, x_2), (y_1, y_2), 1)


def points_to_slopes(lines):
    results = []
    for line in lines:
        parameters = point_to_slope(line)
        results.append(parameters)
    return np.array(results)


def merge_by_slope_sign(lines):
    left = []
    right = []
    for line in lines:
        parameters = point_to_slope(line)
        slope = parameters[0]
        if slope < 0:
            left.append(parameters)
        else:
            right.append(parameters)
    if len(left) != 0:
        left = np.average(left, axis=0)
    if len(right) != 0:
        right = np.average(right, axis=0)
    return np.array([left, right])


def merge_by_kmeans(lines):
    results = points_to_slopes(lines)
    if len(results) < 2:
        return np.array([results[0], []])
    kmeans = KMeans(n_clusters=2, random_state=0).fit(results)
    return kmeans.cluster_centers_


def colapse_neighbours(max_norm, lines):
    line_1, line_2 = lines
    if len(line_1) != len(line_2):
        return lines
    norm = la.norm(line_1-line_2)
    if norm <= max_norm:
        return np.array([np.average(lines, axis=0).astype(int), []])
    return lines


def symmetrize(line):
    if len(line) == 0:
        return np.array([])
    slope, intercept = line
    slope = -slope
    return [slope, intercept]
