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


class MDP:

    def __init__(self, discount,  stream):
        self.event_matrix = np.zeros((6, 6, 3), dtype=int)
        self.value_vector = np.zeros(6, dtype=int)

        self.discount = discount
        self.stream = stream

        self.STATES = [0, 30, 60, 90, 120, 150]
        self.ACTIONS = [-1, 0, 1]

    def extract_frame(self):
        if self.stream.isOpened():
            return self.stream.read()[1]
        return None

    def discretize(self, degree):
        discretization = 0
        for i in np.arange(0, 9):
            if(degree < i*30):
                discretization = (i-1)*30
                break
        return discretization

    def get_state(self, frame):
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

        radian = angle(v1, v2)
        degree = np.degrees(radian)
        state = self.discretize(degree)
        return state

    def get_action(self, current_state):
        next_action = -1
        max_value = 0
        for action in self.ACTIONS:
            value = self.get_reward(current_state)
            for state in self.STATES:
                value += self.discount*self.get_prob(current_state, action,
                                                          state)*self.value_vector[int(state/30)]
            if(value >= max_value):
                max_value = value
                next_action = action
        return [next_action, max_value]

    def get_reward(self, state):
        return 180 - state

    def get_prob(self, current_state, next_action, next_state):
        cs = int(current_state/30)
        na = next_action + 1
        ns = int(next_state/30)
        cs_na_ns = self.event_matrix[cs, ns, na] + 1
        cs_na = len(self.STATES)
        for row in self.event_matrix[cs]:
            cs_na += row[na]
        return cs_na_ns/cs_na

    def learn(self, prev_state, prev_action, prev_value, current_state):
        ps = int(prev_state/30)
        pa = prev_action+1
        cs = int(current_state/30)
        self.value_vector[ps] = prev_value
        self.event_matrix[ps, pa, cs] += 1