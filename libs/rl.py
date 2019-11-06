import sys
from pathlib import Path
import cv2 as cv
import numpy as np
from libs import line, visualization, util


class MDP:

    # debug mode
    # 0: turn off
    # 1: canny debug
    # 2: lines detection debug
    def __init__(self, agent, debug=0):
        self.SCALE = 15
        self.NUM_OF_STATES = int(360/self.SCALE)
        self.STATES = np.arange(0, 360, self.SCALE)
        self.ACTIONS = [-1, 0, 1]
        self.NUM_OF_ACTIONS = len(self.ACTIONS)

        self.event_matrix = np.zeros(
            (self.NUM_OF_STATES, self.NUM_OF_STATES, self.NUM_OF_ACTIONS), dtype=int)
        self.value_vector = np.zeros(self.NUM_OF_STATES, dtype=int)
        self.load_model()
        self.num_of_decisions = np.array(
            [plane.sum() for plane in self.event_matrix]).sum()

        self.noise_rejection = 15
        self.error_tolerant = 20
        self.discount = 0.1
        self.agent = agent
        self.stream = agent.get_snapshot()

        self.debug = debug

    def extract_frame(self):
        return self.stream.get()

    def discretize(self, degree):
        discretization = 0
        degree = round(degree)
        for i in np.arange(0, self.NUM_OF_STATES + 1):
            if(degree < i*self.SCALE):
                discretization = (i-1)*self.SCALE
                break
        return discretization

    def get_state(self):
        error = 0
        while error < self.error_tolerant:
            frame = self.extract_frame()
            canny = visualization.cannize(frame, self.noise_rejection)
            if self.debug == 1:
                cv.imshow("Debug", canny)
                cv.waitKey(10)
            # segment = visualization.cut_the_horizon(canny)
            segment = canny
            hough = cv.HoughLinesP(segment, 1, np.pi / 180, 50,
                                   np.array([]), minLineLength=100, maxLineGap=100)

            if hough is None:
                print("Stop coundown:", error)
                error += 1
            else:
                break

        if error >= self.error_tolerant:
            print("Stoped the car")
            self.agent.stop()
            if self.debug != 0:
                cv.destroyWindow("Debug")
            sys.exit()

        lines = line.merge_by_kmeans(hough)
        lines = line.slopes_to_points(frame, lines)
        # 300: Euler distance in hyperplane
        lines = line.colapse_neighbours(300, lines)
        if self.debug == 2:
            cv_lines = visualization.draw_lines_in_frame(frame, lines)
            output = cv.addWeighted(frame, 0.9, cv_lines, 1, 1)
            cv.imshow("Debug", output)
            cv.waitKey(10)
        lines = line.points_to_slopes(lines)
        left, right = lines
        vectors = None
        if len(left) == 0:
            vectors = [right, right]
        elif len(right) == 0:
            vectors = [left, left]
        else:
            vectors = [left, right]
        v1, v2 = line.slopes_to_points(frame, vectors)

        base_vector = [0, 0, -1, 0]
        degree1 = util.angle(base_vector, v1)
        degree2 = util.angle(base_vector, v2)
        degree = degree1 + degree2
        state = self.discretize(degree)
        return state

    def get_action(self, current_state):
        next_action = -1
        max_value = 0
        # Randomize the init data
        if self.num_of_decisions <= 4000:
            action = self.ACTIONS[0]
            # rand_act = util.random(self.NUM_OF_ACTIONS-1)
            # action = self.ACTIONS[rand_act]
            value = self.get_reward(current_state)
            for state in self.STATES:
                value += self.discount*self.get_prob(current_state, action,
                                                     state)*self.value_vector[int(state/self.SCALE)]
            max_value = value
            next_action = action
        # Start to learn
        else:
            for action in self.ACTIONS:
                value = self.get_reward(current_state)
                for state in self.STATES:
                    value += self.discount*self.get_prob(current_state, action,
                                                         state)*self.value_vector[int(state/self.SCALE)]
                if(value >= max_value):
                    max_value = value
                    next_action = action

        self.num_of_decisions += 1
        print(self.num_of_decisions)
        return [next_action, max_value]

    def get_reward(self, state):
        reward = 360 - abs(180 - state)
        return reward

    def get_prob(self, current_state, next_action, next_state):
        cs = int(current_state/self.SCALE)
        na = next_action + 1
        ns = int(next_state/self.SCALE)
        cs_na_ns = self.event_matrix[cs, ns, na] + 1
        cs_na = len(self.STATES)
        for row in self.event_matrix[cs]:
            cs_na += row[na]
        return cs_na_ns/cs_na

    def learn(self, prev_state, prev_action, prev_value, current_state):
        ps = int(prev_state/self.SCALE)
        pa = prev_action + 1
        cs = int(current_state/self.SCALE)
        self.value_vector[ps] = prev_value
        self.event_matrix[ps, cs, pa] += 1
        self.save_model()

    def save_model(self):
        np.save("event_matrix.npy", self.event_matrix)
        np.save("value_vector.npy", self.value_vector)

    def load_model(self):
        event_matrix = Path("event_matrix.npy")
        if event_matrix.is_file():
            self.event_matrix = np.load("event_matrix.npy")
            print(self.event_matrix)
        value_vector = Path("value_vector.npy")
        if value_vector.is_file():
            self.value_vector = np.load("value_vector.npy")
            print(self.value_vector)
