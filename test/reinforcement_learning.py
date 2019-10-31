import cv2 as cv
import time
from libs import rl


def test_mdp():
    cap = cv.VideoCapture("data/realrun3.mp4")
    mdp = rl.MDP(0.1, cap)

    prev_state = None
    prev_action = None
    prev_value = None
    while cap.isOpened():
        frame = mdp.extract_frame()
        current_state = mdp.get_state(frame)
        if prev_state is not None:
            mdp.learn(prev_state, prev_action, prev_value, current_state)
        next_action, value = mdp.get_action(current_state)
        print(next_action)
        # send cmd to car here
        prev_state = current_state
        prev_action = next_action
        prev_value = value
        # time.sleep(1)
